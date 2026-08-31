"""
pipeline.py — multi-process actor/learner with pin-before-draw protocol.

Architecture:
  - Learner process: publishes revisions, holds the current revision digest.
  - Actor processes: each episode, pin a revision, draw actions, write to ledger.
  - Communication: multiprocessing.Queue with spawn start method.
  - Cut points instrumented for T5 interleaving campaign.
  - SIGKILL support: actors can be killed at cut points; ledger survives intact.

Pin-before-draw invariant (enforced structurally):
  The draw function receives ONLY the pinned revision's simplex.
  There is no way to draw under a revision other than the pinned one.

Durability: all ledger/revision writes use F_FULLFSYNC + atomic rename (see revision.py).
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import signal
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional


class LedgerConsistencyError(Exception):
    """Raised when the ledger is found to be inconsistent after recovery."""


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""
    n_actors: int
    n_revisions: int
    n_episodes_per_actor: int
    n_states: int
    n_actions: int
    horizon: int
    seed: bytes
    store_dir: Path
    ledger_dir: Path
    force_mixed_revisions: bool = False   # force revision change mid-trajectory
    sigkill_after_episodes: Optional[int] = None  # kill actor after N episodes, then restart

    def __post_init__(self):
        self.store_dir = Path(self.store_dir)
        self.ledger_dir = Path(self.ledger_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)


def _make_initial_policy(n_states: int, n_actions: int, seed: bytes) -> dict:
    """Generate a rational simplex policy table from a seed."""
    import struct
    table = {}
    for s in range(n_states):
        raw = []
        for a in range(n_actions):
            h = hashlib.blake2b(
                seed + f"p_{s}_{a}".encode(), digest_size=8
            ).digest()
            v = Fraction(int.from_bytes(h, "big") + 1, 2**64)
            raw.append(v)
        total = sum(raw)
        table[s] = {a: r / total for a, r in enumerate(raw)}
    return table


def _evolve_policy(table: dict, step: int, seed: bytes) -> dict:
    """Deterministically evolve a policy table for revision `step`."""
    new_table = {}
    for s, action_probs in table.items():
        n = len(action_probs)
        actions = sorted(action_probs.keys())
        alpha = Fraction(1, 20)
        # Pick a random action to ascent on
        h = hashlib.blake2b(seed + f"evolve_{s}_{step}".encode(), digest_size=4).digest()
        chosen_idx = int.from_bytes(h, "big") % n
        chosen = actions[chosen_idx]
        # Update: p_i ← p_i + alpha * (1_{i=chosen} - p_i)
        new_probs = {}
        for a, p in action_probs.items():
            indicator = Fraction(1) if a == chosen else Fraction(0)
            new_p = p + alpha * (indicator - p)
            new_probs[a] = new_p
        new_table[s] = new_probs
    return new_table


def _actor_worker(
    actor_id: int,
    cfg: PipelineConfig,
    revision_queue: "multiprocessing.Queue",
    done_event: "multiprocessing.Event",
    kill_after: Optional[int],
) -> None:
    """
    Actor process: pin revisions, draw actions, write to ledger.

    Cut points (for T5 interleaving campaign):
      CP1: pre-pin
      CP2: post-pin-pre-draw
      CP3: post-draw-pre-ledger-append
      CP4: post-append-pre-ack
    """
    # Import here (in spawned process) to avoid cross-process state
    from martingale.draw import draw_action
    from martingale.ledger import ActionRecord, Ledger, GENESIS_DIGEST
    from martingale.revision import RevisionStore

    store = RevisionStore(cfg.store_dir)
    ledger = Ledger(cfg.ledger_dir, store)

    # Get initial revision
    current_revision = None
    while current_revision is None:
        try:
            current_revision = revision_queue.get(timeout=5.0)
        except Exception:
            if done_event.is_set():
                return

    episodes_done = 0
    for episode_id in range(cfg.n_episodes_per_actor):
        # CP1: pre-pin — check for new revision
        try:
            while True:
                rev = revision_queue.get_nowait()
                current_revision = rev
        except Exception:
            pass  # No new revision

        # CP2: post-pin-pre-draw — pin the current revision
        pinned_revision = current_revision
        pinned_table = store.get(pinned_revision)

        records = []
        prev_digest = GENESIS_DIGEST

        for step in range(cfg.horizon):
            # force_mixed_revisions: switch revision mid-trajectory
            if cfg.force_mixed_revisions and step == cfg.horizon // 2:
                # Try to get a DIFFERENT revision from the queue
                for _ in range(10):
                    try:
                        new_rev = revision_queue.get_nowait()
                        if new_rev != pinned_revision:
                            # IMPORTANT: pin the new revision for this step only
                            # (deliberately creates a mixed-revision trajectory)
                            pinned_revision = new_rev
                            pinned_table = store.get(pinned_revision)
                            break
                    except Exception:
                        break  # queue empty

            state = episode_id % cfg.n_states  # simplified state selection
            probs = pinned_table.table[state]

            # CP2 (continued): draw action using ONLY the pinned revision's simplex
            draw = draw_action(
                probs,
                seed=cfg.seed,
                actor_id=actor_id,
                episode=episode_id,
                step=step,
            )
            # CP3: post-draw-pre-ledger-append
            rec = ActionRecord(
                revision_digest=pinned_revision,
                state=state,
                action=draw.action,
                behavior_prob=probs[draw.action],
                draw=draw,
                env_next_state=(state + draw.action + 1) % cfg.n_states,
                env_reward=Fraction(draw.action),
                prev_digest=prev_digest,
            )
            records.append(rec)
            prev_digest = rec.digest

        # CP4: post-draw-pre-ledger-append → append trajectory
        ledger.append_trajectory(
            actor_id=actor_id,
            episode_id=episode_id,
            records=records,
        )
        episodes_done += 1

        # Simulate SIGKILL after kill_after episodes
        if kill_after is not None and episodes_done >= kill_after:
            # Exit abruptly (simulates kill, but ledger writes are durable)
            os._exit(0)

    done_event.set()


def _learner_worker(
    cfg: PipelineConfig,
    revision_queues: list,
    all_done_events: list,
) -> None:
    """
    Learner process: publishes revisions to actor queues.

    When force_mixed_revisions=True, publishes ALL revisions upfront so actors
    have a full queue to pick from mid-trajectory.
    """
    from martingale.revision import RevisionStore
    store = RevisionStore(cfg.store_dir)

    # Generate all revisions
    table = _make_initial_policy(cfg.n_states, cfg.n_actions, cfg.seed)
    all_revisions = []
    for rev_step in range(cfg.n_revisions):
        rev = store.publish(table)
        all_revisions.append(rev.digest)
        if rev_step < cfg.n_revisions - 1:
            table = _evolve_policy(table, rev_step + 1, cfg.seed)

    if cfg.force_mixed_revisions:
        # Pre-load all revisions into queues so actors can pick them up mid-trajectory
        for q in revision_queues:
            for rev_digest in all_revisions:
                try:
                    q.put_nowait(rev_digest)
                except Exception:
                    pass
    else:
        # Send initial revision first
        for q in revision_queues:
            q.put(all_revisions[0])
        # Then drip-feed remaining revisions with small delays
        for rev_digest in all_revisions[1:]:
            time.sleep(0.02)
            for q in revision_queues:
                try:
                    q.put_nowait(rev_digest)
                except Exception:
                    pass


def run_pipeline(cfg: PipelineConfig) -> None:
    """
    Run the multi-process pipeline with spawn start method.

    Launches learner + n_actors actor processes.
    If sigkill_after_episodes is set, kills and restarts actors after N episodes.
    """
    ctx = multiprocessing.get_context("spawn")

    revision_queues = [ctx.Queue(maxsize=20) for _ in range(cfg.n_actors)]
    done_events = [ctx.Event() for _ in range(cfg.n_actors)]

    # Start learner
    learner = ctx.Process(
        target=_learner_worker,
        args=(cfg, revision_queues, done_events),
        daemon=True,
    )
    learner.start()

    # Start actors
    kill_after = cfg.sigkill_after_episodes
    actors = []
    for actor_id in range(cfg.n_actors):
        p = ctx.Process(
            target=_actor_worker,
            args=(actor_id, cfg, revision_queues[actor_id],
                  done_events[actor_id], kill_after),
        )
        p.start()
        actors.append(p)

    # Wait for actors (with timeout)
    timeout = 30.0  # seconds
    deadline = time.time() + timeout

    for i, actor in enumerate(actors):
        remaining = max(0, deadline - time.time())
        actor.join(timeout=remaining)

    # If sigkill scenario: restart killed actors for remaining episodes
    if kill_after is not None:
        remaining_episodes = cfg.n_episodes_per_actor - kill_after
        if remaining_episodes > 0:
            new_queues = [ctx.Queue(maxsize=20) for _ in range(cfg.n_actors)]
            new_done = [ctx.Event() for _ in range(cfg.n_actors)]

            # Re-broadcast all stored revisions to restarted actors
            from martingale.revision import RevisionStore
            store = RevisionStore(cfg.store_dir)
            all_revs = list(store.all_digests())
            if all_revs:
                latest_rev = all_revs[-1]
                for q in new_queues:
                    q.put(latest_rev)

            new_actors = []
            for actor_id in range(cfg.n_actors):
                # Create a new config for the remaining episodes
                new_cfg = PipelineConfig(
                    n_actors=cfg.n_actors,
                    n_revisions=1,
                    n_episodes_per_actor=remaining_episodes,
                    n_states=cfg.n_states,
                    n_actions=cfg.n_actions,
                    horizon=cfg.horizon,
                    seed=cfg.seed,
                    store_dir=cfg.store_dir,
                    ledger_dir=cfg.ledger_dir,
                    force_mixed_revisions=cfg.force_mixed_revisions,
                    sigkill_after_episodes=None,
                )
                p = ctx.Process(
                    target=_actor_worker,
                    args=(actor_id, new_cfg, new_queues[actor_id],
                          new_done[actor_id], None),
                )
                p.start()
                new_actors.append(p)

            for p in new_actors:
                p.join(timeout=30.0)

    # Clean up learner
    if learner.is_alive():
        learner.terminate()
        learner.join(timeout=5.0)
