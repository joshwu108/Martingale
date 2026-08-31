"""
checker/verify.py — independent ledger verifier.

IMPORTANT: This file MUST NOT import anything from src/martingale/.
It reimplements only what is needed to verify a ledger from scratch,
using only stdlib. This isolation is CI-enforced by `make check-imports`.

Verification steps per action record:
  1. Re-derive behavior_prob from the revision store's stored policy table.
  2. Re-run the BLAKE2b draw with the keyed seed and verify that rejection
     sampling with the FULL simplex would have produced the claimed action
     after exactly rejection_count rejections, consuming draw_integer.
  3. Re-compute the record's digest from its fields and check it matches.
  4. Verify prev_digest chain integrity.

Step 2 catches the subtle forgery: if an action is drawn under revision r+1
but the ledger claims revision r, and both revisions happen to assign the
same probability to the chosen action, step 1 passes — but step 2 fails,
because the ordering of the full simplex differs between revisions (different
probability masses in different "slots"), so the same draw_integer maps to
a different action under r vs r+1.
"""
from __future__ import annotations

import hashlib
import json
import struct
from fractions import Fraction
from pathlib import Path
from typing import Iterator


# ===== Constants =====
GENESIS_DIGEST = "0" * 64


# ===== Canonical serialization (must match ledger.py exactly) =====

def _fraction_to_str(f: Fraction) -> str:
    """Serialize a Fraction as 'numerator/denominator' in reduced form."""
    f = Fraction(f)
    return f"{f.numerator}/{f.denominator}"


def _canonical_revision_bytes(table: dict) -> bytes:
    """Canonical bytes for a policy table (must match revision.py)."""
    canonical = {}
    for s in sorted(table.keys(), key=int):
        canonical[str(s)] = {}
        for a in sorted(table[s].keys(), key=int):
            p = Fraction(table[s][a]) if isinstance(table[s][a], str) else table[s][a]
            canonical[str(s)][str(a)] = _fraction_to_str(p)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _record_canonical_bytes(
    revision_digest: str,
    state: int,
    action: int,
    behavior_prob: Fraction,
    draw_integer: int,
    rejection_count: int,
    env_next_state: int,
    env_reward: Fraction,
    prev_digest: str,
) -> bytes:
    """Canonical bytes for an action record (must match ledger.py)."""
    obj = {
        "revision_digest": revision_digest,
        "state": state,
        "action": action,
        "behavior_prob": _fraction_to_str(behavior_prob),
        "draw_integer": draw_integer,
        "rejection_count": rejection_count,
        "env_next_state": env_next_state,
        "env_reward": _fraction_to_str(env_reward),
        "prev_digest": prev_digest,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ===== BLAKE2b draw (must match draw.py) =====

def _make_draw_key(seed: bytes, actor_id: int, episode: int, step: int) -> bytes:
    ctx = struct.pack(">QQQ",
                      actor_id & 0xFFFFFFFFFFFFFFFF,
                      episode & 0xFFFFFFFFFFFFFFFF,
                      step & 0xFFFFFFFFFFFFFFFF)
    return hashlib.blake2b(seed + ctx, digest_size=32).digest()


def _draw_uniform_integer(key: bytes, counter: int) -> int:
    ctr_bytes = struct.pack(">Q", counter & 0xFFFFFFFFFFFFFFFF)
    digest = hashlib.blake2b(ctr_bytes, digest_size=32, key=key).digest()
    return int.from_bytes(digest, byteorder="big")


def _verify_draw(
    simplex: dict[int, Fraction],  # {action: probability}
    seed: bytes,
    actor_id: int,
    episode: int,
    step: int,
    claimed_action: int,
    claimed_draw_integer: int,
    claimed_rejection_count: int,
) -> tuple[bool, str]:
    """
    Re-run the BLAKE2b rejection sampling and verify the draw.

    Returns (ok: bool, reason: str).
    """
    denom = 2 ** 256
    key = _make_draw_key(seed, actor_id, episode, step)
    actions_sorted = sorted(simplex.keys())
    # Build CDF boundaries
    boundaries = []
    cumulative = Fraction(0)
    for a in actions_sorted:
        lower = cumulative
        cumulative += simplex[a]
        if simplex[a] > 0:
            boundaries.append((a, lower, cumulative))

    for counter in range(claimed_rejection_count + 1):
        u = _draw_uniform_integer(key, counter)
        u_rat = Fraction(u, denom)
        for action, lower, upper in boundaries:
            if lower <= u_rat < upper:
                if counter < claimed_rejection_count:
                    # This counter should be a rejection, but found an action
                    return False, (
                        f"Draw {counter} landed on action {action}, "
                        f"but claimed {claimed_rejection_count} rejections"
                    )
                # This is the accepted draw
                if u != claimed_draw_integer:
                    return False, (
                        f"Draw integer mismatch: got {u}, claimed {claimed_draw_integer}"
                    )
                if action != claimed_action:
                    return False, (
                        f"Action mismatch: got {action}, claimed {claimed_action}"
                    )
                return True, "OK"
                break
        # No action found at this counter → it was a rejection (good for counter < claimed_rejections)

    return False, f"Could not reproduce draw after {claimed_rejection_count + 1} attempts"


# ===== Revision store reader (stdlib only) =====

def _load_revision_table(store_dir: Path, digest: str) -> dict[int, dict[int, Fraction]]:
    """Load and return a revision's policy table from the store directory."""
    path = store_dir / f"{digest}.json"
    if not path.exists():
        raise KeyError(f"Revision not found in store: {digest}")
    with open(path, "rb") as f:
        payload = json.loads(f.read())
    stored_digest = payload.get("digest", "")
    table = {
        int(s): {int(a): Fraction(p_str) for a, p_str in action_probs.items()}
        for s, action_probs in payload["table"].items()
    }
    # Re-derive and verify digest
    computed = hashlib.blake2b(_canonical_revision_bytes(table), digest_size=32).hexdigest()
    if computed != digest:
        raise ValueError(
            f"Revision store integrity failure: file digest={digest}, computed={computed}"
        )
    return table


# ===== Main verifier =====

class VerificationError(Exception):
    """Raised when a ledger record fails verification."""


def verify_action_record(
    rec: dict,
    store_dir: Path,
    seed: bytes,
    actor_id: int,
    episode_id: int,
    step_idx: int,
) -> None:
    """
    Verify a single action record. Raises VerificationError on any failure.

    Checks:
      1. Digest integrity: re-compute digest from fields.
      2. Behavior probability: re-derive from revision store.
      3. Draw verification: re-run BLAKE2b rejection sampling.
    """
    # 1. Digest integrity
    behavior_prob = Fraction(rec["behavior_prob"])
    env_reward = Fraction(rec["env_reward"])
    expected_digest = hashlib.blake2b(
        _record_canonical_bytes(
            rec["revision_digest"],
            rec["state"],
            rec["action"],
            behavior_prob,
            rec["draw_integer"],
            rec["rejection_count"],
            rec["env_next_state"],
            env_reward,
            rec["prev_digest"],
        ),
        digest_size=32,
    ).hexdigest()
    if expected_digest != rec["digest"]:
        raise VerificationError(
            f"Record digest mismatch at step {step_idx}: "
            f"expected={expected_digest}, stored={rec['digest']}"
        )

    # 2. Behavior probability from revision store
    table = _load_revision_table(store_dir, rec["revision_digest"])
    state = rec["state"]
    action = rec["action"]
    if state not in table:
        raise VerificationError(f"State {state} not in revision {rec['revision_digest']}")
    if action not in table[state]:
        raise VerificationError(f"Action {action} not in revision {rec['revision_digest']}")
    derived_prob = table[state][action]
    if derived_prob != behavior_prob:
        raise VerificationError(
            f"Behavior probability mismatch at step {step_idx}: "
            f"ledger claims {behavior_prob}, revision has {derived_prob}"
        )

    # 3. Draw verification (catches cross-revision forgeries)
    simplex = table[state]
    ok, reason = _verify_draw(
        simplex=simplex,
        seed=seed,
        actor_id=actor_id,
        episode=episode_id,
        step=step_idx,
        claimed_action=action,
        claimed_draw_integer=rec["draw_integer"],
        claimed_rejection_count=rec["rejection_count"],
    )
    if not ok:
        raise VerificationError(
            f"Draw verification failed at step {step_idx}: {reason}"
        )


def verify_trajectory(
    traj: dict,
    store_dir: Path,
    seed: bytes,
) -> list[str]:
    """
    Verify a complete trajectory. Returns a list of error strings (empty if OK).
    """
    errors = []
    actor_id = traj["actor_id"]
    episode_id = traj["episode_id"]
    records = traj["action_records"]

    # Verify chain integrity
    prev = records[0]["prev_digest"] if records else GENESIS_DIGEST
    for i, rec in enumerate(records):
        if rec["prev_digest"] != prev:
            errors.append(
                f"Chain break at step {i}: "
                f"expected prev={prev}, got {rec['prev_digest']}"
            )
        try:
            verify_action_record(
                rec, store_dir, seed, actor_id, episode_id, step_idx=i
            )
        except VerificationError as e:
            errors.append(str(e))
        prev = rec["digest"]

    # Verify terminal_digest
    if records:
        expected_terminal = records[-1]["digest"]
        if traj.get("terminal_digest") != expected_terminal:
            errors.append(
                f"Terminal digest mismatch: "
                f"expected {expected_terminal}, got {traj.get('terminal_digest')}"
            )

    return errors


def verify_ledger(
    ledger_dir: Path,
    store_dir: Path,
    seed: bytes,
) -> dict:
    """
    Verify all trajectories in a ledger directory.

    Returns a report dict with counts and any errors found.
    """
    report = {"total": 0, "passed": 0, "failed": 0, "errors": []}

    for path in sorted(ledger_dir.glob("actor*.json")):
        with open(path, "rb") as f:
            traj = json.loads(f.read())
        report["total"] += 1
        errors = verify_trajectory(traj, store_dir, seed)
        if errors:
            report["failed"] += 1
            report["errors"].append({"file": str(path.name), "errors": errors})
        else:
            report["passed"] += 1

    return report
