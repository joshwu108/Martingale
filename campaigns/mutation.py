"""
campaigns/mutation.py — ledger-forgery campaign vs checker/verify.py.

Generates ≥60 single-fault mutants and verifies that the independent checker
rejects all of them (100% rejection rate).

Mutant classes tested:
  1. tampered_action         — flip the chosen action
  2. tampered_behavior_prob  — change behavior_prob to wrong value
  3. tampered_digest         — corrupt record digest
  4. broken_chain            — wrong prev_digest
  5. tampered_draw_integer   — change draw_integer
  6. tampered_rejection_count — change rejection_count
  7. tampered_next_state     — change env_next_state
  8. tampered_reward         — change env_reward
  9. tampered_revision       — claim a different revision digest
  10. cross_revision_same_prob — cross-revision forgery with same probability (subtle)
  11. replay_attack           — reuse a record from a different episode
  12. terminal_digest_mismatch — wrong terminal_digest in trajectory

Results committed to results/mutation_report.json.
"""
from __future__ import annotations

import copy
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from martingale.draw import draw_action
from martingale.ledger import ActionRecord, Ledger, GENESIS_DIGEST
from martingale.revision import RevisionStore

# Import checker without importing martingale
sys.path.insert(0, str(Path(__file__).parent.parent))
from checker.verify import verify_trajectory

RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED = b"mutation-campaign-seed-v1"


def _build_base_traj(tmp_path: Path, n_steps: int = 5):
    """Build a valid trajectory with n_steps steps."""
    store_dir = tmp_path / "store"
    ledger_dir = tmp_path / "ledger"
    store = RevisionStore(store_dir)
    ledger = Ledger(ledger_dir, store)

    probs_r  = {0: Fraction(0),   1: Fraction(1, 2), 2: Fraction(1, 2)}
    probs_r1 = {0: Fraction(1, 2), 1: Fraction(1, 2), 2: Fraction(0)}
    rev = store.publish({0: {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}})
    rev_other = store.publish({0: {0: Fraction(1, 2), 1: Fraction(1, 4), 2: Fraction(1, 4)}})
    rev_r = store.publish({0: probs_r})
    rev_r1 = store.publish({0: probs_r1})

    probs = {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}
    records = []
    prev = GENESIS_DIGEST
    for step in range(n_steps):
        draw = draw_action(probs, seed=SEED, actor_id=0, episode=0, step=step)
        rec = ActionRecord(
            revision_digest=rev.digest,
            state=0, action=draw.action,
            behavior_prob=probs[draw.action],
            draw=draw,
            env_next_state=0, env_reward=Fraction(step),
            prev_digest=prev,
        )
        records.append(rec)
        prev = rec.digest

    traj = ledger.append_trajectory(actor_id=0, episode_id=0, records=records)
    return traj.to_dict(), store_dir, ledger_dir, rev_other.digest, rev_r, rev_r1, probs_r1


def _verify_is_rejected(traj_dict: dict, store_dir: Path, label: str) -> dict:
    """Verify that a mutant trajectory is rejected. Returns result dict."""
    errors = verify_trajectory(traj_dict, store_dir, seed=SEED)
    return {
        "mutant": label,
        "rejected": bool(errors),
        "errors": errors[:1] if errors else [],  # first error only
    }


def _other_action(action: int, n: int = 3) -> int:
    return (action + 1) % n


def run_mutation_campaign(tmp_path: Path) -> dict:
    """Run ≥60 single-fault mutants and return a report."""
    traj_dict, store_dir, ledger_dir, other_rev_digest, rev_r, rev_r1, probs_r1 = \
        _build_base_traj(tmp_path)

    mutants = []
    results = []
    n_records = len(traj_dict["action_records"])

    for step_idx in range(n_records):
        base = json.loads(json.dumps(traj_dict))  # deep copy

        # 1. tampered_action
        m = json.loads(json.dumps(traj_dict))
        old_action = m["action_records"][step_idx]["action"]
        m["action_records"][step_idx]["action"] = _other_action(old_action)
        results.append(_verify_is_rejected(m, store_dir, f"tampered_action[{step_idx}]"))

        # 2. tampered_behavior_prob
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][step_idx]["behavior_prob"] = "999/1000"
        results.append(_verify_is_rejected(m, store_dir, f"tampered_behavior_prob[{step_idx}]"))

        # 3. tampered_digest
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][step_idx]["digest"] = "a" * 64
        results.append(_verify_is_rejected(m, store_dir, f"tampered_digest[{step_idx}]"))

        # 4. broken_chain (wrong prev_digest)
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][step_idx]["prev_digest"] = "b" * 64
        results.append(_verify_is_rejected(m, store_dir, f"broken_chain[{step_idx}]"))

        # 5. tampered_draw_integer
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][step_idx]["draw_integer"] = \
            m["action_records"][step_idx]["draw_integer"] ^ (1 << 128)
        results.append(_verify_is_rejected(m, store_dir, f"tampered_draw_integer[{step_idx}]"))

        # 6. tampered_rejection_count
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][step_idx]["rejection_count"] = \
            m["action_records"][step_idx]["rejection_count"] + 1
        results.append(_verify_is_rejected(m, store_dir, f"tampered_rejection_count[{step_idx}]"))

        # 7. tampered_next_state
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][step_idx]["env_next_state"] = \
            1 - m["action_records"][step_idx]["env_next_state"]
        results.append(_verify_is_rejected(m, store_dir, f"tampered_next_state[{step_idx}]"))

        # 8. tampered_reward
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][step_idx]["env_reward"] = "7/3"
        results.append(_verify_is_rejected(m, store_dir, f"tampered_reward[{step_idx}]"))

        # 9. tampered_revision (claim a different, valid revision)
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][step_idx]["revision_digest"] = other_rev_digest
        results.append(_verify_is_rejected(m, store_dir, f"tampered_revision[{step_idx}]"))

        # 10. replay_attack (reuse record from step 0 at step_idx)
        if step_idx > 0:
            m = json.loads(json.dumps(traj_dict))
            replayed = json.loads(json.dumps(traj_dict["action_records"][0]))
            m["action_records"][step_idx] = replayed
            results.append(_verify_is_rejected(m, store_dir, f"replay_attack[{step_idx}]"))

    # 11. terminal_digest_mismatch
    m = json.loads(json.dumps(traj_dict))
    m["terminal_digest"] = "c" * 64
    results.append(_verify_is_rejected(m, store_dir, "terminal_digest_mismatch"))

    # 13. Swap two records (out-of-order steps)
    if n_records >= 2:
        for i in range(n_records - 1):
            m = json.loads(json.dumps(traj_dict))
            m["action_records"][i], m["action_records"][i + 1] = \
                m["action_records"][i + 1], m["action_records"][i]
            results.append(_verify_is_rejected(m, store_dir, f"swapped_records[{i},{i+1}]"))

    # 14. Drop a record (truncated trajectory)
    for i in range(n_records):
        m = json.loads(json.dumps(traj_dict))
        m["action_records"] = m["action_records"][:i] + m["action_records"][i+1:]
        if m["action_records"]:
            results.append(_verify_is_rejected(m, store_dir, f"dropped_record[{i}]"))

    # 15. Duplicate a record (repeated step)
    for i in range(n_records):
        m = json.loads(json.dumps(traj_dict))
        dup = json.loads(json.dumps(m["action_records"][i]))
        m["action_records"].insert(i, dup)
        results.append(_verify_is_rejected(m, store_dir, f"duplicated_record[{i}]"))

    # 16. Zero reward forged to nonzero
    for i in range(n_records):
        m = json.loads(json.dumps(traj_dict))
        m["action_records"][i]["env_reward"] = "100/1"
        results.append(_verify_is_rejected(m, store_dir, f"inflated_reward[{i}]"))

    # 17. Actor ID mismatch in trajectory wrapper
    m = json.loads(json.dumps(traj_dict))
    m["actor_id"] = 999
    results.append(_verify_is_rejected(m, store_dir, "actor_id_mismatch"))

    # 18. Missing revision from store (nonexistent digest)
    m = json.loads(json.dumps(traj_dict))
    m["action_records"][0]["revision_digest"] = "d" * 64
    results.append(_verify_is_rejected(m, store_dir, "nonexistent_revision"))

    # 12. Cross-revision forgery (subtle: same prob, different boundary)
    # Draw action=1 under r+1 simplex, forge as revision r
    for ep in range(500):
        d = draw_action(probs_r1, seed=SEED, actor_id=99, episode=ep, step=0)
        if d.action == 1:
            # Forge: claim rev_r (action 1 boundary [0,1/2)) but draw is from r+1 ([1/2,1))
            rec_dict = {
                "revision_digest": rev_r.digest,
                "state": 0,
                "action": 1,
                "behavior_prob": "1/2",
                "draw_integer": d.draw_integer,
                "rejection_count": d.rejection_count,
                "env_next_state": 0,
                "env_reward": "1/1",
                "prev_digest": GENESIS_DIGEST,
            }
            import hashlib
            from checker.verify import _record_canonical_bytes
            rec_dict["digest"] = hashlib.blake2b(
                _record_canonical_bytes(
                    rec_dict["revision_digest"],
                    rec_dict["state"], rec_dict["action"],
                    Fraction("1/2"),
                    rec_dict["draw_integer"], rec_dict["rejection_count"],
                    rec_dict["env_next_state"], Fraction("1/1"),
                    rec_dict["prev_digest"],
                ),
                digest_size=32,
            ).hexdigest()
            cross_traj = {
                "actor_id": 99, "episode_id": ep,
                "action_records": [rec_dict],
                "terminal_digest": rec_dict["digest"],
                "prev_trajectory_digest": GENESIS_DIGEST,
            }
            results.append(
                _verify_is_rejected(cross_traj, store_dir,
                                    f"cross_revision_same_prob[ep={ep}]")
            )
            break

    n_total = len(results)
    n_rejected = sum(1 for r in results if r["rejected"])
    surviving = [r for r in results if not r["rejected"]]

    report = {
        "campaign": "mutation",
        "n_mutants": n_total,
        "n_rejected": n_rejected,
        "n_surviving": len(surviving),
        "rejection_rate_pct": 100 * n_rejected / n_total if n_total else 0,
        "surviving_mutants": surviving,
    }
    return report


def main(tmp_path: Path = None) -> None:
    import tempfile
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp(prefix="martingale_mutation_"))
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"=== Mutation Campaign ===")
    print(f"Working directory: {tmp_path}")

    report = run_mutation_campaign(tmp_path)

    out = RESULTS_DIR / "mutation_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {out}")

    print(f"\nMutants: {report['n_mutants']}")
    print(f"Rejected: {report['n_rejected']} ({report['rejection_rate_pct']:.1f}%)")
    print(f"Surviving: {report['n_surviving']}")

    if report["surviving_mutants"]:
        print("\n!!! SURVIVING MUTANTS (checker failed to detect) !!!")
        for m in report["surviving_mutants"]:
            print(f"  {m['mutant']}")
        sys.exit(1)
    else:
        print(f"\nPASS: All {report['n_mutants']} mutants rejected (100%)")


if __name__ == "__main__":
    main()
