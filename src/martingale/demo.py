"""
demo.py — `martingale demo`: the whole loop in under a minute, no GPU, no torch.

1. Publishes four LLM revisions (learner steps 0, 1, 2, 4) into a workspace.
2. Records 48 sequences from two actors, some spanning a weight update
   mid-sequence (mixed revision), with synthetic engine log-probs.
3. Scores every sequence at a later step with a lag-dependent drift plus a
   lag-0 engine-mismatch floor, as a trainer would.
4. Prints the ledger head, verifies the export with the independent checker
   anchored on that head, and prints the staleness-vs-mismatch report.
5. Certifies two estimators with the exact bench: sequence-level IS (unbiased)
   and per-token PPO clip (biased), printing one exact bias vector.

Everything is deterministic (hashlib-seeded, IEEE basic ops only, so the
ledger head is identical on every platform). The synthetic log-probs are
not a model; the point is to show what the record, checker and report look
like before you wire the TRL integration to a real run.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

from martingale.record import Recorder, SamplerConfig, bits_to_fraction

SAMPLER = SamplerConfig(temperature=1.0, top_p=1.0, max_tokens=8, logprobs_mode="engine_sampling",
                        dtype="bfloat16", engine="demo-engine", engine_version="0", seed_policy="demo")
STEPS = (0, 1, 2, 4)
N_SEQ = 48
LEN = 8


def _u(seed: bytes, *parts: int) -> float:
    h = hashlib.blake2b(seed + b":" + b",".join(str(p).encode() for p in parts), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def _sym(seed: bytes, *parts: int) -> float:
    """Uniform in [-1, 1) from the hash; only IEEE basic ops, so it is identical on every platform."""
    return 2.0 * _u(seed, *parts) - 1.0


def run_demo(workspace: Path, echo=print) -> dict:
    from checker.verify_tokens import verify_export
    from martingale.diagnostics import decompose, render_markdown
    from martingale.exact import check_unbiased, defendants

    seed = b"martingale-demo"
    workspace = Path(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"{workspace} is not empty; the demo needs a fresh workspace")
    rec = Recorder(workspace)
    revs = [rec.publish_revision(hashlib.blake2b(f"w{s}".encode(), digest_size=32).hexdigest(),
                                 b"demo tokenizer", SAMPLER, step=s) for s in STEPS]
    echo(f"published {len(revs)} revisions at learner steps {STEPS}")

    seqs = []
    for i in range(N_SEQ):
        gen = i % 4                                        # generated under revision index 0..3 (3 = lag 0)
        switch = LEN // 2 if i % 5 == 0 and gen < 3 else None   # some sequences cross a weight update
        with rec.sequence(actor_id=i % 2, sequence_id=f"prompt-{i}", prompt_ids=[1, 2, 3, i]) as s:
            for pos in range(LEN):
                r = revs[gen + 1] if switch is not None and pos >= switch else revs[gen]
                s.token(r.digest, 100 + (i * 7 + pos) % 50, -0.3 - 2.0 * _u(seed, i, pos))
            s.reward(float(i % 2))
        seqs.append((s.record, gen, switch))
    echo(f"recorded {N_SEQ} sequences x {LEN} tokens, {sum(1 for s, _, _ in seqs if s.is_mixed_revision)} mixed-revision")

    train = revs[3]                                         # every sequence is scored at step 4
    for i, (s, _gen, _switch) in enumerate(seqs):
        lps = []
        for pos, tok in enumerate(s.tokens):
            b = float(bits_to_fraction(tok.logprob_bits))
            lag = train.step - rec.ledger.get_revision(tok.revision_digest).step
            floor = 2e-6 * _sym(seed, i, pos, 9)                     # engine-vs-trainer mismatch
            drift = 0.15 * lag * _sym(seed, i, pos, 8) if lag else 0.0
            lps.append(b + floor + drift)
        rec.score(s.digest, train.digest, lps)
    head = rec.ledger.head()
    echo(f"scored every token at step {train.step}; ledger head {head}")

    export = rec.export_for_checker(workspace / "export")
    report = verify_export(export, expected_head=head)
    echo(f"independent checker: ok={report['ok']} revisions={report['n_revisions']} "
         f"sequences={report['n_sequences']} tokens={report['n_tokens']}")
    d = decompose(rec.ledger)
    echo("")
    echo(render_markdown(d))

    bench_ok = check_unbiased(defendants.seq_is, n_configs=2, lags=(1, 4), seed=seed)
    bench_ppo = check_unbiased(defendants.token_ppo_clip(Fraction(1, 5)), n_configs=2, lags=(1, 4), seed=seed)
    worst = max(bench_ppo.certificates, key=lambda c: c.rel_sq_bias or 0)
    echo(f"exact bench: seq_is unbiased in {bench_ok.n_unbiased}/{bench_ok.n_certificates} cells; "
         f"token_ppo_clip(1/5) unbiased in {bench_ppo.n_unbiased}/{bench_ppo.n_certificates}")
    echo(f"  worst token_ppo_clip cell (|S|={worst.n_states}, |A|={worst.n_actions}, H={worst.horizon}, lag={worst.lag}): "
         f"relative squared bias = {worst.rel_sq_bias} (exact)")
    return {"head": head, "verify": report, "report": d, "bench_seq_is": bench_ok.to_dict(),
            "bench_token_ppo_clip": bench_ppo.to_dict()}
