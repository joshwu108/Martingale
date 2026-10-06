"""
diagnostics/reference.py — `martingale recompute`: re-score the record with a reference model.

Given a scoring function (or a HuggingFace checkpoint), compute the reference
log-prob of every recorded completion token and compare it with (a) the
behavior log-prob the engine reported and (b) the trainer's lag-0 score.
Per generation step this answers the question the live doctor can only
suspect: did the engine generate from the weights the trainer thinks it did?

    reference ~= engine,  reference != trainer   ->  engine was serving these (older) weights
    reference ~= trainer, reference != engine    ->  engine was serving something else
    reference ~= both                             ->  fine; the floor is numerics

This is design.md 7.6 tier 2: a report, never a forgery verdict. Needs
prompt_ids in the record (Recorder(keep_prompt_ids=True), the default).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

from martingale.record.bits import bits_to_fraction
from martingale.record.store import TokenLedger

ScoreFn = Callable[[Sequence[int], Sequence[int]], Sequence[float]]
"""(prompt_ids, completion_ids) -> reference log-prob of each completion token."""


@dataclass(frozen=True)
class GenerationAgreement:
    generation_step: int
    n_tokens: int
    mean_abs_ref_vs_engine: float
    mean_abs_ref_vs_trainer: float | None     # lag-0 trainer scores, if any
    mean_abs_engine_vs_trainer: float | None
    verdict: str


def _verdict(ref_eng: float, ref_tr: float | None, tol: float) -> str:
    if ref_tr is None:
        return "engine matches reference" if ref_eng <= tol else "engine differs from reference"
    eng_ok, tr_ok = ref_eng <= tol, ref_tr <= tol
    if eng_ok and tr_ok:
        return "engine and trainer both match the reference: floor is numerics"
    if eng_ok and not tr_ok:
        return "ENGINE GENERATED FROM THE REFERENCE WEIGHTS, TRAINER DID NOT: engine weights are stale (or the reference is the trainer's old checkpoint)"
    if tr_ok and not eng_ok:
        return "trainer matches the reference, engine does not: engine served different weights or inputs"
    return "neither matches the reference"


def recompute(ledger: TokenLedger, score_fn: ScoreFn, tolerance: float = 0.05) -> dict:
    """Score every sequence with prompt_ids; group agreement by the generation step."""
    revs = {r.digest: r for r in ledger.all_revisions()}
    per_gen: dict[int, dict[str, list[float]]] = {}
    skipped = 0
    for seq in ledger.all_sequences():
        if seq.prompt_ids is None:
            skipped += 1
            continue
        cids = [t.token_id for t in seq.tokens]
        ref = [float(x) for x in score_fn(list(seq.prompt_ids), cids)]
        if len(ref) != len(cids):
            raise ValueError("score_fn must return one log-prob per completion token")
        lag0 = {}
        for sc in ledger.scores_for(seq.digest):
            tok = seq.tokens[sc.position]
            if revs[sc.train_revision_digest].step == revs[tok.revision_digest].step:
                lag0[sc.position] = float(bits_to_fraction(sc.logprob_bits))
        g = revs[seq.tokens[0].revision_digest].step
        acc = per_gen.setdefault(g, {"ref_eng": [], "ref_tr": [], "eng_tr": []})
        for pos, tok in enumerate(seq.tokens):
            beh = float(bits_to_fraction(tok.logprob_bits))
            acc["ref_eng"].append(abs(ref[pos] - beh))
            if pos in lag0:
                acc["ref_tr"].append(abs(ref[pos] - lag0[pos]))
                acc["eng_tr"].append(abs(beh - lag0[pos]))
    rows = []
    for g in sorted(per_gen):
        a = per_gen[g]
        ref_eng = statistics.mean(a["ref_eng"]) if a["ref_eng"] else 0.0
        ref_tr = statistics.mean(a["ref_tr"]) if a["ref_tr"] else None
        eng_tr = statistics.mean(a["eng_tr"]) if a["eng_tr"] else None
        rows.append(GenerationAgreement(g, len(a["ref_eng"]), ref_eng, ref_tr, eng_tr, _verdict(ref_eng, ref_tr, tolerance)))
    return {"tolerance": tolerance, "sequences_skipped_without_prompt_ids": skipped,
            "generations": [r.__dict__ for r in rows]}


def hf_score_fn(model_id_or_path: str, dtype: str = "float32", device: str = "cpu") -> ScoreFn:
    """A ScoreFn backed by a HuggingFace causal LM (transformers + torch required)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, dtype=getattr(torch, dtype)).to(device).eval()

    def score(prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> list[float]:
        ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=device)
        with torch.no_grad():
            logits = model(ids).logits[0, len(prompt_ids) - 1: len(prompt_ids) - 1 + len(completion_ids)].float()
        lp = torch.log_softmax(logits, -1)
        return lp[torch.arange(len(completion_ids)), torch.tensor(list(completion_ids), device=device)].tolist()
    return score


def render(report: dict) -> str:
    lines = [f"# martingale recompute (tolerance {report['tolerance']} nats mean |diff|)", ""]
    if report["sequences_skipped_without_prompt_ids"]:
        lines.append(f"skipped {report['sequences_skipped_without_prompt_ids']} sequences without prompt_ids")
    lines += ["| generation step | tokens | ref vs engine | ref vs trainer (lag 0) | engine vs trainer | verdict |",
              "|---|---|---|---|---|---|"]
    for g in report["generations"]:
        rt = "n/a" if g["mean_abs_ref_vs_trainer"] is None else f"{g['mean_abs_ref_vs_trainer']:.4f}"
        et = "n/a" if g["mean_abs_engine_vs_trainer"] is None else f"{g['mean_abs_engine_vs_trainer']:.4f}"
        lines.append(f"| {g['generation_step']} | {g['n_tokens']} | {g['mean_abs_ref_vs_engine']:.4f} | {rt} | {et} | {g['verdict']} |")
    return "\n".join(lines) + "\n"
