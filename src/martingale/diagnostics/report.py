"""diagnostics/report.py — render the decomposition as markdown."""
from __future__ import annotations

from martingale.diagnostics.staleness import _fmt, diagnosis


def render_markdown(d: dict) -> str:
    lines = ["# martingale doctor", ""]
    lines += [f"- {line}" for line in diagnosis(d)]
    lines += ["",
             f"sequences: {d['n_sequences']} (mixed-revision: {d['n_mixed_sequences']}, unscored: {d['n_unscored_sequences']}), "
             f"scored tokens: {d['n_scored_tokens']}", ""]
    if d["lag0_floor"] is None:
        lines.append("No lag-0 tokens: the engine-mismatch floor cannot be measured in this run.")
    else:
        f = d["lag0_floor"]["float_informational"]
        lines.append(f"Engine-mismatch floor (lag 0): mean |log r| = {f['mean_abs_log_ratio']:.3e}, "
                     f"max |log r| = {f['max_abs_log_ratio']:.3e}, ESS/n = {f['ess_fraction']:.3f}")
    lines += ["", "| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | "
              + " | ".join(f"clipped@{e}" for e in d["eps"]) + " |",
              "|---|---|---|---|---|---|---|---|" + "---|" * len(d["eps"])]
    for b in d["by_lag"]:
        f = b["float_informational"]
        clipped = " | ".join(f"{f['clipped_fraction'][str(e)]:.3f}" for e in d["eps"])
        lines.append(f"| {b['lag']} | {b['n_tokens']} | {b['n_sequences']} ({b['n_mixed_sequences']}) | "
                     f"{f['mean_log_ratio']:+.3e} | {f['mean_abs_log_ratio']:.3e} | {f['max_abs_log_ratio']:.3e} | "
                     f"{f['ratio_p95']:.4f} | {f['ess_fraction']:.3f} | {clipped} |")
    bp = d.get("by_provenance") or {}
    if "replayed" in bp:
        lines += ["", "Fresh vs replayed rows:", "",
                  "| provenance | lag | tokens | seqs | mean abs log r | ESS/n | floor source |", "|---|---|---|---|---|---|---|"]
        for prov in ("fresh", "replayed"):
            if prov not in bp:
                continue
            src = bp[prov]["attribution"].get("floor_source") or "none"
            for b in bp[prov]["by_lag"]:
                f = b["float_informational"]
                lines.append(f"| {prov} | {b['lag']} | {b['n_tokens']} | {b['n_sequences']} | {f['mean_abs_log_ratio']:.3e} | "
                             f"{f['ess_fraction']:.3f} | {src} |")
        r = d["replayed"]
        w = r["sequence_weights"]
        lines += ["", f"Replayed sequences: {r['n_sequences']} ({r['n_scored_sequences']} scored, {r['n_without_origin']} "
                  "without a fresh origin in this record)."]
        if w["n"]:
            lines.append(f"Sequence weights over {w['n']} rows ({w['n_overflow']} not representable in float64): declared "
                         f"(importance weight x rescale) ESS/n = {_fmt(w['declared_ess_fraction'], '.3f')}; measured from "
                         f"the trainer's scores ESS/n = {_fmt(w['measured_ess_fraction'], '.3f')}; dispersion of "
                         f"log(declared) - log(measured) after per-step centring: "
                         f"{_fmt(w['log_weight_dispersion'], '.3f')} nats over {w['n_dispersion_rows']} rows.")
    lines += ["", "Exact values (reduced fractions) are in the JSON report; the table is float64 and informational."]
    return "\n".join(lines) + "\n"
