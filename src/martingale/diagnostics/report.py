"""diagnostics/report.py — render the decomposition as markdown."""
from __future__ import annotations

from martingale.diagnostics.staleness import diagnosis


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
    lines += ["", "Exact values (reduced fractions) are in the JSON report; the table is float64 and informational."]
    return "\n".join(lines) + "\n"
