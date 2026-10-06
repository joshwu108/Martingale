# Real-run results (2026-10-06)

The `_t1.0` directory now holds the third run (fixed adapter: all four lags scored, full
`tokens.db`, `worst_tokens.json`). Its headline: TRL 1.13 colocated vLLM served the INITIAL
weights for the generations at steps 4 and 8 while the trainer trained (reference recompute with
the initial checkpoint agrees with vLLM to 0.001 and disagrees with the trainer by 0.8 to 1.0).

Both runs: Qwen2.5-0.5B-Instruct, TRL 1.13.0 GRPO, vLLM 0.28.0, 16 optimizer steps,
`num_iterations=2`, `steps_per_generation=2`, 8 generations per prompt, 2-digit addition
with exact-match reward, seed 0. Each directory holds the doctor report (`report.md`,
`report.json`), the anchored checker verdict (`verify.json`, `head.txt`), the alarms and
per-step metrics (`alarms.json`), TRL's log (`trainer_log.json`) and `config.json`.

| run | mode | temperature | lag-0 floor mean abs log r | lag-0 median ratio | lag-0 max abs log r | checker |
|---|---|---|---|---|---|---|
| `..._t1.0` | colocate, 1x A10G | 1.0 | 4.5e-01 (per generation: 2.7e-3, 0.81, 1.03, 0.15) | 1.0007 | 9.26 | ok, anchored |
| `..._t0.7` | server, 2x A10G | 0.7 | 6.1e-05 | 1.0000 | 0.001 | ok, anchored |

Known defects of the adapter version that produced these runs (fixed the same day):
- the row lookup consumed ids on first use, so the second `num_iterations` pass was not
  scored; the records hold lags 0 and 1 only and `alarms.json` shows `unmatched` errors;
- `tokens.db` was copied before the SQLite write-ahead log was checkpointed and was empty;
  the exports used by the checker and the reports were complete. The empty files were removed.
- `stale_server` fired on the t1.0 run's floor jumps and was then wrongly re-labelled as
  tail numerics because the median ratio stayed at 1. The recompute showed it WAS a stale
  engine. The alarm now uses confident disagreement (engine >= 50% sure, trainer > 2 nats
  lower), which numerics cannot produce; `floor_tail` covers the genuine tail case.
