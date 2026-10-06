# Real-run results (2026-10-06)

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
- `stale_server` fired on the t1.0 run's floor jumps; the median ratio shows those were
  tail tokens, not a shifted distribution. The alarm now requires the median to move and
  a `floor_tail` warning covers this case.
