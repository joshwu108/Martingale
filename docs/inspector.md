# The rollout inspector (`martingale serve`)

One page over a `tokens.db`: the diagnosis, the sequences, every token of one
sequence as a log-ratio heatmap, and whether the record verifies. It makes the
analysis in the README's "First real run" a click.

```bash
uv sync --extra server
uv run martingale serve --dir ./martingale_ws                     # http://127.0.0.1:7373
uv run martingale serve --dir ./martingale_ws --tokenizer Qwen/Qwen2.5-0.5B-Instruct
uv run martingale serve --dir benchmarks/modal/results/trl_grpo_vllm_a10g_16steps_seed0_t1.0
```

- `--dir` holds `tokens.db` (a TRL run's `MartingaleRecorder` workspace, `martingale demo --dir ws`,
  or a copied result directory).
- `--tokenizer` decodes token ids into text through `transformers.AutoTokenizer` when
  transformers is installed; without it the page shows ids only. transformers is never a
  dependency of martingale (`uv run --with transformers martingale serve ...` works).
- `--head` anchors the checker on the ledger head the trainer printed. Without it the
  inspector looks for `head.txt` or `martingale_head.txt` in the workspace, then the newest
  `checkpoint-*/martingale_head.txt`; with none of those the checker runs unanchored and the
  record card says so.
- `--host`, `--port` as before. The old Observatory flags (`--scan-interval`, `--seed`,
  `--halt-on-forgery`) are accepted and ignored.

## The four panels

1. **Diagnosis.** The `martingale doctor` lines, the per-lag table (exact values in the JSON,
   float64 on screen), the alarms a `MartingaleMonitor` raises when replayed over the record
   step by step, and three sparklines per generation step: lag-0 floor, ESS/n at the selected
   lag, confident disagreements at lag 0. The lag selector (one accent colour) drives the
   table, the sequence columns and the heatmap. Clicking a sparkline point filters the
   sequences to that generation step.
2. **Sequences.** One row per sequence: generation step, id, length, advantage (TRL's
   `reward_bits`), mean |log r| at the selected lag, confident disagreements at that lag,
   mixed-revision flag, decoded completion. Column headers sort (exact Fraction compares on
   the server; sequences unscored at the lag sort last). A row opens panel 3.
3. **Sequence detail.** Prompt (ids and text when recorded), then every token as a monospace
   cell coloured by |log r| at the selected lag, with a red outline on confident
   disagreements (engine >= 50% sure, trainer > 2 nats lower). Hover shows the engine
   log-prob and every trainer score by lag. Below, a per-lag matrix with the numbers.
4. **Record and trust.** Revisions, sequences, tokens, scores, the ledger head, and the
   independent checker's verdict (`checker/verify_tokens.py` on a fresh export, anchored on
   the stored head).

The view state (lag, step, sort, selected sequence) lives in the URL hash, so a link to a
finding can be shared.

## API

| route | returns |
|---|---|
| `GET /api/doctor` | `decompose()` + `attribute()` + `diagnosis()` lines + replayed alarms + `AlarmConfig` |
| `GET /api/steps` | per generation step: lag buckets, lag-0 floor, confident disagreements; per optimizer step: the replay |
| `GET /api/sequences?step=&lag=&sort=advantage\|length\|mean_abs_log_ratio\|confident_disagreements&order=&limit=&offset=` | one row per sequence |
| `GET /api/sequence/{digest}` | prompt, tokens with behavior log-prob, trainer scores by lag, log-ratios, confident flag |
| `GET /api/record` | counts, ledger head, stored head and its source, checker result |

Exact values are `"p/q"` strings; anything under `float_informational` is float64 for display
only, following the rule that no float sits on a decision path.

## Replaying the monitor

`martingale.diagnostics.replay.replay(ledger)` groups the scores into contiguous runs by the
train revision's step and folds run *k* into an `IncrementalDiagnosis` as step *k+1* (TRL
increments `global_step` before the callback fires), evaluating the alarms with the same
`AlarmHistory` the live callback keeps. On the real record this raises `stale_server` at
steps 5 and 9 and `floor_tail` at step 13.

## Screenshots

`docs/img/inspector-overview.png` and `docs/img/inspector-sequence-172.png` were taken at
1280 px wide against the real record with the Qwen2.5 tokenizer.
