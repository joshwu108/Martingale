# 0.2.0 release checklist

Prepared on 2026-10-07. This page is a checklist, not a publication record. Do not run the upload or post drafts until the release owner reviews them.

## Distribution name

The PyPI JSON API returned HTTP 404 for both `martingale` and `martingale-rl` on 2026-10-10 and again on 2026-10-13 (the simple index also 404s for `martingale`). Since `martingale` was not taken, the distribution remains `martingale`; its Python import and CLI are also `martingale`. Recheck the name immediately before upload: availability can change, and a 404 does not guarantee PyPI will accept the project name. If it becomes unavailable, change only the distribution name to `martingale-rl` and update this page, README install commands, and CITATION.cff before rebuilding.

## Local gates

- [x] `make check`: 405 passed, 1 skipped; lint, mypy, and checker import isolation passed (2026-10-07).
- [x] `make release`: sdist and wheel built; `twine check dist/*` passed.
- [x] Wheel contains `checker/verify_tokens.py` and `martingale/static/inspector.html`; sdist contains them and excludes `planning/`.
- [x] Fresh scratch venv: `martingale demo --dir /tmp/x`, `martingale doctor --dir /tmp/x`, and `martingale verify --dir /tmp/x --head <demo head>` passed (48 sequences, 384 tokens, anchored).
- [x] `uv pip install --dry-run` resolved the wheel's `server` and `trl` extras together (68 packages). The TRL adapter is tested against TRL 1.13.0; this extra pins that version.
- [x] Reviewed the post and both drafts under `docs/upstream/` against `paper/claim_evidence.md` and `benchmarks/modal/results/README.md`; corrected scope and venue status (2026-10-10).
- [x] All README relative links exist; external badge, CI, and uv installer links returned HTTP 200. LICENSE and CITATION.cff both say Apache-2.0; CITATION.cff has version 0.2.0 and no premature release date.
- [x] Reviewed and finalized the GitHub description/topics below (2026-10-10).
- [x] Rechecked both PyPI names via the JSON API (HTTP 404), ran `make release`, confirmed `dist/` contains only `martingale-0.2.0.tar.gz` and `martingale-0.2.0-py3-none-any.whl`, and passed `demo`, `doctor`, and anchored `verify` from a fresh wheel-only scratch venv (2026-10-10).

### Re-run 2026-10-13

- [x] `make check`: import isolation, ruff, mypy passed; 580 passed, 4 skipped (`trl`, `verl`, `tensordict` not installed). Includes `tests/test_committed_records.py`: every committed record verifies anchored on its `head.txt`, opened read-only, with no file in its workspace changed; the `_t0.7` record was found empty, removed, and documented as lost in `benchmarks/modal/results/README.md`.
- [x] `rm dist/*; make release`: sdist and wheel built; `twine check dist/*` passed. `dist/` holds `martingale-0.2.0.tar.gz`, `martingale-0.2.0-py3-none-any.whl` and the `.gitignore` uv writes there (not matched by `dist/*`).
- [x] Wheel contains `checker/verify_tokens.py` and `martingale/static/inspector.html`; sdist contains both and no `planning/` entry.
- [x] Fresh wheel-only scratch venv: `demo`, `doctor`, and `verify --head <demo head>` passed (4 revisions, 48 sequences, 384 tokens, anchored); `verify` with a wrong head exits 1.
- [x] `uv pip install --dry-run '<wheel>[server,trl]'` resolved (66 packages on top of the installed wheel; `trl==1.13.0`, `torch==2.14.1`).
- [x] README: 11 relative links exist; badge, CI, shields and uv installer URLs returned HTTP 200. LICENSE, `pyproject.toml` and CITATION.cff say Apache-2.0; CITATION.cff version 0.2.0, no release date.
- [x] PyPI JSON API: `martingale` and `martingale-rl` both HTTP 404.
- [x] GitHub description/topics below unchanged. The public API shows none set on `joshwu108/Martingale` yet: applying them is an owner action.
- [x] Post and both `docs/upstream/` drafts re-reviewed against `paper/claim_evidence.md`, the results README, the result JSON files and `docs/nonclaims.md`: every number matches. Fixed the post's "no no-op" typo. Venues via the GitHub API: TRL #5974 and #5195 closed (not planned), discussion #5509 exists, #6945 and #6688 open; verl #7735 closed and PR #8082 merged 2026-10-09 (both now under `verl-project/verl`; the old links redirect).
- [ ] Owner decisions before posting (not applied; `docs/nonclaims.md` was being edited by another session):
  - Post :19-21 and both drafts: add the sync-exactness scope (tensor parallel 1, no sleep mode, no quantization) and say the bf16 floors at steps 8 and 12 were measured on a collapsing policy.
  - `trl-model-version.md:11`: `behavior_logprob_bits` is `logprob_bits` in `design.md` §7.4; the score also binds the sequence digest; if `logprobs_mode` is raised on #6945, add that in server mode it is TRL's flag, not observed from the server, and that the cited runs predate the binding.
  - `verl-7735.md`: cites only the TRL run; add the verl 0.9.1 run (8 steps, 1x T4, lag-0 floor 2.0e-4, zero confident disagreements, served weights not compared) without comparing its floor to TRL's; confirm `selected_token_logprobs` against #8082; update links to `verl-project/verl`.
  - Run counts are stale in `docs/nonclaims.md` ("run four times for real", "four real runs") and `paper/claim_evidence.md:31` ("4 real runs", evidence glob now includes `_t0.7`, whose record is lost).

## Owner actions after review

Upload manually, from the commit that holds this page (the 2026-10-13 `dist/` was built from the working tree before that commit; rebuild if anything else changes):

```sh
cd /Users/joshuawu/Martingale
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/martingale/json   # expect 404
rm -f dist/*.whl dist/*.tar.gz && make release
uv run --with twine twine upload dist/martingale-0.2.0.tar.gz dist/martingale-0.2.0-py3-none-any.whl
```

Then verify the project page and fresh install, create the GitHub release if desired, and post the upstream comments yourself. No upload, GitHub post, issue, PR, or push is part of `make release`.

Final GitHub description: **Per-token diagnostics for RL training: separate rollout staleness from inference-engine and trainer mismatch.**

Final GitHub topics: `reinforcement-learning`, `rlhf`, `grpo`, `ppo`, `vllm`, `trl`, `off-policy`, `training-diagnostics`.
