# 0.2.0 release checklist

Prepared on 2026-10-07. This page is a checklist, not a publication record. Do not run the upload or post drafts until the release owner reviews them.

## Distribution name

The PyPI JSON API returned HTTP 404 for both `martingale` and `martingale-rl` on 2026-10-07. Since `martingale` was not taken, the distribution remains `martingale`; its Python import and CLI are also `martingale`. Recheck the name immediately before upload: availability can change, and a 404 does not guarantee PyPI will accept the project name. If it becomes unavailable, change only the distribution name to `martingale-rl` and update this page, README install commands, and CITATION.cff before rebuilding.

## Local gates

- [x] `make check`: 405 passed, 1 skipped; lint, mypy, and checker import isolation passed (2026-10-07).
- [x] `make release`: sdist and wheel built; `twine check dist/*` passed.
- [x] Wheel contains `checker/verify_tokens.py` and `martingale/static/inspector.html`; sdist contains them and excludes `planning/`.
- [x] Fresh scratch venv: `martingale demo --dir /tmp/x`, `martingale doctor --dir /tmp/x`, and `martingale verify --dir /tmp/x --head <demo head>` passed (48 sequences, 384 tokens, anchored).
- [x] `uv pip install --dry-run` resolved the wheel's `server` and `trl` extras together (68 packages). The TRL adapter is tested against TRL 1.13.0; this extra pins that version.
- [ ] Review the post and the two drafts under `docs/upstream/` against `paper/claim_evidence.md`.
- [x] All README relative links exist; external badge, CI, and uv installer links returned HTTP 200. LICENSE and CITATION.cff both say Apache-2.0; CITATION.cff has version 0.2.0 and no premature release date.
- [ ] Review suggested GitHub description/topics before publishing.
- [ ] Recheck PyPI name and confirm `dist/` contains only the intended 0.2.0 artifacts.

## Owner actions after review

Upload manually: `uv run --with twine twine upload dist/*`

Then verify the project page and fresh install, create the GitHub release if desired, and post the upstream comments yourself. No upload, GitHub post, issue, PR, or push is part of `make release`.

Suggested GitHub description: **Per-token diagnostics for RL training: separate rollout staleness from inference-engine and trainer mismatch.**

Suggested topics: `reinforcement-learning`, `rlhf`, `grpo`, `ppo`, `vllm`, `trl`, `off-policy`, `training-diagnostics`.
