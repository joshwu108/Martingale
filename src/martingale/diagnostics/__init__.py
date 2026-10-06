"""martingale.diagnostics — staleness-versus-mismatch decomposition of a token record."""
from martingale.diagnostics.report import render_markdown
from martingale.diagnostics.staleness import DEFAULT_EPS, Bucket, ScoredToken, decompose, scored_tokens

__all__ = ["DEFAULT_EPS", "Bucket", "ScoredToken", "decompose", "render_markdown", "scored_tokens"]
