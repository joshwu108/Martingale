"""inspector/routes_sequences.py — panels 2 and 3: /api/sequences and /api/sequence/{digest}."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

from martingale.inspector.view import RecordView

SortKey = Literal["advantage", "length", "mean_abs_log_ratio", "confident_disagreements"]


def sequences_router(view: RecordView) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/sequences")
    def sequences(step: int | None = None, lag: int = 0, sort: SortKey = "confident_disagreements",
                  order: Literal["desc", "asc"] = "desc", limit: int = Query(200, ge=1, le=10000),
                  offset: int = Query(0, ge=0)) -> dict[str, Any]:
        """One row per sequence, filtered by generation step, sorted exactly (Fractions) by the chosen key at `lag`."""
        rows = view.sequences(step=step, lag=lag, sort=sort, descending=(order == "desc"))
        return {"total": len(rows), "lags": view.derived().lags, "lag": lag, "sort": sort, "order": order,
                "offset": offset, "limit": limit,
                "rows": [r.to_json(view.decoder) for r in rows[offset:offset + limit]]}

    @router.get("/sequence/{digest}")
    def sequence(digest: str) -> dict[str, Any]:
        """Prompt, every token with its behavior log-prob, every trainer score by lag, and the confident-disagreement flag."""
        try:
            return view.sequence_detail(digest)
        except KeyError:
            raise HTTPException(status_code=404, detail="sequence not found") from None

    return router
