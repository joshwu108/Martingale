"""
server.py — `martingale serve`: the rollout inspector over tokens.db.

    from martingale.server import create_app
    app = create_app("./martingale_ws", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")   # -> uvicorn or TestClient

One page (static/inspector.html, no build step) over five JSON routes:
/api/doctor, /api/steps, /api/sequences, /api/sequence/{digest}, /api/record.
Exact values travel as "p/q" strings; `float_informational` fields are for display.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from martingale.inspector.routes_doctor import doctor_router
from martingale.inspector.routes_sequences import sequences_router
from martingale.inspector.tokenizer import load_tokenizer
from martingale.inspector.view import RecordView
from martingale.record.store import TokenLedger

PAGE = Path(__file__).parent / "static" / "inspector.html"


def create_app(workspace: str | Path, tokenizer: Any = None, head: str | None = None) -> FastAPI:
    """`tokenizer`: a transformers model name (loaded if transformers is installed), an object with decode(), or None."""
    ws = Path(workspace)
    db = ws / "tokens.db"
    if not db.exists():
        raise FileNotFoundError(f"no token record at {db}")
    view = RecordView(TokenLedger(db, readonly=True), load_tokenizer(tokenizer), head=head, workspace=ws)
    app = FastAPI(title="martingale inspector", version="0.2.0")
    app.state.inspector = view
    app.include_router(doctor_router(view))
    app.include_router(sequences_router(view))

    @app.get("/", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    return app
