"""inspector/routes_doctor.py — panels 1 and 4: /api/doctor, /api/steps, /api/record."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from martingale.inspector.view import RecordView


def doctor_router(view: RecordView) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/doctor")
    def doctor() -> dict[str, Any]:
        """decompose() + attribute() + diagnosis() lines + the alarms a replayed MartingaleMonitor raises."""
        return view.derived().doctor

    @router.get("/steps")
    def steps() -> dict[str, Any]:
        """Per generation step: lag buckets, lag-0 floor, confident disagreements; plus the monitor replay per optimizer step."""
        d = view.derived()
        return {"lags": d.lags, "steps": d.steps, "replay": view.replay_summary()}

    @router.get("/record")
    def record() -> dict[str, Any]:
        """Counts, the ledger head, and the independent checker's verdict on a fresh export anchored on the stored head."""
        return view.record()

    return router
