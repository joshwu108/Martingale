"""Shared fixtures: the committed real record (skipped when absent) and a demo workspace built once."""
from pathlib import Path

import pytest

REAL_RECORD = Path(__file__).resolve().parent.parent / "benchmarks" / "modal" / "results" / \
    "trl_grpo_vllm_a10g_16steps_seed0_t1.0"

needs_real_record = pytest.mark.skipif(not (REAL_RECORD / "tokens.db").exists(),
                                       reason="real record benchmarks/modal/results/..._t1.0/tokens.db not present")


@pytest.fixture(scope="session")
def demo_ws(tmp_path_factory) -> Path:
    """A workspace written by `martingale demo` (48 sequences, 4 revisions, every token scored at step 4)."""
    from martingale.demo import run_demo
    ws = tmp_path_factory.mktemp("demo") / "ws"
    run_demo(ws, echo=lambda *a, **k: None)
    return ws
