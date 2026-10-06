"""Import guard for the TRL integration: one place that checks the installed TRL."""
from __future__ import annotations

from typing import Final, NamedTuple

PINNED_TRL_VERSION: Final[str] = "1.13.0"
TESTED_TRL_VERSIONS: Final[tuple[str, ...]] = ("1.13.0",)
REQUIRED_MEMBERS: Final[tuple[str, ...]] = (
    "_generate_and_score_completions",
    "_get_per_token_logps_and_entropies",
)


class TrlSupport(NamedTuple):
    grpo_trainer: type
    trainer_callback: type
    version: str


def require_trl() -> TrlSupport:
    """Import TRL and verify it exposes the two private members the adapter wraps."""
    try:
        import trl
        from transformers import TrainerCallback
        from trl import GRPOTrainer
    except ImportError as exc:
        raise ImportError(
            f"martingale.integrations.trl requires trl and transformers ({exc}); "
            f'install with: pip install "trl=={PINNED_TRL_VERSION}"'
        ) from exc
    version = str(getattr(trl, "__version__", "unknown"))
    missing = [m for m in REQUIRED_MEMBERS if not hasattr(GRPOTrainer, m)]
    if missing:
        raise ImportError(
            f"the installed TRL ({version}) is not supported: trl.GRPOTrainer has no "
            f"{', '.join(missing)}, which martingale.integrations.trl wraps"
        )
    if version not in TESTED_TRL_VERSIONS:
        import warnings
        warnings.warn(
            f"martingale.integrations.trl has been tested with TRL {', '.join(TESTED_TRL_VERSIONS)}; "
            f"the installed version is {version}", RuntimeWarning, stacklevel=2,
        )
    return TrlSupport(grpo_trainer=GRPOTrainer, trainer_callback=TrainerCallback, version=version)
