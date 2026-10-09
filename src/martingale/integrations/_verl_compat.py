"""Pinned verl import guard. The recorder itself has no verl import."""
from __future__ import annotations

from typing import Final, NamedTuple

PINNED_VERL_VERSION: Final[str] = "0.9.1"


class VerlSupport(NamedTuple):
    data_proto: type
    ppo_loss: object
    no_padding_2_padding: object
    version: str


def require_verl() -> VerlSupport:
    try:
        from importlib.metadata import version

        from verl.protocol import DataProto
        from verl.workers.utils.losses import ppo_loss
        from verl.workers.utils.padding import no_padding_2_padding
    except ImportError as exc:
        raise ImportError(
            f'martingale.integrations.verl requires verl=={PINNED_VERL_VERSION}; '
            f'install with: pip install "verl=={PINNED_VERL_VERSION}" ({exc})'
        ) from exc
    installed = version("verl")
    if installed != PINNED_VERL_VERSION:
        raise ImportError(f"verl {installed} is unsupported; install verl=={PINNED_VERL_VERSION}")
    if not all(hasattr(DataProto, name) for name in ("reorder", "to_tensordict")):
        raise ImportError("verl DataProto lacks required batch methods")
    return VerlSupport(DataProto, ppo_loss, no_padding_2_padding, installed)
