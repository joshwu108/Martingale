"""
inspector/tokenizer.py — optional token decoding for the inspector.

`--tokenizer <name>` loads a transformers tokenizer when transformers is
installed; otherwise the page shows token ids only. transformers is never
imported at module load and never a dependency of the package.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

log = logging.getLogger("martingale")


class TokenDecoder:
    """Per-token and whole-sequence decoding over anything with a `decode(ids)` method."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        self._piece: dict[int, str] = {}

    def text(self, ids: Sequence[int]) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=False))

    def pieces(self, ids: Sequence[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if i not in self._piece:
                self._piece[i] = self.text([i])
            out.append(self._piece[i])
        return out


def load_tokenizer(spec: Any) -> TokenDecoder | None:
    """None -> None; an object with decode() -> wrapped; a name -> transformers.AutoTokenizer if available."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        if not hasattr(spec, "decode"):
            raise TypeError("tokenizer must be a model name or an object with a decode() method")
        return TokenDecoder(spec)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.warning("transformers is not installed; --tokenizer %s ignored, showing token ids only", spec)
        return None
    try:
        return TokenDecoder(AutoTokenizer.from_pretrained(spec))
    except Exception as exc:  # network, missing files: the inspector still works with ids
        log.warning("could not load tokenizer %s (%s); showing token ids only", spec, exc)
        return None
