"""
record/bits.py — exact float bit strings and canonical JSON.

A behavior log-prob is a float the inference engine produced. The record binds
its IEEE-754 bits, never a rounded rational:

    "f32:bf800000"   (float32, big-endian hex of the 4 bytes)
    "f64:bff0000000000000"

bits_to_fraction() converts back exactly (every finite float is a dyadic
rational). inf and nan are rejected loudly at this boundary.

checker/verify_tokens.py re-implements these two formats independently.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from fractions import Fraction

DIGEST_SIZE = 32
GENESIS_DIGEST = "0" * 64

_FORMATS = {"f32": (">f", 8), "f64": (">d", 16)}


def float_bits(value: float, dtype: str = "f32") -> str:
    """Encode a finite float as 'f32:<8 hex>' or 'f64:<16 hex>'. Rejects inf/nan."""
    if dtype not in _FORMATS:
        raise ValueError(f"dtype must be one of {sorted(_FORMATS)}, got {dtype!r}")
    value = float(value)
    if math.isinf(value) or math.isnan(value):
        raise ValueError(f"refusing to record non-finite float {value!r}")
    fmt, width = _FORMATS[dtype]
    try:
        packed = struct.pack(fmt, value)
    except OverflowError as exc:
        raise ValueError(f"{value!r} overflows {dtype}") from exc
    # float32 packing rounds; re-check finiteness after the cast
    if math.isinf(struct.unpack(fmt, packed)[0]):
        raise ValueError(f"{value!r} overflows {dtype}")
    return f"{dtype}:{packed.hex()}"


def parse_bits(bits: str) -> tuple[str, float]:
    """Decode a bit string to (dtype, float). Raises ValueError on malformed input."""
    if not isinstance(bits, str) or ":" not in bits:
        raise ValueError(f"malformed bits {bits!r}")
    dtype, hexpart = bits.split(":", 1)
    if dtype not in _FORMATS:
        raise ValueError(f"unknown bits dtype {dtype!r}")
    fmt, width = _FORMATS[dtype]
    if len(hexpart) != width or any(c not in "0123456789abcdef" for c in hexpart):
        raise ValueError(f"malformed {dtype} bits {bits!r}")
    value = struct.unpack(fmt, bytes.fromhex(hexpart))[0]
    if math.isinf(value) or math.isnan(value):
        raise ValueError(f"non-finite bits {bits!r}")
    return dtype, value


def bits_to_fraction(bits: str) -> Fraction:
    """Exact rational value of a bit string (Fraction(float) is exact for finite floats)."""
    _dtype, value = parse_bits(bits)
    return Fraction(value)


def canonical_json(obj) -> bytes:
    """Sorted keys, no whitespace, ASCII only. The only serialisation that is digested."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=DIGEST_SIZE).hexdigest()


def digest_json(obj) -> str:
    return digest_bytes(canonical_json(obj))
