"""martingale.record — attested token-level record for LLM RL (design.md section 7)."""
from martingale.record.bits import GENESIS_DIGEST, bits_to_fraction, float_bits, parse_bits
from martingale.record.recorder import ProtocolViolation, Recorder, SequenceBuilder
from martingale.record.replay import OriginNotFound, ReplayProvenance
from martingale.record.revision import LLMRevision, SamplerConfig, tokenizer_digest, weights_digest
from martingale.record.store import TokenLedger
from martingale.record.tokens import ScoreRecord, SequenceRecord, TokenRecord, build_sequence, prompt_digest

__all__ = [
    "GENESIS_DIGEST", "LLMRevision", "OriginNotFound", "ProtocolViolation", "Recorder", "ReplayProvenance",
    "SamplerConfig", "ScoreRecord",
    "SequenceBuilder", "SequenceRecord", "TokenLedger", "TokenRecord", "bits_to_fraction",
    "build_sequence", "float_bits", "parse_bits", "prompt_digest", "tokenizer_digest", "weights_digest",
]
