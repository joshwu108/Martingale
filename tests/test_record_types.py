"""Tests for martingale.record bits, LLMRevision, TokenRecord, SequenceRecord, ScoreRecord."""
import math
from fractions import Fraction

import pytest

from martingale.record.bits import (
    GENESIS_DIGEST,
    bits_to_fraction,
    canonical_json,
    digest_json,
    float_bits,
    parse_bits,
)
from martingale.record.revision import LLMRevision, SamplerConfig, tokenizer_digest, weights_digest
from martingale.record.tokens import ScoreRecord, SequenceRecord, TokenRecord, build_sequence, prompt_digest

W = "a" * 64
T = "b" * 64


class TestBits:
    @pytest.mark.parametrize("v", [0.0, -0.0, 1.0, -1.5, 1e-30, -123.456, 3.0e38])
    def test_f32_round_trip_is_exact_for_f32_values(self, v):
        import struct
        v32 = struct.unpack(">f", struct.pack(">f", v))[0]
        assert parse_bits(float_bits(v32, "f32"))[1] == v32
        assert bits_to_fraction(float_bits(v32, "f32")) == Fraction(v32)

    def test_f64_round_trip(self):
        v = -0.6931471805599453
        assert parse_bits(float_bits(v, "f64")) == ("f64", v)
        assert bits_to_fraction(float_bits(v, "f64")) == Fraction(v)

    def test_bits_are_exact_not_limit_denominator(self):
        v = -0.1
        f = bits_to_fraction(float_bits(v, "f64"))
        assert f != Fraction(1, 10) * -1          # the float is not exactly -1/10
        assert f == Fraction(v)

    @pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
    def test_non_finite_rejected(self, bad):
        with pytest.raises(ValueError):
            float_bits(bad)

    def test_f32_overflow_rejected(self):
        with pytest.raises(ValueError):
            float_bits(1e300, "f32")

    @pytest.mark.parametrize("bad", ["f32:zz", "f16:0000", "f32:00000000ff", "nope", 12])
    def test_malformed_bits_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_bits(bad)

    def test_canonical_json_is_sorted_and_compact(self):
        assert canonical_json({"b": 1, "a": [1, {"z": None, "y": "é"}]}) == b'{"a":[1,{"y":"\\u00e9","z":null}],"b":1}'


class TestLLMRevision:
    def sampler(self, **kw):
        base = dict(temperature=1.0, top_p=1.0, engine="vllm", engine_version="0.28.0", logprobs_mode="raw")
        base.update(kw)
        return SamplerConfig(**base)

    def test_digest_is_over_manifest_and_deterministic(self):
        a = LLMRevision(W, T, self.sampler(), step=3)
        b = LLMRevision(W, T, self.sampler(), step=3, weights_uri="s3://x")
        assert a.digest == b.digest == digest_json(a.manifest())
        assert "weights_uri" not in a.manifest()

    @pytest.mark.parametrize("change", [dict(step=4), dict(parent_digest="c" * 64)])
    def test_step_and_parent_are_inside_digest(self, change):
        base = dict(weights_digest=W, tokenizer_digest=T, sampler=self.sampler(), step=3)
        assert LLMRevision(**base).digest != LLMRevision(**{**base, **change}).digest

    def test_any_sampler_key_change_is_a_new_revision(self):
        a = LLMRevision(W, T, self.sampler(), step=0)
        for k, v in [("temperature", 0.7), ("top_k", 50), ("logprobs_mode", "processed"), ("tensor_parallel", 2)]:
            assert LLMRevision(W, T, self.sampler(**{k: v}), step=0).digest != a.digest

    def test_round_trip_and_tamper_detection(self):
        rev = LLMRevision(W, T, self.sampler(), step=7, parent_digest="c" * 64, weights_uri="file:///w")
        d = rev.to_dict()
        assert LLMRevision.from_dict(d) == rev
        d["step"] = 8
        with pytest.raises(ValueError, match="digest mismatch"):
            LLMRevision.from_dict(d)

    def test_rejects_bad_hex_and_negative_step(self):
        with pytest.raises(ValueError):
            LLMRevision("abc", T, self.sampler(), step=0)
        with pytest.raises(ValueError):
            LLMRevision(W, T, self.sampler(), step=-1)

    def test_sampler_rejects_unknown_keys(self):
        with pytest.raises(ValueError):
            SamplerConfig.from_dict({"temperature": 1.0, "beam_width": 4})


class TestWeightsDigest:
    def test_raw_triple_path_is_torch_free_and_order_independent(self):
        sd1 = {"b": ("float32", (2,), b"\x00\x00\x80\x3f\x00\x00\x00\x40"), "a": ("int8", (1,), b"\x07")}
        sd2 = dict(reversed(list(sd1.items())))
        assert weights_digest(sd1) == weights_digest(sd2)
        assert weights_digest({"a": sd1["a"]}) != weights_digest(sd1)

    def test_dtype_and_shape_are_bound(self):
        raw = b"\x00" * 8
        assert weights_digest({"w": ("float32", (2,), raw)}) != weights_digest({"w": ("float64", (1,), raw)})
        assert weights_digest({"w": ("float32", (2,), raw)}) != weights_digest({"w": ("float32", (1, 2), raw)})

    def test_torch_tensor_matches_raw_bytes_and_is_stable_across_torch_save(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        raw = t.numpy().astype("<f4").tobytes()
        assert weights_digest({"w": t}) == weights_digest({"w": ("float32", (2,), raw)})
        assert weights_digest({"w": t.clone()}) == weights_digest({"w": t})

    def test_bf16_is_digested_in_its_own_dtype(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([1.0, 2.0]).to(torch.bfloat16)
        assert weights_digest({"w": t}) != weights_digest({"w": t.float()})

    def test_tokenizer_digest_bytes_and_chunks_agree(self):
        assert tokenizer_digest(b"abc") == tokenizer_digest([b"a", b"bc"])


def _rev(step=0):
    return LLMRevision(W, T, SamplerConfig(temperature=1.0), step=step)


class TestTokenAndSequenceRecords:
    def test_build_sequence_chains_from_prev_sequence(self):
        r = _rev()
        seq = build_sequence(0, 0, "p0", [5, 6, 7],
                             [(r.digest, 11, float_bits(-0.5), None), (r.digest, 12, float_bits(-1.0), None)])
        assert seq.tokens[0].prev_digest == GENESIS_DIGEST
        assert seq.tokens[1].prev_digest == seq.tokens[0].digest
        assert seq.terminal_digest == seq.tokens[1].digest
        assert seq.prompt_digest == prompt_digest([5, 6, 7]) and seq.prompt_len == 3
        assert not seq.is_mixed_revision

    def test_mixed_revision_sequence_is_first_class(self):
        a, b = _rev(0), _rev(1)
        seq = build_sequence(0, 0, "p", [1], [(a.digest, 1, float_bits(-0.1), None),
                                            (b.digest, 2, float_bits(-0.2), None)])
        assert seq.is_mixed_revision and seq.revision_digests == (a.digest, b.digest)

    def test_round_trip_and_every_field_is_tamper_evident(self):
        r = _rev()
        seq = build_sequence(1, 4, "p", [1, 2], [(r.digest, 9, float_bits(-0.3), ((9, float_bits(-0.3)), (4, float_bits(-2.0))))],
                             reward_bits=float_bits(1.0), engine_request_id="req-1")
        d = seq.to_dict()
        assert SequenceRecord.from_dict(d) == seq
        for mutate in (lambda x: x["tokens"][0].__setitem__("token_id", 10),
                       lambda x: x["tokens"][0].__setitem__("logprob_bits", float_bits(-0.31)),
                       lambda x: x["tokens"][0].__setitem__("revision_digest", "d" * 64),
                       lambda x: x["tokens"][0]["topk"][1].__setitem__(0, 5),
                       lambda x: x.__setitem__("reward_bits", float_bits(0.0)),
                       lambda x: x.__setitem__("prompt_len", 3),
                       lambda x: x.__setitem__("sequence_index", 5)):
            bad = SequenceRecord.from_dict(seq.to_dict()).to_dict()
            mutate(bad)
            with pytest.raises(ValueError):
                SequenceRecord.from_dict(bad)

    def test_engine_request_id_is_outside_digest(self):
        r = _rev()
        steps = [(r.digest, 1, float_bits(-0.1), None)]
        assert build_sequence(0, 0, "p", [1], steps, engine_request_id="x").digest == \
               build_sequence(0, 0, "p", [1], steps, engine_request_id="y").digest

    def test_bad_position_or_chain_rejected(self):
        r = _rev()
        t0 = TokenRecord(r.digest, 0, 1, float_bits(-0.1), GENESIS_DIGEST)
        t1_bad_pos = TokenRecord(r.digest, 2, 1, float_bits(-0.1), t0.digest)
        with pytest.raises(ValueError, match="position"):
            SequenceRecord(0, 0, "p", prompt_digest([1]), 1, (t0, t1_bad_pos), GENESIS_DIGEST)
        t1_bad_chain = TokenRecord(r.digest, 1, 1, float_bits(-0.1), GENESIS_DIGEST)
        with pytest.raises(ValueError, match="chain"):
            SequenceRecord(0, 0, "p", prompt_digest([1]), 1, (t0, t1_bad_chain), GENESIS_DIGEST)

    def test_empty_sequence_rejected(self):
        with pytest.raises(ValueError):
            SequenceRecord(0, 0, "p", prompt_digest([1]), 1, (), GENESIS_DIGEST)


class TestScoreRecord:
    def test_round_trip_and_tamper(self):
        s = ScoreRecord("e" * 64, 0, _rev(5).digest, float_bits(-0.4), GENESIS_DIGEST)
        assert ScoreRecord.from_dict(s.to_dict()) == s
        d = s.to_dict(); d["logprob_bits"] = float_bits(-0.5)
        with pytest.raises(ValueError):
            ScoreRecord.from_dict(d)
