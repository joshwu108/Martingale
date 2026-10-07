"""Optional token decoding: transformers when installed, ids only otherwise, never a hard dependency."""
import sys

from martingale.inspector.tokenizer import TokenDecoder, load_tokenizer


class Stub:
    def decode(self, ids, skip_special_tokens=False):
        return "|".join(str(i) for i in ids)


def test_none_gives_no_decoder():
    assert load_tokenizer(None) is None


def test_object_with_decode_is_used_directly():
    dec = load_tokenizer(Stub())
    assert isinstance(dec, TokenDecoder)
    assert dec.text([1, 2]) == "1|2" and dec.pieces([1, 2]) == ["1", "2"]


def test_name_without_transformers_yields_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert load_tokenizer("org/model") is None


def test_name_with_transformers_calls_auto_tokenizer(monkeypatch):
    import types
    calls = {}

    class FakeAuto:
        @staticmethod
        def from_pretrained(name):
            calls["name"] = name
            return Stub()
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeAuto))
    dec = load_tokenizer("org/model")
    assert calls["name"] == "org/model" and dec is not None and dec.text([7]) == "7"


def test_pieces_are_cached_per_id():
    class Counting(Stub):
        n = 0

        def decode(self, ids, skip_special_tokens=False):
            self.n += 1
            return super().decode(ids, skip_special_tokens)
    tok = Counting()
    dec = load_tokenizer(tok)
    assert dec is not None
    dec.pieces([5, 5, 6]); dec.pieces([5])
    assert tok.n == 2
