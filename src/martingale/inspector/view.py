"""
inspector/view.py — RecordView: everything the four panels need, derived once per record version.

All per-token arithmetic is the diagnostics module's (exact Fractions through
ScoredToken and Bucket); this file only groups and sorts. Fractions are
rendered as "p/q" strings for JSON, with float64 copies under
`float_informational` for display.
"""
from __future__ import annotations

import tempfile
import threading
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from martingale.diagnostics.alarms import AlarmConfig
from martingale.diagnostics.replay import ReplayStep, replay
from martingale.diagnostics.staleness import Bucket, ScoredToken, _fs, attribute, decompose, diagnosis, scored_tokens
from martingale.inspector.tokenizer import TokenDecoder
from martingale.record.bits import bits_to_fraction
from martingale.record.store import TokenLedger
from martingale.record.tokens import SequenceRecord

HEAD_FILES = ("head.txt", "martingale_head.txt")


@dataclass(frozen=True)
class SequenceRow:
    digest: str
    sequence_id: str
    actor_id: int
    sequence_index: int
    generation_step: int
    length: int
    prompt_len: int
    reward: Fraction | None
    mixed_revision: bool
    mean_abs_log_ratio: dict[int, Fraction]      # per lag, exact
    confident_disagreements: dict[int, int]      # per lag
    token_ids: tuple[int, ...]

    def to_json(self, decoder: TokenDecoder | None) -> dict[str, Any]:
        return {
            "digest": self.digest, "sequence_id": self.sequence_id, "actor_id": self.actor_id,
            "sequence_index": self.sequence_index, "generation_step": self.generation_step,
            "length": self.length, "prompt_len": self.prompt_len,
            "reward": None if self.reward is None else _fs(self.reward), "mixed_revision": self.mixed_revision,
            "mean_abs_log_ratio": {str(k): _fs(v) for k, v in sorted(self.mean_abs_log_ratio.items())},
            "confident_disagreements": {str(k): v for k, v in sorted(self.confident_disagreements.items())},
            "scored_lags": sorted(self.mean_abs_log_ratio),
            "text": None if decoder is None else decoder.text(self.token_ids),
            "float_informational": {
                "reward": None if self.reward is None else float(self.reward),
                "mean_abs_log_ratio": {str(k): float(v) for k, v in sorted(self.mean_abs_log_ratio.items())},
            },
        }


@dataclass
class _Derived:
    version: tuple[int, int, int]
    rows: list[SequenceRow]
    lags: list[int]
    steps: list[dict[str, Any]]
    doctor: dict[str, Any]
    replay: list[ReplayStep]
    checker: dict[str, Any] | None = None


class RecordView:
    def __init__(self, ledger: TokenLedger, decoder: TokenDecoder | None = None, *, head: str | None = None,
                 workspace: Path | None = None, alarm_config: AlarmConfig | None = None) -> None:
        self.ledger = ledger
        self.decoder = decoder
        self.explicit_head = head
        self.workspace = workspace
        self.alarm_config = alarm_config or AlarmConfig()
        self._lock = threading.Lock()
        self._derived: _Derived | None = None
        self._step_of: dict[str, int] = {}

    # ---- caching ----------------------------------------------------------------------

    def _version(self) -> tuple[int, int, int]:
        return (self.ledger.count_revisions(), self.ledger.count_sequences(), self.ledger.count_scores())

    def derived(self) -> _Derived:
        with self._lock:
            v = self._version()
            if self._derived is None or self._derived.version != v:
                self._derived = self._build(v)
            return self._derived

    def step_of(self, revision_digest: str) -> int:
        if revision_digest not in self._step_of:
            self._step_of[revision_digest] = self.ledger.get_revision(revision_digest).step
        return self._step_of[revision_digest]

    def _build(self, version: tuple[int, int, int]) -> _Derived:
        toks = list(scored_tokens(self.ledger))
        rows = self._rows(self.ledger.all_sequences(), toks)
        d = decompose(self.ledger)
        rep = replay(self.ledger, self.alarm_config)
        doctor = {"decomposition": d, "attribution": attribute(d), "diagnosis": diagnosis(d),
                  "alarms": [asdict(a) for st in rep for a in st.alarms], "alarm_config": asdict(self.alarm_config),
                  "replay_steps": len(rep)}
        return _Derived(version, rows, d["lags"], self._steps(rows, toks), doctor, rep)

    # ---- sequences ---------------------------------------------------------------------

    def _rows(self, seqs: Iterable[SequenceRecord], toks: list[ScoredToken]) -> list[SequenceRow]:
        sum_abs: dict[str, dict[int, Fraction]] = {}
        n_scored: dict[str, dict[int, int]] = {}
        n_conf: dict[str, dict[int, int]] = {}
        for t in toks:
            a = sum_abs.setdefault(t.sequence_digest, {})
            a[t.lag] = a.get(t.lag, Fraction(0)) + abs(t.log_ratio)
            n = n_scored.setdefault(t.sequence_digest, {})
            n[t.lag] = n.get(t.lag, 0) + 1
            c = n_conf.setdefault(t.sequence_digest, {})
            c[t.lag] = c.get(t.lag, 0) + int(t.confident_disagreement)
        rows = []
        for s in seqs:
            lags = n_scored.get(s.digest, {})
            rows.append(SequenceRow(
                digest=s.digest, sequence_id=s.sequence_id, actor_id=s.actor_id, sequence_index=s.sequence_index,
                generation_step=self.step_of(s.tokens[0].revision_digest), length=len(s.tokens),
                prompt_len=s.prompt_len,
                reward=None if s.reward_bits is None else bits_to_fraction(s.reward_bits),
                mixed_revision=s.is_mixed_revision,
                mean_abs_log_ratio={lag: sum_abs[s.digest][lag] / n for lag, n in lags.items()},
                confident_disagreements={lag: n_conf[s.digest][lag] for lag in lags},
                token_ids=tuple(t.token_id for t in s.tokens)))
        return rows

    def sequences(self, *, step: int | None, lag: int, sort: str, descending: bool) -> list[SequenceRow]:
        rows = [r for r in self.derived().rows if step is None or r.generation_step == step]

        def key(r: SequenceRow) -> tuple[bool, Fraction]:
            v: Fraction | None
            if sort == "advantage":
                v = r.reward
            elif sort == "length":
                v = Fraction(r.length)
            elif sort == "mean_abs_log_ratio":
                v = r.mean_abs_log_ratio.get(lag)
            else:
                v = None if lag not in r.confident_disagreements else Fraction(r.confident_disagreements[lag])
            return (v is None, Fraction(0) if v is None else (-v if descending else v))   # exact compare; None last
        return sorted(rows, key=key)

    def sequence_detail(self, digest: str) -> dict[str, Any]:
        seq = self.ledger.get_sequence(digest)      # KeyError -> 404 in the route
        scores = self.ledger.scores_for(digest)
        by_pos: dict[int, list[dict[str, Any]]] = {}
        lags: set[int] = set()
        for sc in scores:
            tok = seq.tokens[sc.position]
            t = ScoredToken(actor_id=seq.actor_id, sequence_digest=digest, position=sc.position,
                            behavior_step=self.step_of(tok.revision_digest), train_step=self.step_of(sc.train_revision_digest),
                            log_ratio=bits_to_fraction(sc.logprob_bits) - bits_to_fraction(tok.logprob_bits),
                            mixed_sequence=seq.is_mixed_revision, behavior_logprob=bits_to_fraction(tok.logprob_bits))
            lags.add(t.lag)
            by_pos.setdefault(sc.position, []).append({
                "lag": t.lag, "train_step": t.train_step, "logprob": _fs(bits_to_fraction(sc.logprob_bits)),
                "log_ratio": _fs(t.log_ratio), "confident_disagreement": t.confident_disagreement,
                "float_informational": {"logprob": float(bits_to_fraction(sc.logprob_bits)), "log_ratio": float(t.log_ratio)},
            })
        ids = [t.token_id for t in seq.tokens]
        pieces = None if self.decoder is None else self.decoder.pieces(ids)
        tokens = []
        for tr in seq.tokens:
            lp = bits_to_fraction(tr.logprob_bits)
            tokens.append({
                "position": tr.position, "token_id": tr.token_id, "text": None if pieces is None else pieces[tr.position],
                "revision_step": self.step_of(tr.revision_digest), "behavior_logprob": _fs(lp),
                "scores": {str(s["lag"]): s for s in sorted(by_pos.get(tr.position, []), key=lambda s: s["lag"])},
                "float_informational": {"behavior_logprob": float(lp)},
            })
        reward = None if seq.reward_bits is None else bits_to_fraction(seq.reward_bits)
        gen_steps = sorted({self.step_of(d) for d in seq.revision_digests})
        return {
            "digest": digest, "sequence_id": seq.sequence_id, "actor_id": seq.actor_id,
            "sequence_index": seq.sequence_index, "generation_step": gen_steps[0], "generation_steps": gen_steps,
            "mixed_revision": seq.is_mixed_revision, "lags": sorted(lags),
            "reward": None if reward is None else _fs(reward),
            "prompt": {"digest": seq.prompt_digest, "len": seq.prompt_len,
                       "ids": None if seq.prompt_ids is None else list(seq.prompt_ids),
                       "text": None if (seq.prompt_ids is None or self.decoder is None) else self.decoder.text(seq.prompt_ids)},
            "tokens": tokens, "text": None if self.decoder is None else self.decoder.text(ids),
            "float_informational": {"reward": None if reward is None else float(reward)},
        }

    # ---- steps -------------------------------------------------------------------------

    def _steps(self, rows: list[SequenceRow], toks: list[ScoredToken]) -> list[dict[str, Any]]:
        buckets: dict[tuple[int, int], Bucket] = {}
        conf_all: dict[int, int] = {}
        for t in toks:
            buckets.setdefault((t.behavior_step, t.lag), Bucket(t.lag)).add(t)
            conf_all[t.behavior_step] = conf_all.get(t.behavior_step, 0) + int(t.confident_disagreement)
        gen_steps = sorted({r.generation_step for r in rows} | {g for g, _ in buckets})
        out = []
        for g in gen_steps:
            by_lag = [buckets[(g, lag)].summary() for lag in sorted(lag for gg, lag in buckets if gg == g)]
            lag0 = next((b for b in by_lag if b["lag"] == 0), None)
            mine = [r for r in rows if r.generation_step == g]
            out.append({
                "generation_step": g, "n_sequences": len(mine), "n_tokens": sum(r.length for r in mine),
                "n_scored_tokens": sum(b["n_tokens"] for b in by_lag),
                "n_mixed_sequences": sum(r.mixed_revision for r in mine),
                "lags": [b["lag"] for b in by_lag], "by_lag": by_lag,
                "lag0_floor": None if lag0 is None else lag0["mean_abs_log_ratio"],
                "n_confident_disagreements_lag0": 0 if lag0 is None else lag0["n_confident_disagreements"],
                "n_confident_disagreements": conf_all.get(g, 0),
                "float_informational": {
                    "lag0_floor": None if lag0 is None else lag0["float_informational"]["mean_abs_log_ratio"],
                    "ess_fraction_by_lag": {str(b["lag"]): b["float_informational"]["ess_fraction"] for b in by_lag},
                    "mean_abs_log_ratio_by_lag": {str(b["lag"]): b["float_informational"]["mean_abs_log_ratio"] for b in by_lag},
                },
            })
        return out

    def replay_summary(self) -> list[dict[str, Any]]:
        out = []
        for st in self.derived().replay:
            d = st.diagnosis
            out.append({"step": d.step, "n_new_scores": d.n_new_scores, "n_new_sequences": d.n_new_sequences,
                        "n_negative_lag": d.n_negative_lag, "span_fraction": d.span_fraction,
                        "lag0_floor": None if d.floor_mean_abs is None else _fs(d.floor_mean_abs),
                        "alarms": [a.kind for a in st.alarms], "by_lag": d.by_lag,
                        "float_informational": {"lag0_floor": None if d.floor_mean_abs is None else float(d.floor_mean_abs)}})
        return out

    # ---- record and trust ----------------------------------------------------------------

    def stored_head(self) -> tuple[str | None, str | None]:
        """(head, where it came from): --head, then head.txt / martingale_head.txt in the workspace,
        then the newest checkpoint-*/martingale_head.txt."""
        if self.explicit_head is not None:
            return self.explicit_head, "--head"
        if self.workspace is None:
            return None, None
        candidates = [self.workspace / n for n in HEAD_FILES]
        ckpts = sorted(self.workspace.glob("checkpoint-*/martingale_head.txt"),
                       key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else -1)
        for p in candidates + ckpts[::-1]:
            if p.is_file():
                text = p.read_text().strip()
                if text:
                    return text, str(p)
        return None, None

    def record(self) -> dict[str, Any]:
        from checker.verify_tokens import verify_export

        d = self.derived()
        if d.checker is None:
            head, source = self.stored_head()
            with tempfile.TemporaryDirectory(prefix="martingale_inspector_") as tmp:
                report = verify_export(self.ledger.export_for_checker(Path(tmp) / "export"), expected_head=head)
            d.checker = {**report, "expected_head": head, "head_source": source}
        rows = d.rows
        return {"revisions": d.version[0], "sequences": d.version[1], "scores": d.version[2],
                "tokens": sum(r.length for r in rows), "mixed_sequences": sum(r.mixed_revision for r in rows),
                "head": self.ledger.head(), "stored_head": d.checker["expected_head"],
                "head_source": d.checker["head_source"], "checker": d.checker}
