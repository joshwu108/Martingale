"""benchmarks/modal/replay_inject.py — a synthetic replay buffer for the acceptance run.

Reservoir is the real buffer; the acceptance run (docs/replay-provenance.md, "Acceptance
run") needs only what a buffer does from the record's point of view: after each
generation batch, replace a deterministic fraction of its rows with completions from
EARLIER generations and register each one with the flight recorder as a replayed row
with full provenance (draw id, content digest, exact importance weight, origin digest).
The loss then trains on those rows, the recorder scores them under the train revision,
and the doctor's `by_provenance.replayed` buckets, `replayed.sequence_weights` and the
`martingale/replayed_*` metrics are exercised on real engine and trainer numbers.

The importance weight a buffer would apply is the sequence ratio
exp(sum_t (log pi_now(t) - log pi_behaviour(t))) from one no-grad forward of the
current trainer weights; it is stored exactly (Fraction of the float64 value).
The content digest is BLAKE2b-256 over the prompt and completion ids.

No modal import: tests load this file directly with a fake trainer.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch

from martingale.integrations.trl import OLD_LOGPROBS_KEY, ROW_ID_KEY, SAMPLING_LOGPROBS_KEY, OriginNotFound

MAX_LOG_WEIGHT = 700.0   # exp() overflows float64 above ~709.78


@dataclass(frozen=True)
class PoolEntry:
    prompt: tuple[int, ...]
    completion: tuple[int, ...]
    logprobs: tuple[float, ...]        # the behaviour log-probs the recorder bound (same source preference)
    advantage: float
    behavior_step: int
    origin_digest: str


def content_digest(prompt: tuple[int, ...], completion: tuple[int, ...]) -> str:
    raw = b"".join(int(t).to_bytes(4, "little", signed=True) for t in (*prompt, -1, *completion))
    return hashlib.blake2b(raw, digest_size=32).hexdigest()


def _unpad(ids: torch.Tensor, mask: torch.Tensor, side: str) -> list[int]:
    n = int(mask.sum().item())
    row = ids[-n:] if side == "left" else ids[:n]
    return [int(x) for x in row.tolist()] if n else []


def _logps_tensor(result: Any) -> torch.Tensor:
    if isinstance(result, dict):
        return result["logps"]
    return result[0] if isinstance(result, (tuple, list)) else result


class SyntheticReplay:
    def __init__(self, flight: Any, fraction: Fraction, seed: int = 0, pool_max: int = 256,
                 pad_token_id: int | None = None) -> None:
        fraction = Fraction(fraction)
        if not 0 <= fraction <= 1:
            raise ValueError(f"replay fraction must be in [0, 1], got {fraction}")
        self.flight = flight
        self.fraction = fraction
        self.seed = seed
        self.pool_max = pool_max
        self.pad = pad_token_id
        self._pool: list[PoolEntry] = []
        self.last_replaced: list[int] = []
        self.stats: dict[str, Any] = {"generation_calls": 0, "replaced_rows": 0, "unregistered_rows": 0,
                                      "draw_ids": [], "per_step": []}

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    # ---- the hook ---------------------------------------------------------------------

    def after_generation(self, output: dict, trainer: Any) -> dict:
        """Replace floor(fraction * rows) rows of this batch with earlier completions; returns the batch
        (the same object when nothing is replaced). Call after the flight recorder's on_generation."""
        step = int(trainer.state.global_step)
        self.stats["generation_calls"] += 1
        n = int(output["completion_ids"].size(0))
        candidates = [e for e in self._pool if e.behavior_step < step]
        k = int(self.fraction * n)
        rng = random.Random(f"{self.seed}:{step}")
        rows = sorted(rng.sample(range(n), k)) if k and candidates else []
        picks = [rng.choice(candidates) for _ in rows]
        self._add_to_pool(output, trainer, step)
        self.last_replaced = rows
        if not rows:
            return output
        rids: list[int] = []
        for r, entry in zip(rows, picks):
            draw_id = f"{step}/{r}"
            try:
                rid = self.flight.register_replayed(
                    trainer, origin_digest=entry.origin_digest, draw_id=draw_id,
                    content_digest=content_digest(entry.prompt, entry.completion),
                    is_weight=self._is_weight(entry, trainer), rescale=1, advantage=entry.advantage)
            except OriginNotFound:
                rid = -1
                self.stats["unregistered_rows"] += 1
            rids.append(rid)
            self.stats["draw_ids"].append(draw_id)
        self.stats["replaced_rows"] += len(rows)
        self.stats["per_step"].append({"step": step, "rows": rows, "behavior_steps": [e.behavior_step for e in picks],
                                       "row_ids": rids})
        return self._rewrite(output, rows, picks, rids)

    # ---- pieces -----------------------------------------------------------------------

    def _behaviour_logps(self, output: dict, trainer: Any) -> torch.Tensor:
        """Same preference as MartingaleRecorder: engine sampling log-probs, else TRL's old log-probs,
        else one no-grad forward, so the pool holds the numbers the record bound."""
        for key in (SAMPLING_LOGPROBS_KEY, OLD_LOGPROBS_KEY):
            if output.get(key) is not None:
                return output[key].detach().to("cpu")
        ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
        am = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)
        with torch.no_grad():
            res = trainer._get_per_token_logps_and_entropies(trainer.model, ids, am, output["completion_ids"].size(1))
        return _logps_tensor(res).detach().to("cpu")

    def _add_to_pool(self, output: dict, trainer: Any, step: int) -> None:
        logps = self._behaviour_logps(output, trainer)
        adv = output.get("advantages")
        row_ids = output[ROW_ID_KEY].to("cpu").tolist()
        p_ids, p_mask = output["prompt_ids"].to("cpu"), output["prompt_mask"].to("cpu")
        c_ids, c_mask = output["completion_ids"].to("cpu"), output["completion_mask"].to("cpu")
        for r in range(c_ids.size(0)):
            digest = self.flight.sequence_digest(int(row_ids[r])) if int(row_ids[r]) >= 0 else None
            comp = _unpad(c_ids[r], c_mask[r], "right")
            if digest is None or not comp:
                continue
            lp = [float(x) for x in logps[r][:len(comp)].tolist()]
            if any(math.isnan(x) or math.isinf(x) for x in lp):
                continue
            self._pool.append(PoolEntry(tuple(_unpad(p_ids[r], p_mask[r], "left")), tuple(comp), tuple(lp),
                                        0.0 if adv is None else float(adv[r]), step, digest))
        del self._pool[:-self.pool_max]

    def _is_weight(self, entry: PoolEntry, trainer: Any) -> Fraction:
        """exp(sum(current - behaviour log-prob)) under the trainer's current weights, exact as a Fraction."""
        device = next(trainer.model.parameters()).device
        ids = torch.tensor([list(entry.prompt) + list(entry.completion)], dtype=torch.long, device=device)
        am = torch.ones_like(ids)
        with torch.no_grad():
            res = trainer._get_per_token_logps_and_entropies(trainer.model, ids, am, len(entry.completion))
        cur = [float(x) for x in _logps_tensor(res)[0].detach().to("cpu").tolist()]
        s = sum(c - b for c, b in zip(cur, entry.logprobs))
        return Fraction(math.exp(min(s, MAX_LOG_WEIGHT)))

    def _rewrite(self, output: dict, rows: list[int], picks: list[PoolEntry], rids: list[int]) -> dict:
        if output.get("ref_per_token_logps") is not None:
            raise NotImplementedError("beta > 0: reference log-probs for replayed rows are not available here")
        pad = self.pad if self.pad is not None else int(output["prompt_ids"][0, 0].item())
        P = max(output["prompt_ids"].size(1), max(len(e.prompt) for e in picks))
        C = max(output["completion_ids"].size(1), max(len(e.completion) for e in picks))
        device = output["completion_ids"].device

        def widen(t: torch.Tensor, width: int, side: str, fill: Any) -> torch.Tensor:
            extra = width - t.size(1)
            if extra <= 0:
                return t.clone()
            block = torch.full((t.size(0), extra), fill, dtype=t.dtype, device=t.device)
            return torch.cat([block, t] if side == "left" else [t, block], dim=1)

        new = dict(output)
        new["prompt_ids"] = widen(output["prompt_ids"], P, "left", pad)
        new["prompt_mask"] = widen(output["prompt_mask"], P, "left", 0)
        new["completion_ids"] = widen(output["completion_ids"], C, "right", pad)
        new["completion_mask"] = widen(output["completion_mask"], C, "right", 0)
        for key in (OLD_LOGPROBS_KEY, SAMPLING_LOGPROBS_KEY):
            if output.get(key) is not None:
                new[key] = widen(output[key], C, "right", 0.0)
        new["advantages"] = output["advantages"].clone()
        new[ROW_ID_KEY] = output[ROW_ID_KEY].clone()
        for r, e, rid in zip(rows, picks, rids):
            lp, lc = len(e.prompt), len(e.completion)
            new["prompt_ids"][r] = pad
            new["prompt_ids"][r, P - lp:] = torch.tensor(e.prompt, dtype=new["prompt_ids"].dtype, device=device)
            new["prompt_mask"][r] = 0
            new["prompt_mask"][r, P - lp:] = 1
            new["completion_ids"][r] = pad
            new["completion_ids"][r, :lc] = torch.tensor(e.completion, dtype=new["completion_ids"].dtype, device=device)
            new["completion_mask"][r] = 0
            new["completion_mask"][r, :lc] = 1
            for key in (OLD_LOGPROBS_KEY, SAMPLING_LOGPROBS_KEY):
                if key in new and new[key] is not None:
                    new[key][r] = 0.0
                    new[key][r, :lc] = torch.tensor(e.logprobs, dtype=new[key].dtype, device=device)
            new["advantages"][r] = e.advantage
            new[ROW_ID_KEY][r] = rid
        if torch.is_tensor(new.get("num_items_in_batch")):
            new["num_items_in_batch"] = new["completion_mask"].sum().to(new["num_items_in_batch"].dtype)
        return new
