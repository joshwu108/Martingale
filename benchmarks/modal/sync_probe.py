"""benchmarks/modal/sync_probe.py — elementwise check of TRL's colocated vLLM weight sync.

Wraps ``trainer.vllm_generation.sync_weights`` so that, every time TRL pushes the
trainer's weights into the in-process vLLM model, the probe compares EVERY vLLM
parameter against three references and records the counts:

  vllm vs bf16(trainer)   elements that differ after the sync. 0 means the sync
                          wrote exactly what the trainer holds, rounded to vLLM's dtype.
  vllm vs bf16(initial)   elements that differ from the untouched initial weights.
                          0 means the engine is bit-identical to step 0.
  bf16(trainer) vs bf16(initial)
                          elements the trainer's drift has moved by at least one bf16
                          ulp: the number of changes a correct sync is ABLE to transmit.
  fp32 trainer vs fp32 initial
                          how far the trainer has really moved (max and mean |diff|).

The two hypotheses this separates: (sync no-op) vllm == initial while bf16(trainer)
has flipped elements; (sync fine, rounded away) vllm == bf16(trainer) and the
flipped count is tiny because the updates sit below the bf16 ulp.

Qwen2 only: q/k/v and gate/up are fused in vLLM (``qkv_proj``, ``gate_up_proj``),
concatenated along dim 0 in that order, tensor parallel 1. Tied ``lm_head`` is the
embedding tensor in both frameworks and is therefore compared once.
"""
from __future__ import annotations

import time
from typing import Any

FUSED = {
    "q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1),
}
# Tensors reported individually (the totals cover every parameter).
DETAIL = ("model.embed_tokens.weight", "model.layers.0.self_attn.qkv_proj.weight",
          "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.down_proj.weight",
          "model.layers.0.input_layernorm.weight", "model.norm.weight")


def _vllm_name(hf_name: str) -> tuple[str, int | None]:
    for hf, (fused, idx) in FUSED.items():
        if f".{hf}." in hf_name:
            return hf_name.replace(f".{hf}.", f".{fused}."), idx
    return hf_name, None


def _groups(trainer_model: Any) -> dict[str, list[tuple[int, str]]]:
    """vLLM parameter name -> ordered trainer parameter names that build it."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for name, _ in trainer_model.named_parameters():
        vname, idx = _vllm_name(name)
        groups.setdefault(vname, []).append((idx or 0, name))
    return {k: sorted(v) for k, v in groups.items()}


def _vllm_inner_model(vllm_generation: Any) -> tuple[Any, dict[str, str]]:
    """The nn.Module vLLM executes, unwrapped from the cudagraph wrapper, plus facts about the path."""
    llm = vllm_generation.llm
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    outer = runner.model
    inner = outer.unwrap() if hasattr(outer, "unwrap") else outer
    cc = llm.llm_engine.vllm_config.compilation_config
    facts = {
        "executor": type(llm.llm_engine.model_executor).__name__,
        "model_runner.model": type(outer).__name__,
        "inner_model": type(inner).__name__,
        "same_object_as_get_model": str(runner.get_model() is inner),
        "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
        "compilation_mode": str(getattr(cc, "mode", None)),
        "enable_sleep_mode": str(vllm_generation.enable_sleep_mode),
        "vllm_dtype": str(llm.llm_engine.vllm_config.model_config.dtype),
    }
    return inner, facts


class SyncProbe:
    def __init__(self, trainer: Any) -> None:
        import torch

        self.trainer = trainer
        self.vg = trainer.vllm_generation
        self.inner, self.facts = _vllm_inner_model(self.vg)
        self.vparams = dict(self.inner.named_parameters())
        self.tmodel = trainer.accelerator.unwrap_model(trainer.model)
        self.groups = _groups(self.tmodel)
        self.facts["vllm_param_count"] = str(len(self.vparams))
        self.facts["vllm_param_dtypes"] = str(sorted({str(p.dtype) for p in self.vparams.values()}))
        self.facts["trainer_param_dtypes"] = str(sorted({str(p.dtype) for _, p in self.tmodel.named_parameters()}))
        self.facts["unmapped_vllm_params"] = str(sorted(set(self.vparams) - set(self.groups)))
        self.facts["unmapped_trainer_groups"] = str(sorted(set(self.groups) - set(self.vparams)))
        self.ptrs = {n: p.data_ptr() for n, p in self.vparams.items()}
        # Snapshots of the initial trainer weights, in vLLM's dtype and in fp32, on the CPU.
        self.init_low: dict[str, torch.Tensor] = {}
        self.init_fp32: dict[str, torch.Tensor] = {}
        for vname in self.vparams:
            if vname in self.groups:
                exp = self._expected(vname)
                self.init_fp32[vname] = exp.float().cpu()
                self.init_low[vname] = exp.to(self.vparams[vname].dtype).cpu()
        self.records: list[dict[str, Any]] = []
        self.sync_calls = 0
        original = self.vg.sync_weights

        def wrapped() -> None:
            step = int(trainer.state.global_step)
            pre = self._compare_detail_only()
            t0 = time.perf_counter()
            original()
            dt = time.perf_counter() - t0
            rec = self._compare_all()
            rec.update({"global_step": step, "sync_index": self.sync_calls, "sync_seconds": round(dt, 3),
                        "pre_sync_detail": pre})
            self.sync_calls += 1
            self.records.append(rec)
            print(f"SYNC_PROBE step={step} vllm!=bf16(trainer)={rec['n_diff_vllm_vs_trainer']} "
                  f"vllm!=initial={rec['n_diff_vllm_vs_initial']} trainer_flips={rec['n_flips_trainer_vs_initial']} "
                  f"fp32_drift_max={rec['max_abs_fp32_drift']:.3e} mean={rec['mean_abs_fp32_drift']:.3e}", flush=True)

        self.vg.sync_weights = wrapped

    # ---- helpers ------------------------------------------------------------------

    def _expected(self, vname: str) -> Any:
        """The trainer's tensor(s) for a vLLM parameter, concatenated like vLLM does, fp32 on GPU."""
        import torch

        tparams = dict(self.tmodel.named_parameters())
        parts = [tparams[n].data for _, n in self.groups[vname]]
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def _compare_one(self, vname: str) -> dict[str, Any]:
        vparam = self.vparams[vname].data
        exp32 = self._expected(vname)
        n = min(vparam.shape[0], exp32.shape[0])
        v, e32 = vparam[:n], exp32[:n]
        e_low = e32.to(v.dtype)
        init_low = self.init_low[vname][:n].to(v.device)
        init32 = self.init_fp32[vname][:n].to(v.device)
        drift = (e32 - init32).abs()
        out = {
            "numel": int(v.numel()),
            "n_diff_vllm_vs_trainer": int((v != e_low).sum()),
            "max_abs_vllm_vs_trainer": float((v.float() - e_low.float()).abs().max()),
            "n_diff_vllm_vs_initial": int((v != init_low).sum()),
            "n_flips_trainer_vs_initial": int((e_low != init_low).sum()),
            "max_abs_fp32_drift": float(drift.max()),
            "sum_abs_fp32_drift": float(drift.sum()),
            "shape_mismatch": bool(vparam.shape != exp32.shape),
            "data_ptr_changed": bool(self.vparams[vname].data_ptr() != self.ptrs[vname]),
        }
        return out

    def _compare_detail_only(self) -> dict[str, dict[str, Any]]:
        return {n: self._compare_one(n) for n in DETAIL if n in self.vparams and n in self.groups}

    def _compare_all(self) -> dict[str, Any]:
        tot = {"numel": 0, "n_diff_vllm_vs_trainer": 0, "n_diff_vllm_vs_initial": 0,
               "n_flips_trainer_vs_initial": 0, "sum_abs_fp32_drift": 0.0}
        max_vt, max_drift, ptr_changed, mismatched, detail = 0.0, 0.0, [], [], {}
        for vname in self.vparams:
            if vname not in self.groups:
                continue
            r = self._compare_one(vname)
            for k in tot:
                tot[k] += r[k]
            max_vt = max(max_vt, r["max_abs_vllm_vs_trainer"])
            max_drift = max(max_drift, r["max_abs_fp32_drift"])
            if r["data_ptr_changed"]:
                ptr_changed.append(vname)
            if r["shape_mismatch"]:
                mismatched.append(vname)
            if vname in DETAIL:
                detail[vname] = r
        return {
            "total_numel": tot["numel"],
            "n_diff_vllm_vs_trainer": tot["n_diff_vllm_vs_trainer"],
            "max_abs_vllm_vs_trainer": max_vt,
            "n_diff_vllm_vs_initial": tot["n_diff_vllm_vs_initial"],
            "n_flips_trainer_vs_initial": tot["n_flips_trainer_vs_initial"],
            "max_abs_fp32_drift": max_drift,
            "mean_abs_fp32_drift": tot["sum_abs_fp32_drift"] / max(tot["numel"], 1),
            "data_ptr_changed": ptr_changed,
            "shape_mismatch": mismatched,
            "detail": detail,
        }

    def load_weights_return_check(self) -> dict[str, Any]:
        """Call vLLM's load_weights directly for one tensor and report what it says it loaded."""
        name = "model.norm.weight"
        tparams = dict(self.tmodel.named_parameters())
        returned = self.inner.load_weights([(name, tparams[name].data)])
        return {"name": name, "returned": sorted(returned) if returned is not None else None}

    def summary(self) -> dict[str, Any]:
        return {"facts": self.facts, "sync_calls": self.sync_calls, "records": self.records,
                "load_weights_return_check": self.load_weights_return_check()}
