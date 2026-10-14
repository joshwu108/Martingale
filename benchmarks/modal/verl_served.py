"""benchmarks/modal/verl_served.py — the rollout replica's half of the received-weights check (GPU only).

verl 0.9.1's naive sync streams ``engine.get_per_tensor_param()`` to the vLLM worker in buckets
(verl/workers/engine_workers.py:787-800 -> vllm_rollout/vllm_rollout.py:210-244 ->
vllm_rollout/utils.py:248-330). ``MartingaleWorkerExtension`` digests every bucket as it arrives
(``verl_hooks.receive_with_digest``) and keeps the digest of the whole payload; the driver reads it
through ``MartingaleHttpServer.martingale_received`` before recording each rollout. Imports vLLM:
only the Modal image and the Ray processes it starts load this file; the logic it calls is CPU-tested
in tests/test_verl_grpo_runner.py.

``install()`` registers ``MartingaleReplica`` under verl's "vllm" rollout name
(verl/workers/rollout/replica.py:312-390) in the TaskRunner, before ``trainer.init_workers()`` builds
the replicas. vLLM imports the extension by name in its worker processes, so /repo must be on their
PYTHONPATH (the image sets it).
"""
from __future__ import annotations

from typing import Any

import ray
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

EXTENSION = "verl_served.MartingaleWorkerExtension"


class MartingaleWorkerExtension(vLLMColocateWorkerExtension):
    def update_weights_from_ipc(self, *args, **kwargs):
        from verl_hooks import receive_with_digest

        parent = super().update_weights_from_ipc
        return receive_with_digest(self, lambda: parent(*args, **kwargs))

    def martingale_received_digest(self) -> str | None:
        return getattr(self, "_martingale_received", None)


class MartingaleHttpServer(vLLMHttpServer):
    def _get_worker_extension_cls(self) -> str:
        return EXTENSION

    async def martingale_received(self) -> list[str | None]:
        """One report per TP worker of this server's engine (vLLMHttpServer.collective_rpc drops the
        results). With data parallelism > 1 this reaches one DP shard only (vllm_async_server.py:1310)."""
        return list(await self.engine.collective_rpc("martingale_received_digest"))


class MartingaleReplica(vLLMReplica):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(MartingaleHttpServer)


def install() -> None:
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    RolloutReplicaRegistry.register("vllm", lambda: MartingaleReplica)


def servers(trainer: Any) -> list[Any]:
    """Every rollout server handle of an initialised RayPPOTrainer."""
    return [server for replica in trainer.llm_server_manager.get_replicas() for server in replica.servers]
