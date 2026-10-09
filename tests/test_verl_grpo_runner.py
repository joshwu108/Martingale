"""CPU tests for benchmarks/modal/verl_hooks.py: the verl GRPO runner's hooks on a fake trainer.

The fake mirrors verl 0.9.1's RayPPOTrainer.fit and the actor worker's train_mini_batch:
generate -> union -> balance reorder -> mini-batches x epochs (shuffled) -> micro-batch loss
calls -> optimizer_step, with the driver and the worker holding separate recorders on one
workspace, as the Ray driver and the actor process do on Modal.
"""
import sys
from functools import partial
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "modal"))
import verl_hooks as vh

from checker.verify_tokens import verify_export
from martingale.diagnostics import AlarmConfig, decompose
from martingale.integrations.verl import ROW_ID_KEY, MartingaleMonitor, MartingaleVerlRecorder
from tests.test_verl_integration import make_batch

TOK = "ab" * 32


class Proto:
    def __init__(self, batch):
        self.batch = batch
        self.meta_info = {}

    def reorder(self, idx):
        self.batch = {k: v[idx] for k, v in self.batch.items()}


class Padded(dict):
    def to_padded_tensor(self):
        return self


class FakeEngine:
    def __init__(self, grad_norms):
        self.weights = {"w": torch.ones(3, 2), "b": torch.zeros(2)}
        self.grad_norms = list(grad_norms)
        self.in_loss = False

    def get_per_tensor_param(self):
        assert not self.in_loss, "gathered weights between forward and backward"
        return ((k, v) for k, v in self.weights.items()), None

    def optimizer_step(self):
        norm = self.grad_norms.pop(0)
        if norm == norm and norm != float("inf"):
            self.weights = {k: v + 0.25 for k, v in self.weights.items()}
        return norm


class FakeTrainingWorker:
    def __init__(self, engine):
        self.engine = engine
        self.calls = []
        self.loss_fn = partial(self.ppo_loss, config="actor-config")

    def ppo_loss(self, config, model_output, data, dp_group=None):
        self.calls.append(config)
        return torch.tensor(0.0), {}

    def train_mini_batch(self, batch, *, epochs=2, mini=2, micro=1, shuffle_seed=0):
        """verl's mini-batch x epoch loop; each micro-batch gets one loss call, each mini-batch one step."""
        n = len(batch["responses"])
        gen = torch.Generator().manual_seed(shuffle_seed)
        for _ in range(epochs):
            order = torch.randperm(n, generator=gen)
            for chunk in order.split(n // mini):
                for part in chunk.split(micro):
                    data = Padded({k: v[part] for k, v in batch.items()})
                    lp = data["rollout_log_probs"].clone().requires_grad_(True) - 0.01
                    self.engine.in_loss = True
                    self.loss_fn(model_output={"log_probs": lp}, data=data, dp_group=None)
                    self.engine.in_loss = False
                self.engine.optimizer_step()


class WorkerRPC:
    def __init__(self, hooks):
        self.hooks = hooks

    def snapshot(self):
        return self.hooks.snapshot()

    def status(self):
        return self.hooks.status()


class FakeManager:
    def generate_sequences(self, gen_batch):
        return Proto(make_batch().batch)


class Tracking:
    def __init__(self):
        self.lines = []

    def log(self, data, step, backend=None):
        self.lines.append((step, data))


class FakeTrainer:
    """RayPPOTrainer.fit's order of calls for one GPU."""

    def __init__(self, worker):
        self.worker = worker
        self.async_rollout_manager = FakeManager()
        self.global_steps = 0

    def _update_actor(self, batch):
        self.worker.train_mini_batch(batch.batch)
        out = Proto({})
        out.meta_info["metrics"] = {"actor/pg_loss": [0.0]}
        return out

    def fit(self, steps, tracking):
        for step in range(1, steps + 1):
            self.global_steps = step
            gen = self.async_rollout_manager.generate_sequences(None)
            batch = Proto({**gen.batch, "uid": torch.arange(2)})          # union with the prompt batch
            batch.reorder(torch.tensor([1, 0]))                               # _balance_batch
            out = self._update_actor(batch)
            tracking.log({**out.meta_info["metrics"], "critic/score/mean": 1.0}, step)


@pytest.fixture
def no_verl(monkeypatch):
    monkeypatch.setattr("martingale.integrations._verl_compat.require_verl",
                        lambda: type("S", (), {"no_padding_2_padding": staticmethod(lambda x, _: x)})())


def build(tmp_path, grad_norms):
    workspace = str(tmp_path / "ws")
    worker = FakeTrainingWorker(FakeEngine(grad_norms))
    hooks = vh.WorkerHooks.install(worker, workspace=workspace, tokenizer_digest=TOK, sampler_overrides=None)
    driver = MartingaleVerlRecorder(workspace, TOK)
    view = vh.FlightView(driver)
    monitor = MartingaleMonitor(view, AlarmConfig(min_floor_tokens=1), halt_on_error=False)
    trainer = FakeTrainer(worker)
    dhooks = vh.DriverHooks(driver, monitor, view, WorkerRPC(hooks))
    dhooks.install(trainer)
    return trainer, worker, hooks, driver, dhooks, monitor


def test_fake_fit_records_every_lag_and_verifies(tmp_path, no_verl):
    norms = [1.0, float("inf"), 1.0, 1.0] + [1.0] * 8                  # one fp16 overflow skip in step 1
    trainer, worker, hooks, driver, dhooks, monitor = build(tmp_path, norms)
    tracking_cls = type("T", (Tracking,), {})
    logs = vh.capture_logs(tracking_cls)
    trainer.fit(3, tracking_cls())

    assert worker.calls and set(worker.calls) == {"actor-config"}         # config still reaches ppo_loss
    status = hooks.status()
    assert (status["applied_steps"], status["skipped_steps"], status["loss_time_gathers"]) == (11, 1, 0)
    assert [r["step"] for r in dhooks.rollouts] == [0, 3, 7]                # applied steps at each rollout
    hist = decompose(driver.recorder.ledger)["staleness_histogram"]
    assert set(hist) == {"0", "1", "2", "3"}
    checks = vh.adapter_checks(rollouts=dhooks.rollouts, updates=dhooks.updates, worker=status,
                               revisions=[r.to_dict() for r in driver.recorder.ledger.all_revisions()],
                               log_history=logs)
    assert checks["all_ok"], checks
    assert checks["row_ids_survive_union_reorder"]["updates_with_reordered_rows"] == 3
    assert checks["rollout_digest_equals_trainer_digest"]["steps_compared"] == [0, 3, 7]
    assert all(u["row_ids_match_rollout"] for u in dhooks.updates)
    assert logs[0]["martingale/alarms"] == 0.0 and "actor/pg_loss" in logs[0]
    hooks.close()
    driver.recorder.ledger.checkpoint_wal()
    report = verify_export(driver.recorder.export_for_checker(tmp_path / "exp"), expected_head=driver.head())
    assert report["ok"], report["errors"]


def test_lost_row_ids_and_detached_logps_fail_their_checks(tmp_path, no_verl):
    trainer, worker, hooks, driver, dhooks, _ = build(tmp_path, [1.0] * 4)
    installed = trainer._update_actor

    def union_drops_ids(batch):
        batch.batch.pop(ROW_ID_KEY)
        return installed(batch)

    trainer._update_actor = union_drops_ids
    worker.loss_fn = partial(hooks._inspected(lambda config, model_output, data, dp_group=None: (0, {})),
                             config="c")
    model_output = {"log_probs": torch.zeros(1, 3)}
    worker.loss_fn(model_output=model_output, data=Padded(make_batch().batch), dp_group=None)
    trainer.fit(1, Tracking())
    checks = vh.adapter_checks(rollouts=dhooks.rollouts, updates=dhooks.updates, worker=hooks.status(),
                               revisions=[], log_history=[])
    assert not checks["row_ids_survive_union_reorder"]["ok"]
    assert checks["loss_wrapper_sees_live_logps"]["detached_calls"] == 1
    assert not checks["loss_wrapper_sees_live_logps"]["ok"]
    assert not checks["metrics_in_verl_log"]["ok"] and not checks["all_ok"]


def test_mask_hole_is_audited_then_refused(tmp_path, no_verl):
    _, _, hooks, driver, dhooks, _ = build(tmp_path, [])
    batch = make_batch()
    batch.batch["response_mask"][1] = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="pure"):
        dhooks.on_rollout(Proto(batch.batch))
    assert dhooks.rollouts[-1]["mask_hole_rows"] == 1
    assert vh.mask_holes(make_batch().batch["response_mask"]) == 0


def test_missing_rollout_log_probs_is_audited_then_refused(tmp_path, no_verl):
    _, _, _, _, dhooks, _ = build(tmp_path, [])
    batch = make_batch().batch
    batch.pop("rollout_log_probs")
    with pytest.raises(ValueError, match="calculate_log_probs"):
        dhooks.on_rollout(Proto(batch))
    assert dhooks.rollouts[-1]["missing"] == ["rollout_log_probs"]


def test_install_refuses_an_unexpected_loss(tmp_path):
    worker = FakeTrainingWorker(FakeEngine([]))
    worker.loss_fn = worker.ppo_loss
    with pytest.raises(TypeError, match="partial"):
        vh.WorkerHooks.install(worker, workspace=str(tmp_path / "ws"), tokenizer_digest=TOK)


def test_flight_view_takes_score_stats_from_the_worker(tmp_path):
    driver = MartingaleVerlRecorder(tmp_path / "ws", TOK)
    driver.stats["nan_rows"] = 1
    view = vh.FlightView(driver)
    view.worker_stats = {"scored_sequences": 7, "unmatched_rows": 2, "nan_rows": 3, "sequences": 0}
    assert view.stats["scored_sequences"] == 7 and view.stats["unmatched_rows"] == 2
    assert view.stats["nan_rows"] == 4


def test_task_reward_and_overrides():
    rows = vh.addition_rows(4, seed=0)
    assert rows == vh.addition_rows(4, seed=0) and rows != vh.addition_rows(4, seed=1)
    a, b = (int(x) for x in rows[0]["prompt"][0]["content"].split("?")[0].removeprefix("What is ").split(" + "))
    assert rows[0]["reward_model"]["ground_truth"] == str(a + b) and rows[0]["data_source"] == vh.DATA_SOURCE
    assert vh.compute_score(vh.DATA_SOURCE, " 113.\n", "113") == 1.0
    assert vh.compute_score(vh.DATA_SOURCE, "113 apples", "113") == 0.0
    args = vh.overrides(train_parquet="t.parquet", reward_py="r.py", model_id="m", max_steps=8, seed=0, n=8,
                        train_batch_size=8, ppo_mini_batch_size=4, ppo_epochs=2, lr=1e-6, max_prompt_length=96,
                        max_response_length=16, temperature=1.0)
    for setting in ("trainer.use_v1=false", "algorithm.adv_estimator=grpo", "trainer.total_training_steps=8",
                    "actor_rollout_ref.rollout.calculate_log_probs=True", "actor_rollout_ref.actor.ppo_epochs=2",
                    "reward.custom_reward_function.path=r.py", "trainer.n_gpus_per_node=1"):
        assert setting in args


def test_write_results_writes_the_trl_files(tmp_path):
    res = {"tokens_db": b"db", "report_md": "# r", "head": "h" * 64, "verify": {"ok": True}, "report": {},
           "log_history": [], "alarms": [], "per_step_metrics": [], "worst_tokens": [], "checks": {"all_ok": True},
           "config": {"gpu": "T4"}, "stats": {"driver": {}}}
    out = vh.write_results(res, tmp_path / "run")
    names = {p.name for p in out.iterdir()}
    assert names == {"tokens.db", "report.md", "head.txt", "verify.json", "report.json", "trainer_log.json",
                     "alarms.json", "worst_tokens.json", "adapter_checks.json", "config.json"}
    assert (out / "head.txt").read_text() == "h" * 64 + "\n"
