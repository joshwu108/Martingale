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
from martingale.record import ReceivedWeights
from tests.test_verl_integration import make_batch, multi_turn_batch

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
    def __init__(self, make=make_batch):
        self.make = make

    def generate_sequences(self, gen_batch):
        return Proto(self.make().batch)


class FakeReplica:
    """The vLLM worker extension's half: digests each sync's payload as it arrives."""

    def __init__(self):
        self.received = None
        self.drop_syncs = set()          # sync numbers that never reach the replica (a stale server)
        self.syncs = 0

    def update_weights(self, per_tensor_param):
        self.syncs += 1
        if self.syncs in self.drop_syncs:
            return
        received = ReceivedWeights()
        received.add(list(per_tensor_param))
        self.received = received.digest()

    def received_digest(self):
        return self.received


class Tracking:
    def __init__(self):
        self.lines = []

    def log(self, data, step, backend=None):
        self.lines.append((step, data))


class FakeTrainer:
    """RayPPOTrainer.fit's order of calls for one GPU."""

    def __init__(self, worker, replica, make=make_batch):
        self.worker = worker
        self.replica = replica
        self.async_rollout_manager = FakeManager(make)
        self.global_steps = 0

    def update_weights(self):
        """checkpoint_manager.update_weights: the naive sync sends engine.get_per_tensor_param()."""
        params, _ = self.worker.engine.get_per_tensor_param()
        self.replica.update_weights(params)

    def _update_actor(self, batch):
        self.worker.train_mini_batch(batch.batch)
        out = Proto({})
        out.meta_info["metrics"] = {"actor/pg_loss": [0.0]}
        return out

    def fit(self, steps, tracking):
        self.update_weights()                                                 # fit() syncs before the loop
        for step in range(1, steps + 1):
            self.global_steps = step
            gen = self.async_rollout_manager.generate_sequences(None)
            batch = Proto({**gen.batch, "uid": torch.arange(2)})          # union with the prompt batch
            batch.reorder(torch.tensor([1, 0]))                               # _balance_batch
            out = self._update_actor(batch)
            self.update_weights()                                             # after the actor update
            tracking.log({**out.meta_info["metrics"], "critic/score/mean": 1.0}, step)


@pytest.fixture
def no_verl(monkeypatch):
    monkeypatch.setattr("martingale.integrations._verl_compat.require_verl",
                        lambda: type("S", (), {"no_padding_2_padding": staticmethod(lambda x, _: x)})())


def build(tmp_path, grad_norms, make=make_batch, replica=None):
    workspace = str(tmp_path / "ws")
    worker = FakeTrainingWorker(FakeEngine(grad_norms))
    hooks = vh.WorkerHooks.install(worker, workspace=workspace, tokenizer_digest=TOK, sampler_overrides=None)
    driver = MartingaleVerlRecorder(workspace, TOK)
    view = vh.FlightView(driver)
    monitor = MartingaleMonitor(view, AlarmConfig(min_floor_tokens=1), halt_on_error=False)
    replica = FakeReplica() if replica is None else replica
    trainer = FakeTrainer(worker, replica, make)
    dhooks = vh.DriverHooks(driver, monitor, view, WorkerRPC(hooks), replica)
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
    assert checks["received_weights_equal_rollout_revision"] == {
        "ok": True, "verified_steps": [0, 3, 7], "mismatched_steps": [], "unverified_rollouts": 0}
    assert all(u["row_ids_match_rollout"] for u in dhooks.updates)
    assert logs[0]["martingale/alarms"] == 0.0 and "actor/pg_loss" in logs[0]
    hooks.close()
    driver.recorder.ledger.checkpoint_wal()
    report = verify_export(driver.recorder.export_for_checker(tmp_path / "exp"), expected_head=driver.head())
    assert report["ok"], report["errors"]


def test_skipped_optimizer_update_does_not_publish_weight_version(tmp_path, no_verl):
    trainer, worker, hooks, driver, dhooks, _ = build(tmp_path, [float("inf"), 1.0, 1.0, 1.0])
    counted_step = worker.engine.optimizer_step
    calls = 0

    def inspect_first_step():
        nonlocal calls
        result = counted_step()
        calls += 1
        if calls == 1:
            assert hooks.status()["applied_steps"] == 0
            assert {r.step for r in driver.recorder.ledger.all_revisions()} == {0}
        return result

    worker.engine.optimizer_step = inspect_first_step
    trainer.fit(1, Tracking())
    revisions = list(driver.recorder.ledger.all_revisions())
    assert hooks.status()["skipped_steps"] == 1
    assert hooks.status()["applied_steps"] == 3
    assert {r.step for r in revisions} == {0, 1, 2}
    assert calls == 4
    assert dhooks.updates[0]["applied_steps"] == 3
    hooks.close()


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


def test_mask_hole_rows_are_refused_through_the_fake_fit(tmp_path, no_verl):
    """A multi-turn rollout (tool text inside a response) on verl's driver/worker path: the hole row
    travels with row id -1 through union, reorder, mini-batches and micro-batches, the loss path skips
    it, the other row is scored at every lag, and the record still verifies."""
    trainer, worker, hooks, driver, dhooks, _ = build(tmp_path, [1.0] * 4, make=multi_turn_batch)
    trainer.fit(1, Tracking())
    assert dhooks.rollouts[-1]["mask_hole_rows"] == 1
    assert dhooks.rollouts[-1]["row_ids"] == [-1, 1]
    assert driver.stats["mask_hole_rows"] == 1 and driver.stats["sequences"] == 1
    status = hooks.status()
    assert status["stats"]["skipped_rows"] == 2 and status["stats"]["unmatched_rows"] == 0   # once per epoch
    assert status["stats"]["scored_sequences"] == 2
    checks = vh.adapter_checks(rollouts=dhooks.rollouts, updates=dhooks.updates, worker=status,
                               revisions=[r.to_dict() for r in driver.recorder.ledger.all_revisions()], log_history=[])
    assert checks["mask_holes"] == {"ok": False, "rows_with_holes": 1, "refused_rows": 1, "skipped_loss_rows": 2}
    assert checks["row_ids_survive_union_reorder"]["ok"]
    assert all(u["row_ids_match_rollout"] for u in dhooks.updates)
    assert vh.mask_holes(make_batch().batch["response_mask"]) == 0
    hooks.close()
    with pytest.warns(RuntimeWarning, match="tool tokens"):
        head = driver.head()
    driver.recorder.ledger.checkpoint_wal()
    report = verify_export(driver.recorder.export_for_checker(tmp_path / "exp"), expected_head=head)
    assert report["ok"], report["errors"]


def test_stale_replica_fails_the_sync_check_and_raises_stale_server(tmp_path, no_verl):
    """The sync after the first actor update never reaches the replica: the second rollout's
    revision carries the old received digest, the check names the step, the monitor errors."""
    replica = FakeReplica()
    replica.drop_syncs = {2}
    trainer, worker, hooks, driver, dhooks, monitor = build(tmp_path, [1.0] * 8, replica=replica)
    trainer.fit(2, Tracking())
    rollout_revs = [r for r in driver.recorder.ledger.all_revisions() if r.sampler.logprobs_source == "engine_sampling"]
    assert [r.sync_verified for r in rollout_revs] == [True, False]
    assert rollout_revs[1].received_weights_digest == rollout_revs[0].weights_digest
    checks = vh.adapter_checks(rollouts=dhooks.rollouts, updates=dhooks.updates, worker=hooks.status(),
                               revisions=[r.to_dict() for r in driver.recorder.ledger.all_revisions()], log_history=[])
    assert checks["received_weights_equal_rollout_revision"] == {
        "ok": False, "verified_steps": [0], "mismatched_steps": [4], "unverified_rollouts": 0}
    stale = [a for a in monitor.alarms if a.kind == "stale_server"]
    assert stale and stale[0].evidence["observed"] is True
    hooks.close()


def test_replica_that_cannot_report_is_unverified_not_matched(tmp_path, no_verl):
    class Silent(FakeReplica):
        def received_digest(self):
            return None

    trainer, worker, hooks, driver, dhooks, monitor = build(tmp_path, [1.0] * 4, replica=Silent())
    trainer.fit(1, Tracking())
    assert dhooks.rollouts[-1]["received_weights_digest"] is None
    assert all(r.received_weights_digest is None for r in driver.recorder.ledger.all_revisions())
    checks = vh.adapter_checks(rollouts=dhooks.rollouts, updates=dhooks.updates, worker=hooks.status(),
                               revisions=[r.to_dict() for r in driver.recorder.ledger.all_revisions()], log_history=[])
    assert checks["received_weights_equal_rollout_revision"] == {
        "ok": False, "verified_steps": [], "mismatched_steps": [], "unverified_rollouts": 1}
    assert not [a for a in monitor.alarms if a.kind == "stale_server"]
    hooks.close()


def test_received_digest_from_replica_reports():
    assert vh.received_digest([["c" * 64], ["c" * 64, "c" * 64]]) == "c" * 64      # 2 servers, TP 1 and 2
    assert vh.received_digest([[None, None]]) is None                               # nobody reports: no claim
    assert vh.received_digest([]) is None


@pytest.mark.parametrize("reports", [[["c" * 64], ["d" * 64]], [["c" * 64, None]]])
def test_workers_that_disagree_or_partly_report_are_a_mismatch_not_a_crash(reports):
    """Disagreement (or one worker silent while another reports) is evidence of a stale worker: it
    becomes a digest no weights can have, so the revision records a mismatch and the alarm fires."""
    digest = vh.received_digest(reports)
    assert digest is not None and digest not in {"c" * 64, "d" * 64}
    assert digest == vh.received_digest(list(reversed(reports)))


def test_row_id_ledger_lookup_is_timed_separately(tmp_path, no_verl):
    """The worker resolves every row id through the ledger's row-id table (its own maps are empty);
    that lookup's time is reported apart from the rest of the scoring time."""
    trainer, worker, hooks, driver, dhooks, _ = build(tmp_path, [1.0] * 8)
    trainer.fit(2, Tracking())
    lookup = hooks.status()["row_lookup"]
    assert lookup["calls"] == 8 and lookup["found"] == 8               # 2 steps x 2 epochs x 2 rows
    assert 0 <= lookup["seconds"] <= lookup["score_seconds"]
    assert 0 <= lookup["max_seconds"] <= lookup["seconds"]
    checks = vh.adapter_checks(rollouts=dhooks.rollouts, updates=dhooks.updates, worker=hooks.status(),
                               revisions=[], log_history=[])
    assert checks["row_id_lookup"] == lookup
    hooks.close()


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


class FakeExtension:
    """verl's vLLMColocateWorkerExtension shape: update_weights_from_ipc feeds buckets to _update_weights."""

    def __init__(self, buckets, fail_after=None):
        self.buckets = buckets
        self.fail_after = fail_after
        self.loaded = []

    def _update_weights(self, weights, peft_config=None, base_sync_done=False):
        self.loaded += [name for name, _ in weights]

    def update_weights_from_ipc(self, peft_config=None, base_sync_done=False, use_shm=False):
        for i, bucket in enumerate(self.buckets):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("socket closed")
            self._update_weights(bucket, peft_config=peft_config, base_sync_done=base_sync_done)
        return "done"


def test_receive_with_digest_digests_every_bucket_then_unwraps():
    from martingale.record import weights_digest

    w = {"a": torch.ones(2), "b": torch.arange(3.0), "c": torch.zeros(1, 2)}
    ext = FakeExtension([[("c", w["c"])], [("a", w["a"]), ("b", w["b"])]])
    assert vh.receive_with_digest(ext, lambda: ext.update_weights_from_ipc(use_shm=False)) == "done"
    assert ext._martingale_received == weights_digest(w)
    assert ext.loaded == ["c", "a", "b"]                          # the engine still loads every bucket
    assert "_update_weights" not in vars(ext)                     # the class method is back


def test_receive_with_digest_reports_nothing_after_a_failed_sync():
    ext = FakeExtension([[("a", torch.ones(1))], [("b", torch.ones(1))]], fail_after=1)
    ext._martingale_received = "c" * 64                           # an older sync's digest
    with pytest.raises(RuntimeError, match="socket"):
        vh.receive_with_digest(ext, ext.update_weights_from_ipc)
    assert ext._martingale_received is None
    assert "_update_weights" not in vars(ext)


def test_a_name_received_twice_withholds_the_claim_but_never_breaks_the_sync():
    ext = FakeExtension([[("a", torch.ones(1))], [("a", torch.ones(1))]])
    assert vh.receive_with_digest(ext, ext.update_weights_from_ipc) == "done"
    assert ext.loaded == ["a", "a"] and ext._martingale_received is None
