# Replayed rows: the record API for a replay buffer

A replay buffer (Reservoir is the one this was written for) trains the policy
on rows the current policy did not generate. From the record's point of view
such a row is a second appearance of a generation it already holds: the same
prompt, the same completion, the same behaviour log-prob bits under the same
revision, trained on again at a later step. This page is the contract for
registering those rows so that `martingale doctor` buckets them next to the
fresh ones and `martingale verify` checks their binding. What the checker
cannot say about a replayed row is in [`nonclaims.md`](nonclaims.md).

## 1. What a replayed row is in the record

A replayed row is an ordinary `SequenceRecord` on the recording actor's chain
(it gets the next `sequence_index`, chains from the previous sequence, and its
tokens chain like any other) with two extra fields inside its digest:

```
"provenance": "replayed",
"replay": {
  "draw_id":        "<str>",       # the buffer's identifier of this draw (opaque, non-empty)
  "content_digest": "<64 hex>",    # the buffer's content digest of the row
  "is_weight_num":  <int>,         # importance weight the buffer applied, exact, reduced, >= 0
  "is_weight_den":  <int>,
  "rescale_num":    <int>,         # any policy rescale of that weight (1/1: none), exact, reduced, > 0
  "rescale_den":    <int>,
  "origin_digest":  "<64 hex>" | null   # the fresh sequence in this record the row copies, when known
}
```

Fresh rows carry neither key and their canonical form, digests and exports are
byte for byte what they were before (an absent `provenance` means fresh; the
committed real records still verify). Every token of a replayed row carries the
behaviour log-prob bits and the revision digest of the generation it came from,
so the doctor computes its lag from the record (train step minus the revision's
step), not from anything the buffer declares. The row's `reward_bits` is the
advantage the buffer gave it in its new group, if any.

Weights are exact: a `Fraction` is stored as is, an `int` as `n/1`, a float as
its exact dyadic value (`Fraction(0.1)`, not `1/10`). Reservoir's `is_weights`
are already `Fraction`s; its `StalenessPolicy` rescale is `Fraction(scale_num,
scale_den)`.

## 2. The API

### Record level (`martingale.record`)

```python
from fractions import Fraction
from martingale.record import Recorder, ReplayProvenance, OriginNotFound

rec = Recorder(workspace)

# (a) by origin: copy the prompt, tokens, bits and revisions of a fresh sequence in this record
seq = rec.replay_from(origin_digest,                    # digest of the fresh sequence
                      actor_id=0, sequence_id="step40/replay3",
                      draw_id="17/3", content_digest=buffer_digest,
                      is_weight=Fraction(3, 5), rescale=Fraction(1, 2),
                      reward=advantage)                 # optional; dtype="f32" by default

# (b) explicit: from the buffer's own copy (a buffer restored from another run, another framework)
seq = rec.replayed(actor_id=0, sequence_id="restored/3", prompt_ids=prompt_ids,
                   steps=[(revision_digest, token_id, logprob), ...],   # logprob: float, or "f32:..." bits
                   draw_id="17/3", content_digest=buffer_digest, is_weight=w,
                   rescale=1, origin_digest=None)       # with origin_digest the row must equal that sequence
```

Both return the `SequenceRecord` (`seq.provenance == "replayed"`,
`seq.replay` is the `ReplayProvenance`). The ledger fails closed inside one
transaction: an unpublished revision, a missing origin (`KeyError`), an origin
that is itself replayed (`ValueError`), or tokens or prompt that differ from the
origin (`ValueError`, "differ") leave nothing written. `TokenLedger.find_fresh_sequences(prompt_digest,
completion_ids)` is the lookup `replay_from` builds on; `count_replayed_sequences()`
counts them.

### Trainer level (`martingale.integrations.trl.MartingaleRecorder`)

```python
row_id = flight.register_replayed(
    trainer,
    prompt_ids=..., completion_ids=..., behavior_step=...,   # or origin_digest=... instead of these three
    draw_id="17/3", content_digest=buffer_digest,
    is_weight=Fraction(3, 5), rescale=Fraction(1, 2), advantage=0.75)
```

It finds the fresh sequence generated at `behavior_step` with that prompt and
completion, writes the replayed row through `replay_from`, and returns a new
`martingale_row_id`. Put that id in the batch row the buffer replaced; the loss
path then scores the row as the replayed sequence (`flight.on_scores` matches by
id), so its tokens land in the `replayed` bucket at the right lag. When several
fresh rows of that step share the content (GRPO groups repeat short completions
and their engine log-probs can differ), pass `behavior_logprobs=` (the per-token
log-probs the buffer stored) and the row whose recorded bits equal them is
chosen; with no unique match it raises `AmbiguousOrigin`, a subclass of
`OriginNotFound`. `OriginNotFound` means there is no usable origin (no fresh
sequence matches because the buffer was filled before the recorder attached or
restored from another run; a named digest is missing or is itself replayed; or
the match is ambiguous): put `-1` in the row id and `on_scores` skips the row
(`stats["skipped_rows"]`) instead of mis-attributing it. `flight.sequence_digest(row_id)`
returns the digest behind a row id of the last two generation batches, for a
buffer that prefers to keep the origin digest in its own metadata and pass
`origin_digest=`, which needs no lookup at all.

## 3. What the checker verifies, what the doctor reports

`checker/verify_tokens.py` (tier 1, binding), for every replayed row: the
`replay` block has exactly the keys above with well-formed values (non-empty
draw id, hex content digest, reduced non-negative weight, reduced positive
rescale, hex or null origin), `provenance` and the block agree, the block is
inside the sequence digest, and, when an origin is named, that origin is in the
export, verified, fresh, and equal to the row in prompt digest, prompt length and
every token's `(token_id, revision_digest, logprob_bits, topk)`. The report
carries `n_replayed_sequences`. A consequence shown by
`campaigns/mutation_tokens.py`: a replayed copy binds its origin's digest, so
re-signing or deleting the origin is caught even without an anchor; the
origin's scores are not inside its digest and are not witnessed.

`martingale doctor` keeps `by_lag` over every scored token and adds:

- `by_provenance.{fresh,replayed}`: the same per-lag buckets (tokens, ESS/n,
  mean |log r|, clipped fractions) and an attribution per provenance; a replayed
  bucket with no lag-0 tokens borrows the fresh floor and says so
  (`attribution.floor_source` is `"own lag-0"` or `"fresh lag-0"`).
- `replayed.sequence_weights`: over replayed sequences scored completely under
  some train revision, the ESS/n of the buffer's applied weights (importance
  weight times rescale) against the ESS/n of `exp(sum of train minus behaviour
  log-prob bits)` measured from the trainer's scores, and the per-row dispersion
  of `log(declared importance weight) - measured log-ratio` after centring per
  train step (buffers normalise weights per batch; the normaliser cancels). A
  dispersion above 1 nat gets a plain-language line.
- `replayed.n_without_origin`: rows whose behaviour bits are the buffer's claim only.

The live doctor (`IncrementalDiagnosis` / the TRL callback) adds
`martingale/replayed_tokens`, `martingale/replayed_share`,
`martingale/replayed_ess_fraction_lag{k}` and
`martingale/replayed_mean_abs_log_ratio_lag{k}` to the trainer's metrics; the
existing keys are unchanged.

## 4. Wiring it into Reservoir (the follow-up for the Reservoir session)

Compose the two mixins so Martingale records the fresh batch and allocates row
ids before Reservoir replaces dead rows:

```python
class Trainer(ReservoirReplayMixin, MartingaleGRPOMixin, GRPOTrainer):
    def __init__(self, *args, replay_buffer, flight_recorder, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer, self.flight_recorder = replay_buffer, flight_recorder
        # plus the MartingaleMonitor callback as MartingaleGRPOTrainer adds it
```

`ReservoirReplayMixin._generate_and_score_completions` calls `super()` (which
runs Martingale's `on_generation`) and then `replay_buffer.mix`. In
`ReservoirReplay._replay`, after the kept rows are written (`new = ...`):

```python
from martingale.integrations.trl import ROW_ID_KEY, OriginNotFound

flight = getattr(trainer, "flight_recorder", None)
if flight is not None and ROW_ID_KEY in new:
    ids = new[ROW_ID_KEY].clone()                       # write_rows carries the tensor through unchanged
    for k in kept:
        rollout, row = batch.rollouts[k], dead_rows[k]
        try:
            ids[row] = flight.register_replayed(
                trainer,
                prompt_ids=rollout.metadata["prompt_ids"], completion_ids=rollout.tokens,
                behavior_step=batch.model_versions[k],
                behavior_logprobs=rollout.logprobs,        # tells identical completions of one group apart
                draw_id=f"{batch.op_counter}/{k}",         # the sample record and the draw position
                content_digest=batch.content_digests[k],   # insert with content digests for this to exist
                is_weight=batch.is_weights[k],             # exact Fraction, batch-normalised: fine
                rescale=policy_scale.get(k, Fraction(1)),  # StalenessPolicy's scale_num/scale_den, when D1 lands
                advantage=weighted[k])
        except OriginNotFound:
            ids[row] = -1
            self.stats["martingale_unregistered"] = self.stats.get("martingale_unregistered", 0) + 1
    new = {**new, ROW_ID_KEY: ids}
```

That is the whole integration: about twenty lines plus the composed class.
Notes for the Reservoir side: `write_rows` already keeps `martingale_row_id`
(`pad_batch` shares entries it does not pad, `write_rows` copies the dict, and
`_check_write_args` refuses only per-row tensors of two or more dimensions,
which a 1-D id tensor is not), so nothing in `_trl_rows.py` changes; the fresh
rows Reservoir replaced stay in the record unscored, which is accurate
(generated, not trained on); `rollout.logprobs` are Reservoir's stored
behaviour log-probs (TRL's `old_per_token_logps` or its own no-grad forward),
and they equal Martingale's recorded bits only when Martingale took its bits
from the same tensor, which is the case without vLLM sampling log-probs;
`batch.content_digests` is empty unless groups were inserted with content
digests; with more than one process only the owner rank mixes and the ids it
returns are scattered to other ranks, which §6 covers. One TRL run with both
adapters attached is D2's acceptance test and has not been run yet; §5 is the
Martingale half of it, prepared and not yet launched.

## 5. Acceptance run (prepared 2026-10-08, not yet launched)

`benchmarks/modal/trl_grpo_vllm.py --replay-fraction 1/4` is the headline recipe
(Qwen2.5-0.5B-Instruct, TRL 1.13 GRPO, vLLM 0.28 colocate, bf16 autocast, 16 steps,
`num_iterations=2`, `steps_per_generation=2`, 8 generations per prompt) with a
synthetic replay buffer attached, `benchmarks/modal/replay_inject.py`. After every
generation batch (16 rows), the buffer replaces `floor(16/4) = 4` rows with completions
drawn deterministically (seeded by `seed` and the step) from earlier generation batches
and registers each one with `register_replayed(origin_digest=...)`: draw id
`"<step>/<row>"`, content digest BLAKE2b-256 over the prompt and completion ids, the
importance weight `exp(sum_t log pi_now - log pi_behaviour)` from one no-grad forward of
the current trainer weights, stored exactly, rescale 1, and the row's original advantage.
The row's tensors are rewritten to the origin's tokens and behaviour log-probs, so the
loss trains on them and the recorder scores them by id. The first batch (step 0) has no
earlier rows and is left alone; the batches at steps 4, 8 and 12 each carry 4 replayed
rows, 12 in all. The CPU test of the same path with the fake trainer is
`tests/test_replay_inject.py`.

```bash
modal run benchmarks/modal/trl_grpo_vllm.py --replay-fraction 1/4 --tag accept
# results: benchmarks/modal/results/trl_grpo_vllm_a10g_16steps_seed0_t1.0_replay1of4_accept/
#          (the usual files plus replay.json: rows replaced per step, draw ids, pool size)
```

What the record and the reports should show, given the bf16 baseline run
(`..._t1.0_probe`: 64 sequences, one `floor_jump` at step 13, checker ok):

- `verify.json`: `ok`, anchored on `head.txt`, `n_sequences == 76` (64 fresh + 12 replayed
  on actor 0's chain), `n_replayed_sequences == 12`; every replayed row names an origin in
  the same export and the checker's origin pass accepts it.
- `report.json`: `n_replayed_sequences == 12`; `n_unscored_sequences == 12` (the fresh rows
  the buffer replaced stay recorded and unscored: generated, not trained on);
  `by_provenance.fresh.by_lag` at lags 0..3 as in the baseline; `by_provenance.replayed.by_lag`
  only at lags >= 4 (a row generated at step 0 and trained at steps 4..7 has lags 4..7; the
  batches at 8 and 12 may draw from any earlier batch, so lags up to 15 are possible) with
  `attribution.floor_source == "fresh lag-0"`; `replayed.n_sequences == 12`,
  `n_scored_sequences == 12`, `n_without_origin == 0`; `replayed.sequence_weights.n == 12`,
  `declared_ess_fraction` and `measured_ess_fraction` both reported, `n_dispersion_rows > 0`
  and `log_weight_dispersion` well under the 1-nat line (the declared weight is measured
  with the weights of the generation step; the trainer's first complete score of a row is
  at that step or the next, so the two differ by at most one optimizer step of drift, which
  the baseline puts at 1e-4 to 3e-2 nats per token).
- `report.md`: one "Replayed rows: 12 sequences (... 25% of scored tokens) at lags 4..N"
  line with the declared-versus-measured ESS/n; no "disagree by ... nats" line and no
  "no fresh origin" line.
- `alarms.json`: the baseline's alarms only (a `floor_jump` or `floor_tail` late in the run is
  possible on this collapsing recipe); no `stale_server`, no `unmatched` (replayed rows are
  matched by id: `stats.skipped_rows == 0`, `stats.unmatched_rows == 0`), and
  `martingale/replayed_tokens > 0` with `martingale/replayed_share` near 0.25 from step 5 on,
  plus `martingale/replayed_ess_fraction_lag{k}` for the lags above.
- `replay.json`: `replaced_rows == 12`, `unregistered_rows == 0`, `per_step` with three entries
  (steps 4, 8, 12) of four rows each.

What it proves: on real engine and trainer numbers, rows a buffer hands the trainer are
bound in the record with their provenance, scored by id at the right lag, bucketed apart
from fresh rows by the doctor, and accepted by the independent checker with the origin
binding. What it does not prove: anything about Reservoir's own wiring (§4), its checker,
or the join between the two records; that is the other half of D2 and needs the Reservoir
session's integration first.

## 6. More than one process: the row-id contract

Reservoir's `mix_distributed` gathers every rank's generation slice to rank 0, replaces
dead rows there, and scatters the rewritten slices back; a replayed row therefore reaches
the loss on a rank other than the one whose recorder registered it. The Martingale side
of that is in `MartingaleRecorder` (`tests/test_distributed_rows.py` plays it out with two
fake ranks over one workspace):

- **A row id names its rank.** `martingale_row_id = rank << 40 | local`, with
  `pack_row_id(rank, local)` / `unpack_row_id(row_id)` in `martingale.integrations.trl`;
  the rank is the trainer's `accelerator.process_index` and the local counter is the
  recorder's own. Rank 0 ids are the plain counter, so single-process records are
  unchanged. `-1` stays the skip sentinel.
- **Every rank shares one workspace.** All ranks open the same `tokens.db` path (one node,
  or a filesystem every rank sees); each rank is its own actor chain (`actor_id = rank`)
  and SQLite serialises the writers (30 s busy timeout). Every id a rank allocates, fresh
  or replayed, is bound to its sequence digest in the ledger's `row_ids` table,
  bookkeeping outside the record: not exported, not digested, not checked, kept when
  `tokens.db` is copied, created on first write so reading an old record never rewrites it.
- **A rank resolves any id through the ledger.** `on_scores` looks a row id up in its own
  maps, then in the table (any rank, any age); ids from another rank are counted in
  `stats["foreign_rows"]`. With separate workspaces per rank a foreign id cannot be
  resolved: it counts as `unmatched_rows` and the `unmatched` alarm fires, instead of the
  row being scored against the wrong sequence.
- **The owner registers, the record resolves the origin.** Rank 0 calls
  `register_replayed` on its own recorder; `find_fresh_sequences` is a ledger query, so an
  origin generated on another rank is found through the shared file. The returned id is
  rank 0's; put it in the scattered row. `OriginNotFound` still means `-1`.

What Reservoir must do, and nothing more: carry `martingale_row_id` through
`gather_object`, `concat_shards`, `write_rows` and `slice_shard` unchanged (it already
does: a 1-D per-row tensor is copied, never renumbered); register on the owner rank with
the owner's flight recorder; and launch every rank with the same workspace path. Ranks
that disagree on `global_step` are already refused by `mix_distributed`. A real
multi-GPU run of this path has not happened; the evidence is the two-rank fake.
