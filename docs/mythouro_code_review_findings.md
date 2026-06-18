# MythOuro Code Review — Findings, Fixes, Improvements

External review of `charlatanshost/MythOuro-` @ main (2026-06-09), performed by
Claude (Fable 5). Each finding was independently verified against the code
before being acted on; status lives in
[`review_action_plan.md`](review_action_plan.md).

Scope: `mythouro/main.py`, `mythouro/inference.py`, `mythouro/training_utils.py`,
`training/distill.py`, `eval/harness.py`, `training/1b_fine_web_edu.py`,
`mythouro/grow.py`, `mythouro/checkpointing.py`, `docs/roadmap.md`.

Priorities: **P0** = correctness bug, fix before next training run. **P1** =
measured/likely perf win or measurement-validity issue. **P2** = strategic
improvement. Each item has a verification step — implemented as a test where
marked.

---

## P0 — Correctness bugs

### P0.1 `_init_weights` clobbers deliberate zero-inits
- **Where:** `mythouro/main.py`, `MythOuro.__init__` calls `self._init_weights()`
  *last*; `_init_weights` re-inits every `nn.Linear` to N(0, 0.02).
- **Bug:** Overwrites `CrossLoopAttention.o_proj` (zero-init → identity residual)
  and `UncertaintyHead.net[-1]` (zero weight+bias → initial score 0.5). Both
  design claims in their docstrings are false in the running code. Also
  invalidates the `new_component_warmup` premise that "zero-output-init
  components self-warm".
- **Impact:** All checkpoints v1–v5 trained with cross-loop attention injecting
  random noise into `h` from step 0.
- **Fix:** Have `_init_weights` skip modules that self-initialize (marker
  attribute on the Linear).
- **Verify (test):** post-construction, assert the zero-inits survived.
- **⚠ Discovered consequence (fork follow-up, 2026-06-16 — not part of the original
  review):** the "random noise into `h`" this fix removed was **load-bearing for
  free generation.** It kept the output distribution diffuse and prevented an
  exposure-bias repetition spiral — which is why pre-fix v4 generated *varied* text
  while post-fix checkpoints collapse into repetition under greedy decoding. The fix
  is still correct (representations are healthy; verified via
  `tools/collapse_metrics.py`); the diffuse-distribution benefit must be recovered
  the principled way — **on-policy/GKD training**, not accidental noise. Full
  diagnosis: `docs/training_runs.md` (06-16) + `docs/review_action_plan.md`.

### P0.2 MoE router telemetry only captures the LAST loop iteration
- **Where:** `MoEFFN.forward` stashes `_last_router_logits` /
  `_last_expert_counts`, overwritten every call. ONE MoEFFN instance, called
  `n_loops` times per forward; collectors read once after the forward → see
  only loop K−1.
- **Impact:** load-balance and sparse-activation losses only constrain the
  final loop's routing; the DeepSeek-V3 bias updater balances only the final
  loop's counts; all cv / min% / max% metrics (incl. the v5 "expert-count
  ceiling" conclusion) were measured through this bug — soften that "answered"
  stamp.
- **Fix:** Accumulate per-forward buffers; collectors concatenate across loops;
  counts summed before the bias update. Keep gradient only for the logits.
- **Verify (test):** forward at `n_loops=4` → collected router logits cover
  `4·B·T` rows; counts sum to `4·B·T·topk`.
- **Sequencing:** fix BEFORE the MoE-vs-dense ablation, or the MoE arm is
  handicapped and the pre-registered comparison is ambiguous.

### P0.3 Eval-mode `h_out` under-sums + train/infer emission mismatch
- **Where:** `RecurrentBlock.forward`. Training returns `h_K` (documented
  ACT-collapse fix). Eval/no-cache returns `h_out = Σ w_t·h_t`; the remainder
  mass is committed only on convergence break or threshold crossing — positions
  below threshold at loop exhaustion are magnitude-suppressed.
- **Compounding:** the coda/norm/head were only ever trained on `h_K`; at
  inference they receive a blend weighted by λ's trained only toward a uniform
  KL prior — zero task signal.
- **Impact:** every archived eval number was measured through this
  never-trained path.
- **Fix options:** (1) minimal — final remainder commit; (2) **recommended
  short-term — return `h_K` at inference too** (ACT = early-exit criterion
  only); (3) principled — Ouro-style per-step weighted loss (already spec'd in
  the roadmap; promote it).
- **Cheap experiment:** re-run the v4-class eval with option 2 — if PPL drops,
  that's free measured capability. *(Outcome: it did — v2 PPL 46.3→39.25.)*

### P0.4 `ContinuousDepthwiseBatcher` cross-loop buffer batch mismatch
- **Where:** the batcher passes active-subset hidden states into the cross-loop
  buffer; when the active set shrinks, buffer entries have different batch dims
  → the attention `cat` crashes or rows misalign (a sequence attends to another
  sequence's history). Also inherits P0.3's under-summed `h_out`.
- **Fix:** store full-B snapshots, slice per row at use (or disable cross-loop
  in this batcher).
- **Verify (test):** engineered mixed halt depths + cross-loop on → no crash;
  late-halting row matches a reference forward.

### P0.5 `UncertaintyHead` trained on one distribution, consumed on others
- **Where:** the calibration loss trains the head against training-forward
  (`h_K`) logits; inference scores `h_out` logits; `forward_trajectory` /
  best-of-trajectory score **per-loop** states it was never calibrated on.
- **Impact:** headline ECE doesn't certify the per-loop uses. Directly
  load-bearing for MoDr best-exit labels.
- **Fix:** measure ECE per loop (`forward_trajectory(force_full_depth=True)`)
  on held-out data; if poor, use per-loop CE as the MoDr supervision target.
  *(Outcome: loop 0 badly miscalibrated on v2 AND v4 — per-loop CE mandated.)*

---

## P1 — Performance / measurement

### P1.1 MLA caches RoPE keys expanded per-head (~2–3× cache bloat)
Decoupled RoPE keys are shared across heads; caching the expanded form
multiplies that cache component by `n_heads`. Fix: cache `(B,T,1,rope_dim)`,
broadcast at attention time.

### P1.2 MLA decode lacks weight absorption (O(S) recompute per token per loop)
Each decode step reconstructs K_nope/V over the full cached sequence. Fix
(DeepSeek absorption): fold `W_uk` into the query path and `W_uv` into `wo`;
attention in latent space. Inference-only rewrite; bake into the Rust runtime
from day one.

### P1.3 MoE dispatch host syncs (`sel.any()` per expert per loop)
One host sync per expert per loop per recompute — ~768 syncs/micro-step in the
v5 regime; plausibly a real chunk of its ~33 s/step. Fix: argsort dispatch with
one host transfer; also unblocks `torch.compile`. Verify with `bench_step`.

### P1.4 Multi-scale injection recomputes constant work every loop
`e` is frozen across loops; only the blend weights depend on `t`. Precompute the
three projections once, blend per loop.

### P1.5 ACT gives zero decode speedup; `generate` defaults exceed trained depth
(a) default `n_loops` to `cfg.max_loop_iters`; (b) optional decode early-exit
cache backfill; (c) full per-token version = the Rust ACT-compaction work.

### P1.6 `UncertaintyGatedGenerator` clones the entire KV cache every step
Docstring claims incremental cost; code clones everything. Fix: length-slice
rewind (or better, structure-only snapshot — entries are replaced, never
mutated in place).

### P1.7 `SpeculativeDecoder` is slower than vanilla decode
No-cache drafting (full prefill per draft token), full-prefill verify and bonus
passes, `verify_loops=16` vs trained 4. Fix: cached drafting with rollback,
store the full draft distribution, align depth defaults.

### P1.8 Small hot-path items (batch into one cleanup PR)
LoRA per-loop H2D transfer; `loop_index_embedding` reallocation;
`_causal_mask` materialization (use `is_causal`); dead training-mode `h_out`
accumulation; checkpoint side-effect assertion; fast-tokenizer check; FineWeb
buffer pointer (skip — irrelevant at current scale).

### P1.9 `distillation_loss` soft term ignores `ignore_index`
Hard CE respects it; soft KL averages over ALL positions. Harmless on packed
data, a silent footgun for blended scale-up phases. Mask the KL rows.

---

## P2 — Strategic / structural

- **P2.1** Run the MoE-vs-dense ablation (after P0.2).
- **P2.2** Promote per-step weighted loop loss above Net2Wider — principled fix
  for P0.3/P0.5.
- **P2.3** Consider Muon for from-scratch runs (½ the AdamW state).
- **P2.4** `torch.compile` on the dense ablation arm (after P1.3).
- **P2.5** Quarantine/delete `mythouro/moda.py` (1,063-line unused duplicate;
  name-collides with MoDr).
- **P2.6** Complexity budget: define a frozen minimal config, A/B each
  mechanism back in — several were not doing what their comments claim.
- **P2.7** "Fix release" + re-baseline before rented compute.

---

## Test additions (invariant tests — the suite's gap)

303/303 passing did not catch P0.1 or P0.2 because both are invariant
violations, not logic errors. Add: zero-init invariants; telemetry coverage;
emission-weight normalization; train/eval emission parity; aux-loss gradient
liveness under checkpointing; depthwise-batcher equivalence.

## Suggested order of work

1. P0.1, P0.2, P0.3(option 2), P1.8 small items + invariant tests.
2. Re-run eval suite → diff vs archived; soften open-question #1.
3. P1.3 + P1.4 → `bench_step` before/after.
4. MoE-vs-dense ablation (P2.1), optionally with `torch.compile` (P2.4).
5. Per-step weighted loop loss (P2.2).
6. P1.1/P1.2 MLA fixes (spec into the Rust runtime).
7. P1.5–P1.7 as time allows; P2.5 anytime.
