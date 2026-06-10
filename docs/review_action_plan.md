# Code-review action plan — status tracker

Tracks the external review (`mythouro_code_review_findings.md`, Fable 5,
2026-06-09), independently verified against the code. **Pick up here next
session.** Done items link to their commit; TODO items keep the review's notes.

Legend: ✅ done · ⬜ todo · 🔁 partial

---

## P0 — correctness (fix before next training run)

- ✅ **P0.1** `_init_weights` clobbered zero-inits → `_skip_global_init` marker.
  (commit d7c015c) +2 invariant tests.
- ✅ **P0.2** router telemetry = last loop only → checkpoint-safe per-loop
  accumulation via `_loop_body` return. (commit 50cffa1) +2 tests.
- ✅ **P0.3** eval emitted never-trained `h_out` → return `h_K`. (commit 557affd)
  +1 parity test. **Re-baseline: PPL 46.3→39.25, ECE 0.058→0.042 on v2.**
- ✅ **P0.4** batcher cross-loop buffer batch/row mismatch → CrossLoopAttention
  split into `_maybe_snapshot`+`_attend`; batcher stores full-B snapshots, attends
  on per-row slices, and emits per-row h_K (P0.3-consistent) instead of the h_out
  blend. (commit e0cf187) +2 tests incl. per-row equivalence vs single-row forward.
- ✅ **P0.5** per-loop calibration MEASURED (`tools/per_loop_calibration.py`, run
  on v2+v4 → `reports/per_loop_calibration_p05.md`): loops 1–3 calibrated (ECE
  0.01–0.04), **loop 0 badly miscalibrated** (ECE 0.17–0.22, error understated
  ~0.2 — curriculum starts at 2, loop 0 never an emission loop). Consequences
  applied: `BestOfTrajectoryGenerator` defaults `min_loops=2`; **MoDr supervision
  = per-loop CE (mandated, not just "safer")**; roadmap MoDr section updated.

## Bonus (done while re-baselining)
- ✅ `eval.harness` rebuilds from the checkpoint's own cfg (was hardcoded
  `mythouro_1b`). (commit ea395b2)
- ✅ `eval.harness --tokenizer` default → Ouro 49152 vocab (was gpt-neo 50k →
  out-of-range ids).
- ✅ eval metrics clamp `seq_len` to `cfg.max_seq_len`
  (perplexity/loop_efficiency/ece — done with P0.5's commit).

## Pre-training checklist (status as of 2026-06-09)

Everything that gates a trustworthy training run:

- ✅ P0.1–P0.3 (the bugs that would pollute a fresh run: init, routing balance,
  emission) — fixed + tested.
- ✅ `--seed` + `--start-loops` flags exist in both training scripts (commit
  b7e0b1d) — the ≥2-seed ablation protocol is now actually runnable.
- ⬜ **GPU smoke (user-run, ~10 min, MUST do before any overnight):** this
  session's hot-path changes (P0.2 checkpoint-flow telemetry, P1.3 dispatch,
  the autocast MoE fix, h_out removal) were CPU-validated only. Run ~50–100
  steps on the 5070 and eyeball the log: loss decreasing, no NaN, sane tok/s,
  MoE util line non-degenerate. E.g.
  `python -m training.sft --resume archived_models/mythouro_distill_tiny_v1/step_0005000.pt --total-steps 50 --seed 0`
- ⬜ Optional: the P1.3 GPU A/B (`bench_step` on 31a48b0 vs main).
- **Run-time decision:** `--start-loops 1` (calibrates the head at loop 0, new
  recipe) vs `2` (matches v1–v5). Whichever is picked, use it for BOTH ablation
  arms.

## The high-value next move
- ⬜ **Fresh training run on the fixed code.** The re-baseline only captured
  P0.3's emission gain on *already-trained* weights; v1–v5 were *trained* under
  P0.1 (clobbered inits) + P0.2 (last-loop balance), so a fresh run should
  improve further. Natural candidate: the **MoE-vs-dense ablation** — now both
  *unblocked* (P0.2 fixed → MoE arm balances correctly) and *more trustworthy*.
  `mythouro_distill_tiny` vs `mythouro_distill_tiny_dense`, ≥2 seeds, ~4–5 h/arm
  on the 5070 (per the wired spec in this roadmap's "Gating experiment").

## P1 — performance / measurement

- ✅ **P1.1** MLA rope-key cache bloat → caches compact `(B,S,1,rope_dim)`,
  broadcast at attention (commit f66a524). Cached-decode-vs-full-forward
  equivalence test added. **Finding:** with multi-scale injection / cross-loop
  attention ON, cached single-token decode *legitimately* diverges from a full
  forward (window pooling / loop snapshots only see the current token at
  decode) — pre-existing Part-2 semantics, feeds the P2.6 complexity budget.
- ⬜ **P1.2** MLA decode lacks weight absorption (O(S) `kv_up` recompute per
  token per loop) → DeepSeek absorption: fold `W_uk` into the query path
  (`q_nope @ W_uk^T`, score directly against cached `c_kv`) and `W_uv` into
  `wo`, attention in latent space, zero per-step reconstruction. Inference-only
  rewrite; **bake into the Rust runtime from day one**. Needs GPU validation.
- ✅ **P1.3** MoE dispatch host syncs → argsort-grouped dispatch, ONE host
  transfer per call; torch.compile-friendlier (commit fe1bfbe). Equivalence
  test vs naive per-token reference. CPU bench neutral within noise as
  expected (the syncs only exist on CUDA) — **A/B on the 5070**: before =
  31a48b0, after = fe1bfbe+, ideally on a 96-expert config.
- ✅ **P1.4** multi-scale injection hoist → `precompute()` once per forward +
  `blend_views()` per loop (commit fe1bfbe).
- 🔁 **P1.5** (a) ✅ `generate` defaults to `cfg.max_loop_iters` (was 8 = 2×
  trained depth). (b) ⬜ decode early-exit cache-backfill — *semantic* change
  (backfills deeper-loop K/V with loop-t values), wants GPU validation; (c) =
  the Rust ACT-compaction work.
- ✅ **P1.6** zero-copy cache rewind (commit 352f266) — better than the
  review's length-slice: cache entries are replaced, never mutated in place,
  so a structure-only ref snapshot is a correct rewind with zero copies.
- 🔁 **P1.7** (commit 6dd0506) ✅ verify depth aligned to trained; ✅ residual
  resample reuses stored draft dists (was: full extra draft forward); ✅ bonus
  token sampled from existing verify logits (was: full extra verify forward).
  ⬜ **Remaining — cached drafting (design, ready to implement):** two
  *depth-separate* caches (draft@d, verify@v — loop cache keys differ in
  count, so one shared cache can't serve both). Maintain per-cache
  `committed_len`; each phase feeds the tokens beyond its commit point; after
  acceptance of `a` candidates, roll BOTH caches back by **length-slicing**
  every entry to `committed + fed_prefix + a` along dim 1 (valid because
  causal K/V for a kept position is independent of the dropped suffix; the
  verify pass's own cache writes provide the kept positions). Mind the sink
  offset (prefill cache lengths include sink positions). Needs GPU validation.
- 🔁 **P1.8** ✅ LoRA `t_idx` H2D; ✅ `loop_index_embedding` memo cache; ✅
  vestigial ACT-blend `h_out` accumulation removed entirely (dead since
  P0.3); ✅ ckpt side-effect covered by the P0.2 grad-liveness test. ⬜
  `_causal_mask` → `is_causal` SDPA refactor (touches the attention cascade +
  cached-decode mask alignment; real win — SDPA's fused causal path — but
  wants GPU validation). ⬜ fast-tokenizer check (one-liner, anytime). FineWeb
  buffer pointer: skipped per review (irrelevant at current scale).
- ✅ **P1.9** distillation soft-KL now masked to `ignore_index`-valid positions
  (commit 6998ec5).

## P2 — strategic (the review's items)

- ⬜ **P2.1** Run the MoE-vs-dense ablation (after P0.2 ✅ + ideally P1.3).
- ⬜ **P2.2** Promote per-step weighted loop loss (`--loop-loss per_step_weighted`)
  above Net2Wider — principled fix for P0.3/P0.5, trains exit gates with task
  signal. (P0.3 took option 2 short-term; this is the option-3 upgrade.)
- ⬜ **P2.3** Consider Muon optimizer for from-scratch runs (½ the AdamW state).
- ⬜ **P2.4** `torch.compile` on the dense ablation arm (needs P1.3).
- ✅ **P2.5** `moda.py` quarantined to `examples/` next to its only consumer
  (commit 9b2d1ee); the package no longer ships it.
- ⬜ **P2.6** Define a frozen minimal config (loops + LTI + dense FFN, rest off),
  A/B each mechanism back in — several were not doing what their comments claimed.
- ⬜ **P2.7** "Fix release" re-baseline: full v4 eval + inspector on the same
  checkpoint, diff vs archived. (Partial: v2 ppl/ece/loop_eff done above.)

## Suggested next-session order (updated 2026-06-09, second pass)
1. ✅ ~~P0 tier + eval clamp~~ — complete.
2. ✅ ~~P1.3 + P1.4~~; ✅ P1.1, P1.5a, P1.6, P1.7-subset, P1.8-most, P1.9, P2.5.
3. **On the 5070 (user-run, no training):** A/B `bench_step --device cuda:0`
   before(31a48b0)/after(main) to measure the P1.3 sync win.
4. **Training runs — deliberately deferred by user (2026-06-09: "save
   retraining for later"):** MoE-vs-dense ablation (P2.1, fully unblocked),
   fresh retrain on fixed code, per-step weighted loop loss (P2.2). All wired;
   run when the user gives the go.
5. Code work remaining: **P1.2** (MLA absorption, spec above), **P1.7 cached
   drafting** (design spec'd above), **P1.5b** decode backfill, **P1.8**
   `is_causal` refactor + fast-tokenizer check — all want GPU validation;
   **P2.6** minimal-config definition (doc work, can do anytime); P2.3/P2.4
   ride along with the deferred training runs; **P2.7** re-baseline rides the
   fix-release (v2 partial done).
