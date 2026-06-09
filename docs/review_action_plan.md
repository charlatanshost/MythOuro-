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
- ⬜ **P0.4** `ContinuousDepthwiseBatcher` cross-loop buffer **batch mismatch**
  when the active set shrinks (ragged `cat` in `CrossLoopAttention`). Verified
  *likely real* (inference.py:546–565). *Fix:* index buffer per original row
  (store full-B snapshots, slice by `active_idx`), or disable cross-loop in this
  batcher, or pad/scatter active rows back before buffering. *Test:* mixed-halt
  batch + `use_cross_loop_attention=True` → no crash; late-halting row matches a
  reference full-depth forward. (Note: I already fixed the 3-tuple-unpack crash
  there for P0.2; the *batch-dim* mismatch is the remaining P0.4 issue.)
- ⬜ **P0.5** `UncertaintyHead` calibration validity — trained on `h_K` logits,
  consumed on per-loop states in `forward_trajectory`/best-of-trajectory.
  *Fix before MoDr:* measure ECE per-loop on a held-out set
  (`forward_trajectory(force_full_depth=True)`); if poor, prefer **per-loop CE**
  as the MoDr best-exit supervision target (already the roadmap's safer option).

## Bonus (done while re-baselining)
- ✅ `eval.harness` rebuilds from the checkpoint's own cfg (was hardcoded
  `mythouro_1b`). (commit ea395b2)
- ✅ `eval.harness --tokenizer` default → Ouro 49152 vocab (was gpt-neo 50k →
  out-of-range ids).
- ⬜ eval **metrics don't clamp inputs to `max_seq_len`** (long doc → RoPE/embed
  index error). *Fix:* clamp `seq_len` to `cfg.max_seq_len` in metrics.py
  perplexity/ece/loop_efficiency. (Worked around by small default seq_len.)

## The high-value next move
- ⬜ **Fresh training run on the fixed code.** The re-baseline only captured
  P0.3's emission gain on *already-trained* weights; v1–v5 were *trained* under
  P0.1 (clobbered inits) + P0.2 (last-loop balance), so a fresh run should
  improve further. Natural candidate: the **MoE-vs-dense ablation** — now both
  *unblocked* (P0.2 fixed → MoE arm balances correctly) and *more trustworthy*.
  `mythouro_distill_tiny` vs `mythouro_distill_tiny_dense`, ≥2 seeds, ~4–5 h/arm
  on the 5070 (per the wired spec in this roadmap's "Gating experiment").

## P1 — performance / measurement (the review's items, unstarted)

- ⬜ **P1.1** MLA caches RoPE keys expanded per-head (~2–3× cache bloat) → cache
  `(B,T,1,rope_dim)`, broadcast at attention.
- ⬜ **P1.2** MLA decode lacks weight absorption (O(S) recompute/token/loop) →
  DeepSeek absorption (inference-only rewrite). *Spec into the Rust runtime.*
- ⬜ **P1.3** MoE dispatch host syncs (`sel.any()` per expert per loop) — ~768
  syncs/micro-step in the v5 regime; plausibly a real chunk of the 33 s/step.
  *Fix:* drop the `.any()` guard or argsort+split dispatch. **Do before/with the
  ablation** (also unblocks `torch.compile`). Verify with `bench_step`.
- ⬜ **P1.4** Multi-scale injection recomputes the 3 projections every loop
  (`e` is frozen) → precompute once, blend per loop.
- ⬜ **P1.5** ACT gives no decode speedup; `generate` default `n_loops=8` > trained
  4 → change default to `cfg.max_loop_iters`; (opt) decode early-exit backfill.
- ⬜ **P1.6** `UncertaintyGatedGenerator` clones the whole KV cache every step →
  length-slice rewind.
- ⬜ **P1.7** `SpeculativeDecoder` slower than vanilla (no-cache drafting, verify
  depth 16) → cached drafting + aligned depths.
- ⬜ **P1.8** small hot-path items (one cleanup PR): LoRA `t_idx` H2D transfer;
  `loop_index_embedding` reallocates each call; `_causal_mask` materialized every
  forward (use `is_causal`); skip training-mode `h_out` accumulation (now dead
  after P0.3); gradient-checkpoint side-effect assertion; fast-tokenizer check.
- ⬜ **P1.9** `distillation_loss` soft KL ignores `ignore_index` → mask padded
  rows (harmless now, footgun at scale-up).

## P2 — strategic (the review's items)

- ⬜ **P2.1** Run the MoE-vs-dense ablation (after P0.2 ✅ + ideally P1.3).
- ⬜ **P2.2** Promote per-step weighted loop loss (`--loop-loss per_step_weighted`)
  above Net2Wider — principled fix for P0.3/P0.5, trains exit gates with task
  signal. (P0.3 took option 2 short-term; this is the option-3 upgrade.)
- ⬜ **P2.3** Consider Muon optimizer for from-scratch runs (½ the AdamW state).
- ⬜ **P2.4** `torch.compile` on the dense ablation arm (needs P1.3).
- ⬜ **P2.5** Quarantine/delete `mythouro/moda.py` (1063-line unused duplicate;
  name-collides with MoDr).
- ⬜ **P2.6** Define a frozen minimal config (loops + LTI + dense FFN, rest off),
  A/B each mechanism back in — several were not doing what their comments claimed.
- ⬜ **P2.7** "Fix release" re-baseline: full v4 eval + inspector on the same
  checkpoint, diff vs archived. (Partial: v2 ppl/ece/loop_eff done above.)

## Suggested next-session order (from the review)
1. **P0.4 + P0.5** (finish the P0 tier) + the eval-metric `max_seq_len` clamp.
2. **P1.3** (MoE dispatch) + **P1.4** (MS-injection hoist) — `bench_step` before/after.
3. **MoE-vs-dense ablation** (P2.1), optionally `torch.compile` the dense arm (P2.4).
4. **Per-step weighted loop loss** (P2.2) — if it wins, supersedes P0.3's option 2.
5. P1.1/P1.2 MLA fixes (+ spec into the Rust runtime doc); P1.5–P1.7; P2.5 anytime.
