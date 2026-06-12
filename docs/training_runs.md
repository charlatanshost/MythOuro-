# Training runs — comparison table

Living comparison of every training session's eval stats, built from the raw
eval JSONs where they exist (paths in the last column). Update after each run.

> **Comparability caveats** (read before drawing conclusions):
> 1. **Eval emission path changed 2026-06-09 (P0.3):** v1–v5 numbers were
>    measured through the old under-summed `h_out` blend (pessimistic — fixing
>    it alone moved v2 from PPL 46.3→39.25 on the same weights). Runs from
>    `moe_s0` onward use the trained `h_K` path. Cross-era comparisons are
>    directional, not exact.
> 2. **Absolute PPL is flattered by FineWeb train/eval stream overlap** —
>    applies to all runs equally, so within-protocol comparisons stand; don't
>    quote against external baselines.
> 3. v2's PPL rise vs v1 is expected (SFT specialises away from raw web text).

## Run overview

| Run | Date | Arch / params | Method + recipe deltas | Steps | Final PPL | loop_eff | ECE | Verdict |
|-----|------|---------------|------------------------|------:|----------:|---------:|----:|---------|
| **v1** | 06-01 | MoE 24exp / 278M | distill from scratch; warmup 500, depth-reg 0.3, mb1/ga8 (**the proven recipe**); pre-fix code (P0.1 noise injection active) | 5000 | 37.42 | 0.50 | 0.041 | archived |
| **v2** | 06-04 | = v1 | SFT on v1 (math+code) | 3000 | 46.27¹ | 0.50 | 0.058 | archived |
| **v3** | 06-04/05 | MoE 48exp / 420M | MoE-grown + SFT | 3500 | n/a² | — | — | archived |
| **v4** | 06-05 | = v3 | OpenHermes SFT, fp32 | 3000 | n/a² | — | — | archived; best PoC behaviour |
| **v5** | 06-06 | MoE 96exp / 632M | 2nd MoE growth + SFT | 2887 | n/a² | — | — | archived; expert-ceiling data point |
| *(flatline)* | 06-10 | dense + MoE, both | ablation attempt on **script defaults** (warmup 200, depth-reg 0.1) | ~2000 | 4,462–8,301 | 1.0 (no halt) | — | **dead** — see failure modes; root-caused to recipe defaults |
| **moe_s0** | 06-10 | MoE 24exp / 278M | ablation arm 1: distill from scratch; proven recipe; **post-fix code** | 4000 | **5.72** | 0.500 | 0.015 | ✅ **6.5× better than v1 final in 1k fewer steps** |
| **dense_s0** | 06-10/11 | dense / 180M | ablation arm 2: same recipe/seed as moe_s0, FFN only difference | 4000 | **22.66** | 0.500 | 0.026 | ✅ complete — **MoE wins 4.0× at seed 0**; note dense still beats v1's 37.4 |
| **moe_s1** | 06-11 | MoE 24exp / 278M | ablation arm 3: seed-1 repeat of moe_s0 | 4000 | **22.23** | 0.500 | 0.025 | ✅ complete — **~4× seed spread vs moe_s0 (5.72)**; ≈ dense_s0 (22.66). dense_s1 now decisive |
| **dense_s1** | 06-12 | dense / 180M | ablation arm 4: seed-1 repeat of dense_s0 | 4000 | **20.83** | 0.500 | 0.041 | ✅ complete — **dense BEATS MoE at seed 1** (20.83 < moe_s1 22.23); seeds disagree |

¹ SFT specialises toward chat; web-text PPL rises by design.
² v3–v5 predate per-run eval archiving — only inspector/behavioural results in
their MODEL_CARDs (reconstructed 2026-06-08).

## PPL trajectory (per 1k-step eval)

| Step | v1 (old eval) | **moe_s0** | **dense_s0** | **moe_s1** | **dense_s1** |
|-----:|------:|------:|------:|------:|------:|
| 1000 | 368.4 | 559.9 | 578.0 | 475.4 | 525.9 |
| 2000 | 178.6 | 112.1 | 150.3 | 122.9 | 132.1 |
| 3000 | 81.7 | **11.1** | 33.6 | 34.0 | 31.3 |
| 4000 | 51.8 | **5.72** | 22.7 | 22.2 | 20.8 |

## ABLATION VERDICT (2026-06-12): inconclusive — no robust MoE advantage; seed variance dominates

Complete 2×2 (within-seed is the only valid MoE-vs-dense comparison):

| | seed 0 | seed 1 |
|---|---:|---:|
| **MoE** | 5.72 | 22.23 |
| **dense** | 22.66 | 20.83 |
| winner | MoE 4.0× | **dense 1.07×** |

**The seeds disagree on direction.** Three of four runs cluster at 20.8–22.7;
only `moe_s0` (5.72) is an outlier. Applying the pre-registered rule (keep MoE
only if >5–10% better *across* seeds): **NOT satisfied** — seed 1 favours dense.
Honest conclusion: **`moe_s0`'s 5.72 was a favourable-seed outlier, not an
architecture win; MoE shows no robust advantage at this scale/seed-count.**

The real finding is **seed variance**: a 4× PPL swing from seed alone means
4000-step / ~16M-token runs are *underpowered* to separate architecture from
noise — the comparison is premature, not settled in dense's favour either.
Constant across all four fixed-code runs: loop_eff converges to exactly 0.500.

**Consequences:**
- **MoDr's gate is NOT cleanly passed** — do not commit to the unified
  expert+depth router on this evidence.
- To actually resolve it: (a) more seeds (≥3–5) and/or (b) longer runs (token
  count is the variance lever) before claiming either way; or (c) adopt dense
  for simplicity, since MoE's extra complexity is currently unjustified by data.
- `moe_s0` (5.72) remains the **best single checkpoint** and the right SFT base
  (v6) regardless — "best checkpoint" and "architecture verdict" are separate
  questions. Its luck doesn't make it worse; it just isn't *evidence for MoE*.

Dense sidecars: `checkpoints_ablation_dense_s{0,1}/`.

Note the crossover: moe_s0 starts *slower* (560 vs 368 at step 1000 — longer
warmup in effect) then collapses past v1 between steps 2000–3000, exactly as
the full-depth curriculum phase (step 2000+) kicks in.

## loop_efficiency / ECE trajectory (moe_s0)

| Step | loop_eff | ECE |
|-----:|---------:|----:|
| 1000 | 0.926 (full depth, ACT untrained) | 0.145 |
| 2000 | 0.572 | 0.096 |
| 3000 | 0.496 | 0.011 |
| 4000 | **0.500** (design-band center) | 0.015 |

## Behavioural read: moe_s0 test prompts (2026-06-11)

Inspector on the PPL-5.72 checkpoint (4-prompt set, T=0.7/top_k=40, raw output:
`reports/inspect_moe_s0.txt`). **Context: distill-only, no SFT** — the fair
comparison is v1 (also distill-only), not the SFT'd v2/v4.

| Prompt | Output character | Stop | Notes |
|--------|------------------|------|-------|
| "The recurrent depth transformer is" | degenerate repetition ("compared"×12) | `cycle` ✅ | guard fired |
| ChatML "What is 2+2?" | newline/digit loops | `cycle` ✅ | no chat structure (expected pre-SFT) |
| `def fibonacci(n):` | **correctly indented `
    """` docstring opening**, then collapse | `cycle` ✅ | genuine learned code convention — the one bright spot |
| Roman Empire trivia | "R R R R…" | `cycle` ✅ | guard fired |

**Findings:**
1. Real *structure* learning is visible (the docstring), but open-ended
   generation still collapses into repetition attractors — classic small-LM
   degeneration; halt depth steady at 2.0; cycle guard 4/4.
2. **The PPL-vs-generation gap reinforces the stream-overlap caveat** on the
   absolute 5.72 — hold the absolute number loosely; relative comparisons
   (vs v1, vs dense) remain methodologically sound.
3. Behavioural quality at this scale historically arrives with **SFT** (v4's
   registers/halts all came from it) — this read strengthens the case for
   SFT-ing this base as the next pipeline step.

## Where the raw data lives

| Run | Eval JSONs |
|-----|-----------|
| v1 | `archived_models/mythouro_distill_tiny_v1/distill_step_*.json` |
| v2 | `archived_models/mythouro_distill_tiny_sft_v2/sft_step_*.json` |
| moe_s0 | `checkpoints_ablation_moe_s0/distill_step_*.json` (sidecars) |
| flatline | numbers preserved in roadmap failure-mode entry only |

**Convention going forward:** copy each run's eval JSONs into its checkpoint
dir as sidecars immediately after the run (`eval_results/` filenames collide
across runs — that footgun produced a false alarm on 2026-06-10; a per-run
eval-output path is on the action-plan list).
