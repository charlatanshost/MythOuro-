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
| moe_s1 / dense_s1 | — | — | seed-1 repeats | — | — | — | — | queued |

¹ SFT specialises toward chat; web-text PPL rises by design.
² v3–v5 predate per-run eval archiving — only inspector/behavioural results in
their MODEL_CARDs (reconstructed 2026-06-08).

## PPL trajectory (per 1k-step eval)

| Step | v1 (old eval) | v2-SFT (old eval) | **moe_s0 (fixed code)** | **dense_s0 (fixed code)** |
|-----:|------:|------:|------:|------:|
| 1000 | 368.4 | 48.5 | 559.9 | 578.0 |
| 2000 | 178.6 | 46.5 | 112.1 | 150.3 |
| 3000 | 81.7 | 46.3 | **11.1** | 33.6 |
| 4000 | 51.8 | — | **5.72** | 22.7 |
| 5000 | 37.4 | — | — | — |

**Seed-0 ablation readout:** MoE/dense PPL ratio grows 1.0× → 1.3× → 3.0× →
**4.0×** across training — the sparse capacity (98M idle params at matched
active compute) is increasingly *used* as training matures. Both arms converge
to loop_eff exactly 0.500. Pre-registered rule (keep MoE if >5–10% better):
**MoE retained, pending seed-1 confirmation.** Dense sidecars:
`checkpoints_ablation_dense_s0/`.

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
