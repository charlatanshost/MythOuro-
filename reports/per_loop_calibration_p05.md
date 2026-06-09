# P0.5 — Per-loop UncertaintyHead calibration audit (2026-06-09)

The review's question: the head is trained against logits from the training
forward (the `h_K` emission path), but `forward_trajectory` /
`BestOfTrajectoryGenerator` consume it on **per-loop** states it was never
calibrated on — and MoDr's best-exit teacher labels would inherit whatever
miscalibration exists there. The headline ECE (~0.04) only certifies the
final-loop path. Measured per loop with `tools/per_loop_calibration.py`
(`forward_trajectory(force_full_depth=True)`, FineWeb held-out, 20×256-token
chunks, CPU).

## Results

**v2 (278M, `per_loop_ece_v2.json`):**

| loop | ECE | argmax acc | mean unc (head) | actual err | gap |
|-----:|------:|------:|------:|------:|------:|
| 0 | **0.2172** | 0.140 | 0.643 | 0.860 | **0.217** |
| 1 | 0.0162 | 0.500 | 0.485 | 0.500 | 0.015 |
| 2 | 0.0422 | 0.510 | 0.448 | 0.490 | 0.042 |
| 3 | 0.0212 | 0.499 | 0.479 | 0.501 | 0.021 |

**v4 (420M, `per_loop_ece_v4.json`):**

| loop | ECE | argmax acc | mean unc (head) | actual err | gap |
|-----:|------:|------:|------:|------:|------:|
| 0 | **0.1713** | 0.166 | 0.663 | 0.834 | **0.171** |
| 1 | 0.0129 | 0.546 | 0.457 | 0.454 | 0.003 |
| 2 | 0.0405 | 0.547 | 0.414 | 0.453 | 0.039 |
| 3 | 0.0106 | 0.537 | 0.454 | 0.463 | 0.009 |

## Findings

1. **Loop 0 is badly miscalibrated on both checkpoints** (ECE 0.17–0.22), and
   in the dangerous direction: the head **understates** error by ~0.2 (says
   ~0.65 when truth is ~0.85). Loops 1–3 are well-calibrated (ECE 0.01–0.04 —
   consistent with the headline number).
2. **Clean mechanistic explanation:** the SFT `LoopCurriculum` runs
   `start_loops=2 → 4`, so **loop index 0 was never an emission loop during
   training** — the head was never calibrated against loop-0 states. Loop
   indices 1–3 each served as the final loop at some curriculum stage → all
   calibrated. The miscalibration boundary sits exactly where the curriculum
   says it should.
3. **Retro-correction to the best-of-trajectory findings:** earlier inspector
   runs showed v4 "preferring loop 0" on some prompts (e.g. fibonacci,
   `[0.237, 0.64, 0.61]`). Loop 0's confidence is **inflated** (error
   understated), so those shallow-preference reads were partly a calibration
   artifact, not genuine depth preference. The interior-dip findings at loops
   1–2 stand (those loops are calibrated).

## Actions taken

- `BestOfTrajectoryGenerator` / `best_of_trajectory_generate` **default
  `min_loops` 1 → 2** — loop 0 excluded from the argmin by default, on
  measured evidence (callers can still opt back in).
- **MoDr supervision target: per-loop CE, not uncertainty-argmin** — upgraded
  in the roadmap from "the safer option" to the evidence-mandated default. The
  uncertainty-argmin teacher would systematically over-select loop 0.
- If uncertainty-based selection is ever wanted across *all* loops, the fix is
  a per-loop calibration term during training (BCE against per-loop argmax
  error at every loop, not just the emission loop) — or simply starting the
  curriculum at 1.

Reproduce: `python -m tools.per_loop_calibration -c <ckpt> --max-samples 20`.
