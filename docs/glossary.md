# Glossary

> Project terminology. Split out of `roadmap.md` (2026-06-27 doc reorg).


<!-- ===== moved from docs/roadmap.md (2026-06-27 doc reorg) ===== -->

## Glossary

Project-specific terms used throughout the code and docs:

| Term | Meaning |
|------|---------|
| **RDT** | Recurrent-Depth Transformer — the core architecture (Prelude → looped Recurrent block → Coda) |
| **ACT** | Adaptive Computation Time — per-token learned halting; decides how many loops each token needs |
| **halt distribution** | Per-loop probability mass of where tokens stop looping; should be *spread*, not pinned to loop 1 (pinned = collapse) |
| **loop_efficiency** | avg_halt_depth / max_loops — how much of the available depth the model actually uses (~0.5 = genuinely adaptive) |
| **LTI / ρ(A)** | Linear Time-Invariant injection; ρ(A) is its spectral radius. Must stay < 1 or the recurrence diverges. |
| **MoE** | Mixture of Experts — routed + shared FFN experts; only top-k routed experts fire per token |
| **cv / max% / min%** | MoE utilisation stats: coefficient of variation across experts, and the most/least-used expert's share. Low cv = balanced. |
| **router_bias** | DeepSeek-V3 aux-loss-free load-balancing bias on router logits; nudged outside the optimizer toward uniform expert use |
| **sentinel-bias** | Large negative router_bias on newly-grown experts at promotion, decayed over N steps — keeps MoE expansion function-preserving |
| **depth-reg** | PonderNet × Ouro KL-to-uniform regulariser on the halt distribution; prevents ACT loop-collapse |
| **resp_frac** | Fraction of tokens in an SFT batch that contribute to the loss (response tokens, not prompt/padding) |
| **soft / hard loss** | Distillation: `soft` = distance to teacher logits, `hard` = CE against gold tokens |
| **ECE** | Expected Calibration Error — how well predicted confidence matches actual correctness (lower = better calibrated) |
| **UncertaintyHead** | Per-token uncertainty predictor; the `ConfidenceAwareGenerator` uses it to halt/refuse low-confidence generation |
| **h_K (emission)** | The final loop's hidden state — what the model emits at BOTH training and inference since P0.3. The old eval path emitted the ACT-weighted blend `h_out`, a never-trained mixture (removed). |
| **gnorm** | Gradient norm before clipping (clip caps actual updates at 1.0). Healthy: single digits. Hundreds–thousands = chaotic landscape; the step still has norm 1 but points in noise directions → learning stalls (the flatline-incident signature). |
| **LoopCurriculum / `--start-loops` / `--random-depth`** | Training depth schedule: hold at `start_loops`, ramp linearly to `max_loop_iters` by mid-run. `--random-depth` samples each step's depth uniformly in [start, current cap]. Caveat: depths below `start_loops` are never emission depths → uncertainty head uncalibrated there (the loop-0 finding, P0.5). |
| **warmup (LR)** | Steps before full learning rate. The proven from-scratch recipe needs 500 — 200 destabilised a fresh 4-loop recurrent model (see failure modes 2026-06-10). Distinct from the *curriculum* warmup and the *new-component* warmup. |
| **best-of-trajectory** | Inference experiment: score every loop's output with the UncertaintyHead, emit the most-confident depth per token. Diagnostic-grade (no KV cache); `min_loops=2` default excludes the miscalibrated loop 0. |
| **forced-depth probe (`--force-full-depth`)** | Measurement override suppressing ACT's early exits so all `n_loops` run — exposes the counterfactual loops ACT skips ("would deeper have helped?"). Pure measurement, weights untouched. |
| **per-loop calibration / best-exit teacher** | ECE measured per loop index (`tools/per_loop_calibration.py`). The "best exit" per token (lowest per-loop CE) is MoDr's supervision target — per-loop CE, NOT uncertainty-argmin (loop-0 miscalibration would poison it). |
| **MoDr** | Mixture-of-Depth routing (candidate direction): one learned router emitting expert choice AND per-token recurrence depth, trained teacher/student (forced-depth trajectory = teacher; cheap router = student). Gated behind the MoE-vs-dense ablation. |
| **matched-active (ablation)** | The dense arm's FFN width is sized so its params/FLOPs per token equal the MoE arm's *activated* FFN per token (`expert_dim·topk·(1+n_shared)`) — same compute, different total capacity. |
| **sidecar evals** | Per-run eval JSONs copied into the run's checkpoint dir immediately after a run — because `eval_results/` filenames collide across runs (caused a false alarm 2026-06-10). |
| **proven recipe** | v1's final successful hyperparameters (warmup 500, depth-reg 0.3, mb1/ga8), recorded in its MODEL_CARD provenance and now the `distill.py` defaults. Rule: diff against the model card's command, never trust old defaults. |
| **P0 / P1 / P2** | Review priorities: P0 = correctness (fix before training), P1 = perf/measurement validity, P2 = strategic. Tracker: `docs/review_action_plan.md`. |
| **XPU / `rope_real`** | `torch.xpu` = Intel GPU backend (native ≥ PyTorch 2.5); `mythouro/device.py` abstracts cuda/xpu/cpu. `rope_real` swaps the complex RoPE table for an equivalent cos/sin form for backends without complex-op support. |
| **tok/s (k-suffix!)** | The training log prints `2.5k tok/s` = 2,500. Misreading the k once spawned a multi-day "training is slow" investigation (it wasn't). |
| **exposure bias** | The diagnosed cause of generation collapse: the model, trained only on gold text, never learns to recover from its *own* outputs, so free-running it spirals into a repetition attractor. Decoupled from PPL/ECE — cured by on-policy training, not by any offline objective. |
| **on-policy / GKD** | The cure: the student trains on *its own* generated rollouts under teacher scoring (reverse-KL), directly attacking exposure bias. `--onpolicy-lambda` = fraction of steps on rollouts. Broke collapse domain-wide 2026-06-28. |
| **α / teacher-mix (`--teacher-mix-alpha`)** | MiniLLM teacher-mixed sampling during rollouts: `α·teacher + (1-α)·student`. α=0.0 in a probe = the student's *own* trajectory (the exposure-bias scoreboard); high α = teacher-guided (a capability check). |
| **the plateau** | The finding (30k referendum, n=5, 2026-07) that more *web* tokens stopped improving α=0.0 generation at 278M — a **data-quality** wall, not quantity. First breached by teacher-generated data. |
| **teacher corpus** | Text the teacher (Ouro) *writes* from real-corpus seeds, banked for training (sequence-level KD). `tools/gen_teacher_corpus`, mixed in via `--teacher-data-ratio`. The current data-quality lever. |
| **top_share / distinct1** | Generation-probe metrics: `top_share` = most-frequent token's share (high = repetitive); `distinct1` = unique/total (low = repetitive). ⚠ `top_share` *inverts* at the salad→fluency transition (fluent English repeats "the/of/a") — always read the text too. |
| **`exit_at_step` / UT loops** | Ouro's Universal-Transformer early-exit knob: run N of the 4 loops. `exit_at_step=1` = cheap shallow draft (basis for the queued self-speculative decode), `None` = full 4-loop. |
| **plateau floor** | The α=0.0 `top_share` level of the best pre-teacher-data checkpoints (~0.16, the 8668/13944 regime); the number an intervention must push *below* to count as new frontier progress. |

---

