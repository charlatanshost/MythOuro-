#!/usr/bin/env python3
"""
MythOuro knowledge-distillation training script.

Distils a frozen teacher (e.g. Ouro-1.4B) into an MythOuro student in
the same per-step loop the main pretraining script uses. Reuses the
existing data pipeline (`MixedDataset`), curriculum / annealer, MoE
helpers, and checkpoint infrastructure — only the loss differs.

The student is trained with:

    L = α · distillation(student, teacher, T)   + (1-α) · CE(student, gold)
      + λ_lb · load_balance                                     # MoE health
      + λ_unc · uncertainty_calibration                         # head training
      + λ_sparse · sparse_activation                            # decisive routing

This blends Hinton-style soft-label distillation with the architecture-
specific auxiliary losses MythOuro needs to keep its non-standard
components (MoE router, UncertaintyHead) trained alongside the LM head.

Preconditions
-------------
* Teacher and student MUST share a tokenizer (logit distillation across
  different vocabularies is meaningless). `load_distillation_teacher`
  refuses to return a teacher otherwise.
* Teacher fits in RAM/VRAM alongside the student. For a 12GB Blackwell
  setup distilling a 1.4B teacher into a 1B student, expect to need
  bf16 + grad checkpointing + CPU-offloaded optimizer state, or run
  the teacher on a separate device.

CLI
---
    python training/distill.py \\
        --teacher-id ouro-llm/Ouro-1.4B \\
        --teacher-device cpu \\
        --student-variant mythouro_1b \\
        --total-steps 5000 \\
        --alpha 0.5 \\
        --temperature 2.0 \\
        --eval --eval-every 500

The student variant must already use the teacher's tokenizer. If it
doesn't, switch the student's `vocab_size` (and ideally re-init the
embedding + LM head) before running distillation.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from mythouro import MythOuro
from mythouro.checkpointing import (
    ShutdownHandler,
    list_ckpts,
    load_checkpoint,
    save_checkpoint,
)
from mythouro.tokenizer import MythOuroTokenizer
# Sentinel decay for GROWN checkpoints. tools/grow_checkpoint.py gates new
# experts behind a large negative router_bias; it MUST decay to 0 over
# n_decay_steps or the promoted experts never enter top-k and the expansion
# is inert — a bigger, slower model identical to the source. sft.py has done
# this since growth was built; distill.py never did, so the growth path only
# worked through the channel that collapses this model (found 2026-08-27).
from mythouro.grow import apply_sentinel_to_router_biases
from mythouro.training_utils import (
    _MIX_RATIOS,
    LoopCurriculum,
    MixedDataset,
    ExpertSpecializationProbe,
    ProcessRewardHead,
    apply_component_warmup,
    collect_expert_counts,
    collect_router_logits,
    depth_regularization_loss,
    distillation_loss,
    generate_rollout,
    get_optimizer_groups,
    load_balance_loss,
    load_distillation_teacher,
    log_expert_utilization,
    log_spectral_radius,
    moe_router_bias,
    sparse_activation_loss,
    teacher_logits,
    uncertainty_calibration_loss,
    update_router_bias_from_counts,
)
from mythouro.variants import (
    mythouro_distill_tiny,
    mythouro_distill_tiny_dense,
    mythouro_distill_small,
    mythouro_distill_xl,
    mythouro_1b, mythouro_3b, mythouro_10b, mythouro_50b, mythouro_100b,
    mythouro_500b, mythouro_1t,
)
from mythouro import device as dev
from mythouro.rollout import RolloutBuffer, rollout_with_retry


_VARIANT_FUNCS = {
    # 240M student aligned to Ouro vocab; designed to cohabit with the
    # bf16 teacher on a single 12 GB GPU. Default choice for distillation.
    "mythouro_distill_tiny":  mythouro_distill_tiny,
    # Dense twin of distill_tiny (recurrent MoE -> matched-active dense FFN).
    # The dense arm of the MoE-vs-dense ablation (docs/roadmap.md).
    "mythouro_distill_tiny_dense": mythouro_distill_tiny_dense,
    # Post-MoE-expansion targets (48 / 96 routed experts). Used when resuming
    # a grown checkpoint via `tools/grow_checkpoint.py`.
    "mythouro_distill_small": mythouro_distill_small,
    "mythouro_distill_xl":    mythouro_distill_xl,
    "mythouro_1b":   mythouro_1b,
    "mythouro_3b":   mythouro_3b,
    "mythouro_10b":  mythouro_10b,
    "mythouro_50b":  mythouro_50b,
    "mythouro_100b": mythouro_100b,
    "mythouro_500b": mythouro_500b,
    "mythouro_1t":   mythouro_1t,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distil an MythOuro student from a frozen teacher.",
    )
    p.add_argument("--teacher-id", default="ByteDance/Ouro-2.6B-Thinking",
                   help="HF model id or local path of the teacher. "
                        "Default is Ouro-2.6B-Thinking, the model this "
                        "pipeline is designed around.")
    p.add_argument("--teacher-device", default="cpu",
                   help="Device the teacher runs on. CPU is the safe "
                        "default for mixed-VRAM rigs; switch to cuda if "
                        "your card has room for both teacher and student.")
    p.add_argument("--student-variant", default="mythouro_distill_tiny",
                   choices=list(_VARIANT_FUNCS),
                   help="Default `mythouro_distill_tiny` is the 240M Ouro-"
                        "aligned student sized for a 12 GB GPU.")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking",
                   help="HF tokenizer id. MUST match the teacher's tokenizer "
                        "(load_distillation_teacher refuses to return a "
                        "mismatched teacher).")
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--teacher-data-ratio", type=float, default=0.0,
               help="Fraction of the data stream drawn from the local "
                    "teacher-generated corpus (tools/gen_teacher_corpus). "
                    "0 = off (default, stream unchanged). Start A/Bs at 0.2 "
                    "per docs/teacher_corpus_plan.md.")
    p.add_argument("--teacher-data-files", default="data_teacher/*.jsonl",
               help="Glob of local JSONL shards for --teacher-data-ratio.")
    p.add_argument("--min-lr", type=float, default=None,
               help="Cosine-decay floor. Default None = lr*0.1 (legacy). The "
                    "18k probe (tracker 2026-07-17) showed the legacy floor "
                    "starves late-schedule legs — frontier runs want ~lr*0.3 "
                    "so extended legs keep real signal.")
    p.add_argument("--warmup-steps", type=int, default=500,
               help="v1's proven from-scratch recipe used 500. The old "
                    "default (200) hit full LR too early on a fresh 4-loop "
                    "recurrent model -> transient deep-loop gradient spikes "
                    "-> flatline (see roadmap failure modes, 2026-06-10).")
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Distillation weight: 0=pure CE, 1=pure soft loss.")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Softmax temperature for the distillation term.")
    p.add_argument("--divergence", choices=["fwd_kl", "rev_kl", "jsd"],
                   default="fwd_kl",
                   help="Distillation divergence. fwd_kl = Hinton mode-covering "
                        "(default, = current behaviour). rev_kl = MiniLLM "
                        "mode-seeking (less mass on the teacher's void regions; "
                        "anti-degeneration for a small student / big teacher). "
                        "jsd = interpolation (see --jsd-beta). Tier-1 of the "
                        "on-policy mode-seeking lever (docs/ideas.md).")
    p.add_argument("--jsd-beta", type=float, default=0.5,
                   help="JSD interpolation weight when --divergence jsd "
                        "(β→0 ≈ fwd_kl, β→1 ≈ rev_kl). Try 0.5 or 0.9.")
    p.add_argument("--loop-loss-weighting",
                   choices=["off", "uniform", "progressive", "exit_pdf"],
                   default="off",
                   help="Distil against EVERY recurrent loop instead of the "
                        "final one, combining per-loop losses with weights that "
                        "sum to 1. 'off' (default) = current behaviour, "
                        "bit-identical. This closes our documented divergence "
                        "from Ouro, which trains L = SUM_t p(t|x)*L^(t) while we "
                        "supervise h_K alone (docs/growth_design.md 'loop-loss "
                        "supervision'). uniform = 1/K each. progressive = t^alpha "
                        "normalised (later loops weigh more, see "
                        "--loop-loss-alpha). exit_pdf = the model's own halt "
                        "distribution, i.e. Ouro's p(t|x) exactly. "
                        "CAUTION: exit_pdf interacts with --depth-reg-coeff, "
                        "which pulls the halt distribution TOWARD UNIFORM; at "
                        "0.3 the two weightings largely converge, so 'uniform' "
                        "is the honest control arm, not a strawman. "
                        "MEMORY: K sets of logits stay live for the backward "
                        "(~805MB each at micro-batch 8 / seq 1024 / vocab "
                        "49,152); drop --micro-batch if this OOMs.")
    p.add_argument("--unc-loop-weighting",
                   choices=["off", "match", "uniform", "progressive", "exit_pdf"],
                   default="off",
                   help="Train the UncertaintyHead at EVERY recurrent loop "
                        "instead of the final one. 'off' (default) = current "
                        "behaviour. 'match' reuses --loop-loss-weighting. "
                        "WHY IT IS SEPARATE from --loop-loss-weighting: the two "
                        "fix different heads and are worth measuring apart. The "
                        "head is trained only on final-loop states today, which "
                        "is exactly why per-loop calibration reads ECE 0.013 at "
                        "loop 3 and 0.288 at loop 0 — loop 3 is the only depth "
                        "it has ever seen. Every consumer of shallow-loop "
                        "confidence inherits that: ACT halting, "
                        "BestOfTrajectoryGenerator (which preferred loops 1-2 "
                        "and RAISED the copy rate 41.2%% -> 50.0%%), and "
                        "exit_pdf weighting itself. "
                        "CAUTION: this retrains a head that "
                        "train_depth_policy.py already touched once and made the "
                        "task WORSE (halt 2.00 -> 3.60). Run it with its own "
                        "control arm; do not assume it composes.")
    p.add_argument("--loop-loss-alpha", type=float, default=1.0,
                   help="Exponent for --loop-loss-weighting progressive: "
                        "w_t proportional to (t+1)^alpha, normalised. 0 = "
                        "uniform, 1 = linear ramp, >1 = sharper toward the "
                        "final loop. Ignored by the other weightings.")
    p.add_argument("--lb-coeff", type=float, default=1e-2)
    p.add_argument("--unc-coeff", type=float, default=5e-2)
    p.add_argument("--sparse-coeff", type=float, default=1e-3)
    p.add_argument("--depth-reg-coeff", type=float, default=0.3,
                   help="PonderNet × Ouro KL-to-uniform regulariser on the "
                        "halt distribution; prevents ACT loop-collapse. "
                        "Default 0.3 = v1's proven final recipe (its model-"
                        "card command), not the 1e-1 the earlier help text "
                        "suggested. Pass 0.0 to disable.")
    p.add_argument("--recurrent-state-noise", type=float, default=0.0,
                   help="Training-time Gaussian noise on the recurrent hidden "
                        "state each loop, scaled to σ·RMS(h). Anti-collapse "
                        "regulariser that replaces the accidental P0.1 noise "
                        "which kept free generation from collapsing to a "
                        "fixed point. 0.0 = off. Try 0.02–0.1.")
    # ── On-policy / GKD (the exposure-bias cure; OFF by default) ──
    # See docs/onpolicy_plan.md. λ=0 keeps the current pure-offline behaviour.
    p.add_argument("--onpolicy-lambda", type=float, default=0.0,
                   help="Fraction of steps trained on STUDENT-GENERATED rollouts "
                        "instead of the corpus (GKD/MiniLLM). 0=pure offline "
                        "(current), 1=pure on-policy. The exposure-bias cure: the "
                        "student learns to recover from its OWN trajectories.")
    p.add_argument("--teacher-mix-alpha", type=float, default=0.25,
                   help="Teacher-mixed rollout sampling: draw from "
                        "α·teacher + (1-α)·student. Drags a collapse-prone "
                        "student's rollouts back toward sense (the un-collapse "
                        "lever). 0=pure student sampling. Used only when "
                        "--onpolicy-lambda > 0.")
    p.add_argument("--rollout-len", type=int, default=96,
                   help="Tokens generated per on-policy rollout. Keep SHORT — "
                        "recurrent decode is slow. Used only when "
                        "--onpolicy-lambda > 0.")
    p.add_argument("--onpolicy-temp", type=float, default=1.0,
                   help="Sampling temperature for on-policy rollouts.")
    p.add_argument("--onpolicy-top-k", type=int, default=50,
                   help="Top-k filter for on-policy rollout sampling (0=off). "
                        "Used only when --onpolicy-lambda > 0.")
    p.add_argument("--rollout-batch", type=int, default=16,
                   help="Sequences generated per WIDE rollout call. Decode "
                        "cost is amortised across this batch (latency-bound "
                        "accelerators only win wide), then served to the "
                        "training loop in micro-batch slices via the reuse "
                        "buffer. Scale up against accelerator memory.")
    p.add_argument("--rollout-reuse", type=int, default=2,
                   help="Times each buffered rollout batch is consumed before "
                        "a forced refill. >1 is deliberate mild off-policyness "
                        "(GKD tolerates it) that multiplies on-policy dose per "
                        "decode-second.")
    p.add_argument("--rollout-max-age-steps", type=int, default=50,
                   help="Force a rollout-buffer refill after this many "
                        "optimizer steps regardless of remaining reuse "
                        "(staleness cap).")
    p.add_argument("--rollout-legacy", action="store_true",
                   help="Escape hatch: inline per-micro-step rollout "
                        "generation with full O(L^2) recompute (the "
                        "pre-buffer, pre-KV-cache behaviour).")
    p.add_argument("--compile", choices=("off", "teacher", "student", "both"),
                   default="off",
                   help="torch.compile the TRAINING forwards (automatic kernel "
                        "fusion + launch reduction). Field notes measured +10%% "
                        "end-to-end on XPU with zero graph breaks, and it has "
                        "never been wired into this trainer. Needs "
                        "TRITON_DEFAULT_BACKEND=intel and intel-ocloc. DEFAULT "
                        "MODE ONLY -- max-autotune replaces oneDNN's XMX GEMMs "
                        "with Triton templates and LOSES on PVC (14.9k vs 17.2k "
                        "tok/s, measured). The GENERATION path is deliberately "
                        "left in eager: rollout grows the sequence every token, "
                        "so a compiled graph would recompile per length and can "
                        "easily cost more than the fusion saves.")
    p.add_argument("--profile-steps", type=int, default=0,
                   help="DIAGNOSTIC, NOT TRAINING. Profile this many steps after "
                        "--profile-warmup, print a per-component wall-clock "
                        "breakdown, then EXIT WITHOUT SAVING. Answers 'is the "
                        "teacher forward the bottleneck?' — which decides whether "
                        "teacher-logit caching (and with it, λ as a real "
                        "throughput lever) is worth building. Syncs the device "
                        "around each region, so read the SHARES, not the tok/s.")
    p.add_argument("--profile-warmup", type=int, default=5,
                   help="Steps to run before profiling starts. Must be >0: the "
                        "first steps carry torch.compile, allocator growth and "
                        "SYCL kernel-cache misses that would swamp the real "
                        "shares.")
    p.add_argument("--use-sandwich-norm", action="store_true",
                   help="Huginn sandwich norm (extra post-sublayer RMSNorm in "
                        "every TransformerBlock) — recurrent hidden-state-collapse "
                        "stabiliser, 'required at scale'. Changes architecture → "
                        "FRESH runs only (carried in cfg_dict).")
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="DIAGNOSTIC/WORKAROUND. cfg.gradient_checkpointing=False. "
                        "The 2026-08-28 48-expert segfault stack runs through "
                        "torch.utils.checkpoint at main.py:1794; this removes it. "
                        "Costs activation memory (loop becomes O(n_loops), not "
                        "O(1)) so pair with a smaller --micro-batch.")
    p.add_argument("--use-depth-aware-init", action="store_true",
                   help="Huginn/Takase depth-aware init: residual-output projs get "
                        "std^2=1/(5*h*l). FRESH runs only (no effect on resumed "
                        "weights).")
    p.add_argument("--ckpt-dir", default="checkpoints_distill")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--keep-last", type=int, default=5,
                   help="Most-recent checkpoints to retain (fine-grained). "
                        "Milestones below are kept ON TOP of these.")
    p.add_argument("--ckpt-milestone-every", type=int, default=2000,
                   help="Permanently keep every Nth-step checkpoint — never "
                        "pruned. Preserves the whole trajectory so a mid-leg "
                        "peak can't be silently rotated away (this lost step "
                        "8668 and step 40002). 0 = disable (old keep_last-only "
                        "behaviour).")
    p.add_argument("--ckpt-every-mins", type=float, default=0.0,
                   help="Also checkpoint every N minutes of wall-clock, "
                        "regardless of step count (0=off). Robustness net for "
                        "SLOW runs (on-policy: a single step can take minutes, "
                        "so step-based --ckpt-every may never fire before a "
                        "power cut). Composes with keep_last pruning + the "
                        "Ctrl-C shutdown flush.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader worker subprocesses. **Use 0 for distill** — it's "
                        "teacher-bound (workers buy ~nothing) and 0 makes Ctrl+C a clean "
                        "KeyboardInterrupt (graceful save fires) AND removes worker-death "
                        "crashes. >0 on Windows + streaming data is crash-prone.")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="REQUIRED for the default Ouro teacher (it ships a "
                        "custom modeling_ouro.py). Set whenever the teacher "
                        "repo includes custom modeling code.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds torch / python RNG (model init, depth sampling, "
                        "dropout). Required for the >=2-seed ablation protocol. "
                        "Note: HF streaming data order is not fully seeded.")
    p.add_argument("--start-loops", type=int, default=2,
                   help="LoopCurriculum starting depth. NOTE (P0.5 audit): with "
                        "the default 2, loop index 0 is never an emission loop, "
                        "so the UncertaintyHead ends up badly miscalibrated at "
                        "loop 0 (ECE ~0.2 on v2/v4). Use 1 if you want the head "
                        "calibrated across ALL loops (e.g. for MoDr labels).")
    p.add_argument("--random-depth", action="store_true",
                   help="Per batch, sample unroll depth uniformly in "
                        "[start_loops, curriculum.get(step)] instead of "
                        "using the curriculum value directly. Forces the "
                        "model to be robust across the depth range it has "
                        "been ramped to so far.")
    p.add_argument("--eval", "-e", action="store_true",
                   help="Run the eval harness every `--eval-every` steps. "
                        "Writes a JSON report per eval into `eval_results/`.")
    p.add_argument("--eval-every", type=int, default=500,
                   help="Step cadence for in-loop eval (default: 500).")
    p.add_argument("--eval-max-samples", type=int, default=50,
                   help="Per-benchmark sample cap during in-loop eval.")
    p.add_argument("--eval-benchmarks", nargs="+", default=["all"],
                   help="Benchmarks to run. Default: all. Names: perplexity, "
                        "arc_challenge, gsm8k, loop_efficiency, ece.")
    p.add_argument("--student-device", default=None,
                   help="Device for the student (and AdamW state / aux heads). "
                        "Default: cuda:0 if available else cpu. Pass cuda:1 / "
                        "cuda:2 etc. to put the student on a different GPU "
                        "than the teacher — useful on a multi-card rig where "
                        "the teacher needs the bigger card. The teacher logits "
                        "are transferred to the student device each step.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Step profiler — where does the time actually go?
# ---------------------------------------------------------------------------


class _StepProfiler:
    """Per-component wall-clock breakdown of one training step.

    WHY THIS EXISTS. The λ sweep (2026-07-30) cut on-policy steps 0.7 -> 0.4 and
    moved throughput ~4%, against a predicted 45%. The explanation is visible in
    the loop but had never been measured: `teacher_logits(...)` runs on EVERY
    micro-step, outside the on-policy/offline branch, and the teacher is
    Ouro-2.6B against a 278M student on the SAME card. If the teacher forward
    dominates, then λ *cannot* be a throughput lever and neither can
    --rollout-reuse, because both only touch rollout generation.

    That reasoning has been wrong twice. Hence: measure, don't deduce. The number
    that matters is the teacher share — it decides whether caching top-K teacher
    logits for the offline path (which would also RESURRECT λ as a lever, since
    offline steps would stop paying teacher cost) is worth building.

    ACCURACY NOTE: XPU/CUDA kernel launches are ASYNC, so timing without a
    synchronize measures enqueue time, not execution — every region would look
    instant and the totals would be nonsense. We sync on entry and exit of each
    region. That perturbs absolute throughput slightly (real pipelining is
    suppressed), so read the SHARES, not the tok/s, from a profiled run.
    """

    def __init__(self, device: str, enabled: bool):
        self.enabled = enabled
        self.device = device
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._open: dict[str, float] = {}
        self.steps = 0

    def _sync(self) -> None:
        if self.device.startswith("xpu") and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @contextmanager
    def region(self, name: str):
        if not self.enabled:
            yield
            return
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    # Explicit start/stop for regions whose body is too long to re-indent under a
    # `with` without producing a large, review-hostile diff (the rollout block).
    # A dropped `stop` on an exception is acceptable here: this is a diagnostic
    # mode that exits without saving, and an exception aborts the run anyway.
    def start(self, name: str) -> None:
        if not self.enabled:
            return
        self._sync()
        self._open[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if not self.enabled or name not in self._open:
            return
        self._sync()
        dt = time.perf_counter() - self._open.pop(name)
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.steps = 0

    def report(self) -> None:
        if not self.enabled or not self.totals:
            return
        total = sum(self.totals.values())
        logger.info("=" * 68)
        logger.info(f"STEP PROFILE over {self.steps} steps "
                    f"({total / max(self.steps, 1) * 1e3:.1f} ms/step measured)")
        logger.info(f"  {'region':<18} {'ms/step':>9} {'share':>7}  {'calls/step':>10}")
        for name, t in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            logger.info(f"  {name:<18} {t / self.steps * 1e3:>9.1f} "
                        f"{100 * t / total:>6.1f}% {self.counts[name] / self.steps:>10.1f}")
        logger.info("=" * 68)
        teacher = self.totals.get("teacher_fwd", 0.0)
        share = 100 * teacher / total if total else 0.0
        logger.info(f"TEACHER SHARE: {share:.1f}%")
        if share >= 45:
            logger.info("  => Teacher forward DOMINATES. Caching top-K teacher logits for "
                        "the offline path is the lever; it also makes λ a real lever again.")
        elif share >= 25:
            logger.info("  => Teacher forward is significant but not dominant. Caching helps "
                        "the offline fraction only — weigh against the build cost.")
        else:
            logger.info("  => Teacher forward is NOT the bottleneck. Do not build the logit "
                        "cache; the top region above is where the time is.")
        logger.info("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _loop_weights(
    mode: str,
    K: int,
    halt: "torch.Tensor | None",
    *,
    alpha: float,
    shape: "tuple[int, int]",
    device: "torch.device",
) -> torch.Tensor:
    """
    Per-token weights over the K recurrent loops, for loop-weighted distillation.

    Returns (B, T, K) summing to 1 along K at every position, so the K weighted
    per-loop losses add up to one properly-scaled loss.

    Modes:
      uniform      1/K everywhere. The CONTROL ARM, and not a strawman — see
                   the exit_pdf note below.
      progressive  w_t ∝ (t+1)^alpha, normalised; later loops weigh more.
      exit_pdf     the model's own halt distribution p(t|x) — Ouro's objective
                   exactly, and the only mode whose weights vary PER TOKEN,
                   which is the whole point of adaptive depth.

    ⚠ exit_pdf vs --depth-reg-coeff. `depth_regularization_loss` is a KL from
    the halt distribution to UNIFORM. Run both at strength (the pour uses 0.3)
    and the halt distribution is actively pulled toward 1/K, so exit_pdf
    converges toward uniform. That is not a bug in either piece — Ouro pairs
    per-step weighting WITH entropy regularisation on purpose, because the
    weighting alone hands the optimiser a lever to dump all mass on the easiest
    loop (our own λ₀→1 ACT collapse, docs/growth_design.md). The two are a
    matched pair; tune them together, and read `uniform` as the null against
    which exit_pdf must actually earn its keep.

    Falls back to uniform when the halt distribution is unavailable, loudly:
    silently degrading exit_pdf to uniform would make the A/B meaningless.
    """
    B, T = shape
    if mode == "exit_pdf":
        if halt is None:
            raise RuntimeError(
                "--loop-loss-weighting exit_pdf needs the halt distribution, "
                "but the recurrent block published none (or its shape drifted "
                "from the states). Use 'uniform' or 'progressive', or fix "
                "RecurrentBlock.last_halt_distribution — do not let this "
                "silently fall back, or the A/B measures nothing."
            )
        return halt
    if mode == "uniform":
        w = torch.full((K,), 1.0 / K, device=device)
    elif mode == "progressive":
        t = torch.arange(1, K + 1, device=device, dtype=torch.float32) ** float(alpha)
        w = t / t.sum()
    else:
        raise ValueError(f"unknown --loop-loss-weighting {mode!r}")
    return w.view(1, 1, K).expand(B, T, K)


def main():
    args = _parse_args()

    # Student device — explicit if passed, else cuda:0 (legacy default) or cpu.
    # The teacher device is handled independently via --teacher-device, so a
    # multi-card layout looks like:
    #     --teacher-device cuda:0  --student-device cuda:1
    # We validate the requested device exists before any allocation to fail
    # loudly rather than waste 5 minutes building the model on a phantom GPU.
    if args.student_device is None:
        device = dev.pick_device(None)        # cuda:0 > xpu > cpu
    else:
        device = args.student_device
        if dev.is_accelerator(device):
            if not dev.is_available(device):
                raise RuntimeError(
                    f"--student-device={device!r} but "
                    f"{dev.backend(device)} is unavailable"
                )
            # `cuda`/`xpu` alone means index 0; `cuda:N` / `xpu:N` selects N.
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
            n_devices = dev.device_count(device)
            if idx >= n_devices:
                raise RuntimeError(
                    f"--student-device={device!r} but only {n_devices} "
                    f"{dev.backend(device)} device(s) visible."
                )

    amp_dtype = torch.bfloat16 if dev.bf16_supported(device) else torch.float16

    # Seed BEFORE model construction so init is reproducible per --seed.
    import random as _random_seed
    torch.manual_seed(args.seed)
    _random_seed.seed(args.seed)

    # ------------------------------------------------------------------
    # Tokenizer + student
    # ------------------------------------------------------------------
    encoding = MythOuroTokenizer(args.tokenizer)
    vocab_size = encoding.vocab_size

    cfg = _VARIANT_FUNCS[args.student_variant]()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = args.seq_len
    cfg.recurrent_state_noise = args.recurrent_state_noise
    cfg.use_sandwich_norm = args.use_sandwich_norm
    cfg.use_depth_aware_init = args.use_depth_aware_init
    if args.no_gradient_checkpointing:
        cfg.gradient_checkpointing = False
        logger.info('distill: gradient checkpointing DISABLED '
                    '(activation memory now O(n_loops))')

    # XPU: complex-tensor RoPE (view_as_complex / polar) can segfault on
    # Intel's kernels. The real-valued (cos/sin) path is mathematically
    # identical and safe on all backends — auto-enable when on XPU.
    if dev.backend(device) == "xpu":
        cfg.rope_real = True
        logger.info("distill: XPU detected → rope_real=True (complex ops unsupported)")

    student = MythOuro(cfg).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    logger.info(
        f"distill: student={args.student_variant} params={n_params:,} "
        f"vocab={vocab_size} device={device} amp={amp_dtype}"
    )

    # ------------------------------------------------------------------
    # Teacher — frozen, no grad. Tokenizer alignment is enforced inside.
    # ------------------------------------------------------------------
    teacher = load_distillation_teacher(
        args.teacher_id,
        student_vocab_size=vocab_size,
        device=args.teacher_device,
        dtype=amp_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if teacher is None:
        logger.error(
            "distill: teacher could not be loaded. Aborting — there's no "
            "point running this script without a teacher."
        )
        return

    # ------------------------------------------------------------------
    # Aux heads (mirrors the pretraining script — keeps the
    # UncertaintyHead, ProcessRewardHead, ExpertSpecializationProbe
    # trained even when CE is partially displaced by distillation).
    # ------------------------------------------------------------------
    prm_head  = ProcessRewardHead(cfg.dim).to(device)
    esp_probe = ExpertSpecializationProbe(cfg.n_experts).to(device)

    optimizer = torch.optim.AdamW(
        get_optimizer_groups(
            student,
            base_lr=args.lr,
            weight_decay=args.weight_decay,
            extra_base_params=list(prm_head.parameters())
                             + list(esp_probe.parameters()),
        ),
        betas=(0.9, 0.95),
        fused=dev.fused_adam_supported(device),
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    existing = list_ckpts(args.ckpt_dir)
    resume_extra = None
    if existing:
        logger.info(f"distill: resuming from {existing[-1]}")
        start_step, resume_extra = load_checkpoint(
            student, optimizer, existing[-1], ddp=False, current_cfg=cfg,
        )

    # A checkpoint from tools/grow_checkpoint.py carries growth_metadata in its
    # `extra`. New experts start behind a sentinel bias that must follow the
    # decay SCHEDULE, not the data-driven updater, during the warm-in window.
    growth_metadata = (resume_extra or {}).get("growth_metadata")

    # ⚠ growth_metadata MUST be re-saved into every checkpoint this run writes.
    # It arrives only in the promoted file's `extra`; if a save drops it, the
    # next resume sees no metadata, apply_sentinel_to_router_biases() is never
    # called, and the new experts' bias freezes at its partial-decay value
    # (~-60 mid-window). The DeepSeek updater at bias_lr=1e-3 needs ~60k steps
    # to walk that back, so the new experts never enter top-k and the run
    # silently produces a 460M model that routes exactly like its 278M source.
    # This bites on ANY crash-restart inside the 500-step decay window, and on
    # the nightly resume after the Windows reboot. (Found 2026-08-28; the
    # 123e460 sentinel patch wired the decay in on load but not on save.)
    ckpt_extra = (
        {"growth_metadata": growth_metadata} if growth_metadata is not None else None
    )

    if growth_metadata is not None:
        logger.info(
            "distill: GROWN checkpoint — "
            f"{growth_metadata.get('source_n_experts')} -> "
            f"{growth_metadata.get('target_n_experts')} experts, "
            f"sentinel={growth_metadata.get('sentinel_bias')}, "
            f"decay over {growth_metadata.get('n_decay_steps')} steps"
        )

    # ------------------------------------------------------------------
    # Data + curriculum
    # ------------------------------------------------------------------
    # Teacher-generated corpus mix-in (docs/teacher_corpus_plan.md): ratio R
    # of the stream comes from local JSONL written by tools/gen_teacher_corpus,
    # the real corpora scaled by (1-R). R=0 (default) = exactly the old stream.
    mix_ratios = None
    extra_specs = None
    if args.teacher_data_ratio > 0.0:
        scale = 1.0 - args.teacher_data_ratio
        mix_ratios = {k: v * scale for k, v in _MIX_RATIOS.items()}
        mix_ratios["teacher"] = args.teacher_data_ratio
        extra_specs = [
            ("teacher", f"json:{args.teacher_data_files}", None, "train", "text"),
        ]
    dataset = MixedDataset(
        encoding, args.seq_len, rank=0, world_size=1,
        mix_ratios=mix_ratios, extra_specs=extra_specs,
    )
    loader = DataLoader(
        dataset, batch_size=args.micro_batch, num_workers=args.num_workers, pin_memory=True,
    )

    curriculum = LoopCurriculum(
        start_loops=args.start_loops,
        max_loops=cfg.max_loop_iters,
        warmup_steps=max(args.warmup_steps * 2, args.total_steps // 20),
        total_steps=args.total_steps // 2,
    )

    # `--random-depth` switches per-step depth selection from
    # `curriculum.get(step)` (fixed) to `curriculum.get_sampled(step, rng)`
    # (random uniform in [start, get(step)]). Seeded for reproducibility.
    import random as _random
    depth_rng = _random.Random(args.seed)
    # Independent stream for the per-micro-step on-policy coin flip, so toggling
    # --random-depth never shifts which steps go on-policy (and vice-versa).
    onpolicy_rng = _random.Random(args.seed + 9973)

    amp_ctx = (
        torch.amp.autocast(device_type=dev.autocast_type(device), dtype=amp_dtype)
        if dev.is_accelerator(device) else nullcontext()
    )

    shutdown = ShutdownHandler()
    shutdown.install()

    # Rollout reuse buffer (docs/onpolicy_plan.md phase 5): decouple the wide
    # GENERATION batch from the training micro-batch. --rollout-legacy keeps
    # the old inline per-micro-step path.
    rollout_buffer = (
        None
        if (args.rollout_legacy or args.onpolicy_lambda <= 0.0)
        else RolloutBuffer(
            args.rollout_batch, args.micro_batch,
            reuse=args.rollout_reuse,
            max_age_steps=args.rollout_max_age_steps,
        )
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    student.train()
    data_iter = iter(loader)
    step = start_step
    t0 = time.perf_counter()
    last_ckpt_time = time.perf_counter()
    log_every = args.log_every

    # ── torch.compile aliases (training forwards only) ──
    # The raw `student` / `teacher` handles stay in use for GENERATION and for
    # checkpointing; only the fixed-shape training forwards go through the
    # compiled wrappers. That split matters twice:
    #   * generation grows the sequence each token -> a compiled graph would
    #     recompile per length, and the rollout is 31% of the step;
    #   * save_checkpoint(student, ...) must see the ORIGINAL module, or every
    #     key acquires an `_orig_mod.` prefix and the checkpoint stops loading.
    student_fwd, teacher_fwd = student, teacher
    if args.compile != "off":
        if args.compile in ("student", "both"):
            student_fwd = torch.compile(student)          # default mode ONLY
        if args.compile in ("teacher", "both") and teacher is not None:
            teacher_fwd = torch.compile(teacher)
        logger.info(
            f"torch.compile={args.compile} (default mode); generation and "
            f"checkpointing still use the uncompiled modules. First steps will "
            f"be slow while Inductor builds."
        )

    prof = _StepProfiler(str(device), enabled=args.profile_steps > 0)
    if prof.enabled:
        logger.info(f"PROFILING MODE: {args.profile_warmup} warmup steps, then "
                    f"{args.profile_steps} profiled steps, then exit WITHOUT saving.")
        profile_stop = step + args.profile_warmup + args.profile_steps
        profile_start = step + args.profile_warmup

    while step < args.total_steps:
        if prof.enabled:
            if step == profile_start:
                prof.reset()            # discard warmup: compile + allocator growth
            if step >= profile_stop:
                prof.report()
                logger.info("profiling complete — exiting without saving a checkpoint")
                return
        cur_lr = _cosine_lr(step, args.warmup_steps, args.total_steps,
                             args.lr,
                             args.min_lr if args.min_lr is not None
                             else args.lr * 0.1)
        warmup_factor = apply_component_warmup(
            optimizer, cur_lr, step, cfg.new_component_warmup_steps,
        )
        n_loops = (
            curriculum.get_sampled(step, depth_rng)
            if args.random_depth
            else curriculum.get(step)
        )

        optimizer.zero_grad()
        loss_accum = soft_accum = hard_accum = 0.0
        lb_accum = unc_accum = sparse_accum = depth_accum = 0.0
        op_accum = 0
        accum_expert_counts: dict = {}

        for micro_step in range(args.grad_accum):
            with prof.region("data"):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            # ── On-policy / GKD: with prob λ, train on a STUDENT-generated
            #    rollout instead of the corpus batch. The student continues a
            #    short real-text seed under teacher-mixed sampling, then we
            #    distil (soft divergence, no hard CE) on its OWN sequence — the
            #    exposure-bias cure (docs/onpolicy_plan.md). λ=0 → never fires.
            is_onpolicy = (
                args.onpolicy_lambda > 0.0
                and onpolicy_rng.random() < args.onpolicy_lambda
            )
            if is_onpolicy:
                prof.start("rollout")     # paired with prof.stop below
                seed_len = max(8, args.rollout_len // 4)
                if rollout_buffer is None:
                    # Legacy escape hatch: inline per-micro-step generation.
                    with amp_ctx:
                        rollout = generate_rollout(
                            student, teacher, x[:, :seed_len],
                            n_loops=n_loops,
                            max_new_tokens=args.rollout_len,
                            teacher_mix_alpha=args.teacher_mix_alpha,
                            temperature=args.onpolicy_temp,
                            top_k=args.onpolicy_top_k,
                            use_kv_cache=not args.rollout_legacy,
                        )
                else:
                    if rollout_buffer.needs_refill(step):
                        # Accumulate rollout_batch seed rows from the corpus
                        # stream (the current micro-batch plus as many more
                        # as needed), then ONE wide generate call.
                        seeds = [x[:, :seed_len]]
                        n_rows = x.shape[0]
                        while n_rows < rollout_buffer.rollout_batch:
                            try:
                                xs, _ = next(data_iter)
                            except StopIteration:
                                data_iter = iter(loader)
                                xs, _ = next(data_iter)
                            xs = xs.to(device, non_blocking=True)
                            seeds.append(xs[:, :seed_len])
                            n_rows += xs.shape[0]
                        seed_batch = torch.cat(seeds, dim=0)
                        seed_batch = seed_batch[: rollout_buffer.rollout_batch]
                        with amp_ctx:
                            # use_kv_cache=False: the cached student decode is NOT
                            # distribution-preserving — ACT early-exit converges
                            # per-forward (single token cached vs whole sequence
                            # uncached), shifting logits by up to ~1 nat KL on real
                            # checkpoints (probe tracker 2026-07-16). Wide batching
                            # keeps the phase-5 throughput win; do not re-enable the
                            # cache without a KL equivalence gate like the teacher's.
                            wide = rollout_with_retry(
                                generate_rollout,
                                student, teacher, seed_batch,
                                n_loops=n_loops,
                                max_new_tokens=args.rollout_len,
                                teacher_mix_alpha=args.teacher_mix_alpha,
                                temperature=args.onpolicy_temp,
                                top_k=args.onpolicy_top_k,
                                use_kv_cache=False,
                            )
                        rollout_buffer.fill(wide, step)
                    rollout = rollout_buffer.draw()
                x_in, y_in = rollout[:, :-1], rollout[:, 1:]
                # Sampled tokens aren't gold → pure soft divergence (targets=None
                # makes distillation_loss drop the hard-CE term).
                distill_targets = None
                op_accum += 1
                prof.stop("rollout")
            else:
                x_in, y_in = x, y
                distill_targets = y

            with amp_ctx:
                # ── Teacher forward (no grad, no autograd graph) ──
                # NOTE: this runs on EVERY micro-step, on-policy or offline. It is
                # the prime suspect for the λ null result — see _StepProfiler.
                with prof.region("teacher_fwd"):
                    t_logits = teacher_logits(teacher_fwd, x_in).to(device)

                # ── Student forward ──
                # The per-loop path is needed when EITHER head is trained across
                # loops. Keeping them independent is deliberate: running
                # --unc-loop-weighting alone isolates "does calibrating the head
                # at every depth help" from "does supervising the LM at every
                # depth help", and those are separate claims about separate
                # heads. Confounding them would repeat the mistake of the
                # depth-policy run, which moved two things and could not say
                # which one hurt.
                unc_l_loop = None
                if args.loop_loss_weighting == "off" and args.unc_loop_weighting == "off":
                    with prof.region("student_fwd"):
                        s_logits, unc = student_fwd(x_in, n_loops=n_loops)

                    # ── Distillation (+ CE blend on the offline path) ──
                    distill_total, distill_metrics = distillation_loss(
                        s_logits, t_logits, targets=distill_targets,
                        temperature=args.temperature,
                        alpha=args.alpha,
                        divergence=args.divergence,
                        jsd_beta=args.jsd_beta,
                    )
                else:
                    # ── LOOP-WEIGHTED distillation: supervise EVERY loop ──
                    # Closes the documented divergence from Ouro, which trains
                    # L = SUM_t p(t|x)·L^(t) while we supervise h_K alone. See
                    # docs/growth_design.md "loop-loss supervision".
                    #
                    # Uses the RAW `student`, not `student_fwd`: the compiled
                    # alias wraps `forward`, and calling a different method on a
                    # compiled module falls back to eager anyway. Being explicit
                    # keeps `--compile` and this flag independent.
                    with prof.region("student_fwd"):
                        states, halt = student.forward_loop_states(
                            x_in, n_loops=n_loops,
                        )
                    K = states.shape[2]
                    _mk = lambda mode: _loop_weights(                # noqa: E731
                        mode, K, halt, alpha=args.loop_loss_alpha,
                        shape=states.shape[:2], device=states.device,
                    )
                    w = _mk(args.loop_loss_weighting) \
                        if args.loop_loss_weighting != "off" else None
                    unc_mode = (
                        args.loop_loss_weighting
                        if args.unc_loop_weighting == "match"
                        else args.unc_loop_weighting
                    )
                    if unc_mode == "off" or is_onpolicy:
                        # On-policy micro-steps never train the uncertainty head:
                        # y_in is the student's OWN sample, so the target would be
                        # "did my sample match my argmax" = sampling noise (P1,
                        # 2026-07-01). That rule is depth-independent.
                        w_unc = None
                    else:
                        w_unc = _mk(unc_mode)

                    distill_total = None
                    soft_sum = hard_sum = 0.0
                    for k in range(K):
                        logits_k = student.head(states[..., k, :])
                        if w is not None:
                            loss_k, metrics_k = distillation_loss(
                                logits_k, t_logits, targets=distill_targets,
                                temperature=args.temperature,
                                alpha=args.alpha,
                                divergence=args.divergence,
                                jsd_beta=args.jsd_beta,
                                token_weights=w[..., k],
                            )
                            distill_total = (
                                loss_k if distill_total is None
                                else distill_total + loss_k
                            )
                            soft_sum += metrics_k["soft"]
                            hard_sum += metrics_k["hard"]
                        if w_unc is not None:
                            # Loop k's OWN logits set the error target, so a
                            # shallow loop learns to report that IT is wrong
                            # rather than inheriting the final loop's confidence.
                            # This is the defect behind ECE 0.288 at loop 0.
                            u_k = uncertainty_calibration_loss(
                                logits_k.detach(),
                                student.uncertainty(states[..., k, :]),
                                y, token_weights=w_unc[..., k],
                            )
                            unc_l_loop = (
                                u_k if unc_l_loop is None else unc_l_loop + u_k
                            )
                        if k == K - 1:
                            # Final loop stays the reference output: every
                            # downstream metric is defined against the depth
                            # inference actually emits.
                            s_logits = logits_k
                    unc = student.uncertainty(states[..., K - 1, :])

                    if w is None:
                        # --unc-loop-weighting running ALONE: the LM objective is
                        # untouched (final loop only), exactly as before.
                        distill_total, distill_metrics = distillation_loss(
                            s_logits, t_logits, targets=distill_targets,
                            temperature=args.temperature,
                            alpha=args.alpha,
                            divergence=args.divergence,
                            jsd_beta=args.jsd_beta,
                        )
                    else:
                        distill_metrics = {"soft": soft_sum, "hard": hard_sum}

                # ── Auxiliary losses (keep MoE / uncertainty / sparsity healthy) ──
                router_buf = collect_router_logits(student)
                lb     = load_balance_loss(
                    router_buf, topk=cfg.n_experts_per_tok,
                    router_bias=moe_router_bias(student),
                )
                # Skip uncertainty calibration on on-policy micro-steps: there
                # y_in is the student's OWN sampled rollout, so the head would
                # train on "did my sample match my argmax" = sampling noise, not
                # an error signal (P1, 2026-07-01). On the offline path y_in == y
                # (gold) — pass `y` explicitly so a future y_in refactor can't
                # silently reintroduce the pollution.
                if is_onpolicy:
                    unc_l = torch.tensor(0.0, device=s_logits.device)
                elif unc_l_loop is not None:
                    # Already accumulated across every loop above, with weights
                    # summing to 1, so it is on the same scale as the final-loop
                    # term it replaces and --unc-coeff keeps its meaning.
                    unc_l = unc_l_loop
                else:
                    unc_l = uncertainty_calibration_loss(s_logits.detach(), unc, y)
                sparse = sparse_activation_loss(router_buf)

                # Depth regulariser — PonderNet × Ouro KL-to-uniform on the
                # halt distribution. Skipped entirely when coeff == 0 so the
                # default-off path has zero overhead. Reads the per-loop λ
                # values that the student's forward (just above) stashed on
                # RecurrentBlock.last_halt_distribution.
                if args.depth_reg_coeff > 0.0:
                    depth = depth_regularization_loss(
                        student, prior="uniform", coeff=1.0,
                    ).to(s_logits.device)
                else:
                    depth = torch.tensor(0.0, device=s_logits.device)

                loss = (
                    distill_total
                    + args.lb_coeff * lb
                    + args.unc_coeff * unc_l
                    + args.sparse_coeff * sparse
                    + args.depth_reg_coeff * depth
                )
                loss = loss / args.grad_accum

            with prof.region("backward"):
                loss.backward()
            loss_accum   += loss.item()
            soft_accum   += distill_metrics["soft"] / args.grad_accum
            hard_accum   += distill_metrics["hard"] / args.grad_accum
            lb_accum     += float(lb.item())  / args.grad_accum
            unc_accum    += float(unc_l.item()) / args.grad_accum
            sparse_accum += float(sparse.item()) / args.grad_accum
            depth_accum  += float(depth.item()) / args.grad_accum

            for name, counts in collect_expert_counts(student).items():
                accum_expert_counts[name] = (
                    accum_expert_counts.get(name, 0) + counts
                )

        with prof.region("optimizer"):
            grad_norm = nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        # Aux-loss-free router bias update (DeepSeek-V3 style).
        util_stats = update_router_bias_from_counts(
            student, accum_expert_counts,
            bias_lr=cfg.router_bias_lr, ddp=False,
        )

        # Sentinel override for grown checkpoints. MUST run AFTER the
        # aux-loss-free updater: during warm-in the schedule controls
        # new-expert biases, not traffic counts. No-op when growth_metadata
        # is None and after the decay window closes.
        if growth_metadata is not None:
            apply_sentinel_to_router_biases(student, growth_metadata, step)
        step += 1
        if prof.enabled and step > profile_start:
            prof.steps += 1

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tps = (args.micro_batch * args.grad_accum * args.seq_len
                   * log_every / dt)
            logger.info(
                f"step {step:6d}/{args.total_steps} | loss {loss_accum:.4f} "
                f"| soft {soft_accum:.4f} | hard {hard_accum:.4f} "
                f"| lb {lb_accum:.4f} | unc {unc_accum:.4f} "
                f"| sparse {sparse_accum:.5f} | depth {depth_accum:.4f} "
                f"| n_loops {n_loops} | op {op_accum}/{args.grad_accum} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| wfac {warmup_factor:.2f} | {tps/1e3:.1f}k tok/s"
            )
            t0 = time.perf_counter()

        if step % 100 == 0 and util_stats:
            log_expert_utilization(util_stats, step)
        if step % 500 == 0 and step > 0:
            log_spectral_radius(student, step)

        should_ckpt = (step % args.ckpt_every == 0) or (
            args.ckpt_every_mins > 0
            and (time.perf_counter() - last_ckpt_time) >= args.ckpt_every_mins * 60.0
        )
        if should_ckpt:
            save_checkpoint(
                student, optimizer, step, cfg, vocab_size,
                args.ckpt_dir, ddp=False, master=True,
                keep_last=args.keep_last,
                keep_milestone_every=args.ckpt_milestone_every,
                extra=ckpt_extra,
            )
            last_ckpt_time = time.perf_counter()

        # In-loop eval — mirrors the pretraining script. Runs on master only
        # (single-GPU here), writes JSON to eval_results/, restores train mode.
        if (
            args.eval
            and step % args.eval_every == 0
            and step > 0
        ):
            from eval.harness import run_eval
            eval_out = os.path.join("eval_results", f"distill_step_{step:07d}.json")
            try:
                run_eval(
                    student,
                    encoding,
                    benchmarks=args.eval_benchmarks,
                    max_samples=args.eval_max_samples,
                    output_path=eval_out,
                    verbose=True,
                )
            except Exception as exc:                       # noqa: BLE001
                logger.exception(f"eval at step {step} failed: {exc}")
            student.train()
            t0 = time.perf_counter()                       # reset tok/s timer

        if shutdown.requested:
            logger.warning(f"distill: shutdown at step {step}; flushing")
            save_checkpoint(
                student, optimizer, step, cfg, vocab_size,
                args.ckpt_dir, ddp=False, master=True,
                keep_last=args.keep_last,
                keep_milestone_every=args.ckpt_milestone_every,
                extra=ckpt_extra,
            )
            break

    # Final checkpoint
    if step > start_step and step % args.ckpt_every != 0 and not shutdown.requested:
        save_checkpoint(
            student, optimizer, step, cfg, vocab_size,
            args.ckpt_dir, ddp=False, master=True,
            keep_last=args.keep_last,
            keep_milestone_every=args.ckpt_milestone_every,
            extra=ckpt_extra,
        )
    if shutdown.requested:
        logger.warning("distill: stopped via signal — resume by re-running.")
    else:
        logger.success("distill: training complete.")


def _cosine_lr(step: int, warmup: int, total: int,
                max_lr: float, min_lr: float) -> float:
    """Linear warmup → cosine decay to `min_lr`. Mirrors the pretrain script."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    decay = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


if __name__ == "__main__":
    main()
