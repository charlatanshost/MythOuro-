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
from contextlib import nullcontext

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
from mythouro.training_utils import (
    LoopCurriculum,
    MixedDataset,
    ExpertSpecializationProbe,
    ProcessRewardHead,
    apply_component_warmup,
    collect_expert_counts,
    collect_router_logits,
    depth_regularization_loss,
    distillation_loss,
    get_optimizer_groups,
    load_balance_loss,
    load_distillation_teacher,
    log_expert_utilization,
    log_spectral_radius,
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
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Distillation weight: 0=pure CE, 1=pure soft loss.")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Softmax temperature for the distillation term.")
    p.add_argument("--lb-coeff", type=float, default=1e-2)
    p.add_argument("--unc-coeff", type=float, default=5e-2)
    p.add_argument("--sparse-coeff", type=float, default=1e-3)
    p.add_argument("--depth-reg-coeff", type=float, default=1e-1,
                   help="PonderNet × Ouro KL-to-uniform regulariser on the "
                        "halt distribution. Default 1e-1 actively prevents "
                        "the ACT loop-collapse failure mode (halt distribution "
                        "pinned to one bucket → loops unused). Empirically, "
                        "1e-2 was too weak against the distillation gradient "
                        "pulling λ→1 (~0.1%% of total loss); 1e-1 gives ~7%% "
                        "and actually moves λ. Pass 0.0 to disable.")
    p.add_argument("--ckpt-dir", default="checkpoints_distill")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--trust-remote-code", action="store_true",
                   help="REQUIRED for the default Ouro teacher (it ships a "
                        "custom modeling_ouro.py). Set whenever the teacher "
                        "repo includes custom modeling code.")
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
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    # Student device — explicit if passed, else cuda:0 (legacy default) or cpu.
    # The teacher device is handled independently via --teacher-device, so a
    # multi-card layout looks like:
    #     --teacher-device cuda:0  --student-device cuda:1
    # We validate the requested device exists before any allocation to fail
    # loudly rather than waste 5 minutes building the model on a phantom GPU.
    if args.student_device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.student_device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"--student-device={device!r} but CUDA is unavailable"
                )
            # `cuda` alone means cuda:0; `cuda:N` selects device N
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
            n_devices = torch.cuda.device_count()
            if idx >= n_devices:
                raise RuntimeError(
                    f"--student-device={device!r} but only {n_devices} CUDA "
                    f"device(s) visible (run `nvidia-smi` to check)."
                )

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    # ------------------------------------------------------------------
    # Tokenizer + student
    # ------------------------------------------------------------------
    encoding = MythOuroTokenizer(args.tokenizer)
    vocab_size = encoding.vocab_size

    cfg = _VARIANT_FUNCS[args.student_variant]()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = args.seq_len

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
        fused=torch.cuda.is_available(),
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    existing = list_ckpts(args.ckpt_dir)
    if existing:
        logger.info(f"distill: resuming from {existing[-1]}")
        start_step, _ = load_checkpoint(
            student, optimizer, existing[-1], ddp=False, current_cfg=cfg,
        )

    # ------------------------------------------------------------------
    # Data + curriculum
    # ------------------------------------------------------------------
    dataset = MixedDataset(encoding, args.seq_len, rank=0, world_size=1)
    loader = DataLoader(
        dataset, batch_size=args.micro_batch, num_workers=2, pin_memory=True,
    )

    curriculum = LoopCurriculum(
        start_loops=2,
        max_loops=cfg.max_loop_iters,
        warmup_steps=max(args.warmup_steps * 2, args.total_steps // 20),
        total_steps=args.total_steps // 2,
    )

    # `--random-depth` switches per-step depth selection from
    # `curriculum.get(step)` (fixed) to `curriculum.get_sampled(step, rng)`
    # (random uniform in [start, get(step)]). Seeded for reproducibility.
    import random as _random
    depth_rng = _random.Random(0)

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if "cuda" in device else nullcontext()
    )

    shutdown = ShutdownHandler()
    shutdown.install()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    student.train()
    data_iter = iter(loader)
    step = start_step
    t0 = time.perf_counter()
    log_every = args.log_every

    while step < args.total_steps:
        cur_lr = _cosine_lr(step, args.warmup_steps, args.total_steps,
                             args.lr, args.lr * 0.1)
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
        accum_expert_counts: dict = {}

        for micro_step in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_ctx:
                # ── Teacher forward (no grad, no autograd graph) ──
                t_logits = teacher_logits(teacher, x).to(device)

                # ── Student forward ──
                s_logits, unc = student(x, n_loops=n_loops)

                # ── Distillation + CE blend ──
                distill_total, distill_metrics = distillation_loss(
                    s_logits, t_logits, targets=y,
                    temperature=args.temperature,
                    alpha=args.alpha,
                )

                # ── Auxiliary losses (keep MoE / uncertainty / sparsity healthy) ──
                router_buf = collect_router_logits(student)
                lb     = load_balance_loss(router_buf, topk=cfg.n_experts_per_tok)
                unc_l  = uncertainty_calibration_loss(s_logits.detach(), unc, y)
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

        grad_norm = nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        # Aux-loss-free router bias update (DeepSeek-V3 style).
        util_stats = update_router_bias_from_counts(
            student, accum_expert_counts,
            bias_lr=cfg.router_bias_lr, ddp=False,
        )
        step += 1

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tps = (args.micro_batch * args.grad_accum * args.seq_len
                   * log_every / dt)
            logger.info(
                f"step {step:6d}/{args.total_steps} | loss {loss_accum:.4f} "
                f"| soft {soft_accum:.4f} | hard {hard_accum:.4f} "
                f"| lb {lb_accum:.4f} | unc {unc_accum:.4f} "
                f"| sparse {sparse_accum:.5f} | depth {depth_accum:.4f} "
                f"| n_loops {n_loops} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| wfac {warmup_factor:.2f} | {tps/1e3:.1f}k tok/s"
            )
            t0 = time.perf_counter()

        if step % 100 == 0 and util_stats:
            log_expert_utilization(util_stats, step)
        if step % 500 == 0 and step > 0:
            log_spectral_radius(student, step)

        if step % args.ckpt_every == 0:
            save_checkpoint(
                student, optimizer, step, cfg, vocab_size,
                args.ckpt_dir, ddp=False, master=True,
            )

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
            )
            break

    # Final checkpoint
    if step > start_step and step % args.ckpt_every != 0 and not shutdown.requested:
        save_checkpoint(
            student, optimizer, step, cfg, vocab_size,
            args.ckpt_dir, ddp=False, master=True,
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
