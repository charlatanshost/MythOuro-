#!/usr/bin/env python3
"""
MythOuro supervised fine-tuning (SFT) script.

Stage 2 of the post-pretraining pipeline. Takes a distilled or pretrained
MythOuro checkpoint and teaches it the *behavioural* skill of following
prompts: recognising the ChatML structure, halting on `<|im_end|>`, and
keeping responses on-topic instead of free-associating.

The ONE thing that distinguishes this script from `training/distill.py`:
loss is masked to assistant-response tokens only. Prompt tokens (system
turn, user turn, the `<|im_start|>assistant\\n` header) contribute zero
gradient. See `mythouro.sft_data.MixedSFTDataset` for the contract.

The auxiliary losses (load_balance, uncertainty_calibration,
sparse_activation, depth_regularization) are kept — they apply
identically to SFT because MythOuro's MoE / UncertaintyHead / ACT must
stay healthy regardless of training stage.

CLI
---
    python -m training.sft \\
        --resume archived_models/mythouro_distill_tiny_v1/step_0005000.pt \\
        --total-steps 3000 \\
        --lr 2e-5 \\
        --eval --eval-every 1000

Resume target
-------------
SFT is meant to BUILD ON a pretrained / distilled base, never to train
from scratch. The `--resume` flag is required. The script will refuse
to run without it because pure-SFT-from-scratch on instruction data is
a wasted compute trajectory at this scale.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from mythouro import MythOuro
from mythouro.checkpointing import (
    ShutdownHandler,
    load_checkpoint,
    save_checkpoint,
)
from mythouro.sft_data import MixedSFTDataset
from mythouro.tokenizer import MythOuroTokenizer
from mythouro.training_utils import (
    LoopCurriculum,
    ExpertSpecializationProbe,
    ProcessRewardHead,
    apply_component_warmup,
    collect_expert_counts,
    collect_router_logits,
    depth_regularization_loss,
    get_optimizer_groups,
    load_balance_loss,
    log_expert_utilization,
    log_spectral_radius,
    sparse_activation_loss,
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
from mythouro.grow import apply_sentinel_to_router_biases
from mythouro import device as dev


_VARIANT_FUNCS = {
    "mythouro_distill_tiny":  mythouro_distill_tiny,
    "mythouro_distill_tiny_dense": mythouro_distill_tiny_dense,
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
        description="Supervised fine-tune an MythOuro checkpoint on "
                    "instruction data.",
    )
    p.add_argument("--resume", required=True,
                   help="Path to a pretrained / distilled checkpoint. "
                        "Required — SFT-from-scratch on instruction data "
                        "is wasted compute at this scale.")
    p.add_argument("--student-variant", default="mythouro_distill_tiny",
                   choices=list(_VARIANT_FUNCS),
                   help="Must match the variant of the resumed checkpoint.")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking",
                   help="HF tokenizer id. Must match what the checkpoint "
                        "was trained with.")
    p.add_argument("--total-steps", type=int, default=3000,
                   help="SFT converges faster than pretraining; 3000 steps "
                        "is enough to see clear behavioural change at this "
                        "scale.")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Lower than pretraining/distillation (3e-4) — SFT "
                        "fine-tunes a converged base; aggressive LR would "
                        "erase pretraining signal.")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--lb-coeff", type=float, default=1e-2)
    p.add_argument("--unc-coeff", type=float, default=5e-2)
    p.add_argument("--sparse-coeff", type=float, default=1e-3)
    p.add_argument("--depth-reg-coeff", type=float, default=1e-1,
                   help="Same loop-collapse guard as distillation. SFT can "
                        "also push ACT toward halt-early; keep this on.")
    p.add_argument("--ckpt-dir", default="checkpoints_sft")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--random-depth", action="store_true",
                   help="Per batch, sample unroll depth uniformly in "
                        "[start_loops, curriculum.get(step)].")
    p.add_argument("--data-mix", choices=["clean", "legacy"], default="clean",
                   help="SFT data mix. 'clean' (default since 2026-06-11) = "
                        "zero OpenAI-output provenance (Tulu-3, "
                        "OpenMathInstruct-2, NuminaMath, OpenCodeInstruct, "
                        "MIRIAD, PubMedQA, ChemData — see "
                        "docs/clean_sft_datasets.md); produces distributable "
                        "checkpoints. 'legacy' = the v2/v4-era OpenHermes/"
                        "Magicoder/MetaMathQA mix (reproduction only; carries "
                        "the OpenAI-ToS constraint).")
    p.add_argument("--no-contamination-filter", action="store_true",
                   help="Disable the eval-benchmark contamination guard "
                        "(GSM8K/ARC 13-grams). On by default for the clean "
                        "mix because OpenMathInstruct-2 is augmented from "
                        "GSM8K-style problems.")
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
    p.add_argument("--eval", "-e", action="store_true")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-max-samples", type=int, default=50)
    p.add_argument("--eval-benchmarks", nargs="+", default=["all"])
    p.add_argument("--device", default=None,
                   help="cuda:N / cpu. Default: cuda:0 if available else cpu.")
    p.add_argument("--use-8bit-adam", action="store_true",
                   help="Use bitsandbytes 8-bit AdamW instead of torch's "
                        "fp32 AdamW. Saves ~2.5 GB of VRAM on a ~400M model "
                        "by quantizing the optimizer's first/second moment "
                        "buffers to 8-bit (block-wise dynamic quantization). "
                        "Requires `pip install bitsandbytes`. Convergence is "
                        "near-identical to fp32 AdamW in practice; the small "
                        "optimizer noise from quantization rarely matters at "
                        "SFT scale.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# bitsandbytes CUDA-binary selection
# ---------------------------------------------------------------------------


def _configure_bnb_cuda_version() -> None:
    """
    Point bitsandbytes at the best-matching bundled CUDA binary.

    bitsandbytes ships prebuilt `libbitsandbytes_cudaXXX.dll` (Windows) /
    `.so` (Linux) for a fixed set of CUDA versions. On bleeding-edge CUDA
    (e.g. torch built for cu132 / CUDA 13.2), bnb may not ship an exact
    `cuda132` binary and fails at import with "Configured CUDA binary not
    found at libbitsandbytes_cuda132.dll".

    CUDA binaries are forward-compatible across minor versions within a
    major release, so the `cuda130` binary runs correctly against a 13.2
    runtime. bnb honours the `BNB_CUDA_VERSION` env var (read at import
    time) to override which binary it loads. This helper:

      1. Respects an existing BNB_CUDA_VERSION if the user already set one.
      2. Otherwise scans the installed bnb package for bundled
         `libbitsandbytes_cudaXXX` binaries.
      3. Picks the highest XXX that is <= the torch CUDA version (so we
         never load a *newer* binary than the runtime supports), and sets
         BNB_CUDA_VERSION to it.

    No-op (logs a warning) if bnb isn't importable or no CUDA binary is
    found — the subsequent `import bitsandbytes` will surface the real
    error with bnb's own diagnostics.
    """
    import glob
    import importlib.util
    import os
    import re

    if os.environ.get("BNB_CUDA_VERSION"):
        logger.info(
            f"sft: BNB_CUDA_VERSION already set to "
            f"{os.environ['BNB_CUDA_VERSION']!r}; respecting it."
        )
        return

    # Locate the bnb package directory without importing it (import would
    # trigger the binary load we're trying to configure).
    spec = importlib.util.find_spec("bitsandbytes")
    if spec is None or not spec.submodule_search_locations:
        logger.warning(
            "sft: bitsandbytes not found while configuring CUDA version; "
            "the import below will surface the real error."
        )
        return
    pkg_dir = list(spec.submodule_search_locations)[0]

    # Find bundled CUDA binaries and parse their version numbers.
    patterns = [
        os.path.join(pkg_dir, "libbitsandbytes_cuda*.dll"),
        os.path.join(pkg_dir, "libbitsandbytes_cuda*.so"),
    ]
    available: list[int] = []
    for pat in patterns:
        for path in glob.glob(pat):
            m = re.search(r"cuda(\d+)", os.path.basename(path))
            if m:
                available.append(int(m.group(1)))
    available = sorted(set(available))
    if not available:
        logger.warning(
            f"sft: no bundled libbitsandbytes_cudaXXX binary found in "
            f"{pkg_dir}; letting bnb pick its default."
        )
        return

    # Parse torch's CUDA version (e.g. "13.2" -> 132) as the ceiling.
    torch_cuda = torch.version.cuda  # e.g. "13.2" or None (cpu build)
    if torch_cuda is None:
        logger.warning(
            "sft: torch reports no CUDA; not configuring BNB_CUDA_VERSION."
        )
        return
    major, _, minor = torch_cuda.partition(".")
    torch_cuda_int = int(major) * 10 + int(minor or 0)

    # Highest bundled binary that doesn't exceed the runtime CUDA version.
    candidates = [v for v in available if v <= torch_cuda_int]
    if not candidates:
        # Every bundled binary is newer than the runtime — fall back to the
        # lowest available rather than refusing; better to try than to die.
        chosen = available[0]
        logger.warning(
            f"sft: all bundled bnb CUDA binaries {available} exceed runtime "
            f"CUDA {torch_cuda_int}; falling back to lowest ({chosen})."
        )
    else:
        chosen = candidates[-1]

    os.environ["BNB_CUDA_VERSION"] = str(chosen)
    if chosen != torch_cuda_int:
        logger.info(
            f"sft: torch CUDA is {torch_cuda} but bnb has no exact "
            f"libbitsandbytes_cuda{torch_cuda_int} binary; using cuda{chosen} "
            f"(forward-compatible within the same CUDA major version)."
        )
    else:
        logger.info(f"sft: bnb using matching cuda{chosen} binary.")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy on response tokens only.

    Args:
        logits    : (B, T, V)
        targets   : (B, T)        — long
        loss_mask : (B, T)        — float, 1.0 on response positions

    The naive `F.cross_entropy(..., reduction="none") * loss_mask` works
    but allocates the full (B*T,) per-token loss; for 49152 vocab this
    is fine. We then divide by the sum of the mask, NOT by B*T — the
    latter would falsely shrink the loss when most of the batch is
    prompt/padding.

    A batch with no response tokens at all (all-masked) would divide by
    zero. We guard with a max-clamp; that batch will silently contribute
    zero gradient, which is the correct behaviour (nothing to learn from).
    """
    B, T, V = logits.shape
    per_token = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    masked = per_token * loss_mask
    denom = loss_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    # Device resolution: explicit > cuda:0 > xpu > cpu (CUDA / Intel XPU / CPU).
    device = dev.pick_device(args.device)
    if dev.is_accelerator(device) and not dev.is_available(device):
        raise RuntimeError(
            f"--device={device!r} but {dev.backend(device)} is unavailable"
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

    student = MythOuro(cfg).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    logger.info(
        f"sft: student={args.student_variant} params={n_params:,} "
        f"vocab={vocab_size} device={device} amp={amp_dtype}"
    )

    # ------------------------------------------------------------------
    # Aux heads
    # ------------------------------------------------------------------
    prm_head  = ProcessRewardHead(cfg.dim).to(device)
    esp_probe = ExpertSpecializationProbe(cfg.n_experts).to(device)

    optimizer_groups = get_optimizer_groups(
        student,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        extra_base_params=list(prm_head.parameters())
                         + list(esp_probe.parameters()),
    )
    if args.use_8bit_adam:
        # bitsandbytes AdamW8bit quantizes the optimizer's first/second
        # moment buffers to 8-bit (block-wise dynamic). Saves ~2.5 GB on
        # a ~400M model. The `fused=` kwarg isn't accepted by bnb's
        # interface, and bnb uses its own paged-memory machinery instead.
        #
        # MUST run _configure_bnb_cuda_version() BEFORE importing bnb — it
        # reads BNB_CUDA_VERSION at import time to decide which CUDA binary
        # to load. On bleeding-edge CUDA (e.g. 13.2) bnb may not ship an
        # exact-match binary; the helper points it at the highest bundled
        # binary that's ≤ the runtime CUDA version.
        _configure_bnb_cuda_version()
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "--use-8bit-adam requires bitsandbytes. Install with: "
                "`pip install bitsandbytes`. Original error: " + str(exc)
            )
        optimizer = bnb.optim.AdamW8bit(
            optimizer_groups,
            betas=(0.9, 0.95),
        )
        logger.info("sft: using bitsandbytes AdamW8bit (8-bit optimizer state)")
    else:
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(0.9, 0.95),
            fused=dev.fused_adam_supported(device),
        )

    # ------------------------------------------------------------------
    # Resume — REQUIRED for SFT.
    #
    # We load the model weights from --resume into the freshly-built
    # student, then reset start_step to 0 because the SFT run has its
    # own step counter and LR schedule independent of the upstream
    # pretraining / distillation run.
    # ------------------------------------------------------------------
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(
            f"--resume {args.resume!r} not found. SFT requires a "
            "pretrained or distilled base checkpoint."
        )
    logger.info(f"sft: loading base from {args.resume}")
    _, resume_extra = load_checkpoint(
        student, optimizer, args.resume, ddp=False, current_cfg=cfg,
    )
    # SFT counter starts fresh — the optimizer state from the upstream
    # run isn't a meaningful prior for the new LR schedule.
    for group in optimizer.param_groups:
        group["lr"] = args.lr
    start_step = 0

    # If this checkpoint was produced by `tools/grow_checkpoint.py`, the
    # `extra` dict carries a `growth_metadata` entry describing the
    # promotion. We apply the sentinel-bias decay schedule each step so
    # the new experts are gradually allowed into the top-k selection,
    # rather than entering all at once (which would spike the loss).
    growth_metadata = (resume_extra or {}).get("growth_metadata")
    if growth_metadata is not None:
        logger.info(
            f"sft: detected grown checkpoint — "
            f"source_n_experts={growth_metadata.get('source_n_experts')}, "
            f"target_n_experts={growth_metadata.get('target_n_experts')}, "
            f"sentinel_bias={growth_metadata.get('sentinel_bias')}, "
            f"n_decay_steps={growth_metadata.get('n_decay_steps')}"
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = MixedSFTDataset(
        encoding, args.seq_len, rank=0, world_size=1,
        mix=args.data_mix,
        contamination_filter=(False if args.no_contamination_filter else None),
        seed=args.seed,
    )
    # num_workers=0 runs data loading in the main process. Many of the
    # popular instruction datasets (OpenHermes, Magicoder, MetaMathQA)
    # ship as a single shard, and HF's streaming-sharded iterator throws
    # "list index out of range" when more workers exist than shards.
    # Single-worker also matches the sft_data.py fallback in
    # `_open_source`, which skips sharding when `total_shards == 1`.
    # SFT is forward/backward-bound at this scale; main-thread data
    # loading is not the bottleneck.
    loader = DataLoader(
        dataset, batch_size=args.micro_batch, num_workers=0, pin_memory=True,
    )
    curriculum = LoopCurriculum(
        start_loops=args.start_loops,
        max_loops=cfg.max_loop_iters,
        warmup_steps=max(args.warmup_steps * 2, args.total_steps // 20),
        total_steps=args.total_steps // 2,
    )
    import random as _random
    depth_rng = _random.Random(args.seed)

    amp_ctx = (
        torch.amp.autocast(device_type=dev.autocast_type(device), dtype=amp_dtype)
        if dev.is_accelerator(device) else nullcontext()
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
        loss_accum = ce_accum = 0.0
        lb_accum = unc_accum = sparse_accum = depth_accum = 0.0
        resp_frac_accum = 0.0
        accum_expert_counts: dict = {}

        for micro_step in range(args.grad_accum):
            try:
                x, y, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y, mask = next(data_iter)

            x    = x.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with amp_ctx:
                s_logits, unc = student(x, n_loops=n_loops)

                # ── Masked CE on response tokens only ──
                ce = masked_ce_loss(s_logits, y, mask)

                # ── Auxiliary losses (unchanged from distillation) ──
                router_buf = collect_router_logits(student)
                lb     = load_balance_loss(router_buf, topk=cfg.n_experts_per_tok)
                unc_l  = uncertainty_calibration_loss(s_logits.detach(), unc, y)
                sparse = sparse_activation_loss(router_buf)

                if args.depth_reg_coeff > 0.0:
                    depth = depth_regularization_loss(
                        student, prior="uniform", coeff=1.0,
                    ).to(s_logits.device)
                else:
                    depth = torch.tensor(0.0, device=s_logits.device)

                loss = (
                    ce
                    + args.lb_coeff * lb
                    + args.unc_coeff * unc_l
                    + args.sparse_coeff * sparse
                    + args.depth_reg_coeff * depth
                )
                loss = loss / args.grad_accum

            loss.backward()
            loss_accum   += float(loss.item())
            ce_accum     += float(ce.item()) / args.grad_accum
            lb_accum     += float(lb.item()) / args.grad_accum
            unc_accum    += float(unc_l.item()) / args.grad_accum
            sparse_accum += float(sparse.item()) / args.grad_accum
            depth_accum  += float(depth.item()) / args.grad_accum
            # Diagnostic: fraction of tokens in this micro-batch that
            # actually contribute to the loss. Low values (< 0.1) mean
            # the batch is mostly padding/prompt and the effective batch
            # size is much smaller than nominal.
            resp_frac_accum += float(mask.mean().item()) / args.grad_accum

            for name, counts in collect_expert_counts(student).items():
                accum_expert_counts[name] = (
                    accum_expert_counts.get(name, 0) + counts
                )

        grad_norm = nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        util_stats = update_router_bias_from_counts(
            student, accum_expert_counts,
            bias_lr=cfg.router_bias_lr, ddp=False,
        )

        # Sentinel-decay override for grown checkpoints. Runs AFTER the
        # DeepSeek-V3 updater because we want the schedule, not the
        # data-driven update, to control new-expert biases during the
        # warm-in window. No-op when `growth_metadata is None` (regular
        # SFT runs) and after the decay window closes.
        if growth_metadata is not None:
            apply_sentinel_to_router_biases(student, growth_metadata, step)

        step += 1

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tps = (args.micro_batch * args.grad_accum * args.seq_len
                   * log_every / dt)
            logger.info(
                f"step {step:6d}/{args.total_steps} | loss {loss_accum:.4f} "
                f"| ce {ce_accum:.4f} "
                f"| lb {lb_accum:.4f} | unc {unc_accum:.4f} "
                f"| sparse {sparse_accum:.5f} | depth {depth_accum:.4f} "
                f"| resp_frac {resp_frac_accum:.2f} "
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

        if (
            args.eval
            and step % args.eval_every == 0
            and step > 0
        ):
            from eval.harness import run_eval
            eval_out = os.path.join("eval_results", f"sft_step_{step:07d}.json")
            try:
                run_eval(
                    student,
                    encoding,
                    benchmarks=args.eval_benchmarks,
                    max_samples=args.eval_max_samples,
                    output_path=eval_out,
                    verbose=True,
                )
            except Exception as exc:                          # noqa: BLE001
                logger.exception(f"eval at step {step} failed: {exc}")
            student.train()
            t0 = time.perf_counter()

        if shutdown.requested:
            logger.warning(f"sft: shutdown at step {step}; flushing")
            save_checkpoint(
                student, optimizer, step, cfg, vocab_size,
                args.ckpt_dir, ddp=False, master=True,
            )
            break

    if step > start_step and step % args.ckpt_every != 0 and not shutdown.requested:
        save_checkpoint(
            student, optimizer, step, cfg, vocab_size,
            args.ckpt_dir, ddp=False, master=True,
        )
    if shutdown.requested:
        logger.warning("sft: stopped via signal — resume by re-running.")
    else:
        logger.success("sft: training complete.")


def _cosine_lr(step: int, warmup: int, total: int,
               max_lr: float, min_lr: float) -> float:
    """Linear warmup → cosine decay to `min_lr`."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    decay = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


if __name__ == "__main__":
    main()
