#!/usr/bin/env python3
"""
MythOuro pretraining on FineWeb-Edu with FSDP + AdamW.

Single GPU:
    python training/3b_fine_web_edu.py

Multi-GPU:
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py

CLI flags:
    --eval / -e             Run the eval harness every `eval_every` steps.
    --eval-every N          Override the default eval cadence (1000 steps).
    --eval-max-samples N    Per-benchmark sample cap during in-loop evals.
    --eval-benchmarks ...   Subset of benchmarks to run (default: all).
"""

import os
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from contextlib import nullcontext

from datasets import load_dataset

from mythouro import MythOuro
from mythouro.variants import mythouro_3b
from mythouro.tokenizer import MythOuroTokenizer
from mythouro.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    list_ckpts as _list_ckpts,
    ShutdownHandler,
)
from mythouro.training_utils import (
    LoopCurriculum,
    LoopDepthAnnealer,
    MixedDataset,
    ProcessRewardHead,
    ExpertSpecializationProbe,
    build_fsdp_model,
    combined_loss,
    consistency_loss,
    contrastive_loop_loss,
    get_domain_labels,
    log_spectral_radius,
    process_reward_loss,
    sparse_activation_loss,
    collect_router_logits,
    collect_expert_counts,
    update_router_bias_from_counts,
    log_expert_utilization,
    get_optimizer_groups,
    apply_component_warmup,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    """
    Streaming FineWeb-Edu loader yielding fixed-length (input, target) pairs.

    FineWeb-Edu is trillions of tokens, so `streaming=True` pulls shards on
    demand instead of materializing to disk. Sharding is two-dimensional —
    `world_size` ranks × `num_workers` DataLoader workers per rank — and each
    `(rank, worker_id)` deterministically owns one shard of the global stream.
    That gives disjoint coverage without any cross-process coordination.

    Streaming datasets are not seekable, so a resumed run re-enters its shard
    from the beginning. Acceptable at pretraining scale: the chance of
    re-playing the same tokens before the run ends is negligible versus the
    cost of a true resumable loader.
    """

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        """
        Args:
            encoding   -- tokenizer exposing `.encode(str) -> list[int]`
            seq_len    -- context length; every yielded pair has this many tokens
            subset     -- FineWeb-Edu config name (e.g. "sample-10BT", "default")
            rank       -- global rank of this process within the distributed job
            world_size -- total number of distributed processes
        """
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        """
        Yield `(input_ids, target_ids)` tensors of length `seq_len` forever.

        Inputs and targets are shifted by one for next-token prediction —
        `target[i] == input[i + 1]`. Documents are concatenated into a rolling
        buffer and sliced into fixed-length chunks, packing short docs together
        and splitting long ones. This keeps every step at the same shape,
        which under FSDP avoids recompute from variable-length inputs and
        removes the need for a pad-aware attention mask.
        """
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup → half-cosine decay to `min_lr`.

    Standard language-model pretraining schedule. The warmup phase prevents
    Adam's second-moment estimate from collapsing to a huge LR in the first
    few steps when gradients are noisy. The cosine tail lets the model make
    small, increasingly conservative updates near the end of training rather
    than crashing to `min_lr` at a fixed step.

    Behavior by region:
        step < warmup                 → linear ramp 0 → max_lr
        warmup ≤ step < total         → cosine decay max_lr → min_lr
        step ≥ total                  → clamped at min_lr (safety for
                                        off-by-one step counters at the end
                                        of training)

    Args:
        step    -- current global optimizer step (0-indexed)
        warmup  -- number of warmup steps before cosine decay begins
        total   -- step at which the cosine reaches `min_lr`
        max_lr  -- peak learning rate reached at the end of warmup
        min_lr  -- floor learning rate at and after `total` steps

    Returns:
        Scalar learning rate for this step.
    """
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# Checkpoint helpers (save_checkpoint, load_checkpoint, ShutdownHandler,
# list_ckpts) live in `mythouro.checkpointing` so they can be imported
# without dragging in the heavy training-script imports (datasets, FSDP)
# during testing.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_cli_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """
    Limited CLI surface — kept intentionally small. Hyperparameters live as
    literal constants inside `main()`; the CLI only exposes flags that vary
    legitimately *between* runs (eval cadence, eval subset).
    """
    parser = argparse.ArgumentParser(
        description="MythOuro pretraining + optional in-loop eval harness.",
    )
    parser.add_argument(
        "--eval", "-e", action="store_true",
        help="Run the eval harness every `--eval-every` steps. "
             "Writes a JSON report per eval into `eval_results/`.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=1000,
        help="Step cadence for in-loop eval (default: 1000).",
    )
    parser.add_argument(
        "--eval-max-samples", type=int, default=50,
        help="Per-benchmark sample cap during in-loop eval (default: 50).",
    )
    parser.add_argument(
        "--eval-benchmarks", nargs="+", default=["all"],
        help="Benchmarks to run. Default: all. Names: perplexity, "
             "arc_challenge, gsm8k, loop_efficiency, ece.",
    )
    parser.add_argument(
        "--random-depth", action="store_true",
        help="Per batch, sample unroll depth uniformly in "
             "[start_loops, curriculum.get(step)] instead of using the "
             "curriculum value directly. Forces the model to be robust "
             "across the depth range it has been ramped to so far.",
    )
    parser.add_argument(
        "--depth-reg-coeff", type=float, default=1e-1,
        help="PonderNet × Ouro KL-to-uniform regulariser on the halt "
             "distribution. Default 1e-1 actively prevents the ACT "
             "loop-collapse failure mode (1e-2 was empirically too weak "
             "against the main task gradient pulling λ→1). Pass 0.0 to "
             "disable.",
    )
    return parser.parse_args(argv)


def main():
    """
    End-to-end pretraining entry point.

    Order matters: distributed init must run before any CUDA allocation, the
    tokenizer must exist before the model is built (vocab_size flows into
    cfg), and FSDP must wrap the model before the optimizer is constructed
    (FSDP re-flattens parameters, so an optimizer built on the unwrapped
    model would track stale param objects). Resume then loads state into the
    already-constructed optimizer in-place.

    Lifecycle:
        1. Initialize torch.distributed (NCCL) if launched under torchrun.
        2. Build tokenizer → derive vocab_size.
        3. Construct MythOuro with the 3B variant config.
        4. Wrap in FSDP with FULL_SHARD + bf16/fp16 mixed precision (multi-GPU)
           or move to device + autocast (single-GPU).
        5. Build fused AdamW on (possibly sharded) parameters.
        6. Resume from the latest checkpoint in `ckpt_dir` if one exists.
        7. Stream FineWeb-Edu through grad-accumulation microbatches with
           cosine LR schedule, per-step logging, and periodic checkpoints.
           If `--eval` is set, the harness fires every `eval_every` steps
           on the master rank only.
        8. Write a final checkpoint if the last save wasn't aligned to
           `ckpt_every`, then barrier + tear down the process group.

    All hyperparameters are literal constants in this function by design —
    pretraining runs are long-lived and each run pins exact settings; only
    eval-cadence flags are exposed on the CLI (see `_parse_cli_args`).
    """
    args = _parse_cli_args()
    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        logger.info(
            f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythOuroTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        logger.info(f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    seq_len = 2048
    micro_batch = 4
    target_tokens = 30_000_000_000
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = 2000
    lr = 3e-4
    wd = 0.1
    log_every = 10
    ckpt_every = 1000
    ckpt_dir = "checkpoints"
    dataset_subset = "sample-10BT"  # → sample-100BT or "default" for full run

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = mythouro_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = MythOuro(cfg)

    if ddp:
        # Part 2: HYBRID_SHARD FSDP — shards within NVLink groups and
        # replicates across them, keeping the costly all-reduces on the
        # fast NVLink fabric. Falls back to FULL_SHARD on older torch.
        model = build_fsdp_model(model, local_rank, amp_dtype)
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    # ------------------------------------------------------------------
    # Part 2 auxiliary heads (ProcessRewardHead + ExpertSpecializationProbe)
    # ------------------------------------------------------------------
    # These live outside the model so we can optimise / save them
    # independently. Their losses fire on a fraction of steps (cheap on
    # average) but contribute meaningful supervision signal.
    prm_head = ProcessRewardHead(cfg.dim).to(
        device if not ddp else f"cuda:{local_rank}"
    )
    esp_probe = ExpertSpecializationProbe(cfg.n_experts).to(
        device if not ddp else f"cuda:{local_rank}"
    )

    # FSDP handles its own mixed precision; only need autocast for single-GPU
    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")

    # ------------------------------------------------------------------
    # Optimizer (two param groups: base + warmed-up "new component" group;
    # Part 2 auxiliary heads go into the base group at full LR — they're
    # zero-output-init and don't need a separate ramp.)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        get_optimizer_groups(
            model,
            base_lr=lr,
            weight_decay=wd,
            extra_base_params=list(prm_head.parameters())
                             + list(esp_probe.parameters()),
        ),
        betas=(0.9, 0.95),
        fused=True,
    )

    # ------------------------------------------------------------------
    # Resume from latest checkpoint (if any)
    # ------------------------------------------------------------------
    # Streaming datasets are not resumable by position, so re-iterating from
    # the beginning is accepted — at pretraining scale the loss of dataset
    # position is negligible vs. the cost of discarded training steps.
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from checkpoint: {latest}")
        start_step, _ = load_checkpoint(
            model, optimizer, latest, ddp, current_cfg=cfg,
        )
        if master:
            logger.success(f"Resumed at step {start_step}")

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    # MixedDataset blends three corpora at fixed ratios (40% general /
    # 40% math / 20% code) — see training_utils for the source list and
    # rationale. FineWebEduDataset is kept around for single-corpus
    # experiments but is no longer the default.
    dataset = MixedDataset(encoding, seq_len, rank, world_size, seed=rank)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)

    # Loop-depth curriculum: start shallow, ramp to cfg.max_loop_iters by
    # mid-training. Cheap early steps + full depth where it matters.
    curriculum = LoopCurriculum(
        start_loops=2,
        max_loops=cfg.max_loop_iters,
        warmup_steps=max(warmup_steps * 2, total_steps // 20),
        total_steps=total_steps // 2,
    )

    # Loop-depth annealer (Part 2): in the final ~15% of training, push
    # the loop count modestly beyond cfg.max_loop_iters so the model
    # develops representations that depth-extrapolate cleanly at inference.
    #
    # Cap chosen as base+2 (was base+8) because Ouro's empirical sweep
    # showed measurable accuracy degradation past ~8 loops on hard
    # reasoning tasks — extrapolating too aggressively is actively
    # harmful, not just wasteful. Keeping the bump small lets us teach
    # "small headroom" without venturing into the over-loop regime.
    annealer = LoopDepthAnnealer(
        base_loops=cfg.max_loop_iters,
        max_extra_loops=cfg.max_loop_iters + 2,
        anneal_start=int(total_steps * 0.85),
        total_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Install signal handlers BEFORE the training loop so an interrupt at
    # any point past here triggers a graceful checkpoint flush instead of
    # losing all progress since the last `ckpt_every` save.
    shutdown = ShutdownHandler()
    shutdown.install()

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step

    # Seeded RNG for `--random-depth`. Per-rank seed so DDP ranks don't
    # all sample the same depth (would defeat the diversification point).
    import random as _random
    depth_rng = _random.Random(rank if ddp else 0)

    # Diagnostic / loss-cadence constants. Kept as named values so a
    # train-loop reader sees the cost story at a glance:
    #   consistency_loss + contrastive_loop_loss run extra forwards,
    #   so they fire on a fraction of steps only.
    consistency_every = 25
    contrastive_every = 50
    process_reward_every = 20
    expert_probe_every = 50
    spectral_log_every = 500
    expert_util_log_every = 100      # MoE util / router_bias L2 diagnostic

    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        # Apply per-group LR: the base group gets `cur_lr`; the
        # new-component group (InjectionScheduler / LoRAAdapter.down /
        # MultiScaleInjection) gets `cur_lr * factor(step)` where the
        # factor ramps 0→1 over `cfg.new_component_warmup_steps`. This
        # prevents the non-zero-init Part-1+2 components from destabilising
        # the base block before they've had a chance to settle.
        warmup_factor = apply_component_warmup(
            optimizer, cur_lr, step, cfg.new_component_warmup_steps,
        )

        # Curriculum during the first half of training; annealer takes over
        # in the final ~15% to push depth beyond cfg.max_loop_iters.
        # `--random-depth` flips the curriculum path to sample uniformly in
        # [start, get(step)] each batch. The annealer path stays
        # deterministic — it has a different purpose (depth extrapolation
        # at the end of training, which we want consistently engaged).
        if step >= annealer.anneal_start:
            n_loops = annealer.get(step)
        elif args.random_depth:
            n_loops = curriculum.get_sampled(step, depth_rng)
        else:
            n_loops = curriculum.get(step)

        optimizer.zero_grad()
        loss_accum = 0.0
        ce_accum = 0.0
        lb_accum = 0.0
        unc_accum = 0.0
        depth_accum = 0.0
        cons_accum = 0.0
        cont_accum = 0.0
        prm_accum = 0.0
        sparse_accum = 0.0
        esp_accum = 0.0

        # Aux-loss-free MoE routing: accumulate per-MoE-layer expert counts
        # across the grad-accum micro-steps so the bias updater (after
        # optimizer.step) operates on the full macro-batch distribution.
        accum_expert_counts: "dict[str, torch.Tensor]" = {}

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                # Hook to capture the normed hidden state for the
                # ProcessRewardHead (lives at the input of the LM head).
                hidden_capture: dict = {}
                if step % process_reward_every == 0 and micro_step == 0:
                    head_mod = model.module.head if hasattr(model, "module") else model.head
                    def _grab_hidden(_mod, inp, _out):
                        hidden_capture["h"] = inp[0]
                    prm_hook = head_mod.register_forward_hook(_grab_hidden)
                else:
                    prm_hook = None

                logits, unc = model(x, n_loops=n_loops)

                if prm_hook is not None:
                    prm_hook.remove()

                loss, metrics = combined_loss(
                    model,
                    logits,
                    unc,
                    y,
                    vocab_size=vocab_size,
                    topk=cfg.n_experts_per_tok,
                    depth_reg_coeff=args.depth_reg_coeff,
                )

                # Sparse activation regulariser — cheap, run every step
                # alongside the main combined_loss.
                sparse_l = sparse_activation_loss(collect_router_logits(model))
                loss = loss + sparse_l
                sparse_accum += float(sparse_l.item())

                # Consistency self-distillation across loop depths.
                if (
                    step % consistency_every == 0
                    and micro_step == 0
                    and n_loops >= 4
                ):
                    cons = consistency_loss(
                        model, x,
                        n_loops_low=max(2, n_loops // 2),
                        n_loops_high=n_loops,
                    )
                    loss = loss + 0.1 * cons
                    cons_accum += float(cons.item())

                # Contrastive loop loss — discriminates easy vs hard tokens
                # by hidden-state movement across loop depths.
                if (
                    step % contrastive_every == 0
                    and micro_step == 0
                    and n_loops >= 4
                ):
                    cont = contrastive_loop_loss(
                        model, x, y,
                        n_loops_low=max(2, n_loops // 2),
                        n_loops_high=n_loops,
                    )
                    loss = loss + cont
                    cont_accum += float(cont.item())

                # Process-reward head update — operates on the captured
                # normed hidden state from the same forward pass.
                if prm_hook is not None and "h" in hidden_capture:
                    prm = process_reward_loss(
                        prm_head, hidden_capture["h"], logits, y,
                    )
                    loss = loss + prm
                    prm_accum += float(prm.item())

                # Expert-specialisation probe — needs domain labels from
                # the input texts. We round-trip the ids through the
                # tokenizer's decoder to derive the labels.
                if step % expert_probe_every == 0 and micro_step == 0:
                    texts = [encoding.decode(ids.tolist()) for ids in x]
                    domain_labels = get_domain_labels(
                        texts, x.device if hasattr(x, "device") else device,
                    )
                    esp = esp_probe.loss(
                        collect_router_logits(model), domain_labels,
                    )
                    loss = loss + esp
                    esp_accum += float(esp.item())

                loss = loss / grad_accum

            loss.backward()
            loss_accum  += loss.item()
            ce_accum    += metrics["ce"]    / grad_accum
            lb_accum    += metrics["lb"]    / grad_accum
            unc_accum   += metrics["unc"]   / grad_accum
            depth_accum += metrics["depth"] / grad_accum

            # Snapshot per-MoE-layer expert counts from this micro-step
            # and sum into the macro-step accumulator.
            for name, counts in collect_expert_counts(model).items():
                if name in accum_expert_counts:
                    accum_expert_counts[name] += counts
                else:
                    accum_expert_counts[name] = counts

        # FSDP shards parameters, so `nn.utils.clip_grad_norm_` would clip
        # against each rank's local norm and miss the cross-shard gather.
        # FSDP.clip_grad_norm_ computes the true global norm and returns it.
        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Aux-loss-free MoE routing: nudge router_bias toward uniform
        # expert utilisation. Runs OUTSIDE the optimizer because the bias
        # is a non-gradient buffer. Under DDP/FSDP the counts are
        # all-reduced inside so every rank applies the same delta.
        util_stats = update_router_bias_from_counts(
            model,
            accum_expert_counts,
            bias_lr=cfg.router_bias_lr,
            ddp=ddp,
        )
        step += 1

        if master and step % expert_util_log_every == 0 and util_stats:
            log_expert_utilization(util_stats, step)

        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            tokens_seen = step * global_batch_tok
            # Compose the optional Part 2 metric strings only when their
            # corresponding losses actually fired this step.
            extras = []
            if step % consistency_every == 0 and cons_accum:
                extras.append(f"cons {cons_accum:.4f}")
            if step % contrastive_every == 0 and cont_accum:
                extras.append(f"cont {cont_accum:.4f}")
            if step % process_reward_every == 0 and prm_accum:
                extras.append(f"prm {prm_accum:.4f}")
            if step % expert_probe_every == 0 and esp_accum:
                extras.append(f"esp {esp_accum:.4f}")
            extras_str = ("| " + " | ".join(extras) + " ") if extras else ""
            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| ce {ce_accum:.4f} | lb {lb_accum:.4f} | unc {unc_accum:.4f} "
                f"| sparse {sparse_accum:.5f} | depth {depth_accum:.4f} "
                f"{extras_str}| n_loops {n_loops} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| wfac {warmup_factor:.2f} "
                f"| {tok_per_sec / 1e6:.2f}M tok/s "
                f"| {tokens_seen / 1e9:.1f}B tokens seen"
            )
            t0 = time.perf_counter()

        if master and step % spectral_log_every == 0 and step > 0:
            log_spectral_radius(model, step)

        # ── Optional in-loop eval ────────────────────────────────────────
        # Master rank only — eval is read-only against the model so
        # other ranks can keep idling at the barrier. FSDP-wrapped
        # parameters are gathered transparently inside the harness via
        # standard forward (no extra summon_full_params needed for
        # read-only metric forwards on rank 0 because rank 0 has the
        # full unsharded copy under HYBRID_SHARD; for FULL_SHARD this
        # would need a `with FSDP.summon_full_params(model):` wrap).
        if (
            args.eval
            and master
            and step % args.eval_every == 0
            and step > 0
        ):
            from eval.harness import run_eval
            eval_out = os.path.join("eval_results", f"step_{step:07d}.json")
            try:
                run_eval(
                    model.module if hasattr(model, "module") else model,
                    encoding,
                    benchmarks=args.eval_benchmarks,
                    max_samples=args.eval_max_samples,
                    output_path=eval_out,
                    verbose=True,
                )
            except Exception as exc:                       # noqa: BLE001
                logger.exception(f"eval at step {step} failed: {exc}")
            # Eval flips the model to eval mode — restore train mode.
            model.train()
            t0 = time.perf_counter()                       # reset tok/s timer

        if step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master,
            )

        # Cooperative shutdown poll: a SIGINT / SIGTERM / SIGBREAK earlier
        # in the run flipped `shutdown.requested`. Flush a final checkpoint
        # and exit the loop cleanly so the next launch can resume from
        # exactly this step — losing at most one optimizer step of work
        # rather than the last `ckpt_every` worth.
        if shutdown.requested:
            if master:
                logger.warning(
                    f"Shutdown requested at step {step} — flushing checkpoint"
                )
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master,
            )
            break

    # Final checkpoint — total_steps may not be divisible by ckpt_every, so
    # without this the tail of the run is lost if the schedule doesn't align.
    # Also skipped when a shutdown-triggered save already fired above.
    if step > start_step and step % ckpt_every != 0 and not shutdown.requested:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

    if ddp:
        # Barrier so no rank exits while another is still finishing its
        # checkpoint gather — avoids NCCL "process group destroyed" noise.
        dist.barrier()
        dist.destroy_process_group()

    if master:
        # Report expert-specialisation result — useful sanity check
        # that the routing actually picked up domain structure.
        expert_domains = esp_probe.predict_expert_domains()
        logger.info(f"Final expert-domain assignment: {expert_domains}")
        if shutdown.requested:
            logger.warning("Training stopped via signal — resume by re-running.")
        else:
            logger.success("Training complete.")


if __name__ == "__main__":
    main()
