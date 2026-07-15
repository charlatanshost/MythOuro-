"""
Rollout-path benchmark: legacy full-recompute vs KV-cached decode, swept
across generation batch sizes.

Companion to tools/bench_step.py (which measures the TRAIN phase). This one
measures the phase that actually gates on-policy distillation throughput —
autoregressive rollout generation — and answers the two questions that
decided the phase-5 design (docs/onpolicy_plan.md):

  1. how much does the KV cache save (O(L²) → O(L) student+teacher work)?
  2. how much does going WIDE save (latency-bound accelerators only win
     at batch — the 2026-07-12 bench showed the Max 1100 losing to a 5070
     at batch 1 and beating it ~1.6× at batch 32+)?

Student-only by default (no download). Pass --teacher-id to include the
teacher-mix path (real Ouro teacher → real numbers, needs the HF weights).

Usage
-----
    # Max 1100, student-only sweep (the standard table)
    python -m tools.bench_rollout --device xpu

    # include the real teacher (what training actually runs)
    python -m tools.bench_rollout --device xpu \\
        --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code

Read the tok/s column: generated tokens per wall-second (batch × new_tokens
/ elapsed). The 'speedup' column is cached-vs-legacy at the same batch.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__))
))

import mythouro.variants as variants                          # noqa: E402
from mythouro import device as dev                            # noqa: E402
from mythouro.main import MythOuro                            # noqa: E402
from mythouro.training_utils import (                         # noqa: E402
    _reset_teacher_cache_gate,
    generate_rollout,
    load_distillation_teacher,
)


def bench_one(
    student,
    teacher,
    *,
    device: str,
    batch: int,
    seed_len: int,
    rollout_len: int,
    n_loops: int,
    use_kv_cache: bool,
    dtype: torch.dtype,
    repeats: int = 3,
) -> float:
    """Median tok/s over `repeats` timed rollouts (1 warmup)."""
    vocab = student.cfg.vocab_size
    amp = (
        torch.autocast(device_type=dev.autocast_type(device), dtype=dtype)
        if dev.is_accelerator(device)
        else __import__("contextlib").nullcontext()
    )
    times = []
    for rep in range(repeats + 1):
        prompt = torch.randint(0, vocab, (batch, seed_len), device=device)
        dev.synchronize(device)
        t0 = time.perf_counter()
        with amp:
            generate_rollout(
                student, teacher, prompt,
                n_loops=n_loops, max_new_tokens=rollout_len,
                teacher_mix_alpha=0.25 if teacher is not None else 0.0,
                temperature=1.0, top_k=50,
                use_kv_cache=use_kv_cache,
            )
        dev.synchronize(device)
        if rep > 0:                                   # rep 0 = warmup
            times.append(time.perf_counter() - t0)
    times.sort()
    return batch * rollout_len / times[len(times) // 2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", default="mythouro_distill_tiny")
    p.add_argument("--device", default=None)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32])
    p.add_argument("--seed-len", type=int, default=24)
    p.add_argument("--rollout-len", type=int, default=96)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--teacher-id", default=None,
                   help="Optional HF id; includes teacher-mix in the bench.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--legacy-max-batch", type=int, default=32,
                   help="Skip legacy-path runs above this batch (it's slow).")
    args = p.parse_args()

    device = dev.pick_device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    student = MythOuro(getattr(variants, args.variant)()).to(device).eval()
    n_params = sum(p_.numel() for p_ in student.parameters())

    teacher = None
    if args.teacher_id:
        teacher = load_distillation_teacher(
            args.teacher_id, student_vocab_size=student.cfg.vocab_size,
            device=device, dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        if teacher is None:
            sys.exit("teacher failed to load — see log above")

    print(f"variant : {args.variant} ({n_params:,} params)   device: {device}")
    print(f"shape   : seed={args.seed_len} rollout={args.rollout_len} "
          f"n_loops={args.n_loops} dtype={args.dtype} "
          f"teacher={'yes' if teacher is not None else 'no'}")
    print(f"{'batch':>6} {'legacy tok/s':>14} {'cached tok/s':>14} {'speedup':>9}")
    for b in args.batches:
        _reset_teacher_cache_gate()
        cached = bench_one(
            student, teacher, device=device, batch=b,
            seed_len=args.seed_len, rollout_len=args.rollout_len,
            n_loops=args.n_loops, use_kv_cache=True, dtype=dtype,
        )
        if b <= args.legacy_max_batch:
            legacy = bench_one(
                student, teacher, device=device, batch=b,
                seed_len=args.seed_len, rollout_len=args.rollout_len,
                n_loops=args.n_loops, use_kv_cache=False, dtype=dtype,
            )
            print(f"{b:>6} {legacy:>14,.0f} {cached:>14,.0f} "
                  f"{cached / legacy:>8.1f}x")
        else:
            print(f"{b:>6} {'skipped':>14} {cached:>14,.0f} {'—':>9}")


if __name__ == "__main__":
    main()
