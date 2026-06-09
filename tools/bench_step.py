"""
Standalone training-step benchmark for MythOuro.

Times forward (+ optional backward) on random data — no teacher, no dataset,
no checkpoint — so it runs anywhere in seconds and measures the only number
that actually decides a hardware purchase: **achieved tokens/sec on the real
model**, not peak TFLOPS off a spec sheet.

Why this exists
---------------
Peak dense-BF16 TFLOPS (whether a GPU's tensor cores or a Xeon's AMX tiles) is
a best-case big-square-GEMM figure. MythOuro is full of *small* matmuls (MoE
experts, MLA latents, per-loop LoRA) and runs the recurrent block `n_loops`
times per token, so it sustains only a fraction of peak — and that fraction
differs by hardware. The only way to compare a candidate box (e.g. a rented
Intel Xeon Max / AMX instance) against your current card is to run the same
model on both and read tok/s. This does exactly that.

Usage
-----
    # Current card (CUDA)
    python -m tools.bench_step --variant mythouro_distill_tiny --device cuda:0

    # Rented Xeon Max / AMX box (CPU bf16 -> oneDNN uses AMX automatically)
    python -m tools.bench_step --variant mythouro_distill_tiny --device cpu

    # Dense ablation arm, forward-only, custom shape
    python -m tools.bench_step --variant mythouro_distill_tiny_dense \\
        --batch 1 --seq-len 1024 --n-loops 4 --steps 20 --no-backward

Read the `tokens/sec` line. Note it counts each *input* token once; the
recurrent block ran `n_loops` times per token, so raw matmul throughput is
~n_loops× higher than the sequence-token rate (that loop multiplier is exactly
why a compute-bound platform matters here).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__))
))

import mythouro.variants as variants  # noqa: E402
from mythouro.main import MythOuro, MythOuroConfig  # noqa: E402
from mythouro import device as dev  # noqa: E402


def run_benchmark(
    model: MythOuro,
    device: str,
    *,
    batch: int = 1,
    seq_len: int = 512,
    n_loops: Optional[int] = None,
    steps: int = 20,
    warmup: int = 3,
    backward: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Time `steps` forward(+backward) passes and return throughput metrics.

    Random token ids each step. Uses autocast in `dtype` on the given device.
    `warmup` untimed steps absorb one-off costs (cudnn autotune, AMX warmup,
    allocator growth). Returns a dict with ms/step and tokens/sec.
    """
    model = model.to(device)
    model.train(backward)
    vocab = model.cfg.vocab_size

    autocast_device = dev.autocast_type(device)   # cuda | xpu | cpu

    def _sync():
        dev.synchronize(device)

    # autocast only applies to the low-precision dtypes; fp32 runs plainly.
    if dtype == torch.float32:
        autocast_ctx = __import__("contextlib").nullcontext
    else:
        def autocast_ctx():
            return torch.autocast(device_type=autocast_device, dtype=dtype)

    def one_step():
        x = torch.randint(0, vocab, (batch, seq_len), device=device)
        with autocast_ctx():
            logits, _ = model(x, n_loops=n_loops)
            if backward:
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab).float(), x.reshape(-1)
                )
        if backward:
            model.zero_grad(set_to_none=True)
            loss.backward()

    for _ in range(warmup):
        one_step()
    _sync()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    _sync()
    elapsed = time.perf_counter() - t0

    tokens = batch * seq_len * steps
    ms_per_step = 1000.0 * elapsed / steps
    tokens_per_s = tokens / elapsed
    return {
        "ms_per_step": ms_per_step,
        "tokens_per_s": tokens_per_s,
        "elapsed_s": elapsed,
        "n_loops": n_loops or model.cfg.max_loop_iters,
    }


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", default="mythouro_distill_tiny",
                   help="Variant function name in mythouro.variants.")
    p.add_argument("--device", default=None, help="cuda:N / cpu (default: auto).")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-loops", type=int, default=None,
                   help="Recurrent depth (default: variant's max_loop_iters).")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True,
                   help="Time forward+backward (default) or --no-backward for fwd-only.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--rope-real", action="store_true",
        help="Use the real-valued (cos/sin) RoPE table instead of complex — for "
             "backends without complex-tensor op support (e.g. Intel XPU). Same "
             "math; flip this if `--device xpu` errors inside apply_rope.",
    )
    return p.parse_args(argv)


def main():
    args = _parse_args()
    device = dev.pick_device(args.device)         # explicit, else cuda:0 / xpu / cpu
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    cfg: MythOuroConfig = getattr(variants, args.variant)()
    if args.rope_real:
        cfg.rope_real = True
    model = MythOuro(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"variant : {args.variant}  ({n_params:,} params)")
    print(f"device  : {device}   dtype: {args.dtype}   backward: {args.backward}")
    print(f"shape   : batch={args.batch} seq_len={args.seq_len} "
          f"n_loops={args.n_loops or cfg.max_loop_iters}  "
          f"(recurrent block runs n_loops x per token)")
    print(f"steps   : {args.steps} timed (+{args.warmup} warmup)")
    print("running...")

    r = run_benchmark(
        model, device,
        batch=args.batch, seq_len=args.seq_len, n_loops=args.n_loops,
        steps=args.steps, warmup=args.warmup, backward=args.backward, dtype=dtype,
    )

    print("-" * 56)
    print(f"  ms/step      : {r['ms_per_step']:.1f}")
    print(f"  tokens/sec   : {r['tokens_per_s']:,.0f}   "
          f"(input tokens; raw matmul ~{r['n_loops']}x this)")
    print("-" * 56)
    print("Compare this tokens/sec across machines — it's the real number, "
          "not peak TFLOPS.")


if __name__ == "__main__":
    main()
