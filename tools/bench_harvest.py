"""
Harvest-throughput benchmark (docs/harvest_speedup_plan.md §Benchmark protocol).

Times the teacher's batched decode loop — the harvest's inner engine — across
lever configurations, on one teacher load:

  A  stock cache + CPU sampling          (the v1/v2 baseline path)
  B  stock cache + on-device sampling    (lever 3 alone)
  C  prealloc    + on-device, batch 24   (lever 2, iso-batch)
  D  prealloc    + on-device, batch 32   (the memory headroom claim)
  E  prealloc    + on-device, batch 40   (stretch)

The prealloc equivalence gate runs FIRST and is blocking for C–E. Configs run
in ascending memory order and report incrementally, so an OOM at the stretch
config can't destroy earlier results. Raw generated tok/s (not accepted — the
filters are data-dependent and identical across configs) + peak device memory.

    python -m tools.bench_harvest --device xpu:0 --trust-remote-code
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mythouro.training_utils import load_distillation_teacher  # noqa: E402
from tools.gen_teacher_corpus import _generate_xpu_safe        # noqa: E402
from tools.prealloc_ut_cache import (                          # noqa: E402
    make_prealloc_cache,
    validate_cache_equivalence,
)


def _peak_mem_gb(device: str) -> float:
    try:
        if device.startswith("xpu"):
            return torch.xpu.max_memory_allocated() / 1e9
        return torch.cuda.max_memory_allocated() / 1e9
    except Exception:                                           # noqa: BLE001
        return float("nan")


def _reset_peak(device: str) -> None:
    try:
        if device.startswith("xpu"):
            torch.xpu.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                           # noqa: BLE001
        pass


def bench(teacher, device: str, *, batch: int, gen_tokens: int, seed_len: int,
          cpu_sampling: bool, cache_factory, warm_tokens: int = 16) -> dict:
    ids = torch.randint(1000, 40000, (batch, seed_len), device=device)
    # Warm (JIT/allocator), unmeasured.
    _generate_xpu_safe(teacher, ids, max_new=warm_tokens, temperature=0.9,
                       top_p=0.95, cpu_sampling=cpu_sampling,
                       cache_factory=cache_factory)
    _reset_peak(device)
    t0 = time.perf_counter()
    _generate_xpu_safe(teacher, ids, max_new=gen_tokens, temperature=0.9,
                       top_p=0.95, cpu_sampling=cpu_sampling,
                       cache_factory=cache_factory)
    dt = time.perf_counter() - t0
    return {"tok_s": batch * gen_tokens / dt, "dt": dt,
            "peak_gb": _peak_mem_gb(device)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--teacher-id", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed-len", type=int, default=48)
    p.add_argument("--gen-tokens", type=int, default=128,
                   help="Measured decode steps per config (short is honest for "
                        "prealloc peaks: buffers allocate at full size up "
                        "front; timing is steady-state either way).")
    p.add_argument("--full-len", type=int, default=768,
                   help="max_len the prealloc buffers are sized for (the real "
                        "harvest length) — this is what the memory verdict is "
                        "about.")
    p.add_argument("--batches", type=int, nargs="+", default=[24, 32, 40])
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.teacher_id, trust_remote_code=args.trust_remote_code)
    teacher = load_distillation_teacher(
        args.teacher_id, student_vocab_size=tok.vocab_size,
        device=args.device, dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code)
    if teacher is None:
        raise SystemExit("teacher failed to load")

    total_len = args.seed_len + args.full_len + 8

    # REAL-TEXT probe: random-token gibberish yields flat, near-tie
    # distributions whose greedy argmax flips on bf16 noise — a flaky gate
    # (observed 2026-07-23: stock gate false-failed at step 7 on randint and
    # silently benched the uncached path). Real text = peaked distributions =
    # the same conditions the harvest itself validates under.
    probe = tok("The treatment for a bacterial infection usually involves a",
                return_tensors="pt")["input_ids"].to(args.device)

    print("== stock-cache gate (pins the cached engine for this process) ==",
          flush=True)
    from mythouro.training_utils import _teacher_cache_usable
    if not _teacher_cache_usable(teacher, probe):
        raise SystemExit(
            "STOCK cache gate failed on a real-text probe — the cached engine "
            "is off, so this benchmark would measure the wrong path. Aborting.")
    print("stock gate: PASSED — cached engine ON", flush=True)

    print("== prealloc equivalence gate (blocking for prealloc configs) ==",
          flush=True)
    gate_ok = validate_cache_equivalence(teacher, probe, max_len=total_len)
    print(f"prealloc gate: {'PASSED' if gate_ok else 'FAILED'}", flush=True)

    results = {}

    def run(name, **kw):
        try:
            r = bench(teacher, args.device, gen_tokens=args.gen_tokens,
                      seed_len=args.seed_len, **kw)
            results[name] = r
            print(f"{name:32} {r['tok_s']:8.1f} tok/s   "
                  f"peak {r['peak_gb']:.1f} GB   ({r['dt']:.1f}s)", flush=True)
        except Exception as exc:                                # noqa: BLE001
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name:32} FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)

    b0 = args.batches[0]
    run(f"A stock+cpu-sampling  b{b0}", batch=b0, cpu_sampling=True,
        cache_factory=None)
    run(f"B stock+device-sample b{b0}", batch=b0, cpu_sampling=False,
        cache_factory=None)
    if gate_ok:
        for b in args.batches:
            run(f"{'CDE'[min(args.batches.index(b), 2)]} "
                f"prealloc+device    b{b}", batch=b, cpu_sampling=False,
                cache_factory=lambda: make_prealloc_cache(
                    teacher, max_len=total_len))
    else:
        print("prealloc configs SKIPPED (gate failed)")

    base = results.get(f"A stock+cpu-sampling  b{b0}", {}).get("tok_s")
    if base:
        print("\n== speedups vs baseline A ==")
        for k, r in results.items():
            if "tok_s" in r:
                print(f"{k:32} {r['tok_s']/base:5.2f}×")


if __name__ == "__main__":
    main()
