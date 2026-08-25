"""
Benchmark ON-POLICY ROLLOUT GENERATION — the cost that scales with --rollout-len.

`tools/bench_step.py` times a training step (forward+backward on random data) and
has no rollout path at all. But the cost that decides whether --rollout-len can be
raised lives in `generate_rollout`, and it scales badly: rollouts are hard-pinned
UNCACHED, so generating n tokens re-runs the forward over the whole prefix each
step — O(n^2), not O(n).

WHY UNCACHED (2026-07-16, measured). Cached vs uncached student logits on identical
weights: max |Δlogit| 5.5, KL up to 0.95 nats. Reading mythouro/main.py, both
early-exit paths carry `kv_cache is None` as a condition — so CACHED decode runs
FULL depth while UNCACHED early-exits at ACT depth, which every probe measures at
2.00/4. That 2x depth gap is the distribution shift, and training on cached
rollouts produced the 2026-07-16 regression.

WHAT THIS MEASURES, on the real card rather than from a complexity argument:
  * uncached wall-time and tok/s at each --rollout-len (the training path)
  * cached, for reference — how much the O(n) path would buy IF the depth gap
    were ever resolved. NOT a recommendation to train on it.
  * observed mean halt depth, since that is what sets the uncached cost

    python -m tools.bench_rollout -c checkpoints_codemix/step_0149500.pt \\
        --device xpu:0 --lens 64,128,256 --batch 32
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inspect_checkpoint import _load_model                       # noqa: E402
from mythouro.training_utils import (                            # noqa: E402
    generate_rollout, load_distillation_teacher)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--lens", default="64,128,256",
                   help="Comma-separated --rollout-len values to time.")
    p.add_argument("--one", action="store_true",
                   help="Time ONE (len,batch) config and print a single TSV line. "
                        "A GPU page fault aborts the process, so the sweep driver "
                        "runs each config in its own subprocess and keeps the "
                        "results that survived.")
    p.add_argument("--batch", type=int, default=32, help="--rollout-batch.")
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--teacher-id", default="",
                   help="⚠ WITHOUT THIS THE BENCH UNDERSTATES BADLY. Training runs "
                        "--teacher-mix-alpha 0.45, so the 2.6B teacher forwards at "
                        "EVERY generated token too. Measuring the student alone "
                        "(the 2026-08-24 mistake) gave 8.5 s/step against an "
                        "actual 28.9 — a 3.4x miss that cost most of a night.")
    p.add_argument("--teacher-mix-alpha", type=float, default=0.45,
                   help="Match the training recipe. 0.0 = student only.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--cached-too", action="store_true",
                   help="Also time the cached path, for reference only.")
    a = p.parse_args()

    model, cfg, step = _load_model(a.checkpoint, a.device)
    model.eval()
    teacher = None
    if a.teacher_id:
        teacher = load_distillation_teacher(
            a.teacher_id, student_vocab_size=getattr(cfg, "vocab_size", 49152),
            device=a.device, dtype=torch.bfloat16,
            trust_remote_code=a.trust_remote_code)
        if teacher is None:
            raise SystemExit("teacher failed to load")
    elif a.teacher_mix_alpha > 0:
        print("\n  ⚠ no --teacher-id given: measuring the STUDENT ALONE. Training "
              "mixes a 2.6B teacher at every token, so this UNDERSTATES the real "
              "cost — by 3.4x when this was last done wrong.\n")
    lens = [int(x) for x in a.lens.split(",")]
    vocab = getattr(cfg, "vocab_size", 49152)
    g = torch.Generator(device="cpu").manual_seed(1234)
    prompt = torch.randint(0, vocab, (a.batch, a.prompt_len), generator=g).to(a.device)

    def sync():
        if a.device.startswith("xpu"):
            torch.xpu.synchronize()
        elif a.device.startswith("cuda"):
            torch.cuda.synchronize()

    print(f"\n  step {step} | batch {a.batch} "
          f"| prompt {a.prompt_len} | n_loops {a.n_loops}")
    if a.one:
        L = lens[0]
        ts = []
        for _ in range(a.reps):
            sync(); t0 = time.perf_counter()
            with torch.no_grad():
                generate_rollout(model, teacher, prompt, n_loops=a.n_loops,
                                 max_new_tokens=L,
                                 teacher_mix_alpha=a.teacher_mix_alpha,
                                 temperature=1.0, top_k=50, use_kv_cache=False)
            sync(); ts.append(time.perf_counter() - t0)
        best = min(ts)
        print(f"RESULT\t{L}\t{a.batch}\t{best:.3f}\t{a.batch*L/best:.1f}", flush=True)
        return

    print(f"\n  {'len':>6}{'cache':>8}{'sec':>9}{'tok/s':>10}{'vs 64':>8}")
    base = {}
    for cached in ([False, True] if a.cached_too else [False]):
        for L in lens:
            ts = []
            for _ in range(a.reps):
                sync(); t0 = time.perf_counter()
                with torch.no_grad():
                    generate_rollout(model, teacher, prompt, n_loops=a.n_loops,
                                     max_new_tokens=L,
                                     teacher_mix_alpha=a.teacher_mix_alpha,
                                     temperature=1.0, top_k=50,
                                     use_kv_cache=cached)
                sync(); ts.append(time.perf_counter() - t0)
            best = min(ts)
            tag = "cached" if cached else "uncached"
            if not cached and L == lens[0]:
                base["t"] = best
            rel = best / base["t"] if "t" in base else float("nan")
            print(f"  {L:>6}{tag:>8}{best:>9.2f}{a.batch*L/best:>10.0f}{rel:>7.1f}x")

    print(f"\n  A leg is 6,000 steps; rollouts regenerate every "
          f"--rollout-reuse steps (8 in the current recipe), so multiply the "
          f"per-rollout time by 6000/8 = 750 to get the leg's rollout cost.")


if __name__ == "__main__":
    main()
