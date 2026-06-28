#!/usr/bin/env python
"""
On-policy rollout probe — the de-risk gate for on-policy distillation.

Loads a (possibly collapsed) student checkpoint + the teacher, then generates
rollouts at several teacher-mix α values from a few real-text seeds, printing the
decoded text + a repetition metric. Lets you SEE whether teacher-mixed sampling
de-collapses a degenerate student — and pick the α to launch training with —
BEFORE spending any training compute. See docs/onpolicy_plan.md (phase-2 gate).

What to look for: as α rises, `top_share` should fall and `distinct1/2` should
rise (the `is is is` attractor breaking up). Pick the smallest α that yields
clearly non-degenerate, on-topic continuations — that's the α to start training
with (anneal it down later as the student de-collapses).

Example:
    python -m tools.onpolicy_rollout_probe \
        --ckpt-dir checkpoints_revkl_stable \
        --student-device cuda:0 --teacher-device cuda:2 \
        --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code
"""
from __future__ import annotations

import argparse
from collections import Counter

import torch

from mythouro import MythOuro
from mythouro.checkpointing import list_ckpts
from mythouro.tokenizer import MythOuroTokenizer
from mythouro.training_utils import generate_rollout, load_distillation_teacher
from mythouro.variants import (
    mythouro_distill_small,
    mythouro_distill_tiny,
    mythouro_distill_tiny_dense,
    mythouro_distill_xl,
)

_VARIANTS = {
    "mythouro_distill_tiny": mythouro_distill_tiny,
    "mythouro_distill_tiny_dense": mythouro_distill_tiny_dense,
    "mythouro_distill_small": mythouro_distill_small,
    "mythouro_distill_xl": mythouro_distill_xl,
}

_SEEDS = [
    # general prose
    "In the morning the weather was clear, so we decided to",
    # medical (the mission domain) — multiple registers so one sticky seed
    # doesn't define the whole domain read (avoid the n=1 trap)
    "The treatment for a bacterial infection usually involves",
    "Common symptoms of type 2 diabetes include",
    "Ibuprofen is a nonsteroidal anti-inflammatory drug used to treat",
    # code
    "def fibonacci(n):",
    # math
    "To solve the quadratic equation x^2 - 5x + 6 = 0, we",
]


def _rep_numbers(ids: "list[int]") -> "tuple[float, float, float]":
    """(distinct1, distinct2, top_share) for a generated token list."""
    if not ids:
        return (0.0, 0.0, 1.0)
    n = len(ids)
    distinct1 = len(set(ids)) / n
    bigrams = list(zip(ids, ids[1:]))
    distinct2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    _, top_cnt = Counter(ids).most_common(1)[0]
    return (distinct1, distinct2, top_cnt / n)


def _agg(xs: "list[float]") -> str:
    """mean [min-max] across samples (or just the value for a single sample)."""
    m = sum(xs) / len(xs)
    return f"{m:.2f} [{min(xs):.2f}-{max(xs):.2f}]" if len(xs) > 1 else f"{m:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="On-policy rollout probe (de-risk gate).",
    )
    p.add_argument("--ckpt-dir", default="checkpoints_revkl_stable")
    p.add_argument("--student-variant", default="mythouro_distill_tiny",
                   choices=list(_VARIANTS))
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--teacher-device", default="cuda:2")
    p.add_argument("--teacher-id", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.7])
    p.add_argument("--seeds", nargs="+", default=_SEEDS,
                   help="Prompt seeds to probe (override the defaults for ad-hoc "
                        "testing, e.g. --seeds \"A patient with chest pain should\" "
                        "\"def quicksort(arr):\").")
    p.add_argument("--rollout-len", type=int, default=96)
    p.add_argument("--seed-len", type=int, default=16)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--samples", type=int, default=3,
                   help="Rollouts per (seed, α). >1 reports mean [min-max] so a "
                        "single unlucky sample can't mislead the read.")
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--no-sandwich-norm", action="store_true",
                   help="Build WITHOUT sandwich norm (default: WITH, matching "
                        "the rev-KL-stable 6675 recipe).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    sdev = args.student_device

    tok = MythOuroTokenizer(args.tokenizer)

    cfg = _VARIANTS[args.student_variant]()
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = args.seq_len
    cfg.use_sandwich_norm = not args.no_sandwich_norm
    cfg.recurrent_state_noise = 0.0

    student = MythOuro(cfg).to(sdev)
    ckpts = list_ckpts(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoints found in {args.ckpt_dir!r}")
    # Load model weights directly — a probe doesn't need the optimizer (and the
    # saved optimizer has grouped param-groups a plain one can't accept). Mirror
    # load_checkpoint: drop the RoPE freqs buffers (sized for the saved seq_len;
    # the fresh model already has correct ones) and load strict=False.
    ckpt = torch.load(ckpts[-1], map_location=sdev, weights_only=False)
    model_state = dict(ckpt["model"])
    for key in ("freqs_cis", "freqs_cis_mla"):
        model_state.pop(key, None)
    student.load_state_dict(model_state, strict=False)
    step = ckpt.get("step", "?")
    student.eval()
    print(f"[probe] loaded {ckpts[-1]} (step {step}) on {sdev}")

    teacher = load_distillation_teacher(
        args.teacher_id, student_vocab_size=tok.vocab_size,
        device=args.teacher_device, dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )
    if teacher is None:
        raise SystemExit("teacher failed to load")
    print(f"[probe] teacher {args.teacher_id} on {args.teacher_device}\n")

    for seed_text in args.seeds:
        seed_ids = tok.encode(seed_text)[: args.seed_len]
        prompt = torch.tensor([seed_ids], device=sdev)
        print("=" * 78)
        print(f"SEED: {seed_text!r}  ({len(seed_ids)} tok)")
        for alpha in args.alphas:
            d1s, d2s, tss, example = [], [], [], None
            for _ in range(args.samples):
                roll = generate_rollout(
                    student, teacher, prompt,
                    n_loops=args.n_loops,
                    max_new_tokens=args.rollout_len,
                    teacher_mix_alpha=alpha,
                    temperature=args.temp,
                    top_k=args.top_k,
                )
                gen_ids = roll[0, len(seed_ids):].tolist()
                d1, d2, ts = _rep_numbers(gen_ids)
                d1s.append(d1); d2s.append(d2); tss.append(ts)
                if example is None:
                    example = tok.decode(gen_ids)
            print(f"\n  α={alpha:<4} | top_share {_agg(tss)} | "
                  f"distinct1 {_agg(d1s)} | distinct2 {_agg(d2s)} | n={args.samples}")
            print(f"    e.g.: {example!r}")
        print()


if __name__ == "__main__":
    main()
