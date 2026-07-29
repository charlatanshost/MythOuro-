"""
Teacher-exhaustion gauge — is there still signal left in this teacher?

`docs/teacher_data_curriculum.md` names the soft-KL floor as the most direct
Rung-3 gate:

> "The soft-KL loss on held-out text IS the student-teacher distribution gap.
>  When it stops falling and sits near its irreducible floor ... the student has
>  absorbed what this teacher can transfer."

This measures exactly that, across a series of checkpoints, on a FIXED held-out
sample so the numbers are comparable. Unlike the rollout probe it is
deterministic (no n=5 sampling noise) and it does not suffer the top_share
metric-inversion problem — it is the actual training objective, read out.

Reading the result (the decision this feeds):
  * soft-KL still FALLING          -> teacher has transferable signal left at
                                      this capacity; keep pouring tokens.
  * soft-KL FLAT + α=0.0 << α=0.7  -> capacity-limited -> GROW (curriculum Rung 2).
  * soft-KL at floor + α-gap CLOSED-> genuinely exhausted -> Rung 3, graduate
                                      the teacher (tokenizer graduation).

The α-gap half comes from `tools/onpolicy_rollout_probe.py`; this supplies the
continuous half. Both are needed — the asymmetry in the curriculum doc (gap
closed vs progress stalled) is what separates "grow" from "graduate".

⚠ STATUS 2026-07-29: WORKS AND IS STABLE — BUT ITS READING IS CONFOUNDED.
First real series, 46k→58k every 2000, 8x4x512 = 16,384 held-out tokens:

    46000 2.2451 | 48000 2.2180 | 50000 2.2100 | 52000 2.2687
    54000 2.3165 | 56000 2.2636 | 58000 2.3216
    mean 2.2634   half-to-half drift +0.0682   step-noise (rms) 0.0460

So soft-KL **RISES** ~1.5x the point-to-point scatter. Sample size (hypothesis 1
from the 07-28 smoke) is no longer the explanation — but the rise is NOT
attributable to teacher exhaustion, because two confounds survive:

  * MIX CHANGE INSIDE THE SERIES. The values FALL through 46k→50k, then jump
    +0.0587 at **52,000 and stay elevated** — and 52,000 is exactly where the
    uniform mix took effect (code 20%→33%, general 40%→34%). The one clearly
    non-random feature in the series sits on a training-distribution change.
    **Compare checkpoints within a single mix regime.**
  * THE ON-POLICY CAVEAT (unresolved, and the important one). At
    `--onpolicy-lambda 0.7` ~70% of steps train on the student's OWN rollouts,
    not offline web text. Drifting away from the teacher's web-text distribution
    is then EXPECTED, not evidence of exhaustion — this gauge may simply be
    pointed at the wrong distribution for an on-policy run.

DISCRIMINATING TEST (not yet done): measure soft-KL over **student rollouts**
instead of web text. If rollout-KL falls while web-KL rises, the on-policy caveat
is confirmed and this tool should be re-pointed. That is a code change, not a
re-run.

**Still do NOT use this alone for a grow-vs-graduate call.** Cross-check against
the α=0.0 vs α=0.7 gap (`onpolicy_rollout_probe`), which as of 58k is NOT closing
(~+0.03-0.04) — that is the capacity-limited signature, and it is the more
trustworthy of the two right now.

Usage
-----
    python -m tools.kd_exhaustion --ckpt-dir checkpoints_onpolicy_fixed \\
        --device xpu:0 --trust-remote-code --every 2000
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_checkpoint import _load_model                        # noqa: E402
from mythouro.training_utils import (                             # noqa: E402
    MixedDataset,
    load_distillation_teacher,
    teacher_logits,
)


def _step_of(path: str) -> int:
    m = re.search(r"step_0*(\d+)\.pt", os.path.basename(path))
    return int(m.group(1)) if m else -1


@torch.no_grad()
def soft_kl(student, teacher, batches, device: str, n_loops: int,
            temperature: float = 1.0) -> float:
    """
    Mean KL(teacher || student) over the held-out batches, in nats/token.

    Same shape as the training soft term (T^2-scaled KL on the full vocab), read
    in fp32 for stability. Lower = the student's distribution is closer to the
    teacher's = less left to transfer.
    """
    total, count = 0.0, 0
    for ids in batches:
        ids = ids.to(device)
        t_log = F.log_softmax(teacher_logits(teacher, ids).float() / temperature, dim=-1)
        s_out = student(ids, n_loops=n_loops)
        s_logits = s_out[0] if isinstance(s_out, (tuple, list)) else s_out
        s_log = F.log_softmax(s_logits.float() / temperature, dim=-1)
        # KL(teacher || student), summed over vocab, averaged over tokens.
        kl = (t_log.exp() * (t_log - s_log)).sum(-1)
        total += float(kl.mean()) * ids.shape[0]
        count += ids.shape[0]
    return total / max(count, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ckpt-dir", default="checkpoints_onpolicy_fixed")
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--teacher-id", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--every", type=int, default=2000,
                   help="Only measure checkpoints whose step is a multiple of "
                        "this (the milestone series). 0 = every checkpoint.")
    p.add_argument("--batches", type=int, default=8,
                   help="Held-out batches to average over. Fixed seed → the same "
                        "text for every checkpoint, so the series is comparable.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--seed", type=int, default=12345,
                   help="Held-out sample seed. Deliberately NOT the training "
                        "seed (0) so this text is unseen.")
    p.add_argument("--cache", default=None,
                   help="Path for the cached held-out sample. Default derives "
                        "from the sample params. Delete it to re-draw. Keeping "
                        "ONE cache across checkpoints/growth steps is what makes "
                        "the soft-KL series comparable.")
    args = p.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "step_*.pt")), key=_step_of)
    if args.every:
        ckpts = [c for c in ckpts if _step_of(c) % args.every == 0]
    if not ckpts:
        raise SystemExit(f"no checkpoints matched in {args.ckpt_dir}")
    print(f"measuring {len(ckpts)} checkpoints: "
          f"{', '.join(str(_step_of(c)) for c in ckpts)}", flush=True)

    # MixedDataset wants the project's tokenizer wrapper (it reaches for
    # `.tokenizer` internally), not a bare HF tokenizer.
    from mythouro.tokenizer import MythOuroTokenizer
    encoding = MythOuroTokenizer(args.teacher_id)
    tok = encoding.tokenizer
    teacher = load_distillation_teacher(
        args.teacher_id, student_vocab_size=tok.vocab_size, device=args.device,
        dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code)
    if teacher is None:
        raise SystemExit("teacher failed to load")

    # Fixed held-out batches — built once, CACHED TO DISK, reused forever.
    # Streaming these from the Hub costs minutes (fineweb-edu metadata alone),
    # and this is a diagnostic meant to be re-run at every checkpoint/rung — so
    # pay it once. Caching also GUARANTEES the series is comparable across runs
    # and across growth steps, which is the whole point of a floor measurement.
    cache = Path(args.cache) if args.cache else Path(
        f"reports/.heldout_s{args.seed}_b{args.batches}x{args.batch_size}"
        f"_L{args.seq_len}.pt")
    if cache.exists():
        batches = torch.load(cache)
        print(f"held-out: loaded cached sample {cache}", flush=True)
    else:
        print("held-out: streaming a fresh sample (one-time; then cached) …",
              flush=True)
        ds = MixedDataset(encoding, seq_len=args.seq_len, rank=0, world_size=1,
                          seed=args.seed)
        it = iter(ds)
        batches, buf = [], []
        while len(batches) < args.batches:
            x, _y = next(it)          # MixedDataset yields (inputs, targets) tensors
            buf.append(x)
            if len(buf) == args.batch_size:
                batches.append(torch.stack(buf)); buf = []
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(batches, cache)
        print(f"held-out: cached -> {cache}", flush=True)
    print(f"held-out: {len(batches)} x {batches[0].shape[0]} x "
          f"{batches[0].shape[1]} tok (seed {args.seed}, unseen)\n", flush=True)

    print(f"{'step':>8}  {'soft-KL (nats/tok)':>20}  {'Δ vs prev':>10}")
    print("-" * 44)
    prev = None
    rows = []
    for c in ckpts:
        student, _cfg, _st = _load_model(c, args.device)
        student.eval()
        kl = soft_kl(student, teacher, batches, args.device, args.n_loops)
        d = "" if prev is None else f"{kl - prev:+.4f}"
        print(f"{_step_of(c):>8}  {kl:>20.4f}  {d:>10}", flush=True)
        rows.append((_step_of(c), kl)); prev = kl
        del student

    if len(rows) >= 3:
        vals = [v for _, v in rows]
        n = len(vals)
        mean = sum(vals) / n
        # Point-to-point scatter is the honest noise floor here: adjacent
        # checkpoints are ~2000 steps apart, so a real trend should dwarf it.
        deltas = [vals[i + 1] - vals[i] for i in range(n - 1)]
        noise = (sum(d * d for d in deltas) / len(deltas)) ** 0.5
        drift = (sum(vals[n // 2:]) / len(vals[n // 2:])
                 - sum(vals[: n // 2]) / len(vals[: n // 2]))
        sign_flips = sum(1 for i in range(len(deltas) - 1)
                         if deltas[i] * deltas[i + 1] < 0)

        print(f"\n  mean {mean:.4f}   half-to-half drift {drift:+.4f}   "
              f"step-noise (rms) {noise:.4f}   sign-flips {sign_flips}/{len(deltas)-1}")

        # Only claim a trend when it clears the scatter. Earlier versions printed
        # a confident FLATTENING/GROW verdict off two half-series slopes, which
        # on the first real run (46k→58k, 2026-07-29) fired on what was plainly
        # noise + a mix-change confound. Under-claiming is the right failure mode
        # for something feeding a capital decision.
        if abs(drift) < noise:
            print("  → INCONCLUSIVE: |drift| < step-noise. This series cannot "
                  "distinguish\n    exhaustion from scatter. Do NOT read a "
                  "grow-vs-graduate decision from it.")
        elif drift < 0:
            print("  → FALLING beyond the noise: transferable signal remains at "
                  "this capacity —\n    keep pouring tokens before spending on growth.")
        else:
            print("  → RISING beyond the noise: the student is moving AWAY from the "
                  "teacher on\n    this text. Before reading that as exhaustion, rule "
                  "out (a) a training-mix\n    change inside the series, and (b) the "
                  "on-policy caveat in the module docstring\n    (at high "
                  "--onpolicy-lambda the student trains on its OWN rollouts, so "
                  "drifting\n    from the teacher's web-text distribution is expected).")
        print("  ⚠ Cross-check any reading against the α=0.0 vs α=0.7 gap from "
              "onpolicy_rollout_probe;\n    neither signal is sufficient alone.")


if __name__ == "__main__":
    main()
