"""
Does a learnable depth policy EXIST? Measure the best-exit target before building
anything that trains against it.

WHY THIS COMES FIRST. The plan (docs/ideas.md "MoDr", docs/growth_design.md
"loop-loss supervision") is to supervise the halt decision against a BEST-EXIT
target — the loop with the lowest per-loop CE — because the uncertainty head is
measurably the wrong signal: on 2026-07-31 `BestOfTrajectoryGenerator` selected
loop 1 or 2 for 98% of tokens and drove the prompt-copy rate UP 41.2% -> 50.0%.
Argmin-over-uncertainty is an echo-seeking objective (copying a token just seen
is the most confident prediction available).

But a policy is only learnable if the target VARIES. Three questions, all
answered here, none requiring a training run:

  1. Does the best exit differ per token, or is it the same loop everywhere?
     Constant target => nothing to learn => the whole direction is moot.
  2. How much CE is on the table? Compare CE at the ORACLE best exit against CE
     at the final loop (what training emits) and at the loop the uncertainty
     head would pick. That gap is the ceiling on what a perfect depth router
     could buy — if it is ~0, do not build the router.
  3. Does the uncertainty head already agree with the CE-optimal exit? Agreement
     is what a supervised halt branch would have to improve on.

⚠ A DOCUMENTED CONTRADICTION THIS RESOLVES. docs/ideas.md records from the
2026-06-08 forced-depth probe that "some prompts bottom out at loop 7 (2x the
trained depth), others at loop 0" and that "uncertainty keeps dropping
monotonically to loop 7". Today's best-of-trajectory run never selected past
loop 3. Those disagree — different checkpoint (70,500 steps later), different
decode (greedy then, T=0.4 sampled now), different prompts. This measures it on
current weights against gold tokens, which is neither of those conditions and is
the one that matters for supervision.

Uses per-loop CE against the GOLD next token, which is the target the docs
mandate ("per-loop CE, NOT uncertainty-argmin" — tools/per_loop_calibration.py
prints this as its verdict). No teacher forward needed, so it is cheap.

Usage
-----
    python -m tools.best_exit_probe -c checkpoints_reuse8/step_0070500.pt \\
        --device xpu:0 --n-loops 8 --chunks 12 --seq-len 256
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_checkpoint import _load_model                      # noqa: E402


@torch.no_grad()
def probe(model, tok, device: str, n_loops: int, chunks: int,
          seq_len: int, mix_ratios: "dict | None" = None) -> dict:
    from mythouro.training_utils import MixedDataset

    # rank/world_size are REQUIRED positionally — matches training/distill.py
    # (`MixedDataset(encoding, seq_len, rank=0, world_size=1, ...)`).
    #
    # `mix_ratios` is how --by-domain isolates one corpus: passing
    # {"math": 1.0} yields a PURE math stream. Done this way on purpose rather
    # than by labelling chunks inside MixedDataset, for two reasons. First, the
    # token buffer is shared across sources — `buf` carries the tail of one
    # document into the next — so a single chunk can legitimately contain
    # tokens from two corpora and there is no honest single label for it.
    # Second, MixedDataset is the live training data path that distill.py and
    # sft.py both run on; a measurement tool should not reach into it.
    ds = MixedDataset(tok, seq_len, rank=0, world_size=1, seed=0,
                      mix_ratios=mix_ratios)
    it = iter(ds)

    best_hist: Counter = Counter()
    head_hist: Counter = Counter()
    agree = total = 0
    ce_best = ce_final = ce_head = ce_rand = 0.0
    per_loop_ce_sum = [0.0] * n_loops

    for _ in range(chunks):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        # (B,T,K,V) and (B,T,K). force_full_depth so every loop is scored — the
        # point is the counterfactual, not what ACT would have run.
        logits_traj, unc_traj = model.forward_trajectory(
            x, n_loops=n_loops, force_full_depth=True,
        )
        B, T, K, V = logits_traj.shape

        # Per-loop CE against the gold next token, per position. Computed loop by
        # loop so we never hold (B,T,K,V) in float32 — at training shape that
        # tensor alone is ~3.2GB in bf16 (vocab 49,152).
        ce = torch.empty(B, T, K, device=device, dtype=torch.float32)
        for k in range(K):
            ce[:, :, k] = F.cross_entropy(
                logits_traj[:, :, k, :].reshape(-1, V).float(),
                y.reshape(-1), reduction="none",
            ).view(B, T)

        best_k = ce.argmin(dim=-1)                       # (B,T) the ORACLE exit
        head_k = unc_traj.argmin(dim=-1)                 # (B,T) what the head picks

        best_hist.update(best_k.flatten().tolist())
        head_hist.update(head_k.flatten().tolist())
        agree += int((best_k == head_k).sum())
        total += best_k.numel()

        # RANDOM-EXIT CONTROL. `ce_oracle_best` is a MINIMUM over K noisy
        # per-loop CEs, and min-of-K is biased low by construction: a model with
        # NO genuine per-token depth preference still shows positive "headroom"
        # purely from that selection. Without this control the headroom number
        # cannot distinguish real depth structure from noise — and it would look
        # roughly equal across domains either way, which is exactly the
        # ambiguity the 2026-08-10 --by-domain run hit.
        #
        # A uniformly random exit is the honest floor: it captures none of the
        # selection bias and none of the signal. Interpretation:
        #   oracle much better than random  => real per-token structure exists
        #   oracle ~ random                 => the "headroom" is selection noise
        rand_k = torch.randint(0, K, best_k.shape, device=ce.device)
        ce_rand += float(ce.gather(-1, rand_k.unsqueeze(-1)).mean())
        ce_best += float(ce.gather(-1, best_k.unsqueeze(-1)).mean())
        ce_head += float(ce.gather(-1, head_k.unsqueeze(-1)).mean())
        ce_final += float(ce[:, :, K - 1].mean())
        for k in range(K):
            per_loop_ce_sum[k] += float(ce[:, :, k].mean())

    n = max(1, min(chunks, max(1, total // (seq_len or 1))))
    return {
        "n_loops": n_loops, "positions": total,
        "best_exit_hist": dict(sorted(best_hist.items())),
        "head_exit_hist": dict(sorted(head_hist.items())),
        "agreement": agree / total if total else 0.0,
        "ce_oracle_best": ce_best / n,
        "ce_random_exit": ce_rand / n,
        "ce_head_choice": ce_head / n,
        "ce_final_loop": ce_final / n,
        "ce_per_loop": [c / n for c in per_loop_ce_sum],
    }


def _by_domain(model, enc, args, step: int) -> None:
    """
    Per-corpus headroom — is depth benefit a property of the SUBJECT?

    The aggregate probe answers "does a depth policy exist"; it cannot answer
    "for whom". Growing depth for the topics that benefit (docs/roadmap.md)
    presumes the benefit VARIES by topic, and that presumption has never been
    measured. This measures it, read-only, no training.

    READ THE SPREAD, NOT THE LEVEL. Absolute CE differs across corpora for
    reasons that have nothing to do with depth — code is more predictable than
    prose, so its CE is lower everywhere. What matters is whether the HEADROOM
    (CE at trained depth minus CE at the oracle exit) and the best-exit
    DISTRIBUTION differ. Similar headroom everywhere => one global depth policy
    is the right shape and per-subject depth is a dead end.
    """
    from mythouro.training_utils import _MIX_RATIOS

    domains = args.domains or list(_MIX_RATIOS)
    rows, failed = [], []
    for d in domains:
        if d not in _MIX_RATIOS and not args.domains:
            continue
        print(f"  probing {d} ...", flush=True)
        try:
            r = probe(model, enc, args.device, args.n_loops, args.chunks,
                      args.seq_len, mix_ratios={d: 1.0})
        except Exception as exc:                                  # noqa: BLE001
            # One unreachable corpus must not cost the other five — the same
            # non-fatal rule run_full_validation.sh uses.
            print(f"    FAILED: {exc}")
            failed.append(d)
            continue
        if not r["positions"]:
            print("    no positions — corpus yielded nothing; skipping")
            failed.append(d)
            continue
        ti = min(args.trained_loops, len(r["ce_per_loop"])) - 1
        r["domain"] = d
        r["ce_trained_depth"] = r["ce_per_loop"][ti]
        r["headroom_nats"] = r["ce_per_loop"][ti] - r["ce_oracle_best"]
        rows.append(r)

    if not rows:
        print("\n  no domain produced data — nothing to compare.")
        return

    print(f"\n  {'domain':16} {'CE@trained':>11} {'CE@oracle':>10} {'CE@random':>10} "
          f"{'headroom':>9} {'vs RANDOM':>10} {'agr':>5}")
    for r in rows:
        pct = 100 * r["headroom_nats"] / max(r["ce_trained_depth"], 1e-9)
        r["oracle_vs_random"] = r["ce_random_exit"] - r["ce_oracle_best"]
        print(f"  {r['domain']:16} {r['ce_trained_depth']:11.4f} "
              f"{r['ce_oracle_best']:10.4f} {r['ce_random_exit']:10.4f} "
              f"{r['headroom_nats']:8.4f} ({pct:4.1f}%) "
              f"{r['oracle_vs_random']:9.4f} {100*r['agreement']:4.0f}%")

    # The control decides whether the headroom column means anything AT ALL,
    # so it is checked BEFORE the per-domain spread. A spread computed over
    # numbers that are mostly selection bias is a spread in the noise.
    sig = [r["oracle_vs_random"] for r in rows]
    if max(sig) < 0.02:
        print("\n  ⚠ VERDICT (control failed): the oracle barely beats a RANDOM "
              "exit in ANY domain. The headroom column is dominated by min-of-K "
              "selection bias, so neither the levels NOR the per-domain spread "
              "support a depth policy. Do not grow depth on this evidence.")
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"step": step, "by_domain": rows, "failed": failed}, indent=2))
            print(f"  wrote {args.json}")
        return

    hr = [r["headroom_nats"] for r in rows]
    spread = max(hr) - min(hr)
    rel = spread / max(max(hr), 1e-9)
    hi = max(rows, key=lambda r: r["headroom_nats"])["domain"]
    lo = min(rows, key=lambda r: r["headroom_nats"])["domain"]
    print(f"\n  headroom spread: {spread:.4f} nats "
          f"({hi} highest, {lo} lowest) = {100*rel:.0f}% of the largest")
    if rel < 0.25:
        print("  VERDICT: headroom is essentially UNIFORM across subjects. "
              "Per-subject depth specialisation is NOT supported — one global "
              "depth policy is the right shape. Do not grow depth for topics.")
    else:
        print(f"  VERDICT: headroom VARIES by subject ({hi} wants materially "
              f"more depth than {lo}). Per-subject depth has a target worth "
              f"training toward — but note this is teacher-forced CE, not "
              f"generation quality.")
    if failed:
        print(f"  (no data: {', '.join(failed)})")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"step": step, "by_domain": rows, "failed": failed}, indent=2))
        print(f"  wrote {args.json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--n-loops", type=int, default=8,
                   help="Scored depth. Above the trained 4 is deliberate: the "
                        "question is whether USEFUL depth exists beyond where "
                        "ACT stops, not what ACT does.")
    p.add_argument("--trained-loops", type=int, default=4,
                   help="The depth training actually emits from (cfg."
                        "max_loop_iters). Headroom is measured against THIS, not "
                        "against the last SCORED loop: with --n-loops 8 the final "
                        "loop is off-distribution and bad, which inflates the gap "
                        "and flatters a depth router that would never be compared "
                        "against it in production.")
    p.add_argument("--chunks", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--json", default=None)
    p.add_argument("--by-domain", action="store_true",
                   help="Run the probe SEPARATELY per corpus and compare the "
                        "headroom. THE QUESTION THIS EXISTS FOR: the aggregate "
                        "number cannot say whether some SUBJECTS want more "
                        "depth than others, which is the whole premise of "
                        "training loops past 4 for the topics that benefit "
                        "(docs/roadmap.md). If every domain reports the same "
                        "headroom, depth specialisation is disconfirmed before "
                        "a single training step is spent on it.")
    p.add_argument("--domains", nargs="+", default=None,
                   help="Subset of corpora for --by-domain (default: all of "
                        "_MIX_RATIOS). Each runs as a PURE stream.")
    args = p.parse_args()

    from mythouro.tokenizer import MythOuroTokenizer
    enc = MythOuroTokenizer(args.tokenizer)
    model, _cfg, step = _load_model(args.checkpoint, args.device)
    model.eval()
    print(f"checkpoint step {step} | n_loops {args.n_loops} | "
          f"{args.chunks} chunks x {args.seq_len} tok\n")

    if args.by_domain:
        _by_domain(model, enc, args, step)
        return

    r = probe(model, enc, args.device, args.n_loops, args.chunks, args.seq_len)
    tot = r["positions"]

    print(f"  positions scored: {tot:,}\n")
    print("  ORACLE best exit (argmin per-loop CE) — does the target VARY?")
    for k, v in r["best_exit_hist"].items():
        print(f"    loop {k}: {v:>7,}  {100 * v / tot:5.1f}%  {'#' * (v * 40 // tot)}")
    print("\n  What the UNCERTAINTY HEAD picks:")
    for k, v in r["head_exit_hist"].items():
        print(f"    loop {k}: {v:>7,}  {100 * v / tot:5.1f}%  {'#' * (v * 40 // tot)}")
    print(f"\n  head agrees with oracle: {100 * r['agreement']:.1f}%")

    print("\n  CE per loop:", "  ".join(f"{k}:{c:.3f}"
                                        for k, c in enumerate(r["ce_per_loop"])))
    ti = min(args.trained_loops, len(r["ce_per_loop"])) - 1
    ce_trained = r["ce_per_loop"][ti]
    gap = ce_trained - r["ce_oracle_best"]
    r["ce_trained_depth"] = ce_trained
    r["headroom_nats"] = gap
    print(f"\n  CE at loop {ti} (trained depth) {ce_trained:.4f}   <- what "
          f"training emits")
    print(f"  CE at last scored loop  {r['ce_final_loop']:.4f}   (off-distribution "
          f"if > trained depth — NOT the baseline)")
    print(f"  CE at head choice       {r['ce_head_choice']:.4f}")
    print(f"  CE at RANDOM exit       {r['ce_random_exit']:.4f}   <- the control: "
          f"min-of-K selection bias with NO signal")
    print(f"  CE at ORACLE exit       {r['ce_oracle_best']:.4f}")
    _signal = r["ce_random_exit"] - r["ce_oracle_best"]
    _bias = ce_trained - r["ce_random_exit"]
    print(f"  oracle beats random by  {_signal:.4f} nats   "
          f"(random vs trained depth: {_bias:+.4f})")
    if _signal < 0.02:
        print("  ⚠ the oracle barely beats a RANDOM exit — the headroom above is "
              "mostly min-of-K selection bias, NOT learnable structure.")
    print(f"  headroom vs trained depth: {gap:.4f} nats "
          f"({100 * gap / max(ce_trained, 1e-9):.1f}%)")
    if r["ce_head_choice"] > ce_trained:
        print(f"  ⚠ the head's selection is WORSE than a fixed depth of {ti} — "
              f"the current selector is actively harmful.")

    n_used = len(r["best_exit_hist"])
    print()
    if n_used <= 1:
        print("  VERDICT: best exit is CONSTANT — no policy to learn. "
              "Do not build the depth router.")
    elif gap < 0.02:
        print("  VERDICT: target varies but the CE headroom is negligible — "
              "a perfect router would buy almost nothing.")
    else:
        print(f"  VERDICT: target varies across {n_used} depths with "
              f"{gap:.3f} nats of headroom — a learnable policy EXISTS, and the "
              f"head currently captures only {100 * r['agreement']:.0f}% of it.")

    if args.json:
        Path(args.json).write_text(json.dumps({"step": step, **r}, indent=2))
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
