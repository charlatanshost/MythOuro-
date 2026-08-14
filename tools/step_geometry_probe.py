"""
Does the LATENT GEOMETRY predict the best exit better than the trained head does?

WHY THIS EXISTS. Three efforts have gone into a LEARNED halting signal and all
three failed: ACT thresholding settles at depth 2.0 while loop 3 is the most
accurate; `BestOfTrajectoryGenerator` drove the prompt-copy rate UP 41.2% -> 50.0%
because argmin-over-uncertainty is an echo-seeking objective; and
`train_depth_policy.py` improved per-loop CE while making the task worse. The head
is measurably overconfident exactly where it is consumed — ECE 0.288 at loop 0,
predicting 1.68% error where it makes 30.4%.

Three independent 2026 sources reach for the same alternative — halt on the loop's
OWN DYNAMICS rather than a trained confidence signal:

  * Two-Scale Latent Dynamics (arXiv 2509.23314): early exit on SECOND-ORDER
    DIFFERENCES IN STEP SIZE, reported to beat KL-based early exit on
    performance, stability AND efficiency. Its picture: within a looped block,
    updates are small-scale refinements; across blocks, states undergo
    larger-scale drift.
  * FPRM (arXiv 2606.18206): fixed-point convergence as the halt.
  * duongtrongnguyen123/recurrent-depth-ttc: a HARDCODED halt rule, no learned
    stopping head at all.

MEASURE BEFORE BUILDING. This does NOT wire a halt into decode. It asks the only
question that justifies doing so: on real data, does a geometric rule agree with
the ORACLE best exit (argmin per-loop CE against the gold token) more often than
the UncertaintyHead does? The head's agreement is 19-28% depending on corpus,
against 12.5% chance at 8 loops. If geometry cannot beat that, a decode-time rule
is not worth building and this is a cheap negative.

⚠ CAVEAT ON WHAT IS MEASURED. `forward_trajectory(return_states=True)` returns
the POST-CODA, POST-NORM state per loop — what `head` and `uncertainty` actually
consume. 2509.23314 studies the raw recurrent iterates. Post-coda states are the
defensible first measurement (they determine the output and need no new model
code), but a null result here does not fully clear the idea: the geometry may
live in the pre-coda trajectory. Stated so a negative is not over-read.

Usage
-----
    python -m tools.step_geometry_probe -c checkpoints_newmix/step_0108471.pt \\
        --device xpu:0 --n-loops 8 --chunks 12 --seq-len 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_checkpoint import _load_model                      # noqa: E402


def _rules(states: torch.Tensor) -> dict:
    """
    Geometric exit predictions from the per-loop state trajectory.

    states -- (B, T, K, D) post-coda hidden state per loop.

    Returns a dict of rule-name -> (B, T) predicted exit index. Every rule uses
    ONLY the trajectory: no trained head, no gold token.
    """
    # Relative step size between consecutive loops: s_k = |h_k - h_{k-1}| / |h_{k-1}|
    # Relative rather than absolute so the scale of the state cannot dominate.
    d = (states[:, :, 1:, :] - states[:, :, :-1, :]).norm(dim=-1)     # (B,T,K-1)
    n = states[:, :, :-1, :].norm(dim=-1).clamp_min(1e-6)
    s = d / n                                                         # (B,T,K-1)
    K = states.shape[2]
    out = {}

    # 1. CONVERGENCE: first loop whose relative step falls below a threshold —
    #    the fixed-point reading (FPRM). +1 because s_k describes the move INTO
    #    loop k+1.
    for thr in (0.05, 0.02, 0.01):
        below = s < thr
        idx = torch.where(below.any(-1), below.float().argmax(-1) + 1,
                          torch.full_like(below[..., 0], K - 1, dtype=torch.long))
        out[f"converge<{thr}"] = idx.clamp(max=K - 1)

    # 2. SECOND-ORDER DIFFERENCE (2509.23314): the step sizes stop shrinking —
    #    refinement has flattened out, so further loops add drift, not detail.
    if s.shape[-1] >= 2:
        d2 = s[:, :, 1:] - s[:, :, :-1]                               # (B,T,K-2)
        flat = d2 > 0                       # step grew again = no longer refining
        idx = torch.where(flat.any(-1), flat.float().argmax(-1) + 1,
                          torch.full_like(flat[..., 0], K - 1, dtype=torch.long))
        out["second_diff"] = idx.clamp(max=K - 1)

    # 3. MIN-STEP: the loop after the SMALLEST move — the most converged point.
    out["min_step"] = (s.argmin(-1) + 1).clamp(max=K - 1)
    return out


@torch.no_grad()
def probe(model, tok, device: str, n_loops: int, chunks: int, seq_len: int) -> dict:
    from mythouro.training_utils import MixedDataset

    ds = MixedDataset(tok, seq_len, rank=0, world_size=1, seed=0)
    it = iter(ds)
    agree: dict = {}
    ce_of: dict = {}
    head_agree = total = 0
    ce_oracle = ce_head = ce_final = 0.0
    per_loop = [0.0] * n_loops
    nb = 0

    for _ in range(chunks):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        _, unc_traj, states = model.forward_trajectory(
            x, n_loops=n_loops, force_full_depth=True, return_states=True)
        B, T, K, _ = states.shape

        # Per-loop CE against the gold token, one loop at a time — (B,T,K,V) is
        # ~3.2GB in bf16 at training shape, so it is never materialised.
        ce = torch.empty(B, T, K, device=device, dtype=torch.float32)
        for k in range(K):
            lg = model.head(states[:, :, k, :])
            ce[:, :, k] = F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1),
                reduction="none").view(B, T)

        best_k = ce.argmin(dim=-1)                    # ORACLE
        head_k = unc_traj.argmin(dim=-1)              # what the trained head picks
        head_agree += int((best_k == head_k).sum())
        total += best_k.numel()
        ce_oracle += float(ce.gather(-1, best_k.unsqueeze(-1)).mean())
        ce_head += float(ce.gather(-1, head_k.unsqueeze(-1)).mean())
        ce_final += float(ce[:, :, K - 1].mean())
        for k in range(K):
            per_loop[k] += float(ce[:, :, k].mean())

        for name, idx in _rules(states).items():
            idx = idx.clamp(max=K - 1)
            agree[name] = agree.get(name, 0) + int((best_k == idx).sum())
            ce_of[name] = ce_of.get(name, 0.0) + float(
                ce.gather(-1, idx.unsqueeze(-1)).mean())
        nb += 1

    nb = max(nb, 1)
    return {
        "n_loops": n_loops, "positions": total,
        "chance": 1.0 / n_loops,
        "head_agreement": head_agree / total if total else 0.0,
        "ce_oracle": ce_oracle / nb,
        "ce_head": ce_head / nb,
        "ce_final_loop": ce_final / nb,
        "ce_per_loop": [c / nb for c in per_loop],
        "rules": {k: {"agreement": agree[k] / total if total else 0.0,
                      "ce": ce_of[k] / nb} for k in agree},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--n-loops", type=int, default=8)
    p.add_argument("--trained-loops", type=int, default=4,
                   help="Depth training emits from. CE at THIS loop is the "
                        "baseline any exit rule must beat — not the last SCORED "
                        "loop, which is off-distribution and flatters a router.")
    p.add_argument("--chunks", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    from mythouro.tokenizer import MythOuroTokenizer
    enc = MythOuroTokenizer(args.tokenizer)
    model, _cfg, step = _load_model(args.checkpoint, args.device)
    model.eval()
    r = probe(model, enc, args.device, args.n_loops, args.chunks, args.seq_len)

    print(f"checkpoint step {step} | {r['positions']:,} positions | "
          f"{args.n_loops} loops (chance {100*r['chance']:.1f}%)\n")
    print(f"  {'selector':18} {'agrees w/ oracle':>17} {'CE':>9}")
    print(f"  {'ORACLE (argmin CE)':18} {'100.0%':>17} {r['ce_oracle']:9.4f}")
    print(f"  {'UncertaintyHead':18} {100*r['head_agreement']:16.1f}% {r['ce_head']:9.4f}")
    for name, d in sorted(r["rules"].items(), key=lambda kv: -kv[1]["agreement"]):
        print(f"  {name:18} {100*d['agreement']:16.1f}% {d['ce']:9.4f}")
    ti = min(args.trained_loops, len(r["ce_per_loop"])) - 1
    ce_trained = r["ce_per_loop"][ti]
    r["ce_trained_depth"] = ce_trained
    print(f"\n  CE per loop: " + "  ".join(f"{k}:{c:.3f}"
                                           for k, c in enumerate(r["ce_per_loop"])))
    print(f"  fixed depth {ti} (WHAT INFERENCE EMITS): {ce_trained:.4f}  <- the number to beat")
    print(f"  last scored loop {len(r['ce_per_loop'])-1}: {r['ce_final_loop']:.4f}  "
          f"(off-distribution above the trained depth — NOT the baseline)")

    # RANK BY CE, NOT AGREEMENT. A rule can agree with the argmin less often and
    # still produce far lower CE, because when it disagrees it picks a
    # nearly-as-good loop. Agreement counts exact matches; CE measures what the
    # model actually emits. Ranking by agreement on 2026-08-14 declared a rule
    # dead (14.9% vs the head's 23.3%) whose CE was 0.309 against the head's
    # 0.492 — the opposite conclusion.
    best = min(r["rules"].items(), key=lambda kv: kv[1]["ce"])
    print()
    if best[1]["ce"] >= ce_trained:
        print(f"  VERDICT: no geometric rule beats a FIXED trained depth "
              f"({best[0]} {best[1]['ce']:.4f} vs {ce_trained:.4f}). Halting on "
              f"latent dynamics buys nothing here — do not build the decode rule. "
              f"(Caveat in the module docstring: POST-CODA states, not the raw "
              f"recurrent iterates.)")
    else:
        gain = 100 * (1 - best[1]["ce"] / max(ce_trained, 1e-9))
        vs_head = ("and beats the UncertaintyHead"
                   if best[1]["ce"] < r["ce_head"] else
                   "though the UncertaintyHead is lower")
        print(f"  VERDICT: {best[0]} gives CE {best[1]['ce']:.4f} vs "
              f"{ce_trained:.4f} at the fixed trained depth — {gain:.0f}% lower "
              f"{vs_head} ({r['ce_head']:.4f}). Worth wiring into decode; needs "
              f"NO training. NOTE it agrees with the oracle only "
              f"{100*best[1]['agreement']:.1f}% of the time, which is fine: "
              f"exact-match agreement is not the objective, emitted CE is.")

    if args.json:
        Path(args.json).write_text(json.dumps({"step": step, **r}, indent=2))
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
