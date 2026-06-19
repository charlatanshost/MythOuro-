#!/usr/bin/env python
"""
collapse_metrics.py — quantify recurrent hidden-state collapse.

Turns "it collapses" into a tracked number, using the diagnostics from the
recurrent-depth literature:

  * Token correlation  (Huginn, arXiv 2502.05171): mean off-diagonal cosine
    similarity between token hidden states at each loop. -> 1.0 means every
    token has the same representation = collapse. Their "Bad Run 1" hit 1.0.
  * Effective rank     (MeSH, arXiv 2510.07739, Fig 5): exp(entropy of the
    singular-value spectrum) of the (T x D) hidden-state matrix per loop.
    Collapse shows as fast spectral decay -> effective rank near 1.

Both are read from `model.recurrent.last_trajectory` (B, T, K, D), the per-loop
committed hidden states the block already stashes when `collect_trajectory` is
set. Forward-only; quick; no training. Run it on any checkpoint to baseline the
collapse, and to judge whether the Huginn recipe / MeSH-style changes actually
fix it (watch correlation stop climbing toward 1 and rank stop decaying).

Usage:
  python tools/collapse_metrics.py -c checkpoints_noise_test/step_0011000.pt --device cuda:0
"""

import argparse
import os
import statistics
import sys

import torch
import torch.nn.functional as F

# Make the repo root importable so we can reuse the inspector's loader/tokenizer.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inspect_checkpoint import _DEFAULT_PROMPTS, _load_model  # noqa: E402
from mythouro.tokenizer import MythOuroTokenizer  # noqa: E402

# Categorised probe sets — several prompts per domain so we can watch how each
# capability learns, instead of inferring from a single prompt (n=1). The original
# 4 default prompts are kept (one per format) so older reports stay comparable; the
# rest extend each category. Format matters as much as subject on a distill-only
# model: prose / code / math are *in-format* for the distill corpus (FineWeb-Edu /
# codeparrot / open-web-math, the 40/20/40 mix); chat (ChatML) and qa (Q:/A:) are
# OOD formats until SFT introduces them. Newlines are embedded so the format matches
# exactly regardless of shell quoting.
_PROBE_SETS = {
    "prose": [
        "The recurrent depth transformer is",
        "The history of the Roman Empire began",
        "Photosynthesis is the process by which",
        "The largest planet in our solar system is",
    ],
    "code": [
        "def fibonacci(n):",
        "def is_prime(n):",
        "import numpy as np\n",
        "class Stack:\n    def __init__(self):",
    ],
    "math": [
        "The derivative of x^2 with respect to x is",
        "The sum of the first 10 positive integers is",
        "To solve 2x + 3 = 7, subtract 3 from both sides to get",
        "The area of a circle with radius r is",
    ],
    "chat": [
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain gravity in one sentence.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n",
    ],
    "qa": [
        "Q: Roughly what year was the Roman Empire founded?\nA:",
        "Q: What is the capital of France?\nA:",
        "Q: How many days are in a week?\nA:",
        "Q: What color is the sky on a clear day?\nA:",
    ],
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def token_correlation(H: torch.Tensor) -> float:
    """
    Mean off-diagonal cosine similarity between the T token vectors in H (T, D).

    -> 1.0 means all tokens collapsed to the same direction (Huginn's collapse
    signature). T must be >= 2.
    """
    Hn = H / (H.norm(dim=-1, keepdim=True) + 1e-8)
    sims = Hn @ Hn.t()                                  # (T, T)
    T = H.shape[0]
    off_sum = sims.sum() - sims.diagonal().sum()
    return float(off_sum / (T * (T - 1)))


def effective_rank(H: torch.Tensor) -> float:
    """
    exp(Shannon entropy of the normalised singular-value spectrum) of H (T, D).

    In [1, min(T, D)]. Low -> the representation lives in a few dimensions
    (MeSH's "fast spectral decay" = collapse). High -> full-dimensional use.
    """
    s = torch.linalg.svdvals(H.float())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * (p + 1e-12).log()).sum()
    return float(torch.exp(entropy))


# ---------------------------------------------------------------------------
# Capture + report
# ---------------------------------------------------------------------------


@torch.no_grad()
def per_loop_trajectory(model, ids: torch.Tensor, n_loops: int) -> torch.Tensor:
    """
    Run a forward pass capturing every loop's committed hidden state.

    Returns the real-token trajectory (T, K, D) for the single batch row, with
    sink positions stripped. Forces full depth so K == n_loops (no ACT exit).
    """
    rec = model.recurrent
    prev_collect, prev_full = rec.collect_trajectory, rec.force_full_depth
    rec.collect_trajectory = True
    rec.force_full_depth = True
    try:
        model(ids, n_loops=n_loops)                    # no kv_cache -> populates
        traj = rec.last_trajectory                     # (B, T_ext, K, D) or None
    finally:
        rec.collect_trajectory = prev_collect
        rec.force_full_depth = prev_full
        rec.last_trajectory = None

    if traj is None:
        raise RuntimeError("last_trajectory is None — capture failed.")
    sink_len = int(getattr(getattr(model, "sink", None), "n_tokens", 0) or 0)
    return traj[0, sink_len:, :, :].float()            # (T, K, D)


def report_prompt(model, tokenizer, prompt: str, device: str, n_loops: int) -> list:
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    traj = per_loop_trajectory(model, ids, n_loops)    # (T, K, D)
    T, K, _ = traj.shape
    rows = []
    for k in range(K):
        Hk = traj[:, k, :]                             # (T, D)
        corr = token_correlation(Hk) if T >= 2 else float("nan")
        erank = effective_rank(Hk)
        rows.append((k, corr, erank))

    print(f"\nprompt: {prompt!r}   (T={T} real tokens, K={K} loops)")
    print("  loop |  token_corr (->1 = collapse) | eff_rank (->1 = collapse)")
    for k, corr, erank in rows:
        print(f"   {k:>3} |  {corr:>26.3f} | {erank:>8.2f}")
    if K >= 2:
        d_corr = rows[-1][1] - rows[0][1]
        d_rank = rows[-1][2] - rows[0][2]
        trend = ("recurrence DRIVES collapse" if d_corr > 0.05 and d_rank < 0
                 else "stable across loops" if abs(d_corr) <= 0.05
                 else "recurrence diversifies")
        print(f"  loop0->lastK: dcorr={d_corr:+.3f} drank={d_rank:+.2f}  [{trend}]")
    return rows


@torch.no_grad()
def generation_diagnostic(model, tokenizer, prompt: str, device: str,
                          n_loops: int, max_new: int = 32,
                          temperature: float = 0.0, top_k: int = 0) -> dict:
    """
    Generate `max_new` tokens (full recompute) to locate the degeneration:
    in the REPRESENTATIONS or in the OUTPUT DISTRIBUTION.

    temperature == 0 -> greedy (worst case for the repetition spiral). >0 ->
    sample (optionally top-k). The greedy-vs-sampled contrast tests whether the
    degeneration is a *greedy decoding* failure (high early entropy means
    sampling can escape the attractor — the principled version of v4's accidental
    inference-time noise) or something sampling can't fix.

    Reports the generated text; the output distribution's top-token prob and
    entropy as generation proceeds (entropy -> 0 / top-prob -> 1 = a confident
    repetition attractor); and rep metrics (token_corr, eff_rank) on the PROMPT
    positions vs the GENERATED positions at the final loop.

      reps stay healthy + output degenerate  -> output / exposure-bias path
      generated-token reps collapse          -> free-running rep degradation
    """
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    T0 = ids.shape[1]
    top_probs, entropies, gen = [], [], []
    for _ in range(max_new):
        logits, _ = model(ids, n_loops=n_loops)
        last = logits[0, -1].float()
        logp = F.log_softmax(last, dim=-1)
        p = logp.exp()
        top_probs.append(float(p.max()))               # raw model confidence
        entropies.append(float(-(p * logp).sum()))     # (comparable across temps)
        if temperature and temperature > 0:
            scaled = last / temperature
            if top_k and top_k > 0:
                kth = torch.topk(scaled, min(top_k, scaled.numel())).values[-1]
                scaled = scaled.masked_fill(scaled < kth, float("-inf"))
            nxt = int(torch.multinomial(scaled.softmax(dim=-1), 1))
        else:
            nxt = int(last.argmax())                    # greedy = expose the attractor
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)

    # Rep metrics on the full (prompt+generated) sequence, split by region.
    traj = per_loop_trajectory(model, ids, n_loops)    # (T, K, D), sink stripped
    last_loop = traj[:, -1, :]                          # (T, D)
    prompt_H, gen_H = last_loop[:T0], last_loop[T0:]

    def _m(H):
        return (token_correlation(H) if H.shape[0] >= 2 else float("nan"),
                effective_rank(H))
    pc, pr = _m(prompt_H)
    gc, gr = _m(gen_H)
    mean_ent, mean_tp = statistics.fmean(entropies), statistics.fmean(top_probs)

    decode = "greedy" if not (temperature and temperature > 0) else (
        f"T={temperature}" + (f" top_k={top_k}" if top_k else ""))
    print(f"\nprompt: {prompt!r}")
    print(f"  generated ({decode}, {max_new} tok): {tokenizer.decode(gen)!r}")
    print(f"  output dist:  mean_entropy={mean_ent:.3f} nats  final_entropy={entropies[-1]:.3f}"
          f"  mean_top_prob={mean_tp:.3f}  final_top_prob={top_probs[-1]:.3f}")
    print("  first steps (top_prob, entropy): "
          + ", ".join(f"({tp:.2f},{e:.2f})" for tp, e in list(zip(top_probs, entropies))[:8]))
    print(f"  rep @final loop:  PROMPT corr={pc:.3f} rank={pr:.2f}  |  "
          f"GENERATED corr={gc:.3f} rank={gr:.2f}")

    gen_healthy = (not (gc != gc)) and gc < 0.6 and gr > max(2.0, 0.4 * gen_H.shape[0])
    out_degen = mean_ent < 1.0 or mean_tp > 0.8
    if gen_healthy and out_degen:
        verdict = ("OUTPUT / exposure-bias path — reps stay healthy, distribution "
                   "degenerate. -> reverse-KL / GKD")
    elif not gen_healthy:
        verdict = "FREE-RUNNING rep degradation — generated-token reps collapse"
    else:
        verdict = "inconclusive / not clearly degenerate at this length"
    print(f"  -> {verdict}")
    return {"prompt_rep": (pc, pr), "gen_rep": (gc, gr),
            "mean_entropy": mean_ent, "mean_top_prob": mean_tp}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", "-c", required=True, help="step_*.pt to analyse")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--prompt", "-p", action="append", default=None,
                   help="Prompt to test; repeat -p for several. Default = the "
                        "inspector's 4-prompt set. Note: shells turn '\\n' literal, "
                        "so use --probe-set for prompts that need real newlines.")
    p.add_argument("--probe-set", choices=sorted(_PROBE_SETS) + ["all"],
                   default=None,
                   help="Run a built-in categorised probe set (several prompts per "
                        "domain, real newlines). One of: "
                        f"{', '.join(sorted(_PROBE_SETS))}, all. 'all' runs every "
                        "category with a per-category header so you can watch each "
                        "capability learn separately.")
    p.add_argument("--qa-probe", action="store_true",
                   help="Back-compat alias for --probe-set qa.")
    p.add_argument("--n-loops", type=int, default=None,
                   help="Recurrent depth. Default: cfg.max_loop_iters.")
    p.add_argument("--generate", action="store_true",
                   help="Generation-time mode: greedy-generate and measure whether "
                        "degeneration is in the reps or the output distribution.")
    p.add_argument("--max-new", type=int, default=32,
                   help="Tokens to generate in --generate mode.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="--generate sampling temperature. 0 = greedy (default). "
                        ">0 samples — tests whether sampling escapes the spiral "
                        "(the principled version of v4's inference-time noise).")
    p.add_argument("--top-k", type=int, default=0,
                   help="Top-k for --generate sampling (0 = off).")
    p.add_argument("--inference-noise", type=float, default=0.0,
                   help="Apply representation noise (sigma) at INFERENCE — the "
                        "principled version of v4's accidental noise. Tests whether "
                        "it escapes the spiral. 0 = off. Try 0.05-0.1.")
    args = p.parse_args()

    # Windows consoles default to cp1252, which can't encode the U+FFFD
    # replacement char that byte-level BPE can emit when sampling splits a
    # multi-byte token — that raised UnicodeEncodeError mid-print and aborted
    # the T>0 read. Force UTF-8 with a safe fallback so the diagnostic always
    # completes regardless of the host locale.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

    print(f"loading: {args.checkpoint}\ndevice:  {args.device}")
    model, cfg, step = _load_model(args.checkpoint, args.device)
    tokenizer = MythOuroTokenizer(args.tokenizer)
    n_loops = args.n_loops or cfg.max_loop_iters
    print(f"step={step}  n_loops={n_loops}  "
          f"sandwich_norm={getattr(cfg, 'use_sandwich_norm', False)}  "
          f"state_noise={getattr(cfg, 'recurrent_state_noise', 0.0)}")

    # Build (category, prompt) pairs so the generate path can print per-category
    # headers; `prompts` stays a flat list for the rep/rank path below.
    probe_set = "qa" if args.qa_probe else args.probe_set
    if probe_set == "all":
        selected = [(cat, pr) for cat in sorted(_PROBE_SETS) for pr in _PROBE_SETS[cat]]
    elif probe_set:
        selected = [(probe_set, pr) for pr in _PROBE_SETS[probe_set]]
    elif args.prompt:
        selected = [("custom", pr) for pr in args.prompt]
    else:
        selected = [("default", pr) for pr in _DEFAULT_PROMPTS]
    prompts = [pr for _, pr in selected]

    if args.generate:
        mode = "greedy" if args.temperature <= 0 else f"T={args.temperature}"
        if args.inference_noise > 0:
            model.recurrent.inference_noise = True
            model.recurrent.state_noise_sigma = args.inference_noise
            mode += f" +infnoise_sigma={args.inference_noise}"
        print(f"\n[generation-time mode] {mode} decode; locating the degeneration")
        cur_cat = None
        for cat, prompt in selected:
            if cat != cur_cat:
                print(f"\n=== category: {cat} ===")
                cur_cat = cat
            generation_diagnostic(model, tokenizer, prompt, args.device,
                                  n_loops, args.max_new, args.temperature, args.top_k)
        return

    all_last_corr, all_last_rank = [], []
    for prompt in prompts:
        rows = report_prompt(model, tokenizer, prompt, args.device, n_loops)
        if rows:
            all_last_corr.append(rows[-1][1])
            all_last_rank.append(rows[-1][2])

    if all_last_corr:
        mc = statistics.fmean(c for c in all_last_corr if c == c)
        mr = statistics.fmean(all_last_rank)
        print("\n" + "=" * 60)
        print(f"SUMMARY (final loop, mean over prompts): "
              f"token_corr={mc:.3f}  eff_rank={mr:.2f}")
        verdict = ("COLLAPSED" if mc > 0.9 or mr < 2.0
                   else "partial" if mc > 0.6 or mr < 4.0
                   else "healthy")
        print(f"verdict: {verdict}   (corr>0.9 or rank<2 = collapsed; "
              f"lower corr / higher rank = better)")


if __name__ == "__main__":
    main()
