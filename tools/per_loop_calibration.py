"""
Per-loop calibration audit for the UncertaintyHead (P0.5).

The head is trained on ONE distribution — logits from the training forward
(the h_K path) — but consumed on others: `forward_trajectory` /
`BestOfTrajectoryGenerator` score **per-loop** states the head was never
calibrated on, and MoDr's best-exit teacher labels would inherit whatever
mis-calibration exists there. The headline ECE (0.04-class) only certifies the
final-loop path; this tool measures ECE **per loop index** so the per-loop
uses rest on evidence.

For each held-out chunk it runs `forward_trajectory(force_full_depth=True)`
(ACT early-exits suppressed → all K loops scored), then per loop k computes:
  - ECE_k        — calibration of unc[..., k] against argmax-error at loop k
  - accuracy_k   — next-token argmax accuracy of loop k's logits
  - mean_unc_k   — the head's average predicted error at loop k
  - gap_k        — |mean_unc_k − actual error rate|, the crude bias readout

Usage:
    python -m tools.per_loop_calibration \\
        --checkpoint archived_models/mythouro_distill_tiny_sft_v2/step_0003000.pt \\
        --max-samples 20 --seq-len 256 --out reports/per_loop_ece_v2.json

Interpretation guide (for the MoDr decision):
  - ECE_k flat-ish across loops  → the head generalises; uncertainty-argmin
    best-exit labels are usable.
  - ECE_k good at the last loop but poor at shallow loops → the head is only
    calibrated where it was trained; use **per-loop CE** as the MoDr
    supervision target instead (the roadmap's safer option).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mythouro.main import MythOuro  # noqa: E402
from mythouro.tokenizer import MythOuroTokenizer  # noqa: E402
from eval.metrics import _safe_load_dataset  # noqa: E402


def _ece(preds: torch.Tensor, errors: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error — same binning as eval.metrics."""
    edges = torch.linspace(0, 1, n_bins + 1)
    total = preds.numel()
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        in_bin = (preds >= lo) & (preds < hi if i < n_bins - 1 else preds <= hi)
        n_in = int(in_bin.sum().item())
        if n_in == 0:
            continue
        ece += (n_in / total) * abs(
            float(preds[in_bin].mean()) - float(errors[in_bin].mean())
        )
    return ece


@torch.no_grad()
def per_loop_calibration(
    model: MythOuro,
    tokenizer,
    *,
    device: str = "cpu",
    n_loops: "int | None" = None,
    max_samples: int = 20,
    seq_len: int = 256,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
) -> dict:
    """Collect per-loop (uncertainty, argmax-error) pairs and compute per-loop
    ECE / accuracy. Returns a dict with one row per loop index."""
    K = n_loops or model.cfg.max_loop_iters
    t0 = time.perf_counter()

    ds = _safe_load_dataset(
        dataset_name, name=dataset_config, split="train", streaming=True,
    )
    if ds is None:
        raise RuntimeError("could not open eval corpus")

    preds_k: "list[list[float]]" = [[] for _ in range(K)]
    errs_k: "list[list[float]]" = [[] for _ in range(K)]
    n_chunks = 0
    buf: list[int] = []

    for sample in ds:
        if n_chunks >= max_samples:
            break
        text = sample.get("text") or ""
        if not text:
            continue
        buf.extend(tokenizer.encode(text))
        while len(buf) >= seq_len + 1 and n_chunks < max_samples:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]
            ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            tgt = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

            logits_traj, unc_traj = model.forward_trajectory(
                ids, n_loops=K, force_full_depth=True,
            )                                   # (1,T,K,V), (1,T,K)
            k_run = logits_traj.shape[2]
            assert k_run == K, (k_run, K)
            for k in range(K):
                pred = logits_traj[0, :, k, :].argmax(dim=-1)      # (T,)
                wrong = (pred != tgt[0]).float()
                preds_k[k].extend(unc_traj[0, :, k].float().cpu().tolist())
                errs_k[k].extend(wrong.cpu().tolist())
            n_chunks += 1

    rows = []
    for k in range(K):
        p = torch.tensor(preds_k[k])
        e = torch.tensor(errs_k[k])
        rows.append({
            "loop": k,
            "ece": round(_ece(p, e), 4),
            "accuracy": round(1.0 - float(e.mean()), 4),
            "mean_uncertainty": round(float(p.mean()), 4),
            "actual_error_rate": round(float(e.mean()), 4),
            "gap": round(abs(float(p.mean()) - float(e.mean())), 4),
            "positions": int(p.numel()),
        })

    return {
        "n_loops": K,
        "samples": n_chunks,
        "seq_len": seq_len,
        "per_loop": rows,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--n-loops", type=int, default=None,
                   help="Loops to score (default: cfg.max_loop_iters).")
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MythOuro(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()
    print(f"checkpoint: {args.checkpoint} (step {int(ckpt.get('step', 0))})")

    tokenizer = MythOuroTokenizer(args.tokenizer)
    result = per_loop_calibration(
        model, tokenizer, device=args.device, n_loops=args.n_loops,
        max_samples=args.max_samples, seq_len=args.seq_len,
    )

    print(f"\nper-loop calibration  ({result['samples']} chunks x "
          f"{result['seq_len']} tok, {result['elapsed_s']}s)")
    print(f"{'loop':>4} | {'ECE':>7} | {'acc':>7} | {'mean unc':>8} | "
          f"{'err rate':>8} | {'gap':>6}")
    print("-" * 56)
    for r in result["per_loop"]:
        print(f"{r['loop']:>4} | {r['ece']:>7.4f} | {r['accuracy']:>7.4f} | "
              f"{r['mean_uncertainty']:>8.4f} | {r['actual_error_rate']:>8.4f} | "
              f"{r['gap']:>6.4f}")

    last = result["per_loop"][-1]["ece"]
    worst = max(r["ece"] for r in result["per_loop"])
    print("-" * 56)
    if worst > 2.5 * max(last, 1e-6) and worst - last > 0.05:
        print(f"VERDICT: head is calibrated at the final loop (ECE {last:.3f}) but "
              f"NOT at shallower loops (worst {worst:.3f}).")
        print("-> use per-loop CE, not uncertainty-argmin, as the MoDr best-exit target.")
    else:
        print(f"VERDICT: per-loop ECE roughly uniform (final {last:.3f}, worst "
              f"{worst:.3f}) — uncertainty-based per-loop selection is defensible.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
