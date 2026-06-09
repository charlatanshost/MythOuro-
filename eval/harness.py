"""
MythOuro evaluation harness — runs the metric suite, prints a summary,
and optionally writes a JSON report.

Usage as a library:

    from eval.harness import run_eval
    results = run_eval(model, tokenizer, benchmarks=["perplexity", "arc"])

Usage from the command line:

    python -m eval.harness \\
        --checkpoint checkpoints/step_0050000.pt \\
        --benchmarks all \\
        --max-samples 50 \\
        --output eval_results/step_0050000.json

The CLI defaults are tuned for a quick local smoke run (~5 minutes on
CPU at `--max-samples 50`); raise the cap for full benchmark runs on GPU.

Design notes
------------
- Each metric is run independently; one failing benchmark never blocks
  the others. Failures are recorded in the returned dict with an
  `"error"` key.
- The model is put into eval mode for the duration and restored to its
  prior mode afterwards.
- GPU/CPU is auto-selected from the model's actual device unless
  overridden via `device=...`.
- The JSON report includes config + per-benchmark timing so two
  consecutive runs can be diffed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger

from eval.metrics import (
    perplexity,
    arc_challenge,
    gsm8k,
    loop_efficiency,
    expected_calibration_error,
)


_BENCHMARKS = {
    "perplexity":      perplexity,
    "arc":             arc_challenge,
    "arc_challenge":   arc_challenge,
    "gsm8k":           gsm8k,
    "loop_efficiency": loop_efficiency,
    "ece":             expected_calibration_error,
}

# Default ordering for "all" — cheap metrics first, slow generative ones last.
_DEFAULT_ORDER = (
    "perplexity",
    "loop_efficiency",
    "ece",
    "arc_challenge",
    "gsm8k",
)


@dataclass
class EvalSummary:
    """Compact summary of one harness run. Used internally for printing."""
    results: dict
    elapsed_s: float
    n_benchmarks: int


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_eval(
    model,
    tokenizer,
    *,
    benchmarks: Optional[list[str]] = None,
    max_samples: int = 50,
    n_loops: Optional[int] = None,
    device: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run a benchmark suite and return a dict of `{benchmark_name: result_dict}`.

    Args:
        model         -- a (trained or untrained) MythOuro instance.
        tokenizer     -- something exposing `.encode(str) -> list[int]` and
                          `.decode(list[int]) -> str`.
        benchmarks    -- list of names; pass `None` or `["all"]` for the default
                          set. Unknown names log a warning and are skipped.
        max_samples   -- per-benchmark cap. ~50 fits a CPU smoke; 500–1000
                          for a "real" eval on GPU.
        n_loops       -- override for the model's recurrent depth during eval.
                          Defaults to `cfg.max_loop_iters`.
        device        -- "cuda" / "cpu" / explicit device. Defaults to
                          the device of the first model parameter.
        output_path   -- if set, write the full results dict as pretty JSON.
                          Parent directory is created if missing.
        verbose       -- print a per-benchmark line and a final summary.

    Returns: dict keyed by benchmark name. Includes a top-level
        `_meta` key with the run's config and total elapsed time.
    """
    if benchmarks is None or benchmarks == ["all"]:
        names = list(_DEFAULT_ORDER)
    else:
        names = list(benchmarks)

    if device is None:
        device = str(next(model.parameters()).device)

    if n_loops is None:
        n_loops = model.cfg.max_loop_iters

    prev_training = model.training
    model.eval()

    t0_total = time.perf_counter()
    results: dict = {}
    n_run = 0

    for name in names:
        if name not in _BENCHMARKS:
            logger.warning(f"eval: unknown benchmark {name!r} — skipped")
            continue
        fn = _BENCHMARKS[name]
        if verbose:
            logger.info(f"eval: running {name} (max_samples={max_samples})…")
        try:
            res = fn(
                model, tokenizer,
                max_samples=max_samples,
                n_loops=n_loops,
                device=device,
            )
        except Exception as exc:                            # noqa: BLE001
            logger.exception(f"eval: {name} crashed ({exc})")
            res = {"name": name, "error": str(exc)}
        results[name] = res
        n_run += 1
        if verbose:
            _log_result(res)

    if prev_training:
        model.train()

    elapsed = time.perf_counter() - t0_total
    results["_meta"] = {
        "elapsed_s": elapsed,
        "n_benchmarks": n_run,
        "max_samples": max_samples,
        "n_loops": n_loops,
        "device": device,
    }

    if verbose:
        _print_summary(EvalSummary(results=results, elapsed_s=elapsed, n_benchmarks=n_run))

    if output_path:
        _write_json(output_path, results)
        if verbose:
            logger.success(f"eval: wrote results to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _log_result(res: dict) -> None:
    """One-line summary of a single benchmark's result."""
    name = res.get("name", "?")
    if "error" in res:
        logger.warning(f"  ← {name}: ERROR — {res['error']}")
        return
    # Pick the canonical scalar to highlight per metric
    headline = _headline_value(res)
    elapsed = res.get("elapsed_s", 0.0)
    samples = res.get("samples", res.get("positions", 0))
    logger.info(f"  ← {name}: {headline}  (n={samples}, {elapsed:.1f}s)")


def _headline_value(res: dict) -> str:
    name = res.get("name", "")
    if name == "perplexity":
        return f"ppl={res.get('ppl', float('nan')):.3f}"
    if name == "arc_challenge":
        acc = res.get("accuracy", float("nan"))
        return f"acc={acc:.3f}  (random≈0.250)"
    if name == "gsm8k":
        return f"acc={res.get('accuracy', float('nan')):.3f}"
    if name == "loop_efficiency":
        return (
            f"avg_depth={res.get('avg_halt_depth', float('nan')):.2f}/"
            f"{res.get('max_depth', '?')}  "
            f"eff={res.get('efficiency', float('nan')):.3f}"
        )
    if name == "expected_calibration_error":
        return f"ECE={res.get('ece', float('nan')):.4f}"
    return "ok"


def _print_summary(summary: EvalSummary) -> None:
    """Tidy table at the end of a run."""
    rows: list[tuple[str, str, str, str]] = []
    for name, res in summary.results.items():
        if name == "_meta":
            continue
        rows.append((
            name,
            _headline_value(res) if "error" not in res else f"ERROR: {res['error']}",
            str(res.get("samples", res.get("positions", "—"))),
            f"{res.get('elapsed_s', 0.0):.1f}s",
        ))
    width = max((len(r[0]) for r in rows), default=10) + 2
    logger.info("=" * 80)
    logger.info(f"MythOuro eval — {summary.n_benchmarks} benchmark(s), "
                f"{summary.elapsed_s:.1f}s total")
    logger.info("-" * 80)
    for name, val, n, t in rows:
        logger.info(f"  {name.ljust(width)}  {val.ljust(40)}  n={n.rjust(6)}  {t}")
    logger.info("=" * 80)


def _write_json(path: str, results: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_checkpoint_into_model(model, checkpoint_path: str) -> int:
    """Loads weights only (not optimizer state) from a saved checkpoint.
    Returns the recorded step number, or 0 if absent."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    return int(ckpt.get("step", 0))


def _build_model_from_config(checkpoint_path: Optional[str] = None):
    """Construct an MythOuro and (optionally) load weights from disk.

    When a checkpoint is given, the model is rebuilt from the checkpoint's OWN
    saved `cfg` — so any variant (distill_tiny … 1b) loads correctly. (Previously
    this hardcoded `mythouro_1b`, so the standalone CLI could only eval a 1b
    checkpoint and size-mismatched on everything else; the archived eval JSONs
    came from in-training eval, not this path.) With no checkpoint, falls back to
    a `mythouro_1b` skeleton for a bare smoke.
    """
    from mythouro import MythOuro

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg")
        if cfg is None:                          # legacy checkpoints w/o pickled cfg
            from mythouro.variants import mythouro_1b
            cfg = mythouro_1b()
        model = MythOuro(cfg)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        logger.info(f"eval: loaded checkpoint from step {int(ckpt.get('step', 0))}")
        return model

    from mythouro.variants import mythouro_1b
    return MythOuro(mythouro_1b())


def main():
    parser = argparse.ArgumentParser(
        description="Run the MythOuro evaluation harness.",
    )
    parser.add_argument(
        "--checkpoint", "-c", default=None,
        help="Path to a .pt checkpoint. If omitted, a fresh randomly-"
             "initialised model is evaluated (useful for harness smoke tests).",
    )
    parser.add_argument(
        "--benchmarks", "-b", nargs="+", default=["all"],
        help="One or more benchmark names, or 'all'. Available: "
             + ", ".join(_DEFAULT_ORDER),
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=50,
        help="Per-benchmark sample cap. 50 is a CPU smoke; 500-1000 for GPU.",
    )
    parser.add_argument(
        "--n-loops", type=int, default=None,
        help="Override model.cfg.max_loop_iters for this eval run.",
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda / cpu / cuda:0 — defaults to the model's current device.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to write a JSON report. Parent directories are created.",
    )
    parser.add_argument(
        "--tokenizer", default="ByteDance/Ouro-2.6B-Thinking",
        help="Tokenizer model id (anything MythOuroTokenizer accepts). Default "
             "matches the Ouro-aligned vocab (49152) the distill checkpoints use; "
             "a mismatched-vocab tokenizer yields out-of-range token ids.",
    )
    args = parser.parse_args()

    from mythouro.tokenizer import MythOuroTokenizer
    tokenizer = MythOuroTokenizer(args.tokenizer)

    model = _build_model_from_config(args.checkpoint)

    from mythouro import device as dev
    device = dev.pick_device(args.device)   # cuda:0 > xpu > cpu
    model = model.to(device)

    run_eval(
        model, tokenizer,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        n_loops=args.n_loops,
        device=device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
