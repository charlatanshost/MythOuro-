"""
Comparative tokenizer evaluation.

Why this matters
----------------
The default `MythOuroTokenizer` falls back to GPT-2 BPE, which compresses
prose well but is poor on code and very poor on math (each LaTeX command
expands into many subword tokens). Picking the right tokenizer is a
one-time decision that affects every token of pretraining — and unlike
most architectural choices, it can't be revisited without re-training
from scratch.

This module compares a list of candidate tokenizers on a corpus mix that
mirrors your intended training distribution. The output is a table of
compression ratios (chars per token) per domain, plus the vocabulary
overhead each tokenizer adds.

How to interpret the numbers
----------------------------
Lower `chars_per_token` is better (more text per token = fewer steps to
cover the same content). On a mixed corpus:
    * GPT-2 BPE       — ~3.5–4.0 on prose, ~2.5 on code, ~2.0 on math
    * Llama-3 / Qwen2 — ~4.5–5.0 prose, ~3.5 code, ~2.8 math
    * DeepSeek BPE    — ~4.0 prose, ~3.8 code (specifically tuned for it)

The "right" tokenizer is the one that gives you the lowest weighted
average across your training mix, with a vocabulary size that doesn't
blow up the embedding table for your model dim.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Iterable, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Sample corpora
# ---------------------------------------------------------------------------


# Built-in tiny samples used when no user corpus is supplied. They're
# intentionally short — enough to compute meaningful compression ratios
# without requiring a network fetch. For a "real" eval, load FineWebEdu /
# codeparrot-clean / open-web-math slices via `samples_from_hf`.
_BUILTIN_SAMPLES: dict[str, str] = {
    "prose": (
        "The recurrent-depth transformer extends a single block of "
        "parameters by looping it T times, with input injection at every "
        "loop to prevent representational drift. Deeper loops at inference "
        "are an emergent property — the model can reason for longer "
        "without changing its architecture."
    ),
    "code": (
        "def merge_sort(items: list[int]) -> list[int]:\n"
        "    if len(items) <= 1:\n"
        "        return items\n"
        "    mid = len(items) // 2\n"
        "    left = merge_sort(items[:mid])\n"
        "    right = merge_sort(items[mid:])\n"
        "    return _merge(left, right)\n"
    ),
    "math": (
        r"Theorem (Cauchy-Schwarz). For vectors $u, v \in \mathbb{R}^n$, "
        r"$|\langle u, v \rangle| \leq \|u\| \cdot \|v\|$ with equality "
        r"iff $u$ and $v$ are linearly dependent. Proof: consider "
        r"$f(\lambda) = \|u - \lambda v\|^2 \geq 0$ for all $\lambda$."
    ),
    "instruction": (
        "<|user|>\nWrite a Python function that returns the n-th Fibonacci "
        "number using memoisation.\n<|assistant|>\nHere is a memoised "
        "implementation using `functools.cache`:\n"
    ),
}


# ---------------------------------------------------------------------------
# Tokenizer loading (transformers, with a clear failure path)
# ---------------------------------------------------------------------------


def _load_hf_tokenizer(name: str):
    """
    Lazy `AutoTokenizer.from_pretrained(name)`. Returns the tokenizer or
    None on failure (network, auth, model id typo). Errors are logged but
    don't propagate — the eval harness can still report results for the
    tokenizers that *did* load.
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:                                # noqa: BLE001
        logger.warning(f"tokenizer_eval: could not load {name!r} ({exc})")
        return None


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------


def _compression(tokenizer, text: str) -> dict:
    """
    Returns chars-per-token and tokens-per-char for a single text on a
    single tokenizer.
    """
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text).get("input_ids", [])
    n_tokens = len(ids)
    n_chars = len(text)
    return {
        "chars": n_chars,
        "tokens": n_tokens,
        "chars_per_token": n_chars / max(n_tokens, 1),
        "tokens_per_char": n_tokens / max(n_chars, 1),
    }


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------


def compare_tokenizers(
    tokenizer_names: list[str],
    samples: Optional[dict] = None,
    weights: Optional[dict] = None,
) -> dict:
    """
    Compare tokenizers across multiple text domains.

    Args:
        tokenizer_names -- HF model ids or local paths.
        samples         -- {domain_name: text}. Defaults to the built-in
                            tiny samples; pass FineWebEdu / the-stack /
                            open-web-math slices for a real eval.
        weights         -- {domain_name: float} domain mixing weights for the
                            weighted-average ratio. Defaults to the user's
                            MixedDataset ratios (40/30/20/10).

    Returns: dict shaped like
        {
            "results": {tokenizer_name: {domain_name: <_compression dict>}},
            "summary": {tokenizer_name: {
                "weighted_chars_per_token": float,
                "vocab_size":                int | None,
                "elapsed_s":                 float,
            }}
        }
    """
    samples = samples or _BUILTIN_SAMPLES
    if weights is None:
        weights = {
            "prose":       0.40,
            "code":        0.30,
            "math":        0.20,
            "instruction": 0.10,
        }
        # Renormalise to whatever domains are actually present.
        weights = {k: v for k, v in weights.items() if k in samples}
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

    results: dict = {}
    summary: dict = {}
    for name in tokenizer_names:
        t0 = time.perf_counter()
        tok = _load_hf_tokenizer(name)
        if tok is None:
            summary[name] = {"error": "load_failed"}
            continue
        per_domain = {}
        for dom, text in samples.items():
            per_domain[dom] = _compression(tok, text)
        weighted = sum(
            per_domain[dom]["chars_per_token"] * weights.get(dom, 0)
            for dom in per_domain
        )
        vocab = getattr(tok, "vocab_size", None)
        results[name] = per_domain
        summary[name] = {
            "weighted_chars_per_token": weighted,
            "vocab_size":                int(vocab) if vocab else None,
            "elapsed_s":                 time.perf_counter() - t0,
        }
    return {"results": results, "summary": summary, "weights": weights}


def print_summary(report: dict) -> None:
    """Pretty-print the result of `compare_tokenizers` as a table."""
    summary = report["summary"]
    weights = report.get("weights", {})
    if not summary:
        logger.warning("tokenizer_eval: nothing to print")
        return

    header = (
        f"{'tokenizer':<40} "
        f"{'weighted c/t':>14} "
        f"{'vocab':>10} "
        f"{'time s':>8}"
    )
    logger.info("=" * len(header))
    logger.info(f"Tokenizer comparison (weights: {weights})")
    logger.info("-" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    # Sort highest compression first
    ordered = sorted(
        summary.items(),
        key=lambda kv: -(kv[1].get("weighted_chars_per_token") or 0),
    )
    for name, s in ordered:
        if "error" in s:
            logger.info(f"{name:<40} {'ERROR: ' + s['error']:>14}")
            continue
        logger.info(
            f"{name[:40]:<40} "
            f"{s['weighted_chars_per_token']:>14.3f} "
            f"{(s['vocab_size'] or 0):>10,} "
            f"{s['elapsed_s']:>8.1f}"
        )
    logger.info("=" * len(header))


# ---------------------------------------------------------------------------
# HF-corpus sample helper
# ---------------------------------------------------------------------------


def samples_from_hf(
    n_chars_per_domain: int = 20_000,
) -> dict:
    """
    Build a `samples` dict from short streaming slices of the user's
    actual training corpora. Useful for a more realistic comparison.

    Reads `n_chars_per_domain` characters per domain from FineWebEdu /
    codeparrot-clean / open-web-math. Falls back to the built-in samples
    if streaming fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("tokenizer_eval: `datasets` missing — using builtin samples")
        return _BUILTIN_SAMPLES.copy()

    specs = [
        ("prose", "HuggingFaceFW/fineweb-edu",    "sample-10BT", "text"),
        ("code",  "codeparrot/codeparrot-clean",  None,          "content"),
        ("math",  "open-web-math/open-web-math",  None,          "text"),
    ]
    out: dict = {}
    for domain, repo, config, field in specs:
        try:
            ds = load_dataset(repo, name=config, split="train", streaming=True)
            buf = ""
            for sample in ds:
                buf += sample.get(field) or ""
                if len(buf) >= n_chars_per_domain:
                    break
            out[domain] = buf[:n_chars_per_domain]
            logger.info(f"tokenizer_eval: sampled {len(out[domain]):,} chars for '{domain}'")
        except Exception as exc:                            # noqa: BLE001
            logger.warning(f"tokenizer_eval: '{domain}' sampling failed ({exc}) — using builtin")
            out[domain] = _BUILTIN_SAMPLES.get(domain, "")
    if "instruction" in _BUILTIN_SAMPLES:
        out["instruction"] = _BUILTIN_SAMPLES["instruction"]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_TOKENIZERS = [
    "openai-community/gpt2",                # current default (worst on code/math)
    "Qwen/Qwen2.5-0.5B",                    # strong all-around
    "meta-llama/Llama-3.2-1B",              # strong all-around (auth required)
    "deepseek-ai/DeepSeek-V2-Lite",         # code-biased
    "openai-community/openai-gpt",          # older baseline
]


def _main(argv: "list[str] | None" = None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare candidate tokenizers on a representative sample.",
    )
    parser.add_argument(
        "--tokenizers", "-t", nargs="+", default=_DEFAULT_TOKENIZERS,
        help="HF model ids or local paths.",
    )
    parser.add_argument(
        "--use-hf-samples", action="store_true",
        help="Sample text from FineWebEdu / codeparrot-clean / open-web-math "
             "(requires network). Default: built-in tiny samples.",
    )
    parser.add_argument(
        "--n-chars", type=int, default=20_000,
        help="Per-domain sample size in chars when --use-hf-samples is set.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args(argv)

    samples = (
        samples_from_hf(n_chars_per_domain=args.n_chars)
        if args.use_hf_samples else None
    )
    report = compare_tokenizers(args.tokenizers, samples=samples)
    print_summary(report)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.success(f"tokenizer_eval: report → {args.output}")


if __name__ == "__main__":
    _main()
