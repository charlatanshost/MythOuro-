"""
Training-vs-eval contamination filter.

Why this matters
----------------
If your training corpus contains the verbatim text of an eval benchmark's
test questions, your eval numbers are meaningless — the model is recalling,
not reasoning. This is a real issue at scale: ARC questions appear in
educational web crawls, GSM8K examples leak via blogs and tutorials,
HumanEval prompts surface in Stack Overflow answers and GitHub READMEs.

How it works
------------
For each eval benchmark we extract a stable set of normalised character
n-grams from every test question and (when present) its gold answer text.
A training document is flagged contaminated if it contains any one of
those exact n-grams. The 13-gram default matches the OpenAI / Llama 3
threshold — long enough to avoid collisions on common phrases, short
enough to catch verbatim leakage.

This is intentionally a precision-first design: we want zero false
positives at eval time (no spurious "your model is contaminated" alerts),
which is why the bar is verbatim 13-gram match. Paraphrased contamination
slips through; for that you'd need a semantic-similarity filter, which is
expensive and out of scope.

Loading benchmarks
------------------
Each benchmark has its own loader function (`_load_arc`, `_load_gsm8k`,
`_load_humaneval`) that returns an iterable of test-question strings. To
add a benchmark, write a loader and register it in `_BENCHMARK_LOADERS`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Iterable, Iterator, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Text normalisation (must match training-time normalisation to be useful)
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalise(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace.

    This is more aggressive than the dedup normalisation — it has to be,
    or you miss obvious leaks like "What is 2+2?" vs "what is 2 plus 2".
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _ngrams(text: str, n: int) -> Iterator[str]:
    """Sliding word n-grams. Returns no items if text is shorter than n words."""
    words = _normalise(text).split()
    if len(words) < n:
        return
    for i in range(len(words) - n + 1):
        yield " ".join(words[i : i + n])


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------


def _safe_load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
        return load_dataset(*args, **kwargs)
    except Exception as exc:                                # noqa: BLE001
        logger.warning(f"contamination: load_dataset failed ({exc})")
        return None


def _load_arc(max_samples: Optional[int] = None) -> Iterator[str]:
    """ARC-Challenge test questions + answer choices."""
    ds = _safe_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if ds is None:
        return
    items = ds if max_samples is None else ds.select(range(min(max_samples, len(ds))))
    for sample in items:
        yield sample.get("question") or ""
        # Choice text is short but worth indexing — leaked answer rationale
        # is a common contamination vector.
        for choice in (sample.get("choices") or {}).get("text") or []:
            yield choice


def _load_gsm8k(max_samples: Optional[int] = None) -> Iterator[str]:
    """GSM8K test questions and full chain-of-thought answers."""
    ds = _safe_load_dataset("openai/gsm8k", "main", split="test")
    if ds is None:
        return
    items = ds if max_samples is None else ds.select(range(min(max_samples, len(ds))))
    for sample in items:
        yield sample.get("question") or ""
        yield sample.get("answer") or ""


def _load_humaneval(max_samples: Optional[int] = None) -> Iterator[str]:
    """HumanEval prompts (function signatures + docstrings)."""
    ds = _safe_load_dataset("openai_humaneval", split="test")
    if ds is None:
        return
    items = ds if max_samples is None else ds.select(range(min(max_samples, len(ds))))
    for sample in items:
        yield sample.get("prompt") or ""


_BENCHMARK_LOADERS: dict[str, Callable[[Optional[int]], Iterator[str]]] = {
    "arc":       _load_arc,
    "gsm8k":     _load_gsm8k,
    "humaneval": _load_humaneval,
}


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class ContaminationFilter:
    """
    Build an n-gram index from one or more eval benchmarks and use it to
    flag training documents that contain verbatim leakage.

    Default n=13 matches Llama-3 / OpenAI convention. Shorter n catches
    more leakage but produces false positives on common phrases; longer
    n is overly conservative.

    Usage:
        f = ContaminationFilter(["arc", "gsm8k", "humaneval"])
        f.build_index()
        for text in training_stream:
            if not f.is_contaminated(text):
                yield text
    """

    def __init__(
        self,
        benchmarks: list[str],
        n: int = 13,
        max_samples_per_benchmark: Optional[int] = None,
    ):
        unknown = set(benchmarks) - set(_BENCHMARK_LOADERS)
        if unknown:
            raise ValueError(
                f"Unknown benchmark(s): {unknown}. "
                f"Known: {sorted(_BENCHMARK_LOADERS)}"
            )
        self.benchmarks = benchmarks
        self.n = n
        self.max_samples = max_samples_per_benchmark
        self._index: set[str] = set()
        self._built = False

    @property
    def stats(self) -> dict:
        return {
            "benchmarks": self.benchmarks,
            "n":          self.n,
            "ngrams":     len(self._index),
            "built":      self._built,
        }

    def build_index(self) -> int:
        """
        Materialise the n-gram index from the configured benchmarks.

        Returns the number of unique n-grams collected. Idempotent — a
        second call is a no-op.
        """
        if self._built:
            return len(self._index)
        for bench in self.benchmarks:
            loader = _BENCHMARK_LOADERS[bench]
            count_before = len(self._index)
            for text in loader(self.max_samples):
                for gram in _ngrams(text, self.n):
                    self._index.add(gram)
            added = len(self._index) - count_before
            logger.info(f"contamination: {bench} → {added:,} unique {self.n}-grams")
        self._built = True
        return len(self._index)

    def add_custom_text(self, text: str) -> int:
        """
        Add n-grams from an arbitrary string (e.g. a private holdout set
        not loadable via HF). Marks the filter as built. Returns the
        number of new n-grams added.
        """
        before = len(self._index)
        for gram in _ngrams(text, self.n):
            self._index.add(gram)
        self._built = True
        return len(self._index) - before

    def is_contaminated(self, text: str) -> bool:
        """
        True iff any n-gram of `text` appears in the eval-benchmark index.

        Builds the index lazily on first call if not already built — but
        you should call `build_index()` once up front in production so
        the cost is paid before the streaming loop.
        """
        if not self._built:
            self.build_index()
        if not text.strip():
            return False
        for gram in _ngrams(text, self.n):
            if gram in self._index:
                return True
        return False

    def filter_stream(
        self,
        source: Iterable["tuple[str, str]"],
        log_every: int = 10_000,
    ) -> Iterator["tuple[str, str]"]:
        """
        Filter a stream of `(doc_id, text)` pairs, dropping contaminated docs.
        """
        if not self._built:
            self.build_index()
        seen = 0
        dropped = 0
        for doc_id, text in source:
            seen += 1
            if self.is_contaminated(text):
                dropped += 1
                continue
            yield doc_id, text
            if seen % log_every == 0:
                logger.info(
                    f"contamination: seen={seen:,} dropped={dropped:,} "
                    f"({dropped / max(seen, 1) * 100:.2f}%)"
                )


# ---------------------------------------------------------------------------
# Convenience: filter a JSONL file
# ---------------------------------------------------------------------------


def filter_jsonl(
    input_path: str,
    output_path: str,
    benchmarks: list[str],
    text_field: str = "text",
    n: int = 13,
) -> dict:
    """Read JSONL → remove contaminated docs → write JSONL."""
    f = ContaminationFilter(benchmarks=benchmarks, n=n)
    f.build_index()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    seen = 0
    written = 0
    with open(input_path,  encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            if f.is_contaminated(rec.get(text_field, "")):
                continue
            fout.write(json.dumps(rec) + "\n")
            written += 1
    logger.success(
        f"contamination: wrote {written:,} / {seen:,} docs "
        f"({(seen - written) / max(seen, 1) * 100:.2f}% dropped) → {output_path}"
    )
    return {"seen": seen, "written": written, "dropped": seen - written}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: "list[str] | None" = None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Drop training docs that overlap eval benchmarks (n-gram match)."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument(
        "--benchmarks", "-b", nargs="+", default=["arc", "gsm8k", "humaneval"],
        choices=list(_BENCHMARK_LOADERS),
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--n", type=int, default=13,
                        help="Word n-gram length (default 13).")
    args = parser.parse_args(argv)
    filter_jsonl(
        args.input, args.output,
        benchmarks=args.benchmarks,
        text_field=args.text_field,
        n=args.n,
    )


if __name__ == "__main__":
    _main()
