"""
Unit tests for the MythOuro data pipeline (`data/`).

Pinning the contracts:

    1. MinHashDeduplicator flags exact and near-duplicates, keeps unrelated docs.
    2. ContaminationFilter flags docs containing eval n-grams (with a
       mocked eval-loader path so tests don't require a network).
    3. compare_tokenizers gives lower chars-per-token for tokenizers
       known to compress better.

Tests that would otherwise need HF dataset downloads use the
`add_custom_text` injection path on ContaminationFilter so they stay
fully offline.
"""

from __future__ import annotations

import json

import pytest

from data.dedup import MinHashDeduplicator, dedup_jsonl, stream_dedup
from data.contamination import ContaminationFilter, _ngrams, _normalise
from data.tokenizer_eval import _BUILTIN_SAMPLES

try:
    import datasketch  # noqa: F401 — optional `data` extra
    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


_DOC_A = (
    "The quick brown fox jumps over the lazy dog. "
    "MythOuro is a Recurrent-Depth Transformer that loops a single "
    "block of parameters T times with input injection at every step to "
    "prevent representational drift across loop iterations. The looped "
    "architecture allows deeper reasoning chains at inference without "
    "increasing the parameter count, an emergent property of the design."
)
_DOC_A_DUPLICATE = _DOC_A
# One-word substitution leaves Jaccard ≈ 0.95 on 5-gram shingles for a
# doc this length — well above the 0.8 default threshold.
_DOC_A_NEAR = _DOC_A.replace("quick", "swift")
_DOC_B = (
    "Python's dataclasses make it easy to define record-like structures "
    "with sensible defaults and type hints. The standard library also "
    "provides typing helpers, attrs and pydantic offer richer alternatives "
    "for runtime validation and schema definition in production systems."
)
_DOC_C = (
    "The history of the Roman Empire spans roughly five centuries, from "
    "27 BC under Augustus to the deposition of Romulus Augustulus in 476 "
    "AD, traditionally marking the end of the Western Empire. The Eastern "
    "Roman Empire, later called Byzantium, persisted for nearly a thousand "
    "more years before falling to the Ottomans in 1453."
)


@pytest.mark.skipif(
    not _HAS_DATASKETCH,
    reason="datasketch not installed (optional `data` extra) — "
           "MinHashDeduplicator raises ImportError on construction",
)
class TestMinHashDeduplicator:
    def test_first_doc_is_kept(self):
        d = MinHashDeduplicator()
        assert d.add("doc_a", _DOC_A) is True
        assert d.stats["indexed"] == 1
        assert d.stats["dropped"] == 0

    def test_exact_duplicate_is_dropped(self):
        d = MinHashDeduplicator()
        d.add("doc_a", _DOC_A)
        assert d.add("doc_a_copy", _DOC_A_DUPLICATE) is False
        assert d.stats["dropped"] == 1

    def test_near_duplicate_is_dropped(self):
        # A one-word substitution stays well above the 0.8 Jaccard threshold
        # on char-5-grams for a sentence this short.
        d = MinHashDeduplicator(threshold=0.8)
        d.add("doc_a", _DOC_A)
        assert d.add("doc_a_near", _DOC_A_NEAR) is False

    def test_unrelated_docs_are_kept(self):
        d = MinHashDeduplicator()
        assert d.add("a", _DOC_A) is True
        assert d.add("b", _DOC_B) is True
        assert d.add("c", _DOC_C) is True
        assert d.stats["indexed"] == 3
        assert d.stats["dropped"] == 0

    def test_empty_text_is_dropped(self):
        d = MinHashDeduplicator()
        assert d.add("empty", "") is False
        assert d.add("blank", "    \n  ") is False

    def test_is_duplicate_does_not_mutate(self):
        d = MinHashDeduplicator()
        d.add("a", _DOC_A)
        n_before = d.stats["seen"]
        # Querying should not advance the seen counter
        d.is_duplicate(_DOC_A)
        assert d.stats["seen"] == n_before

    def test_stream_dedup_helper(self):
        docs = [
            ("1", _DOC_A),
            ("2", _DOC_A),                # exact dup of 1
            ("3", _DOC_B),
            ("4", _DOC_A_NEAR),           # near-dup of 1
            ("5", _DOC_C),
        ]
        kept = list(stream_dedup(docs))
        kept_ids = [k for k, _ in kept]
        assert kept_ids == ["1", "3", "5"]

    def test_dedup_jsonl_roundtrip(self, tmp_path):
        in_path  = tmp_path / "in.jsonl"
        out_path = tmp_path / "out.jsonl"
        records = [
            {"id": "1", "text": _DOC_A},
            {"id": "2", "text": _DOC_A},
            {"id": "3", "text": _DOC_B},
            {"id": "4", "text": _DOC_A_NEAR},
            {"id": "5", "text": _DOC_C},
        ]
        in_path.write_text("\n".join(json.dumps(r) for r in records))
        stats = dedup_jsonl(str(in_path), str(out_path), id_field="id")
        # 3 unique docs (A, B, C) — the near-duplicate of A is dropped.
        assert stats["indexed"] == 3
        survivors = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        assert {r["id"] for r in survivors} == {"1", "3", "5"}


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------


class TestContaminationNgrams:
    def test_normalise_lowercases_and_strips_punct(self):
        assert _normalise("Hello, World!!  ") == "hello world"

    def test_ngrams_word_n(self):
        out = list(_ngrams("the quick brown fox jumps", n=3))
        assert out == [
            "the quick brown",
            "quick brown fox",
            "brown fox jumps",
        ]

    def test_ngrams_too_short_returns_empty(self):
        assert list(_ngrams("only four words here", n=10)) == []


class TestContaminationFilter:
    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            ContaminationFilter(["arc", "made_up"])

    def test_custom_text_flags_match(self):
        # Skip HF loading entirely by injecting via add_custom_text.
        f = ContaminationFilter(benchmarks=["arc"], n=5)
        # Don't auto-build from HF; add a known "eval" string by hand.
        f.add_custom_text("Photosynthesis converts light energy into chemical energy.")
        assert f.is_contaminated(
            "Some intro text. Photosynthesis converts light energy into "
            "chemical energy. End."
        )
        # Disjoint doc must not be flagged.
        assert not f.is_contaminated("The capital of France is Paris.")

    def test_empty_text_not_flagged(self):
        f = ContaminationFilter(benchmarks=["arc"], n=5)
        f.add_custom_text("Photosynthesis converts light into chemical energy.")
        assert f.is_contaminated("") is False
        assert f.is_contaminated("   \n  ") is False

    def test_stream_filter_drops_contaminated(self):
        f = ContaminationFilter(benchmarks=["arc"], n=5)
        f.add_custom_text("the mitochondria is the powerhouse of the cell")
        docs = [
            ("1", "Random text about programming languages and compilers."),
            ("2", "It is well known that the mitochondria is the powerhouse of the cell, and"
                  " this fact is widely taught."),
            ("3", "Completely unrelated content about ocean currents."),
        ]
        kept = list(f.filter_stream(docs))
        assert [k for k, _ in kept] == ["1", "3"]


# ---------------------------------------------------------------------------
# Tokenizer eval
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """
    Test-only tokenizer that emulates different compression levels.

    `mode="word"` splits on whitespace (~4 chars/token on English prose).
    `mode="char"` returns one token per character (1 char/token — the
    worst-compression baseline). `str.split("")` raises in Python so we
    can't use the empty-separator trick; the explicit branch keeps the
    contract clear.
    """

    def __init__(self, mode: str, vocab_size: int = 1000):
        if mode not in ("word", "char"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.mode = mode
        self.vocab_size = vocab_size

    def encode(self, text: str):
        if not text:
            return []
        return text.split() if self.mode == "word" else list(text)


class TestTokenizerEval:
    def test_compare_produces_summary_per_tokenizer(self, monkeypatch):
        # Replace the HF loader with our fake to avoid network.
        from data import tokenizer_eval as te

        def _fake_loader(name: str):
            return _FakeTokenizer(mode=name)

        monkeypatch.setattr(te, "_load_hf_tokenizer", _fake_loader)
        report = te.compare_tokenizers(
            ["word", "char"],
            samples={"prose": "the quick brown fox" * 10},
            weights={"prose": 1.0},
        )
        assert set(report["summary"]) == {"word", "char"}
        # char-split → 1 char per token; word-split → ~4 chars per token.
        # higher chars_per_token = better compression.
        word = report["summary"]["word"]["weighted_chars_per_token"]
        char = report["summary"]["char"]["weighted_chars_per_token"]
        assert word > char

    def test_load_failure_recorded_not_raised(self, monkeypatch):
        from data import tokenizer_eval as te
        monkeypatch.setattr(te, "_load_hf_tokenizer", lambda name: None)
        report = te.compare_tokenizers(["bogus/path"])
        assert report["summary"]["bogus/path"] == {"error": "load_failed"}

    def test_builtin_samples_cover_four_domains(self):
        # The builtin set should expose at least the four domains we mix
        # in MixedDataset — eval should remain meaningful even offline.
        assert {"prose", "code", "math", "instruction"}.issubset(_BUILTIN_SAMPLES)
