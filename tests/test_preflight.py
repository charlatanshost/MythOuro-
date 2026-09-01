"""Preflight assertions must catch the three 2026-08 silent failures."""
import json, os, tempfile, pytest
from mythouro.preflight import (
    PreflightError, check_teacher_shard_schemas, check_corpus_dirs,
    check_growth_metadata, check_corpus_rows,
)


def _shard(path, keys, n=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(json.dumps({k: ("x" if k != "seed_len" else 1) for k in keys}) + "\n")


class TestSchemaMismatch:
    """Bug #3 — med/v2 carried seed_len, code/math did not; HF dropped them."""

    def test_the_real_2026_08_31_failure_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            _shard(f"{d}/code/shard_0.jsonl", ["text", "source"])
            _shard(f"{d}/math/shard_0.jsonl", ["text", "source"])
            _shard(f"{d}/med/shard_0.jsonl", ["text", "source", "seed_len"])
            _shard(f"{d}/v2/shard_0.jsonl", ["text", "source", "seed_len"])
            with pytest.raises(PreflightError) as e:
                check_teacher_shard_schemas(f"{d}/*/shard_*.jsonl")
            m = str(e.value)
            assert "MORE THAN ONE SCHEMA" in m
            assert "normalize_teacher_shards" in m      # says how to fix it
            assert "med" in m and "code" in m           # names the culprits

    def test_consistent_schemas_pass(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("code", "math", "med", "v2"):
                _shard(f"{d}/{n}/shard_0.jsonl", ["text", "source"])
            r = check_teacher_shard_schemas(f"{d}/*/shard_*.jsonl")
            assert r["dirs"] == 4 and r["files"] == 4

    def test_no_files_is_an_error_not_a_silent_pass(self):
        with pytest.raises(PreflightError):
            check_teacher_shard_schemas("/nonexistent/*.jsonl")


class TestCorpusDirs:
    def test_glob_that_misses_a_directory_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            _shard(f"{d}/code/shard_0.jsonl", ["text"])
            _shard(f"{d}/math/shard_0.jsonl", ["text"])
            with pytest.raises(PreflightError) as e:
                check_corpus_dirs(f"{d}/*/shard_*.jsonl", expected=4)
            assert "NOT the corpus the run intends" in str(e.value)

    def test_matching_count_passes(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("a", "b"):
                _shard(f"{d}/{n}/shard_0.jsonl", ["text"])
            check_corpus_dirs(f"{d}/*/shard_*.jsonl", expected=2)


class TestGrowthMetadata:
    """Bug #1 — dropped on save; resume then trains a bigger identical model."""

    class Cfg:
        def __init__(self, n): self.n_experts = n

    def test_grown_model_without_metadata_is_refused(self):
        with pytest.raises(PreflightError) as e:
            check_growth_metadata({}, self.Cfg(48), source_variant_experts=24)
        assert "bigger, slower and identical" in str(e.value)

    def test_grown_model_with_metadata_passes(self):
        check_growth_metadata({"growth_metadata": {"source_n_experts": 24}},
                              self.Cfg(48), source_variant_experts=24)

    def test_ungrown_model_needs_no_metadata(self):
        check_growth_metadata({}, self.Cfg(24), source_variant_experts=24)
        check_growth_metadata(None, self.Cfg(24), source_variant_experts=24)


class TestCorpusRows:
    def test_short_corpus_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            _shard(f"{d}/a/shard_0.jsonl", ["text"], n=3)
            with pytest.raises(PreflightError):
                check_corpus_rows(f"{d}/a/shard_*.jsonl", minimum=100)

    def test_sufficient_corpus_passes(self):
        with tempfile.TemporaryDirectory() as d:
            _shard(f"{d}/a/shard_0.jsonl", ["text"], n=50)
            assert check_corpus_rows(f"{d}/a/shard_*.jsonl", minimum=10) == 50
