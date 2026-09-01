"""
Startup assertions — fail in SECONDS on conditions that invalidate a run.

WHY THIS EXISTS. Three separate times in the week of 2026-08-25 a training job
ran for hours doing the wrong thing while logging only a WARNING:

1. `growth_metadata` was dropped by every `save_checkpoint` call, so any resume
   silently stopped applying the sentinel decay — producing a 397M model that
   routes exactly like its 278M source. Caught by an unrelated audit.
2. `tools.code_eval`'s documented `-c $DIR/step_0*.pt` expanded to 22 paths
   against an argparse `-c` that takes one. Would have died the moment a 9.5h
   leg finished.
3. `data_teacher_med` and `data_teacher_v2` carried a `seed_len` column the
   other corpora lacked, so HuggingFace's JSON loader dropped them ENTIRELY.
   410 skipped batches in 50 minutes while the corpus banner cheerfully printed
   all four directory names. `run_grown48_broadmix.sh` had never worked.

Every one of those is cheap to detect before a single GPU kernel launches. The
rule this module encodes: **a condition that invalidates the run is a hard
failure at startup, not a warning at step 40.**

Each check raises `PreflightError` with a message that says what to do about it.
Callers should let it propagate — do not catch and continue.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

from loguru import logger


class PreflightError(RuntimeError):
    """A condition that invalidates the run, detected before it starts."""


def _resolve(globs: str) -> "list[str]":
    return sorted(f for p in globs.split(",") for f in glob.glob(p.strip()))


def check_teacher_shard_schemas(globs: str, *, sample_rows: int = 200) -> dict:
    """Every teacher shard must expose the SAME JSON keys.

    `load_dataset("json", data_files=[...])` locks its schema to the first file
    in sorted order and refuses to cast the rest, dropping them with a warning.
    This is bug #3 above, and it silently halved the corpus for 50 minutes.
    """
    files = _resolve(globs)
    if not files:
        raise PreflightError(f"teacher corpus: no files match {globs!r}")
    schemas: "dict[tuple, list[str]]" = {}
    for f in files:
        keys = None
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i >= sample_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    k = tuple(sorted(json.loads(line)))
                except json.JSONDecodeError:
                    continue
                keys = k if keys is None else keys
                if k != keys:
                    raise PreflightError(
                        f"teacher corpus: {f} has rows with DIFFERENT keys "
                        f"({sorted(keys)} vs {sorted(k)}). Normalise it:\n"
                        f"    python3 -u -m tools.normalize_teacher_shards"
                    )
        if keys is not None:
            schemas.setdefault(keys, []).append(f)
    if len(schemas) > 1:
        lines = []
        for k, fs in schemas.items():
            dirs = sorted({os.path.dirname(f) or "." for f in fs})
            lines.append(f"    {list(k)}  <- {', '.join(dirs)}")
        raise PreflightError(
            "teacher corpus: shards carry MORE THAN ONE SCHEMA. HuggingFace's "
            "JSON loader locks to the first file and DROPS the rest with only a "
            "warning, so part of your corpus would silently never load.\n"
            + "\n".join(lines)
            + "\n  Fix:  python3 -u -m tools.normalize_teacher_shards"
        )
    keys = next(iter(schemas))
    by_dir = Counter(os.path.dirname(f) or "." for f in files)
    logger.info(
        "preflight: {} teacher shards, one schema {}, {} dirs — {}",
        len(files), list(keys), len(by_dir),
        ", ".join(f"{d}: {n}" for d, n in sorted(by_dir.items())),
    )
    return {"files": len(files), "dirs": len(by_dir), "keys": list(keys)}


def check_corpus_dirs(globs: str, expected: int) -> None:
    """The resolved directory count must match what the script intends.

    A glob that quietly matches fewer directories than the caller wrote is the
    difference between a broadened-corpus leg and a repeat of the narrow one.
    """
    files = _resolve(globs)
    dirs = sorted({os.path.dirname(f) or "." for f in files})
    if len(dirs) != expected:
        raise PreflightError(
            f"teacher corpus: expected {expected} directories, resolved "
            f"{len(dirs)} ({', '.join(dirs) or 'none'}). A glob missed — this "
            f"is NOT the corpus the run intends."
        )


def check_growth_metadata(extra: "dict | None", cfg, source_variant_experts: "int | None") -> None:
    """A grown checkpoint must still carry its growth metadata.

    Without it `apply_sentinel_to_router_biases` never runs, the new experts'
    bias freezes mid-decay, and the run produces a bigger model that routes
    exactly like its source. This is bug #1 above.
    """
    gm = (extra or {}).get("growth_metadata")
    n = int(getattr(cfg, "n_experts", 0))
    if gm is None and source_variant_experts and n > source_variant_experts:
        raise PreflightError(
            f"resume: model has n_experts={n} (> the {source_variant_experts} "
            f"of an ungrown model) but the checkpoint carries NO "
            f"growth_metadata. The sentinel decay would never run and the new "
            f"experts would never enter top-k. Refusing to train a model that "
            f"is bigger, slower and identical to its source."
        )


def check_corpus_rows(globs: str, minimum: int) -> int:
    """Total rows across the corpus must clear a floor."""
    n = 0
    for f in _resolve(globs):
        with open(f) as fh:
            n += sum(1 for line in fh if line.strip())
    if n < minimum:
        raise PreflightError(
            f"teacher corpus: {n:,} rows, expected at least {minimum:,}. "
            f"A shard is missing or truncated."
        )
    return n
