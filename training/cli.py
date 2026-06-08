"""
Console-script entry points for MythOuro training.

These exist because the training script filenames (`3b_fine_web_edu.py`,
`1b_fine_web_edu.py`) start with a digit and so aren't importable as
Python modules. The `runpy` shim lets `pyproject.toml`'s
`[tool.poetry.scripts]` table point at them anyway.

CLI argument handling is left to the underlying scripts — when invoked
via `mythouro-train --eval ...`, `sys.argv[1:]` reaches the script's
own `argparse` parser unchanged.
"""

from __future__ import annotations

import os
import runpy
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)


def _run_script(rel_path: str) -> None:
    """Execute a script file as if it were called by `python <path>`."""
    path = os.path.join(_REPO_ROOT, rel_path) if os.path.isabs(rel_path) is False and not rel_path.startswith("training") \
           else os.path.join(_REPO_ROOT, rel_path)
    if not os.path.exists(path):
        sys.exit(f"openmythos: training script not found at {path!r}")
    runpy.run_path(path, run_name="__main__")


def fineweb_3b_main() -> None:
    """`mythouro-train` — start/resume FineWebEdu pretraining (3B variant).

    Accepts the same CLI flags as `python training/3b_fine_web_edu.py`:
        --eval / -e
        --eval-every N
        --eval-max-samples N
        --eval-benchmarks ...
    """
    _run_script(os.path.join("training", "3b_fine_web_edu.py"))


def fineweb_1b_main() -> None:
    """`mythouro-train-1b` — same as the 3B entry point but on the 1B variant."""
    _run_script(os.path.join("training", "1b_fine_web_edu.py"))


def tiny_main() -> None:
    """`mythouro-train-tiny` — single-sentence memorisation sanity check.

    Sub-second to seconds end-to-end on CUDA; the canonical "is the
    architecture wired correctly" smoke test before kicking off a real run.
    """
    _run_script("train_tiny_mythos.py")


def distill_main() -> None:
    """`mythouro-distill` — distil a frozen teacher into an MythOuro student.

    See `training/distill.py` for required `--teacher-id`, alpha /
    temperature / variant flags. Teacher and student tokenisers MUST
    match — the loader refuses to return a mismatched pair.
    """
    _run_script(os.path.join("training", "distill.py"))
