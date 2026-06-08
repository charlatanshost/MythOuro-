"""
Unified CLI dispatcher for the MythOuro data pipeline.

Usage:
    python -m data dedup        --input X.jsonl --output Y.jsonl
    python -m data contamination --input X.jsonl --output Y.jsonl
    python -m data tokenizer-eval --use-hf-samples

Each subcommand defers to the per-module CLI (`_main(argv)`). Argument
parsing is left to each module so subcommand help (`python -m data dedup
--help`) reaches the right argparse instance.
"""

from __future__ import annotations

import sys


_SUBCOMMANDS = {
    "dedup":          "data.dedup",
    "contamination":  "data.contamination",
    "tokenizer-eval": "data.tokenizer_eval",
    "tokenizer_eval": "data.tokenizer_eval",     # underscore alias
}


def _print_help() -> None:
    print("Usage: python -m data <subcommand> [--help] [args...]")
    print()
    print("Subcommands:")
    print("  dedup            MinHash LSH near-duplicate removal")
    print("  contamination    Drop docs overlapping ARC / GSM8K / HumanEval")
    print("  tokenizer-eval   Compare candidate tokenizers' compression")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        return 0
    sub = sys.argv[1]
    if sub not in _SUBCOMMANDS:
        print(f"unknown subcommand: {sub!r}\n")
        _print_help()
        return 1
    import importlib
    module = importlib.import_module(_SUBCOMMANDS[sub])
    return module._main(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main() or 0)
