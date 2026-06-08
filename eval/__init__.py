"""MythOuro evaluation harness.

Public entry point: `eval.harness.run_eval(model, tokenizer, ...)`.
Individual metric helpers live in `eval.metrics`.
"""

from eval.harness import run_eval
from eval.metrics import (
    perplexity,
    arc_challenge,
    gsm8k,
    loop_efficiency,
    expected_calibration_error,
)

__all__ = [
    "run_eval",
    "perplexity",
    "arc_challenge",
    "gsm8k",
    "loop_efficiency",
    "expected_calibration_error",
]
