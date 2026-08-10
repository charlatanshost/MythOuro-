"""
code_eval's degeneracy metrics — and the blind spot that motivated the third one.

On 2026-08-10 a collapsed SFT checkpoint produced this, and EVERY metric the tool
had scored it healthy (`max_line_repeat` 1, `lrs_frac` 0.071, `char_degenerate`
0, `looped_frac` 0.013):

    is is is not only only one one one one one of them:):
    ThisThisThis_is_tweakcasecasecase(1000) as as as many, but but then

`_max_line_repeat` counts identical LINES. `_lrs_frac` finds the longest
substring occurring twice and reads LOW here precisely because the repeats are
short and local. Neither fires on ADJACENT duplicates — the actual failure mode
of a collapsing run — so 48 of 80 samples were certified clean by both.

These tests pin the fix and, more importantly, pin the FALSE-POSITIVE side: a
degeneracy flag that fires on correct code is worse than no flag, because it
would mark real progress as collapse.
"""
import sys

import pytest

sys.path.insert(0, "tools")
from code_eval import _adjacent_repeat, _lrs_frac, _max_line_repeat  # noqa: E402


# Real completions that scored L3+ at step 108,471 — these MUST stay clean.
CORRECT = [
    "return a + b",
    "def add_two(a, b):\n    return a + b",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "return [[0] * n for _ in range(n)]",
    "return sorted(set(x for x in xs if x > 0))",
    "if not xs:\n    return None\nreturn max(xs)",
]

# Verbatim from reports/code_v8_bigbatch_2127_pen1.15.json.
COLLAPSED = [
    "is is is not only only one one one one one of them:):",
    "ThisThisThis_is_tweakcasecasecase(1000) as as as many, but but then",
    "def def __init__(self, self): # #.",
    "** ** ** ** ** **",
]

# The subset the OLDER metrics genuinely miss. `** ** ** ** ** **` is excluded
# on purpose: in isolation its lrs_frac is 0.824, so the character metric DOES
# catch it — it only slipped through in the real run because it sat inside a
# much longer completion that diluted the ratio. Overstating the blind spot
# would make this test lie in the tool's favour.
BLIND_TO_OLDER = COLLAPSED[:3]


class TestNoFalsePositives:
    @pytest.mark.parametrize("src", CORRECT)
    def test_correct_code_is_not_flagged(self, src):
        frac, _ = _adjacent_repeat(src)
        assert frac < 0.10, f"correct code flagged as degenerate: {frac}"

    def test_short_and_empty_inputs_are_safe(self):
        for src in ("", " ", "x", "return"):
            frac, run = _adjacent_repeat(src)
            assert frac == 0.0 and run == 1


class TestCatchesTheBlindSpot:
    @pytest.mark.parametrize("src", COLLAPSED)
    def test_collapsed_output_is_flagged(self, src):
        frac, _ = _adjacent_repeat(src)
        assert frac >= 0.10, f"degenerate output not flagged: {frac}"

    @pytest.mark.parametrize("src", BLIND_TO_OLDER)
    def test_the_older_metrics_really_are_blind(self, src):
        """
        The regression this guards. If someone later 'improves' _lrs_frac or
        _max_line_repeat into catching these, that is fine — but this test
        documents that on 2026-08-10 they did NOT, which is why the third
        metric exists. It fails loudly if the premise stops holding, rather
        than leaving a redundant metric nobody understands.
        """
        assert _max_line_repeat(src) < 3
        assert _lrs_frac(src) < 0.35

    def test_within_token_periodicity(self):
        """'casecasecase' == 'case' * 3 — degenerate with no whitespace at all."""
        assert _adjacent_repeat("casecasecase")[0] >= 0.10
        assert _adjacent_repeat("ThisThisThis")[0] >= 0.10
        # ...but a word that merely CONTAINS a repeat is not periodic.
        assert _adjacent_repeat("indentation")[0] == 0.0
        assert _adjacent_repeat("banana")[0] == 0.0

    def test_run_length_is_reported(self):
        _, run = _adjacent_repeat("one one one one one")
        assert run == 5
