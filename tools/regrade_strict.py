"""
Re-grade SAVED code_eval completions against much stronger tests. No GPU, no
regeneration — it reads the report JSONs that already exist.

WHY. 2026-08-21: the α-anneal comparison was run entirely on L3+, which means
"parses and runs on the check inputs" — NOT "answers the question". Reading the
completions behind the 82.5% "record" at step 143,500 found guard clauses and
topic drift: `add_two` -> `if a == b: return 0`; `sum_list` -> `if not nums:
return 0`; `double_it` -> a comment about binary search. 66/80 graded L3, 0/80
graded L4.

Worse, L4 itself has false positives, because the shipped checks are thin. The
is_even test is TWO assertions on TWO values:

    assert is_even(4) is True or is_even(4) == True
    assert not is_even(3)

`return (n % 2 == 0) and (n % 3 == 1)` passes both and is wrong. It was graded L4
at step 149,500.

WHAT THIS DOES. Reuses `score_sample` from tools.code_eval unchanged — same
truncate-to-parseable, same AST checks, same subprocess runner — and swaps in the
strict suites below. So any difference in the result is the TESTS, nothing else.

    python -m tools.regrade_strict                      # all bare-framed reports
    python -m tools.regrade_strict --glob 'reports/code_anneal*_pen115.json'
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.code_eval import score_sample, _TASKS            # noqa: E402

_PROMPTS = {n: p for n, p, _ in _TASKS}

# Strict suites. Edge cases first — zero, empty, negative, identity — because the
# fragments this is meant to catch are guard clauses that return early.
_STRICT = {
    "add_two": "assert add_two(2, 3) == 5\nassert add_two(-1, 1) == 0\n"
               "assert add_two(0, 0) == 0\nassert add_two(100, -50) == 50\n"
               "assert add_two(-4, -6) == -10",
    "double_it": "assert double_it(4) == 8\nassert double_it(0) == 0\n"
                 "assert double_it(-3) == -6\nassert double_it(7) == 14\n"
                 "assert double_it(1) == 2",
    "get_first": "assert get_first([7, 8, 9]) == 7\nassert get_first([1]) == 1\n"
                 "assert get_first(['a', 'b']) == 'a'\n"
                 "assert get_first([3, 1, 2]) == 3",
    "sum_list": "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\n"
                "assert sum_list([-1, 1]) == 0\nassert sum_list([5]) == 5\n"
                "assert sum_list([10, 20, 30]) == 60",
    "count_items": "assert count_items([1, 2, 3]) == 3\nassert count_items([]) == 0\n"
                   "assert count_items([1]) == 1\n"
                   "assert count_items(['a', 'b', 'c', 'd']) == 4",
    "is_even": "assert is_even(4) == True\nassert is_even(3) == False\n"
               "assert is_even(0) == True\nassert is_even(7) == False\n"
               "assert is_even(-2) == True\nassert is_even(1) == False\n"
               "assert is_even(100) == True\nassert is_even(99) == False",
    "reverse_string": "assert reverse_string('abc') == 'cba'\n"
                      "assert reverse_string('') == ''\n"
                      "assert reverse_string('a') == 'a'\n"
                      "assert reverse_string('hello') == 'olleh'",
    "max_of_two": "assert max_of_two(3, 9) == 9\nassert max_of_two(5, 2) == 5\n"
                  "assert max_of_two(0, 0) == 0\nassert max_of_two(-5, -2) == -2\n"
                  "assert max_of_two(7, 7) == 7",
    "fibonacci": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\n"
                 "assert fibonacci(2) == 1\nassert fibonacci(3) == 2\n"
                 "assert fibonacci(6) == 8\nassert fibonacci(10) == 55",
    "is_prime": "assert is_prime(7) == True\nassert is_prime(8) == False\n"
                "assert is_prime(2) == True\nassert is_prime(1) == False\n"
                "assert is_prime(9) == False\nassert is_prime(13) == True\n"
                "assert is_prime(4) == False",
}


def regrade(path: str) -> dict:
    d = json.load(open(path))
    if d.get("chat_template"):
        return {}                       # bare-framed only; chat completions are extracted
    n = l3 = l4 = l4_old = 0
    fixed = []
    for t in d["tasks"]:
        fname = t["task"]
        prompt, checks = _PROMPTS.get(fname), _STRICT.get(fname)
        if not prompt or not checks:
            continue
        for s in t["samples"]:
            n += 1
            l4_old += s["rung"] == 4
            r = score_sample(prompt, s["completion"], fname, checks)["rung"]
            l3 += r >= 3
            l4 += r == 4
            if s["rung"] == 4 and r != 4:
                fixed.append(f"{fname} (was L4, now L{r})")
    return {"n": n, "l3": l3, "l4": l4, "l4_old": l4_old, "downgraded": fixed}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--glob", default="reports/code_*_pen115.json")
    a = p.parse_args()
    rows = []
    for f in sorted(_glob.glob(a.glob)):
        m = re.search(r"_(\d{6})_", f)
        r = regrade(f)
        if r:
            rows.append((int(m.group(1)) if m else 0, os.path.basename(f), r))
    rows.sort()
    print(f"\n  {'step':>8}  {'L3+ (runs)':>11}{'L4 shipped':>12}{'L4 STRICT':>11}")
    for step, name, r in rows:
        print(f"  {step:>8}  {100*r['l3']/r['n']:>10.1f}%{r['l4_old']:>9}/{r['n']}"
              f"{r['l4']:>8}/{r['n']}")
    print()
    for step, name, r in rows:
        for f in r["downgraded"]:
            print(f"  downgraded @{step}: {f}")


if __name__ == "__main__":
    main()
