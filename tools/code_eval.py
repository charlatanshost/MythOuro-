"""
Executable code eval — the project's first CAPABILITY metric.

Everything we measure today is a *degeneracy floor* (`top_share`, `distinct1`,
self-repetition): it tells us whether output is broken, never whether it is
RIGHT. That gap cost real time — on 2026-07-29/30 the same generation was scored
"better", then "worse", then "better for a different reason", because the metrics
are blind to correctness and the text had to be argued about. Code is the one
domain where that is fully fixable: **generate a function, RUN it, check the
answer.** No judge model, no interpretation.

⚠ GRADED LADDER, NOT PASS/FAIL — this is the key design choice.
At the current quality (`def n(n): return n`, arithmetic blobs) a binary
pass-rate would read 0% for months and show no progress. So each sample is scored
on the furthest rung it reaches:

    L0  nothing         — no parseable code at all
    L1  syntax          — prompt+completion parses as Python
    L2  defines         — actually defines the requested function
    L3  runs            — calling it on real args raises nothing
    L4  correct         — returns the right answers

L1→L3 will move long before L4 does, which is exactly what makes this usable now.
NOTE L1 alone is a weak signal: `def f(n): return 8553.99` parses fine — that is
why syntactic validity was rejected as a standalone metric (`relevance_probe`).
The ladder earns its keep from L2/L3/L4.

SAFETY: generated code is executed in a subprocess with a timeout, in a temp dir,
never in-process. It is model output, so treat it as untrusted.

Usage
-----
    python -m tools.code_eval -c checkpoints_onpolicy_fixed/step_0064000.pt \\
        --device xpu:0 --samples 3
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_checkpoint import _load_model                      # noqa: E402

# (name, prompt, call-and-check snippet). Deliberately EASY — the point is to
# detect the first flicker of capability, not to rank a frontier model. Several
# prompts match the generation-probe seeds so the two instruments line up.
_TASKS = [
    ("add_two", "def add_two(a, b):\n",
     "assert add_two(2, 3) == 5\nassert add_two(-1, 1) == 0"),
    ("double_it", "def double_it(n):\n",
     "assert double_it(4) == 8\nassert double_it(0) == 0"),
    ("get_first", "def get_first(items):\n",
     "assert get_first([7, 8, 9]) == 7"),
    ("sum_list", "def sum_list(nums):\n",
     "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0"),
    ("count_items", "def count_items(items):\n",
     "assert count_items([1, 2, 3]) == 3"),
    ("is_even", "def is_even(n):\n",
     "assert is_even(4) is True or is_even(4) == True\nassert not is_even(3)"),
    ("reverse_string", "def reverse_string(s):\n",
     "assert reverse_string('abc') == 'cba'"),
    ("max_of_two", "def max_of_two(a, b):\n",
     "assert max_of_two(3, 9) == 9\nassert max_of_two(5, 2) == 5"),
    ("fibonacci", "def fibonacci(n):\n",
     "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(6) == 8"),
    ("is_prime", "def is_prime(n):\n",
     "assert is_prime(7)\nassert not is_prime(8)"),
]

_RUNNER = """\
import sys
{code}
try:
{checks}
except Exception as e:
    print("FAIL:" + type(e).__name__); sys.exit(2)
print("OK")
"""


def _truncate_to_parseable(src: str) -> "str | None":
    """Longest prefix of whole lines that parses. Generations end mid-statement."""
    lines = src.split("\n")
    for cut in range(len(lines), 0, -1):
        cand = "\n".join(lines[:cut])
        if not cand.strip():
            continue
        try:
            ast.parse(cand)
            return cand
        except (SyntaxError, ValueError):
            continue
    return None


def score_sample(prompt: str, completion: str, fname: str,
                 checks: str, timeout: float = 5.0) -> int:
    """Return the highest ladder rung (0-4) this generation reaches."""
    raw = (prompt + completion).replace("\\n", "\n").replace("\\t", "\t")
    code = _truncate_to_parseable(raw)
    if code is None:
        return 0                                          # L0
    rung = 1                                              # L1 syntax
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return 0
    if not any(isinstance(n, ast.FunctionDef) and n.name == fname
               for n in ast.walk(tree)):
        return rung                                       # stops at L1
    rung = 2                                              # L2 defines
    # L3/L4 come from ONE execution of the real checks, distinguished by the
    # exception type. An earlier design smoke-called the function with `0` for
    # every argument to test L3 separately — that wrongly failed CORRECT string
    # and list functions (`reverse_string(0)` raises), i.e. it punished the very
    # code we want to reward. Using the real check args avoids all type guessing:
    #   AssertionError  -> the function RAN on valid inputs, answer wrong  -> L3
    #   any other error -> it blew up                                      -> L2
    #   clean exit      -> correct                                         -> L4
    # Subprocess only: this is untrusted model output.
    indented = "\n".join("    " + ln for ln in checks.split("\n"))
    script = _RUNNER.format(code=code, checks=indented)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.py")
        with open(path, "w") as fh:
            fh.write(script)
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=timeout, cwd=td)
        except subprocess.TimeoutExpired:
            return rung                                   # hang = no further credit
    if r.returncode == 0 and "OK" in r.stdout:
        return 4                                          # L4 correct
    if "FAIL:AssertionError" in r.stdout:
        return 3                                          # L3 ran, wrong answer
    return rung                                           # L2 defined but errored


@torch.no_grad()
def generate(model, tok, prompt: str, device: str, max_new: int,
             n_loops: int, temperature: float) -> str:
    ids = torch.tensor([tok.encode(prompt)], device=device)
    out = []
    for _ in range(max_new):
        o = model(ids, n_loops=n_loops)
        logits = (o[0] if isinstance(o, (tuple, list)) else o)[0, -1].float()
        if temperature > 0:
            # CPU sampling: topk/multinomial segfault on XPU (field notes).
            probs = (logits / temperature).cpu().softmax(-1)
            nxt = int(torch.multinomial(probs, 1))
        else:
            nxt = int(logits.argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--samples", type=int, default=3,
                   help="Generations per task; the ladder score is the BEST of "
                        "these (pass@k semantics).")
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.4,
                   help="0 = greedy. A little sampling helps a weak model find a "
                        "working form; report the temperature alongside results.")
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--json", default=None, help="Also write results here.")
    args = p.parse_args()

    from mythouro.tokenizer import MythOuroTokenizer
    enc = MythOuroTokenizer(args.tokenizer)
    tok = enc.tokenizer
    model, _cfg, step = _load_model(args.checkpoint, args.device)
    model.eval()
    print(f"checkpoint step {step} | {args.samples} samples/task | "
          f"T={args.temperature}\n")

    names = {0: "L0 nothing", 1: "L1 syntax", 2: "L2 defines",
             3: "L3 runs", 4: "L4 correct"}
    rows, hist = [], {k: 0 for k in names}
    for fname, prompt, checks in _TASKS:
        best, best_txt = 0, ""
        for _ in range(args.samples):
            comp = generate(model, tok, prompt, args.device, args.max_new,
                            args.n_loops, args.temperature)
            s = score_sample(prompt, comp, fname, checks)
            if s > best:
                best, best_txt = s, comp
        hist[best] += 1
        rows.append({"task": fname, "rung": best, "label": names[best]})
        print(f"  {fname:16} {names[best]}")
        if best >= 2:
            print(f"       {best_txt[:110].strip()!r}")

    n = len(_TASKS)
    print("\n  ladder distribution (best-of-%d per task):" % args.samples)
    for k in sorted(names):
        bar = "#" * hist[k]
        print(f"    {names[k]:12} {hist[k]:>2}/{n}  {bar}")
    reach = lambda r: sum(v for k, v in hist.items() if k >= r)
    print(f"\n  reached L2+ (defines fn): {reach(2)}/{n}"
          f"   L3+ (runs): {reach(3)}/{n}   L4 (CORRECT): {reach(4)}/{n}")
    print("  ⚠ L1 alone is weak — `def f(n): return 8553.99` parses. Track L2+.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"step": step, "temperature": args.temperature,
             "samples": args.samples, "tasks": rows,
             "reached": {f"L{r}+": reach(r) for r in (1, 2, 3, 4)}}, indent=2))
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
