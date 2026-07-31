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

⚠ L1 AND L2 ARE THE SAME RUNG IN PRACTICE — measured, not theorised. Across the
46k-66k sweep L1+ equalled L2+ at all 11 checkpoints, and again at n=8. The cause
is structural: the PROMPT supplies the `def fname(...):` line, so any parseable
prefix already defines the function, while a prefix short enough to lose the def
line does not parse at all (a bare `def f(a):` with no body is a SyntaxError) and
scores L0. So L1-without-L2 is very nearly unreachable by construction. Read the
ladder as L0 / L1-2 / L3 / L4, and track L3 and L4 — those are the live rungs.

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

# ── FRAMING: does the prompt look like the data the model was TRAINED on? ──
#
# The distillation mix feeds `codeparrot/codeparrot-clean` for code — whole
# GitHub Python FILES: module docstring, imports, several functions in sequence.
# The eval has always prompted with a BARE signature and expected a solved body,
# which is a task the training distribution never demonstrates. So a persistent
# L4=0 may be measuring a FORMAT MISMATCH rather than a capability wall — and
# that distinction decides whether `grow_width.py` is the right next spend.
#
# `--framing file` wraps the same signature in realistic file context. Same
# tasks, same ladder, same checks: only the door the model is asked to come
# through changes. Compare bare-vs-file on ONE checkpoint; if L3/L4 jump, the
# wall is the framing.
#
# The preamble is deliberately generic and unrelated to any task (no helper the
# model could crib from), so it supplies CONTEXT, not answers.
_FILE_PREAMBLE = '''"""Small utility helpers."""
import math
import os


def _clamp(value, low, high):
    """Bound `value` to the inclusive range [low, high]."""
    return max(low, min(value, high))


'''

_RUNNER = """\
import sys
{code}
try:
{checks}
except Exception as e:
    print("FAIL:" + type(e).__name__); sys.exit(2)
print("OK")
"""


def _truncate_to_parseable(src: str) -> "tuple[str | None, int]":
    """Longest prefix of whole lines that parses, plus how many lines were DROPPED.

    Generations end mid-statement, so some truncation is normal and meaningless.
    The count matters for a different reason — see `_max_line_repeat`.
    """
    lines = src.split("\n")
    for cut in range(len(lines), 0, -1):
        cand = "\n".join(lines[:cut])
        if not cand.strip():
            continue
        try:
            ast.parse(cand)
            return cand, len(lines) - cut
        except (SyntaxError, ValueError):
            continue
    return None, len(lines)


def _max_line_repeat(src: str) -> int:
    """Largest number of times any single non-blank line appears verbatim.

    THE REPETITION ATTRACTOR, MADE NUMERIC. This exists because of what the
    46k-66k sweep showed: L3+ rose from 1/10 to 7/10 while L4 stayed at 0/10 in
    all 110 task-evaluations. But `_truncate_to_parseable` SALVAGES a runnable
    stub from a looping generation — `if n == 0:\\n    return 0` repeated forever
    truncates to a function that defines, executes, and returns the wrong answer,
    i.e. scores L3. So a rising L3 is ambiguous: real progress, or just cleaner
    stubs to salvage? This number disambiguates. >=3 identical lines means the
    generation looped, and its L3 was salvage rather than a finished function.
    """
    from collections import Counter
    lines = [ln.strip() for ln in src.split("\n") if ln.strip()]
    return max(Counter(lines).values(), default=0)


def _body_stmts(tree: ast.AST, fname: str) -> int:
    """Statements in the target function's body — a complete-ness proxy."""
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == fname:
            return len(n.body)
    return 0


def score_sample(prompt: str, completion: str, fname: str,
                 checks: str, timeout: float = 5.0) -> dict:
    """Score one generation: ladder rung 0-4 plus repetition diagnostics.

    Returns a dict rather than a bare int so every run records WHY a rung was
    reached. Re-analysis then costs no card time — the old version kept only the
    rung and 110 truncated characters, so any new question meant regenerating.
    """
    raw = (prompt + completion).replace("\\n", "\n").replace("\\t", "\t")
    # Two repetition figures. `max_line_repeat` counts the whole prompt+completion
    # and is what every report before 2026-07-31 used — keep it for continuity.
    # `max_line_repeat_completion` counts ONLY what the model produced, which is
    # the figure to use when comparing framings: a long `--framing file` preamble
    # adds distinct lines and would otherwise make the same looping look milder.
    d = {"rung": 0, "max_line_repeat": _max_line_repeat(raw),
         "max_line_repeat_completion": _max_line_repeat(
             completion.replace("\\n", "\n").replace("\\t", "\t")),
         "lines_dropped": 0, "body_stmts": 0, "completion": completion}
    code, dropped = _truncate_to_parseable(raw)
    d["lines_dropped"] = dropped
    if code is None:
        return d                                          # L0
    rung = 1                                              # L1 syntax
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return d
    d["rung"] = rung
    if not any(isinstance(n, ast.FunctionDef) and n.name == fname
               for n in ast.walk(tree)):
        return d                                          # stops at L1
    rung = 2                                              # L2 defines
    d["rung"] = rung
    d["body_stmts"] = _body_stmts(tree, fname)
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
            return d                                      # hang = no further credit
    if r.returncode == 0 and "OK" in r.stdout:
        d["rung"] = 4                                     # L4 correct
    elif "FAIL:AssertionError" in r.stdout:
        d["rung"] = 3                                     # L3 ran, wrong answer
    return d                                              # else stays L2


@torch.no_grad()
def generate(model, tok, prompt: str, device: str, max_new: int,
             n_loops: int, temperature: float, rep_penalty: float = 1.0) -> str:
    """Sample a continuation, optionally with a CTRL-style repetition penalty.

    WHY THE PENALTY IS HERE — it is a DIAGNOSTIC, not a scoring aid. At step
    64,000, 69% of samples loop (median 6 identical lines) and 77% of L3 scores
    are salvaged out of a loop rather than earned by finished code. That leaves
    two very different explanations, which this flag separates:
      penalty lifts L4 off zero -> the model CAN finish a function; the attractor
                                   is a decoding failure, and the fix is cheap.
      L4 stays at zero          -> it is a training problem (unlikelihood/DiverseKD),
                                   and more tokens alone will not get there.
    Always report the penalty alongside results; a penalised number is not
    comparable to the unpenalised 46k-66k sweep.
    """
    prompt_ids = tok.encode(prompt)
    ids = torch.tensor([prompt_ids], device=device)
    out = []
    for _ in range(max_new):
        o = model(ids, n_loops=n_loops)
        logits = (o[0] if isinstance(o, (tuple, list)) else o)[0, -1].float()
        if rep_penalty != 1.0 and out:
            # Keskar et al. 2019 style, with TWO deliberate deviations:
            #
            # (1) COUNT-SCALED, not fixed. Plain CTRL applies one flat divisor per
            #     distinct seen token, which measurably fails against a SUSTAINED
            #     loop — the exact thing we are probing. Verified on a fake model
            #     whose argmax is pinned to one token: a flat penalty breaks the
            #     loop for a single step and then reverts forever, because once the
            #     alternative has also been seen both are divided equally and the
            #     attractor still wins. Scaling by occurrence count makes the
            #     pressure escalate the longer a token repeats, so it cannot sit in
            #     a loop indefinitely.
            # (2) GENERATED TOKENS ONLY, never the prompt. The prompt is
            #     `def add_two(a, b):` and a correct body must reuse `a` and `b`;
            #     penalising it would suppress exactly the tokens correct code needs
            #     and manufacture a failure we would then misread as model weakness.
            #
            # Sampling already runs on CPU (topk/multinomial segfault on XPU, see
            # field notes), so stay on CPU from here.
            logits = logits.cpu()
            seen, counts = torch.unique(torch.tensor(out), return_counts=True)
            scale = rep_penalty ** counts.float()
            v = logits[seen]
            logits[seen] = torch.where(v > 0, v / scale, v * scale)
        if temperature > 0:
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
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="1.0 = off (comparable to the 46k-66k sweep). >1 penalises "
                        "already-GENERATED tokens (never the prompt). Diagnostic: "
                        "does the repetition attractor hide real capability? Sweep "
                        "1.0/1.15/1.3 — too high destroys code, which is itself a "
                        "result worth recording.")
    p.add_argument("--framing", choices=("bare", "file"), default="bare",
                   help="'bare' = prompt is just the signature (the historical "
                        "default; every report before 2026-07-31 used it). "
                        "'file' = the same signature wrapped in realistic Python "
                        "FILE context, matching codeparrot-clean, which is what "
                        "the model is actually trained on. Run both on ONE "
                        "checkpoint: if L3/L4 jump under 'file', the L4=0 wall is "
                        "a format mismatch, not a capability limit — which "
                        "changes whether growing the model is the right spend.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the sampler so RE-MEASURING A CHECKPOINT IS "
                        "REPRODUCIBLE. Added 2026-07-31 after an unseeded rerun "
                        "of the SAME weights gave task-level L3+ 4/10 then 7/10 "
                        "and a training run was stopped over the difference. "
                        "Use the same seed to compare checkpoints; VARY it (or "
                        "omit) to measure the instrument's own spread.")
    p.add_argument("--json", default=None, help="Also write results here.")
    args = p.parse_args()

    from mythouro.tokenizer import MythOuroTokenizer
    enc = MythOuroTokenizer(args.tokenizer)
    tok = enc.tokenizer
    model, _cfg, step = _load_model(args.checkpoint, args.device)
    model.eval()
    if args.seed is not None:
        torch.manual_seed(args.seed)        # sampling is on CPU, so this is enough
    print(f"checkpoint step {step} | {args.samples} samples/task | "
          f"T={args.temperature} | rep_penalty={args.repetition_penalty} | "
          f"seed={args.seed} | framing={args.framing}")
    if args.framing == "file":
        print("  prompts wrapped in file context — NOT comparable to bare-framing "
              "reports; compare the two on the SAME checkpoint")
    print()

    names = {0: "L0 nothing", 1: "L1 syntax", 2: "L2 defines",
             3: "L3 runs", 4: "L4 correct"}
    rows, hist, all_samples = [], {k: 0 for k in names}, []
    for fname, prompt, checks in _TASKS:
        # ONE transform, used for BOTH generation and scoring — they must see
        # the identical prompt or the ladder scores a different string than the
        # model continued.
        prompt = _FILE_PREAMBLE + prompt if args.framing == "file" else prompt
        best, best_txt, samples = 0, "", []
        for _ in range(args.samples):
            comp = generate(model, tok, prompt, args.device, args.max_new,
                            args.n_loops, args.temperature,
                            args.repetition_penalty)
            d = score_sample(prompt, comp, fname, checks)
            samples.append(d)
            if d["rung"] > best:
                best, best_txt = d["rung"], comp
        hist[best] += 1
        all_samples.extend(samples)
        looped = sum(1 for s in samples if s["max_line_repeat"] >= 3)
        rows.append({"task": fname, "rung": best, "label": names[best],
                     "looped": looped, "samples": samples})
        print(f"  {fname:16} {names[best]:12} looped {looped}/{len(samples)}")
        if best >= 2:
            print(f"       {best_txt[:110].strip()!r}")

    n = len(_TASKS)
    print("\n  ladder distribution (best-of-%d per task):" % args.samples)
    for k in sorted(names):
        bar = "#" * hist[k]
        print(f"    {names[k]:12} {hist[k]:>2}/{n}  {bar}")
    reach = lambda r: sum(v for k, v in hist.items() if k >= r)
    # ── PRIMARY NUMBER: per-sample L3+ over ALL generations ──
    # Task-level best-of-k over 10 tasks is a COARSE, high-variance statistic:
    # two unseeded runs of the SAME 66,000 checkpoint gave 4/10 and 7/10, and a
    # training run was stopped over that difference (2026-07-31). The per-sample
    # rate uses samples*10 observations instead of 10 and is what actually
    # separated the reuse=8 trajectory (21% -> 42% -> 54%) from its control.
    # Quote THIS with its interval; task-level L3+ is context, not the headline.
    tot_s = len(all_samples)
    k_l3 = sum(1 for s in all_samples if s["rung"] >= 3)
    p_l3 = k_l3 / tot_s if tot_s else 0.0
    ci95 = 1.96 * (p_l3 * (1 - p_l3) / tot_s) ** 0.5 if tot_s else 0.0
    print(f"\n  ** per-sample L3+ : {k_l3}/{tot_s} = {100 * p_l3:.1f}% "
          f"±{100 * ci95:.1f} (95% CI) **   <- PRIMARY, quote this")
    print(f"  task-level (best-of-{args.samples}, n=10 — coarse, high variance):"
          f" L2+ {reach(2)}/{n}  L3+ {reach(3)}/{n}  L4 {reach(4)}/{n}")
    print("  ⚠ L1 alone is weak — `def f(n): return 8553.99` parses. Track L2+.")
    if args.seed is None:
        print("  ⚠ UNSEEDED — a rerun draws fresh samples. Pass --seed to compare "
              "checkpoints reproducibly.")

    # Repetition diagnostics: is a rising L3 real progress, or just cleaner
    # stubs salvaged out of looping generations? `looped_in_L3` is the number to
    # watch — if it falls while L3+ holds, the model is finishing functions.
    tot = len(all_samples)
    looped = sum(1 for s in all_samples if s["max_line_repeat"] >= 3)
    l3s = [s for s in all_samples if s["rung"] == 3]
    l3_looped = sum(1 for s in l3s if s["max_line_repeat"] >= 3)
    med_rep = sorted(s["max_line_repeat"] for s in all_samples)[tot // 2]
    diag = {"samples_total": tot, "looped": looped,
            "per_sample_l3plus": round(p_l3, 4), "per_sample_l3plus_ci95": round(ci95, 4),
            "looped_frac": round(looped / tot, 3) if tot else 0.0,
            "median_max_line_repeat": med_rep,
            "l3_samples": len(l3s), "l3_looped": l3_looped,
            "l3_looped_frac": round(l3_looped / len(l3s), 3) if l3s else None}
    print(f"\n  repetition: {looped}/{tot} samples looped (>=3 identical lines), "
          f"median max-repeat {med_rep}")
    if l3s:
        print(f"  of L3 samples, {l3_looped}/{len(l3s)} were SALVAGED from a loop "
              f"({100 * l3_looped / len(l3s):.0f}%) — the rest were finished code.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"step": step, "temperature": args.temperature,
             "repetition_penalty": args.repetition_penalty,
             "seed": args.seed,
             "framing": args.framing,
             "samples": args.samples, "tasks": rows,
             "reached": {f"L{r}+": reach(r) for r in (1, 2, 3, 4)},
             "diagnostics": diag}, indent=2))
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
