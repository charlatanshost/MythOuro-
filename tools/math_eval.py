"""
Executable math eval — the second CAPABILITY metric, after `code_eval`.

WHY MATH NEEDED ITS OWN INSTRUMENT. Math is 26% of the distillation mix and, on
the @70,500 probe, the worst-performing domain — but that judgement rested on ONE
seed read by eye. Code got an objective instrument; math had nothing, so we could
say it looked bad but not by how much, nor whether it was moving.

Math is in fact EASIER to measure than code: the answer is checkable without
executing anything, and problems can be authored with known answers.

⚠ GRADED LADDER, NOT PASS/FAIL — same reason as code_eval. A binary correct-rate
would read 0% for months and show no progress:

    L0  nothing      — no number in the output at all
    L1  a number     — emits some number
    L2  right form   — the right COUNT and TYPE of values (two roots for a
                       quadratic, one integer where one is expected)
    L3  buried       — the correct answer APPEARS somewhere but is not presented
                       as the answer
    L4  correct      — the correct answer, stated as the answer

L3 mirrors code_eval's "runs but wrong" and catches exactly the @70,500 failure:
`The roots of the equation ... are given:` is the right FRAME with no roots in it.

BUILT WITH code_eval's FOUR HARD-WON DEFECTS ALREADY FIXED (2026-07-29..31):
  1. `--seed` from the start — an unseeded rerun of identical weights gave
     L3+ 4/10 then 7/10 and stopped a healthy training run.
  2. per-sample rate as the PRIMARY statistic (samples*n_tasks observations),
     with task-level best-of-k demoted to context.
  3. every completion saved, so re-analysis costs no card time.
  4. `lrs_frac` (character-level repetition) alongside the line metric, because
     the line metric is blind to within-line degeneracy like `0.0000...000`.

Usage
-----
    python -m tools.math_eval -c checkpoints_reuse8/step_0070500.pt \\
        --device xpu:0 --samples 8 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_checkpoint import _load_model                      # noqa: E402
from tools.code_eval import _lrs_frac, _max_line_repeat, generate  # noqa: E402
from mythouro.inference import BestOfTrajectoryGenerator          # noqa: E402

# (name, prompt, answers, n_expected, kind)
#   answers    — every value that must appear for L4 (order-insensitive)
#   n_expected — how many distinct values a well-formed answer has (drives L2)
# Deliberately EASY, and phrased as continuations rather than chat turns, to
# match how the model is actually trained (raw-text distillation, no chat
# template). Several mirror the generation-probe seeds so the two instruments
# line up.
_TASKS = [
    ("add", "12 + 7 =", [19], 1, "int"),
    ("sub", "45 - 18 =", [27], 1, "int"),
    ("mul", "6 * 7 =", [42], 1, "int"),
    ("div", "84 / 4 =", [21], 1, "int"),
    ("pct", "10% of 250 is", [25], 1, "int"),
    ("avg", "The average of 4, 8, and 12 is", [8], 1, "int"),
    ("linear", "Solving 3x + 6 = 21 gives x =", [5], 1, "int"),
    ("quadratic", "To solve the quadratic equation x^2 - 5x + 6 = 0, we find x =",
     [2, 3], 2, "roots"),
    ("area", "The area of a rectangle 7 cm by 3 cm is", [21], 1, "int"),
    ("frac", "1/2 + 1/4 =", [0.75], 1, "frac"),
]

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def _to_float(tok: str) -> "float | None":
    try:
        if "/" in tok:
            a, b = tok.split("/")
            return float(a) / float(b) if float(b) else None
        return float(tok)
    except (ValueError, ZeroDivisionError):
        return None


def extract_numbers(text: str) -> "list[float]":
    """Every number in the completion, in order, fractions resolved."""
    out = []
    for m in _NUM_RE.finditer(text.replace("\\n", "\n")):
        v = _to_float(m.group())
        if v is not None:
            out.append(v)
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


def score_sample(completion: str, answers: "list[float]", n_expected: int,
                 kind: str, prompt: str = "") -> dict:
    """Ladder rung 0-4 plus diagnostics, for one generation.

    TWO SUB-L4 SIGNALS, because the ladder alone is insensitive at the floor.
    At step 70,500 the model scores L2 on 97.5% of samples and L4 on 0%, so a
    comparison against that baseline (e.g. an --n-loops sweep, or the
    task-shaped mix change) could read flat-at-zero and leave us unable to tell
    "this did not help" from "there was nothing yet for it to help".

      `copied`  — is the emitted number lifted straight from the prompt?
                  This measures the MECHANISM found at 70,500: "12 + 7 =" -> 12,
                  "10% of 250 is" -> 100%, "x^2 - 5x + 6 = 0, x =" -> 0. It is
                  slot-filling by echo, not arithmetic. A falling copy rate means
                  computation is starting even while answers stay wrong.
      `rel_err` — relative error of the CLOSEST emitted number to the answer.
                  18 and 12 both score L2 for "12 + 7", but 18 is nearly right
                  and 12 is an echo. Graded, so it moves before the ladder does.
    """
    comp = completion.replace("\\n", "\n").replace("\\t", "\t")
    nums = extract_numbers(comp)
    prompt_nums = extract_numbers(prompt) if prompt else []
    copied = bool(nums) and any(_close(nums[0], p) for p in prompt_nums)
    rel = None
    if nums and answers:
        target = answers[0]
        denom = abs(target) if abs(target) > 1e-9 else 1.0
        rel = round(min(abs(v - target) for v in nums) / denom, 4)
    d = {
        "rung": 0,
        "n_numbers": len(nums),
        "copied_from_prompt": copied,
        "rel_err": rel,
        "max_line_repeat_completion": _max_line_repeat(comp),
        "lrs_frac": round(_lrs_frac(completion), 4),
        "completion": completion,
    }
    if not nums:
        return d                                            # L0
    d["rung"] = 1                                           # L1 emitted a number

    # L2: right FORM. The answer is normally the first thing emitted for a
    # continuation prompt ("12 + 7 =" -> "19"), so form is judged on the leading
    # values: enough of them, and of the right type.
    head = nums[:n_expected]
    if len(head) == n_expected:
        if kind == "int":
            form_ok = all(float(x).is_integer() for x in head)
        elif kind == "roots":
            form_ok = len(set(head)) == n_expected      # two DISTINCT roots
        else:
            form_ok = True
        if form_ok:
            d["rung"] = 2                                   # L2 right form

    matched = [a for a in answers if any(_close(a, v) for v in nums)]
    if len(matched) < len(answers):
        return d                                            # nothing better

    # Every required value appears somewhere. L4 only if they lead the output —
    # i.e. presented AS the answer rather than buried in surrounding arithmetic.
    d["rung"] = 3                                           # L3 buried
    lead = nums[:max(n_expected, len(answers))]
    if all(any(_close(a, v) for v in lead) for a in answers):
        d["rung"] = 4                                       # L4 stated as answer
    return d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--samples", type=int, default=8,
                   help="Generations per task. Task-level score is the BEST of "
                        "these; the PRIMARY statistic is the per-sample rate.")
    p.add_argument("--max-new", type=int, default=48,
                   help="Short on purpose: these answers are a few tokens, and a "
                        "long budget just invites the model to ramble past a "
                        "correct answer into contradicting itself.")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--decoder", choices=("act", "best_of_trajectory"),
                   default="act",
                   help="'act' = the normal path: the recurrent block's ACT\n"
                        "halting decides the exit. 'best_of_trajectory' scores\n"
                        "EVERY loop with the UncertaintyHead and emits the most\n"
                        "confident one per token — keeping the best step it saw\n"
                        "rather than running deeper and trying to undo a bad\n"
                        "one. The @70,500 sweep showed depth is inert under ACT;\n"
                        "this is a DIFFERENT SELECTOR over the same trajectory\n"
                        "and is untested against task correctness (the P0.5 audit\n"
                        "measured calibration only). Needs no training. Slow:\n"
                        "B=1, full recompute per token, no KV cache.")
    p.add_argument("--force-full-depth", action="store_true",
                   help="Suppress ACT early exit so the block actually runs the\n"
                        "full --n-loops. WITHOUT this, --n-loops sets a BUDGET,\n"
                        "not a depth: positions halt at act_threshold=0.99 and a\n"
                        "larger allowance may simply go unused, so a flat sweep\n"
                        "cannot distinguish 'depth does not help' from 'the extra\n"
                        "depth was never taken'. Reported halt depth tells you\n"
                        "which happened.")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the sampler so re-measuring a checkpoint is "
                        "REPRODUCIBLE. Omitting it means a rerun draws fresh "
                        "samples — that cost a training run once already.")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    from mythouro.tokenizer import MythOuroTokenizer
    enc = MythOuroTokenizer(args.tokenizer)
    tok = enc.tokenizer
    model, _cfg, step = _load_model(args.checkpoint, args.device)
    model.eval()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    bot = (BestOfTrajectoryGenerator(
               model, n_loops=args.n_loops, min_loops=2,
               force_full_depth=args.force_full_depth)
           if args.decoder == 'best_of_trajectory' else None)
    block = getattr(model, 'recurrent', None)
    if args.force_full_depth and block is not None:
        block.force_full_depth = True
    print(f"checkpoint step {step} | {args.samples} samples/task | "
          f"T={args.temperature} | seed={args.seed} | "
          f"n_loops={args.n_loops} | force_full_depth={args.force_full_depth} | "
          f"decoder={args.decoder}\n")

    names = {0: "L0 nothing", 1: "L1 a number", 2: "L2 right form",
             3: "L3 buried", 4: "L4 CORRECT"}
    rows, hist, all_samples = [], {k: 0 for k in names}, []
    for name, prompt, answers, n_exp, kind in _TASKS:
        best, best_txt, samples = 0, "", []
        for _ in range(args.samples):
            chosen = None
            if bot is not None:
                ids = torch.tensor([tok.encode(prompt)], device=args.device)
                r = bot.generate(ids, max_new_tokens=args.max_new,
                                 temperature=args.temperature, top_k=50)
                # Returns "sequences" = prompt + completion (1, T), NOT just the
                # new tokens — slice the prompt off or every completion silently
                # begins with the question restated, which would read as the
                # model echoing and corrupt copied_from_prompt.
                seq = r["sequences"][0].tolist()
                comp = tok.decode(seq[ids.shape[1]:])
                chosen = list(r.get("chosen_loops") or [])
            else:
                comp = generate(model, tok, prompt, args.device, args.max_new,
                                args.n_loops, args.temperature,
                                args.repetition_penalty)
            s = score_sample(comp, [float(a) for a in answers], n_exp, kind,
                             prompt=prompt)
            hs = getattr(block, 'last_halt_step', None) if block is not None else None
            s['halt_depth'] = round(float(hs.float().mean()), 2) if hs is not None else None
            if chosen:
                s['chosen_loops_mean'] = round(sum(chosen) / len(chosen), 2)
                s['chosen_loops'] = chosen
            samples.append(s)
            if s["rung"] > best:
                best, best_txt = s["rung"], comp
        hist[best] += 1
        all_samples.extend(samples)
        rows.append({"task": name, "prompt": prompt, "answers": answers,
                     "rung": best, "label": names[best], "samples": samples})
        print(f"  {name:11} {prompt:48} {names[best]}")
        if best >= 2:
            print(f"       {best_txt[:90].strip()!r}")

    n = len(_TASKS)
    tot = len(all_samples)
    k4 = sum(1 for s in all_samples if s["rung"] == 4)
    k3 = sum(1 for s in all_samples if s["rung"] >= 3)
    ci = 1.96 * ((k4 / tot) * (1 - k4 / tot) / tot) ** 0.5 if tot else 0.0
    reach = lambda r: sum(v for kk, v in hist.items() if kk >= r)

    print(f"\n  ** per-sample L4 (CORRECT): {k4}/{tot} = {100 * k4 / tot:.1f}% "
          f"±{100 * ci:.1f} (95% CI) **   <- PRIMARY, quote this")
    print(f"  per-sample L3+ (answer present anywhere): {k3}/{tot} = "
          f"{100 * k3 / tot:.1f}%")
    print(f"  task-level (best-of-{args.samples}, n={n} — coarse): "
          f"L2+ {reach(2)}/{n}  L3+ {reach(3)}/{n}  L4 {reach(4)}/{n}")
    lrs = sorted(s["lrs_frac"] for s in all_samples)
    print(f"  repetition: median lrs_frac {lrs[tot // 2]:.3f}, "
          f"{sum(1 for s in all_samples if s['lrs_frac'] >= 0.35)}/{tot} degenerate")
    # Sub-L4 signals — these move before the ladder does (see score_sample).
    copied = sum(1 for s in all_samples if s["copied_from_prompt"])
    errs = sorted(s["rel_err"] for s in all_samples if s["rel_err"] is not None)
    med_err = errs[len(errs) // 2] if errs else None
    near = sum(1 for e in errs if e <= 0.10)
    hd = [s["halt_depth"] for s in all_samples if s.get("halt_depth") is not None]
    if hd:
        print(f"  ACTUAL halt depth: mean {sum(hd)/len(hd):.2f} of "
              f"{args.n_loops} budgeted  <- if this is flat across a --n-loops "
              f"sweep, the extra depth was never taken")
    print(f"  COPIED from prompt: {copied}/{tot} = {100 * copied / tot:.1f}%  "
          f"(echo, not arithmetic — falling means computation is starting)")
    print(f"  rel. error of closest number: median "
          f"{('%.3f' % med_err) if med_err is not None else 'n/a'}, "
          f"within 10% on {near}/{len(errs) or 1}")
    if args.seed is None:
        print("  ⚠ UNSEEDED — a rerun draws fresh samples. Pass --seed.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"step": step, "temperature": args.temperature, "seed": args.seed,
             "samples": args.samples,
             "repetition_penalty": args.repetition_penalty, "tasks": rows,
             "decoder": args.decoder,
             "n_loops": args.n_loops,
             "force_full_depth": args.force_full_depth,
             "mean_halt_depth": (sum(hd)/len(hd)) if hd else None,
             "copied_from_prompt": copied,
             "median_rel_err": med_err,
             "within_10pct": near,
             "per_sample_l4": round(k4 / tot, 4) if tot else 0.0,
             "per_sample_l3plus": round(k3 / tot, 4) if tot else 0.0,
             "reached": {f"L{r}+": reach(r) for r in (1, 2, 3, 4)}}, indent=2))
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
