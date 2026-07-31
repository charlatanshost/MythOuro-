"""
Prompt→output reading instrument. DELIBERATELY SCORELESS.

WHY THIS EXISTS, AND WHY IT COMPUTES NOTHING
--------------------------------------------
Every other instrument in this repo produces numbers, and for output QUALITY the
numbers have pointed the wrong way three times:

  * 66,000 -> 70,500 at α=0.0, `top_share` got WORSE (+0.013) while
    prompt-relevance clearly improved — diabetes went from a first-person stomach
    anecdote ("the problem with my stomach... like a 50 ppm and 75") to an actual
    symptom list naming two real T2D associations; bacterial infection went from
    circular drug->drug->drug to a purpose clause with a correct connective.
  * An UNSEEDED `code_eval` rerun of identical weights gave L3+ 4/10 then 7/10,
    and a healthy training run was stopped over the difference.
  * A partially-written probe file produced "collapse events 3 -> 0" when the
    complete file said 3 -> 2, and that the medical prompts had stabilised while
    CODE became the worst offender.

`top_share` and `distinct1` are DEGENERACY proxies: they answer "is the output
broken", never "is it answering the question". Coherence, relevance and fluency
against the actual prompt are what text is for, and they are read, not measured.

An earlier attempt at this problem — `tools/relevance_probe.py` — responded to
"the numbers miss it" by adding three more numbers, only one of which validated.
So this tool scores NOTHING on purpose. It lays the seed beside what each
checkpoint produced from it and gets out of the way.

⚠ BUT DO NOT READ THIS AS "METRICS ARE UNRELIABLE" — that is an overcorrection,
and it was made here first. The two kinds of instrument catch each other's
failures, demonstrated in a single afternoon (2026-07-31):

  * READING caught a metric blind spot: `self.get_value(self.get_value(...` and
    a 95-zero run both scored max_line_repeat=1, i.e. perfectly clean. Hence
    `_lrs_frac` in code_eval.
  * The METRIC then caught a reading error: from those two examples the
    conclusion drawn was "degeneracy moved but didn't decrease". Computing the
    distribution showed the opposite — file framing genuinely is cleaner
    (median lrs_frac 0.528 -> 0.312, 57/80 -> 34/80 degenerate). Two vivid
    cases had misled a human read; only the numbers found it.
  * The METRICS carried the trend that mattered: per-sample L3+ 21% -> 42% ->
    54% over three checkpoints is what validated --rollout-reuse 8. Reading five
    generations could not have established that.
  * READING carried what no aggregate reported: the "the risk of..." opener
    persisting across steps 46k/52k/58k/64k and then breaking.

And metrics are what make 21 probe reports COMPARABLE at all — nobody eyeballs
21 probes. The numbers index the history and say where to look; the text says
what changed there. Use both; distrust either one alone.

Usage
-----
    # α=0.0 is the real-use case: no teacher help.
    python -m tools.text_diff reports/onpolicy_rollout_probe_66000_reuse8_n5.txt \\
                              reports/onpolicy_rollout_probe_70500_reuse8_n5.txt \\
        -o reports/text_diff_66000_vs_70500.md

    # every α, to see where the teacher is still carrying the content
    python -m tools.text_diff a.txt b.txt --alpha all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_SEED_RE = re.compile(r"SEED:\s*(.*?)\s+\(\d+ tok\)")
_ALPHA_RE = re.compile(r"α=([\d.]+)")


def parse_report(path: str) -> "dict[str, dict[str, list[str]]]":
    """{seed: {alpha: [texts]}} from a probe report or its richer .json.

    ⚠ A .txt report contains ONE sample per (seed, α) — and it is sample #1,
    kept by position, not chosen for being representative. The probe generated
    `--samples` of them and threw the rest away, keeping only their numbers.
    Comparing checkpoints on 1-of-5 is comparing a small part, not the whole:
    with n=5 you can pick two runs of an unchanged model and "see" a difference.
    Pass the .json (probe `--json`) to read every sample. Historical .txt reports
    are all we have for older checkpoints — their other samples are gone.
    """
    if path.endswith(".json"):
        import json
        blob = json.loads(Path(path).read_text())
        return {seed: {a: list(v["texts"]) for a, v in alphas.items()}
                for seed, alphas in blob.get("seeds", {}).items()}
    out: dict[str, dict[str, list[str]]] = {}
    lines = Path(path).read_text(errors="ignore").split("\n")
    seed = alpha = None
    for i, line in enumerate(lines):
        m = _SEED_RE.search(line)
        if m:
            seed = m.group(1).strip().strip("'\"")
            out.setdefault(seed, {})
            alpha = None
            continue
        a = _ALPHA_RE.search(line)
        if a and "top_share" in line:
            alpha = a.group(1)
            continue
        if "e.g.:" in line and seed is not None and alpha is not None:
            txt = line.split("e.g.:", 1)[1].strip()
            if txt[:1] in "'\"" and txt[-1:] in "'\"":
                txt = txt[1:-1]
            out[seed][alpha] = [txt]          # .txt holds only sample #1
            alpha = None
    return out


def label_for(path: str) -> str:
    m = re.search(r"probe_(\d+)_([a-z0-9]+)", Path(path).name)
    return f"step {m.group(1)} ({m.group(2)})" if m else Path(path).stem


def render(paths: "list[str]", alphas: "list[str] | None") -> str:
    reports = [(label_for(p), parse_report(p)) for p in paths]
    seeds: list[str] = []
    for _, r in reports:
        for s in r:
            if s not in seeds:
                seeds.append(s)

    o: list[str] = []
    o.append("# Prompt → output, read side by side\n")
    o.append("**No scores here, on purpose.** `top_share`/`distinct1` measure "
             "degeneracy and have pointed the wrong way on relevance three times "
             "— see the module docstring. Read the pairs.\n")
    o.append("Compared: " + ", ".join(lbl for lbl, _ in reports) + "\n")

    for seed in seeds:
        present = sorted({a for _, r in reports for a in r.get(seed, {})},
                         key=float)
        show = present if not alphas else [a for a in present if a in alphas]
        if not show:
            continue
        o.append("\n---\n")
        o.append(f"\n## PROMPT\n\n> {seed}\n")
        for a in show:
            o.append(f"\n### α={a}" + ("  *(no teacher help — the real-use case)*"
                                       if float(a) == 0.0 else ""))
            for lbl, r in reports:
                txts = r.get(seed, {}).get(a)
                if not txts:
                    o.append(f"\n**{lbl}**\n\n*(not in this report)*\n")
                    continue
                note = ("  *(only sample #1 survives in the .txt — the rest were "
                        "discarded)*" if len(txts) == 1 else
                        f"  *(all {len(txts)} samples)*")
                o.append(f"\n**{lbl}**{note}\n")
                for i, txt in enumerate(txts, 1):
                    body = txt.replace("\\n", "\n").replace("\\t", "\t")
                    head = f"\nsample {i}/{len(txts)}\n" if len(txts) > 1 else "\n"
                    o.append(head + "```\n" + body.strip() + "\n```\n")
        o.append("\n**Reading notes:** \n")   # left blank to be filled in by hand
    return "".join(o)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("reports", nargs="+", help="probe report .txt files, oldest first")
    p.add_argument("--alpha", default="0.0",
                   help="'0.0' (default — the no-teacher-help case), a "
                        "comma-separated list, or 'all'.")
    p.add_argument("-o", "--out", default=None, help="write markdown here")
    args = p.parse_args()

    alphas = None if args.alpha == "all" else [a.strip() for a in args.alpha.split(",")]
    md = render(args.reports, alphas)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}  ({len(md.splitlines())} lines)")
        print("Blank 'Reading notes' left under each prompt — fill them in; the "
              "written judgement IS the record.")
    else:
        print(md)


if __name__ == "__main__":
    main()
