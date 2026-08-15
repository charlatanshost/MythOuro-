"""
Compare two teacher corpora — is a faster teacher writing WORSE answers?

WHY THIS EXISTS. On 2026-08-15 Ouro-1.4B-Thinking harvested at 66 tok/s against
Ouro-2.6B-Thinking's 45 (1.47x, from half the layers). A faster teacher is only a
win if the corpus is still good, and "still good" has been argued from two
cherry-picked samples before. The chat-mix leg already showed corpus CHARACTER
matters more than corpus SIZE, so this measures character.

THE METRIC THAT MATTERS IS GROUNDING. These are instruction rows: a real corpus
snippet in the prompt, the teacher's answer after it. A good answer talks about
the passage. A drifting one invents its own topic — which is exactly what the
continuation harvest did when it produced "the PAWL study" and 30-mg ibuprofen
against a real 200-400mg. `grounding` is the fraction of the answer's content
words that also appear in the passage. It is a floor, not a ceiling: high
grounding does not prove correctness, but LOW grounding proves drift.

Everything else here is a guard against reading grounding wrong: a model that
parrots the passage back verbatim scores perfect grounding and is useless, so
`copy_rate` (longest shared run) is reported beside it.

Usage
-----
    python -m tools.compare_corpora data_teacher_chat_clean /tmp/ouro14b_test
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_STOP = set("""a an the and or but if then than that this these those is are was were be been
being to of in on at by for with from as it its it's we you they he she i not no do does did
have has had will would can could should may might must there their them our your my me him her
what which who whom whose when where why how all any both each few more most other some such only
own same so too very s t don now""".split())


def _words(t: str) -> list:
    return [w for w in re.findall(r"[a-z0-9']+", t.lower())
            if w not in _STOP and len(w) > 2]


def _longest_shared_run(a: list, b: list) -> int:
    """Longest run of consecutive words appearing in both — verbatim copying."""
    bs = set(zip(b, b[1:])) if len(b) > 1 else set()
    best = run = 0
    for i in range(len(a) - 1):
        if (a[i], a[i + 1]) in bs:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best + 1 if best else 0


def _split(text: str) -> "tuple[str, str]":
    """(passage, answer) from one ChatML row."""
    user = text.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
    passage = user.split("\n\n", 1)[1] if "\n\n" in user else user
    ans = text.split("</think>\n\n", 1)[-1] if "</think>" in text else \
        text.split("<|im_start|>assistant\n", 1)[-1]
    return passage, ans.rsplit("<|im_end|>", 1)[0]


def profile(d: str) -> dict:
    rows = [json.loads(l) for f in sorted(glob.glob(f"{d}/shard_*.jsonl"))
            for l in open(f)]
    if not rows:
        return {}
    ground, copyr, alen = [], [], []
    for r in rows:
        p, a = _split(r["text"])
        pw, aw = _words(p), _words(a)
        alen.append(len(aw))
        if aw:
            ground.append(sum(1 for w in aw if w in set(pw)) / len(aw))
            copyr.append(_longest_shared_run(aw, pw))
    return {
        "rows": len(rows),
        "sources": dict(Counter(r["source"] for r in rows)),
        "answer_words_median": st.median(alen) if alen else 0,
        "grounding_median": st.median(ground) if ground else 0,
        "grounding_p10": sorted(ground)[len(ground) // 10] if ground else 0,
        "copy_run_median": st.median(copyr) if copyr else 0,
        "copy_run_p90": sorted(copyr)[int(.9 * len(copyr))] if copyr else 0,
        "reopened_think": sum(1 for r in rows if r["text"].count("<think>") > 1),
        "malformed": sum(1 for r in rows if r["text"].count("<|im_start|>") != 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("dirs", nargs=2, help="two corpus directories to compare")
    p.add_argument("--samples", type=int, default=2,
                   help="rows to print in full from each, per the standing rule "
                        "that aggregates have pointed the wrong way five times")
    args = p.parse_args()

    profs = [(d, profile(d)) for d in args.dirs]
    for d, pr in profs:
        if not pr:
            raise SystemExit(f"no shards found in {d}")

    keys = [("rows", "rows", "{:,}"),
            ("answer_words_median", "answer words (median)", "{:.0f}"),
            ("grounding_median", "GROUNDING median", "{:.3f}"),
            ("grounding_p10", "  grounding p10 (worst tenth)", "{:.3f}"),
            ("copy_run_median", "copy run median", "{:.0f}"),
            ("copy_run_p90", "  copy run p90", "{:.0f}"),
            ("reopened_think", "reopened <think>", "{:.0f}"),
            ("malformed", "malformed", "{:.0f}")]
    w = max(len(lbl) for _, lbl, _ in keys) + 2
    print(f"\n  {'metric':<{w}}" + "".join(f"{Path(d).name:>26}" for d, _ in profs))
    for k, lbl, fmt in keys:
        print(f"  {lbl:<{w}}" + "".join(f"{fmt.format(pr[k]):>26}" for _, pr in profs))

    a, b = profs[0][1], profs[1][1]
    print(f"\n  GROUNDING is the number to read: the fraction of the answer's")
    print(f"  content words that appear in the passage. Lower = the teacher is")
    print(f"  drifting off the source, which is how the continuation harvest")
    print(f"  produced fabrications. COPY RUN guards the other side — a corpus")
    print(f"  that parrots the passage scores perfect grounding and teaches")
    print(f"  nothing.")
    d = b["grounding_median"] - a["grounding_median"]
    if abs(d) < 0.03:
        print(f"\n  VERDICT: grounding is comparable ({d:+.3f}). Judge on speed "
              f"and on the text below.")
    elif d < 0:
        print(f"\n  VERDICT: {Path(profs[1][0]).name} is LESS grounded "
              f"({d:+.3f}) — it drifts further from the passage. A faster "
              f"teacher writing less grounded answers is not a win.")
    else:
        print(f"\n  VERDICT: {Path(profs[1][0]).name} is MORE grounded "
              f"({d:+.3f}) — but check copy run: if that rose too, it is "
              f"parroting rather than summarising.")

    for d_, _ in profs:
        rows = [json.loads(l) for f in sorted(glob.glob(f"{d_}/shard_*.jsonl"))
                for l in open(f)]
        print(f"\n{'='*74}\n  {Path(d_).name}\n{'='*74}")
        for r in rows[:args.samples]:
            pas, ans = _split(r["text"])
            print(f"\n[{r['source']}] PASSAGE: {pas[:200].strip()}")
            print(f"    ANSWER: {ans[:420].strip()}")


if __name__ == "__main__":
    main()
