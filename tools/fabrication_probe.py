"""
Does the teacher invent DEFINITIONS the source never gave?

WHY THIS EXISTS. `tools/compare_corpora.py` measures GROUNDING — how much of an
answer's vocabulary comes from the passage. It reported Ouro-1.4B-Thinking as
comparable to the 2.6B (0.473 vs 0.462). But reading one 1.4B code sample:

    passage: a GPL header by Rafael Senties Martinelli, importing AKBL.Bindings
    answer : "interacts with an AKBL (Advanced Knowledge Binding Language)
              application ... enables the script to interact with AKBL's
              knowledge base"

AKBL is **Alienware KeyBoard Lights**. The model invented an expansion and then
built two further sentences on it. Grounding cannot catch this: the answer stays
topically anchored while fabricating one specific claim. It is the same shape as
"the PAWL study" and 30-mg ibuprofen (real dosing 200-400mg) from the continuation
harvest — and on a project whose stated mission is medical, a corpus that teaches
confident fabrication is worse than a smaller honest one.

WHAT IS DETECTED. Definitional constructions in the ANSWER whose content is absent
from the PASSAGE:

    ACRONYM (Expanded Form Here)        <- the AKBL case
    Term (some gloss)
    X stands for Y / X refers to Y / X is short for Y

A gloss is UNSUPPORTED when fewer than `--support` of its content words appear in
the passage. That threshold matters: a legitimate restatement reuses the source's
words, an invention does not.

DELIBERATELY NARROW. This finds one failure mode precisely rather than all of them
vaguely. A clean report does NOT mean the corpus is free of fabrication — it means
this particular, easily-detected kind is absent. Reported as a RATE per 100 rows so
corpora of different sizes compare.

Usage
-----
    python -m tools.fabrication_probe data_teacher_chat_clean /tmp/ouro14b_test
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_corpora import _split, _words          # noqa: E402

# ACRONYM (expansion)  — the highest-precision signal: an all-caps token followed
# by a parenthetical of ordinary words is almost always a definition.
_ACRONYM_GLOSS = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\s*\(([^)]{4,80})\)")
# Term (gloss) — capitalised word followed by a parenthetical.
_TERM_GLOSS = re.compile(r"\b([A-Z][a-zA-Z]{2,})\s*\(([^)]{4,80})\)")
# X stands for / refers to / is short for Y
_PHRASE_DEF = re.compile(
    r"\b([A-Z][A-Za-z0-9]{1,15})\s+(?:stands for|refers to|is short for)\s+"
    r"([^.;\n]{4,80})", re.I)


_QUALIFIER = re.compile(r"^\s*(e\.?g\.?|i\.?e\.?|for|see|exact|approx|such as|"
                        r"including|and|or|not|no|but|excluding|per|via|from|to)\b",
                        re.I)


def _is_expansion(term: str, gloss: str) -> bool:
    """Do the gloss's word initials spell the acronym? A real expansion does.

    THE PRECISION FIX. Without this the probe flags every parenthetical:
    "Complications (e.g., bleeding, infection)" is an example list, "USB (for slow
    control)" a qualifier, "Friday (exact day unspecified)" a hedge. None are
    definitional claims. Requiring the initials to match keeps only constructions
    that ASSERT what an acronym stands for — the AKBL case — and drops the rest.
    """
    letters = [w[0].lower() for w in re.findall(r"[A-Za-z]+", gloss)]
    t = term.lower()
    if len(letters) < 2:
        return False
    # allow small-word skips: "Digital Display Systems for X" still expands DDS
    i = 0
    for ch in t:
        while i < len(letters) and letters[i] != ch:
            i += 1
        if i >= len(letters):
            return False
        i += 1
    return True


def _unsupported(answer: str, passage: str, support: float) -> list:
    """Definitions in `answer` whose content words are absent from `passage`."""
    pw = set(_words(passage))
    out, seen_keys = [], set()
    for rx, kind in ((_ACRONYM_GLOSS, "acronym"),
                     (_TERM_GLOSS, "term"),
                     (_PHRASE_DEF, "phrase")):
        for term, gloss in rx.findall(answer):
            gw = _words(gloss)
            if len(gw) < 2:                     # "(n)" / "(see below)" — not a claim
                continue
            if _QUALIFIER.match(gloss):         # example list / hedge, not a definition
                continue
            if kind in ("acronym", "term") and not _is_expansion(term, gloss):
                continue                        # parenthetical, but not a DEFINITION
            key = (term.lower(), gloss.strip().lower())
            if key in seen_keys:      # acronym+term patterns both match "X (...)"
                continue
            hit = sum(1 for w in gw if w in pw) / len(gw)
            if hit < support:
                seen_keys.add(key)
                # A term the passage never mentions at all is a weaker signal —
                # the model may be introducing context rather than misdefining.
                # Flag it, but mark it so the reader can discount it.
                seen = term.lower() in pw
                out.append({"kind": kind, "term": term, "gloss": gloss.strip(),
                            "support": round(hit, 2), "term_in_passage": seen})
    return out


def profile(d: str, support: float) -> dict:
    rows = [json.loads(l) for f in sorted(glob.glob(f"{d}/shard_*.jsonl"))
            for l in open(f)]
    hits, rows_with = [], 0
    for r in rows:
        p, a = _split(r["text"])
        found = _unsupported(a, p, support)
        if found:
            rows_with += 1
            for f in found:
                f["source"] = r["source"]
            hits.extend(found)
    return {"rows": len(rows), "rows_with": rows_with, "hits": hits,
            "per100": 100 * rows_with / len(rows) if rows else 0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("dirs", nargs="+")
    p.add_argument("--support", type=float, default=0.34,
                   help="A gloss is UNSUPPORTED when fewer than this fraction of "
                        "its content words appear in the passage. 0.34 means "
                        "'at least a third should be traceable to the source'.")
    p.add_argument("--show", type=int, default=6, help="examples per corpus")
    args = p.parse_args()

    profs = [(d, profile(d, args.support)) for d in args.dirs]
    print(f"\n  UNSUPPORTED DEFINITIONS (support threshold {args.support})\n")
    print(f"  {'corpus':<28}{'rows':>8}{'rows w/ fabrication':>22}{'per 100 rows':>14}")
    for d, pr in profs:
        print(f"  {Path(d).name:<28}{pr['rows']:>8,}{pr['rows_with']:>22}"
              f"{pr['per100']:>13.1f}")

    for d, pr in profs:
        print(f"\n{'='*76}\n  {Path(d).name} — examples\n{'='*76}")
        if not pr["hits"]:
            print("  none detected")
            continue
        # term_in_passage first: those are the strong cases (the passage USES the
        # term, so the model is defining something it should have read, not
        # introducing outside context).
        for h in sorted(pr["hits"], key=lambda x: (not x["term_in_passage"],
                                                   x["support"]))[:args.show]:
            mark = "TERM IN PASSAGE" if h["term_in_passage"] else "term not in passage"
            print(f"  [{h['source']:<14}] {h['term']} ({h['gloss'][:60]})")
            print(f"      support {h['support']:.2f}  {mark}")

    print("\n  Read the strong cases first (TERM IN PASSAGE): the source uses the")
    print("  term, so an unsupported gloss is the model defining something it")
    print("  should have read. 'term not in passage' is weaker — it may be")
    print("  legitimate outside context rather than invention.")
    print("\n  ⚠ A clean report does NOT mean the corpus is fabrication-free. This")
    print("  detects ONE precise failure mode, not all of them.")


if __name__ == "__main__":
    main()
