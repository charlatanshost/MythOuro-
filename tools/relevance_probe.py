"""
Relevance / grounding scorer — the axis the degeneracy metrics cannot see.

`top_share` / `distinct1` never look at the prompt: swap the seed entirely and
they don't move. They measure a *fluency floor* ("is the output degenerate?"),
not whether the model responded to what was asked. Coherency and relevancy —
the two properties that make output a *conversation* rather than generated text —
have been eyeballed, which is why single-sample reads kept flipping.

This scores EXISTING probe reports (no GPU, no re-generation), so a whole
checkpoint history can be retro-scored into a trend.

⚠ HONEST SCORECARD (validated against real reports 2026-07-29). THREE metrics
were built; only ONE survived contact with data. Kept as-is, with the failures
documented, because the failures are the useful part.

* ✅ self-repetition (USE THIS) — repeated 4-grams inside the completion.
  Strong discriminator: a phrase-loop ("- The following is a list of the
  following: - The following is ...") scores 0.619 where healthy text scores
  0.000. Complements top_share, which rates that same loop as *healthy* because
  its tokens vary. On the real α=0.0 series it shows a genuine trend:
  52k 0.000 → 58k 0.017 → 60k 0.068 (phrase looping increasing).

* ❌ novel-content rate (UNINFORMATIVE — do not steer on it). Intended to catch
  prompt-echo, motivated by the weather seed where α=0.0 recycled prompt words
  ("the weather ... the winter ... the wind") while α=0.5 introduced new content
  ("the zoo ... a picnic ... the park"). It orders them correctly but by only
  0.909 vs 1.000 — and across the real series it is essentially constant
  (0.992 / 0.978 / 0.966). Why it fails: the prompts are ~10 words and the
  completions ~80, so echoing a handful of words barely moves a rate computed
  over all content words. The thing actually wrong with that weather output —
  that it discusses the weather instead of proposing an activity — is SEMANTIC,
  and this confirms rules can't reach it.

* ❌ code validity (BROKEN twice over). (a) Implementation: the trailing-line
  retry reduces a single-line input to "", and `ast.parse("")` succeeds — so it
  could return True for anything. (b) Conceptual, and fatal: a degenerate blob
  IS valid Python — `def f(n): return n - 8553442926685.994633` parses fine.
  Syntactic validity cannot distinguish the number-blob failure mode from real
  code, which is exactly the distinction we needed. Fixing (a) would not rescue
  (b). A real code metric has to EXECUTE the function against test cases.

CONCLUSION: relevancy/coherency are not rule-checkable at useful resolution.
Self-repetition is a genuine addition to the degeneracy floor, and it is the
only claim this module makes. Real task scoring needs either executable tests
(code — the tractable case) or a judge model (Ouro is on the box; costs GPU).

Usage
-----
    python -m tools.relevance_probe reports/onpolicy_rollout_probe_*_n5.txt
    python -m tools.relevance_probe reports/collapse_metrics_58000_xpu_aligned.txt
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys
from collections import Counter

# Function words carry no content, so echoing them is not evidence of anything.
_STOP = set("""
a an the this that these those and or but so if then than as of in on at to for
from with without by is are was were be been being am do does did doing have has
had having will would can could shall should may might must not no nor it its
they them their he she his her we us our you your i me my there here what which
who whom when where why how all any both each few more most other some such only
own same too very s t just also into about over under again further once because
while during before after above below up down out off on
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'_-]*")


def _content_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)
            if w.lower() not in _STOP and len(w) > 1]


def novel_content_rate(prompt: str, completion: str) -> "float | None":
    """Share of the completion's content words that are NOT in the prompt.

    Low = the model is recycling the prompt (the failure the weather seed showed);
    high = it is introducing new material. Returns None if the completion has no
    content words to score.
    """
    pw = set(_content_words(prompt))
    cw = _content_words(completion)
    if not cw:
        return None
    return sum(1 for w in cw if w not in pw) / len(cw)


def self_repetition(completion: str, n: int = 4) -> "float | None":
    """Share of 4-grams in the completion that are repeats.

    Catches phrase-level looping that token-level stats miss — e.g.
    "- The following is a list of the following: - The following is a list ..."
    scores well on distinct1 because the tokens vary, but is plainly degenerate.
    """
    toks = _WORD.findall(completion.lower())
    if len(toks) < n * 2:
        return None
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    return 1.0 - (len(counts) / len(grams))


def code_parses(prompt: str, completion: str) -> "bool | None":
    """Does prompt+completion parse as Python? None if this isn't a code prompt.

    Truncated generations legitimately end mid-statement, so a trailing-garbage
    retry is attempted by trimming to the last complete line before giving up.
    """
    if not re.search(r"\bdef |\bclass |\bimport |sorted\(", prompt):
        return None
    src = prompt + completion
    src = src.replace("\\n", "\n").replace("\\t", "\t")
    for candidate in (src, "\n".join(src.split("\n")[:-1])):
        if not candidate.strip():
            continue          # ast.parse("") SUCCEEDS — never count that as valid
        try:
            ast.parse(candidate)
            return True
        except (SyntaxError, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

_SEED = re.compile(r"SEED: '(.*?)'")
_ALPHA = re.compile(r"α=([0-9.]+)")
_EG = re.compile(r"e\.g\.: (.+)$")
_CM_PROMPT = re.compile(r"^prompt: '(.*)'\s*$")
_CM_GEN = re.compile(r"generated \([^)]*\): '(.*)'\s*$")


def parse_report(path: str) -> list[dict]:
    """Extract (prompt, completion, mode) triples from either report format."""
    out, seed, alpha = [], None, None
    for line in open(path, errors="replace"):
        m = _SEED.search(line)
        if m:
            seed, alpha = m.group(1), None
            continue
        m = _ALPHA.search(line)
        if m:
            alpha = m.group(1)
            continue
        m = _EG.search(line)
        if m and seed is not None:
            body = m.group(1).strip()
            if len(body) > 1 and body[0] in "\"'":
                body = body[1:-1] if body[-1] == body[0] else body[1:]
            out.append({"prompt": seed, "completion": body,
                        "mode": f"α={alpha}" if alpha else "?"})
            continue
        m = _CM_PROMPT.match(line)
        if m:
            seed = m.group(1)
            continue
        m = _CM_GEN.search(line)
        if m and seed is not None:
            out.append({"prompt": seed, "completion": m.group(1),
                        "mode": "greedy" if "greedy" in line else "sampled"})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("reports", nargs="+",
                   help="Probe report files (globs OK). Scored per file so a "
                        "checkpoint series becomes a trend.")
    p.add_argument("--mode", default=None,
                   help="Only score this mode, e.g. 'α=0.0' or 'greedy'. "
                        "α=0.0 is the load-bearing unaided case.")
    args = p.parse_args()

    files = sorted(f for pat in args.reports for f in glob.glob(pat))
    if not files:
        raise SystemExit("no report files matched")

    print(f"{'report':46} {'n':>3} {'selfrep':>8} {'novel!':>7} {'code!':>7}")
    print("-" * 76)
    for f in files:
        rows = parse_report(f)
        if args.mode:
            rows = [r for r in rows if r["mode"] == args.mode]
        if not rows:
            print(f"{os.path.basename(f)[:46]:46} {'—':>3}  (no rows matched)")
            continue
        nov = [novel_content_rate(r["prompt"], r["completion"]) for r in rows]
        rep = [self_repetition(r["completion"]) for r in rows]
        cod = [code_parses(r["prompt"], r["completion"]) for r in rows]
        nov = [x for x in nov if x is not None]
        rep = [x for x in rep if x is not None]
        cod = [x for x in cod if x is not None]
        m = lambda v: (sum(v) / len(v)) if v else float("nan")
        code_s = f"{100*m(cod):6.0f}%" if cod else "     —"
        print(f"{os.path.basename(f)[:46]:46} {len(rows):>3} "
              f"{m(rep):>8.3f} {m(nov):>7.3f} {code_s:>7}")

    print("\n  selfrep = repeated 4-grams in the completion — THE ONLY VALIDATED "
          "COLUMN.\n            Lower is better. Catches phrase looping that "
          "top_share rates as healthy.")
    print("  novel!  = prompt-echo proxy — UNINFORMATIVE, near-constant in "
          "practice. Do not steer on it.")
    print("  code!   = syntactic validity — MISLEADING: a degenerate number-blob "
          "is valid Python.\n            Real code scoring needs EXECUTION "
          "against tests.")
    print("  See the module docstring for why the last two failed; relevancy is "
          "not rule-checkable\n  at useful resolution.")


if __name__ == "__main__":
    main()
