"""Does open-web-math actually teach how to SOLVE, or just discuss mathematics?"""
import re, sys
from datasets import load_dataset

N = 400
ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
pats = {
    "worked solution marker": r"(?i)\b(solution|solve for|step 1|therefore|hence)\b",
    "explicit Answer:":       r"(?i)\banswer\s*[:=]",
    "equation with =":        r"[a-zA-Z0-9\)\]]\s*=\s*[^=]",
    "quadratic-ish":          r"x\s*\^?2|x²",
    "'we get/we have'":       r"(?i)\bwe (get|have|obtain|find)\b",
    "LaTeX math":             r"\$|\\\\begin\{|\\\\frac",
    "question mark":          r"\?",
}
hits = {k: 0 for k in pats}
both = 0            # has a question AND a worked-solution marker
lens = []
for i, row in enumerate(ds):
    if i >= N: break
    t = row.get("text") or ""
    lens.append(len(t))
    f = {k: bool(re.search(p, t)) for k, p in pats.items()}
    for k, v in f.items(): hits[k] += v
    if f["question mark"] and f["worked solution marker"]: both += 1

print(f"sampled {len(lens)} docs, median length {sorted(lens)[len(lens)//2]:,} chars\n")
for k, v in sorted(hits.items(), key=lambda kv: -kv[1]):
    print(f"  {k:24} {v:>4}/{len(lens)}  {100*v/len(lens):5.1f}%")
print(f"\n  question + worked-solution marker: {both}/{len(lens)} ({100*both/len(lens):.1f}%)")
print("\n--- first 2 docs, opening 500 chars each ---")
ds2 = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
for i, row in enumerate(ds2):
    if i >= 2: break
    print(f"\n[doc {i}]\n{(row.get('text') or '')[:500]}")
