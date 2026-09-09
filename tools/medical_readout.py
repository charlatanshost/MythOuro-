"""
Medical-seed readout from prose-probe reports.

    python3 tools/medical_readout.py reports/prose_instruct_*.json

WHY SEPARATE. `run_prose_readout.sh` averages all six seeds, which buries the
medical ones — the mission axis — under prose and code. Asked on 2026-09-09
whether the instruction leg had harmed medical, the aggregates could not answer
and reading three samples by eye gave the WRONG answer: it looked like exam-style
Q&A scaffolding was leaking in from the instruction corpus, and counting showed
the instruct checkpoints were actually LOWER than the seed (13% and 7% vs 27%).

The three medical seeds are indices 1-3 of the probe's six:
    1  'The treatment for a bacterial infection usually involves'
    2  'Common symptoms of type 2 diabetes include'
    3  'Ibuprofen is a nonsteroidal anti-inflammatory drug used to treat'

Reported per checkpoint, over α=0.0 and α=0.25 (the low-teacher-help alphas the
owner reads, where the student is most on its own):

  stutter    a 4+ char run doubled inside one word ('immunoconductduct')
  looping    any non-trivial line repeated >=3x
  QA         exam scaffolding ('Question 3:' / 'Answer:') in medical prose
  diabetes   canonical type-2 symptoms named on seed 2 (thirst, urination,
             blurred vision, fatigue, weight loss)

⚠ n is 10 per checkpoint for the diabetes column and 30 for the rest. The
lineage range on diabetes is 0-20% and on QA 7-27%, so a single checkpoint moving
inside those bands is NOISE. Read >=3 checkpoints, and do not call a difference
without them. exit_pdf@7,200 read 0% on diabetes where base and exit_pdf@6,000
both read 20% — one bad draw, which made leg 1 look like a recovery when it was
regression to the mean.
"""
import json, os, re, sys
from collections import Counter

MED = {1: "bacterial", 2: "diabetes", 3: "ibuprofen"}
ALPHAS = ("0.0", "0.25")
QA = re.compile(r"\bQuestion\s*\d*\s*[:.]|\bAnswer\s*\d*\s*[:.]", re.I)
DX = re.compile(r"thirst|urinat|polyuria|polydipsia|blurred vision|fatigue|weight loss", re.I)


def _stutter(t):
    return bool(re.search(r"\b\w*?(\w{4,})\1\w*\b", t))


def _looping(t):
    L = [l.strip() for l in t.split("\n") if len(l.strip()) > 3]
    return bool(L) and max(Counter(L).values()) >= 3


def readout(path):
    d = json.load(open(path))
    S = list(d["seeds"].items())
    st = lo = qa = n = 0
    dx = dn = 0
    for i in MED:
        if i >= len(S):
            continue
        for a in ALPHAS:
            for t in S[i][1].get(a, {}).get("texts", []):
                n += 1
                st += _stutter(t); lo += _looping(t); qa += bool(QA.search(t))
                if i == 2:
                    dn += 1; dx += bool(DX.search(t))
    return d.get("step", "?"), st, lo, qa, n, dx, dn


def main(paths):
    rows = [readout(p) for p in paths if os.path.exists(p)]
    if not rows:
        sys.exit("no readable reports")
    print("\n" + "=" * 70)
    print("  MEDICAL SEEDS (bacterial / diabetes / ibuprofen), α=0.0 and 0.25")
    print("=" * 70)
    print(f"  {'step':>8} {'stutter':>9} {'looping':>9} {'QA-scaffold':>12} {'diabetes sx':>12}")
    for step, st, lo, qa, n, dx, dn in rows:
        print(f"  {step:>8} {f'{st}/{n}':>9} {f'{lo}/{n}':>9} "
              f"{f'{qa}/{n}':>12} {f'{dx}/{dn}':>12}")
    if len(rows) >= 2:
        N = sum(r[4] for r in rows); DN = sum(r[6] for r in rows)
        print(f"\n  MEAN over {len(rows)} checkpoints:")
        print(f"    stutter  {sum(r[1] for r in rows)}/{N}"
              f"   looping {sum(r[2] for r in rows)}/{N}"
              f"   QA {sum(r[3] for r in rows)}/{N}"
              f"   diabetes {sum(r[5] for r in rows)}/{DN}")
    print("\n  LINEAGE REFERENCE (2026-09-09):")
    print("    base 157,000    stutter 0/30  looping 1/30  QA  5/30  diabetes 2/10")
    print("    exit_pdf 6,000  stutter 0/30  looping 0/30  QA  2/30  diabetes 2/10")
    print("    exit_pdf 7,200  stutter 1/30  looping 1/30  QA  8/30  diabetes 0/10")
    print("    instruct 3,000  stutter 0/30  looping 0/30  QA  4/30  diabetes 2/10")
    print("\n  ⚠ Ranges are wide (diabetes 0-20%, QA 7-27%) and n is 10 and 30.")
    print("    A single checkpoint inside those bands is noise, not a result.")
    print("  ⚠ NONE of this measures whether the medicine is CORRECT. It counts")
    print("    degeneracy and whether canonical symptoms are named. The model")
    print("    still produces confident wrong statements (ibuprofen 'to reduce")
    print("    the risk of cardiovascular disease' — it raises it). READ THE TEXT.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
