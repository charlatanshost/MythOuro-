"""
Math-slice corpus quality — and a warning about how little it can measure.

  python3 tools/corpus_math_check.py --arith -n 4000        # mechanical LOWER bound
  python3 tools/corpus_math_check.py --review 10 --seed 7   # dump rows for HAND adjudication

Companion to tools/corpus_correctness.py, which measures `data_teacher_code`.
The two slices need different instruments and give very different answers:

  CODE   rows self-adjudicate. The PROMPT carries `**Sample Input/Output**`, so
         the row contains an independent check of its own answer. 96.4% correct.
  MATH   rows do NOT self-adjudicate. The only answer present is the teacher's
         own `\\boxed{}` claim. There is nothing to check it against.

⚠ ARITHMETIC CHECKING IS NEARLY BLIND HERE — this is the main finding.
`--arith` verifies explicit numeric claims (`47 * 93 = 4361` -> false, it is 4371)
and flags ~4% of assertions, of which hand-inspection found ~40% real, so ~2%.
That number is almost meaningless as a quality measure, because the errors that
matter are REASONING errors with perfect arithmetic. Demonstration, from
shard_0000 row 1: "sum of all three-digit integers whose cubes end in 125" is
answered 125+375+625+875 = 2000. Every addition is correct. But 375^3 ends in
375, 625^3 in 625, 875^3 in 875 — three of the four do not satisfy the condition,
18 valid values are missing, and the answer is 11990. No arithmetic checker can
see this. Do not quote --arith as the corpus error rate.

⚠ EXTRACTION ARTIFACTS in --arith, found by inspection and only partly fixed:
  - LaTeX renders `\\div` as `:`, so `(105+20) : 26` splits like an equation
  - stripping variables turns `14x = -26` into `14 = -26`, and `x^2 - 57600 = 0`
    into `2 - 57600 = 0` (this one alone produced a phantom 18.69% before the
    boundary checks were added)
  - factorials: `15!(1 + 4896) = 15! * 4897` loses the `15!`
  - rounding: `43 / 3 = 14.33` is not an error (handled by `agree()`)
Every flagged expression is printed WITH CONTEXT for this reason. Classify by
hand before believing any of it.

⇒ THE ONLY TRUSTWORTHY METHOD IS `--review`: read rows and adjudicate them.
Measured 2026-09-26, n=20, seeds 7 and 99: **13 clean / 7 defective ~ 65%**,
against 96.4% for the code slice. The failures concentrate in the
reasoning-dense problems (octagon area, cubic inequality, distance-to-parabola,
a perfect-square count, real-vs-all roots under Vieta) plus rows whose PROBLEM is
defective — one unanswerable question the model answers anyway after stating it
cannot be answered. Simple arithmetic word problems were all correct.
"""
from __future__ import annotations
import argparse, ast, glob as _glob, json, operator, random, re

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def _ev(n):
    if isinstance(n, ast.Expression):
        return _ev(n.body)
    if isinstance(n, ast.Constant):
        if isinstance(n.value, (int, float)):
            return n.value
        raise ValueError
    if isinstance(n, ast.BinOp):
        f = OPS.get(type(n.op))
        if f is None:
            raise ValueError
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and (abs(a) > 1e6 or abs(b) > 64):
            raise ValueError                      # keep 10^99999 from hanging us
        return f(a, b)
    if isinstance(n, ast.UnaryOp):
        f = OPS.get(type(n.op))
        if f is None:
            raise ValueError
        return f(_ev(n.operand))
    raise ValueError


def _calc(s: str):
    s = s.strip()
    if not s or not re.search(r"\d", s):
        raise ValueError
    return _ev(ast.parse(s.replace("^", "**"), mode="eval"))


def _norm(t: str) -> str:
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    for junk in ("\\left", "\\right", "\\,", "\\!", "\\;"):
        t = t.replace(junk, "")
    for _ in range(3):
        t = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", t)
    for junk in ("$", "\\[", "\\]", "\\(", "\\)"):
        t = t.replace(junk, "")
    return re.sub(r"(?<=\d),(?=\d\d\d(?!\d))", "", t)


TOK = r"[0-9.()+\-*/^ ]"
# A chain must be bounded by characters that cannot belong to a numeric
# expression. A letter, ^, _, {, }, \ or [ next to it means a variable was
# stripped and the "equation" is an artifact, not a claim.
BOUND_BAD = re.compile(r"[A-Za-z_^{}\\\]\[|:]")


def _agree(a: float, b: float) -> bool:
    if abs(a - b) <= 1e-6 * max(1, abs(a), abs(b)):
        return True
    for v, w in ((a, b), (b, a)):                 # is w a rounding of v?
        s = repr(w)
        d = len(s.split(".")[1]) if "." in s else 0
        if d <= 6 and abs(round(v, d) - w) <= 1e-9:
            return True
    return False


def assertions(text: str):
    t = _norm(text)
    out = []
    for m in re.finditer(r"(?:%s)+(?:=(?:%s)+)+" % (TOK, TOK), t):
        s, e = m.span()
        if (t[max(0, s - 1):s] and BOUND_BAD.search(t[max(0, s - 1):s])) or \
           (t[e:e + 1] and BOUND_BAD.search(t[e:e + 1])):
            continue
        parts = m.group(0).split("=")
        if len(parts) < 2 or not any(re.search(r"[*/+\-^]", p) for p in parts):
            continue
        vals = []
        try:
            for p in parts:
                vals.append(_calc(p))
        except Exception:
            continue
        if len(vals) >= 2:
            out.append((m.group(0).strip(), vals, t[max(0, s - 90):e + 25]))
    return out


def load(pattern: str):
    rows = []
    for f in sorted(_glob.glob(pattern)):
        with open(f) as fh:
            rows += fh.readlines()
    if not rows:
        raise SystemExit(f"no rows match {pattern}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data_teacher_math/shard_*.jsonl")
    ap.add_argument("-n", "--samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arith", action="store_true", help="mechanical LOWER bound")
    ap.add_argument("--review", type=int, default=0, help="dump N rows to adjudicate by hand")
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()
    rows = load(args.glob)

    if args.review:
        for i, l in enumerate(random.Random(args.seed).sample(rows, args.review), 1):
            t = json.loads(l)["text"]
            q = t.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0].strip()
            a = (t.split("<|im_start|>assistant", 1)[1].replace("<think>", "")
                 .replace("</think>", "").replace("<|im_end|>", "").strip())
            box = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", a)
            print(f"===== [{i}] Q: {q}")
            print(f"      CLAIMED ANSWER: {box[-1] if box else '(no box)'}")
            print(f"      WORK: {a[:700]}")
            print()
        print("Adjudicate each by hand. --arith cannot substitute for this.")
        return

    if not args.arith:
        raise SystemExit("pass --arith or --review N (see the module docstring)")

    samp = rows if args.samples >= len(rows) else random.Random(args.seed).sample(rows, args.samples)
    nrow = bad_row = noassert = nass = badass = 0
    flagged = []
    for l in samp:
        body = json.loads(l)["text"].split("<|im_start|>assistant", 1)[-1]
        A = assertions(body)
        nrow += 1
        if not A:
            noassert += 1
            continue
        hit = False
        for expr, vals, ctx in A:
            nass += 1
            if any(not _agree(v, vals[0]) for v in vals[1:]):
                badass += 1
                hit = True
                if len(flagged) < args.show:
                    flagged.append((expr[:70], [round(v, 4) for v in vals], ctx.replace("\n", " ")))
        bad_row += hit
    chk = nrow - noassert
    print(f"{args.glob}: {len(rows):,} rows; sampled {nrow}")
    print(f"  with >=1 checkable assertion  {chk}  ({chk/nrow:.1%})")
    print(f"  assertions checked            {nass}")
    print(f"  assertions flagged FALSE      {badass}  ({badass/max(nass,1):.2%})")
    print(f"  ROWS flagged                  {bad_row}  ({bad_row/max(chk,1):.2%} of checkable)")
    print("\n  ⚠ ~40% of flags were artifacts on hand-inspection, and REASONING")
    print("    errors are invisible to this check. This is a LOWER bound only.\n")
    for e, v, ctx in flagged:
        print(f"  {e}   -> {v}")
        print(f"      ctx: ...{ctx}...")


if __name__ == "__main__":
    main()
