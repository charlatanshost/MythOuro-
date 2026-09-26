"""
Is the TEACHER CORPUS itself correct? — CPU-only, runs while the card pours.

  python3 tools/corpus_correctness.py --glob 'data_teacher_code/shard_*.jsonl' -n 3000

WHY THIS EXISTS. 2026-09-26: the owner asked whether distillation should need
fewer tokens because the tokens are higher quality. The honest way to answer is
to measure the quality rather than assume it. A prior claim in the same
conversation — "the teacher is 70.6% L4, so ~a third of the corpus is wrong" —
was WRONG: 70.6% is the teacher's score under the PROBE's adversarial settings
(bare framing, 96-token budget, T=0.4). The corpus is full chat-formatted
generations with a think block and a complete answer. Different instrument,
different number. This measures the corpus on its own terms.

HOW. `data_teacher_code` rows are OpenCodeInstruct-seeded chat transcripts whose
PROMPT carries `**Sample Input:**` / `**Sample Output:**` fenced blocks and whose
ANSWER carries a ```python block. Where the answer defines exactly one top-level
function of one argument and both sample blocks literal_eval cleanly, the row
self-adjudicates: call the function on the input, compare to the output.

⚠ TWO VERDICTS, AND THE STRICT ONE UNDERSTATES CORRECTNESS.
  strict      `==`. Counts `(1,2)` vs `[1,2]` and `'42'` vs `42` as failures.
  normalised  tuple~list, numeric-string~number, dict key order, float 1e-6.
Read the normalised number. Inspecting the residue by hand (2026-09-26) found it
is dominated by NON-defects: functions that sort IN PLACE and return None,
`random`-based functions whose sample output cannot be reproduced, set-order
differences, and rows where the SAMPLE OUTPUT ITSELF is wrong and the code is
right (e.g. calculate_total_sales: 120.5+199.75 = 320.25, prompt says 310.25).

⚠ COVERAGE. Only ~28% of rows self-adjudicate, and they skew simple — one
argument, literal I/O. Treat the result as an upper bound on the whole corpus.

SAFETY: corpus code is model output. It is executed in a subprocess, in a temp
dir, under a timeout, never in-process — the same rule as tools/code_eval.py.
Output is read from after an `@@@BEGIN@@@` marker so a row's own `print()` calls
can never be mistaken for a verdict (that bug produced three phantom failures
before it was caught).
"""
from __future__ import annotations
import argparse, ast, glob as _glob, json, os, random, re, subprocess, sys, tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

SPLIT = "<|im_start|>assistant"
IN_RE = re.compile(r"\*\*Sample Input[^*]*:?\*\*\s*\n```\s*\n(.*?)```", re.S)
OUT_RE = re.compile(r"\*\*Sample Output[^*]*:?\*\*\s*\n```\s*\n(.*?)```", re.S)

RUNNER = '''
import sys, json
{code}
def _norm(x):
    if isinstance(x, (tuple, list)): return ["#seq"] + [_norm(v) for v in x]
    if isinstance(x, dict):  return ["#map", sorted((str(k), _norm(v)) for k, v in x.items())]
    if isinstance(x, bool):  return ["#b", x]
    if isinstance(x, (int, float)): return ["#n", round(float(x), 6)]
    if isinstance(x, str):
        s = x.strip()
        try: return ["#n", round(float(s), 6)]
        except Exception: return ["#s", s]
    return ["#o", repr(x)]
print("@@@BEGIN@@@")
for _a, _e in json.loads(sys.argv[1]):
    try: _r = {fname}(_a)
    except Exception as _ex:
        print(json.dumps(["RAISE", type(_ex).__name__])); continue
    _s = (_r == _e)
    try: _l = _s or (_norm(_r) == _norm(_e))
    except Exception: _l = _s
    print(json.dumps(["V", bool(_s), bool(_l), repr(_a)[:100], repr(_e)[:60], repr(_r)[:60]]))
'''


def extract(line: str):
    """Return (code, fname, pairs) if the row self-adjudicates, else a reason string."""
    d = json.loads(line)
    text = d["text"]
    if SPLIT not in text:
        return "no_assistant"
    prompt, answer = text.split(SPLIT, 1)
    blocks = re.findall(r"```python\s*\n(.*?)```", answer, re.S)
    if not blocks:
        return "no_python_block"
    code = blocks[0]
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "syntax_error"
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1:
        return "not_one_fn"
    if len(fns[0].args.args) != 1:
        return "multi_arg"
    ins, outs = IN_RE.findall(prompt), OUT_RE.findall(prompt)
    if not ins or not outs:
        return "no_io_block"
    if len(ins) == len(outs) > 1:                    # "Sample Input 1/2" style
        il, ol = [i.strip() for i in ins], [o.strip() for o in outs]
    else:
        il = [x for x in ins[0].strip().split("\n") if x.strip()]
        ol = [x for x in outs[0].strip().split("\n") if x.strip()]
    if len(il) != len(ol):
        return "len_mismatch"
    pairs = []
    for i, o in zip(il, ol):
        try:
            pairs.append((ast.literal_eval(i.strip()), ast.literal_eval(o.strip())))
        except Exception:
            return "unparseable_literal"
    return (code, fns[0].name, pairs)


_CASES: list = []


def _run(i: int):
    code, fname, pairs = _CASES[i]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.py")
        with open(p, "w") as fh:
            fh.write(RUNNER.format(code=code, fname=fname))
        try:
            r = subprocess.run([sys.executable, p, json.dumps(pairs)],
                               capture_output=True, text=True, timeout=10, cwd=td)
        except subprocess.TimeoutExpired:
            return i, []
        except Exception:
            return i, []
    if "@@@BEGIN@@@" not in r.stdout:
        return i, []
    out = []
    for ln in r.stdout.split("@@@BEGIN@@@", 1)[1].strip().split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return i, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data_teacher_code/shard_*.jsonl")
    ap.add_argument("-n", "--samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16,
                    help="keep modest: a training leg shares this box")
    ap.add_argument("--json", default=None)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    files = sorted(_glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    rows: list[str] = []
    for f in files:
        with open(f) as fh:
            rows += fh.readlines()
    random.seed(args.seed)
    samp = rows if args.samples >= len(rows) else random.sample(rows, args.samples)
    print(f"{args.glob}: {len(rows):,} rows in {len(files)} shards; sampling {len(samp):,}")

    why = Counter()
    for line in samp:
        got = extract(line)
        (_CASES.append(got) if isinstance(got, tuple) else why.update([got]))
    n_adj = len(_CASES)
    print(f"  self-adjudicating rows: {n_adj} ({n_adj/len(samp):.1%})   "
          f"skipped: {dict(why.most_common())}")
    if not n_adj:
        raise SystemExit("nothing to adjudicate")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = dict(ex.map(_run, range(n_adj), chunksize=8))

    strict = lenient = n = 0
    residue = []
    for i, verd in res.items():
        if not verd:
            continue
        n += 1
        vs = [v for v in verd if v[0] == "V"]
        raised = any(v[0] == "RAISE" for v in verd)
        if vs and not raised and all(v[1] for v in vs):
            strict += 1
        if vs and not raised and all(v[2] for v in vs):
            lenient += 1
        else:
            bad = [v for v in vs if not v[2]]
            if bad:
                residue.append((_CASES[i][1], bad[0]))
    print(f"\n  rows with a verdict   {n}")
    print(f"  STRICT ==             {strict:5d}  {strict/n:6.1%}   (understates: counts (1,2) vs [1,2])")
    print(f"  TYPE-NORMALISED       {lenient:5d}  {lenient/n:6.1%}   <- read this one")
    print(f"  residue               {n-lenient:5d}  {(n-lenient)/n:6.1%}   mostly in-place sorts, random(), bad sample outputs")
    if args.show and residue:
        print(f"\n  residue sample (inspect before calling any of it a defect):")
        for fname, v in residue[:args.show]:
            print(f"   fn={fname:26s} arg {v[3][:46]:48s} expected {v[4][:24]:26s} got {v[5][:24]}")
    if args.json:
        with open(args.json + ".tmp", "w") as fh:
            json.dump({"glob": args.glob, "sampled": len(samp), "adjudicable": n_adj,
                       "verdicts": n, "strict": strict, "normalised": lenient,
                       "skipped": dict(why)}, fh, indent=2)
        os.replace(args.json + ".tmp", args.json)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
