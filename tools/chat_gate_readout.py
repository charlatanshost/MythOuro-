"""
Read a chat-framed code_eval report and answer ONE question: does the model
close `<think>` and answer, or was it simply interrupted?

    python3 tools/chat_gate_readout.py reports/chat_instruct_3000_512.json [more.json ...]

WHY THIS IS SEPARATE from the summary block inside run_chat_eval.sh: that block
shipped with a broken truncation rule (2026-09-07). It flagged truncation when
`(max-min)/max < 0.5`, and the fully-truncated max_new=96 run it was written for
measured **0.508** — just past the threshold, so it stayed silent on the exact
case that motivated it. Measured separation on real reports:

    report                     >=80% of longest    truncated?
    chat_instruct_3000  (96)         85%           yes, all 320
    code_instruct_3000  (96)         26%           no
    code_exitpdf_7200   (96)         18%           no

Piling near the ceiling separates cleanly where spread does not, so the rule is
now "60%+ of completions within 80% of the longest".
"""
import json, sys, os


def readout(path):
    d = json.load(open(path))
    S = [s for t in d["tasks"] for s in t["samples"]]
    n = len(S)
    if not n:
        print(f"{path}: no samples"); return
    L = sorted(len((s.get("completion") or "")) for s in S)
    mx = d.get("max_new")
    ceil_frac = sum(1 for x in L if x >= 0.8 * L[-1]) / n
    truncated = ceil_frac >= 0.60

    # DEGENERACY. The tag counts are meaningless if the text is token soup —
    # 2026-09-08: 20.3% of instruct completions "closed and answered", and the
    # answers were "three three three four four four five five seven seven".
    # A closer emitted amid noise is not an answer.
    dg = sum(1 for s in S if float(s.get("adj_repeat_frac", 0) or 0) >= 0.10) / n
    lp = sum(1 for s in S if int(float(s.get("max_line_repeat", 1) or 1)) >= 3) / n

    op = cl = ans = emp = 0
    for s in S:
        c = s.get("completion") or ""
        o, k = c.count("<think>"), c.count("</think>")
        if o:
            op += 1
            if k:
                cl += 1
                if c.rsplit("</think>", 1)[-1].strip():
                    ans += 1
                if not c.split("<think>", 1)[1].split("</think>", 1)[0].strip():
                    emp += 1

    print("=" * 68)
    print(f"  {os.path.basename(path)}   step {d.get('step')}   "
          f"chat={d.get('chat_template')}   max_new={mx}   n={n}")
    print("=" * 68)
    print(f"  completion chars: min {L[0]}  median {L[len(L)//2]}  max {L[-1]}")
    print(f"  within 80% of longest: {ceil_frac:.0%}"
          f"   -> {'TRUNCATED — the cap is the result' if truncated else 'generation is stopping naturally'}")
    print(f"  adjacent-token degenerate: {dg:.1%}   line-looping: {lp:.1%}")
    if dg >= 0.5:
        print("  ⚠⚠ THE TEXT IS COLLAPSING. Tag counts below describe noise, not")
        print("     behaviour: a `</think>` emitted amid token soup is not an answer.")
    if truncated:
        print("  ⚠ Tag counts below are NOT interpretable. Raise MAXNEW and re-run.")
    print()
    for lbl, k in (("opened <think>", op), ("...and CLOSED it", cl),
                   ("...answered after it", ans), ("...with an EMPTY think block", emp)):
        print(f"  {lbl:30s} {k:4d}/{n:<4d} {k/n:6.1%}")

    print()
    if truncated:
        print("  VERDICT: inconclusive. The model ran out of budget, which is what")
        print("           rung 8 measured (median 376-414 chars, closed 1-3/80).")
    elif dg >= 0.5:
        print("  VERDICT: DEGENERATE at this length. Not a closure result either")
        print("           way — the model collapses into token repetition before it")
        print("           finishes. Raising the budget EXPOSED this; it did not")
        print("           cause it. Compare bare framing at the SAME max_new")
        print("           before blaming the framing rather than the length.")
    elif op and cl / op > 0.9 and ans / op > 0.9:
        print("  VERDICT: it CLOSES and ANSWERS. Rung 8's failure does not reproduce")
        print("           when the budget allows it -> that failure was the INSTRUMENT.")
    elif op and cl / max(op, 1) < 0.5:
        print("  VERDICT: opens and does not close, WITH room to. This is a real")
        print("           model behaviour, not the budget. Rung 8's finding stands.")
    else:
        print("  VERDICT: mixed — read the completions before calling it.")
    if op and emp / op > 0.5:
        print("  ⚠ Most blocks are EMPTY: it learned the corpus's no-think SHAPE")
        print("    (99.5% of rows), which is not the same as learning to answer.")
    print("\n  ⚠ Read the completions in the json. Counts are a screen, not the call.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        readout(p)
