"""
Convert OpenCodeInstruct into teacher-corpus JSONL shards — the shape
`--teacher-data-files` consumes. NO GPU, no teacher generation.

WHY THIS EXISTS. 2026-08-22: form is fixed and content never moved. Strict L4 has
been 0-2/320 since step 64,000 and median `body_stmts` is 1 at every α ever
measured — the model writes one plausible statement and stops, because nothing in
its training signal ever rewarded FINISHING a task. It trained almost entirely on
CONTINUATION data (codeparrot files, teacher continuations of web text).

The obvious asset, `data_teacher_chat/`, turns out to be the wrong shape: its
templates are "Explain the following passage in your own words" and only ~17% of
rows contain code at all. It teaches passage QA, not task completion.

OpenCodeInstruct is the right shape and has been configured all along
(`mythouro/sft_data.py`, 200,000 samples) — task in, complete solution out, with
`tests_execution_status` recording real unit-test results. But it is only
reachable from `training/sft.py`, and SFT has collapsed this model at every dose
and every batch size. The channel that WORKS — teacher-data through distillation —
has never been fed it.

So: convert, don't harvest. The solutions are pre-verified by their own test
suites, which also means no fabrication risk (unlike the medical harvest's "PAWL
study"). Accept only rows where EVERY test passed, reusing `_to_messages_opencode`
so the filter matches what SFT would have used.

    python -m tools.make_code_corpus --inspect            # checks only, no writes
    python -m tools.make_code_corpus --target-rows 100000 --out-dir data_teacher_code
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mythouro.sft_data import _to_messages_opencode              # noqa: E402

_SYS = "You are a helpful assistant."


def _body_stmts(code: str) -> int:
    """Statements in the first function body — the metric the model is stuck at 1 on."""
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return -1
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            return len(n.body)
    return 0


def _fenced(out: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", out, re.S)
    return m.group(1) if m else out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--dataset", default="nvidia/OpenCodeInstruct")
    p.add_argument("--split", default="train")
    p.add_argument("--sample", type=int, default=2000,
                   help="Rows to stream for --inspect.")
    p.add_argument("--target-rows", type=int, default=0,
                   help="Accepted rows to write. 0 = inspect only.")
    p.add_argument("--out-dir", default="data_teacher_code")
    p.add_argument("--rows-per-shard", type=int, default=20000)
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset(a.dataset, split=a.split, streaming=True)

    seen = kept = 0
    stmts, chars, toks = [], [], []
    writer = None
    shard = 0
    written = 0
    os.makedirs(a.out_dir, exist_ok=True) if a.target_rows else None

    for row in ds:
        seen += 1
        msgs = _to_messages_opencode(row)
        if msgs:
            kept += 1
            sol = _fenced(msgs[1]["content"])
            s = _body_stmts(sol)
            if s >= 0:
                stmts.append(s)
            chars.append(len(msgs[0]["content"]) + len(msgs[1]["content"]))
            if a.target_rows:
                if writer is None or written % a.rows_per_shard == 0:
                    if writer:
                        writer.close()
                    path = os.path.join(a.out_dir, f"shard_{shard:04d}.jsonl")
                    writer = open(path, "w")
                    shard += 1
                text = (f"<|im_start|>system\n{_SYS}<|im_end|>\n"
                        f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n"
                        f"<|im_start|>assistant\n{msgs[1]['content']}<|im_end|>")
                writer.write(json.dumps({"text": text,
                                         "source": "opencodeinstruct"}) + "\n")
                written += 1
                if written >= a.target_rows:
                    break
        if not a.target_rows and seen >= a.sample:
            break
    if writer:
        writer.close()

    print(f"\n  streamed {seen} rows, accepted {kept} ({100*kept/max(seen,1):.1f}%)")
    if stmts:
        print(f"  solution body_stmts: median {st.median(stmts):.0f} "
              f"mean {st.mean(stmts):.1f} "
              f"| 1-stmt {100*sum(1 for x in stmts if x <= 1)/len(stmts):.0f}% "
              f"| >=3 {100*sum(1 for x in stmts if x >= 3)/len(stmts):.0f}%")
        print(f"    ^ the model is stuck at median 1. Data must be HIGHER than that.")
    if chars:
        c = sorted(chars)
        print(f"  chars/example: p50 {c[len(c)//2]} p90 {c[int(.9*len(c))]} "
              f"max {c[-1]}   (~4 chars/token; seq_len is 1024 tok ≈ 4096 chars)")
        over = sum(1 for x in chars if x > 4096)
        print(f"    over 4096 chars (would truncate): {100*over/len(chars):.0f}%")
    if written:
        print(f"  wrote {written} rows across {shard} shard(s) in {a.out_dir}/")


if __name__ == "__main__":
    main()
