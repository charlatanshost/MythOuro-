"""
Keep only instruction rows whose reasoning a 278M student can actually execute.

    python3 tools/filter_think_corpus.py data_teacher_chat_think --max-think-tokens 200
    python3 tools/filter_think_corpus.py data_teacher_chat_think --apply

WHY THIS EXISTS. `gen_teacher_corpus.py --no-think` was chosen on 2026-08-14 for
two documented reasons, and the second one is the important one:

    "Ouro-2.6B-Thinking emits very long traces: at --max-new 1536, 50% of
     responses never finished reasoning and were rejected as unterminated, at
     ~9 tok/s. Beyond throughput, A 278M STUDENT CANNOT EXECUTE 1500-TOKEN CoT
     — TRAINING ON TRACES TEACHES RAMBLING."

That prediction has to be taken seriously, because rambling is exactly what the
model does now: under chat framing at 512 tokens it opens `<think>`, never
closes (3.1% closure), and 95.3% of samples collapse into token repetition.

But it was trained on ZERO traces and rambles anyway. So the no-think choice did
not buy what it was meant to buy, and "traces teach rambling" is untested at a
length the student can reach.

This filter is the middle path the 08-14 decision did not consider: keep traces,
but only SHORT ones. A row survives only if it

  * opens `<think>` exactly once and closes it exactly once,
  * has a NON-EMPTY reasoning body (the whole point — the old corpus was 99.5%
    empty), and that body is <= --max-think-tokens,
  * has a non-empty answer after `</think>`,
  * terminates with `<|im_end|>`.

The bound matters: the student's own eval budget is 512 tokens. A 200-token
think block leaves ~300 for the answer, so every retained row demonstrates a
trace the student can finish INSIDE the window it is measured in. Rows teaching
reasoning longer than the student can execute are the ones that teach rambling.

Token counts use the teacher tokenizer when available and fall back to a
chars/4 estimate, which is close enough for a length bound.
"""
import argparse, glob, json, os, sys
from collections import Counter

TAG_O, TAG_C, EOT = "<think>", "</think>", "<|im_end|>"
ASSISTANT = "<|im_start|>assistant"


def _assistant_turn(text):
    """Everything the TEACHER generated, excluding the prompt.

    ⚠ Counting tags over the whole row is wrong and silently so. The
    `--think-brief` system prompt contains the literal string "<think>" (it has
    to — it is telling the model what to keep short), so every brief-harvested
    row carries one extra opener from the PROMPT. Measured against the raw text
    on 2026-09-10, the brief pilot read as 22 rows of (1,0) and 39 of (2,1) and
    reported "no closed blocks" — when in fact 39 rows had a perfectly well
    formed block in the assistant turn. Split first, then count.
    """
    return text.split(ASSISTANT, 1)[1] if ASSISTANT in text else text


def _counter(tokenizer_id):
    if not tokenizer_id:
        return lambda s: max(1, len(s) // 4)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    except Exception as exc:
        print(f"  (tokenizer unavailable: {exc}; using chars/4)", file=sys.stderr)
        return lambda s: max(1, len(s) // 4)


def verdict(text, ntok, max_think):
    """None = keep; else the reason it was dropped."""
    turn = _assistant_turn(text)
    if turn.count(TAG_O) != 1 or turn.count(TAG_C) != 1:
        return "bad_tags"
    body = turn.split(TAG_O, 1)[1].split(TAG_C, 1)[0]
    if not body.strip():
        return "empty_think"
    n = ntok(body)
    if n > max_think:
        return "think_too_long"
    if not turn.rsplit(TAG_C, 1)[-1].replace(EOT, "").strip():
        return "no_answer"
    if EOT not in turn:
        return "unterminated"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--max-think-tokens", type=int, default=200)
    ap.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite the shards in place (.prefilter backups). "
                         "Without this it only reports.")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.dir, "shard_*.jsonl")))
    if not shards:
        sys.exit(f"no shards under {a.dir}/")
    ntok = _counter(a.tokenizer)

    stats, kept_tok, lens = Counter(), 0, []
    per_shard = {}
    for p in shards:
        rows = [json.loads(l) for l in open(p)]
        keep = []
        for r in rows:
            why = verdict(r.get("text", ""), ntok, a.max_think_tokens)
            stats[why or "KEPT"] += 1
            if why is None:
                keep.append(r)
                kept_tok += ntok(r["text"])
                body = _assistant_turn(r["text"]).split(TAG_O, 1)[1].split(TAG_C, 1)[0]
                lens.append(ntok(body))
        per_shard[p] = keep

    total = sum(stats.values())
    print(f"\n  {a.dir}: {total} rows, max_think_tokens={a.max_think_tokens}")
    for k, v in stats.most_common():
        print(f"    {k:16s} {v:6d}  {v/total:6.1%}")
    if lens:
        lens.sort()
        print(f"\n  kept think-block tokens: min {lens[0]}  "
              f"median {lens[len(lens)//2]}  max {lens[-1]}")
        print(f"  kept corpus: ~{kept_tok/1e6:.2f}M tokens "
              f"(the no-think corpus it replaces is ~1.0M)")
    if not a.apply:
        print("\n  (report only — pass --apply to rewrite the shards)\n")
        return
    for p, keep in per_shard.items():
        bak = p + ".prefilter"
        if not os.path.exists(bak):
            os.replace(p, bak)
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            for r in keep:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)
    print(f"\n  applied — backups are *.prefilter\n")


if __name__ == "__main__":
    main()
