"""
Repair the `</think>` artifacts in data_teacher_chat_clean/.

WHY. The 2026-08-14 harvest ran with `--no-think`, which PREFILLS a closed empty
reasoning block (`<think>\n\n</think>\n\n`) into the assistant turn. In 519 of
2,545 rows the model then emitted a `</think>` of its OWN, so the row carries two
closers. The 2026-08-14 structural check tested for unclosed and REOPENED
`<think>` — both genuinely 0 — but a duplicate CLOSER is a third failure mode
that was never enumerated, so it passed.

Two distinct populations, repaired differently:

  443 rows  `<think>\n\n</think>\n\n</think>\n\nANSWER`
            Nothing between the closers. The model just re-emitted the tag.
            -> drop the surplus closer.

   88 rows  `<think>\n\n</think>\n\nDRAFT\n</think>\n\nANSWER`
            The model IGNORED the prefill and answered anyway, then closed and
            rewrote a refined answer. The DRAFT is real generated content sitting
            OUTSIDE the block its own tag closes.
            -> move it INSIDE: `<think>\nDRAFT\n</think>\n\nANSWER`.

            This is the conservative reading. The alternative — deleting the
            draft — throws away 88 of the only draft-then-refine exemplars in the
            corpus. Leaving it as-is is the one option that is clearly wrong: it
            trains the model to emit prose after a CLOSED think block, which is
            the exact shape the run is trying to test for.

Backups: `<shard>.predup`, matching the `.preschema` convention.
Idempotent — a repaired corpus is a no-op on re-run.
"""
import glob, json, os, re, sys

TAG_O, TAG_C = "<think>", "</think>"


def repair(text: str):
    """Repair to a FIXED POINT and return (new_text, kind).

    Single-pass is not enough: 12 rows carry three or four closers, so one pass
    leaves the row still unbalanced. Loop until nothing changes.
    """
    # Consecutive closers separated only by whitespace are always surplus —
    # including after a block this pass has just made legitimate, which the
    # non-empty-body guard below (correctly) refuses to reopen.
    kind = None
    collapsed = re.sub(r"</think>\s*(?=</think>)", "", text)
    if collapsed != text:
        text, kind = collapsed, "dropped"
    for _ in range(8):
        text, k = _repair_once(text)
        if k is None:
            break
        kind = k if kind in (None, k) else "moved"
    return text, kind


def _repair_once(text: str):
    """One surplus closer. Returns (new_text, None | 'dropped' | 'moved')."""
    if text.count(TAG_C) < 2 or TAG_O not in text:
        return text, None
    o = text.index(TAG_O)
    c1 = text.index(TAG_C, o)
    if text[o + len(TAG_O):c1].strip():
        return text, None          # a real, non-empty block — never touch it
    c2 = text.index(TAG_C, c1 + len(TAG_C))
    body = text[c1 + len(TAG_C):c2]
    head, tail = text[:o], text[c2 + len(TAG_C):]
    if not body.strip():
        return head + TAG_O + "\n\n" + TAG_C + tail, "dropped"
    return head + TAG_O + "\n" + body.strip() + "\n" + TAG_C + tail, "moved"


def main(d="data_teacher_chat_clean", apply=False):
    shards = sorted(glob.glob(os.path.join(d, "shard_*.jsonl")))
    if not shards:
        sys.exit(f"no shards under {d}/")
    tot = {"dropped": 0, "moved": 0, "clean": 0}
    for path in shards:
        rows = [json.loads(l) for l in open(path)]
        out = []
        for r in rows:
            new, kind = repair(r["text"])
            tot[kind or "clean"] += 1
            r["text"] = new
            out.append(r)
        if apply:
            bak = path + ".predup"
            if not os.path.exists(bak):
                os.replace(path, bak)
            else:                                   # already backed up; rewrite
                pass
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                for r in out:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
    print(f"{d}: {sum(tot.values())} rows  "
          f"surplus-closer dropped {tot['dropped']}  draft moved inside {tot['moved']}  "
          f"untouched {tot['clean']}")

    if apply:
        rows = [json.loads(l) for f in shards for l in open(f)]
        multi = sum(1 for r in rows if r["text"].count(TAG_C) > 1)
        unclosed = sum(1 for r in rows
                       if r["text"].count(TAG_O) != r["text"].count(TAG_C))
        reopened = sum(1 for r in rows if r["text"].count(TAG_O) > 1)
        nonempty = sum(1 for r in rows if TAG_O in r["text"]
                       and r["text"].split(TAG_O, 1)[1].split(TAG_C, 1)[0].strip())
        print(f"  after: >1 closer {multi}   unclosed {unclosed}   "
              f"reopened {reopened}   non-empty think {nonempty}")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
