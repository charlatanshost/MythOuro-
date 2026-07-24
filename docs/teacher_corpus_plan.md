# Teacher-generated corpus (token supply) — design

## What this is, in one breath (read this first when returning after a gap)

**The teacher is writing a textbook for its student, because the student stopped improving
from reading the raw internet.** We give Ouro (the 2.6B teacher) short snippets of real
corpus text as writing prompts, let it continue them at length, filter the junk, and bank
the results (`data_teacher/`). Why it should help, in priority order: (1) the 30k plateau
(tracker 2026-07-19) says more *web* tokens no longer improve the student's own generation —
teacher text is the cheap test of "the data was too noisy" before concluding "the model is
too small"; (2) distillation signal is sharper on text the teacher itself wrote — its
logits there are confident, not mushy (sequence-level KD); (3) it's an unlimited token
supply we control (the #1 recorded bottleneck). It's the mirror of on-policy training:
on-policy = student writes, teacher grades (fixes the student's habits); this = teacher
writes, student imitates (provides good examples). The literature blends both. It enters
training as a 20% mix-in (`--teacher-data-ratio 0.2`), NOT a replacement — an all-teacher
diet risks collapsing the student into one model's style. The A/B probe after the R=0.2
leg decides: score moves → data quality was the wall, lean in; flat → the wall is likely
model size → v6 SFT / scale-up conversations.

**Status: IMPLEMENTED 2026-07-18** (`tools/gen_teacher_corpus.py` + `--teacher-data-ratio`
in distill; smoke-validated on the 5070, suite green). Built after the mid-leg 24,010 probe
read "frontier plateau" — plan-B input. **The A/B (R=0.2 vs 0) stays gated on the 30k verdict.**

**⚠ Measured throughput (final, 2026-07-19 — corrects BOTH earlier estimates):** decode is
**launch-bound on every backend** — wall-clock per step is ~flat in batch, so tokens/s ∝
batch, and batch is MEMORY-capped. Measured: 5070 ~25 accepted tok/s (batch 4); **Max
~56 accepted tok/s at batch 24** = **~4.8M tok/day** (not the 40M/day guessed from the
short-rollout bench). Batch 48 × 768-tok continuations **OOMs the 48 GB card** — the HF
KV-cache `torch.cat` transiently doubles the cache, and an XPU OOM can leave a **zombie
process holding all VRAM** (field-notes gotcha #7; `pkill` + `xpu-smi -m 18` check before
relaunch). Batch 24 peaks ~43 GB. Future throughput levers (backlog): compiled decode step,
on-device top-p (sort appears XPU-safe; topk/multinomial are not), preallocated cache.

**Filter calibration (2026-07-19, from reject-reason telemetry):** the v1 distinct-1 floor
of 0.30 rejected 62% of output — telemetry showed 100% `low_distinct1`, and measurement of
REAL corpus text at ~768 tok (general p10=0.38 / math 0.26 / code **0.23** — distinct-1
falls with length, code is naturally repetitive) proved the floor would reject ~half of
genuine code. **Recalibrated to 0.20** → 75% acceptance, 56 tok/s, mix balance restored;
`top_share` (never fired) remains the true degeneracy guard. Lesson for any length-changed
filter: calibrate against the real corpus at the same length first.

**Spot-read of the admitted 0.20–0.30 band (shard_0002, 2026-07-19): PASS** — contains genuinely good Python (docstringed functions the old floor would have rejected) plus a tolerable template-prose tax; mix on-target (45/35/20). **⚠ v1 SEEDING BUG (found 2026-07-21, fixed): head-seeding harvested boilerplate.** v1 took
`ids[:seed_len]` — the FIRST 48 tokens of each document — and documents open with boilerplate:
source files with license headers, scraped math pages with nav cruft. Measured on the 5.84M v1
corpus: **57% of CODE samples are the teacher faithfully continuing an Apache/copyright header**
(~600k tokens of legalese); math 0.7%, general 0.5% — the bias is code-specific because code
files have the most stereotyped openings. Almost certainly the cause of the fibonacci/quadratic
regression seen in the 34,500 tripwire probe. **Fix:** seed from a RANDOM WINDOW of the first
2048 tokens + a `boilerplate` reject filter (≥2 license-ish matches in the first 800 chars).
The v1 corpus is retained (the running A/B trains on it) — which makes that A/B a **lower
bound**: it improved prose *despite* ~10% of the corpus being license spam.

**Did this affect earlier training runs? NO — measured 2026-07-21.** (a) Every leg before the
R=0.2 A/B ran `--teacher-data-ratio 0.0`, so no teacher text existed in them. (b) The REAL
corpora are clean: sampling 300 docs each, license-opening docs are code 44.3% / math 0% /
general 0.7%, but as a share of *tokens* that is **code 3.9%, math 0%, general 0.2%** ≈ **0.8%
of all training tokens** — the honest distribution of real source files, not an artifact.
`MixedDataset` packs whole documents, so there is no head bias on the real side either.
**The lesson:** head-seeding *amplified* a benign feature into a pathology — a license header
is ~4% of a real file, but a 768-token continuation *seeded on* that header is ~100% legalese.
Same material, ~25× the concentration. Any future seeded-generation pipeline should assume the
seed distribution, not the corpus distribution, determines what you get.

**⚠ MIX DRIFT — the seed mix is NOT the corpus mix (measured 2026-07-23, fixed).** Seeds are
drawn at `_MIX_RATIOS` 40/40/20, but the **accepted** corpus landed at **45.3 / 38.0 / 16.7**
by token. Two compounding causes, both measured on the first 4.28M of v2:
| source | acceptance | mean len | usable tokens per seed |
|---|---|---|---|
| general | 82.5% | 649.6 | 535.8 |
| math | 58.8% | 765.6 | 450.0 |
| code | **51.6%** | 767.6 | **396.3** |
Code passes the filter at *half* general's rate (it is naturally repetitive, so `low_distinct1`
— 96% of all rejects — hits it hardest), and general emits EOS earlier so its samples are
shorter. A code seed therefore returns **26% fewer usable tokens** than a general seed. Left
uncorrected this starves exactly the slice the 07-21 A/B flagged as still regressing.
**Fix: `--seed-mix` (harvest-local override, 2026-07-23).** `_MIX_RATIOS` is **shared with the
training `MixedDataset`** (`training/distill.py`) and must never be moved for harvest reasons —
hence a separate flag. Weights `general=0.3211, math=0.4237, code=0.2552` over-draw code+math
so the **full 12M corpus** lands on 40/40/20, compensating for the drift already banked.
Recompute if the target, the filters, or the measured acceptance rates change. **Generalised
lesson (the sibling of the head-seeding lesson above): the seed distribution is not the corpus
distribution — anything downstream of a filter must be measured at the OUTPUT, not assumed
from the input.**

**⚠ CROSS-SESSION SEED REUSE — every session re-read the same documents (found + fixed 2026-07-23).**
`load_dataset(streaming=True)` iterates from document #1 each time the *process* starts, and the
`--seed` RNG only picks the window *within* a document — so session N re-harvested exactly the
material session N−1 had already used. Measured on the first 4.28M of v2: **6,038 rows came from
only 3,886 distinct seed documents** — 1,588 seed prefixes repeated, *every* repeat spanning
different shards (= different sessions), and 564 documents were used **three** times. Because
sampling is stochastic (T=0.9) the *texts* are all distinct, which is exactly why row-level dedup
never caught it: **the redundancy sits one level up, in the source material.** Direct proof of the
mechanism — pulling seeds twice with the old code gives **100% document overlap** on all three
corpora; with the fix and two different `--stream-seed` values, **0%**.
**Fix: `.shuffle(seed, buffer_size)` on the streaming dataset**, which reorders the **shards** (all
three corpora are many-file) so a session starts somewhere else entirely; the seed is bumped per
epoch so a stream that exhausts and restarts doesn't repeat either. Default is a fresh random
`stream_seed` per session, **recorded in the manifest** — verified reproducible (same seed → 100%
overlap), so a session can still be replayed exactly. `--no-stream-shuffle` is the rollback.
Costs ~100–150 s of one-time startup while the reservoir fills (0.3% of a 15 h run), which drags
the *first* cumulative tok/s print down to ~60; it climbs to steady state after a few batches.
**Generalised lesson, and the third instance of the same shape in this pipeline** (after head-seeding
and mix drift): *every property we care about must be measured on the OUTPUT corpus.* Head-seeding
was assumed-uniform seeding, mix drift was assumed-uniform acceptance, this was assumed-fresh
traversal. Row-level checks (duplicate texts, distinct-1) cannot see any of them.

**Harvest v1 running** (2026-07-19, batch 24, Max): ~4.8M/day → A/B-ready ~10M in ~2 days;
30M ≈ 6 days. At R=0.2 a ~9k-step leg consumes ~10M teacher tokens; launching on ~6.5M
means ~1.5 epochs of teacher data (acceptable mild repetition — owner's call). Backlog items it implements: *teacher-generated synthetic
data* + *sequence-level KD* (ideas.md — one build, two entries). Attacks the #1
bottleneck (token SUPPLY), feeds main-thread #2 (the token curve).

## Why one build

Sequence-level KD (Kim & Rush 2016) = train the student on teacher-generated
sequences. Generating those sequences at scale IS the synthetic-data pipeline.
The training side needs zero new loss code: teacher text enters the ordinary
offline path (hard CE on teacher tokens + soft KL against teacher logits on
them — the KL is computed by the existing step, and on the teacher's own text
its logits are maximally informative).

## Generator — `tools/gen_teacher_corpus.py`

- **Runs on the 5070** (venv-cuda; transformers repinned to <5 on 2026-07-17,
  teacher loads fine) — the Max keeps training. Same division of labor as the
  mid-run probes.
- **Seeding**: draw 32–64-token seeds from the real corpora at `_MIX_RATIOS`
  (general 40 / math 40 / code 20) and let the teacher continue them
  512–1024 tokens. Keeps the topical distribution anchored to the real mix;
  the *text* is teacher-clean. (Pure unconditional generation drifts to the
  teacher's priors; seeded continuation is the standard fix.)
- **Sampling**: temperature ~0.9, top-p 0.95 (sampled, not greedy — greedy
  teacher text is low-entropy and KD-poor). Batch 8–16 (12 GB budget:
  teacher 5.2 GB bf16 + KV cache).
- **Cheap filters at write time**: min length; distinct-1 floor (drop
  degenerate continuations); exact-prefix dedup across a session. No LLM
  judging — keep it dumb and fast.
- **Output**: sharded JSONL `{"text": ..., "source": <seed corpus>, "seed_len": n}`
  under `data_teacher/` (gitignored) + a `MANIFEST.json` per session recording
  teacher id, sampling params, filter stats, date — provenance in the
  dataset_selection.md spirit. Clean-data status: Ouro output, OpenAI-free.

## Loader integration — `MixedDataset`

- Teach `_open_stream` to accept a local spec:
  `("teacher", "json:data_teacher/*.jsonl", None, "train", "text")` →
  `load_dataset("json", data_files=..., streaming=True)`.
- New flag `--teacher-data-ratio R` (default **0.0** = exactly current
  behavior): mix becomes `{general, math, code} · (1−R) + teacher · R`.
  Start R=0.2; it's a knob, fully reversible.

## Throughput / dose math

~~Original estimate (500–1,000 tok/s on the 5070 → 40–80M/day)~~ — **wrong by
~10×; see the measured-throughput block at the top** (launch-bound decode,
memory-capped batch: Max ~56 accepted tok/s ≈ 4.8M/day). The scale framing
survives: the entire distill history is ~200M tokens, so even at measured
rates a week of harvest is a double-digit-percent addition to the model's
lifetime clean-token diet.

## Validation (before trusting it)

1. Spot-read 20 samples per source bucket (the probe-tracker lesson: read the
   text, not just metrics).
2. A/B: continue the main run with R=0.2 vs R=0 for one probe interval
   (~50M tokens), same instrument (`--no-kv-cache`, 5070). Keep only if the
   α=0.0 probe is ≥ flat — the hypothesis is teacher text helps coherence,
   the risk is distribution narrowing.

## Non-goals

- No student involvement in generation (that's the on-policy path, already
  running on the Max) — this is the OFFLINE complement, per the GKD blend.
- No quality curation beyond dumb filters in v1 (phi-style curation is its own
  backlog item; layer later if the A/B is promising but noisy).
- No new divergence/loss code.
