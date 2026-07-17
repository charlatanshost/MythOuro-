# Teacher-generated corpus (token supply) — design

**Status: DESIGN, 2026-07-17.** Implementation gated on the 18,000 probe passing
(frontier past 8668). Backlog items it implements: *teacher-generated synthetic
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

Ouro-2.6B batched KV-cached generation on the 5070, batch ~12, seq ~768:
plausibly 500–1,000 tok/s sustained → **~40–80M filtered tokens/day** of
teacher-clean text, generated for free alongside training. For scale: the
entire distill history to date is ~200M tokens. A week of background
generation ≈ doubles the clean-token supply.

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
