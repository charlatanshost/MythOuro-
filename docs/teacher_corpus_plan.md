# Teacher-generated corpus (token supply) — design

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
