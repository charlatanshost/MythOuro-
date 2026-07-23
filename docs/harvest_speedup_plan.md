# Harvest throughput speedups — design & benchmark plan

**Status: DESIGN, 2026-07-21.** Goal: more teacher-corpus tokens per card-hour.
The A/B verdict (tracker 2026-07-21) proved teacher data is the lever moving the
plateau, so faster harvest = faster iteration on the one intervention that's
working — and, paired with a second card, the difference between corpus
generation being a bottleneck vs a solved problem.

## Baseline (measured 2026-07-19/21)

- `tools/gen_teacher_corpus._generate_xpu_safe`, batch 24, 768-tok continuations,
  Ouro-2.6B teacher (4 UT loops), bf16, Max 1100.
- **~55 accepted tok/s ≈ 4.8M tok/day.** Decode is **launch-bound** (per-step
  wall-clock ~flat in batch → throughput ∝ batch), and batch is **memory-capped
  at 24** because the HF KV-cache `torch.cat` transiently doubles the cache each
  step (batch 48 OOMs the 48 GB card; batch 24 peaks ~43 GB).
- Sampling runs on **CPU** (`.cpu()` each step) — XPU `topk`/`multinomial`
  segfault; `sort`/`cumsum` are XPU-safe.

## Two classes of lever — keep them separate

**SAFE (distribution-preserving):** the teacher's per-token output distribution
is byte-for-byte unchanged; only the schedule/packing/memory changes. No corpus-
quality risk. Prefer these.

**RISKY (distribution-changing):** the teacher emits a *different* (worse)
distribution. Trades corpus quality — the exact thing the A/B proved is our
lever — for speed. Avoid unless a benchmark proves the quality cost is
negligible, gated like §Correctness below.

---

## ⚠ RANKING REVISED BY MEASUREMENT (2026-07-22) — read before the catalog

Two measurements taken before building (the doc's own benchmark-first rule,
applied one step earlier):
1. **The EOS-waste that motivated continuous batching is small.** Measured on
   1,006 accepted v2 samples: p50 = 768 (the cap), **85% run ≥750 of 768**,
   post-EOS idle = **7.8%** of accepted lanes. In a launch-bound regime, idle
   lanes cost ~no time anyway — the only real lever is MORE lanes. **Continuous
   batching demoted from top pick to "small win, high complexity — deferred."**
   (Reject lanes ~30% still waste compute, but reclaiming them needs the same
   mixed-length cache surgery; not worth it at this payoff.)
2. **Every op needed for on-device sampling is XPU-safe on this card/driver**
   (sort, cumsum, rand, searchsorted, cmp+sum, gather — micro-tested 07-22;
   searchsorted and cmp+sum indexing agree exactly). The `topk`/`multinomial`
   segfaults are the only bad ops.

**⚡ BENCHMARKED (2026-07-23, `tools/bench_harvest.py`, gates PASSED at 1.0e-03 nats on
real-text probes):** A stock+cpu b24 = 116.4 raw tok/s → B +device-sampling = 138.4
(**+19% — the host-sync tax was real**) → D prealloc b32 = **178.6 (1.53×)**, peak 47.1 GB.
b40 needs ~58 GB → OOM (caught in-config; incremental-results design worked). Prealloc at
iso-batch is ~neutral speed but **saves ~6 GB** (36.6 flat vs ~43 grown+transient). Scaling
b24→b32 near-linear — the launch-bound model confirmed. **Adopted config: `--prealloc-cache
--batch 30`** (~44.5 GB, ~3.5 GB margin for multi-day runs; ~1.43× ≈ **~70 accepted tok/s**
vs 49). Bench gotcha discovered: greedy-argmax gates FALSE-FAIL on random-token probes (flat
distributions → bf16 near-ties; first bench run silently measured the uncached path) — gates
must probe with REAL text, and benches must ABORT, not fall back, when an engine is off.

**✅ PRODUCTION-VALIDATED (2026-07-23 afternoon): 101 accepted tok/s = 2.06× the old 49 —
BEATING the bench's 1.43× projection.** Why the bench understated: it timed 128-token
generations, but the stock cache's `torch.cat` copy cost **grows with sequence length** —
by token ~700 it's copying ~700-token buffers across all 96 UT-cache entries every step,
while prealloc's slice-write stays O(1). At the real 768-token workload prealloc's win is
much larger than at bench length. Lesson for the protocol: **bench throughput levers at the
production sequence length** (short benches are only honest for memory peaks, and only with
prealloc). Live config: `--batch 30 --prealloc-cache` + default on-device sampling, 45.3 GB
resident (~2.7 GB headroom). v2 acceptance note: 67% vs v1's 75% (mid-document random-window
seeds produce slightly more `low_distinct1` rejects — the price of the boilerplate fix,
net hugely positive). New rate ≈ **8.7M tok/day**.

**⚠ Manifest bug found (2026-07-23, fixed in code for future sessions):** `MANIFEST.json`
counters were per-session — each relaunch reset and overwrote, so multi-session corpora
under-reported (v2 read "2.13M" when 2.84M were on disk). Now the manifest keeps a
`sessions` list and cumulative totals; rows-on-disk remain the ground truth for old corpora.

**Build status (2026-07-22):** lever 3 (on-XPU sampling) **BUILT** — default on,
`--cpu-sampling` rollback flag. Lever 2 (prealloc cache) **BUILT + unit-tested**
(`tools/prealloc_ut_cache.py`, 5 CPU tests green; runtime subclass of the
teacher's own UT cache class, slice-write update, reorder support) behind
`--prealloc-cache` with a **blocking KL equivalence gate** at startup. Cache
overridability spike: ANSWERED — subclassing works; the stock class even ships
`reorder_cache` (batch index_select) if slot surgery is ever wanted. **Both
levers await the on-card benchmark** (card busy harvesting): equivalence gate,
then batch 32/40 sweep vs the 24-baseline.

## Standing directive (owner, 2026-07-23)

> **The harvest should be the fastest and best we can make it — squeeze every
> lever, when windows allow.** Quality levers may outrank speed levers in
> scheduling, but no known speed lever is ever "done enough" to delete from
> this list; they queue for idle card windows.

**❌ SEED-PREFETCH KILLED BY MEASUREMENT (2026-07-23 evening) — honest negative.**
The queued estimate was "5–15 s idle per ~230 s cycle ≈ 3–6%". Measured directly
(`_seed_streams` + the real post-generate decode/filter path, batch 30, n=5):
**seed fetch = 0.21 s median** (min 0.09, max 0.51), **decode+filter = 0.01 s**.
Against the *current* 143 s cycle (b30 @ 101 tok/s — the 230 s figure was the old
b24 config) that is **0.15% + 0.01% = 0.16% total GPU-idle CPU work**. The
estimate was ~50× too pessimistic. Why it's so cheap: the seed corpora are
`streaming=True` over HTTP with **metadata-only local caches** (36–60 KB each),
and HF streams *parquet row groups* — one range request yields thousands of
documents, so per-seed cost amortises to ~7 ms; only row-group boundaries cost
anything (that's the 0.51 s max). **Do not build it** — the ceiling is 0.16%
even if prefetch were free and perfect. Lesson (same shape as the 07-22
continuous-batching demotion): *measure the idle before building machinery to
hide it.* Two for two — both "obvious" scheduling wins evaporated under a
stopwatch, because this loop is genuinely ~100% GPU-bound.

**Corollary — the ONLY remaining software levers are launch-count levers.** With
CPU-side idle at 0.16%, nothing is left to overlap. Throughput now moves only by
(a) more lanes per launch (batch) or (b) fewer/cheaper launches (compile,
speculation). That is exactly the launch-bound model, now confirmed from the
idle side too.

**Remaining squeeze queue after the 07-23 productionization (101 tok/s live):**
1. **Continuous batching / reject-lane refill** — **re-promoted 2026-07-23**, ~1.28×
   (see the re-promotion note in the catalog). Gated on reading a full run's
   `--telemetry` to fix per-source abort thresholds.
2. **torch.compile the decode step** — +10% on training; launch-bound decode may
   gain more. The top lever that needs no new measurement.
3. **b32 revisit** — +6–7% over b30, gated on observing long-run fragmentation
   behaviour at 45.3 GB (a multi-day run that never creeps → the 0.9 GB margin
   at b32 may be acceptable; or shave `--max-new` to ~704 to buy the margin).
3. **Speculative decode** — the wildcard (0 to +2×); benchmark-gated as designed.
4. **Second card** — the hardware multiplier (+~100%/card); see hardware_options.
5. ~~Seed-prefetch thread~~ — **killed by measurement, see above.**

## The catalog (ranked by value × safety ÷ effort)

### ⚠ 1. Continuous batching — SAFE — **RE-PROMOTED 2026-07-23** (demoted 07-22 on the wrong number)

**The 07-22 demotion measured the wrong waste.** It sized continuous batching on
**post-EOS idle = 7.8%** and shelved it. But the dominant waste is **REJECTS: 33%
of all generations** (9,035 attempts → 6,038 accepted) each burn a full 768
tokens and are thrown away — **4× the EOS waste the demotion was based on.**
Abort a doomed lane at token 256 and refill it and the recoverable compute is
`2,997 × 512 / (9,035 × 768)` ≈ **22% → ~1.28×**, larger than compile (+10%) or
b32 (+6–7%). The prealloc cache (lever 2) is already built, gated, and in
production — and this doc already notes it is the natural substrate for the
ragged per-slot cursor continuous batching needs. The 07-22 note stands only for
the *EOS* slice; it should not have carried the reject slice with it.

**Safety half measured (2026-07-23), on accepted v2 rows — false-kill rate if we
abort at N tokens below a distinct-1 threshold:**
| N | thr 0.20 | 0.25 | 0.30 | 0.35 |
|---|---|---|---|---|
| 128 | 0.23% | 0.45% | 1.09% | 2.68% |
| 256 | 0.62% | 1.73% | 4.16% | 10.18% |
| 384 | 0.65% | 2.84% | 10.18% | 25.47% |
Conservative aborts are nearly free of false kills (0.62% at N=256/thr 0.20).
**The unknown is the CATCH rate** — rejects never reach the shards, so their
early-token behaviour was unmeasurable. Hence `--telemetry` (added 2026-07-23):
logs `d1_128/256/384`, `d1_final`, `top_share`, source and reject reason for
**every** sample, accepted or rejected. First live sample already shows clean
separation on code (accepted `d1_256`=0.535 vs rejected 0.328). **Build only
after reading a full run's telemetry** — and note the threshold must be
**per-source**, because distinct-1 aborts would preferentially kill *code* (the
naturally-repetitive slice we are simultaneously trying to strengthen; see the
mix-drift fix in teacher_corpus_plan.md), which would silently undo that fix.

### 1a. Continuous batching — original write-up

**The waste, seen in code:** `_generate_xpu_safe` runs `for i in range(max_new)`
with **no early stop**. Every sequence generates the full 768 tokens even after
it emits `<|endoftext|>` at token 50 (main() trims at EOS only *after*). On top
of that, ~25% of finished samples are dropped by the reject filter — their full
768-token compute is pure waste. In a static batch of 24, a slot that finishes
early (EOS or doomed-to-reject) idles until the *longest* sequence in the batch
reaches 768.

**The fix:** refill each batch slot with a fresh seed the moment it frees
(EOS emitted, or an online degeneracy check trips). All 24 slots always do
useful work. Recovers BOTH the post-EOS waste and the reject waste. This is what
vLLM's continuous batching does; we implement a minimal version in the manual
loop.

**Why it's the top pick:** larger *expected* win than speculation and far more
certain — it attacks waste we can *see*, not a theoretical latency gain that may
not survive the batched regime. Completely distribution-preserving. **Stacks
with every other lever below.**

**Effort:** medium — rewrite the batch loop to track per-slot state (position,
done-flag), evict+refill finished slots, handle the ragged KV cache (each slot
at a different length). The KV-cache raggedness is the real work; a
preallocated cache (lever 2) makes it much easier, so build them together.

**Sub-win — online degeneracy abort:** check distinct-1 / top-share at ~256
tokens; if a sample is already doomed to fail the reject filter, evict it now
instead of finishing 768. Recovers the reject compute *before* it's spent.

### 2. KV-cache preallocation — SAFE — **stack-mate of #1**

The `torch.cat` doubling caps batch at 24. Preallocate the cache to
`prompt+max_new` up front and write in place (no `cat`) → the transient
doubling disappears → **batch 32–40 fits at 768**. Launch-bound ⇒ ~1.4–1.7×
throughput. Also the natural substrate for continuous batching's ragged cache
(fixed buffer, per-slot write cursor). No distribution change.

**Effort:** medium, **risk unknown until checked** — depends on whether Ouro's
`UniversalTransformerCache` allows a preallocated / in-place-write mode or must
be subclassed. First task: read the cache class, determine overridability. If
the custom cache resists, a fallback is a hand-rolled cache in our decode loop
that calls the model layer-by-layer (more invasive).

### 3. On-XPU sampling — SAFE — small, low-risk

Replace the per-step `.cpu()` + `multinomial` with an XPU-native top-p sample
(`sort` → `cumsum` → threshold → `searchsorted`/`gather`, all XPU-safe). Removes
a host↔device round-trip every token — meaningful in a launch-bound loop.
Distribution-identical (same top-p math). Low effort; good warm-up task.

### 4. torch.compile the decode step — SAFE — medium risk

+10% measured on the *training* step (hardware_options.md); launch-bound decode
may gain more because compile fuses kernels and cuts exactly the launch overhead
that dominates. Risk: `torch.compile` + XPU + the custom UT cache + dynamic
shapes (growing sequence) may not compile cleanly or may hit graph breaks. Try
after 1–3; measure, keep only if it helps.

### 5. Self-speculative decode (`exit_at_step`) — SAFE\* — high effort, uncertain

The one the owner first asked for. **\*Safe only behind a proven equivalence
gate** — a subtle bug silently skews the distribution (the 07-16 cached-decode
failure mode exactly). See §Correctness and §The open question. Kept in the plan
because it's the *only* way to get the loop-reduction speedup WITHOUT the quality
cost of lever 7 — but ranked below 1–3 because its benefit in our batched
throughput regime is genuinely uncertain and it's the most bug-prone build.

Feasibility (from reading the model):
- **Ouro exposes `exit_at_step`** in `OuroForCausalLM.forward`: `exit_at_step=1`
  → 1-UT-loop logits (cheap draft), `None` → full 4-loop (verify). No model
  surgery; the draft is the same weights at reduced depth.
- **The hard part is cache rollback.** `UniversalTransformerCache` has per-UT-
  loop slots; speculative decoding must roll the cache back to the accepted
  length on every rejection, across all 4 loop sub-slots. Bug-prone.
- **Gate template exists:** `_validate_teacher_cache` (KL(uncached‖cached) <
  5e-2 nats + argmax match) is exactly the equivalence check shape needed.

### 6. Shorter continuations + higher batch — SAFE(dist) — zero code

Drop `--max-new` 768→384: halves KV memory → ~batch 40–48 → more parallel
sequences. Zero code (flags only). Distribution-preserving per token, but the
*corpus* changes (shorter samples). 384 tokens is still substantial for seq-KD,
so this is a legitimate quick lever if a fast win is wanted before 1–2 are
built. Note: not free lunch — shorter samples mean more seeds/less context per
sample.

### 7. Fewer teacher loops, unconditionally — RISKY — **do not, except as a dial**

`exit_at_step=2` for ALL generation ≈ 2× faster, but the teacher runs at half
depth → a *different, dumber* distribution. Directly trades the corpus quality
the A/B proved is our lever. Recorded only so it's not re-proposed as "free 2×":
it isn't free, it's quality-for-speed, and speculation (5) exists precisely to
get the loop-reduction win *without* this cost.

### 8. Greedy instead of sampled — RISKY — no

Faster (no sampling step) but greedy teacher text is low-entropy and KD-poor
(teacher_corpus_plan.md §Generator). Rejected.

---

## Correctness gate (blocking, for ANY distribution-touching lever)

Mirror `_validate_teacher_cache`: at startup, generate from a fixed prompt both
the new way and the reference full-4-loop way; assert token-distribution
KL under tolerance AND greedy-argmax match for N steps. **Refuse to run on
failure.** A silent skew here poisons the corpus, and unlike a training bug it's
invisible until after the model has trained on it (cf. the 07-16 cached-decode
incident — a "distribution unchanged" claim that was false by ~1 nat and cost a
week). Levers 1–4 and 6 don't need this (they don't change the distribution),
but the gate is cheap insurance — run it whenever the decode path changes at all.

## The open question the literature can't answer for us (re: lever 5)

Speculative decoding is a **single-stream latency** optimization; the 2–3×
figures are batch-1 numbers. **We run batch 24 — a throughput regime** where
launch overhead is already amortized across 24 sequences, and continuous
batching (1) fills idle slots that speculation would instead fill with draft
compute. Speculation's win here may be small or negative. **Measure on our
workload; do not assume.** Honest-negative candidate.

## Benchmark protocol (how "faster" is decided)

Single harness, same seed corpus, same batch budget, report **accepted tok/s**
(not raw tok/s — rejects don't count) plus peak VRAM:
1. Baseline (current path) — the number to beat.
2. Each lever in isolation, then stacked (1+2, 1+2+3, …).
3. For speculation: sweep draft depth (`exit_at_step` 1 vs 2) × block size K
   (2/4/8); report acceptance rate — low acceptance means the 1-loop draft is
   too weak.
Adopt a lever only if it **beats baseline on accepted tok/s AND passes the gate.**

## Build order

1. **On-XPU sampling** (3) — small, safe, self-contained warm-up; immediate.
2. **KV-cache preallocation** (2) — unblocks bigger batch AND is the substrate
   for continuous batching; do the cache-overridability spike first.
3. **Continuous batching** (1) — the big safe win, on top of 2.
4. Benchmark 1+2+3 stacked vs baseline → this is the likely production config.
5. **Speculative decode** (5) — prototype + gate + benchmark head-to-head vs the
   1+2+3 stack. Adopt only if it adds on top.
6. **torch.compile** (4) — last, opportunistic +10%.

All GPU work waits for the card to free after the current v2 harvest; the
logic/cache-spike/prototype work does not.

## Non-goals

- No change to the sampling distribution, ever, without the gate proving it.
- Nothing launched into an unattended overnight harvest until benchmarked + gated.
- No corpus-quality-for-speed trades (7, 8) without an explicit quality benchmark.
