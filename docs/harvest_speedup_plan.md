# Harvest throughput speedups — design & benchmark plan

**Status: DESIGN, 2026-07-21.** Goal: more teacher-corpus tokens per card-hour.
Baseline (measured): ~55 accepted tok/s, batch 24, 768-tok continuations,
launch-bound decode (docs/teacher_corpus_plan.md). The A/B verdict (tracker
2026-07-21) proved teacher data is the lever moving the plateau, so faster
harvest = faster iteration on the one intervention that's working.

## Feasibility findings (from reading modeling_ouro.py + training_utils.py)

- **Ouro exposes `exit_at_step`** in `OuroForCausalLM.forward` (native early
  exit): `exit_at_step=1` → logits from 1 UT loop (cheap draft), `None` → full
  4 loops (verify). So self-speculation needs NO model surgery — the draft is
  the same weights at reduced depth.
- **The hard part is the KV cache.** Ouro builds a `UniversalTransformerCache`
  with per-UT-loop slots (`total_layers * total_ut_steps`; a plain DynamicCache
  silently corrupts — 0.30 divergence, caught 2026-07-14). Speculative decoding
  needs **cache rollback on rejected tokens**, and rolling back a UT cache
  across all 4 loop sub-slots is the bug-prone, correctness-critical step.
- **The equivalence gate already has a template:** `_validate_teacher_cache`
  (KL(uncached‖cached) < 5e-2 nats, argmax match) is exactly the shape of the
  gate a speculative path needs.

## ⚠ The open question the literature doesn't answer for us

Speculative decoding is a **single-stream latency** optimization. **We run
batch 24 — a throughput regime** where launch overhead is already amortized.
The 2–3× figures are batch-1 numbers. In our batched case the win may be small,
and draft compute + rollback overhead could cancel it. **Must be measured on
our workload, not assumed.** This is an honest-negative candidate.

## Plan (benchmark-first, correctness-gated)

1. **Prototype** `--speculative` in gen_teacher_corpus (default off; old path is
   the reference). Draft K tokens at `exit_at_step=1`, verify in one 4-loop
   parallel forward, standard speculative accept rule (sampling-correct), roll
   back both caches to the accepted length.
2. **Distribution-equivalence gate** (blocking): at startup, generate from a
   fixed prompt both ways (speculative vs full-4-loop) and assert token-dist KL
   under tolerance, mirroring `_validate_teacher_cache`. Refuse to run if it
   diverges — a silent skew here poisons the corpus (cf. the 07-16 cached-decode
   incident: the exact failure mode to avoid).
3. **Benchmark** accepted tok/s speculative vs baseline at batch 24. Also try
   the acceptance rate at K=2/4/8 (low acceptance = draft too weak at 1 loop →
   try 2-loop draft).
4. **Decision:** adopt only if it BEATS baseline AND passes the gate. If not,
   log the honest negative and pursue a throughput-native lever instead:
   - **KV-cache preallocation** — the cat-doubling caps batch at 24; a
     preallocated cache could allow batch 32–40 at 768 (~1.4–1.7×), no
     distribution change.
   - **torch.compile the decode step** — +10% measured on training; launch-bound
     decode may gain more.

## Non-goals

- No change to the sampling distribution, ever, without the gate proving it.
- Not launched into an unattended overnight harvest until benchmarked + gated.
