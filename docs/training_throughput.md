# Training throughput — where the step time actually goes

**Status 2026-07-30.** Single Max 1100 (48GB), dual-boot duty cycle, no second
rig. The 1550 OAM card is bought but not present and needs parts + testing, so
everything here is about extracting more from the card on hand.

> **The one-line lesson.** Three separate predictions about this loop's cost
> structure were wrong before anything was measured: λ's mechanism, "the teacher
> dominates", and "skip `--rollout-reuse`". Every correction came from
> `--profile-steps`, not from reasoning harder. **Profile before arguing about
> this loop.**

---

## The measurement

`training/distill.py --profile-steps N` runs `--profile-warmup` steps (default 5,
discarding `torch.compile`, allocator growth and SYCL kernel-cache misses),
profiles N, prints ms/step + share per region, and **exits without saving**.

Device syncs wrap each region — kernel launches are async, so timing without a
sync measures enqueue, not execution. That suppresses pipelining, so **a profiled
run's tok/s is pessimistic; read the SHARES.** For real throughput, run without
the flag and use wall-clock (checkpoint timestamps).

## Baseline: production config @64,000, reuse=2

| region | ms/step | share |
|---|---|---|
| rollout | 7,747.6 | **72.2%** |
| teacher_fwd | 2,280.1 | 21.2% |
| backward | 479.6 | 4.5% |
| student_fwd | 182.9 | 1.7% |
| data / optimizer | 47.7 | 0.4% |

**Rollout generation dominates, not the teacher.** The teacher forward runs on
every micro-step regardless of λ (`distill.py`, outside the on-policy branch),
Ouro-2.6B against a 278M student on one card — which *looked* decisive and is
only 21%.

### Why the rollout is so expensive

`rollout_batch=32, micro_batch=8, reuse=2` → `reuse * (32/8) = 8` draws per fill
(`mythouro/rollout.py`). At λ=0.7 with `grad_accum=2` that's 1.4 on-policy
micro-steps/step, so the buffer refills every **~5.7 steps**. Total rollout time
÷ refill period ≈ **41 seconds per wide generation** of 32×64 = 2,048 tokens —
about **50 tok/s**. It is slow because generation runs `use_kv_cache=False` on
purpose (the ACT early-exit KL finding, probe tracker 2026-07-16), so every
decode step re-runs the whole sequence: O(L²).

### This finally explains the λ null result

The λ sweep cut on-policy steps 0.7 → 0.4 and moved throughput ~4% against a
predicted 45%. λ changes *which input* the teacher sees, never *whether it runs*.
It does change refill frequency — so λ should have helped — but reuse is the
knob that touches refills directly and without spending on-policy dose.

---

## The reuse ladder (the lever)

Raising `--rollout-reuse` cuts refill frequency. Measured, production config,
25 profiled steps each:

| reuse | ms/step | tok/s | speedup | rollout share | teacher share |
|---|---|---|---|---|---|
| 2 | 11,083.4 | 1,478 | — | 69.8% | 23.7% |
| 4 | 6,370.6 | 2,572 | 1.74x | 48.4% | 40.8% |
| 8 | 4,900.4 | 3,343 | **2.26x** | 31.2% | **54.2%** |

*(post-SDPA figures; pre-SDPA were 10,737.9 / 6,478.3 / 4,949.7 — see below)*

**Confirmed in live training**, wall-clock from checkpoint timestamps, reuse=8:
**4.81 s/step ≈ 3,400 tok/s = 2.30x**. Production slightly beats the profiled
number, as expected once the per-region syncs are gone. Predictions from the cost
model landed within 0.5% at reuse=8 — the model is now trustworthy.

In calendar terms: **91M tok/day → ~209M tok/day** at the same duty cycle. The
pour to 2B drops from ~22 days of card time to under 10.

### The bottleneck ROTATES — plan the sequence, not the fix

At reuse=8 the teacher is **54.2%**, so the next lever is the one the reuse=2
profile said to abandon:

1. **`--rollout-reuse 8`** — 2.3x. *Gated on quality, see below.*
2. **Top-K teacher-logit cache for the offline path.** Offline steps feed *fixed
   corpus text* that recycles completely every ~18h, and its teacher logits are
   recomputed identically every pass. ~512 bytes/token → the 13.6M-token teacher
   corpus is ~7GB on disk. Not built.
3. **λ, which only then becomes a real lever** — once offline steps stop paying
   teacher cost, lowering λ finally buys throughput the way it never did.

### The cost is staleness, and it is NOT yet validated

reuse=8 serves 32 draws per fill ≈ **23 optimizer steps of rollout drift** at
λ=0.7. Inside `max_age_steps=50`, so nothing trips — but 4x the staleness the
project has actually run. On-policy dose is what broke the collapse, so this
gates on a probe, not a throughput argument. A/B: `run_reuse8_ab.sh`, one arm
(64,000→66,000, reuse=8) against the existing reuse=2 control in
`checkpoints_lambda07`, same λ, same data, one flag different.

---

## XPU SDPA: real kernel win, zero end-to-end

Re-enabled for GQA (commit b83d1dc) after re-probing on torch 2.13.0+xpu —
8–14x on the attention kernel, numerically gated at 0.00200 nats next-token KL.
Details and the MLA carve-out: `docs/max1100_field_notes.md` workaround #1.

**Step-time effect, same config before and after:**

| reuse | pre-SDPA | post-SDPA | change |
|---|---|---|---|
| 2 | 10,737.9 | 11,083.4 | +3.2% |
| 4 | 6,478.3 | 6,370.6 | −1.7% |
| 8 | 4,949.7 | 4,900.4 | −1.0% |

All inside run-to-run noise; the rollout region itself moved 0.1%. **Attention is
not where this model spends its seconds** — 48 MoE experts run 4x per forward
dominate, and rollout decodes at sequence lengths of 16–80 where attention is
nearly free. Kept because it is free, safe, and pays off if seq-len grows or the
rollout KV cache is ever fixed. FlashAttention-2 would land in the same place and
is unavailable on this hardware regardless.

---

## Not the bottleneck (measured, stop proposing these)

- **Teacher forward at reuse=2** — 21%. Becomes the target only *after* reuse.
- **λ as a throughput knob** — ~4%, until the teacher cache lands.
- **Attention kernels** — ~1% end-to-end despite 8–14x on the kernel.
- **Data loading** (0.3–0.7%) and **optimizer** (0.1–0.3%) — noise.
- **Anti-repetition training** — separate axis; ruled out for *capability* by the
  code eval, see `generation_probe_tracker.md` 2026-07-30.

## Untested throughput ideas, cheapest first

- `--micro-batch 16 --grad-accum 1` — identical tokens/step (16,384) and
  mathematically identical optimization, half the kernel launches. Likely
  memory headroom at 48GB / seq-len 1024. One run to check.
- `--rollout-len` / `--rollout-batch` — change the O(L²) generation cost
  directly, but both alter the on-policy signal, so they are quality decisions.
- Fixing the rollout KV cache (needs a KL equivalence gate like the teacher's)
  would attack the 41s/generation at its root, and would also make GQA's 4x
  smaller KV cache actually pay off in training instead of only at inference.
