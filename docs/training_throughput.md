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

- ~~`--micro-batch 16 --grad-accum 1`~~ — **STALE, do not run (2026-09-12).**
  Written before `exit_pdf` made the loss per-loop. `run_exitpdf.sh:44` measured
  the per-loop distillation loss at **~19.3 GB at micro-batch 8** with all K
  autograd graphs live — the cause of the "NotPresent / PDE / Write" page fault,
  an OOM in disguise. mb2 is ~4.8 GB, so mb16 is ~38 GB before the 4.8 GB teacher
  and ~4.2 GB of weights/optimiser: it cannot fit in 48 GB. The surviving version
  of this idea is **`--micro-batch 4 --grad-accum 4`** (~9.6 GB), same
  tokens/step, 4x fewer launches than mb2 — probed by `run_throughput_probe.sh`.
- `--rollout-len` / `--rollout-batch` — change the O(L²) generation cost
  directly, but both alter the on-policy signal, so they are quality decisions.
- Fixing the rollout KV cache (needs a KL equivalence gate like the teacher's)
  would attack the 41s/generation at its root, and would also make GQA's 4x
  smaller KV cache actually pay off in training instead of only at inference.

---

## 2026-09-12 — step 2 of the rotation landed: the teacher cache is built, and λ is now a real lever

The ladder above said: reuse 8, then the **top-K teacher-logit cache** ("Not
built"), and only then *"λ, which only then becomes a real lever."* Step 2 is
now built and measured. This document predicted the outcome; it is recorded here
rather than anywhere else so the sequence stays in one place.

**OPT-B (`optim/`, top-K=32 + lumped tail) measured end-to-end:** 12.34 → **8.83
s/step, 1.40x**, on the current recipe (24 experts, `micro-batch 2 / grad-accum
8`, `--onpolicy-lambda 0.7`, `rollout-len 64 / batch 8 / reuse 8`). Cache: 4
shards, 6,836 rows, mean top-K mass **99.415%**, 909 MB for 7M tokens.

**Profile with B applied** (5 warmup + 10 steps):

| region | ms/step | share | calls/step |
|---|---|---|---|
| **rollout** | 7,095.6 | **69.7%** | 5.9 |
| backward | 2,130.5 | 20.9% | 8.0 |
| student_fwd | 779.4 | 7.7% | 8.0 |
| teacher_fwd | 109.0 | **1.1%** | 8.2 |
| optimizer | 38.5 | 0.4% | 1.0 |
| data | 25.4 | 0.2% | 8.0 |

Compare the reuse=8 row of the ladder above (rollout 31.2%, teacher 54.2%): the
cache took the teacher from 54.2% to 1.1%, and **rollout is back on top at
69.7%** — the rotation this document described, one full turn later.

### ⚠ The 1.1% teacher share is a region boundary, not the teacher being cheap

`teacher_fwd` wraps only the explicit `teacher_logits(...)` calls. `generate_rollout`
runs the teacher at **every generated token** (`teacher_mix_alpha=0.45`,
`α·softmax(teacher/T) + (1−α)·softmax(student/T)`), and that time is billed to
`rollout`. So an unknown but probably large fraction of the 69.7% is still
teacher work, now hidden rather than removed.

**This is directly measurable and has not been measured:** a profile run with
`--teacher-mix-alpha 0` isolates it, since α=0 skips the teacher in generation
entirely. It is a ~5-minute PROFILE run and it decides which lever is real:

* if `rollout` collapses at α=0 → the teacher still dominates, and α (or a
  cheaper mixing schedule) is the lever, not rollout geometry;
* if `rollout` barely moves → it is genuinely student decode, and the O(L²)
  KV-cache fix listed under "Untested throughput ideas" is the root attack.

⚠ α is not free to lower: it is the un-collapse lever from `docs/onpolicy_plan.md`
that made on-policy viable from a degenerate checkpoint. Lowering it in TRAINING
is a quality decision. Using α=0 in a throw-away PROFILE run is not.

### Consequence for the bigger-student plan

`rollout + backward + student_fwd` = **98.3%** of the step, and all three scale
with student size (the teacher component inside rollout does not, which is
exactly why the α measurement above matters). The 2026-09-10 costing assumed the
teacher was ~78% of a step and would amortise a 2.55x-activated student to
roughly today's step time. That assumption is dead: on this recipe a 2.55x
student lands near **26 s/step ≈ 21.6 h per 3,000-step leg** unless rollout cost
comes down first.

**⇒ Rollout cost is now a prerequisite for scaling the model, not a separate
optimisation.**

---

## 2026-09-12 (evening) — the teacher is 84% of rollout, and α is a SWITCH not a dial

Four PROFILE variants of the production recipe, dense teacher (OPT-B applied but
no cache passed, so every offline micro-step takes the dense path). 5 warmup +
10 profiled steps each, nothing saved.

| variant | ms/step | rollout | backward | student_fwd | teacher_fwd | vs base |
|---|---|---|---|---|---|---|
| A baseline (mb2/ga8, α=0.45) | 12,492 | 54.4% | 16.8% | 6.2% | 22.1% | 1.00x |
| B `--micro-batch 4 --grad-accum 4` | 10,792 | 63.3% | 11.9% | 4.0% | 20.3% | **1.16x** |
| C `--teacher-mix-alpha 0` | **6,751** | **15.7%** | 31.3% | 11.2% | 40.9% | **1.85x** |
| D `--rollout-len 32` | 8,557 | 35.0% | 24.1% | 8.7% | 31.4% | 1.46x |

### The teacher is 84% of rollout generation

In absolute terms the `rollout` region is 6,796 ms at α=0.45 and **1,060 ms at
α=0**. Generation itself — the student decoding 64 tokens with `use_kv_cache=False`
— costs about 1.1 s. The other **5.7 s is the 2.6B teacher**, running once per
generated token for the mix `α·softmax(teacher/T) + (1−α)·softmax(student/T)`.

This resolves the ambiguity in the morning's post-B profile, where `teacher_fwd`
read 1.1% and looked like the teacher had been eliminated. It had not: the cache
removes the teacher from *offline* micro-steps only, and the teacher's real cost
had simply moved into a region that does not carry its name.

### α is a SWITCH, not a dial

`mythouro/training_utils.py`: `use_teacher = teacher is not None and
teacher_mix_alpha > 0.0`. **Any α > 0 pays the full teacher generation cost.**
There is no α=0.2 that buys half the saving — the choice is teacher-mixed
rollouts at 12.5 s/step or pure-student rollouts at 6.8 s/step.

### What to take, and what to gate

* **B (mb4/ga4) is 1.16x and is NOT free — corrected 2026-09-13.** Tokens/step
  is identical (16,384) and the main loss is unaffected, but "mathematically
  identical optimisation" was wrong for an MoE model. `load_balance_loss` is
  `E·Σ_i f_i·P_i` with **both** `f_i` and `P_i` averaged over the micro-batch; a
  product of two batch means is not linear in the batch, so 8 accumulations over
  N=2,048 and 4 over N=4,096 give different auxiliary gradients, and the
  router-bias update cadence changes with them. Cheap, still worth taking, but
  it is a routing change and gets its own gate — routing is what the 24→48
  expert programme showed this model is sensitive to. Memory stays ~9.6 GB against the ~19.3 GB
  that page-faulted at mb8.
* **C (α=0) is 1.85x and is a QUALITY DECISION, not a free win.** α is the
  un-collapse lever from `onpolicy_plan.md`: it "drags a collapsed student's
  rollouts back toward the teacher's support." It was set to 0.45 when the model
  *was* collapsing. It no longer is — code L0 is 2-7% against a 4-31% historical
  band, prose LOOPING is 1/90, halt is 2.88. There is also a principled argument
  that α=0 is *more* correct: on-policy distillation is meant to train on the
  student's own distribution, and teacher-mixing is a crutch. **Neither argument
  is evidence.** Gate it on a leg.
* **D (rollout-len 32) is 1.46x** and likewise alters the on-policy signal. It is
  strictly dominated by C right now — less speedup for a similar class of risk —
  so it is only interesting if C fails its gate.

### Consequence for the token curve

At mb4 + α=0 the step would be ~5.8 s (if the two compose; **unmeasured**, and B
was measured at α=0.45 where rollout still dominated, so do not assume they
multiply). That is the difference between ~10 nights and ~5 for the same curve —
which is why this is worth one gating leg before starting it.

