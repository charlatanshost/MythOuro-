# On-policy distillation (GKD / MiniLLM) — implementation plan

**Status:** ✅✅ **COLLAPSE BROKEN DOMAIN-WIDE, 2026-06-28** (step 6906, ~231 on-policy steps).
A 6-seed probe shows α=0.0 `top_share` low on **every** seed (0.06–0.31) — **no hard attractor
left anywhere**, not just prose. The exposure-bias blocker that stalled the project for months
is **cured**. Regime is now "varied but incoherent word-salad" = a *normal undertrained small
model*, so the remaining gap is **coherence/capability = tokens + scale** (the lever tokens
actually move). Capability is present at α≥0.5 (diabetes α=0.7 listed the correct symptoms;
fibonacci wrote real code). **Next = pour tokens on the un-collapsed base** (throughput → the
Max 1100). Also: single-sample probes are high-variance (a noisy bacterial draw briefly read
`the the the`); probe now multi-samples (`--samples`). Full verdict: generation_probe_tracker.md
2026-06-28.

**Update 2026-06-30 — α-anneal begins (step 7242, ~218 steps past the 7024 probe).** Pouring
tokens at *fixed* α=0.6 (7024→7242) left α=0.0 **flat** — still varied-but-incoherent salad, no
coherence jump — and the training loss **plateaued** (~1.5 soft / ~0.85 total over 190 steps; an
earlier "loss dropping" read was noise off a lucky 7030 sample). BUT capability is clearly
**present** at α≥0.5: bacterial α=0.7 gave a correct antibiotic/antifungal/antiviral/antiparasitic
taxonomy; diabetes α=0.7 "increased thirst and urination"; ibuprofen "pain, fever and inflammation"
+ real brand names (Advil/Motrin). **Diagnosis:** capability present but **not internalized into the
student's own (α=0.0) trajectory** — the exposure-bias gap, which fixed α=0.6 isn't closing (60%
teacher-driven rollouts → the student rarely has to recover from its *own* errors). **Decision:**
begin the Phase-4 **α-anneal** the design always called for ("anneal α down as the student
de-collapses") — un-collapse is achieved, so step training α **0.6 → 0.5** (tonight's run, resuming
from 7242). **Hypothesis:** α=0.5 forces more student-driven rollouts → α=0.0 coherence should rise
*faster* than the flat fixed-0.6 grind. **Watch:** (a) loss may tick *up* — expected/good (harder
own-rollouts = real exposure-bias training, not regression); (b) the next α=0.0 probe is the read,
*not* the loss; (c) **re-collapse on fragile seeds** — bacterial-LaTeX hit top_share 0.47 on one
α=0.0 sample, math/code drift to number-salad; if those climb back toward the attractor, α dropped
too fast → back off to 0.55. Full 7242 probe: generation_probe_tracker.md 2026-06-30.

**Result 2026-07-01 (α=0.5 verdict, step 7458, ~216 steps).** Anneal is **SAFE** — no re-collapse,
the fragile bacterial/LaTeX seed *de-fragilized* (α=0.0 top_share 0.47→0.18). Metric **moved right**
(mean α=0.0 top_share 0.18→0.12, 4/6 seeds down) — more than the flat-0.6 grind's flat. **But α=0.0
text is still incoherent salad — no coherence jump**; lower top_share alone is ambiguous
(less-repetitive vs more-random). Also: my "loss will tick up at α=0.5" call was **wrong** (it
*fell* — easier student-native rollouts lower the rev-KL by construction, so loss isn't comparable
across α). Anneal's payoff = *safe + distribution-nudged*, not a breakthrough. **Decision: HOLD
α=0.5, grind tokens** (~1,000+ across sessions; the Max makes it cheap), re-probe on a bigger dose
before stepping α further — **bottleneck is token volume, not α.** Full read:
generation_probe_tracker.md 2026-07-01.

**✅✅✅ RESULT 2026-07-06 — THE DE-TAX WORKED (step 8668, ~1,210 de-taxed steps).** Full-strength
on-policy (A1) + EOS (A2) over the weekend produced a **second regime shift**: α=0.0 went from
varied-salad → **rambling-grammatical English** (weather/bacterial = connected on-topic clauses; fib
= real Python syntax). Clearest α=0.0 movement of the project — moved *far* more than the flat taxed
run, confirming the fixes paid off. **Methodological catch:** top_share *inverted* (0.12→0.16, fluent
English repeats function words); the TEXT is the read, `distinct1` the honest metric. **Boundary:**
fluent-not-meaningful (grammar+topic, no correct reasoning yet) — still ~120M tokens, undertrained.
Laggards: ibuprofen α=0.0 still salad, fib α=0.0 high-variance. **Decision: α=0.5 is working → HOLD
it, pour TOKENS (the Max)** to push fluency→meaning; 0.45 anneal optional/secondary. Full read:
generation_probe_tracker.md 2026-07-06.

**⚑ LOG MARKER — Batch A fixes landed 2026-07-01 (commit `ca071e8`); weekend run resumes from
7458 with them ACTIVE.** Loss-surface shift, so the curve *will* jog at 7458 — expected, not
regression: (a) **A1/P2** — on-policy soft was backpropping 0.5× the intended gradient the whole
run (`args.alpha=0.5`); now full-strength, so on-policy `distill_total` roughly **doubles** (raw
`soft` metric is continuous — logged pre-scale); (b) **A2/EOS** — `<|endoftext|>` now separates
packed docs (model can finally learn to stop), a small input-distribution shift → `soft` may move
slightly, real signal; (c) A3 (on-policy unc skip → `unc`=0 on op-steps) and A4 (biased
load-balance) also live. Config unchanged otherwise: λ=0.7, **α=0.5 held** (grinding tokens per the
anneal decision). P0.6 LoRA fold (`73a0f73` tooling) **deferred to an attended boundary before the
Max** — B's gradient wakes ~50×, don't run it into an unattended weekend.

**Prior (2026-06-27, partial):** first run (6675→6771, ~96 steps, λ=0.5) un-collapsed the α=0.0
prose seed (top_share 0.45→0.14) — first movement on the blocker ever. Read as "dose-limited,
prose-first"; the 06-28 higher-dose 6-seed probe showed it generalized domain-wide (and that the
"medical still collapsed" read was partly single-sample noise). Throughput problem (5.8 min/step)
→ the Max-1100 case (batched rollouts).

Flags + rollout engine (`generate_rollout`) + on-policy step + the α-probe tool all landed,
**default-off** (λ=0 → no behaviour change).
**α-probe done (2026-06-25) — read from the RAW text, not the eyeball:**
- α=0.0 **collapsed** (`the the the`, top_share 0.89) — collapse shows under *sampling*
  too, not just greedy (earlier "greedy artifact" claim retracted).
- α=0.25 **also collapsed** (`the the a a a`, top_share 0.76) — the 0.25 mix is too weak
  to break the 0.99 attractor.
- α=0.5 floor of real content but degrades into loops (`bacteria bacteria`, `is is is`).
- α=0.7 first *mostly* coherent (a clean weather narrative); code still degenerate.

**Picked α=0.6** (was wrongly 0.25): need ≈0.6–0.7 for workable rollouts. Trade-off: high
α = more teacher-driven (less "pure" on-policy), BUT rollouts still carry the student's
signature errors (`the the`, `if if if`) in coherent context — rev-KL to the teacher on
those is exactly what suppresses repetition. **Anneal α down** as the student de-collapses
(Phase 4). Next: first overnight run from 6675.

## Why
Every offline divergence we swept (fwd-KL, rev-KL, JSD — all on a *stable*, well-
calibrated footing) **mode-collapses with tokens** into a sharp `is is is` attractor.
The cause is **exposure bias**: offline distillation only ever sees *teacher-forced*
sequences, so the student never learns to recover from its *own* trajectories. It is
**decoupled from every formal metric** (PPL 1.759, ECE 0.0152, stability, reps all
healthy at the collapsed checkpoint 6675). More offline tokens — continued OR fresh —
only sharpen the attractor. The cure is a **different objective**, not more data:
train the student on **its own rollouts** under teacher correction.

This is the one lever the whole week's evidence points at. It is also teacher-agnostic
(base vs -Thinking distill nearly identically on plain corpus), so the teacher question
is correctly parked until *here*, where the teacher actually generates/scores.

## The key structural insight
An on-policy step is the **same loss computation** as today's offline step, just on
**student-generated** sequences instead of corpus ones. So `teacher_logits(...)` and
`distillation_loss(..., divergence="rev_kl")` are **reused unchanged** — the only new
piece is *generating the rollout*. `mythouro/inference.py` already has the primitives
(`_sample` = top-k/temp; the kv-cache incremental decode path).

```
offline (now):   x ← corpus;            t = teacher_logits(x);  s = student(x);  loss = div(s,t)
on-policy (new): x ← generate_rollout(); t = teacher_logits(x);  s = student(x);  loss = div(s,t)
                                          ^^^^^^^^^^^^^^^^^^^^^^ the only new code
```

## Resolved decisions
- **Warm-start from checkpoint 6675** (stable, ECE 0.0152). Reuses 109M tokens; the
  teacher-mix is the un-collapse mechanism. *Not* a fresh restart.
- **Objective:** rev-KL (mode-seeking) on the rollouts — the MiniLLM setting.
- **Default off:** `--onpolicy-lambda 0.0` preserves current behaviour exactly.

## Flags (landed)
- `--onpolicy-lambda` — fraction of steps trained on rollouts (0=offline, 1=on-policy).
- `--teacher-mix-alpha` — sample rollouts from `α·teacher + (1-α)·student` (un-collapse
  lever; default 0.25).
- `--rollout-len` — tokens per rollout (default 96; keep short — recurrent decode is slow).
- `--onpolicy-temp` — rollout sampling temperature.

## Perf realities (design around these)
1. **Recurrent decode is slow** — each rollout token = K loops. On-policy steps are far
   slower than offline. → short rollouts (64–128), small batch.
2. **Teacher-mix needs teacher logits *every* step.** `teacher_logits` runs no-cache
   (Ouro's `cache_position` bug forces `use_cache=False`), so naive per-step teacher
   calls during generation are O(L²). Phase-2 problem; mitigate with short L first, a
   teacher kv-cache path later.
3. **The collapsed-rollout risk.** A 0.99-sharp student sampled freely emits `is is is` —
   useless rollouts. The mix `α·teacher + 0.8·student` may not break a 0.99 spike at
   α=0.25; **may need α≈0.5 early, annealed down** as the student de-collapses. Watch the
   first few hundred steps: rollouts should get less repetitive. This is the make-or-break
   knob — tune it empirically.

## Build phases (incremental, test each)
1. **✅ Flags** — landed, default-off, no behaviour change.
2. **✅ Rollout engine** — `generate_rollout(...)` in `mythouro/training_utils.py`:
   step-wise, sample from the α-mix, `@torch.no_grad`, eval(), returns token ids.
   (Simple O(L²) version — full forward each step; optimise in phase 5.)
3. **✅ On-policy step** — `training/distill.py`: with prob `λ`, the student continues a
   short real-text seed (`generate_rollout`), then the *existing* `teacher_logits` +
   `distillation_loss(targets=None)` (pure soft rev-KL — no hard CE on sampled tokens).
   Backprop unchanged. Log shows `op N/grad_accum` (on-policy micro-steps that step) so
   you *see* it firing.
4. **🔄 Schedule + polish** — λ-ramp (start small), length-norm; **α-anneal ACTIVE (2026-06-30):
   manual step 0.6 → 0.5, probe-gated. Lower further if α=0.0 improves + fragile seeds hold.**
5. **✅ Optimize (2026-07-14, branch `feat/batched-cached-rollouts`)** — teacher kv-cache,
   larger batch/rollouts, throughput. Landed: (a) cached student decode in
   `generate_rollout` (mirrors `MythOuro.generate`; `use_kv_cache=False` = exact legacy
   path); (b) cached teacher via model-built cache (Ouro's `UniversalTransformerCache`)
   behind a **KL-based startup validation gate** that falls back to full recompute on any
   mismatch — never silently trains against wrong teacher logits (bf16 max-abs logit noise
   reaches 0.34 on a *correct* cache, so the gate compares softmax KL ~1e-3 nats + greedy
   tokens instead); (c) `RolloutBuffer` (`mythouro/rollout.py`) — one wide generate call
   serves micro-batch slices with reuse budget + staleness cap; flags `--rollout-batch/
   --rollout-reuse/--rollout-max-age-steps/--rollout-legacy`. **Measured (Max 1100, bf16,
   teacher incl., rollout 96):** old inline path 23 tok/s → cached @ batch 32 = 134 tok/s
   (5.8×), ×2 reuse ≈ **11.7× effective on-policy dose per decode-second**. Remaining
   floor is PVC per-step kernel-launch latency → follow-up ticket: `torch.compile` on the
   decode step (default mode only; max-autotune is a measured regression on PVC).

## How to run
**Step 1 — α-probe (no training; pick α).** Generates rollouts from 6675 at each α and
prints repetition stats. Look for `top_share` falling / `distinct` rising as α rises:
```
python -m tools.onpolicy_rollout_probe --ckpt-dir checkpoints_revkl_stable \
    --student-device cuda:0 --teacher-device cuda:2 \
    --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code
```

**Step 2 — warm-start the on-policy run from 6675** (one-time copy into a fresh dir so
the pure-rev-KL lineage is preserved), then launch with the α the probe picked:
```
mkdir checkpoints_onpolicy
cp checkpoints_revkl_stable/step_0006675.pt checkpoints_onpolicy/
python -m training.distill --student-variant mythouro_distill_tiny \
    --student-device cuda:0 --teacher-device cuda:2 \
    --teacher-id ByteDance/Ouro-2.6B-Thinking \
    --seq-len 1024 --micro-batch 1 --grad-accum 16 \
    --total-steps 12000 --warmup-steps 500 --lr 1e-4 --depth-reg-coeff 0.3 \
    --divergence rev_kl --use-sandwich-norm --use-depth-aware-init \
    --onpolicy-lambda 0.7 --teacher-mix-alpha 0.6 --rollout-len 64 \
    --ckpt-every-mins 15 --num-workers 0 --trust-remote-code \
    --ckpt-dir checkpoints_onpolicy
```
**Cross-GPU (teacher on `cuda:2`) — single-card doesn't fit.** Tried it (best for PCIe
avoidance) but the 5.2 GB teacher + the student's training peak **OOMs a 12 GB 5070** even
at `--seq-len 512` with `expandable_segments` — the teacher just doesn't cohabit. So
teacher on `cuda:2`: the proven layout 6675 trained under. **The PCIe worry doesn't carry
over to on-policy:** the 0.3k tok/s was *offline*, where each transfer was a full
`(B,1024,V)` ≈ 100 MB tensor with only one forward of compute. On-policy slices to the
**last-token logits (~100 KB) before transfer**, with **128 forwards** of compute per
rollout — transfer is dwarfed by compute, so the cross-GPU penalty should be ~negligible.
**Measure `tok/s`**; only if genuinely slow do we need Phase-5 (teacher kv-cache) or a
quantized/smaller-footprint teacher to enable single-card.

**Throughput reality:** still slow — sequential recurrent decode + per-token teacher
forward (no kv-cache yet) is the floor. Expect *tens* of slow steps overnight, not
thousands — fine for a first signal. **`--ckpt-every-mins 15`** saves every 15 min of
wall-clock (step-based `--ckpt-every` won't fire at this speed → power-cut net; keep_last
prunes to 3). Watch `op N/16` (firing) + the morning α=0.0 re-probe (un-collapse).

## Success signal
The `is is is` collapse breaks: free-generation probes at matched depth produce varied,
on-topic continuations (not the sharp attractor), while PPL/ECE stay healthy. That is the
*generation-coherence* metric the whole project has been blocked on — and the first time
any lever has touched it, because it's the first that attacks exposure bias directly.

**Concrete morning check (clean A/B):** re-run the probe pointed at the on-policy
checkpoint and read the **α=0.0** rows (pure student, *no* teacher-mix). At the pre-train
6675 baseline those are **collapsed** (`the the the`, top_share ~0.5–0.9, distinct1
~0.06–0.15). If on-policy worked, α=0.0 should improve — top_share *down*, distinct *up*,
text less repetitive — i.e. the *student's own unaided* distribution got better, which is
the whole point. (α>0 rows are teacher-assisted, so they don't isolate the student's gain.)
```
python -m tools.onpolicy_rollout_probe --ckpt-dir checkpoints_onpolicy \
    --student-device cuda:0 --teacher-device cuda:2 \
    --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code
```
