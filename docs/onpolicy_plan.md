# On-policy distillation (GKD / MiniLLM) — implementation plan

**Status:** in progress (started 2026-06-24). Flags + rollout engine (`generate_rollout`)
+ on-policy step + the α-probe tool all landed, **default-off** (λ=0 → no behaviour change).
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
4. **⏭ Schedule + polish** — λ-ramp (start small), length-norm, α-anneal.
5. **⏭ Optimize** — teacher kv-cache, larger batch/rollouts, throughput.

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
    --onpolicy-lambda 0.5 --teacher-mix-alpha 0.6 --rollout-len 64 \
    --num-workers 0 --trust-remote-code --ckpt-dir checkpoints_onpolicy
```
**Throughput reality:** on-policy micro-steps are ~100× an offline one (96/64 sequential
recurrent forwards × student+teacher). λ=0.5 ⇒ ~half the 16 micro-steps generate ⇒ steps
~50× slower than offline. For the first overnight run that's fine — even a few hundred
on-policy steps should *visibly* move `rollout` repetition. Tune for speed: lower `λ`,
shorter `--rollout-len`, or smaller `--grad-accum`. Watch the `op N/16` field to confirm
on-policy steps are firing and `loss`/rollouts to confirm un-collapse.

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
