# On-policy distillation (GKD / MiniLLM) — implementation plan

**Status:** in progress (started 2026-06-24). Flags landed in `training/distill.py`;
rollout engine + on-policy step next.

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
2. **Rollout engine** — `generate_rollout(student, teacher, prompt, α, temp, n_loops, L)`:
   step-wise, sample from the α-mix, `@torch.no_grad`, return token ids. Student uses its
   kv-cache path; teacher recompute (short L). **Unit-test it in isolation** (does it
   produce non-degenerate ids from 6675 at α≈0.5?) before wiring into training.
3. **On-policy step** — with prob `λ`, swap the corpus batch for a rollout, then the
   existing forward + `distillation_loss`. Backprop unchanged. Log a `rollout/rep` metric
   (n-gram repetition of the rollouts) so we *see* the un-collapse happen.
4. **Schedule + polish** — λ-ramp (start small), length-norm, α-anneal.
5. **Optimize** — teacher kv-cache, larger batch/rollouts, throughput.

## Success signal
The `is is is` collapse breaks: free-generation probes at matched depth produce varied,
on-topic continuations (not the sharp attractor), while PPL/ECE stay healthy. That is the
*generation-coherence* metric the whole project has been blocked on — and the first time
any lever has touched it, because it's the first that attacks exposure bias directly.
