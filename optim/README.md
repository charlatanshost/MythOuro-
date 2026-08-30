# Throughput optimisations A and B — teacher-forward elimination

## The measurement that motivates both

`_StepProfiler` (2026-08-28, 48-expert config, `--profile-steps`):

```
region            ms/step   share   calls/step
teacher_fwd        5077.8   78.4%      4.0
backward            901.6   13.9%      4.0
student_fwd         404.4    6.2%      4.0
data                 51.5    0.8%      4.0
optimizer            38.9    0.6%      1.0
rollout               0.1    0.0%      2.0
```

**The 2.6B teacher forward is 78.4% of a step.** `rollout` is already 0.0%, so
`--rollout-len`, `--rollout-batch` and `--rollout-reuse` are NOT throughput
levers — which is also why the λ sweep moved throughput ~4% against a predicted
45%. On-policy micro-steps are **72.6%** of the total (measured from the
`op N/4` log lines over the 2026-08-29 leg).

## ⚠️ CORRECTION 2026-08-30 — A IS WORTH ~5%, NOT 2x. B IS THE LEVER.

**Measured on the first A leg: 11.98 s/step (2700→2750, contains a checkpoint
save) against an 11.46 s/step pre-A baseline. No usable speedup.**

The error was mine and it is a specific one: I read the profiler's
`teacher_fwd ... 4.0 calls/step` together with the `op 2/4` log lines and
concluded on-policy micro-steps were **72.6% of the teacher's work**. They are
72.6% of the teacher's **CALLS**. The two are wildly different here, because the
two paths run at completely different sequence lengths:

```
on-policy x_in = seed_len(16) + rollout_len(64) - 1 =   79 tokens
offline   x_in = seq_len(1024)                    - 1 = 1023 tokens
```

Teacher tokens per optimizer step: on-policy **632 (7.2%)**, offline
**8,184 (92.8%)**. So A can touch at most 5.6% of step time and saves 7/8 of
that = **4.9%** — 11.5 → 10.93 s/step, which is invisible under checkpoint noise
and is exactly what was measured.

**A is still correct, just small.** The loss gate passed cleanly (0.7485 at step
2700 against 0.9013 pre-A, continuous — mispaired logits would have spiked it to
5-10+). It is a real ~5% win with no downside; keep it.

**B is where the 3x lives**, and for the same reason A is not: B removes the
teacher from the **offline** path, which is 92.8% of teacher token-work.
**11.5 → ~3.13 s/step.** Every "+A" row below overstates A and understates B's
relative importance; the s/step figures for A+B combined remain roughly right
because they were dominated by the offline term all along.

**The general lesson, worth carrying:** a profiler's `calls/step` is not
`work/step` when call shapes differ. Multiply by the actual tensor sizes before
predicting a saving.

---

## A — teacher-logit reuse cache (on-policy). NUMERICALLY EQUIVALENT.

`RolloutBuffer` sets `_draws_left = reuse * (rollout_batch // micro_batch)`
= `8 * (8//4)` = **16 draws**, and `draw()` wraps a cursor over an 8-row store in
4-row slices — so **each slice is served 8 times, byte-identical**. distill.py
nevertheless called `teacher_logits(teacher_fwd, x_in)` on every micro-step.
The teacher was being asked the same question 8 times.

A runs the teacher **once per fill**, over the wide rollout, and slices the
cached logits alongside the tokens.

| | s/step | steps per 10h night |
|---|---|---|
| current | 11.5 | 3,130 |
| **+A** | **5.77** | **6,236  (1.99x)** |

Saving = `0.784 x 0.726 x 7/8` = **49.8%** of wall clock.

**Caveat, stated honestly:** not bit-identical. The teacher now sees 8 rows at
once instead of two 4-row batches, so reduction order differs in the last bits.
Numerically equivalent, not bitwise — which is what the A/B below is for.

**Memory:** one `(rollout_batch, L-1, vocab)` bf16 tensor per fill: ~88 MB at
L=112, ~805 MB at L=1024.

## B — precomputed top-K teacher logits (offline). AN APPROXIMATION.

The teacher answers the offline corpus **once, offline**, and training reads it
back. Storage is top-K logits plus one number for everything else:
`tail_lse = logsumexp(tail_logits / T)`.

| | s/step | steps per 10h night |
|---|---|---|
| +A | 5.77 | 6,236 |
| **+A+B** | **3.30** | **10,902  (3.48x total)** |

**B is NOT equivalent, and this is the whole reason it is a separate leg.** The
divergence is computed on a coarsened event space (K symbols + one tail bucket)
for both teacher and student. By the data-processing inequality that is a
**lower bound** on the true KL, so B trains against a slightly softened
objective. The gap shrinks as K rises. Verified in `test_optB.py`: at K = V the
sparse loss reproduces `distillation_loss` exactly (fwd_kl / rev_kl / jsd,
T=1 and 2, with and without targets), and it is monotone in K.

**Storage** (`4K + 4` bytes/token; vocab 49152 fits in uint16):

| K | B/token | 6,000-step leg | full 67,749-step pour |
|---|---|---|---|
| 16 | 68 | 1.8 GB | 20.7 GB |
| **32** | **132** | **3.6 GB** | **40.1 GB** |
| 64 | 260 | 7.0 GB | 79.1 GB |

**Pick K from measurement, not taste.** `precompute_teacher_logits.py
--report-only` prints the mean top-K probability mass captured on a sample.
Below ~99% the lumped tail is carrying real signal and K should go up. (The
toy-vocab numbers in `test_optB.py` look much worse than reality — random
logits are far flatter than a real LM's.)

**Two further costs, both real:**
1. **The precompute is a GPU job.** It runs the teacher over the whole corpus
   once. Forward-only at large batch, so much faster than in-training, but it is
   not free — budget it against what it saves.
2. **It FIXES the offline corpus.** Streaming fineweb-edu is effectively
   infinite; a cache is finite. `TeacherLogitCache.epochs_for()` reports the
   re-read dose and the trainer logs a WARNING past 1.35 epochs — the dose that
   measurably cost 6.2pp of code L3+ in the chat-mix post-mortem.

## Order of work

**Nothing here is applied to the repo.** The patches are scripts; apply them
when no trainer is running.

### Leg 1 — A
```bash
bash optim/apply.sh A            # refuses while a trainer is live
python3 -m pytest tests/ -q -k rollout
bash run_grown48.sh              # unchanged; A is transparent to it
```
Gate: `s/step` should fall ~11.5 → ~5.8, and the loss curve should track the
pre-A leg. If loss diverges, A is wrong — revert with `optim/apply.sh revert`.

### Leg 2 — B, compared against leg 1
```bash
bash optim/apply.sh B                       # A must already be applied
python -u -m tools.precompute_teacher_logits --report-only ...   # choose K
python -u -m tools.precompute_teacher_logits ...                 # build cache
bash run_grown48.sh --teacher-logit-cache data_teacher_logits_k32
```
Gate: `s/step` ~5.8 → ~3.3. **The comparison that matters is quality, not
speed** — B changes the objective, A does not. Compare the B leg's L3+/L4/prose
readout against the A leg's at the same step count from the same checkpoint.
If B is even slightly worse, it is not worth 1.75x; A alone is the safe win.

## Files

| file | what |
|---|---|
| `patchA_apply.py` | A: `mythouro/rollout.py` + `training/distill.py` |
| `patchB_apply.py` | B: installs 3 modules, patches `distill.py` |
| `sparse_kd.py` | → `mythouro/sparse_kd.py` — top-K + lumped-tail KD |
| `logit_cache.py` | → `mythouro/logit_cache.py` — cache reader + dose report |
| `precompute_teacher_logits.py` | → `tools/` — builds the cache |
| `test_optA.py`, `test_optA_regression.py` | A correctness + no regression |
| `test_optB.py`, `test_optB_loader.py` | K=V exactness, bound, alignment |
| `apply.sh` | guarded apply / revert |
