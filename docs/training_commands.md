# MythOuro Training Commands

Copy-paste-ready command reference. Companion to
[`training_runs.md`](training_runs.md) (which records what the runs *produced*)
— this file is *how to run them*.

GPU map on this rig: **Linux side (CURRENT): `xpu:0` = Intel Max 1100 (48 GB) — teacher AND
student on one card; `cuda:0` = RTX 5070 (12 GB, display + A/B benches).** The 4060/5060 were
removed 2026-07-12 (power connectors freed for the Max) — old `cuda:1`/`cuda:2` commands below
are Windows-era references.

---

## ⭐⭐ XPU / Max 1100 (native Ubuntu) — THE CURRENT ENVIRONMENT (2026-07-14)

**Shell setup — start EVERY session with this** (fresh terminals have no `python`; the three
env vars are load-bearing: SYCL cache skips the brutal first-JIT, the allocator flag cuts
ACT-batch fragmentation, the Triton var fixes the dual-GPU `2 active drivers` crash):
```bash
cd /media/charlatanshost/94C4EE28C4EE0C74/MythOuro-main/MythOuro-main
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_DEFAULT_BACKEND=intel
```

**The main run — on-policy distill, teacher+student on one card, phase-5 buffered rollouts**
(resumes automatically from the latest ckpt in `--ckpt-dir`; expect
`teacher KV-cache validation PASSED` after the teacher loads, then 1.5–3k tok/s):
```bash
python -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --total-steps 12000 --warmup-steps 500 --lr 1e-4 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --onpolicy-lambda 0.7 --teacher-mix-alpha 0.5 --rollout-len 64 \
  --rollout-batch 32 --rollout-reuse 2 \
  --ckpt-every-mins 15 --num-workers 0 --trust-remote-code --log-every 5 \
  --ckpt-dir checkpoints_onpolicy_xpu
```
Micro-batch 8 × accum 2 = the old effective batch 16 (optimizer-state coherent with the
5070-era runs); go WIDE, never narrow — the Max loses to a 5070 at batch 1.
`--rollout-legacy` = escape hatch to the old inline rollout path if quality ever looks off.
**α=0.5, not 0.6** (fixed 2026-07-15): this block was originally copied from the pre-anneal
2026-06-27 command and silently carried `--teacher-mix-alpha 0.6`; the probe-validated decision
(7458 anneal verdict → 8668 "hold 0.5, pour tokens") is **0.5**. The ~100 XPU steps of 2026-07-14
(9780→9881) and the smoke tests likely ran at 0.6 — see generation_probe_tracker.md 2026-07-15.

**Monitoring (second terminal):**
```bash
xpu-smi dump -d 0 -m 1,2,3,18 -i 2    # power, core °C, mem °C, mem-used, every 2 s
watch -n 2 xpu-smi stats -d 0          # coarser dashboard (temps show N/A here; use dump)
```
Temp target ≤85 °C sustained (fan holds ~68–70 °C; training's fine at ~80 °C with teacher).
If cooling ever regresses: `sudo xpu-smi config -d 0 --powerlimit 225` (resets on reboot).

**Benchmarks:**
```bash
python -m tools.bench_step --variant mythouro_distill_tiny --device xpu --batch 64   # train phase
python -m tools.bench_rollout --device xpu                                           # rollout, student-only
python -m tools.bench_rollout --device xpu --batches 8 32 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code                      # rollout, real teacher
```

**Smoke test (run before any long run after env/code changes)** — 50 steps, ~7 min:
same command as the main run but `--total-steps 50 --warmup-steps 10 --onpolicy-lambda 0.5
--rollout-batch 16 --ckpt-dir /tmp/smoke_xpu`. Pass = falling losses + gate PASSED + ckpt saved.

**Sanity checks:**
```bash
xpu-smi discovery                      # kernel driver sees the card
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
pytest tests/ -q                       # full suite (365+; all green expected)
```

**XPU gotchas (all solved — don't re-debug):**
- `transformers` must be **<5** (pinned in requirements.txt) — Ouro's custom code breaks on 5.x.
- `TRITON_DEFAULT_BACKEND=intel` + `apt install intel-ocloc` are required for any
  `torch.compile` run (dual-GPU driver clash / missing kernel compiler).
- `torch.compile`: default mode only — **`max-autotune` is a measured regression on PVC.**
- **Teardown abort after "training complete"** (`terminate called without an active
  exception` + core dump) is cosmetic — checkpoint is already saved; ignore it.
- Segfault workarounds (no SDPA on XPU, bmm attention, CPU sampling, forced rope_real) are
  committed and load-bearing — see hardware_options.md before "cleaning them up".
- One flaky segfault at rollout start was seen 2026-07-12; rollouts auto-retry once now.
  If a run dies hard, just rerun — `--ckpt-every-mins 15` bounds the loss.

---

## ⚠️ Python environment (WINDOWS side) — USE THE CUDA BUILD (env gotcha, 2026-06-24)

Automated/fresh shells resolve `python` to a **CPU-only `.venv`** (`d:\MythOuro-main\.venv`) →
`AssertionError: Torch not compiled with CUDA enabled`. The working **CUDA build**
(torch 2.13.0+cu132) is:
```
C:\Users\danie\AppData\Local\Python\pythoncore-3.14-64\python.exe
```
Your interactive PowerShell already uses it; this only bites fresh/automated shells. Call it
explicitly when needed:
```powershell
& "C:\Users\danie\AppData\Local\Python\pythoncore-3.14-64\python.exe" tools/collapse_metrics.py -c <path.pt> --device cuda:0 --generate --probe-set all
```

---

## ⭐ CURRENT (2026-06-27): on-policy / GKD — the VALIDATED cure (continue from 6771)

On-policy distillation **un-collapsed the α=0.0 prose seed** — first movement on the generation
blocker ever (training_runs.md / generation_probe_tracker.md 06-27). Partial (medical/code still
dose-limited) → now a throughput/dose problem. **Continue from 6771 at λ=0.7** (gnorm had headroom):
```powershell
python -m training.distill --student-variant mythouro_distill_tiny --student-device cuda:0 --teacher-device cuda:2 --teacher-id ByteDance/Ouro-2.6B-Thinking --seq-len 1024 --micro-batch 1 --grad-accum 16 --total-steps 12000 --warmup-steps 500 --lr 1e-4 --depth-reg-coeff 0.3 --divergence rev_kl --use-sandwich-norm --use-depth-aware-init --onpolicy-lambda 0.7 --teacher-mix-alpha 0.6 --rollout-len 64 --ckpt-every-mins 15 --num-workers 0 --trust-remote-code --ckpt-dir checkpoints_onpolicy
```
Cross-GPU (teacher `cuda:2`) — single-card OOMs a 12 GB 5070 (5.2 GB teacher won't cohabit; the Max
1100's 48 GB is the fix → batched rollouts, see hardware_options.md). **Probe the result** (read the
**α=0.0** rows vs the collapsed baseline — top_share down / distinct up = working):
```powershell
python -m tools.onpolicy_rollout_probe --ckpt-dir checkpoints_onpolicy --student-device cuda:0 --teacher-device cuda:2 --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code
```
Warm-start a fresh on-policy dir from a checkpoint first: `Remove-Item <dir>\*.pt; copy <src>.pt <dir>\; dir <dir>`
then confirm the launch log says `resuming from …<step>.pt` (not step 1). Full design + perf notes:
`docs/onpolicy_plan.md`.

---

## ✅ DONE (2026-06): rev-KL STABILITY run — stability solved; but pure rev-KL collapses

The gnorm-explosion collapse at lr 3e-4 (which killed JSD at the `n_loops 2→3` transition) is
fixed by **lr 1e-4 + `--use-sandwich-norm --use-depth-aware-init`**. Healthiest run yet
(gnorm flat ~1.0, cv 0.18); **survived the `n_loops 2→3` transition** that detonated JSD. Resume /
continue (auto-resumes from the latest checkpoint in `--ckpt-dir`):
```powershell
python -m training.distill --student-variant mythouro_distill_tiny --student-device cuda:0 --teacher-device cuda:2 --seq-len 1024 --micro-batch 1 --grad-accum 16 --total-steps 12000 --warmup-steps 500 --lr 1e-4 --depth-reg-coeff 0.3 --divergence rev_kl --use-sandwich-norm --use-depth-aware-init --num-workers 0 --trust-remote-code --ckpt-dir checkpoints_revkl_stable
```
**Outcome (depth-matched, step 6675):** stability held to 6675, **but pure rev-KL mode-COLLAPSES**
(`is is is`) — best-ever PPL (1.759) + good ECE (**0.0152**; the earlier "0.20" was a depth-mismatch
artifact, **retracted**) + healthy reps, and free-gen still collapses. Exposure bias is decoupled from
every formal metric → no offline divergence reaches coherence → **on-policy** (the CURRENT run above).
This checkpoint (`checkpoints_revkl_stable/step_0006675.pt`) is the **warm-start base** for on-policy.

## ⛔ DEPRIORITIZED: stable-JSD (offline-divergence avenue closed)

JSD on the stable footing. **Deprioritized 2026-06-27:** fwd/rev-KL *and* JSD all mode-collapse — no
offline divergence alone reaches coherence (on-policy is validated). Kept for the record; not the path.
Fresh from random init:
```powershell
python -m training.distill --student-variant mythouro_distill_tiny --student-device cuda:0 --teacher-device cuda:2 --seq-len 1024 --micro-batch 1 --grad-accum 16 --total-steps 12000 --warmup-steps 500 --lr 1e-4 --depth-reg-coeff 0.3 --divergence jsd --jsd-beta 0.5 --use-sandwich-norm --use-depth-aware-init --num-workers 0 --trust-remote-code --ckpt-dir checkpoints_jsd_stable
```
Calibration levers (separate axis from divergence — see ideas.md / training_runs.md 06-23): bump
**`--unc-coeff`** (the `uncertainty_calibration_loss`); post-hoc **temperature scaling** at inference.

## NEXT (B): base-teacher A/B (concise vs thinking-heavy target) — ⛔ DEFERRED

> **⛔ DEFERRED (2026-06-24) — blocked by env *and* likely ~a no-op anyway:**
> 1. **Env blocker.** The base `Ouro-2.6B` repo ships *pre-5.x* `modeling_ouro.py` (uses
>    `ROPE_INIT_FUNCTIONS['default']`, which transformers **5.8.1 removed** — only scaling variants
>    remain: linear/dynamic/yarn/longrope/llama3/proportional; the standalone default-rope fn is gone too).
>    No clean fix: hand-writing the rope risks *silently-wrong* targets (worse than a crash); downgrading
>    transformers risks breaking MythOuro (built on 5.8.1); cache-swapping `-Thinking`'s newer code works
>    but is fragile (re-download reverts it). `-Thinking` loads because its repo has the newer code.
> 2. **Likely ~no-op for OFFLINE CORPUS distillation.** The teacher computes next-token logits on *plain*
>    FineWeb/math/code text — it does NOT *generate* the verbose thinking monologue (that only appears in
>    chat). So base vs `-Thinking` give **nearly-identical corpus targets** (a modest fine-tune shift, not
>    a thinking-vs-concise chasm). The student's `is is is` collapse is **plain-text exposure bias**,
>    teacher-agnostic → cure is **on-policy**, not teacher choice.
> **Teacher-style matters only when the teacher GENERATES** (sequence-level KD, on-policy) or on chat/CoT
> data — i.e. the post-coherence/on-policy stage, on a controlled env (Linux/Max rig, pinned transformers).
> Revisit there. (The `pad_token_id` backfill in `load_distillation_teacher` stays — harmless robustness.)

**Teacher-style hypothesis (2026-06-24):** Ouro-2.6B-**Thinking** emits verbose CoT monologues
("Okay, let me figure out… but wait… let me check…"); a tiny 278M student may learn a **concise**
target (base `Ouro-2.6B` → `return fibonacci(n-1) + fibonacci(n-2)`) more coherently, *and* the
thinking-heavy distribution plausibly amplifies the high-frequency collapse (`is`/`the`/`Let me`/`So`).
**No code change — `--teacher-id` already exists.** Clean single-variable A/B vs the rev-KL-stable run
(everything identical *except* the teacher → isolates teacher-style):
```powershell
python -m training.distill --student-variant mythouro_distill_tiny --student-device cuda:0 --teacher-device cuda:2 --teacher-id ByteDance/Ouro-2.6B --seq-len 1024 --micro-batch 1 --grad-accum 16 --total-steps 12000 --warmup-steps 500 --lr 1e-4 --depth-reg-coeff 0.3 --divergence rev_kl --use-sandwich-norm --use-depth-aware-init --num-workers 0 --trust-remote-code --ckpt-dir checkpoints_revkl_base_teacher
```
**Use the HF id `ByteDance/Ouro-2.6B`, NOT the local `D:\LLMs\Ouro-2.6B`** — verified 2026-06-24: MythOuro
runs **transformers 5.8.1**, but the local copy's `modeling_ouro.py` is patched for **4.54.x** (a *major*
version behind) → likely crashes on 5.8.1. The HF id uses the hub modeling code MythOuro *already* loads
fine for `-Thinking`. **Do NOT pin the teacher depth / set `early_exit_threshold`:** verified from
`modeling_ouro.py` (line 818) the default **`threshold=1.0` already = never-early-exit = full 4 steps**
(stable targets); the "set 0.0" advice is **backwards** (0.0 → exit at step 1 = shallow/degraded targets).
`load_distillation_teacher` uses the default config — already correct. **Expectation:** easier
offline target → maybe less/different collapse, *but* exposure bias is offline-inherent → likely still
collapses; **on-policy remains the deep cure**. Informative single-variable test, not a guaranteed fix.

## Probe / eval the stability checkpoints (use the CUDA python above)
```powershell
& "C:\Users\danie\AppData\Local\Python\pythoncore-3.14-64\python.exe" tools/collapse_metrics.py -c checkpoints_revkl_stable/step_0004000.pt --device cuda:0 --generate --probe-set all
& "C:\Users\danie\AppData\Local\Python\pythoncore-3.14-64\python.exe" tools/collapse_metrics.py -c checkpoints_revkl_stable/step_0004000.pt --device cuda:0 --generate --probe-set all --temperature 0.8 --top-k 40
& "C:\Users\danie\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m eval.harness --checkpoint checkpoints_revkl_stable/step_0004000.pt --device cuda:0 --max-samples 500 --output checkpoints_revkl_stable/eval_step_4000.json
```

---

## Earlier: v6 Clean SFT (on moe_s0 base)

> **Pre-flight:** clear or rename `checkpoints_v6_clean_sft/` if restarting
> from scratch (old checkpoints from the code-data-starved attempt live there).

```bash
python -m training.sft --resume checkpoints_ablation_moe_s0/step_0004000.pt --device cuda:0 --seq-len 1024 --micro-batch 1 --grad-accum 16 --total-steps 3000 --warmup-steps 100 --lr 1e-5 --depth-reg-coeff 0.1 --random-depth --seed 0 --eval --eval-every 1000 --ckpt-dir checkpoints_v6_clean_sft
```

**What this does:** SFT on the clean data mix (no OpenAI provenance) using
the best distill checkpoint (moe_s0, PPL 5.72). Uses `--data-mix clean` by
default. OpenCodeInstruct code-data fix applied 2026-06-12 (commit bf20338).

---

## Ablation: Distillation runs

### MoE seed 0 (complete — PPL 5.72)
```bash
python -m training.distill --trust-remote-code --teacher-device cuda:2 --warmup-steps 500 --depth-reg-coeff 0.3 --micro-batch 1 --grad-accum 8 --start-loops 2 --random-depth --seed 0 --total-steps 4000 --eval --eval-every 1000 --ckpt-dir checkpoints_ablation_moe_s0
```

### MoE seed 1 (complete — PPL 22.23)
```bash
python -m training.distill --trust-remote-code --teacher-device cuda:2 --warmup-steps 500 --depth-reg-coeff 0.3 --micro-batch 1 --grad-accum 8 --start-loops 2 --random-depth --seed 1 --total-steps 4000 --eval --eval-every 1000 --ckpt-dir checkpoints_ablation_moe_s1
```

### Dense seed 0 (complete — PPL 22.66)
```bash
python -m training.distill --trust-remote-code --teacher-device cuda:2 --student-variant mythouro_distill_tiny_dense --warmup-steps 500 --depth-reg-coeff 0.3 --micro-batch 1 --grad-accum 8 --start-loops 2 --random-depth --seed 0 --total-steps 4000 --eval --eval-every 1000 --ckpt-dir checkpoints_ablation_dense_s0
```

### Dense seed 1 (complete — PPL 20.83)
```bash
python -m training.distill --trust-remote-code --teacher-device cuda:2 --student-variant mythouro_distill_tiny_dense --warmup-steps 500 --depth-reg-coeff 0.3 --micro-batch 1 --grad-accum 8 --start-loops 2 --random-depth --seed 1 --total-steps 4000 --eval --eval-every 1000 --ckpt-dir checkpoints_ablation_dense_s1
```

> Ablation verdict (both seeds): **inconclusive — seed variance > architecture
> effect at 278M.** See `training_runs.md`. MoE re-tested at scale, not closed.

---

## Generation-degeneration cure test (next away-run)

Diagnosis: exposure bias (see `training_runs.md` 06-16). Cheapest on-target test —
**mode-seeking (reverse-KL) distillation**, continued from the 24-expert base:
```powershell
New-Item -ItemType Directory -Force checkpoints_revkl_test
Copy-Item checkpoints_distill_cont/step_0008000.pt checkpoints_revkl_test/step_0008000.pt
python -m training.distill --student-variant mythouro_distill_tiny --student-device cuda:0 --teacher-device cuda:2 --seq-len 1024 --micro-batch 1 --grad-accum 16 --total-steps 14000 --warmup-steps 500 --lr 3e-4 --depth-reg-coeff 0.3 --divergence rev_kl --trust-remote-code --ckpt-dir checkpoints_revkl_test
```
Judge it: `python tools/collapse_metrics.py -c checkpoints_revkl_test/step_0014000.pt --device cuda:0 --generate`.
Tier-2 (full on-policy + teacher-mix) is gated on this result. (Note: distill uses
`--student-device`, not `--device`.)

---

## Compound commands (chained with `;`)

`cmd_a ; cmd_b` runs b after a finishes (even if a errors) — for overnights.
Both training scripts auto-resume from their `--ckpt-dir`.

```bash
python -m training.distill ... --ckpt-dir checkpoints_ablation_dense_s1 ; python -m training.sft ... --ckpt-dir checkpoints_v6_clean_sft
```

---

## Inspection & Eval

The flag is `--checkpoint` (or `-c`), **not** `--ckpt`.

### Run eval harness on a checkpoint
```bash
python -m eval.harness --checkpoint <path.pt> --device cuda:0 --benchmarks all --max-samples 50
```

### Inspect a checkpoint (generation + diagnostics)
```bash
python inspect_checkpoint.py --checkpoint <path.pt> --device cuda:0
```
Default prompt set + the v6+ domain/honesty extension prompts are documented in
[`training_runs.md`](training_runs.md) ("Test prompt suite"); pass one with
`--prompt "..."`.

### Per-loop calibration audit (P0.5 tool)
```bash
python -m tools.per_loop_calibration --checkpoint <path.pt> --max-samples 20
```

### Benchmark step speed
```bash
python -m tools.bench_step --variant mythouro_distill_tiny --device cuda:0
```

### Collapse / degeneration diagnostics (2026-06-16)
Quantify generation degeneration (per-loop token-correlation, effective-rank,
output entropy). Fast, forward-only — runnable in a couple minutes at home.
```bash
# static: reps healthy if eff_rank high / token_corr low
python tools/collapse_metrics.py -c <path.pt> --device cuda:0
# generation-time: locate the degeneration (reps vs output distribution)
python tools/collapse_metrics.py -c <path.pt> --device cuda:0 --generate
# sampling / inference-noise variants (both shown NOT to escape the spiral):
python tools/collapse_metrics.py -c <path.pt> --device cuda:0 --generate --temperature 0.8 --top-k 40
python tools/collapse_metrics.py -c <path.pt> --device cuda:0 --generate --inference-noise 0.1
```
Verdict (06-16): degeneration is **exposure bias** (a learned repetition
attractor), NOT recurrent collapse. Full chain in `training_runs.md`.

---

## Key flags reference

| Flag | Purpose |
|------|---------|
| `--resume <ckpt>` | Start SFT from a distill checkpoint |
| `--device cuda:0` | GPU for the student (the 5070) |
| `--teacher-device cuda:2` | GPU for the teacher = 5060 (distill only) |
| `--seq-len 1024` | Sequence length (≥1024 for multi-turn) |
| `--micro-batch 1` | Per-step batch size (OOM safety) |
| `--grad-accum 16` | Effective batch = micro-batch × grad-accum |
| `--warmup-steps` | LR warmup (500 for distill, 100 for SFT) |
| `--lr 1e-5` | Peak learning rate |
| `--depth-reg-coeff` | Depth regulariser (0.3 distill, 0.1 SFT) |
| `--random-depth` | Sample loop count from the curriculum |
| `--start-loops 2` | Curriculum starts at 2 loops |
| `--seed 0` | RNG seed (model init, depth sampling, dropout) |
| `--eval` / `--eval-every 1000` | Run eval harness at checkpoints, frequency |
| `--data-mix clean` | Clean SFT mix (default; no OpenAI provenance) |
| `--data-mix legacy` | v2/v4-era OpenHermes mix (reproduction only) |
| `--data-mix clean_chat` | Chat-heavy clean mix (Tulu-dominant, low-structured) |
| `--no-contamination-filter` | Disable GSM8K/ARC 13-gram guard (on by default for clean) |
| `--divergence {fwd_kl,rev_kl,jsd}` | Distill divergence; rev_kl/jsd = mode-seeking (anti-degeneration). Default fwd_kl = prior behaviour |
| `--jsd-beta 0.5` | JSD interpolation weight when `--divergence jsd` (β→0≈fwd, β→1≈rev) |
| `--recurrent-state-noise 0.05` | Training-time hidden-state noise regulariser (default 0 = off) |
| `--use-sandwich-norm` | Huginn sandwich norm — **PROMOTED** (2026-06): part of the stability recipe (lr 1e-4 + this + depth-aware-init) that fixed the gnorm-explosion collapse. Fresh runs only |
| `--use-depth-aware-init` | Huginn/Takase depth-aware init — **PROMOTED** (same stability recipe). Fresh runs only |
| `--num-workers 0` | Dataloader workers; **0 = clean `Ctrl+C` graceful checkpoint** (used in the stability runs) |
| `--unc-coeff` | Weight on `uncertainty_calibration_loss` — **calibration lever**; bump to fight the ECE regression (training_runs.md 06-23) |
| `--lr 1e-4` | **Stability-recipe peak LR** (3e-4 caused the gnorm-explosion collapse) |
| `--use-8bit-adam` | Quantize optimizer state to 8-bit (bitsandbytes) — see VRAM playbook |
| `--ckpt-dir <dir>` | Where to save checkpoints |

---

## VRAM playbook (12 GB 5070)

Baseline: v6 SFT (278M, seq 1024, mb1/ga16) sits at **~9.5 GB**. Past grown-MoE
runs (v3–v5) ran stably up to **~11.7 GB**, so ~2.5 GB headroom here is
comfortable — *don't add compression to runs that already fit.* Pull a lever
only when you deliberately push past the ceiling.

**Micro-batch throughput (measured 2026-06-13):** student-only bench (seq 1024,
5070) — mb1 2,865 tok/s, **mb2 4,616 (1.61×)**, mb4 5,557 (saturating), mb8 OOM.
BUT the gain only materialises when the *student* is the bottleneck:
- **SFT (no teacher): use `--micro-batch 2`** — ~1.6× free throughput, fits
  (student 8.6 GB w/ grad-checkpointing, far below the bench's no-checkpoint 10 GB).
- **Distillation: micro-batch does NOT help** — the 2.6B teacher forward
  dominates and scales with batch, so mb2 ≈ mb1 end-to-end (~0.8k tok/s, both
  cards fit: student 8.6/12, teacher 5.9/8, no spill needed). Distill token
  volume = wall-clock or rent, not batch.

**Reduction options** (bang-for-buck order):

| Lever | Saves | Cost | Status |
|-------|-------|------|--------|
| `--use-8bit-adam` | ~2 GB (quantizes Adam m/v state, the biggest consumer) | none meaningful; convergence near-identical | **ready** — flag exists, bitsandbytes 0.49.2 installed. Smoke-test ~20 steps first (v4 hit a bnb CUDA-binary snag on 13.2; `_configure_bnb_cuda_version` handles it) |
| Gradient checkpointing | already saving (recurrent loops recompute) | recompute time | **already ON** by default |
| Shorter `--seq-len` | ~linear in activations | multi-turn / long-doc quality | avoid unless needed |
| LoRA-only SFT (freeze base, train per-loop adapters) | *huge* (no optimizer state on frozen base) | slightly less plasticity; needs wiring | architecture HAS per-loop LoRA; not yet wired for freeze-base SFT |
| Paged AdamW (bnb) | spills optimizer to RAM on spikes | speed on spill | one bnb swap |

**When to pull which** (the three pressure cases that eat the 2.5 GB reserve):

- **Longer sequence** (seq-len > 1024, long-doc/multi-turn): activations scale
  ~linearly → `--use-8bit-adam` to reclaim the headroom.
- **Bigger batch** (micro-batch 2): roughly doubles activation memory →
  `--use-8bit-adam`, and/or paged AdamW for spike safety.
- **Bigger base model** (the 1B-on-the-5070 milestone): fp32 Adam state alone
  is ~8 GB and won't coexist with model + activations → `--use-8bit-adam`
  becomes **mandatory**, LoRA-only SFT becomes attractive. This is the wall
  that caps single-card model size.

---

## Notes

- **Proven recipe (distill):** warmup 500, depth-reg 0.3, mb1/ga8, start-loops 2
  — recovered from v1's MODEL_CARD after the script defaults flatlined a run
  (now the defaults). Always diff a command against the model-card provenance.
- **SFT recipe:** warmup 100, depth-reg 0.1, mb1/ga16, lr 1e-5, seq-len 1024.
- **Teacher placement:** `--teacher-device cuda:2` (the 5060) keeps the 5070
  free for the student. Cohabiting teacher+student on the 5070 OOMs at mb2.
- **Clean code fix (2026-06-12):** OpenCodeInstruct `tests_execution_status`
  is a JSON list, not a single string — fixed in `sft_data.py` (commit
  bf20338); pre-flight confirms all 7 clean sources yield.
- **eval_results/ collision:** filenames collide across runs — copy a run's
  eval JSONs into its checkpoint dir as sidecars right after it finishes.
