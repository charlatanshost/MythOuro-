# MythOuro Training Commands

Copy-paste-ready command reference. Companion to
[`training_runs.md`](training_runs.md) (which records what the runs *produced*)
— this file is *how to run them*.

GPU map on this rig (not intuitive — verify with `torch.cuda.get_device_name`):
`cuda:0` = RTX 5070 (12 GB, student), `cuda:1` = RTX 4060, `cuda:2` = RTX 5060
(teacher host for distillation).

---

## Current: v6 Clean SFT (on moe_s0 base)

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
| `--no-contamination-filter` | Disable GSM8K/ARC 13-gram guard (on by default for clean) |
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
