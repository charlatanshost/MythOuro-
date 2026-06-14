# Hardware Options — the scale-up decision

Decision doc for what hardware (if any) to add for the next phase. Companion to
the roadmap's measured hardware analysis (bench numbers, the 8480 memory-
bandwidth findings) — this file is the **forward-looking purchase/rent
decision**, kept separate so it doesn't get buried.

Owner context: solo, self-funded ($1.5k is a significant, planned purchase),
**prefers to stay local** (dedicated rig planned), **Linux is fine**.

---

## What we're actually trying to buy past

Established 2026-06-13/14 (see `docs/training_runs.md`): **the bottleneck is
token volume → coherence, and the local rig can't reach it.** The 278M ceiling
is thoroughly mapped — more tokens lower PPL but don't buy coherence; behavior
needs ≥420M; content fluency needs the real scale-up (more params **and** ~1000×
more tokens than the ~32M we can grind locally). So the scale-up needs **a
faster / bigger-VRAM device + throughput**, not a recipe change.

Current rig & its bottlenecks:
- **RTX 5070, 12 GB** — the student/training card. VRAM-bound (caps batch/model size).
- **Xeon 8480 ES, 56 cores** — CPU; data pipeline runs here. Note: 2/8 DDR5
  channels = memory-bandwidth-starved (roadmap failure modes).
- **RTX 4060 / 5060, 8 GB each** — teacher host (5060) / spare (4060).
- Distillation is **teacher-bound + data-supply-bound** (not compute-bound);
  multi-GPU data-parallel distill is impossible locally (teacher+student OOMs
  8 GB — roadmap). Throughput lever: dataloader `num_workers` (queued).

---

## Rent vs buy

| | Rent (A100/H100, Linux cloud) | Buy local |
|---|---|---|
| Friction | **zero** — CUDA, everything works | setup + drivers + capital |
| Cost shape | ~$1–2/hr (~$tens for a 1B-token run) | one-time, then "free" |
| Fit | best for a **one-off** scale-up validation | best for **ongoing** local capacity |
| Owner pref | — | ✅ "stay local" |

For a *single* scale-up run, renting an A100 is the lowest-friction, likely
cheapest path. For *ongoing* local work (the stated preference), a cheap
big-VRAM card wins — **if** the ecosystem friction is cleared first.

---

## Card options

| Card | VRAM | Ecosystem | Rel. cost | Friction for THIS project |
|------|-----:|-----------|-----------|---------------------------|
| **Intel Max 1100** (Ponte Vecchio) | 48 GB HBM2e | XPU (IPEX/oneAPI) | **cheapest** | teacher-on-XPU untested; no bnb; Linux-only; driver/IPEX setup |
| **AMD MI210** (CDNA2) | 64 GB HBM2e | ROCm | mid | ROCm > XPU maturity but < CUDA; HF custom-modeling hit/miss |
| **NVIDIA A100** | 40/80 GB | CUDA | priciest | **none** — gold standard, everything works |
| Used **3090 / 4090** | 24 GB | CUDA | low–mid | **none** — drop-in; 24 GB < the others but zero friction |

---

## Intel Max / Ponte Vecchio (the card under consideration)

**Attractive:** 48 GB HBM2e for a fraction of an A100/MI210; the XPU port
already exists (`mythouro/device.py` cuda/xpu abstraction + `rope_real`
complex-op fallback) — built specifically so the **MythOuro model** runs on
Intel; good HBM bandwidth (unlike the bandwidth-starved 8480).

**The three frictions, and where they land:**
1. **The HF Ouro teacher is the gating risk.** The model is XPU-ready, but
   distillation needs the **Ouro-2.6B teacher** — HF *custom modeling* code,
   far less tested on XPU than CUDA. If its forward breaks/crawls on XPU, the
   distill pipeline (the main lever) stalls. **This is not our code, so the XPU
   port doesn't cover it — it's the make-or-break.**
2. **bitsandbytes is CUDA-only** (no 8-bit Adam on XPU). See alternatives
   below — and largely moot at 48 GB.
3. **Linux datacenter part** — no Windows support (fine per owner). IPEX +
   Level Zero + oneAPI is a real setup project. The **1100 is PCIe dual-slot**
   (workstation-viable); the **1550 is OAM** (needs a baseboard — not a
   workstation card).

Also: Intel's data-center GPU roadmap has been turbulent → some long-term
software-support uncertainty.

### 8-bit Adam on XPU
bitsandbytes won't run, but:
- **torchao low-bit optimizers** (`AdamW8bit/4bit`) — native-PyTorch-based, the
  most portable "8-bit Adam" equivalent; verify XPU coverage.
- **Adafactor** — the safe bet: factored second moment (~√ the state of Adam),
  pure PyTorch → runs anywhere incl. XPU. Needs LR tuning; battle-tested (T5).
- **DeepSpeed ZeRO-Offload** — optimizer state → CPU RAM (you have 64 GB);
  Intel contributes XPU support; *more* savings than 8-bit Adam.
- **Most importantly: 48 GB largely obviates 8-bit Adam.** It was a 12 GB-budget
  crutch. fp32 Adam state for a 1B model is ~8 GB; model + grads + activations
  on top still fit 48 GB. Just run plain fused AdamW. The VRAM *is* the fix.

---

## The de-risk gate (do BEFORE buying)

**Rent a PVC hour on Intel's Tiber Developer Cloud and run `training.distill`
end-to-end (teacher + student) on XPU.** This tests the one thing that could
kill the purchase — does the **Ouro teacher run on XPU?** — for ~free.
- Teacher loads + distill works on XPU → a 1100 purchase is justified.
- Teacher breaks on XPU → you saved the money (and learned to pick a
  CUDA card or rent instead).

Same discipline that's served the project: validate cheap before committing
capital.

---

## Recommendation (2026-06-14)

- **One-off scale-up validation:** rent an A100 — zero friction, everything
  works, cheap per-hour. Good for the first "does 1B tokens fix coherence" run.
- **Ongoing local capacity (owner's preference):** the **Max 1100 (48 GB)** is
  genuinely attractive on VRAM/$ and the XPU port covers the model — **but gated
  on the Tiber teacher-test.** Don't buy until that passes. 8-bit Adam is a
  non-issue (alternatives + 48 GB obviates it).
- **Lowest-friction local option if XPU disappoints:** a used 24 GB CUDA card
  (3090/4090) — less VRAM than the 1100, but drop-in and everything works.

> Caveat: assessment as of early-2026 knowledge. Verify current IPEX maturity,
> PVC pricing, and HF XPU custom-modeling support before deciding — those move
> fast.
