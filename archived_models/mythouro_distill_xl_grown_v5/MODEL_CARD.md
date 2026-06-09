# MythOuro Distill XL — v5 (2nd MoE expansion) — ceiling data point

`mythouro_distill_small_v4` (420M, 48 experts) promoted to
`mythouro_distill_xl` (632M, **96 routed experts**) via a *second* MoE
expansion, then SFT-continued with 8-bit AdamW. **This is a negative result by
design:** the 2nd expansion hit the **expert-count ceiling** — net-comparable to
v4 — and is the data point that closed the MoE-growth axis. Trained 2026-06-06.

> **Reconstructed card.** v5 had no contemporaneous MODEL_CARD or eval JSONs.
> Config from roadmap/CHANGES; timing from checkpoint mtimes; eval figures are
> the values recorded in [`docs/roadmap.md`](../../docs/roadmap.md).

---

## Lineage

```
… distill_small_v4 (48exp)
   → [MoE expansion 48→96, grow.py] + SFT  ← this step
      → distill_xl_grown_v5 = THIS checkpoint (step 2887)
```

## Architecture / growth

- **Variant:** `mythouro_distill_xl` (dim=1280, **96 routed** + 2 shared experts,
  top-4, expert_dim=1280). Same function-preserving promotion mechanism as v3
  (zeroed `down`, sentinel `router_bias`, linear decay).
- **8-bit AdamW now working:** `_configure_bnb_cuda_version()` auto-detects the
  bundled `cuda130` binary against the CUDA 13.2 runtime (the issue that forced
  v4 to fp32). Saved ~2.5 GB optimizer state — what made the 632M model fit.

## Training

| Item | Value |
|------|-------|
| Method | MoE expansion (48→96) + ~2.4K SFT steps |
| Base | `archived_models/mythouro_distill_small_v4/step_0003000.pt`, promoted to 96 experts |
| Checkpoint | `step_0002887.pt` (run stopped early — ceiling already evident) |
| Optimizer | **8-bit AdamW** (bitsandbytes, cuda130 auto-detect) |
| Seq len / batch | 768 / micro-batch 1, grad-accum 16 (per the run command) |
| Hardware | RTX 5070 (cuda:0, 12 GB), student only |
| **Per-step time (measured)** | **~33 s/step** (ckpt 2000 @ 06-06 13:53 → 2887 @ 22:01 = ~33 s/step; ~8 h for 887 steps) — ~370 tok/s, the slowest run of the PoC |

The ~33 s/step (≈ **2.7× v4**, ≈ **7× v2**) is the cost of 632M params + 96-expert
MoE dispatch + micro-batch 1, all squeezed into 12 GB. This run is the clearest
evidence that the **12 GB VRAM budget — not raw compute — gates the heavy runs**:
micro-batch 1 is forced, which is the worst case for GPU utilisation.

## Eval / findings (from roadmap record) — the ceiling

| Metric | Value | Verdict |
|--------|------:|---------|
| MoE cv | **~0.5** (would not tighten below) | vs v3/v4's 0.19–0.20 — routing stopped improving |
| MoE min% | ~0.1–0.4% | new experts stayed under-utilised |
| 7-prompt inspector | **net-comparable to v4** (2 better, 2 worse on the standard 4) | 2nd expansion did **not** compound |

**Conclusion (answered Open Research Q#1, 2026-06-06):** MoE expansion does **not**
compound across rounds at this scale. Round 1 (24→48, v3) clearly helped; round 2
(48→96, v5) hit the expert-count ceiling — at 632M / ~20–40M tokens the model
can't find distinct work for 96 experts. **Do not do a third expansion.** The next
lever is width (Net2Wider) or scale, not more experts. This negative result is
v5's actual value: it *mapped a ceiling*.

## Known limitations

- **No improvement over v4** — same gibberish content, comparable behaviour, worse
  routing balance. Kept as the ceiling data point, not as a capability upgrade.
- Run stopped at step 2887 (not a round number) because the ceiling was already
  evident and further steps weren't earning their (expensive) keep.

## How to use

```python
import torch
from mythouro import MythOuro
ckpt = torch.load("archived_models/mythouro_distill_xl_grown_v5/step_0002887.pt", weights_only=False)
model = MythOuro(ckpt["cfg_dict"]); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
```

## Provenance

- Reconstructed 2026-06-08 from roadmap/CHANGES + checkpoint mtimes.
- **Terminal checkpoint of the growth line.** Forward path is from-scratch
  distilled at scale on rented/HBM compute (see roadmap scale-up plan), not
  further growth.
