# MythOuro Distill Small — v4 (OpenHermes-augmented SFT)

`mythouro_distill_small_grown_v3` (420M, 48 experts) continued with an
**OpenHermes-augmented SFT** at `seq_len=768`. The run where **all three halt
mechanisms fire on 4/4 prompts** — the behavioural high-water mark of the
proof-of-concept. Trained 2026-06-05.

> **Reconstructed card.** v4 had no contemporaneous MODEL_CARD or eval JSONs.
> Config from roadmap/CHANGES; timing from checkpoint mtimes; eval figures are
> the values recorded in [`docs/roadmap.md`](../../docs/roadmap.md).

---

## Lineage

```
… distill_small_grown_v3 (48exp)
   → [OpenHermes-augmented SFT, seq_len=768]  ← this step
      → distill_small_v4 = THIS checkpoint
```

## Architecture

Identical to v3 (`mythouro_distill_small`, 420M, 48 routed + 2 shared experts).
No growth this step — pure SFT continuation with a broader data mix.

## Training

| Item | Value |
|------|-------|
| Method | SFT (masked CE on responses), OpenHermes re-added |
| Base | `archived_models/mythouro_distill_small_grown_v3/step_0003500.pt` |
| Checkpoint | `step_0003000.pt` |
| Seq len | **768** (planned 1024; reduced to fit fp32 AdamW in 12 GB) |
| Optimizer | **fp32 AdamW** — bitsandbytes 8-bit Adam could not load on this machine's CUDA 13.2 (no matching prebuilt binary at the time; fixed later for v5) |
| Precision | bf16 autocast |
| Depth-reg coeff | 0.1 · random-depth on |
| Dataset mix | OpenHermes-2.5 (general chat) + Magicoder (code) + MetaMathQA (math) |
| Hardware | RTX 5070 (cuda:0, 12 GB); peak VRAM ~10.7 GB (tight but stable) |
| **Per-step time (measured)** | **~13 s/step** (ckpt 2000 @ 06-05 07:09 → 3000 @ 10:47 = ~218 min / 1000 steps) |

The ~13 s/step (vs v3's ~8) reflects the longer seq_len (768) and **fp32**
optimizer (no 8-bit Adam), both forced by the 12 GB budget.

## Eval / findings (from roadmap record)

**Headline — all three halt mechanisms now fire (4/4 prompts halt cleanly):**

| Prompt | v3 stop | **v4 stop** | Mechanism |
|--------|---------|-------------|-----------|
| Math (2+2) | `eos` | `eos` | end-of-turn |
| Code (fib) | `max_new_tokens` | **`confidence`** | UncertaintyHead |
| Trivia (France) | `max_new_tokens` | **`cycle`** | repetition detector |
| Hello | `max_new_tokens` | **`confidence`** | UncertaintyHead |

- "Say hello" now opens **"Sure,"** and halts on `confidence` (v3 ran 50 tokens of
  code-register gibberish) — the OpenHermes data unlocked the social register.
- Trivia fires `cycle` (v3 ran out the clock).
- MoE cv **0.20**, min% **1.4** (better than v3's 1.1); no loop collapse; ρ(A) 0.34–0.39.

The OpenHermes variety re-tightened calibration that had drifted v2→v3 and
generalised the confidence-halt across every prompt type. **Best behavioural
checkpoint of the PoC** — every architectural mechanism demonstrably working.

## Known limitations

- **Content still gibberish at 420M** — every *mechanism* fires correctly, but the
  text isn't coherent. Scale ceiling, not a design problem.
- OpenHermes acceptance only ~40% at seq_len=768 (multi-turn truncation);
  `seq_len=1024` would raise it but didn't fit fp32 AdamW in 12 GB.

## How to use

```python
import torch
from mythouro import MythOuro
ckpt = torch.load("archived_models/mythouro_distill_small_v4/step_0003000.pt", weights_only=False)
model = MythOuro(ckpt["cfg_dict"]); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
```

## Provenance

- Reconstructed 2026-06-08 from roadmap/CHANGES + checkpoint mtimes.
- Base for **v5** (2nd MoE expansion 48→96).
