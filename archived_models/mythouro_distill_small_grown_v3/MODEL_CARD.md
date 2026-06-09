# MythOuro Distill Small — v3 (MoE-grown)

First **model-growth** deliverable: `mythouro_distill_tiny_sft_v2` (278M, 24
routed experts) promoted to `mythouro_distill_small` (420M, **48 routed
experts**) via function-preserving MoE expansion, then SFT-continued.
Trained 2026-06-04 → 06-05.

> **Reconstructed card.** v3 had no contemporaneous MODEL_CARD or eval JSONs.
> Config is from the roadmap/CHANGES; timing is derived from checkpoint
> save-timestamps; eval figures are the values recorded in
> [`docs/roadmap.md`](../../docs/roadmap.md) (capability-criteria + sessions
> log), not a re-run.

---

## Lineage

```
distill_tiny_v1 (distill, 278M/24exp)
   → distill_tiny_sft_v2 (SFT)
      → [MoE expansion 24→48, grow.py]  ← this step
         → distill_small_grown_v3 (SFT) = THIS checkpoint
```

## Architecture / growth

- **Variant:** `mythouro_distill_small` (dim=1280, 48 routed + 2 shared experts,
  top-4, expert_dim=1280; recurrent depth × MLA × ACT identical to tiny).
- **Promotion:** `tools/grow_checkpoint.py` doubled the routed pool 24→48. New
  experts initialised with **zeroed `down` projections** + large negative
  **sentinel `router_bias`**, so the promoted model is byte-identical in output
  to v2 at step 0 (function-preserving). A linear **sentinel decay** eases the
  new experts into top-k routing; the DeepSeek-V3 aux-loss-free bias updater then
  rebalances the pool. See [`docs/growth_design.md`](../../docs/growth_design.md).

## Training

| Item | Value |
|------|-------|
| Method | MoE expansion (24→48) + supervised fine-tuning (masked CE on responses) |
| Base | `archived_models/mythouro_distill_tiny_sft_v2/step_0003000.pt`, promoted to 48 experts |
| Checkpoint | `step_0003500.pt` |
| Dataset mix | Magicoder-Evol-Instruct (code) + MetaMathQA (math) |
| Optimizer / precision | AdamW, bf16 autocast |
| Depth-reg coeff | 0.1 |
| Hardware | RTX 5070 (cuda:0, 12 GB), student only |
| **Per-step time (measured)** | **~8 s/step** (ckpt 2500 @ 06-04 22:32 → 3500 @ 06-05 00:44 = ~131 min / 1000 steps) |

## Eval / findings (from roadmap record)

| Metric | Value | Note |
|--------|------:|------|
| Function preservation at promotion | logits == v2 within fp tol | validated by `tests/test_grow.py` |
| Loss spike at promotion | none (ce stayed ~1.4) | sentinel decay worked |
| MoE cv (after stabilisation) | **0.19** | exceptional — *better* than v2's 0.34 |
| MoE min% / max% | 1.1% / 3.0% | all 48 experts get traffic by step 3000 |
| ρ(A) (LTI injection) | 0.34–0.39 | stable |
| Inspector | math prompt → `eos`; more domain-relevant content than v2 on math+code | — |

**Headline:** the first MoE expansion *compounded* — it recovered the halt
mechanisms and tightened routing (`cv 0.34 → 0.19`) while preserving function.
This is the round-1 growth success that later motivated (and bounded) v5.

## Known limitations

- **Content still gibberish** — 420M is still far below the coherence threshold.
  Growth added capacity, not coherence (the parameter-count ceiling holds).
- Social/chat register still weak (math+code data only — addressed in v4).

## How to use

```python
import torch
from mythouro import MythOuro
ckpt = torch.load("archived_models/mythouro_distill_small_grown_v3/step_0003500.pt", weights_only=False)
model = MythOuro(ckpt["cfg_dict"]); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
```

## Provenance

- Reconstructed 2026-06-08 from roadmap/CHANGES + checkpoint mtimes.
- Superseded by **v4** (OpenHermes-augmented SFT on this checkpoint).
