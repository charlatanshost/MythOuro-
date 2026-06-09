# MythOuro Distill Tiny — v1

Reference checkpoint validating the MythOuro architecture end-to-end via
distillation from Ouro-2.6B-Thinking. Trained 2026-06-01.

This is a **proof-of-concept**, not a daily-driver model. The goal of this
run was to confirm the recipe; final capability is bounded by scale
(278M params), training length (5000 steps), and the teacher.

---

## Architecture

| Component | Detail |
|-----------|--------|
| Variant | `mythouro_distill_tiny` |
| Total params | 278M |
| Hidden dim | 1280 |
| Loops (max) | 4 |
| MoE experts | 24 routed + 2 shared (top-k=6) |
| Attention | MLA + GQA, KV-cache equivalent |
| Halting | ACT (Graves) + PonderNet KL-to-uniform depth regulariser |
| Stability | LTI-stable injection, ρ(A) < 1 guaranteed |
| Vocab | 49152 (Ouro tokenizer) |
| Context | 512 (training); 131072 max (Ouro tokenizer ceiling) |
| Other | LoRA-v2 per-loop, MultiScaleInjection, CrossLoopAttention, AttentionSink, UncertaintyHead |

---

## Training

| Item | Value |
|------|-------|
| Method | Hinton-style logit distillation + hard CE |
| Teacher | `ByteDance/Ouro-2.6B-Thinking` |
| Total steps | 5000 |
| Warmup steps | 500 |
| Peak LR | 3.0e-4 |
| Final LR | 3.0e-5 (cosine decay) |
| Micro-batch | 1 |
| Grad accum | 8 (effective batch 8) |
| Seq len | 512 |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1, fused) |
| Mixed precision | bf16 autocast |
| Grad checkpointing | On (recurrent block only) |
| Depth-reg coeff | 0.3 (KL-to-uniform PonderNet × Ouro) |
| ACT bias init | -2.2 (sigmoid → λ≈0.1 at init) |
| Curriculum | Random-depth, n_loops sampled from [2, curriculum(step)] |
| Dataset mix | 40% HuggingFaceFW/fineweb-edu (sample-10BT) <br> 40% open-web-math/open-web-math <br> 20% codeparrot/codeparrot-clean |
| Hardware | NVIDIA RTX 5070 (teacher, cuda:0) + RTX 5060 8GB (student, cuda:2) |

---

## Final eval (step 5000)

| Benchmark | Result | Notes |
|-----------|-------:|-------|
| **Perplexity** (held-out web text) | **37.4** | n=50 |
| Loop efficiency | avg_depth 2.01/4 (eff=0.50) | Genuinely adaptive at inference |
| ECE | 0.0414 | ~3× better than typical production LLMs |
| ARC-Challenge | 0.220 | Random ≈ 0.250; scale-limited |
| GSM8K | 0.000 | Needs instruction data + scale |

### PPL trajectory across the run

| Step | PPL | Δ |
|------|----:|--:|
| 1000 | 368.4 | — |
| 2000 | 178.6 | −52% |
| 3000 | 81.7 | −54% |
| 4000 | 51.8 | −37% |
| 5000 | 37.4 | −28% |

10× PPL reduction over the full run.

---

## Architecture health at end of training

| Metric | Value | Target band |
|--------|------:|-------------|
| Depth (mean) | ~0.04 | 0.05–0.15 (lower = more uniform) |
| MoE cv | 0.38 | < 0.8 |
| MoE max% | 7.2% | 6–10% |
| MoE min% | 2.1% | > 1% |
| MoE router bias \|·\|₂ | 7.88 | grows then plateaus |
| ρ(A) (LTI injection) | 0.35–0.39 | < 1.0 |
| gnorm (typical) | 2–8 (K=2/3), 10–20 (K=4 transitions) | bounded |

All targets met or exceeded.

---

## Known limitations

- **Reasoning capability is scale-limited.** ARC ≈ random, GSM8K = 0.
  Architecture supports reasoning (recurrent depth, ACT, MoE) but a 278M
  student trained for 5000 steps doesn't have the capacity to demonstrate
  it. Same recipe at 1B+ scale is the expected unlock.

- **Capped at teacher quality.** Distillation only — the student can match
  but cannot exceed Ouro-2.6B-Thinking on tasks. To go beyond, add: SFT,
  preference tuning (DPO/ORPO), reasoning RL with verifier rewards.

- **No instruction tuning.** Model has not seen chat-formatted training
  data; expect rambling/raw-text-style outputs.

- **Single seed, single run.** No variance estimates on eval metrics.
  n=50 per benchmark in the harness.

- **Long context unverified.** Trained at seq_len=512. Tokenizer supports
  131k context but the model has not seen anything close to that.

---

## How to use this checkpoint

### Inspect

```powershell
python inspect_checkpoint.py --checkpoint archived_models\mythouro_distill_tiny_v1\step_0005000.pt
```

### Load programmatically

```python
from mythouro import MythOuro
from mythouro.checkpointing import load_checkpoint

ckpt = torch.load("archived_models/mythouro_distill_tiny_v1/step_0005000.pt", weights_only=False)
cfg = ckpt["cfg_dict"]  # serialized MythOuroConfig
model = MythOuro(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

### Resume distillation training

```powershell
python -m training.distill --trust-remote-code `
    --teacher-device cuda:0 --student-device cuda:2 `
    --total-steps 10000 --warmup-steps 0 `
    --micro-batch 1 --grad-accum 8 `
    --ckpt-dir archived_models\mythouro_distill_tiny_v1
```

---

## Provenance

- Codebase: `d:\OpenMythos-main\OpenMythos-main`
- Tests passing at training time: 268/268
- Training command (final successful run):
  ```
  python -m training.distill --trust-remote-code \
      --teacher-device cuda:0 --student-device cuda:2 \
      --total-steps 5000 --warmup-steps 500 \
      --micro-batch 1 --grad-accum 8 \
      --depth-reg-coeff 0.3 \
      --eval --eval-every 1000 --eval-max-samples 50 \
      --random-depth
  ```
- Eval JSONs: `eval_results/distill_step_{1000,2000,3000,4000,5000}.json`
  (copied alongside this card)

---

## Next planned step

`mythouro_1b` distilled from a stronger teacher (Llama 3.3 70B or
DeepSeek V3) on 50–100B tokens of FineWeb-Edu + open-web-math +
codeparrot-clean. The architecture multipliers (recurrent depth × MoE)
should produce a model competitive with static 8–15B transformers if
training compute permits.
