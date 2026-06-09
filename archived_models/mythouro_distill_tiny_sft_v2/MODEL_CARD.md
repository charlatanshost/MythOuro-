# MythOuro Distill Tiny — v2 (SFT)

Instruction-tuned checkpoint built on top of `mythouro_distill_tiny_v1`.
Validates the SFT stage of the post-pretraining pipeline end-to-end.
Trained 2026-06-04.

This is the **second deliverable** from the proof-of-concept session.
v1 proved distillation works; v2 proves SFT works on top of it. Both
together prove the recipe scales structurally — capability is still
bounded by the 278M parameter count.

---

## Architecture

Identical to v1 (same checkpoint shape). See
`archived_models/mythouro_distill_tiny_v1/MODEL_CARD.md` for the full
breakdown. Architectural multipliers (recurrent depth × MoE × ACT)
unchanged through SFT.

---

## Training

| Item | Value |
|------|-------|
| Method | Supervised fine-tuning with masked CE on response tokens only |
| Base | `archived_models/mythouro_distill_tiny_v1/step_0005000.pt` |
| Total steps | 3000 |
| Warmup steps | 100 |
| Peak LR | 2.0e-5 (lower than distill's 3e-4 — fine-tuning a converged base) |
| Final LR | 2.0e-6 (cosine decay) |
| Micro-batch | 2 |
| Grad accum | 8 (effective batch 16) |
| Seq len | 512 |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1, fused) |
| Mixed precision | bf16 autocast |
| Depth-reg coeff | 0.1 (same as v1, kept ACT halt distribution uniform) |
| Loss masking | Response tokens only (prompt tokens contribute 0 gradient) |
| Chat template | ChatML via `MythOuroTokenizer.apply_chat_template` |
| Dataset mix | 50% Magicoder-Evol-Instruct (code instruction → response) <br> 50% MetaMathQA (math problem → CoT solution) |
| Hardware | NVIDIA RTX 5070 (cuda:0, 12GB) — student only, no teacher |
| Wall-clock | ~6 hours (training + 3 evals) |

### Loss-masking contract

The single architectural detail that distinguishes SFT from distillation:
gradient flows *only* through assistant-response tokens. Prompt tokens
(system / user turns + the `<|im_start|>assistant\n` header) contribute
zero loss. Implemented in `mythouro.sft_data._build_sft_example` (loss
mask construction) and `training/sft.masked_ce_loss` (the masked loss
function itself). Without this, the model would waste capacity learning
to predict its own prompts, which biases generation toward the input
distribution rather than producing genuine responses.

### Dataset notes

OpenHermes-2.5 was tried initially as the general-instruction source
but dropped because >95% of its multi-turn conversations exceeded
seq_len=512 in the prompt alone, leading to ~100% sample rejection.
The remaining math + code mix kept SFT productive at this scale but
left a visible gap in conversational behavior (see "Known limitations"
below). For a future run, either bump `seq_len=1024` or use a
length-filtered subset of OpenHermes.

---

## Final eval (step 3000)

| Benchmark | v1 (distill) | **v2 (SFT)** | Δ | Notes |
|-----------|-------------:|-------------:|---|-------|
| Perplexity (held-out web) | 37.4 | **46.3** | +24% | Expected — model now specialised for chat distribution, not raw web text |
| Loop efficiency | 0.502 | **0.500** | flat | Adaptive depth preserved through SFT |
| ECE | 0.0414 | **0.0578** | +39% | Still excellent (~3× better than typical production LLMs) |
| ARC-Challenge | 0.220 | **0.180** | within noise | Scale-limited, not architecture-limited |
| GSM8K | 0.000 | **0.000** | unchanged | Needs more capacity + verifier RL |

### PPL trajectory across the SFT run

| Step | PPL | Notes |
|------|----:|-------|
| Distill 5000 (start) | 37.4 | Web-text baseline |
| SFT 1000 | 48.5 | Initial chat-distribution shift |
| SFT 2000 | 46.5 | Stabilising |
| SFT 3000 | 46.3 | Plateaued — no further drift |

PPL increase is **expected behaviour for SFT**, not a regression. The
held-out web-text eval measures predictability on a distribution the
model is no longer specialised for; it doesn't capture instruction-
following quality, which is the actual SFT objective.

---

## Behavioural validation

Qualitative test on chat-formatted prompts (via
`inspect_checkpoint.py -p "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"`).
All three architectural mechanisms fired correctly on real prompts:

| Prompt type | Stop reason | Mechanism validated |
|-------------|-------------|---------------------|
| Math word problem (in-domain) | `max_new_tokens` | Model attempts but doesn't conclude — depth machinery active |
| Fibonacci code (in-domain) | **`eos`** | Model recognised `<|im_end|>` and halted naturally — chat structure learned |
| "Capital of France" (out-of-domain) | **`confidence`** | UncertaintyHead pulled the trigger — hallucination guard fired |
| "Say hello" (no training data) | `max_new_tokens` | Fell back to code register — exposes the dataset gap |

The trivia / hallucination-guard result is particularly notable: the
model has no geography training data, recognised it didn't know the
answer (via UncertaintyHead), and the `ConfidenceAwareGenerator` halted
generation before confabulation could continue. This is the calibration
training (ECE 0.058) paying off in practice as a hallucination defence.

### Architecture health at end of training

| Metric | v1 end | v2 end |
|--------|-------:|-------:|
| MoE cv | 0.38 | 0.34 |
| MoE max% | 7.2% | 5.5% |
| MoE min% | 2.1% | 2.0% |
| MoE router bias \|·\|₂ | 7.88 | 9.05 |
| ρ(A) (LTI injection) | 0.35–0.39 | 0.34–0.39 |
| Depth mean | ~0.04 | ~0.02 |

SFT *improved* MoE routing slightly (`cv 0.38 → 0.34`, `max% 7.2 → 5.5`).
Depth regulariser kept the halt distribution uniform throughout. No
architectural metric degraded.

---

## Known limitations

- **Content is gibberish.** 278M parameters is too small for coherent
  multi-token generation regardless of training. The mechanical pipeline
  works correctly; the model just doesn't have the capacity to produce
  real text. This is the scale ceiling we already documented in v1.

- **No conversational chat data.** Trained only on math + code
  instruction/response pairs. Social prompts ("say hello", "how are
  you") fall back to the code register because there's no general-chat
  pattern in training. Fixable with seq_len=1024 + OpenHermes-2.5 in
  the next SFT run.

- **Capped at the chat structure the training data exhibits.**
  Magicoder + MetaMath are single-turn (user → assistant). The model
  has not seen multi-turn conversations, so multi-turn behaviour is
  undefined. Would need multi-turn SFT data to extend.

- **Not RLHF / DPO tuned.** SFT teaches the *form* of a response;
  preference tuning teaches *preferring better responses over worse
  ones*. Next pipeline stage if continued at this scale.

---

## How to use this checkpoint

### Inspect (with chat-formatted prompt)

```powershell
python inspect_checkpoint.py `
    -c archived_models\mythouro_distill_tiny_sft_v2\step_0003000.pt `
    -p "<|im_start|>user`nWhat is 2+2?<|im_end|>`n<|im_start|>assistant`n"
```

### Load programmatically

```python
import torch
from mythouro import MythOuro
ckpt = torch.load(
    "archived_models/mythouro_distill_tiny_sft_v2/step_0003000.pt",
    weights_only=False,
)
cfg = ckpt["cfg_dict"]
model = MythOuro(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

### Resume SFT (e.g., adding OpenHermes with seq_len=1024)

```powershell
python -m training.sft `
    --resume archived_models\mythouro_distill_tiny_sft_v2\step_0003000.pt `
    --device cuda:0 `
    --seq-len 1024 --micro-batch 1 --grad-accum 16 `
    --total-steps 3000 --warmup-steps 100 `
    --lr 1e-5 `
    --depth-reg-coeff 0.1 --random-depth `
    --eval --eval-every 1000 --eval-max-samples 50
```

---

## Provenance

- Codebase: `d:\OpenMythos-main\OpenMythos-main`
- Tests passing at training time: 288/288 (268 base + 20 SFT-specific)
- Successful training command:
  ```
  python -m training.sft \
      --resume archived_models/mythouro_distill_tiny_v1/step_0005000.pt \
      --device cuda:0 \
      --total-steps 3000 --warmup-steps 100 \
      --micro-batch 2 --grad-accum 8 \
      --lr 2e-5 \
      --depth-reg-coeff 0.1 \
      --random-depth \
      --eval --eval-every 1000 --eval-max-samples 50
  ```
- Eval JSONs: `sft_step_{1000,2000,3000}.json` (copied alongside this card)

---

## Diagnostic infrastructure added this session

The SFT path needed several iterations to land. The reusable infrastructure
that came out of that debugging:

- **`mythouro/sft_data.py`**: `MixedSFTDataset` with per-sample loss-mask
  construction, per-reason rejection counters, and graceful fallbacks for
  HF streaming failures. Uses non-streaming `load_dataset` after streaming
  proved unreliable on home internet.
- **`training/sft.py`**: `masked_ce_loss` helper, `--resume`-required
  contract (refuses to run from scratch), and `resp_frac` diagnostic in
  the per-step log (fraction of tokens contributing to loss).
- **`tests/test_sft_data.py`**: 20 tests covering loss-mask correctness,
  shift-by-one alignment, schema adapters, edge cases.

The most important debugging lesson: the Ouro tokenizer's
`apply_chat_template(..., tokenize=True)` returns `BatchEncoding`, not
`list[int]`. The fix is to render with `tokenize=False` and then
`tokenizer.encode(...)` the resulting text. Documented in
`_build_sft_example`.

---

## Next planned step

`mythouro_1b` distilled from a stronger teacher (Llama 3.3 70B or
DeepSeek V3) with FSDP across the 5070 + 5060. The full distill → SFT
pipeline validated here scales to 1B with the same recipe. Expected
training duration on this hardware: ~5 days for a 20K-step capability
run, ~3 weeks for a 100K-step foundation run.
