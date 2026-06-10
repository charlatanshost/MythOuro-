# MythOuro — Implementation Changes

Comprehensive record of every change layered onto the base `kyegomez/OpenMythos`
fork (renamed to **MythOuro** in this fork). Self-contained — meant to be
readable by a fresh contributor (human or agent) who hasn't seen the
conversation history.

> **Attribution note:** the upstream foundation is the work of Kye Gomez
> (`kyegomez/OpenMythos`, MIT) and is credited with thanks. Everything recorded
> in *this* changelog is independent fork work — Kye Gomez has no involvement
> in, and no responsibility for, these changes or the fork's direction. See the
> README "Acknowledgements" section.

**Status as of 2026-06-10:** 313+ tests pass. External code review fixed 5
correctness bugs (see the 2026-06-09/10 section at the bottom); first run on
the fixed code beat v1's final PPL by 6.5×. Full distillation → SFT →
model-growth pipeline validated end-to-end on consumer hardware. Trained
reference checkpoints v1 (278M distilled) through v5 (632M, 2nd MoE expansion to
96 experts) archived. **v5 hit the expert-count ceiling** — net-comparable to v4
(420M), so MoE growth is now considered tapped out; the next lever is width/scale
(Net2Wider toward ~1B single-card) or from-scratch distilled 3B on rented
compute, not more experts. See [`docs/roadmap.md`](docs/roadmap.md) for the
checkpoint lineage and forward plan.

> Earlier milestone (2026-05-31): 249/249 tests, pre-train roadmap complete
> plus working Ouro-aligned distillation verified against
> `ByteDance/Ouro-2.6B-Thinking`.

---

## 2026-06: Post-training pipeline (SFT + model growth)

Work layered on after the distillation pipeline was validated. All of this
was driven by actually training models on a 12 GB consumer GPU and fixing
what broke.

### Supervised fine-tuning (SFT)

- **[`training/sft.py`](training/sft.py)** — new SFT trainer built from the
  distillation loop. The defining difference is **loss masking**: cross-entropy
  is computed on assistant-response tokens only (`masked_ce_loss`), so prompt
  tokens contribute zero gradient. `--resume` is required (SFT builds on a
  trained base; from-scratch SFT is rejected). Keeps all architecture-specific
  auxiliary losses (MoE load balance, uncertainty calibration, sparse
  activation, depth regularisation). Logs a `resp_frac` diagnostic = fraction
  of tokens contributing to the loss.
- **[`mythouro/sft_data.py`](mythouro/sft_data.py)** — `MixedSFTDataset`
  interleaves instruction corpora (OpenHermes-2.5, Magicoder-Evol-Instruct,
  MetaMathQA), applies the ChatML template, and emits
  `(input_ids, target_ids, loss_mask)` triples. Per-reason rejection counters
  + real-time diagnostic logging. Loads datasets **non-streaming** (HF
  streaming proved unreliable on home internet — see failure modes in roadmap).
- **[`tests/test_sft_data.py`](tests/test_sft_data.py)** — 20 tests covering
  loss-mask correctness (the critical invariant), shift-by-one alignment,
  schema adapters, and edge cases.

### Model growth — MoE expansion

- **[`mythouro/grow.py`](mythouro/grow.py)** — function-preserving promotion of
  a trained checkpoint to a larger routed-expert pool (e.g. 24 → 48). At
  promotion the new experts have **zeroed `down` projections** and a large
  negative **sentinel router_bias**, so the model output is byte-identical to
  the source. A linear **sentinel decay** over the first N steps eases the new
  experts into the top-k routing, after which the DeepSeek-V3 aux-loss-free
  bias updater rebalances the pool.
- **[`tools/grow_checkpoint.py`](tools/grow_checkpoint.py)** — CLI wrapper;
  embeds growth metadata in the promoted checkpoint so the SFT trainer applies
  the sentinel-decay schedule automatically.
- **[`mythouro/variants.py`](mythouro/variants.py)** — added
  `mythouro_distill_small` (420M, 48 routed experts), the MoE-expansion target
  of `mythouro_distill_tiny`.
- **[`tests/test_grow.py`](tests/test_grow.py)** — 15 tests, including a
  bit-exact function-preservation check (promoted model's logits match the
  source within fp tolerance) and a post-promotion training-step smoke.
- **[`docs/growth_design.md`](docs/growth_design.md)** — full design rationale,
  including why MoE expansion is preferred over Net2Wider/Net2Deeper for this
  architecture (SiLU is non-idempotent, so width/depth growth isn't strictly
  function-preserving).

### Memory: bitsandbytes 8-bit AdamW

- **[`training/sft.py`](training/sft.py)** — `--use-8bit-adam` flag uses
  `bitsandbytes.optim.AdamW8bit` (block-wise 8-bit optimizer state, ~2.5 GB
  saved on a 400M model). `_configure_bnb_cuda_version()` auto-detects the
  best bundled CUDA binary before import — works around bnb not shipping an
  exact-match binary for bleeding-edge CUDA (e.g. picks `cuda130` for a
  CUDA 13.2 runtime; binaries are forward-compatible within a major version).

### Checkpointing robustness for staged training

- **[`mythouro/checkpointing.py`](mythouro/checkpointing.py)**:
  - Optimizer state is now optional on load — grown checkpoints carry an empty
    optimizer dict (shapes don't match the promoted model), so the trainer's
    fresh optimizer is kept instead of crashing on `KeyError: 'param_groups'`.
  - `max_seq_len` removed from the shape-compat shape-fields and RoPE buffers
    (`freqs_cis`, `freqs_cis_mla`) are dropped from the loaded state dict — they
    are deterministic functions of cfg, so a checkpoint trained at one
    `max_seq_len` loads cleanly into a model built with another (enables
    raising seq_len between training stages).

---

## Table of contents

1. [Architecture (Part 1 + Part 2)](#architecture-part-1--part-2)
2. [Training utilities](#training-utilities)
3. [Inference utilities](#inference-utilities)
4. [Training script integration](#training-script-integration)
5. [Eval harness (§7)](#eval-harness-7)
6. [Aux-loss-free MoE routing (§5)](#aux-loss-free-moe-routing-5)
7. [Checkpoint robustness (§4)](#checkpoint-robustness-4)
8. [Data pipeline](#data-pipeline)
9. [Flash-Attention cascade (§1)](#flash-attention-cascade-1)
10. [New-component LR warmup (§3)](#new-component-lr-warmup-3)
11. [Test coverage (§12)](#test-coverage-12)
12. [Packaging (§11)](#packaging-11)
13. [Bugs caught + fixed](#bugs-caught--fixed)
14. [Deliberately deferred](#deliberately-deferred)

---

## Architecture (Part 1 + Part 2)

All changes in [`mythouro/main.py`](mythouro/main.py).

### Part 1
- `MythOuro.forward()` returns `(logits, uncertainty)` tuple. **Every existing
  call site (training, tests, examples, README, docs) was updated to unpack.**
- `AttentionSink` — learnable register tokens prepended on prefill, stripped
  before the LM head. Stabilises attention in deep recurrent stacks.
- `UncertaintyHead` — 2-layer MLP, zero-init output, per-token confidence-of-
  error in (0, 1).
- `MoEFFN.forward` — **vectorised scatter dispatch** via `index_add_`. Each
  expert now runs once per forward (was: up to `topk` times). Also stashes
  `_last_router_logits` and `_last_expert_counts` per call so aux losses and the
  routing-bias updater can read them without re-running the router.
- `LTIInjection.get_A` — clamp tightened from `−20` to `−15`. At `−20`,
  `exp(-exp(-20)) ≈ 1 − 2e-9` rounds to exactly `1.0` in float32 and the
  ρ(A) < 1 stability guarantee silently fails under aggressive gradient steps.
- `RecurrentBlock` — added **gradient checkpointing** (training-only, no
  kv_cache), **convergence-detection early exit** (inference-only), and a
  `last_halt_step` telemetry tensor (used by `loop_efficiency` eval metric).
- **9 new `MythOuroConfig` fields**: `n_sink_tokens`, `gradient_checkpointing`,
  `convergence_eps`, `use_multiscale_injection`, `ms_window_size`,
  `use_cross_loop_attention`, `cross_loop_store_every`, `injection_decay`,
  `router_bias_lr`, `new_component_warmup_steps`.

### Part 2
- `LoRAAdapter` **v2** — per-loop B matrix (was: shared B with per-loop scale).
  `B` zero-init so the adapter starts as identity perturbation.
- `InjectionScheduler` — per-loop scalar magnitude with cosine init
  `1.0 → 0.1`. Lives inside the new `LTIInjection`.
- `LTIInjection` rewritten — `B_dir` (direction) × scheduler magnitude.
- `MultiScaleInjection` — blends fine / coarse / global views of `e` with
  learned per-loop weights.
- `CrossLoopAttention` — lightweight multi-head attention over a buffer of past
  loop hidden states. `o_proj` zero-init so it starts as identity residual.
- `RecurrentBlock` wires all of the above. Part 2 features default **on** in
  `MythOuroConfig`.

---

## Training utilities

All in [`mythouro/training_utils.py`](mythouro/training_utils.py).

### Part 1
- `combined_loss(model, logits, uncertainty, targets, vocab_size, topk, ...)` —
  CE + load_balance + uncertainty_calibration. Returns `(total, metrics_dict)`.
- `load_balance_loss`, `uncertainty_calibration_loss`, `consistency_loss`
  (KL self-distillation across loop depths).
- `LoopCurriculum` — linear ramp of `n_loops` from start to max.
- `MixedDataset` — streams 4 HF corpora at **40% FineWebEdu / 30% the-stack-smol
  / 20% open-web-math / 10% no_robots**.
- `log_spectral_radius`, `collect_router_logits`.

### Part 2
- `contrastive_loop_loss` — discriminates easy vs hard tokens by hidden-state
  movement between shallow and deep forward passes.
- `ProcessRewardHead` + `process_reward_loss` — small MLP predicting "answer
  correct?".
- `LoopDepthAnnealer` — pushes `n_loops` beyond `cfg.max_loop_iters` in the
  final ~15% of training.
- `sparse_activation_loss` — **per-token entropy** of router probabilities
  (was buggy L1; see [Bugs caught](#bugs-caught--fixed)).
- `ExpertSpecializationProbe` + `get_domain_labels`.
- `build_fsdp_model` — HYBRID_SHARD FSDP for NVLink-paired clusters.

### Added during §5
- `collect_expert_counts(model)` — snapshot `_last_expert_counts` per MoE layer.
- `update_router_bias_from_counts(model, counts_by_layer, bias_lr, ddp)` —
  DeepSeek-V3 nudge `sign(target − count) × bias_lr`. DDP-aware all-reduce.
- `log_expert_utilization(stats, step)` — CV / min% / max% / bias L2 diagnostic.

### Added during §3
- `_collect_new_component_param_ids(model)` — IDs of risk-prone Part 1+2 params.
- `get_optimizer_groups(model, base_lr, weight_decay, extra_base_params)` —
  returns two named groups: `base` and `new_component`.
- `ComponentWarmup(warmup_steps)` — stateless linear ramp 0→1 with `.factor()`.
- `apply_component_warmup(opt, base_lr, step, warmup_steps)` — per-step LR
  mutator. Drop-in replacement for the existing
  `for g in opt.param_groups: g["lr"] = lr` line.

---

## Inference utilities

All in [`mythouro/inference.py`](mythouro/inference.py).

### Part 1
- `UncertaintyGatedGenerator` — cheap-loops first, redo at `max_loops` if the
  uncertainty head flags the last position as uncertain.
- `SpeculativeDecoder` — single-model speculative decoding (shallow drafts,
  deep verifies). No separate draft model required.
- `CrossLoopKVCache` + `compress_kv_cache` — post-forward late-loop cache
  merger. ~17% memory reduction observed at `share_after=2`.
- `ComponentGradNormLogger` — per-block grad-norm diagnostic split into
  prelude / recurrent / coda / head / uncertainty / sink / embed / norm.

### Part 2
- `ContinuousDepthwiseBatcher` — per-sequence early-exit batching. Halted rows
  drop out of subsequent loop iterations.
- `RetrievalAugmentedInjector` — inject a retrieved-doc embedding at specific
  loop iterations. Pluggable `retriever(query) → list[str]` callable.
- `CoTDistillationTrainer` — distil explicit CoT traces into latent loop
  states (loss helper only; user supplies the dataset).
- `ActivationOffloader` — move Prelude/Coda activations to CPU during forward.
- `apply_int8_quantization` + `quantization_aware_training_hooks` — INT8
  dynamic quantisation post-training, skipping stability-critical params.

---

## Training script integration

[`training/3b_fine_web_edu.py`](training/3b_fine_web_edu.py) wires everything:

- `MixedDataset` replaces `FineWebEduDataset` as the default loader.
- HYBRID_SHARD FSDP via `build_fsdp_model`.
- `LoopCurriculum` (first half) + `LoopDepthAnnealer` (final ~15%) drive
  `n_loops` per step.
- `combined_loss` + `sparse_activation_loss` every step.
- `consistency_loss` every 25 steps, `contrastive_loop_loss` every 50.
- `process_reward_loss` every 20 steps.
- `ExpertSpecializationProbe.loss` every 50 steps.
- `update_router_bias_from_counts` every macro-step (after `optimizer.step()`).
- `log_spectral_radius` every 500 steps.
- `log_expert_utilization` (CV / min% / max% / bias L2) every 100 steps.
- Two-group AdamW via `get_optimizer_groups`; per-step LR via
  `apply_component_warmup`.
- ShutdownHandler installed before the loop; cooperative save-on-interrupt.
- CLI flags: `--eval`, `--eval-every`, `--eval-max-samples`, `--eval-benchmarks`.
- Log line shows `ce / lb / unc / sparse / cons / cont / prm / esp / n_loops /
  wfac (warmup factor) / lr / gnorm / tok/s`.

---

## Eval harness (§7)

New [`eval/`](eval/) package — orchestrator + 5 metrics + CLI.

- [`eval/harness.py`](eval/harness.py) — `run_eval()` + `python -m eval.harness`.
- [`eval/metrics.py`](eval/metrics.py) — 5 metrics:
  - `perplexity` (FineWebEdu stream)
  - `arc_challenge` (cloze log-likelihood)
  - `gsm8k` (greedy generate + `#### N` regex)
  - `loop_efficiency` (reads `RecurrentBlock.last_halt_step`)
  - `expected_calibration_error` (10-bin ECE on the uncertainty head)
- HumanEval deliberately skipped (sandboxed code execution out of v1 scope).
- Training script's `--eval` flag runs the harness every N steps on rank 0
  and dumps JSON to `eval_results/step_NNNNNNN.json`.

---

## Aux-loss-free MoE routing (§5)

**Bug fix**: `MoEFFN.router_bias` was a zero buffer that nothing updated. The
"aux-loss-free DeepSeek-V3 routing" claim in the comments was fiction.

- `MoEFFN.forward` stashes `_last_expert_counts` (a `(n_experts,)` tensor from
  the topk decision).
- `update_router_bias_from_counts` applies the DeepSeek-V3 nudge
  `sign(target − count) × router_bias_lr` outside the optimizer (it's a
  non-gradient buffer). All-reduces counts across ranks under DDP/FSDP.
- `MythOuroConfig.router_bias_lr = 1e-3`.
- Training loop accumulates counts across grad-accum micro-steps, applies the
  update post-`optimizer.step()`, logs MoE utilisation every 100 steps.
- **5 new tests** pin: counts shape correctness, collector roundtrip, bias
  direction (underused → +, overused → −), no-crash on zero counts, end-to-end
  drift over training steps.

**Verification**: smoke run showed bias L2 climb `0 → 0.75` over 5 steps; max
expert utilisation drop `50% → 47.5%`.

---

## Checkpoint robustness (§4)

New [`mythouro/checkpointing.py`](mythouro/checkpointing.py) — extracted
from training script so tests don't drag in `datasets`/pandas (which segfaults
on Python 3.14 + Windows).

**Schema v2** adds:
- `checkpoint_version` (int) — bumped on incompatible schema changes.
- `cfg_dict` — canonical for resume-time compatibility checks.
- `rng_state` — torch CPU + CUDA + Python + numpy. Single-process only; under
  DDP/FSDP each rank has its own state.
- `scaler_state` — `torch.amp.GradScaler` for fp16 paths (None on bf16).
- `extra` — free-form dict for arbitrary side-state.

**Compatibility guard**: `_SHAPE_FIELDS` allow-list. Shape-affecting fields
(`dim`, `n_heads`, `n_experts`, ...) must match exactly between save and load;
LR / dropout / ratios / loss coefficients may change between stages (logged,
not blocked). This is exactly the contract staged training needs.

**`ShutdownHandler`** installs SIGINT + SIGTERM + (Windows) SIGBREAK. First
signal sets `requested=True`; loop flushes a checkpoint at the next iteration
boundary; second signal force-exits.

**11 new tests** in [`tests/test_checkpoint.py`](tests/test_checkpoint.py):
roundtrip preserves model/optimizer/router_bias buffer, version mismatch
raises, override allows resume, shape-incompatible cfg raises, benign drift
allowed, RNG state restored, shutdown flag behavior.

**Verification**: end-to-end interrupt+resume smoke confirmed param-exact
restore (46.79 → 0.000000 L1 diff between fresh model and resumed model).

---

## Data pipeline

New [`data/`](data/) package — CPU-only preprocessing.

- [`data/dedup.py`](data/dedup.py) — MinHash LSH near-dup removal via
  `datasketch`. Defaults match Llama-3 / DeepSeek convention (char-5-gram
  shingles, 128 perms, 0.8 Jaccard).
- [`data/contamination.py`](data/contamination.py) — verbatim 13-word-gram
  match against ARC-Challenge / GSM8K / HumanEval test prompts. Long enough
  to avoid false positives, short enough to catch verbatim leakage.
- [`data/tokenizer_eval.py`](data/tokenizer_eval.py) — comparative compression
  analysis across HF tokenizers on a 4-domain weighted mix matching
  `MixedDataset` ratios. Built-in offline samples + `samples_from_hf` for
  real-corpus eval.
- [`data/__main__.py`](data/__main__.py) — `python -m data {dedup |
  contamination | tokenizer-eval}` dispatcher.
- **18 tests** in [`tests/test_data.py`](tests/test_data.py).

**Smoke result on GPT-2 tokenizer**: 3.67 weighted chars/token across the
40/30/20/10 mix. Llama-3 / Qwen-2.5 will likely score 4.5–5.0 on the same mix
— a ~20–30% throughput tax that's worth measuring before locking the
tokenizer for staged training.

---

## Flash-Attention cascade (§1)

`mythouro/main.py` gained a `CAPABILITIES` singleton that probes
`has_flash_attn_import`, `has_sdpa`, and `cuda_cc` once at module load.

- `fa2_usable` requires CC ≥ 8.0 — this is the bug §1 was filed to fix.
  Flash-attn imports cleanly on Volta (V100, CC 7.0) and Turing (CC 7.5) but
  its kernels crash at launch. The CC gate refuses to dispatch.
- **GQA cascade**: `FA2 → SDPA → manual`. SDPA branch uses `enable_gqa=True`
  on torch ≥ 2.3, falls through to `repeat_interleave` on older.
- **MLA cascade**: `SDPA → manual`. FA2 skipped — MLA's nope+rope key
  concatenation doesn't map cleanly onto FA2's API.
- `warn_once(key, msg)` deduplicates per process so the cascade is audible at
  startup without spamming every forward.

**9 tests** in [`tests/test_attention_fallback.py`](tests/test_attention_fallback.py)
pin CC enforcement (Volta/Turing refused, Ampere/Blackwell allowed),
warn-once dedup, and **GQA + MLA SDPA ↔ manual numerical equivalence** (the
critical contract: a checkpoint trained on one path must compute the same
outputs on the other).

**On user's actual machine** (Blackwell, CC 12.0, no flash-attn installed):
cascade picks SDPA — fast path on Blackwell since flash-attn hasn't reliably
supported SM 10.x yet.

---

## New-component LR warmup (§3)

**Re-scoped from AGENT_TASKS.md.** Original list included zero-output-init
components that self-warm anyway (`CrossLoopAttention.o_proj`,
`UncertaintyHead`, `ProcessRewardHead`). Actual risk surface narrowed to three
genuinely-active-at-step-0 components:

- `InjectionScheduler.log_scale` — cosine-init, immediately scales `B(t)·e`.
- `LoRAAdapter.down` — `std=0.02` init; gradient leaks into the recurrent
  block before `B` is trained up.
- `MultiScaleInjection.*` — `std=0.02` projections + non-uniform blend.

Mechanism: two-group AdamW (`base` + `new_component`), with the
`new_component` group's LR multiplied by `factor(step) = min(step / warmup_steps, 1.0)`
each step. `MythOuroConfig.new_component_warmup_steps = 2000`. Warmup factor
appears in the training log line as `wfac 0.50`.

**23 tests** in [`tests/test_component_warmup.py`](tests/test_component_warmup.py)
pin membership invariants (the right params are IN, the rest are OUT),
optimizer-group partition correctness, factor curve correctness, and per-step
LR mutation behavior.

---

## Test coverage (§12)

**+50 new tests across two files**:

- [`tests/test_training.py`](tests/test_training.py) — 33 tests covering every
  loss helper (combined / consistency / contrastive / load_balance /
  uncertainty calibration / sparse_activation / process_reward), the
  curriculum and annealer schedules, the MoE collectors, the ExpertSpecializationProbe,
  the domain-labels heuristic, and the spectral-radius diagnostic.
- [`tests/test_inference.py`](tests/test_inference.py) — 17 tests covering
  `UncertaintyGatedGenerator`, `SpeculativeDecoder`,
  `ContinuousDepthwiseBatcher`, `CrossLoopKVCache` + `compress_kv_cache`,
  `ComponentGradNormLogger`.

**Test suite total**: **221/221 pass** (started at 70 in `tests/test_main.py`).

---

## Packaging (§11)

**Re-scoped**. Original spec wanted extras (`bnb` / `gptq` / `awq` / `gguf`
/ `vllm`) and entry points (`mythouro-serve`, `mythouro-export`) for
integration code that doesn't exist. Shipping those would mislead users.

[`pyproject.toml`](pyproject.toml) ships only what's real:

| Extra | Adds | Use case |
|---|---|---|
| `flash` | `flash-attn ≥ 2.8.3` | Faster GQA on Ampere+; cascade falls back gracefully |
| `data`  | `datasketch`         | MinHash LSH dedup |
| `train` | `wandb`              | Experiment tracking slot |
| `all`   | everything above     | Convenience |

`loguru` promoted from extras to **core** (used in every module).

**Console scripts** (5):
- `mythouro-train` / `mythouro-train-1b` / `mythouro-train-tiny`
- `mythouro-eval`
- `mythouro-data` (dispatcher)

[`training/`](training/) is now a package ([`__init__.py`](training/__init__.py))
with a [`training/cli.py`](training/cli.py) `runpy` shim — the script
filenames start with digits and aren't directly importable as Python modules.

[`README.md`](README.md) updated with the install-extras table, an **honest**
hardware-tier table (consumer / mid-range / prosumer / server / CPU-only),
and a console-scripts section.

---

## Bugs caught + fixed

1. **`LTIInjection.get_A` clamp boundary** (Part 1). Pre-existing flaky test:
   at `clamp(-20, 20)`, `exp(-exp(-20)) ≈ 1 − 2e-9` rounds to exactly `1.0`
   in float32 and the ρ(A) < 1 stability guarantee silently breaks. Tightened
   to `clamp(-15, 20)` so `exp(-exp(-15)) ≈ 1 − 3e-7` stays representable.

2. **`MoEFFN.router_bias` dead code** (§5). The buffer existed but nothing
   updated it; the comment claiming "aux-loss-free routing" was fiction.
   Now driven by `update_router_bias_from_counts` every macro-step.

3. **`sparse_activation_loss` no-op** (§12). The implementation was
   `coeff * probs.abs().mean()`. For a softmax distribution the L1 norm is
   identically 1 and the L1 mean is identically `1/E`, regardless of routing
   pattern. The "loss" had zero gradient. Replaced with per-token entropy,
   which actually distinguishes uniform from peaked routing.

4. **Test bugs in `tests/test_main.py`** (Part 1B). Four `setup_method`s in
   GQA / MLA / TransformerBlock / RecurrentBlock test classes built freqs of
   length `max_seq_len=32` and passed them with input length `T=8`. `apply_rope`
   requires matched lengths; tests had never actually been run. Sliced freqs
   to `[:T]` so they match production usage.

---

## Deliberately deferred

The following AGENT_TASKS items are **not** implemented, with reasons:

| § | Item | Why deferred |
|---|---|---|
| §2 | Per-component grad clipping | Needs measurement first via `ComponentGradNormLogger` on a real run; the AGENT_TASKS clip values are guesses |
| §6 | Memorisation vs reasoning loss split | Premise unverified — needs eval-harness data to confirm looping hurts memorisation before adding complexity |
| §8 | Hardware auto-detect + 4 VRAM tiers | Premature until one config produces a usable model |
| §9 | Quantization export (GGUF / GPTQ / AWQ / BnB) | Over-scoped — each is a separate project. Pick one when needed (probably GGUF via llama-cpp-python) |
| §10 | Inference backend abstraction (PyTorch / llama.cpp / vLLM / HF) | Over-scoped — vLLM doesn't support custom architectures; llama.cpp needs C++ recurrent block reimpl |
| — | HumanEval in eval harness | Needs sandboxed code execution; out of v1 scope |

These are still useful eventually, but ordering them before there's an actual
training run that needs them inverts the priority.

---

## Next concrete steps

1. **Tokenizer evaluation** — `mythouro-data tokenizer-eval --use-hf-samples
   --tokenizers openai-community/gpt2 Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B
   deepseek-ai/DeepSeek-V2-Lite --output reports/tokenizer_eval.json`. Lock
   the tokenizer choice before any further architecture work.
2. **MoE-vs-dense ablation** (if you have GPU time to spare): train two
   identical configs at matched active-parameter count, let the eval harness
   pick the winner. Could eliminate ~60% of architectural complexity.
3. **Staged training** with the current setup once tokenizer is locked. §4
   makes long multi-stage runs interruption-safe.

---

# 2026-05-31 update — what landed since the original CHANGES.md

This block is appended in chronological order; everything above is unchanged.
**Repo total now 249/249 tests.**

## ConfidenceAwareGenerator (+14 tests)

New generator in [`mythouro/inference.py`](mythouro/inference.py) that
addresses the "right answer + rambles confidently while uncertain" failure
mode observed in deployed Ouro (and reasoning models in general).

Four stop reasons, checked in order each generated token:
- `eos` — exact match on `eos_token_id`; bypasses `min_new_tokens` floor.
  Default `None` (disabled) — caller must pass their tokenizer's EOS.
- `confidence` — sustained low `UncertaintyHead` output for
  `confidence_window` tokens AND the latest token is in `break_token_ids`.
  Default `break_token_ids=None` disables this entirely (fail-closed —
  the previous "any token is a break" default was a footgun).
- `cycle` — literal repeated n-gram (`cycle_min_len`) in the last
  `cycle_window` tokens. Detected by `_has_cycle` static method.
- `max_new_tokens` — hard cap.

Returns a dict with `sequences`, `stop_reason`, and `uncertainty_trace`
(per-generated-token uncertainty score) for tuning the threshold offline.

Enforces `B=1` with a clear assertion — the per-token break check
doesn't generalise to batched rows that want to stop at different
positions; deferred until a real batched-inference need arises.

## Config tightening per Ouro empirical evidence

Ouro (Zhu et al. 2025, 7.7T tokens) measured peak accuracy at 3–4 loops
with **measurable degradation past 8**. Three config changes:

- [`mythouro/main.py`](mythouro/main.py): `MythOuroConfig.max_loop_iters`
  default lowered `16 → 6` with inline citation.
- [`mythouro/variants.py`](mythouro/variants.py): all 7 variants
  updated. `mythouro_1b` / `mythouro_3b` → 6 (in Ouro's tested range);
  `mythouro_10b` / `mythouro_50b` → 8; `mythouro_100b`+ → 12 (cautious
  extrapolation, flagged as informed-guess beyond evidence).
- [`training/3b_fine_web_edu.py`](training/3b_fine_web_edu.py):
  `LoopDepthAnnealer.max_extra_loops` capped at `base + 2` (was
  `base + 8`). Don't extrapolate into the over-loop regime that
  actively hurts accuracy.

## Distillation pipeline (+14 tests)

New module surface for Hinton-style logit distillation from a frozen
teacher into an MythOuro student. Motivated by the "RL only surfaces
existing base-model capacity" finding — starting from a higher ceiling
matters more than chasing post-training tricks.

**[`mythouro/training_utils.py`](mythouro/training_utils.py)** (new helpers, appended):
- `distillation_loss(s, t, targets, *, temperature, alpha)` — T²-scaled
  KL + optional CE blend. fp32 KL for stability under bf16. Refuses
  mismatched shapes with explicit "tokenisers are misaligned" hint.
- `load_distillation_teacher(model_id, student_vocab_size, *, device,
  dtype, trust_remote_code)` — frozen `AutoModelForCausalLM` wrapper.
  Enforces vocab alignment + freezes all params + moves to device/dtype.
  Returns `None` on load failure or mismatch (caller falls back to CE).
- `teacher_logits(teacher, input_ids)` — no-grad forward; handles both
  `CausalLMOutputWithPast` and bare-tensor returns. Passes
  `use_cache=False, past_key_values=None` to dodge Ouro's
  `modeling_ouro.py:get_mask_sizes` int-vs-tensor bug.

**[`training/distill.py`](training/distill.py)** — full distillation training
script. Reuses MixedDataset, LoopCurriculum, optimizer-groups +
warmup, MoE bias updater, checkpointing, ShutdownHandler. Loss:
`α·distill + (1−α)·CE + λ_lb·load_balance + λ_unc·uncertainty_cal + λ_sparse·sparse_activation`.

CLI: `mythouro-distill` (registered in `training/cli.py` and
`pyproject.toml`).

## Ouro alignment (real teacher verified)

End-to-end verified against `ByteDance/Ouro-2.6B-Thinking` on the
user's machine. Specific values:
- vocab_size = 49152 (custom ByteDance BPE)
- tokenizer = `ByteDance/Ouro-2.6B-Thinking` (GPT2Tokenizer class)
- Requires `trust_remote_code=True` (ships custom `modeling_ouro.py`)
- Special tokens: `<|endoftext|>` (0), `<|im_start|>` (1, BOS),
  `<|im_end|>` (2, EOS), `<think>` (3), `</think>` (4), `<file_sep>` (5)
- Chat template: Qwen-style ChatML with optional `<think>` blocks
- Architecture: dense (NOT MoE), 48 layers × `total_ut_steps=4`
  recurrent passes, 2048 hidden, 16 heads (no GQA), bf16 = ~5.2 GB

**`MythOuroTokenizer` rewrite** ([`mythouro/tokenizer.py`](mythouro/tokenizer.py)):
- Default changed: `EleutherAI/gpt-neo-125m` / `openai/gpt-oss-20b` → `ByteDance/Ouro-2.6B-Thinking`
- New properties: `bos_token_id`, `eos_token_id`, `pad_token_id`
- New method: `apply_chat_template(messages, *, enable_thinking=False, ...)`
  — supports Ouro's `<think>` block via the flag, falls back with a
  clear error on tokenizers that lack a chat template
- `encode` now defaults `add_special_tokens=False` (matches the
  training-pipeline packing convention)

**`mythouro_distill_tiny` variant** ([`mythouro/variants.py`](mythouro/variants.py)):
- 279M params, 0.56 GB bf16 weights
- vocab_size=49152 (Ouro-aligned, non-negotiable)
- dim=1280, GQA 16/4, 2+2 prelude/coda
- MoE: 24 routed + 2 shared experts, expert_dim=1280
- `max_loop_iters=4` matches Ouro's `total_ut_steps=4`
- Sized to cohabit with the 5.2 GB bf16 Ouro teacher on a 12 GB GPU
  (5070): teacher + student weights + grads + AdamW + activations ≈ 9 GB

**`distill.py` CLI defaults now Ouro-first**: `--teacher-id` and
`--tokenizer` default to `ByteDance/Ouro-2.6B-Thinking`;
`--student-variant` defaults to `mythouro_distill_tiny`; `--trust-remote-code`
helptext flagged as required for the default teacher.

The zero-config invocation is now:
```bash
mythouro-distill --trust-remote-code --total-steps 200 \
    --eval --eval-every 100
```

## Bugs caught + fixed in this update

1. **`ConfidenceAwareGenerator` API footguns (now fixed)**:
   - `eos_token_id=2` default — token id 2 is arbitrary; not any real
     tokenizer's EOS. Now defaults to `None` (disabled until caller
     opts in).
   - `break_token_ids=[]` empty meant "any token is a break" — made
     confidence stops trivially-firing. Now defaults to `None`
     (disabled until caller opts in with explicit ids).
   - `unc[:, -1].mean()` silently averaged across batch rows. Now
     `assert input_ids.shape[0] == 1` with a clear error.

2. **Ouro `modeling_ouro.py:get_mask_sizes` int-vs-tensor crash** —
   `cache_position` is an `int` when called outside `model.generate`,
   but the method tries `.shape` on it. Worked around in our
   `teacher_logits` wrapper by passing `use_cache=False,
   past_key_values=None`.

## User's hardware reference (3 GPUs, mixed, no NVLink)

5070 12 GB Blackwell (CC 12.0), 5060 8 GB Blackwell, 4060 8 GB Ada
(CC 8.9). FSDP across mixed generations is painful; recommended
distillation layouts:

| Layout | Teacher | Student | When |
|---|---|---|---|
| **Single-card** (default) | 5070 cuda | 5070 cuda | Both fit (~9 GB total for distill_tiny + teacher). Other cards idle. |
| **Two-card relaxed** | 5070 | 5060 / 4060 | Frees 5070 for bigger student; pays PCIe latency for teacher forward. |
| **CPU teacher** | CPU | any GPU | Fits any student size on any card. Throughput drops noticeably. |

## Depth regulariser (PonderNet × Ouro, +15 tests)

Layered on top of the existing Graves-style ACT halting (unchanged
inference behaviour) rather than replacing it. Captures the per-loop
halt probabilities as they're computed and post-processes into a proper
PonderNet-style distribution; regularises that distribution toward a
uniform prior via KL.

**[`mythouro/main.py`](mythouro/main.py)** — additions to
`RecurrentBlock.forward`:
- New `loop_halt_probs: list[Tensor]` accumulates each loop's λ_t (the
  per-token sigmoid output of `ACTHalting`). Lambdas stay in the
  autograd graph — the regulariser drives ACTHalting.
- After the loop completes, post-processes via PonderNet formula:
  `P(halt at n) = λ_n · ∏_{i<n}(1 − λ_i)`, with the last step absorbing
  any residual mass so each (B, T) row sums to 1.
- Exposed as `self.last_halt_distribution` (B, T, K) where K is the
  number of loops that *actually* ran (can be less than `n_loops` when
  ACT cumulative-threshold halt or convergence-detection fires —
  pinned by `test_K_can_be_less_than_n_loops_when_ACT_short_circuits`).
- New `cfg.depth_reg_coeff` field (default `0.0` = off). Recommended
  `1e-3` to `1e-2` when enabling.

**[`mythouro/training_utils.py`](mythouro/training_utils.py)**:
- `collect_halt_distributions(model)` — walks `model.modules()` for
  RecurrentBlocks and gathers their `last_halt_distribution` tensors.
- `depth_regularization_loss(model, *, prior="uniform", coeff=1e-2, eps=1e-12)` —
  computes KL(P || uniform) per token, averages over (B, T) and across
  RecurrentBlocks. Only `prior="uniform"` is implemented; `prior="geometric"`
  raises `NotImplementedError` with a citation of Ouro's empirical evidence
  against the geometric prior (it under-trains late loops). Returns
  `torch.tensor(0.0)` when no distribution exists yet (no forward done).
- `combined_loss` extended with `depth_reg_coeff: float = 0.0` parameter.
  When 0 (default), the collection walk + KL math are skipped entirely
  so the default `combined_loss` cost is unchanged. When > 0, the
  regulariser is added to the total and a `"depth"` metric is reported.
  The `"depth"` key is always present in metrics for log-line stability.

**[`tests/test_depth_regulariser.py`](tests/test_depth_regulariser.py)** —
15 tests covering:
- Halt-distribution shape, normalisation (sums to 1), non-negativity.
- Variable-K contract (K can be < n_loops when ACT short-circuits).
- KL math: 0 when uniform, exactly `log(K)` for a one-hot distribution,
  linear in `coeff`.
- Gradient flow: confirms `depth_regularization_loss` actually
  back-propagates into `ACTHalting.halt.weight`.
- Only "uniform" prior accepted; "geometric" raises.
- `combined_loss` integration: depth metric always present; total loss
  unchanged when `depth_reg_coeff=0`; total rises by exactly
  `coeff × depth` when enabled.

**Design choice (worth noting for future contributors)**: this is a
*parallel* tracker. The halt CRITERION is still Graves cumulative-
threshold (unchanged inference behaviour); the lambdas drive both the
existing ACT mechanism AND a new regularisation signal. We did NOT swap
the criterion to PonderNet's pure-distribution sampling because:
1. The existing ACT criterion has been working correctly through 264
   tests and three rounds of integration; replacing it for the
   regulariser's sake risks correctness regressions elsewhere.
2. The two semantics are mathematically compatible — both use the same
   per-step λ values; PonderNet just sums them differently downstream.
3. The user can run with depth_reg_coeff > 0 to get Ouro's
   training-time depth-shaping behaviour without touching the inference
   path, which is exactly what we want when distilling from Ouro
   (inherit Ouro's halt distribution via the soft-label loss; nudge our
   own halt distribution toward uniform with the regulariser).

If full PonderNet semantics are wanted later (criterion AND
regularisation), the swap is a separate ~80-line change in `RecurrentBlock`
that keeps `last_halt_distribution` intact and just changes how `h_out`
is computed (weighted sum by P(halt) instead of cumulative-threshold).

---

# 2026-06-08 update — best-of-trajectory emission

Inference-side experiment to extract more from the existing depth machinery
without retraining. Default-off; the normal `forward`/`generate` path is
byte-for-byte unchanged.

**Motivation.** Standard decoding emits the recurrent block's ACT-weighted blend
over loops, and the existing `UncertaintyGatedGenerator` only ever loops *more*
when uncertain. Neither can emit an *earlier* loop's prediction when that loop
was the most confident — yet extra loops can legitimately raise entropy on hard
tokens before resolving them. Best-of-trajectory keeps the lowest-uncertainty
step instead of running extra loops and trying to undo a bad one.

**[`mythouro/main.py`](mythouro/main.py)**:
- `RecurrentBlock` — opt-in `collect_trajectory` flag stashes the per-loop
  committed hidden states into `last_trajectory` (B, T, K, D). Active only at
  inference (no kv_cache, not training); zero cost otherwise.
- `MythOuro.forward_trajectory(input_ids, n_loops)` — runs each captured loop
  state through Coda + LM head + UncertaintyHead and returns
  `(logits_traj (B,T,K,V), unc_traj (B,T,K))`, where K is the number of loops
  actually run (≤ n_loops; fewer if convergence early-exit fires). Full
  recompute, no KV cache — O(K) Codas, an inspector/experiment path.

**[`mythouro/inference.py`](mythouro/inference.py)**:
- `BestOfTrajectoryGenerator` + `best_of_trajectory_generate` — B=1 decode that
  selects the argmin-uncertainty depth per token. Has a `min_loops` floor
  (excludes shallow depths from selection when the head is miscalibrated early)
  and returns a `chosen_loops` telemetry trace so you can see whether selection
  diverges from "always deepest".

**[`tests/test_inference.py`](tests/test_inference.py)** — +8 tests
(`TestBestOfTrajectory`): forward_trajectory shapes + valid probabilities, no
state leakage after the call, single-step at `n_loops=1`, generation
length/prefix, argmin-selection contract (deterministic match against
`forward_trajectory`), `min_loops` floor, EOS wiring, B=1 assertion. Full
inference suite 31 → 39 tests, all passing.

**Status.** Validation-ready against v4/v5. It's a *measurement* tool first —
the param-count gibberish ceiling may mask the effect until the model is larger.
The learned generalisation of this (a trained depth policy unified with expert
routing) is tracked as **MoDr** in [`docs/roadmap.md`](docs/roadmap.md), gated
behind the MoE-vs-dense ablation.

### Forced-depth probe (`--force-full-depth`)

`forward_trajectory(..., force_full_depth=True)` suppresses ACT's
convergence + halt-all early-exit during trajectory capture (via a
`RecurrentBlock.force_full_depth` measurement flag) so the loop runs the full
`n_loops` instead of stopping where ACT chose. This exposes the counterfactual
loops ACT skips — letting us tell "deeper loops genuinely hurt" from "deeper
loops never ran". Threaded through `BestOfTrajectoryGenerator(force_full_depth=)`
and an `inspect_checkpoint.py --force-full-depth` switch that prints an `[A/B]`
verdict comparing ACT's learned halt depth to the forced-depth uncertainty
minimum. Pure measurement — no weight change, normal forward/generate untouched.
+3 tests (forced K == n_loops, extrapolation past trained depth, generator
contract). Also fixed: `inspect_checkpoint.py` now forces UTF-8 stdout so
redirecting output doesn't crash on exotic tokens in the model's (gibberish)
generations on a cp1252 Windows console.

**Result (v4/v5, `reports/inspect_v{4,5}_forced*.txt`).** Whether ACT halts too
early is **prompt-dependent** — on some prompts the skipped loops *do* lower
uncertainty (ACT too early), on others they don't (ACT justified); and on v5 at
`n_loops=8` one prompt reaches its global uncertainty minimum at **loop 7** (2×
the trained depth), partial evidence for depth-extrapolation. Full analysis +
the "single global halt threshold is structurally wrong -> motivates MoDr"
takeaway are in [`docs/roadmap.md`](docs/roadmap.md).

### MoE-vs-dense ablation: `recurrent_dense` flag + variant

Wires up the gating experiment that decides whether the recurrent MoE earns its
complexity (and gates MoDr). See the full spec / protocol / decision rule in
[`docs/roadmap.md`](docs/roadmap.md).

- **[`mythouro/main.py`](mythouro/main.py)** — `MythOuroConfig` gains
  `recurrent_dense` + `recurrent_dense_ffn_dim`. When `recurrent_dense=True`,
  `RecurrentBlock` builds a dense `Expert(dim, d_ff)` recurrent FFN instead of
  `MoEFFN`, with auto width `expert_dim · n_experts_per_tok · (1 + n_shared)` —
  sized so the dense FFN's params/FLOPs per token equal the MoE arm's *activated*
  FFN per token (matched compute). `TransformerBlock` gained a `dense_ffn_dim`
  override to support this.
- **[`mythouro/variants.py`](mythouro/variants.py)** — `mythouro_distill_tiny_dense()`,
  a `dataclasses.replace` of `distill_tiny` differing *only* in the two dense
  fields (provably identical otherwise). Registered in `mythouro/__init__.py`
  and both training CLIs (`training/sft.py`, `training/distill.py`).
- **[`tests/test_dense_ablation.py`](tests/test_dense_ablation.py)** — 8 tests:
  flag swaps MoEFFN→Expert, explicit/auto width, the matched-active invariant
  (`dense_FFN_params == MoE_active_FFN_params`, exact), dense forward+backward,
  the MoE-aux helpers no-op on a dense model, and the variant differs from
  `distill_tiny` only in the FFN fields.

Verified on the real variant: MoE arm **278.9 M** total / dense arm **180.5 M**
(98 M idle routed-expert capacity removed), dense recurrent FFN width **15360**.
The MoE-only aux losses already short-circuit to 0 on a model with no MoE layers,
so the dense arm runs through the existing training scripts unchanged.

---

# 2026-06-09/10 update — external code review, fixes, and the first run on fixed code

The largest correctness pass since the original build. An external review
(Claude Fable 5, full text: [`docs/mythouro_code_review_findings.md`](docs/mythouro_code_review_findings.md))
found 5 correctness bugs (P0), 9 perf/measurement issues (P1), and 7 strategic
items (P2). Every finding was independently verified against the code before
acting. Status tracker: [`docs/review_action_plan.md`](docs/review_action_plan.md).
Cross-run stats: [`docs/training_runs.md`](docs/training_runs.md).

## The five correctness bugs (what was wrong, what we did, why it matters)

1. **P0.1 — `_init_weights` clobbered deliberate zero-inits.** The blanket
   N(0,0.02) re-init ran *after* submodules set their zero-inits, so
   `CrossLoopAttention.o_proj` (meant to start as an identity residual) and
   `UncertaintyHead.net[-1]` (meant to start at neutral 0.5) were random — i.e.
   **v1–v5 trained their entire runs with noise injected into the hidden state
   every loop.** Fix: protected layers mark `_skip_global_init`. This is the
   prime suspect for why the post-fix run (below) trains so much better.
2. **P0.2 — MoE router telemetry only saw the LAST loop.** The single recurrent
   MoEFFN runs n_loops× per forward but its telemetry was overwritten each
   call, so the aux losses and the DeepSeek-V3 bias updater only balanced loop
   K−1 — and the v5 "expert-ceiling" cv numbers were measured through this
   (conclusion softened in the roadmap). Fix: telemetry returned through the
   gradient-checkpoint boundary and accumulated across all loops.
3. **P0.3 — eval emitted a never-trained path.** Training returns `h_K`; eval
   returned the ACT blend `h_out`, which under-summed for non-halting positions
   AND was never seen by the coda/head during training. Fix: emit `h_K`
   everywhere; ACT is purely an early-exit criterion. **Re-baseline on the same
   v2 weights: PPL 46.3→39.25, ECE 0.058→0.042 — free measured capability.**
   All archived eval numbers were pessimistic.
4. **P0.4 — depthwise batcher attended across the wrong rows.** Cross-loop
   buffer held active-subset snapshots; as rows halted, batch dims went ragged
   and row identities drifted (a sequence could attend to another sequence's
   loop history). Fix: full-batch snapshots, per-row slicing, h_K emission.
   Pinned by a per-row equivalence test vs single-row forwards.
5. **P0.5 — uncertainty head consumed on distributions it was never calibrated
   on.** Measured per-loop ECE (new tool `tools/per_loop_calibration.py`) on v2
   and v4: loops 1–3 calibrated (0.01–0.04), **loop 0 badly miscalibrated
   (0.17–0.22, error understated ~0.2)** — the loop curriculum starts at 2, so
   loop 0 was never an emission loop. Consequences: best-of-trajectory defaults
   `min_loops=2`; **MoDr supervision target = per-loop CE (mandated)**; the
   earlier "model prefers loop 0" reads were partly a calibration artifact.

Plus: eval harness now rebuilds from the checkpoint's own cfg (was hardcoded to
`mythouro_1b`), defaults to the Ouro tokenizer (was wrong-vocab gpt-neo), and
clamps eval seq_len to `max_seq_len`.

## P1 fixes (perf / measurement)

Sorted MoE dispatch (one host sync instead of ~768/micro-step in the v5
regime); MLA caches compact shared rope keys (~n_heads× less rope cache);
multi-scale injection projections hoisted out of the loop; `generate` defaults
to the trained depth; zero-copy KV rewind in UncertaintyGatedGenerator;
SpeculativeDecoder loses two redundant full forwards per step; LoRA H2D
transfer and loop-embedding reallocation removed; dead ACT-blend accumulation
deleted; distill soft-KL masked to valid positions. Remaining P1 items (MLA
absorption, cached drafting, is_causal refactor) carry written specs in the
action plan and want GPU validation. `moda.py` (1,063-line unused upstream
duplicate) quarantined to `examples/` (P2.5).

## New invariant tests

The 303-green suite missed P0.1/P0.2 because they're invariant violations, not
logic errors. Added: zero-init survival, telemetry coverage, train/eval
emission parity, aux-gradient liveness under checkpointing, batcher per-row
equivalence, dispatch-vs-naive equivalence, per-loop-uncertainty contracts.
Suite now 313+ passing.

## Two operational incidents (both documented in roadmap failure modes)

1. **Script defaults ≠ proven recipe.** The first ablation attempt used
   `distill.py` defaults (warmup 200, depth-reg 0.1) — both arms flatlined
   identically (CE stuck ~7.6–8, transient deep-loop gnorm spikes to 2743,
   weights sane). Diagnosis chain (depth-sweep eval, init comparison, weight
   forensics, resume probe, MoE control arm, data decode) ended at v1's
   MODEL_CARD provenance command: **warmup 500, depth-reg 0.3, mb1/ga8**.
   Defaults now encode the proven recipe. Rule: *diff against the model card's
   command; never trust script defaults.*
2. **Eval filename collision.** `eval_results/distill_step_*.json` collide
   across runs — a stale v1 file masqueraded as a live regression. Convention:
   copy each run's eval JSONs into its checkpoint dir as sidecars; a per-run
   eval path is on the tooling list.

## First training run on the fixed code: ablation arm 1 (MoE, seed 0)

**Final PPL 5.72 vs v1's 37.4 — 6.5× better in 1,000 fewer steps** (trajectory
560 → 112 → 11.1 → 5.72; loop_eff 0.500 dead-center; ECE 0.015). Same
architecture, data, and teacher as v1 — the delta is the fixed code (P0.1/P0.2)
plus the proven recipe. Caveat: absolute PPL is flattered by FineWeb train/eval
stream overlap (applies equally to v1, so the relative gain stands). Dense arm
and seed-1 runs are wired and **user-gated**.

## Also this period

- **CUDA→XPU device abstraction** (`mythouro/device.py`) + real-valued RoPE
  fallback (`rope_real`) — `--device xpu` is turnkey for Intel Arc/Battlemage;
  NVIDIA path byte-identical. Plus `tools/bench_step.py` (achieved-tok/s
  benchmark) and the hardware analysis it powered (measured: 5070 = 6,852
  tok/s @ b8; Xeon-8480-on-2-channels = 373; conclusion: VRAM, not TFLOPS, is
  the local bottleneck).
- **Reconstructed MODEL_CARDs for v3/v4/v5** (from roadmap + checkpoint
  timestamps) and versioned all cards + eval JSONs (archived_models/ was
  accidentally blanket-gitignored).
- `--seed` / `--start-loops` flags in both training scripts; training-runs
  comparison doc; science+medical domain expansion added to the data roadmap.
