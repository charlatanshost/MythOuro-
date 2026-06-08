# MythOuro vs. OpenMythos — code-level comparison

A verified, file-by-file account of how this fork (**MythOuro**,
`charlatanshost/MythOuro-`) diverges from its upstream foundation
(**OpenMythos**, `kyegomez/OpenMythos`, MIT). Numbers below come from an actual
`diff` of the two trees, not a summary of intent.

> **Attribution.** The upstream architecture is Kye Gomez's work
> (`kyegomez/OpenMythos`, MIT) and is credited with thanks. He has no
> involvement in this fork beyond that foundation. The teacher
> (`ByteDance/Ouro-2.6B-Thinking`) is Apache 2.0. See the README
> "Acknowledgements" section.

## Verdict

Not a rebrand. `main.py` alone went from 1,085 → 1,911 lines with **1,076
changed/added lines** — only ~1% of the upstream file survives untouched. The
architecture *skeleton* (RMSNorm, RoPE, GQA/MLA attention, MoE, the
Prelude→Recurrent→Coda loop with LTI injection and ACT halting) is inherited;
nearly everything around it is new or reworked, and a full distillation → SFT →
model-growth training pipeline was built on top.

## 1. File-level diff (quantified)

| File | OpenMythos | MythOuro | Changed lines | Status |
|------|-----------:|---------:|--------------:|--------|
| `main.py` | 1,085 L / 12 classes | 1,911 L / **18 classes** | **1,076** | Near-total rewrite |
| `tokenizer.py` | 64 L | 155 L | 171 | Heavily rewritten (Ouro alignment) |
| `variants.py` | 198 L | 351 L | 211 | Heavily changed (distill variants) |
| `__init__.py` | 55 L | 61 L | 50 | Renamed exports + new variants |
| `moda.py` | 1,063 L | 1,063 L | **0** (line-endings only) | **Genuinely unchanged** |

`moda.py` is the only truly shared module — a self-contained DeepSeek-style MoE
implementation (`MoDAConfig`, `DeepSeekExpert`, `DeepSeekGate`, `DeepSeekMoE`)
carried over verbatim and **not wired into the main `MythOuro` model** (parallel
/ vestigial inherited code).

## 2. Package & layout

| | OpenMythos | MythOuro |
|---|---|---|
| Package | `open_mythos` (install `open-mythos`) | `mythouro` (install `mythouro`) |
| New core modules | — | `checkpointing.py`, `grow.py`, `inference.py`, `sft_data.py`, `training_utils.py` |
| `training/` | `3b_fine_web_edu.py` | + `distill.py`, `sft.py`, `cli.py`, `1b_fine_web_edu.py`, `__init__.py` |
| Real test modules | 3 (`test_main`, `test_tokenizer`, `test_rope_debug`) | 14 (added attention-fallback, checkpoint, component-warmup, data, depth-regulariser, distillation, grow, inference, sft-data, smoke-e2e, training) |
| New top-level dirs/files | — | `data/`, `eval/`, `tools/`, `eval_results/`, `inspect_checkpoint.py`, `launch_training.py`, `CHANGES.md` |

**pyproject:** name `open-mythos` → `mythouro`; `loguru` promoted from extra to
core dependency; extras `data` (datasketch) / `train` (wandb) / `all`; six
console scripts via `training.cli` (`mythouro-train`, `-train-1b`, `-train-tiny`,
`mythouro-distill`, `mythouro-eval`, `mythouro-data`).

## 3. Core model — net-new classes

Upstream's 12 classes are all present (renamed where relevant:
`MythosConfig`→`MythOuroConfig`, `OpenMythos`→`MythOuro`). MythOuro adds six:

| New class | Purpose |
|-----------|---------|
| `_Capabilities` | One-shot FA2/SDPA/compute-capability probe; drives the attention cascade (FA2 → SDPA → manual) with a CC ≥ 8.0 gate |
| `InjectionScheduler` | Per-loop injection magnitude (cosine / linear / constant) |
| `MultiScaleInjection` | Fine / coarse / global blend of the injected embedding |
| `CrossLoopAttention` | Attention over a buffer of prior loop hidden states |
| `AttentionSink` | Learnable register tokens, prepended on prefill, stripped before the LM head |
| `UncertaintyHead` | Per-token confidence-of-error predictor |

### Forward signature (breaking change)
- OpenMythos: `OpenMythos.forward(x) -> torch.Tensor` (logits only).
- MythOuro: `return logits, unc` — a `(logits, uncertainty)` tuple. Every call
  site (training, tests, examples, docs) was updated to unpack.

### Config changes
- `max_loop_iters` default **16 → 6** (per Ouro's finding that accuracy peaks at
  3–4 loops and degrades past 8).
- **+11 new fields**: `n_sink_tokens`, `gradient_checkpointing`,
  `convergence_eps`, `use_multiscale_injection`, `ms_window_size`,
  `use_cross_loop_attention`, `cross_loop_store_every`, `injection_decay`,
  `router_bias_lr`, `new_component_warmup_steps`, `depth_reg_coeff`.

## 4. Substantive (not just additive) change: the MoE router

The clearest example that the diff isn't all new scaffolding — some of it fixes
inherited no-ops:

- **OpenMythos**: `router_bias` is a zero buffer added to the logits, but
  **nothing ever updates it** — dead code; the "aux-loss-free routing" claim was
  aspirational.
- **MythOuro**: the router stashes `_last_router_logits` and
  `_last_expert_counts`, dispatch is vectorised via `index_add_` (each expert
  runs once per forward instead of up to top-k times), and an external
  DeepSeek-V3 updater (`router_bias_lr`) actually nudges the bias toward balanced
  utilisation. The dead buffer became a live load-balancer.

## 5. Pipeline & artifacts

| | OpenMythos | MythOuro |
|---|---|---|
| Training | Basic pretrain script (AdamW, FineWeb-Edu, DDP) | Distillation from Ouro-2.6B-Thinking (`distill.py`), masked-CE SFT (`sft.py` + `sft_data.py`), function-preserving MoE growth (`grow.py`) |
| Memory recipes | — | 8-bit AdamW (bnb, CUDA-version auto-detect), gradient checkpointing, staged seq-len resume |
| Eval | — | Harness with perplexity / ARC / GSM8K / loop_efficiency / ECE |
| Data tooling | — | MinHash dedup, contamination check, tokenizer eval (`data/`) |
| Extra heads | — | `UncertaintyHead` (main.py), `ProcessRewardHead` (training_utils.py) |
| Trained artifacts | None | 278M → 632M reference checkpoints with eval results + roadmap |

## 6. Documentation

- **OpenMythos**: theory-heavy (stability proofs, scaling-law discussion).
- **MythOuro**: retains the theory but prepends a "Project identity & lineage"
  section (fork acknowledgement, original contributions, honest small-scale
  limitations — "no coherent text yet at this scale"), plus practical
  hardware-tier tables, a changelog (`CHANGES.md`), a roadmap, and this
  comparison.

---

*Method: comparison generated by diffing a `--depth 1` clone of
`kyegomez/OpenMythos` against this tree. Reproduce with
`diff -r open_mythos/ mythouro/` after cloning upstream.*
