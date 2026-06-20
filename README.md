# MythOuro

<p align="left">
  <a href="https://pypi.org/project/mythouro/" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/pypi/v/mythouro?style=for-the-badge&color=3670A0" media="(prefers-color-scheme: dark)">
      <img alt="Version" src="https://img.shields.io/pypi/v/mythouro?style=for-the-badge&color=3670A0">
    </picture>
  </a>
  <a href="https://pytorch.org" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/badge/PyTorch-Implemented-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" media="(prefers-color-scheme: dark)">
      <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Implemented-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    </picture>
  </a>
</p>

> **Disclaimer:** MythOuro is an independent research project on recurrent-depth transformers, built on publicly available research and open-source components. It is not affiliated with, endorsed by, or connected to Anthropic or any of their proprietary systems.

MythOuro implements a Recurrent-Depth Transformer (RDT) with three stages: **Prelude** (transformer blocks), a looped **Recurrent Block** (up to `max_loop_iters`), and a final **Coda**. Attention is switchable between MLA and GQA, and the feed-forward uses a sparse MoE with routed and shared experts ideal for exploring compute-adaptive, depth-variable reasoning.

---

## Project identity & lineage

> **MythOuro is a research project by Daniel Hardy on recurrent-depth
> transformers (RDTs)** — their training dynamics, distillation efficiency, and
> calibrated honesty — with the applied goal of a small, private, **local-first
> model for medical information** (to help lower the cost of and improve access
> to medical information for people underserved by the current system). It began
> as a fork of Kye Gomez's **OpenMythos** (credited below) and is distilled from
> **ByteDance Ouro-2.6B** (the teacher), but has diverged into its own trained
> pipeline and research program. The name is **Myth + Ouro** — the *Ouro* (the
> recurrent loop / the Ouro teacher) is the half that carries the identity;
> OpenMythos and the "Mythos" origin are **credited lineage, not the focus.**

It is a **custom recurrent-depth Mixture-of-Experts language model** — a
hybrid that draws on three distinct lineages but is identical to none of them:

| Draws from | What was taken | How this project differs |
|---|---|---|
| **OpenMythos** (`kyegomez/OpenMythos`) — architecture | The RDT hypothesis — Prelude/Recurrent/Coda, MoE, MLA/GQA, LTI-stable injection, ACT halting | Extended into a full distillation → SFT → model-growth pipeline with uncertainty-aware generation and function-preserving MoE expansion; produced *trained checkpoints*, not just a reference implementation. (This fork renamed the project to **MythOuro** — Myth + Ouro.) |
| **ByteDance Ouro-2.6B-Thinking** (teacher) | Used as the frozen distillation teacher; vocab/tokenizer alignment | Not Ouro — Ouro is a teacher *signal*, not the architecture. The student is a different, MoE-based design grown on consumer hardware |
| **Anthropic "Claude Mythos"** (inspiration only) | The original *hypothesis* that motivated the architecture | Purely speculative inspiration — no affiliation, no weights, nothing proprietary (see disclaimer above) |

**What's genuinely original here** (not inherited from any of the above):
- The distillation → SFT → MoE-growth training pipeline and its tooling
- Function-preserving **MoE expansion** (sentinel-bias promotion) on a recurrent-MoE architecture — [`docs/growth_design.md`](docs/growth_design.md)
- The trained reference checkpoints (278M v1 → 632M v5, plus the post-fix ablation runs) and their end-to-end validation — [`docs/training_runs.md`](docs/training_runs.md)
- A single-card consumer-hardware training recipe (8-bit Adam, staged seq-len, growth-based scaling)
- A pre-registered **MoE-vs-dense ablation** at matched active compute, and the measured depth/calibration findings feeding the MoDr direction — [`docs/roadmap.md`](docs/roadmap.md)

**Current state (2026-06-20):** the active research is **fixing free-running
generation degeneration** at small scale, *before* spending on tokens/compute.
It's been diagnosed as **exposure bias** (a learned repetition attractor) —
*not* a recurrent/hidden-state collapse; the recurrent representations stay
healthy (verified with [`tools/collapse_metrics.py`](tools/collapse_metrics.py)).
The live thread is the **distillation objective**: forward-KL collapses;
reverse-KL escaped the attractor early but mode-collapsed with more tokens; **JSD
is the current test**. (An earlier external code review also found and fixed 5
correctness bugs — notably a clobbered zero-init that was silently injecting
noise into the hidden state each loop.) Full record:
[`docs/training_runs.md`](docs/training_runs.md) ·
[`docs/generation_probe_tracker.md`](docs/generation_probe_tracker.md) ·
[`docs/review_action_plan.md`](docs/review_action_plan.md).

**Honest scale note:** the trained checkpoints are **278M–632M proof-of-concept
models.** They validate that the architecture + recipe work end-to-end (stable
training, balanced MoE routing, calibrated uncertainty, all three halt
mechanisms firing) — but they do **not** produce coherent text. That's a
parameter-count ceiling, not a design flaw. This is a research / architecture
project, not a deployable model.

**One-line description:** *a research project on recurrent-depth MoE
transformers — distillation efficiency, training dynamics, and calibrated
uncertainty — distilled from Ouro-2.6B-Thinking and forked from OpenMythos,
aimed at a small, private, local medical-information model.*

See [`docs/roadmap.md`](docs/roadmap.md) for the full checkpoint lineage,
eval results, and forward plan.

---

## Installation

```bash
pip install mythouro

# uv pip install mythouro
```

MythOuro defines four installation extras, each pulling in the optional
dependencies for one workflow:

| Extra | Adds | Use case |
|---|---|---|
| `flash` | `flash-attn ≥ 2.8.3` | Faster `GQAttention` on Ampere+ (CC ≥ 8.0). Falls back to torch SDPA when absent. |
| `data`  | `datasketch`         | MinHash LSH dedup in `python -m data dedup`. |
| `train` | `wandb`              | Experiment tracking slot for the training scripts. |
| `all`   | every optional dep   | Everything above. |

```bash
pip install "mythouro[flash]"        # inference on Ampere/Blackwell
pip install "mythouro[data,train]"   # pretraining prep + tracking
pip install "mythouro[all]"          # everything
```

### Hardware tiers (what actually runs today)

This table reflects what's *shipped and tested*, not aspirational backends.
GGUF / GPTQ / AWQ exports and the vLLM / llama.cpp inference backends are
deliberately out of scope until there's working integration code for them.

| Tier            | VRAM     | Inference                                    | Training                                                  |
|-----------------|----------|----------------------------------------------|-----------------------------------------------------------|
| Consumer GPU    | 8 GB     | Tiny variants in bf16                        | `train_tiny_mythos.py` sanity check; CPU offload for 1B   |
| Mid-range GPU   | 12–16 GB | 1B variant in bf16                           | Tiny custom configs; aggressive grad-accum on 1B          |
| Prosumer GPU    | 24 GB+   | 3B variant in bf16; INT8 via dynamic quant   | 1B training with grad checkpointing                       |
| Server multi-GPU| 80 GB+   | 3B variant in bf16 across GPUs               | 1B–10B via FSDP HYBRID_SHARD (NVLink-paired clusters)     |
| CPU only        | n/a      | Tiny variants for testing / smoke runs       | Eval + data pipeline only (`data/` and `eval/` packages)  |

### Console scripts

After installation, the following commands are on `$PATH`:

```bash
mythouro-train-tiny           # train_tiny_mythos.py — memorisation smoke test
mythouro-train                # training/3b_fine_web_edu.py — full FineWebEdu pretrain
mythouro-train-1b             # training/1b_fine_web_edu.py — 1B variant pretrain
mythouro-eval --benchmarks all --max-samples 50 --output report.json
mythouro-data dedup -i in.jsonl -o out.jsonl --threshold 0.8
mythouro-data contamination -i in.jsonl -o out.jsonl -b arc gsm8k humaneval
mythouro-data tokenizer-eval --use-hf-samples
```

## Usage

```python

import torch
from mythouro.main import MythOuro, MythOuroConfig


attn_type = "mla"  # or "gqa"

base = {
    "vocab_size": 1000,
    "dim": 256,
    "n_heads": 8,
    "max_seq_len": 128,
    "max_loop_iters": 4,
    "prelude_layers": 1,
    "coda_layers": 1,
    "n_experts": 8,
    "n_shared_experts": 1,
    "n_experts_per_tok": 2,
    "expert_dim": 64,
    "lora_rank": 8,
    "attn_type": attn_type,
}

if attn_type == "gqa":
    cfg = MythOuroConfig(**base, n_kv_heads=2)
else:
    cfg = MythOuroConfig(
        **base,
        n_kv_heads=8,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )

model = MythOuro(cfg)
total = sum(p.numel() for p in model.parameters())
print(f"\n[{attn_type.upper()}] Parameters: {total:,}")

ids = torch.randint(0, cfg.vocab_size, (2, 16))
# forward() returns (logits, uncertainty) — per-token confidence in (0, 1)
logits, uncertainty = model(ids, n_loops=4)
print(f"[{attn_type.upper()}] Logits shape: {logits.shape}")
print(f"[{attn_type.upper()}] Uncertainty shape: {uncertainty.shape}")

out = model.generate(ids, max_new_tokens=8)   # n_loops defaults to the trained depth;
                                              # raise it explicitly for depth extrapolation
print(f"[{attn_type.upper()}] Generated shape: {out.shape}")

A = model.recurrent.injection.get_A()
rho = torch.linalg.eigvals(A).abs().max().item()
print(
    f"[{attn_type.upper()}] Spectral radius ρ(A) = {rho:.4f} (must be < 1)"
)
```



## Model Variants

Pre-configured scales, from the trainable distillation students up to the
aspirational 1T frontier config:

```python
from mythouro import (
    mythouro_distill_tiny,
    mythouro_distill_small,
    mythouro_1b,
    mythouro_3b,
    mythouro_10b,
    mythouro_50b,
    mythouro_100b,
    mythouro_500b,
    mythouro_1t,
    MythOuro,
)

cfg = mythouro_distill_tiny()  # returns a MythOuroConfig
model = MythOuro(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
```

| Variant | `dim` | Routed experts | `expert_dim` | Loop iters | Context | Notes |
|---|---|---|---|---|---|---|
| `mythouro_distill_tiny`  | 1280 | 24 | 1280 | 4 | 2k | **278M** — Ouro-aligned distillation student; fits a 12 GB GPU alongside the teacher |
| `mythouro_distill_small` | 1280 | 48 | 1280 | 4 | 2k | **420M** — MoE-expansion target of `distill_tiny` (see [growth_design](docs/growth_design.md)) |
| `mythouro_1b` | 2048 | 64 | 2048 | 6 | 4k | research / fine-tune scale |
| `mythouro_3b` | 3072 | 64 | 4096 | 6 | 4k | compact inference model |
| `mythouro_10b` | 4096 | 128 | 5632 | 8 | 8k | mid-scale |
| `mythouro_50b` | 6144 | 256 | 9728 | 8 | 8k | large reasoning |
| `mythouro_100b` | 8192 | 256 | 13568 | 12 | 1M | frontier-class |
| `mythouro_500b` | 12288 | 512 | 23040 | 12 | 1M | ultra-scale MoE |
| `mythouro_1t` | 16384 | 512 | 34560 | 12 | 1M | maximum scale |

The `distill_tiny` and `distill_small` variants are the ones actually
trained and validated end-to-end on consumer hardware (see [Practical
training pipeline](#practical-training-pipeline) below). The 1B–1T configs
are pre-defined scale targets, not trained checkpoints.

---

## Training

The training script for the 3B model on FineWeb-Edu is at [`training/3b_fine_web_edu.py`](training/3b_fine_web_edu.py).

**Single GPU:**
```bash
python training/3b_fine_web_edu.py
```

**Multi-GPU (auto-detects GPU count):**
```bash
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

Key design choices:

| Feature | Detail |
|---|---|
| Optimizer | AdamW |
| Dataset | `HuggingFaceFW/fineweb-edu` (`sample-10BT` by default, swap to `sample-100BT` or `default` for full run) |
| Tokenizer | `openai/gpt-oss-20b` via `MythOuroTokenizer` |
| Parallelism | PyTorch DDP via `torchrun`, sharded streaming dataset |
| Precision | bfloat16 on H100/A100, float16 + GradScaler on older GPUs |
| Schedule | Linear warmup (2000 steps) → cosine decay |
| Target | 30B tokens (~Chinchilla-adjusted for looped architecture) |

---

## Practical training pipeline

Beyond the FineWeb-Edu pretrain above, the repo contains a complete,
hardware-validated **distillation → SFT → model-growth** pipeline that was
run end-to-end on consumer GPUs. This is the path that produced the
trained reference checkpoints.

```
Ouro-2.6B-Thinking (teacher)
        │  logit distillation  (training/distill.py)
        ▼
distill_tiny  278M  ──── SFT ────►  distill_tiny  278M  (instruction-tuned)
 (v1)                (training/sft.py)        (v2)
                                              │  MoE expansion  (tools/grow_checkpoint.py)
                                              ▼
                                       distill_small  420M  ──── SFT ───►  distill_small  420M
                                        (v3, 24→48 experts)                  (v4, +OpenHermes)
```

| Stage | Script | What it does |
|---|---|---|
| **Distillation** | [`training/distill.py`](training/distill.py) | Hinton-style logit distillation from a frozen `ByteDance/Ouro-2.6B-Thinking` teacher into a tiny student, with the architecture's auxiliary losses (MoE load balance, uncertainty calibration, depth regularisation). |
| **SFT** | [`training/sft.py`](training/sft.py) | Supervised fine-tuning with **loss masked to response tokens only**. Mixes instruction corpora (OpenHermes-2.5, Magicoder, MetaMathQA) via [`mythouro/sft_data.py`](mythouro/sft_data.py). `--use-8bit-adam` enables bitsandbytes 8-bit optimizer state (~2.5 GB saved; auto-detects a compatible CUDA binary). |
| **Model growth** | [`tools/grow_checkpoint.py`](tools/grow_checkpoint.py) | Function-preserving **MoE expansion** (e.g. 24 → 48 routed experts) via [`mythouro/grow.py`](mythouro/grow.py). New experts are zero-gated + sentinel-biased at promotion so the model's output is byte-identical, then a decay schedule eases them into the routing. See [`docs/growth_design.md`](docs/growth_design.md). |
| **Inspect** | [`inspect_checkpoint.py`](inspect_checkpoint.py) | Per-prompt diagnostics: generated text, per-token uncertainty trace, ACT halt distribution, MoE utilisation. |

`training/distill.py`'s defaults encode the **proven recipe** (warmup 500,
depth-reg 0.3 — recovered from v1's model-card provenance after the script
defaults flatlined a run; see the roadmap's failure modes). Cross-run eval
results live in [`docs/training_runs.md`](docs/training_runs.md); the full
lineage, hardware notes, failure-mode recovery patterns, and forward plan in
[`docs/roadmap.md`](docs/roadmap.md).

---

## Documentation

| Page | Description |
|---|---|
| [`docs/roadmap.md`](docs/roadmap.md) | **Start here when resuming.** Checkpoint lineage, capability milestones, memory-reduction options, hardware-scaling analysis, failure-mode recovery patterns, and the forward plan. |
| [`docs/mythouro.md`](docs/mythouro.md) | Full API reference for the `MythOuro` class — constructor, `forward`, `generate`, all sub-modules, configuration reference, and usage examples |
| [`docs/growth_design.md`](docs/growth_design.md) | MoE-expansion / model-growth design notes — the function-preserving promotion algorithm and its training contract |
| [`docs/datasets.md`](docs/datasets.md) | Recommended training datasets with token budget guidance per model size |
| [`docs/training_runs.md`](docs/training_runs.md) | **Cross-run results table** — every training session's eval stats, trajectories, and behavioural reads |
| [`docs/review_action_plan.md`](docs/review_action_plan.md) | Code-review (P0–P2) status tracker — what was broken, what's fixed, what's queued |
| [`docs/mythouro_code_review_findings.md`](docs/mythouro_code_review_findings.md) | The external code review itself (the source document for the fixes) |
| [`docs/fork_vs_openmythos.md`](docs/fork_vs_openmythos.md) | Verified code-level diff of this fork against upstream OpenMythos |

---

## The Central Hypothesis

Claude Mythos is suspected to be a **Recurrent-Depth Transformer (RDT)** — also called a Looped Transformer (LT). Rather than stacking hundreds of unique layers, a subset of layers is recycled and run through multiple times per forward pass. Same weights. More loops. Deeper thinking.

This is not chain-of-thought. There is no intermediate token output. All of this reasoning happens **silently, inside a single forward pass**, in continuous latent space.

---

## Architecture

A looped transformer divides its layers into three functional blocks:

```
Input
  ↓
[Prelude P]        — standard transformer layers, run once
  ↓
[Recurrent Block R] — looped T times
  ↑_______↓         (hidden state h updated each loop with input injection e)
  ↓
[Coda C]           — standard transformer layers, run once
  ↓
Output
```

The recurrent block update rule at each loop step t:

```
h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
```

Where:
- `h_t` is the hidden state after loop t
- `e` is the encoded input (from the Prelude), injected at every loop
- `A` and `B` are learned injection parameters
- The Transformer blocks apply attention and MLP as usual

The injection of `e` at every step is what prevents the model from drifting — it keeps the original input signal alive throughout the entire recurrence depth.

The full implementation is in [`mythouro/main.py`](mythouro/main.py). See the [`MythOuro` class reference](docs/mythouro.md) for a detailed API walkthrough, configuration options, and usage examples.

### Attention Implementations

The attention layer is switchable via `cfg.attn_type`:

| Option | Class | Description |
|---|---|---|
| `"gqa"` | `GQAttention` | Grouped Query Attention (Ainslie et al., 2023) — fewer KV heads than Q heads (`n_kv_heads < n_heads`), reducing KV-cache memory by `n_heads / n_kv_heads`. Uses **Flash Attention 2** (Dao et al., 2023) when `flash-attn>=2.8.3` is installed: GQA is handled natively (no KV head expansion), I/O-bound-optimal, with a transparent fallback to manual scaled dot-product attention when the package is absent. |
| `"mla"` | `MLAttention` | Multi-Latent Attention (DeepSeek-V2) — caches a compressed KV latent (`kv_lora_rank`) rather than full K/V, with split RoPE / no-RoPE head dims for position-aware compression. |

RoPE is applied to Q and K before caching, so cached values do not need to be re-rotated on retrieval.

---

## Why This Explains Mythos

### 1. Systematic Generalization

Vanilla transformers fail to combine knowledge in ways they have never seen during training. Looped transformers pass this test. The ability emerges through a **three-stage grokking process**:

1. Memorization — model fits training distribution
2. In-distribution generalization — model handles known compositions
3. Systematic generalization — model handles novel compositions OOD, abruptly and suddenly

This is why Mythos feels qualitatively different from other models on novel questions — the capability phase-transitions in, rather than emerging gradually.

### 2. Depth Extrapolation

Train on 5-hop reasoning chains. Test on 10-hop. Vanilla transformer fails. Looped transformer succeeds — by running more inference-time loops. This maps directly to the observation that Mythos handles deeply compositional problems (multi-step math, long-horizon planning, layered arguments) without explicit chain-of-thought.

More loops at inference = deeper reasoning chains = harder problems solved.

### 3. Latent Thoughts as Implicit Chain-of-Thought

Each loop iteration is the functional equivalent of one step of chain-of-thought, but operating in continuous latent space rather than token space. A looped model running T loops implicitly simulates T steps of CoT reasoning. This has been formally proven (Saunshi et al., 2025).

Furthermore, continuous latent thoughts — unlike discrete token outputs — can encode **multiple alternative next steps simultaneously**. This allows something closer to breadth-first search over the reasoning space, rather than a single committed reasoning path. The model is effectively exploring many possible directions inside each forward pass before converging.

### 4. No Parameter Explosion

A looped model with k layers run L times achieves the quality of a kL-layer non-looped model, with only k layers worth of parameters. For Mythos-scale deployments, this matters enormously:

- Memory footprint does not grow with reasoning depth
- Inference-time compute scales with loop count, not model size
- This makes deeper reasoning "free" in terms of parameters

---

## The Stability Problem (and How It Was Likely Solved)

Training looped models is notoriously unstable. Two failure modes dominate:

- **Residual explosion** — the hidden state `h_t` grows unboundedly across loops
- **Loss spikes** — training diverges suddenly due to large spectral norms in injection parameters

### The Dynamical Systems View

Recast looping as a discrete linear time-invariant (LTI) dynamical system over the residual stream. Ignoring the nonlinear Transformer contribution, the recurrence becomes:

```
h_{t+1} = A·h_t + B·e
```

For this LTI system, stability is governed entirely by the **spectral radius** of A:
- `ρ(A) < 1` → stable, convergent
- `ρ(A) ≥ 1` → unstable, divergent

Empirically, every divergent training run learns `ρ(A) ≥ 1`. Every convergent run maintains `ρ(A) < 1`.

### The Fix

Constrain the injection parameters so that stability is guaranteed **by construction**:

1. Parameterize A as a continuous negative diagonal matrix
2. Discretize using ZOH/Euler schemes: `A_discrete = exp(Δt · A_continuous)`
3. Enforce negativity via `A := Diag(-exp(log_A))` with a learned scalar `Δt`
4. This ensures `ρ(A) < 1` always holds, regardless of learning rate or batch noise

The result: the looped model becomes significantly more robust to hyperparameter selection and trains cleanly even at high learning rates. This is the Parcae architecture (Prairie et al., 2026), and it represents the most likely class of solution Anthropic used to make Mythos trainable.

---

## Scaling Laws for Looped Models

Parcae establishes the first predictable scaling laws for looped training:

- **Training**: For a fixed FLOP budget with fixed parameters, increasing mean recurrence and reducing token count yields a lower loss than training with minimal loops on more data. Optimal recurrence and optimal token count both follow **power laws** with consistent exponents across scales.
- **Inference**: More test-time loops improves quality following a **predictable, saturating exponential decay** — gains are real but diminishing. This mirrors the inference-time scaling of chain-of-thought.

At 770M parameters, a looped model achieves the downstream quality of a 1.3B fixed-depth Transformer trained on the same data — roughly **half the parameters for the same quality**.

Applied to Mythos: if trained under these scaling laws, Mythos could be dramatically more parameter-efficient than it appears, with a large fraction of its apparent "capability" coming from loop depth rather than raw parameter count.

---

## The Loop Index Embedding Hypothesis

A key open question is whether the looped block behaves **identically** on every iteration, or whether it can learn to do different things at different loop depths.

Without any positional signal across loops, the same weights must handle both early-stage pattern matching and late-stage refinement — a tight constraint. A **RoPE-like embedding of the loop index** injected alongside the input at each step would allow the same parameters to implement functionally distinct operations across iterations, much like how RoPE allows the same attention heads to behave differently at different sequence positions.

If Mythos uses this technique, each loop is not a repetition — it is a distinct computational phase, all sharing weights but operating in different representational regimes. This would substantially increase the expressiveness of the recurrent block without increasing parameter count.

---

## The Overthinking Problem

More loops is not always better. Beyond a certain depth, excessive recurrence **degrades predictions** — the hidden state drifts past the solution and into noise. This is the "overthinking" failure mode.

The original Universal Transformer (Dehghani et al., 2018) addressed this with an **Adaptive Computation Time (ACT)** halting mechanism: a learned scalar per position that dynamically decides when to stop looping. Positions that are harder to process receive more computation; simple tokens halt early.

Mythos almost certainly has some version of this. The model cannot naively run the maximum number of loops on every input — it needs a learned signal for when the answer has converged. The ACT mechanism also makes the model **Turing-complete** under certain assumptions, which has theoretical implications for the class of problems it can solve.

---

## Mixture of Experts — Suspected for Large Parameter Counts

The looped transformer explains the depth of Mythos's reasoning, but not the breadth. Handling wildly different domains — code, math, literature, science, law — with the same weights requires **Mixture of Experts (MoE)**. The suspected design replaces every FFN in the Recurrent Block with a fine-grained MoE layer: each FFN is split into many small experts (1/m the normal size), a router selects the top-mK of them per token via learned affinity scores, and a small number of **shared experts** are always activated regardless of routing to absorb common cross-domain knowledge — syntax, basic reasoning, general context — that would otherwise be redundantly learned by every routed expert. Routing collapse is prevented through a bias term on the router logits adjusted dynamically during training, keeping load balanced across experts without distorting the loss signal. 

As the hidden state `h_t` evolves across loop iterations, the router may select different expert subsets at each depth, making every loop computationally distinct despite shared weights. MoE provides breadth; looping provides depth. If the activation ratio is ~5%, Mythos could hold hundreds of billions of total parameters while activating only a small fraction per token — the true parameter count, if ever disclosed, would be a storage number, not a compute number.

---

## The Memorization-Reasoning Tradeoff

Looped models exhibit an interesting dichotomy: looping improves reasoning but can hurt memorization. The recurrent structure is optimized for iterative composition — running a reasoning chain forward — but does not inherently improve the storage of rote facts.

This maps to an observable characteristic of Mythos: it reasons exceptionally well about novel problems it has never seen, but its factual recall can be inconsistent. The architecture is structurally biased toward composition over memorization.

Looping-based regularization (Saunshi et al., 2025) can be used to balance this tradeoff during training — applying stronger looping constraints for reasoning tasks while relaxing them for retrieval tasks.

---

## Parameter Reuse via LoRA Adaptation

A complementary approach from Relaxed Recursive Transformers (Bae et al., 2024): rather than requiring fully identical weights at every loop, add a small **depth-wise LoRA module** at each iteration. This preserves the compactness of weight sharing while allowing each loop to adapt its behavior slightly.

The result:
- Each loop shares a large common weight matrix (the recursive base)
- A small rank-r adaptation matrix shifts behavior per iteration depth
- The total parameter overhead is minimal

This bridges the gap between pure weight-tying (maximally parameter-efficient, less expressive) and fully distinct layers (maximally expressive, no parameter savings). Mythos likely sits somewhere on this spectrum.

---

## Continuous Depth-wise Batching

A downstream consequence of the recursive architecture: **Continuous Depth-wise Batching**. Because all tokens share the same recurrent block, the model can exit the loop at different depths for different tokens or sequences — processing easy inputs quickly and hard inputs with more iterations, all within the same batch.

Theoretical analysis suggests 2-3x improvements in inference throughput. For a deployed model like Mythos serving many users simultaneously, this would be a substantial efficiency gain.

---

## Summary: What Mythos Probably Is

| Property | Description |
|---|---|
| Architecture | Recurrent-Depth Transformer (Prelude + Looped Recurrent Block + Coda) |
| FFN layer | Suspected MoE — fine-grained experts + always-on shared experts |
| Parameter count | Very large total; small fraction activated per token (~5% estimate) |
| Reasoning mechanism | Implicit multi-hop via iterative latent updates — no token output between steps |
| Inference-time scaling | More loops = deeper reasoning, following predictable exponential decay |
| Training stability | LTI-constrained injection parameters with spectral radius < 1 |
| Loop differentiation | Likely uses loop-index positional embedding (à la RoPE) per iteration |
| Halting | Adaptive Computation Time or learned convergence criterion |
| Attention | GQA (with optional Flash Attention 2) or MLA with compressed KV latent cache |
| Scaling law | Optimal training scales looping and data together, not parameters alone |
| Reasoning vs. memory | Structurally biased toward composition; memorization requires separate treatment |
| Deployment | Continuous Depth-wise Batching enables variable compute per request |

---

## References

### Twitter / X

- Why Claude Mythos is so good — looped transformer theory (Sigrid Jin): https://x.com/realsigridjin/status/2044620031410266276
- LT implicit reasoning over parametric knowledge unlocks generalization (Yuekun Yao): https://x.com/yuekun_yao/status/2044229171627639004
- Looped transformer cyclic trajectories and input injection (rosinality): https://x.com/rosinality/status/2043953033428541853
- Parcae scaling laws for stable looped language models — thread (Hayden Prairie): https://x.com/hayden_prairie/status/2044453231913537927
- RoPE-like loop index embedding idea to differentiate functions across iterations (davidad): https://x.com/davidad/status/2044453231913537927
- On the Looped Transformers Controversy by ChrisHayduk: https://x.com/ChrisHayduk/status/2045947623572688943
- On the Looped Transformers Controversy Summary by @realsigridjin https://x.com/realsigridjin/status/2046012743778766875


### Papers

- Fine-grained expert segmentation and shared expert isolation in MoE: https://arxiv.org/abs/2401.06066
- Loop, Think, & Generalize — Implicit Reasoning in Recurrent Depth Transformers: https://arxiv.org/pdf/2604.07822
- Parcae — Scaling Laws for Stable Looped Language Models: https://arxiv.org/abs/2604.12946
- Parcae blog: https://sandyresearch.github.io/parcae/
- Universal Transformers: https://arxiv.org/pdf/1807.03819
- Reasoning with Latent Thoughts — On the Power of Looped Transformers: https://arxiv.org/abs/2502.17416
- Training Large Language Models to Reason in a Continuous Latent Space: https://arxiv.org/abs/2412.06769
- Relaxed Recursive Transformers — Effective Parameter Sharing with Layer-wise LoRA: https://arxiv.org/pdf/2410.20672
- Mixture-of-Depths Attention: https://arxiv.org/abs/2603.15619
- Hyperloop Transformers: https://arxiv.org/abs/2604.21254
- The Recurrent Transformer: Greater Effective Depth and Efficient Decoding: https://arxiv.org/abs/2604.21215

**Distillation teacher & training/scaling techniques (this fork):**

- Ouro — *Scaling Latent Reasoning via Looped Language Models* (Zhu et al., 2025) — the distillation teacher (`ByteDance/Ouro-2.6B-Thinking`, Apache 2.0). Paper: https://arxiv.org/abs/2510.25741 · Project: https://ouro-llm.github.io/ · Models: https://huggingface.co/ByteDance/Ouro-2.6B-Thinking
- Net2Net — Accelerating Learning via Knowledge Transfer (Chen et al., 2015) — basis for the function-preserving growth discussion: https://arxiv.org/abs/1511.05641
- MoE-LPR — MoE expansion via post-pretraining with frozen experts (2024) — basis for the sentinel-bias MoE-expansion recipe: https://arxiv.org/abs/2408.11396
- DeepSeek-V3 — aux-loss-free load balancing for MoE routing: https://arxiv.org/abs/2412.19437
- Distilling the Knowledge in a Neural Network (Hinton et al., 2015) — the soft-label distillation objective: https://arxiv.org/abs/1503.02531

---

## Acknowledgements

This project stands on two pieces of others' work and credits them explicitly:

- **[Kye Gomez](https://github.com/kyegomez)** — author of the upstream
  **[OpenMythos](https://github.com/kyegomez/OpenMythos)** project (MIT) that
  this is forked from. His work provided the foundation everything here builds
  on: the Recurrent-Depth Transformer architecture (Prelude/Recurrent/Coda,
  MoE, MLA/GQA, LTI-stable injection, ACT halting). Genuine credit — without
  that base, none of the rest would exist. The original copyright is preserved
  in [`LICENSE`](LICENSE). (This fork renamed the project from *OpenMythos* to
  **MythOuro** — Myth + Ouro, for the distillation teacher.)

  > **To be clear:** Kye Gomez has had **no involvement in this fork** beyond
  > authoring the upstream foundation. The distillation → SFT → model-growth
  > pipeline, the function-preserving MoE expansion, the trained checkpoints,
  > and every direction the project has taken since forking are independent
  > work. He bears no responsibility for, and has not endorsed, the choices or
  > current state of this fork.
- **[Ouro — *Scaling Latent Reasoning via Looped Language Models*](https://ouro-llm.github.io/)**
  (Zhu et al., 2025; `ByteDance/Ouro-2.6B-Thinking`, Apache 2.0) — the frozen
  teacher distilled from to produce the trained checkpoints. Used under
  Apache 2.0 with attribution. Paper: [arXiv:2510.25741](https://arxiv.org/abs/2510.25741).

The distillation → SFT → model-growth pipeline, the function-preserving MoE
expansion, the trained checkpoints, and the project's direction since forking
are **original work by Daniel Hardy** (the MythOuro fork). See
[Licensing & data provenance](#licensing--data-provenance) for the full
dependency accounting, and the [References](#references) for the papers behind
each architectural component.

---

## Citation

If you use MythOuro, please cite this project. Also cite the upstream OpenMythos
project (the architecture it builds on) and, if you use the distilled
checkpoints, the Ouro teacher:

```bibtex
@software{hardy2026mythouro,
  author    = {Daniel Hardy},
  title     = {MythOuro: A Recurrent-Depth MoE Language Model with Distillation
               and Function-Preserving Model Growth},
  year      = {2026},
  note      = {Custom recurrent-depth MoE model forked from OpenMythos,
               distilled from Ouro-2.6B-Thinking, and scaled via
               function-preserving MoE expansion}
}

@software{gomez2026openmythos,
  author    = {Kye Gomez},
  title     = {OpenMythos: A Theoretical Reconstruction of the Claude Mythos Architecture},
  year      = {2026},
  url       = {https://github.com/kyegomez/OpenMythos},
  note      = {Recurrent-Depth Transformer with MoE, MLA, LTI-stable injection, and ACT halting. MythOuro is a fork of OpenMythos.}
}

@article{zhu2025ouro,
  title   = {Scaling Latent Reasoning via Looped Language Models},
  author  = {Zhu, Rui-Jie and Wang, Zixuan and Hua, Kai and Zhang, Tianyu and
             Li, Ziniu and Que, Haoran and Wei, Boyi and Wen, Zixin and Yin, Fan and
             others},
  journal = {arXiv preprint arXiv:2510.25741},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.25741}
}
```

---

## License

MIT License — Copyright (c) 2026 Kye Gomez. See [`LICENSE`](LICENSE) for the full text.

The MythOuro **code** is MIT. That license covers the source only — it does
**not** automatically extend to model weights trained with it, which inherit
constraints from the teacher and training data (see below).

## Licensing & data provenance

Honest accounting of what the trained checkpoints depend on, so any future
decision to **distribute weights or use them commercially** is made with the
facts in hand. (Not legal advice.)

| Source | License | Implication |
|--------|---------|-------------|
| **Code** (this repo) | MIT | Permissive; reuse freely with attribution |
| **Teacher** — `ByteDance/Ouro-2.6B-Thinking` | **Apache 2.0** | ✅ Permits distillation, derivatives, redistribution, commercial use — with attribution (include the license, state changes, retain copyright). Card adds an advisory "research purposes only" note (not a license restriction). |
| **Distillation data** — FineWeb-Edu, OpenWebMath (ODC-By), CodeParrot-clean (per-file) | open / per-file | Standard web/code provenance; CodeParrot files carry their own upstream licenses |
| **SFT data** — OpenHermes 2.5, MetaMathQA, Magicoder-Evol-Instruct | ⚠️ **OpenAI-output provenance** | All three contain data **generated by OpenAI models**. OpenAI's terms restrict using outputs to train competing models. This is the **main constraint** on distributing/commercialising the SFT'd checkpoints. |

**Bottom line:**
- **As a research project that doesn't distribute weights** → low practical risk; everything above is used under permissive terms or fair research use.
- **If distributing checkpoints or going commercial** → the teacher is clean (Apache 2.0), but the **SFT datasets' OpenAI-generated provenance is the gating issue**. To get a cleanly-distributable checkpoint, retrain SFT on corpora without OpenAI-generated content (Dolly-15k human-written, documented-provenance Tulu-3 subsets, or data self-generated from an Apache/MIT model). See [`docs/datasets.md`](docs/datasets.md) for the per-dataset breakdown.

**Third-party dependency:** `bitsandbytes` (MIT) — optional, enables the
`--use-8bit-adam` 8-bit optimizer path.
