# MythOuro

<p align="left">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20%C2%B7%20not%20released-orange?style=for-the-badge">
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

**Current state (2026-06-27):** the long-standing blocker — **free-running
generation degeneration** at small scale — has had its **first real break.** It
was diagnosed as **exposure bias** (a learned repetition attractor; *not* a
recurrent/hidden-state collapse — the recurrent representations stay healthy,
verified with [`tools/collapse_metrics.py`](tools/collapse_metrics.py)) and shown
to be **decoupled from every formal metric** (best-ever PPL, good calibration,
and solved training stability all coexisted with hard mode-collapse). The offline
**distillation-objective** avenue was swept and closed: forward-KL, reverse-KL,
*and* JSD all mode-collapse with tokens, stable or not. The cure is **on-policy
distillation (GKD / MiniLLM)** — the student trains on *its own* rollouts under
teacher correction, attacking exposure bias directly.

**On-policy is now implemented and partially validated (2026-06-27):** the first
on-policy run produced the **first movement on the unaided-generation metric in
the project's history** — a collapsed prose seed went from a stuck `the the the`
attractor to varied sentences (top-token share 0.45 → 0.14, distinct-1
0.15 → 0.66). It's *partial* — under-represented domains (medical, code) still
need more on-policy dose — so reaching coherence is now a **throughput / scaling
problem**, not an open research question. Full record:
[`docs/training_runs.md`](docs/training_runs.md) ·
[`docs/generation_probe_tracker.md`](docs/generation_probe_tracker.md) ·
[`docs/onpolicy_plan.md`](docs/onpolicy_plan.md).

**Honest scale note:** the trained checkpoints are **278M–632M proof-of-concept
models.** They validate that the architecture + recipe work end-to-end (stable
training, balanced MoE routing, calibrated uncertainty, all three halt
mechanisms firing). Their free-running generation was mode-collapsed (exposure
bias) — **on-policy distillation is now un-collapsing it** (partial; see current
state above) — but full **content fluency** still needs the real scale-up (more
parameters **and** ~1000× more tokens than can be ground out locally). This
remains a research / architecture project, not a deployable model.

**One-line description:** *a research project on recurrent-depth MoE
transformers — distillation efficiency, training dynamics, and calibrated
uncertainty — distilled from Ouro-2.6B-Thinking and forked from OpenMythos,
aimed at a small, private, local medical-information model.*

See [`docs/roadmap.md`](docs/roadmap.md) for the full checkpoint lineage,
eval results, and forward plan.

---

## Installation

> **Not released.** MythOuro is a private research project — there is **no PyPI
> package and no HuggingFace release**. Use it from source:

```bash
git clone <repo-url> mythouro && cd mythouro
pip install -e .
```

Optional dependency groups (in `pyproject.toml`), install as needed:

| Group | Adds | Use case |
|---|---|---|
| `flash` | `flash-attn ≥ 2.8.3` | Faster `GQAttention` on Ampere+ (CC ≥ 8.0). Transparent fallback to torch SDPA when absent — including on Blackwell, where FA2 is currently unavailable. |
| `data`  | `datasketch`         | MinHash LSH dedup in `python -m data dedup`. |
| `train` | `wandb`              | Experiment-tracking slot for the training scripts. |

```bash
pip install -e ".[flash]"        # + flash-attn (Ampere; no-op on Blackwell)
pip install -e ".[data,train]"   # data prep + tracking
```

### Hardware — what's actually been run

The trained checkpoints (278M–632M) were produced on a **single consumer GPU**
(RTX 5070, 12 GB) with the frozen teacher hosted on a second card. That is the only
configuration **validated end-to-end**. Larger-model, multi-GPU, and FSDP tiers are
**design targets, not tested configs**, and export backends (GGUF/GPTQ/AWQ, vLLM,
llama.cpp) are out of scope until there's integration code. The real scale-up plan —
a single 48 GB Intel Max 1100 — is in [`docs/hardware_options.md`](docs/hardware_options.md).

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
| [`docs/roadmap.md`](docs/roadmap.md) | **Start here when resuming.** Forward plan, checkpoint lineage, capability milestones, decision rules — and the **complete documentation index** (master router to every doc). |
| [`docs/onpolicy_plan.md`](docs/onpolicy_plan.md) | The **on-policy / GKD** work — the live cure for generation collapse (design, run commands, status). |
| [`docs/failure_modes.md`](docs/failure_modes.md) | Failure modes encountered + recovery patterns — the debugging / lessons-learned reference. |
| [`docs/mythouro.md`](docs/mythouro.md) | Full API reference for the `MythOuro` class — constructor, `forward`, `generate`, all sub-modules, configuration reference, and usage examples |
| [`docs/growth_design.md`](docs/growth_design.md) | MoE-expansion / model-growth design notes — the function-preserving promotion algorithm and its training contract |
| [`docs/datasets.md`](docs/datasets.md) | Recommended training datasets with token budget guidance per model size |
| [`docs/training_runs.md`](docs/training_runs.md) | **Cross-run results table** — every training session's eval stats, trajectories, and behavioural reads |
| [`docs/review_action_plan.md`](docs/review_action_plan.md) | Code-review (P0–P2) status tracker — what was broken, what's fixed, what's queued |
| [`docs/mythouro_code_review_findings.md`](docs/mythouro_code_review_findings.md) | The external code review itself (the source document for the fixes) |
| [`docs/fork_vs_openmythos.md`](docs/fork_vs_openmythos.md) | Verified code-level diff of this fork against upstream OpenMythos |

---

## How MythOuro works — and the research behind it

The sections below document **MythOuro's architecture and why each piece is there.** The design originated from a research question — what makes looped models reason — that also motivated the upstream OpenMythos project (credited lineage); MythOuro builds and *trains* it as its own system. Nothing here is a claim about any proprietary model (see the disclaimer up top).

MythOuro is a **Recurrent-Depth Transformer (RDT)** — a Looped Transformer. Rather than stacking hundreds of unique layers, a subset of layers is recycled and run through multiple times per forward pass. Same weights, more loops, deeper computation. This is **not** chain-of-thought: there is no intermediate token output — the iteration happens silently inside a single forward pass, in continuous latent space.

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

## Why recurrent-depth + MoE — the design rationale

### 1. Systematic Generalization

Vanilla transformers fail to combine knowledge in ways they have never seen during training. Looped transformers pass this test. The ability emerges through a **three-stage grokking process**:

1. Memorization — model fits training distribution
2. In-distribution generalization — model handles known compositions
3. Systematic generalization — model handles novel compositions OOD, abruptly and suddenly

This phase-transition behaviour — capability appearing abruptly rather than emerging gradually — is a documented property of looped models and a core reason the architecture is worth pursuing at small scale.

### 2. Depth Extrapolation

Train on 5-hop reasoning chains. Test on 10-hop. Vanilla transformer fails. Looped transformer succeeds — by running more inference-time loops. This is the bet for compositional problems (multi-step math, long-horizon planning, layered arguments): handle them by running more inference-time loops, without an explicit chain-of-thought token stream.

More loops at inference = deeper reasoning chains = harder problems solved.

### 3. Latent Thoughts as Implicit Chain-of-Thought

Each loop iteration is the functional equivalent of one step of chain-of-thought, but operating in continuous latent space rather than token space. A looped model running T loops implicitly simulates T steps of CoT reasoning. This has been formally proven (Saunshi et al., 2025).

Furthermore, continuous latent thoughts — unlike discrete token outputs — can encode **multiple alternative next steps simultaneously**. This allows something closer to breadth-first search over the reasoning space, rather than a single committed reasoning path. The model is effectively exploring many possible directions inside each forward pass before converging.

### 4. No Parameter Explosion

A looped model with k layers run L times achieves the quality of a kL-layer non-looped model, with only k layers worth of parameters. For a small, local-first model like MythOuro, this matters enormously:

- Memory footprint does not grow with reasoning depth
- Inference-time compute scales with loop count, not model size
- This makes deeper reasoning "free" in terms of parameters

---

## The stability problem (and how MythOuro solves it)

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

The result: the looped model becomes significantly more robust to hyperparameter selection and trains cleanly even at high learning rates. This is the Parcae approach (Prairie et al., 2026), and it is exactly what MythOuro uses — the LTI-constrained injection in [`mythouro/main.py`](mythouro/main.py) keeps ρ(A) < 1 *by construction* (verified each run via the spectral-radius check).

---

## Scaling Laws for Looped Models

Parcae establishes the first predictable scaling laws for looped training:

- **Training**: For a fixed FLOP budget with fixed parameters, increasing mean recurrence and reducing token count yields a lower loss than training with minimal loops on more data. Optimal recurrence and optimal token count both follow **power laws** with consistent exponents across scales.
- **Inference**: More test-time loops improves quality following a **predictable, saturating exponential decay** — gains are real but diminishing. This mirrors the inference-time scaling of chain-of-thought.

At 770M parameters, a looped model achieves the downstream quality of a 1.3B fixed-depth Transformer trained on the same data — roughly **half the parameters for the same quality**.

Applied to MythOuro: under these laws, a large fraction of capability is meant to come from **loop depth rather than raw parameter count** — which is the core efficiency bet of a small, local-first model trained on a consumer budget.

---

## The Loop Index Embedding Hypothesis

A key open question is whether the looped block behaves **identically** on every iteration, or whether it can learn to do different things at different loop depths.

Without any positional signal across loops, the same weights must handle both early-stage pattern matching and late-stage refinement — a tight constraint. A **RoPE-like embedding of the loop index** injected alongside the input at each step would allow the same parameters to implement functionally distinct operations across iterations, much like how RoPE allows the same attention heads to behave differently at different sequence positions.

MythOuro uses this: a loop-index embedding makes each loop a **distinct computational phase** rather than a repetition — all sharing weights but operating in different representational regimes — increasing the expressiveness of the recurrent block without adding parameters.

---

## The Overthinking Problem

More loops is not always better. Beyond a certain depth, excessive recurrence **degrades predictions** — the hidden state drifts past the solution and into noise. This is the "overthinking" failure mode.

The original Universal Transformer (Dehghani et al., 2018) addressed this with an **Adaptive Computation Time (ACT)** halting mechanism: a learned scalar per position that dynamically decides when to stop looping. Positions that are harder to process receive more computation; simple tokens halt early.

MythOuro implements this as **ACT halting** (`ACTHalting`): a learned per-position signal decides when the answer has converged, so easy tokens halt early and hard ones get more compute. ACT also makes the model **Turing-complete** under certain assumptions, with theoretical implications for the class of problems it can solve.

---

## Mixture of Experts — breadth across domains

Looping explains reasoning *depth*, but not *breadth*. Handling wildly different domains — code, math, literature, science, medicine — with the same weights is what **Mixture of Experts (MoE)** buys. MythOuro replaces every FFN in the Recurrent Block with a fine-grained MoE layer: each FFN is split into many small experts (1/m the normal size), a router selects the top-mK of them per token via learned affinity scores, and a small number of **shared experts** are always activated regardless of routing to absorb common cross-domain knowledge — syntax, basic reasoning, general context — that would otherwise be redundantly learned by every routed expert. Routing collapse is prevented through a bias term on the router logits adjusted dynamically during training, keeping load balanced across experts without distorting the loss signal. 

As the hidden state `h_t` evolves across loop iterations, the router may select different expert subsets at each depth, making every loop computationally distinct despite shared weights. MoE provides breadth; looping provides depth. At a ~5% activation ratio this lets the larger MythOuro configs hold a big *total* parameter count while activating only a small fraction per token — a storage number, not a compute number.

---

## The Memorization-Reasoning Tradeoff

Looped models exhibit an interesting dichotomy: looping improves reasoning but can hurt memorization. The recurrent structure is optimized for iterative composition — running a reasoning chain forward — but does not inherently improve the storage of rote facts.

The architecture is structurally biased toward **composition over memorization** — it favours reasoning about novel problems over rote factual recall. For MythOuro's medical-information goal this is a feature, not a bug: factual recall is handled by **retrieval** (RAG) rather than asked of the weights, so the model reasons over retrieved sources instead of memorising them.

Looping-based regularization (Saunshi et al., 2025) can be used to balance this tradeoff during training — applying stronger looping constraints for reasoning tasks while relaxing them for retrieval tasks.

---

## Parameter Reuse via LoRA Adaptation

A complementary approach from Relaxed Recursive Transformers (Bae et al., 2024): rather than requiring fully identical weights at every loop, add a small **depth-wise LoRA module** at each iteration. This preserves the compactness of weight sharing while allowing each loop to adapt its behavior slightly.

The result:
- Each loop shares a large common weight matrix (the recursive base)
- A small rank-r adaptation matrix shifts behavior per iteration depth
- The total parameter overhead is minimal

This bridges the gap between pure weight-tying (maximally parameter-efficient, less expressive) and fully distinct layers (maximally expressive, no parameter savings). MythOuro uses **per-loop LoRA**, sitting deliberately on this spectrum.

---

## Continuous Depth-wise Batching

A downstream consequence of the recursive architecture: **Continuous Depth-wise Batching**. Because all tokens share the same recurrent block, the model can exit the loop at different depths for different tokens or sequences — processing easy inputs quickly and hard inputs with more iterations, all within the same batch.

Theoretical analysis suggests 2–3× inference-throughput improvements. For MythOuro serving many requests at once, that's a real efficiency gain on the deployment side.

---

## Architecture summary

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

MIT License — Copyright (c) 2026 Kye Gomez (original OpenMythos architecture) and
Daniel Hardy (MythOuro fork and subsequent work). See [`LICENSE`](LICENSE) for the
full text.

The MythOuro **code** is MIT. That license covers the source only — it does
**not** automatically extend to model weights trained with it, which inherit
constraints from the teacher and training data, and which may be released under a
**separate weights license** (see below).

## Licensing & data provenance

Honest accounting of what the trained checkpoints depend on, so any future
decision to **distribute weights or use them commercially** is made with the
facts in hand. (Not legal advice.)

| Source | License | Implication |
|--------|---------|-------------|
| **Code** (this repo) | MIT | Permissive; reuse freely with attribution |
| **Teacher** — `ByteDance/Ouro-2.6B-Thinking` | **Apache 2.0** | ✅ Permits distillation, derivatives, redistribution, commercial use — with attribution (include the license, state changes, retain copyright). Card adds an advisory "research purposes only" note (not a license restriction). |
| **Distillation data** — FineWeb-Edu, OpenWebMath (ODC-By), CodeParrot-clean (per-file) | open / per-file | Standard web/code provenance; CodeParrot files carry their own upstream licenses |
| **SFT data (legacy v1–v5 only)** — OpenHermes 2.5, MetaMathQA, Magicoder-Evol-Instruct | ⚠️ **OpenAI-output provenance** | Contains data **generated by OpenAI models** → OpenAI's terms restrict using outputs to train competing models. **Applies to the old v1–v5 SFT checkpoints only** — the current direction does not use these. |
| **Current / planned training path** | clean | **Distillation from the Apache-2.0 Ouro teacher on open distill data** (FineWeb-Edu, OpenWebMath — ODC-By; CodeParrot-clean — per-file upstream licenses). Clean-SFT replacements (no OpenAI-derived data) tracked in [`docs/clean_sft_datasets.md`](docs/clean_sft_datasets.md). This is the cleanly-distributable path. |
| **Model weights (on release)** | **planned: responsible-AI license** | Code is MIT, but released *weights* will carry a **separate weights license** — intended to be a responsible-AI / behavioral-use license (OpenRAIL-M-style) with **medical-safety clauses** (not a medical device; not for clinical diagnosis without validation; no harmful use). Keeps the model **free and open to use** while restricting *misuse*. To be finalized and reviewed before any weights are distributed. |

**Bottom line:**
- **Weight provenance is clean:** the teacher (Ouro) is **Apache 2.0** and the code is **MIT** — the model derives from permissively-licensed sources.
- **The only legacy constraint** is the OpenAI-generated SFT data in the *old* v1–v5 checkpoints; the **current distillation direction avoids it**, with clean-SFT replacements planned ([`docs/clean_sft_datasets.md`](docs/clean_sft_datasets.md)).
- **On release:** code stays **MIT**; weights ship under a **responsible-AI / medical-safety** license (row above). *Not legal advice — finalize the weights license with review before distributing.*

**Third-party dependency:** `bitsandbytes` (MIT) — optional, enables the
`--use-8bit-adam` 8-bit optimizer path.

**Development tooling:** built with **Claude Code** (Anthropic) as a coding/research
assistant. Claude is **not** used to generate training data, synthetic datasets, or
model outputs for training — it is an engineering aid only.
