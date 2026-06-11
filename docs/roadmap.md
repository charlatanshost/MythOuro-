# MythOuro Roadmap

Living document tracking what's been built, what's queued, and what's
deliberately out of scope. Updated as we make decisions.

**What this project is:** a custom recurrent-depth MoE language model — a hybrid
that takes its architecture from the OpenMythos (`kyegomez/OpenMythos`) RDT
reconstruction, distills from ByteDance Ouro-2.6B-Thinking as a teacher, and is
independently scaled via a function-preserving model-growth pipeline. It is
*not* OpenMythos (we extended it into a trained pipeline), *not* Ouro (that's the
teacher), and *not* Claude Mythos (speculative inspiration only). Trained
checkpoints are 278M–420M proof-of-concept models — they validate the
architecture and recipe, not deployable quality. Full lineage writeup is in the
README's "Project identity & lineage" section.

**Attribution:** the upstream architecture is Kye Gomez's work
(`kyegomez/OpenMythos`, MIT) and is credited with thanks. He has no involvement
in this fork beyond that foundation, and no responsibility for its direction or
current state. The teacher (`ByteDance/Ouro-2.6B-Thinking`) is Apache 2.0. See
the README "Acknowledgements" and "Licensing & data provenance" sections.

Compute constraint baseline: **single workstation — RTX 5070 (12 GB) + RTX 5060
(8 GB) + RTX 4060 (8 GB), all native bf16, overnight-only training windows.**
Anything multi-week of continuous compute belongs to the "needs cloud or
hardware upgrade" tier.

---

## ⚡ Resume quickstart (read this first if returning after a gap)

**State in one sentence:** the distill → SFT → MoE-growth pipeline is built and
validated through v5; the latest checkpoint is `mythouro_distill_xl_grown_v5`
(632M, 96 experts) — but the **2nd MoE expansion (48 → 96) hit the expert-count
ceiling** (net-comparable to the 420M v4, `cv` wouldn't tighten below ~0.5), so
**MoE growth is now considered tapped out** and output is still gibberish at this
scale (parameter-count ceiling, not a bug).

**The next axis is width/scale, not more experts.** Two real options:

- **Near-term, single-card (Path A):** build **Net2Wider** width growth
  (`grow_width.py`, ~2 sessions — does not exist yet) to push `dim` up toward
  ~1B on the 5070. Function-preserving with SiLU (only Net2Deeper is blocked by
  SiLU non-idempotence — see [`growth_design.md`](growth_design.md)). This is the
  remaining unproven growth axis.
- **The actual destination (rented compute):** **from-scratch *distilled* 3B**
  from a stronger teacher (Llama 3.3 70B / Qwen 2.5 72B), then quantize to INT4
  to fit a 24 GB card. Growth can't deliver the gibberish→coherent jump; scale +
  real data can. See [Scale-up execution plan](#scale-up-execution-plan-the-destination).

> **Do NOT re-run the 48 → 96 MoE expansion** — that was v5 (2026-06-06) and it
> hit the ceiling. The previous version of this block told you to run it; that
> advice is obsolete.

**Where to look:** checkpoint lineage + criteria → [Capability success criteria](#capability-success-criteria-per-milestone);
something broke → [Failure modes](#failure-modes-encountered--recovery-patterns);
which memory/growth technique → [Decision rules](#decision-rules);
hardware questions → [Hardware-scaling analysis](#hardware-scaling-analysis).

**Hard limits to remember:** current rig caps at ~1B single-card (~1.5B via FSDP,
PCIe-penalised); 3B needs an Ampere+ card with ≥24 GB; coherent text needs scale
the workstation can't reach — this is a *recipe-validation* project, not a
deployable-model project.

---

## Documentation index

If you're returning to this project after a break, these are the
authoritative documents and where to find what:

| File | What's in it |
|------|--------------|
| [docs/roadmap.md](roadmap.md) | This file. Forward plan, milestones, decision rules, failure-mode memory. **Start here when resuming.** |
| [docs/mythouro.md](mythouro.md) | Architecture overview — what's in `MythOuro` and why each piece exists. |
| [docs/datasets.md](datasets.md) | Dataset reference — what corpora we use and how. |
| [docs/growth_design.md](growth_design.md) | MoE expansion / model growth design notes. Read before promoting a checkpoint. |
| [docs/fork_vs_openmythos.md](fork_vs_openmythos.md) | Verified code-level diff of this fork against upstream `kyegomez/OpenMythos`. |
| [docs/mythouro_code_review_findings.md](mythouro_code_review_findings.md) | The external code review itself (Fable 5, 2026-06-09) — the source document for the action plan. |
| [docs/review_action_plan.md](review_action_plan.md) | Code-review (P0–P2) status tracker — what's fixed, what's left. **Resume here.** |
| [docs/training_runs.md](training_runs.md) | Comparison table of every training session's eval stats (PPL/loop_eff/ECE trajectories, recipes, raw-data paths). Update after each run. |
| `archived_models/<name>/MODEL_CARD.md` | Per-checkpoint provenance: training config, eval results, behavioural validation, limitations. One per shipped reference checkpoint. |
| `CHANGES.md` | Changelog at the codebase root — features added per session. |

Where to find code that implements specific concepts:

| Concept | File / function |
|---------|-----------------|
| Core model | [mythouro/main.py](../mythouro/main.py) (`MythOuro`, `RecurrentBlock`, `MoEFFN`, `ACTHalting`) |
| Architectural variants | [mythouro/variants.py](../mythouro/variants.py) |
| Tokenizer wrapper | [mythouro/tokenizer.py](../mythouro/tokenizer.py) |
| Training utilities (losses, MoE bias updater, FSDP wrap) | [mythouro/training_utils.py](../mythouro/training_utils.py) |
| Checkpointing + shutdown handler | [mythouro/checkpointing.py](../mythouro/checkpointing.py) |
| SFT dataset + masked-loss contract | [mythouro/sft_data.py](../mythouro/sft_data.py) |
| MoE expansion algorithm | [mythouro/grow.py](../mythouro/grow.py) |
| Confidence-aware generation | [mythouro/inference.py](../mythouro/inference.py) |
| Distillation training loop | [training/distill.py](../training/distill.py) |
| SFT training loop | [training/sft.py](../training/sft.py) |
| Pretraining loop (kept for reference) | [training/3b_fine_web_edu.py](../training/3b_fine_web_edu.py) |
| Eval harness | [eval/harness.py](../eval/harness.py) |
| Inspector | [inspect_checkpoint.py](../inspect_checkpoint.py) |
| Checkpoint promotion CLI | [tools/grow_checkpoint.py](../tools/grow_checkpoint.py) |

---

## Where we are (as of 2026-06-10)

### Shipped reference checkpoints

| Checkpoint | Variant | Params | Method | Status |
|------------|---------|-------:|--------|--------|
| `mythouro_distill_tiny_v1` | distill_tiny | 278M | Logit distillation from Ouro-2.6B-Thinking (5K steps) | ✓ Archived — **pre-fix code** (trained under P0.1/P0.2 bugs) |
| `mythouro_distill_tiny_sft_v2` | distill_tiny | 278M | SFT on Magicoder + MetaMath (3K steps) | ✓ Archived (pre-fix) |
| `mythouro_distill_small_grown_v3` | distill_small | 420M | MoE expansion (24→48 experts) + 3K SFT steps | ✓ Archived (pre-fix) |
| `mythouro_distill_small_v4` | distill_small | 420M | OpenHermes-augmented SFT @ seq_len=768 | ✓ Archived (pre-fix) |
| `mythouro_distill_xl_grown_v5` | distill_xl | 632M | MoE expansion 48→96 + ~2.4K SFT steps | ✓ Archived (pre-fix) — **expert-count ceiling** data point (cv evidence softened post-P0.2; inspector read still supports it) |
| **`moe_s0`** (ablation arm 1) | distill_tiny | 278M | From-scratch distillation on **fixed code + proven recipe** (4K steps, seed 0) | ✓ **Current best: PPL 5.72 vs v1's 37.4 (6.5× better, 1K fewer steps)**, loop_eff 0.500, ECE 0.015. `checkpoints_ablation_moe_s0/` |

Full cross-run stats: [docs/training_runs.md](training_runs.md).

### Code state (post-review, 2026-06-09/10)

External code review (Fable 5) found and we fixed **5 correctness bugs** —
notably P0.1 (v1–v5 all trained with noise injected via a clobbered zero-init)
and P0.3 (eval emitted a never-trained path) — plus most of the P1 perf items.
The moe_s0 run above is the proof the fixes matter. Trackers:
[review_action_plan.md](review_action_plan.md) ·
[mythouro_code_review_findings.md](mythouro_code_review_findings.md).

### Pipeline infrastructure built and tested

- **Distillation training** ([training/distill.py](../training/distill.py)) — Hinton-style logit distillation; defaults now encode the proven recipe
- **SFT training** ([training/sft.py](../training/sft.py)) — masked-CE on response tokens only, growth-checkpoint-aware
- **MoE expansion** ([mythouro/grow.py](../mythouro/grow.py)) — function-preserving promotion with sentinel-bias decay
- **Dense ablation arm** ([mythouro/variants.py](../mythouro/variants.py) `mythouro_distill_tiny_dense`) — matched-active dense twin for the MoE-vs-dense gating experiment
- **Eval harness** ([eval/harness.py](../eval/harness.py)) — perplexity, ECE, loop_efficiency, ARC, GSM8K; loads any checkpoint's own cfg
- **Checkpoint inspector** ([inspect_checkpoint.py](../inspect_checkpoint.py)) — per-prompt diagnostics incl. best-of-trajectory A/B + forced-depth probe
- **Calibration audit** ([tools/per_loop_calibration.py](../tools/per_loop_calibration.py)) — per-loop ECE (the P0.5 tool)
- **Step benchmark** ([tools/bench_step.py](../tools/bench_step.py)) — achieved tok/s on the real model, CUDA/XPU/CPU
- **Device abstraction** ([mythouro/device.py](../mythouro/device.py)) — `--device xpu` turnkey for Intel cards; NVIDIA path unchanged
- **Tests**: 313+ passing (incl. the new invariant tests the review showed were missing)

---

## Near-term: next 1–3 overnights

### Immediate (v4 + v5 both done — MoE growth tapped out)

v4 validated (all 3 halt mechanisms fire, 4/4 prompts halt cleanly) and v5 (2nd
MoE expansion) hit the expert-count ceiling on 2026-06-06. The decision is made:
**stop expanding experts.** The two live next axes are:

1. **Net2Wider width growth** (`grow_width.py`, ~2 sessions — does not exist yet)
   to push `dim` toward ~1B single-card. Function-preserving with SiLU; the
   remaining unproven growth axis.
2. **From-scratch distilled 3B on rented compute** — the actual destination for
   coherent output (see [Scale-up execution plan](#scale-up-execution-plan-the-destination)).
   Growth can't deliver the gibberish→coherent jump.

### Memory-reduction stack (apply as needed)

Ranked by leverage for the current single-card budget:

| Technique | Status | Savings | Effort | Unlocks |
|-----------|--------|--------:|--------|---------|
| **8-bit AdamW** (bitsandbytes) | ✓ Working — `--use-8bit-adam`, auto-detects cuda130 binary on CUDA 13.2 | ~2.5 GB | done | 420M comfortable, ~700M tight |
| **GaLore** (gradient low-rank projection) | 📋 Queued | ~4–6 GB | 4–6h | ~1B on 5070 alone, full-param training |
| **Paged AdamW** (bnb 8-bit + CPU offload) | 📋 Queued | spike resilience | 1h | survives transient memory spikes |
| **Activation checkpointing on more layers** | 📋 Queued (partial coverage today) | ~0.5 GB | 1h | minor headroom |
| **4-bit optimizers** (thu-ml/low-bit-optimizers) | 📋 Optional | ~3.5 GB total | 2h | further squeeze if 8-bit isn't enough |
| **Q-GaLore** (GaLore + 4-bit projection) | 📋 Optional, newer | ~6–8 GB | 6–8h | ~1.5B+ on 5070 alone, less battle-tested |

**Deliberately NOT applicable** (revisit if requirements change):

| Technique | Why excluded |
|-----------|--------------|
| QLoRA | Freezes base model → blocks full-param training + growth |
| GPTQ / AWQ / HQQ / AQLM | Inference-only quantization |
| Zeroth-order optimization | ~100× slower convergence |
| TorchAO | Less mature than bnb/GaLore for our training case |

### Growth axes (next promotion candidates)

Proven: **MoE expansion** round 1 (24 → 48 experts, v3) via [mythouro/grow.py](../mythouro/grow.py).
**Tapped out:** MoE expansion round 2 (48 → 96, v5) hit the expert-count ceiling
(2026-06-06) — do not repeat.

Remaining growth axes:

| Axis | Approach | Effort | Notes |
|------|----------|--------|-------|
| ~~**MoE expansion round 2** (48 → 96 experts)~~ | ~~Same `grow_moe_checkpoint`~~ | done (v5) | **Ceiling hit — net-comparable to v4. Do not repeat.** |
| **Net2Wider** (hidden dim growth) | Custom `grow_width.py` module | 2 sessions | **Next single-card lever.** SiLU non-idempotency means ~0.3 nat loss spike, recoverable |
| **Net2Deeper** (layer count) | Custom `grow_depth.py` module | 2 sessions | Same SiLU caveat, identity-init layers |
| **Loop expansion** | Bump `max_loop_iters` | Trivial | Already free at inference, training would need depth-reg retune |

---

## Mid-term: next month, given overnight-only compute

### Path A — Stay single-card, push to ~1B

```
v4 (420M) → 8-bit Adam validated
        ↓
    [done] MoE expansion round 2 → v5 (632M, 96 experts) — CEILING HIT, no compound gain
        ↓
    GaLore wrapper (4–6h work, one session)            ← resume here
        ↓
    Net2Wider promotion to dim=1536 or 1792 (~1B)      ← the real next lever
        ↓
    20K SFT steps overnight chunks (~2 weeks calendar)
```

Endpoint: a ~1B MythOuro that comfortably fits on the 5070 alone, with all the architectural multipliers stacked. Calendar cost: ~3 weeks of overnights.

### Path B — Add FSDP, push to ~1.5B

```
v4 validated
        ↓
    FSDP wiring (3–4h work, one session)
        ↓
    Direct 1B distillation from a stronger teacher
    (Llama 3.3 70B local quantized OR DeepSeek V3 API)
        ↓
    20K-step distillation overnight chunks
        ↓
    SFT phase with full data mix
```

Endpoint: a ~1B model distilled from a stronger teacher (breaks the Ouro-2.6B quality ceiling). Calendar cost: ~3–4 weeks of overnights.

### My recommendation

**Path A** for the next month — leverages the growth infrastructure already built and stays inside the single-card comfort zone. FSDP wiring is the right Path B work but it's the kind of debugging that's painful in overnight chunks (you discover the sync bug at 7 AM with no time to fix before workday). Path A wins from a "how do I make progress while sleeping" angle.

---

## Scale-up execution plan (the destination)

The biggest roadmap gap, now filled: *what to actually run when compute arrives.*
Decided 2026-06-06 after the v5 expert-ceiling finding.

### Grow vs. from-scratch — decision: **growth is tapped out; destination is from-scratch (distilled)**

Growth (MoE expansion / Net2Wider) was the **proof-of-concept + compute-thrift** tool and it did its job — validated the recipe and *mapped the ceilings*. But the evidence says it's near its useful end for this base:
- **2nd MoE expansion (v5) hit a ceiling** — measured, not theoretical
- The model is **data-starved** (~20–40M tokens) — growth adds capacity a starved model can't fill
- **Warm-start benefit erodes** as growth ops stack (Net2Net literature)

So the gibberish→coherent jump needs **from-scratch at scale with real data**, which growth can't deliver.

**Key nuance — from-scratch ≠ pure pretraining.** For hardware-constrained reality, **from-scratch *distilled* from a strong teacher** (Llama 3.3 70B / DeepSeek V3) is far more sample-efficient than raw CE pretraining — reaches coherence on *much* less data. That's the actual destination.

### The "train big, then quantize to fit" strategy

Confirmed intent (2026-06-06): on rented compute, **train a model bigger than 1B (e.g., 3B), then quantize it to fit a small card for local deployment** — *quantize, not distill-down*. The distinction matters:
- **Quantize** (INT4): a 3B model → still 3B params, ~¼ the memory. **3B-INT4 ≈ fits a 24GB consumer card.** Same model, small footprint. ← **this is the plan**
- (Distillation to a genuinely smaller param count remains an *option*, but the chosen path is quantize-to-fit — keep the 3B capability, shrink the footprint.)

This also **validates the quantization roadmap**: quant was modest at 632M, but **at 3B it pays off** — exactly where INT4 footprint savings let a big model run on consumer hardware. Train big → quantize → run the *full-capability* model locally.

### Execution sequence (rented-compute phase)

```
1. Rent NVIDIA (A100-class) → from-scratch BIG MythOuro (3B+), distilled from a strong teacher
2. Quantize the 3B → 3B-INT4 (torchao) so it fits a 24GB consumer card at full capability
3. Rust + candle runtime for the frozen, quantized model   (deployment phase)
   — separately & cheaply: rent an Intel instance to test whether the B70 ports
     (CUDA→XPU) and its achieved tok/s, before any $800–1000 B70 purchase
```

> **XPU port: DONE (2026-06-09), turnkey for B70 testing.** The CUDA→XPU port is
> built and merged — `mythouro/device.py` abstracts cuda/xpu/cpu (torch.xpu is a
> native 1:1 mirror of torch.cuda since PyTorch 2.5; no IPEX/Linux needed —
> Windows is supported), wired through sft/distill/eval/inspect/bench_step. **The
> NVIDIA path is byte-identical** (on a cuda device every helper returns what the
> old hardcoded code did; complex RoPE is still the default; 217 tests pass). The
> only XPU op risk — RoPE's complex ops (`view_as_complex`/`polar`) — has a
> `rope_real` fallback (real cos/sin, *identical* rotation, equivalence-tested),
> exposed as `bench_step --rope-real`. So on a B70 the whole test is:
> `pip install torch --index-url …/whl/xpu` →
> `python -m tools.bench_step --device xpu` (add `--rope-real` if apply_rope
> errors) → real tok/s. Only hardware can confirm op coverage + MFU; the code is
> ready.

### Sequencing rule: Rust comes AFTER training, never before
Rust is a deployment optimization for a *frozen, coherent* model; from-scratch produces that model, and the architecture may change during it — building Rust first means rebuilding it. Order: **train → freeze → quantize → Rust.**

> **The one exception (note it):** if you do **reasoning RL (GRPO)** later, that
> needs *many inference rollouts during training*, so fast inference would speed
> the RL loop — but even then you'd want the architecture **frozen first**. So
> Rust still lands after the base model exists, never before.

### Objective: distillation vs. training from just datasets (pure pretraining)

Two ways to train the big model. The recipe already blends them via
`α·soft(teacher) + (1−α)·hard(CE on data)` — so "just datasets" = α=0.

> **What the pipeline actually does today (phase pattern, verified):**
> - **Distillation phase** (`training/distill.py`, made **v1**): **hybrid**,
>   `α=0.5` (default, not overridden) — 50% soft teacher logits (Ouro-2.6B) +
>   50% hard CE on data tokens.
> - **SFT phase** (`training/sft.py`, made **v2–v5**): **datasets-only** —
>   `masked_ce_loss` on response tokens, **no teacher**.
>
> So the project already uses *both* modes, in different stages. The **scale-up
> repeats this two-phase shape at larger scale**: hybrid distillation (big run,
> α-blend) → datasets-only SFT on top. The α knob tunes the teacher-vs-data
> balance, or goes α=0 for pure datasets if the teacher is ever dropped.

| | Distillation (teacher) | Pure pretraining (datasets only, α=0) |
|---|---|---|
| Capability ceiling | capped at teacher | **none** — can exceed any model |
| Sample efficiency | **high** (soft targets) | lower (hard CE) |
| Data needed | billions of tokens | **way more** (~10–60B for 3B, Chinchilla ~20 tok/param) |
| Compute | less | more |
| Teacher constraints | hosting + tokenizer + licensing | none (free vocab, clean provenance) |
| Per-step cost | teacher forward each step | cheaper/step |

**Decision: distillation is the pragmatic primary** for a budget-constrained
run. Pure pretraining at 3B (~250–3000× the ~20–40M tokens used so far) is
likely beyond even the rented-compute budget — distillation's sample-efficiency
is what makes coherence reachable affordably. Pure-datasets (α=0) is the
**fallback** if the teacher situation breaks (no hostable teacher, tokenizer/
licensing problems) or if abundant data+compute ever materialise. Note: both
paths still need a large data jump — the data-acquisition problem bites either
way, just bigger for pretraining.

### Teacher choice — decision framework (resolve at run time)

The gating factor is **not "which model is smartest" — it's "which can you run
locally to read its logits."** Logit distillation needs the teacher's full
per-token distribution, which means self-hosting open weights with an
adoptable tokenizer.

| Teacher | Self-host for logits? | Vocab | Verdict |
|---------|----------------------|-------|---------|
| Llama 3.3 70B | ✅ (quantized ~40GB, A100 80GB) | ~128k | **Safe default** |
| Qwen 2.5 72B | ✅ (quantized) | ~150k | Strong code/reasoning alt |
| Gemma (latest, ~27B) | ✅ (fits easily) | **~256k → big student embedding** | OK if you accept the vocab cost. (Gemma 4 specifics unverified.) |
| DeepSeek V3 | ❌ for logits (671B; API gives no full logits) | — | Out for *logit* distill; viable for *synthetic-data* SFT |

Two coupled facts:
- **Teacher = student vocab.** Logit distillation forces a shared tokenizer, so
  picking the teacher silently sets the student's embedding/LM-head size (Gemma's
  256k vocab = a much bigger embedding at 3B).
- **Ouro is out for scale-up** — it's 2.6B, too small to teach a *bigger* student
  to exceed 2.6B. Scaling up *requires* a bigger teacher.

**Split-teacher strategy** (best of both): logit-distill from a self-hostable
teacher (Llama/Qwen/Gemma) for the backbone, **and** generate synthetic
reasoning/code data from a stronger model you can't self-host (DeepSeek V3 API)
for capability injection. (Synthetic API data carries the OpenAI-style ToS
provenance flag — fine for research, a constraint if distributing.)

Resolve the exact teacher **closer to the run** — the open-model landscape moves
fast; there may be a clearly-best option by the time you rent compute.

### Hardware is the upstream gate — and it keeps *both* objectives open

Both distillation and pure-datasets need better/rented compute, so the hardware
investment is required either way — and **it doesn't force the objective choice;
it enables both.** With multiple high-quality cards or a rented node you can fit
*either* a decent teacher (Llama 70B quantized ~40GB) for distillation *or* large
streamed datasets for pretraining. Two consequences:

- **Good hardware makes distillation *easier*, not just possible** — on a
  multi-A100 / multi-card node you host a big teacher *and* the student
  comfortably, with room for bigger batches. The cramped teacher-hosting that
  forced the v1 teacher onto the 5060 disappears.
- **The objective can be deferred to run-time** — rent the node, see what fits
  and what the budget allows, then choose. Default lean stays **distillation**
  (sample-efficient → *fewer rented hours → cheaper total run*, even with the
  teacher's per-step overhead); go pure-datasets only to exceed the teacher and
  if you can afford the extra hours.

### Cloud rental — providers to evaluate (research in progress)

Renting is the realistic unstick path. Providers to compare (verify current
pricing — it moves):

| Provider | Note |
|----------|------|
| **RunPod** | Popular, cheap, per-hour A100/H100, community + secure tiers — common budget pick |
| **Vast.ai** | Marketplace, often cheapest (community hosts), variable reliability |
| **Lambda Labs** | ML-focused, A100/H100, solid middle ground |
| **Paperspace / DigitalOcean** | Managed, easy onboarding |
| **CoreWeave, Hyperbolic, TensorDock** | Newer / larger-scale options worth checking |
| **AWS / GCP / Azure** | Most reliable, most expensive; spot instances cheaper |
| **Intel Tiber Developer Cloud** | For the **B70/Intel port test** specifically (Arc/Max GPUs) |

Two separate rentals, different purposes (don't conflate):
- **NVIDIA (A100-class)** → the actual from-scratch-distilled training run
- **Intel (Arc/B70-class)** → the cheap port-feasibility + tok/s test before any B70 purchase

### Still to decide (remaining open sub-questions)
- **Target size** (3B confirmed as "train big"; could go larger if compute allows)
- **Data volume/mix** for the run — the data-scale plan (billions of tokens) is its own undertaking
- **Cloud cost budget** (provider rate × A100-hours for the chosen step count) — pick provider first, then the math
- **Cloud provider** — evaluate the table above against current pricing

---

## Long-term: requires hardware upgrade or cloud compute

### Out of scope on current hardware

| Direction | Why parked |
|-----------|-----------|
| 3B+ training from scratch | 50+ days continuous compute |
| Real foundation pretraining (100B+ tokens) | Even at 1B, multiple months on this hardware |
| Frontier model competition | Different problem class, ~$M of compute |

### Hardware-scaling analysis

#### Current rig — three GPUs, use them by *role*, not FSDP

The workstation has three cards, all **native bf16** (no fp16 conversion needed).

> **Status: PROPOSED role-separation — not yet exercised.** The 3-card setup has
> only been *discussed*, never run. Actual history: **v1 distillation used 2
> cards** (5070 student + 5060 teacher); **v2–v5 (SFT + growth) used 1 card**
> (5070 only). The **4060 has never been used in a training session.** The table
> below is the *suggested* allocation if/when you run a multi-card workflow, not
> a description of what's been done.

| Card | VRAM | Gen | Proposed role (not yet used as a trio) |
|------|------|-----|-----------|
| RTX 5070 | 12 GB | Blackwell (fastest) | **Primary training** — single-card, native bf16, no sync overhead (this *is* what v2–v5 used) |
| RTX 5060 | 8 GB | Blackwell | **Teacher host** during distillation — where the v1 teacher ran (Ouro-2.6B ~5.2 GB bf16) |
| RTX 4060 | 8 GB | Ada | **Parallel eval** (proposed) — run the harness on saved checkpoints while the 5070 trains. *Never used yet.* |

**Parallel training runs on this rig (assessed 2026-06-11):** placement, not
aggregate VRAM, is the blocker. Four pieces (2 students + 2 teachers) don't fit
3 cards cleanly — with teachers on the 5060 + 4060 (both proven hosts), the
second *student* has no slot, and squeezing two students onto the 5070 halves
each run's compute (parallel ≈ sequential wall-clock, plus OOM risk). **The
unlock is a shared teacher server:** both runs use the identical frozen
teacher, so ONE copy on the 5060 serving logits to two training processes
frees the 4060 for a second student — true 2× run throughput, comfortable
margins. ~A session of code (IPC + batched serving); worth building before any
seed sweep or the P2.6 config matrix. (A 24 GB card achieves the same by
hosting a full student+teacher pair on one device — another concrete entry in
the more-VRAM ledger.)

**Do NOT FSDP these three together for training.** MythOuro is *compute-bound*
(the recurrent loops multiply compute per step without multiplying
communication), so the PCIe-sync penalty (no NVLink) hurts, and the
heterogeneous cards mean the slowest gates every step. Role-separation
(teacher on 4060, eval on 5060, train on 5070) sidesteps the sync cost
entirely — the teacher forward is one transfer/step, not gradient all-reduce.

Ceiling with this rig: **~1B single-card** (8-bit Adam + growth), stretchable to
**~1.3–1.5B** via FSDP student on 5070+5060 with the teacher on the 4060, if you
accept the PCIe penalty.

#### The bf16-generation rule for any GPU purchase

The single most important hardware caveat, learned the hard way: **buy Ampere
generation or newer.** bf16 was introduced with Ampere (A100). Older datacenter
/ workstation cards lack hardware bf16:

| Card | Gen | Native bf16? | Verdict for this project |
|------|-----|--------------|--------------------------|
| V100 (any) | Volta | ❌ | bf16 runs at fp32 speed (no tensor cores); fp16 needs revalidation |
| Quadro RTX 8000 / 6000 | Turing | ❌ | Same trap — 48 GB is tempting but no bf16 |
| A100 40/80 GB | Ampere | ✅ | Ideal — native bf16 + tf32, datacenter-grade |
| RTX A6000 48 GB | Ampere | ✅ | Excellent — most VRAM, ECC, workstation |
| RTX 6000 Ada 48 GB | Ada | ✅ | Newer, faster, pricier |
| RTX 5090 32 GB | Blackwell | ✅ | Consumer, fast, fits 3B (tight) |

Running the whole validated pipeline on a non-bf16 card means converting to
fp16 and re-validating every stability property (ACT collapse, MoE balance,
LTI spectral radius, depth regulariser). MythOuro's *bounded-activation* design
(ρ(A) < 1, normalized routing, sigmoid halting) makes fp16 tractable, but it's
work you avoid entirely by staying on Ampere+.

#### Purchase options for the 3B goal

3B needs **~24 GB minimum, ~28–32 GB comfortable** (8-bit Adam). The current
20 GB pooled can't fit it — 3B requires new hardware regardless of growth-vs-
from-scratch. Compared honestly:

| Option | $ | VRAM | bf16 | Reaches 3B? | Caveats |
|--------|---|------|------|-------------|---------|
| **2× V100 16 GB SXM2 + baseboard** | ~$500 | 32 GB (NVLink) | ❌ fp16 | Yes (FSDP) | SXM2 = build project (cooling/power); fp16 revalidation; old, slow compute. Viable for a maker with shop skills (AIO + milled bracket); the cheap $/GB path. |
| **RTX 5090 32 GB** | ~$2k | 32 GB | ✅ | Yes (tight) | Single card, native bf16, newest compute |
| **Used RTX A6000 48 GB** | ~$3–4k | 48 GB | ✅ | Yes, comfortable | Single card, ECC, native bf16, NVLink-bridge — the "do it properly" pick |
| **Used A100 40 GB** | ~$3k | 40 GB | ✅ | Yes, comfortable | Datacenter HBM2, native bf16, NVLink-capable |
| **Intel Xeon Max 9480** (1S, 64 GB HBM2e) | ~$3k/chip (used) + SPR board | 64 GB HBM | ✅ (AMX) | Yes, comfortable | **CPU with AMX matrix units + on-package HBM.** ~95 effective dense-BF16 TFLOPS single-socket (~3× a lone 5070's ~34), passes the native-bf16 gate. Capacity play: 64 GB fits 3B + teacher with no FSDP. See assessment below. |

Because MythOuro is compute-bound, a **single fast native-bf16 card beats a
fast interconnect between slow cards**: even 2× V100 NVLink in software-bf16 is
slower per step than the current 5070+5060 over PCIe (the V100's emulated bf16
runs at ~fp32 speed, and the recurrent loops make compute, not comms, the
bottleneck). The V100 path only wins on *price* and on *fitting* models that
don't fit otherwise — not on speed.

##### Intel Xeon Max 9480 (AMX + HBM) — assessed 2026-06-08

Considered as an upgrade from the current consumer rig (a lone 5070 sustains
~34 dense-BF16 TFLOPS; 5070+5060 pooled ~55). Verdict: **a legitimate
single-socket upgrade — but verify the real number before buying.**

- **For — vs. the *actual* current baseline** (not vs. an H100): single-socket
  9480 is ~95 effective dense-BF16 TFLOPS (≈3× the 5070) **and** a 64 GB HBM
  pool that finally fits a 3B + teacher with no mixed-card FSDP. AMX has native
  BF16, so it dodges the V100 fp16-revalidation trap. On every axis that matters
  vs. today's hardware, it's up.
- **Against — the caveats that shrink the win:**
  - **The recurrent-loop tax.** Headline tok/s estimates assume a dense
    single-pass model; MythOuro runs the recurrent block `n_loops`× per token
    (×4 train), so real sequence throughput is ~/4 of the dense-model figure.
    A "~46k tok/s on a 650M dense model" estimate is closer to **~11–13k tok/s**
    for an equivalent MythOuro.
  - **Small-matmul derating.** ~95 TFLOPS is big-square-GEMM peak; MythOuro's
    small matmuls (MoE experts, MLA, per-loop LoRA) sustain a fraction of it on
    *any* engine. Whether AMX holds a higher fraction than a GPU here is an
    empirical question — could break either way.
  - **Dual-socket ≠ one big accelerator.** 2S is *two* 64 GB HBM pools over UPI
    (~hundreds of GB/s cross-socket, not a unified 3 TB/s); NUMA reintroduces
    sharding penalties — partly the thing you're trying to escape. The clean
    story is **single-socket**.
  - **CPU/AMX software maturity.** The pipeline is CUDA-validated; moving to CPU
    BF16 (oneDNN/IPEX) needs re-validation, and custom ops (recurrent loop, MoE
    `index_add_` dispatch) may not hit optimal AMX kernels without tuning.
- **Decision rule: measure, don't extrapolate.** Peak TFLOPS is a spec sheet;
  the buy hinges on *achieved* tok/s on MythOuro. Rent an AMX instance (Intel
  Tiber Developer Cloud — already listed below for the B70 port test) for an
  hour and run [`tools/bench_step.py`](../tools/bench_step.py) on it and on the
  5070 for a true apples-to-apples (`python -m tools.bench_step --variant
  mythouro_distill_tiny --device {cpu|cuda:0}`). If single-socket lands ≥2×
  real-world *and* fitting 3B locally is worth ~$3k+platform to you, it's a
  defensible buy. If it lands ~1–1.5×, rent GPU hours instead.

**Measured on the on-hand ES 8480 (2026-06-08) — AMX runs, but DDR5 starves it.**
Benchmarked on a Sapphire Rapids **8480 engineering sample** (56C, 4.7 GHz turbo,
**2-of-8** DDR5-4800 channels populated, ASUS W790, Windows, stock
`torch==2.12.0+cpu`).

*Correcting an earlier wrong call in this doc:* AMX **is** engaged.
`ONEDNN_VERBOSE=1` shows bf16 matmuls dispatching to the
`brg_matmul:avx10_1_512_amx` kernel — the AMX path, exactly as HWiNFO's
feature list reports. The bottleneck is **memory bandwidth, not the tiles.**

Steady-state bf16 GEMM throughput vs working-set size (≈100 MB L2+L3):

| Matrix | TFLOPS | Note |
|--------|-------:|------|
| 1024³ | 7 | cache-resident |
| 2048³ | **16** | ~25 MB, fits → AMX shining |
| 4096³ | 4 | ~100 MB, thrashes → tiles starve |
| 8192³ | 9 | DRAM-bound |

Cache-resident matrices hit ~16 TFLOPS; once the working set spills to DRAM the
AMX tiles starve and throughput collapses — the classic signature of a
**2-of-8-channel** config (~¼ the platform's bandwidth). This is *precisely* why
an 8480 on DDR5 underperforms a Xeon Max on HBM: identical AMX compute, but HBM
(>1 TB/s) keeps the tiles fed. `tools.bench_step` `distill_tiny` (b1×s256) ran
**85 tok/s** here — the small per-op matmuls + MoE scatter + recurrent loop sit
largely in the memory-bound regime.

- **The bandwidth lever — and why the RAM market tilts it toward HBM.** Filling
  2→8 DDR5 channels is ~4× bandwidth and would lift the memory-bound throughput.
  *But* at current ECC DDR5 RDIMM shortage pricing (32 GB ~$600–800 ea, 16 GB
  ~$300–500), six more DIMMs is **~$3.6–4.8k**, not "a few hundred" — and even
  full 8-channel DDR5-4800 (~300 GB/s) still partially starves AMX on large
  problems. A used **Xeon Max 9480 (~$3k)** bundles **64 GB HBM2e (>1 TB/s)** —
  ~3× the bandwidth of a maxed DDR5 rig, no DIMM tax — aimed exactly at the
  16→4 TFLOPS collapse measured above. So in *this* RAM market, building around
  an HBM Max is both faster and cheaper than feeding the 8480 with RDIMMs.
- **Decision (2026-06-08): dedicated 1S C741 Xeon Max rig.** Build a *separate*
  box — Gigabyte **MS33-CE0** (single-socket LGA4677 / C741, 8-channel) or
  similar MS33/MS03 — around a **Xeon Max 9480**, run **HBM-only mode** (64 GB,
  no DIMMs → dodges the RDIMM shortage entirely; fits a 3B + teacher with
  *streamed* data). This resolves the earlier platform-compat worry (proper
  server board, not the W790 workstation board the 8480 ES currently sits in) and
  frees the work rig. **1S (MS33), not 2S (MS73):** for a single training job one
  socket avoids the cross-socket NUMA penalty (2S = two 64 GB HBM pools over UPI,
  *not* a unified 1.6 TB/s); reserve 2S for parallel jobs or a 128 GB need.
- **Build gotchas:** 9480 is 350 W and needs a narrow-ILM LGA4677 server cooler
  + real airflow; a Max *ES* carries the same clock/stability caveats as the
  8480 ES. Expectation: raw bf16 GEMM ~3–5× the starved 8480, but end-to-end
  MythOuro tok/s gains *less* (small matmuls + recurrent-loop tax persist) — the
  honest target is "finally trains a 3B at usable speed," not "95 TFLOPS of model
  throughput." Run `tools/bench_step.py` on it day one for the real number.
- Still **additive compute + 3B-capacity, not a GPU replacement**: even HBM-fed,
  the Max's AMX (~95–175 TFLOPS 1S–2S) is below a modern GPU's tensor throughput,
  and MythOuro's small matmuls + recurrent loop won't saturate it — but it *fits*
  a 3B the 12 GB card can't, with HBM removing the starvation.
- **Revised takeaway:** AMX is real and working on this box — it's
  *bandwidth-limited, not software-limited*. My earlier "AMX not engaged / needs
  IPEX+Linux" and "just cheaply fill 8 channels" framings were both wrong. Given
  RAM pricing, an HBM Xeon Max is the rational build target for an AMX path;
  validate platform compat, and remember it complements (not replaces) the GPU.

*(Side effect of this exercise: it surfaced and fixed a latent autocast bug —
`MoEFFN.index_add_` dispatch wasn't dtype-consistent under mixed precision. The
CUDA training path dodged it by dtype coincidence; now fixed for all paths.)*

##### Xeon Max build checklist (decided 2026-06-08)

Committed to a dedicated Max rig — the deciding factor is **RAM-shortage
arbitrage**: at current ECC DDR5 RDIMM pricing, the 64 GB HBM is effectively
*free fast memory* bundled into the CPU, sidestepping ~$4k of DIMMs *and* fitting
a 3B. This holds regardless of the exact tok/s. (Pre-buy benchmarking is off the
table — Intel Tiber Cloud's Xeon Max signup is dead and the part is too niche to
rent elsewhere — so the plan is **buy, measure day one**.)

- **CPU:** Xeon Max 9462 / 9460 / 9480 — all share **64 GB HBM2e + ~1.6 TB/s**;
  the only difference is AMX compute (~68 / 78 / 95 TFLOPS peak ≈ 32 / 40 / 56
  cores). HBM feeds them equally, so **more cores = strictly faster** for this
  compute-bound workload — pick by budget. An **ES** is the cheap route (accept
  the clock/stability variance, as on the current 8480 ES).
- **Board:** all three 1S Gigabyte C741 boards **support the Xeon CPU Max
  Series** (verified on Gigabyte's own spec pages — third-party retailer listings
  understate this):
  - **MS03-CE0** — ATX, 8 DIMM, 7× PCIe Gen5. **Pick for this build:** ATX fits a
    standard case + the on-hand LGA4677 AIO, and has the most GPU/expansion slots.
    *(Note: "MS33-CE0" doesn't exist — CE0 is the MS03 ATX line.)*
  - **MS33-AR0** — E-ATX, 16 DIMM, 8× SATA. Valid Max board; choose if you want
    E-ATX with more DIMM/storage headroom (irrelevant on HBM-only).
  - **MS33-CP0** — E-ATX, 16 DIMM, OCP 3.0 + MCIO. Choose for OCP networking.
  - **Not the 2S MS73** for single-job training (two HBM pools over UPI = NUMA,
    not a unified pool).
  Verify the exact Max SKU is on the board's CPU QVL and the BIOS exposes HBM
  mode before buying, especially for an ES chip.
- **Memory mode:** **HBM-only** (64 GB, no DIMMs) — the whole point. Fits a 3B +
  activations with *streamed* data. Add DDR (HBM-caching mode) only if a slower
  capacity tier is later needed; for training it isn't.
- **Cooler:** LGA4677 **AIO** (already running one on the 8480 — carries over;
  handles the 350 W).
- **Software:** AMX works **out of the box on stock Windows `torch+cpu`** —
  confirmed: `ONEDNN_VERBOSE` shows `avx10_1_512_amx`. No IPEX/Linux required
  (that earlier worry was wrong); they're optional later tuning.
- **Day one:** `python -m tools.bench_step --variant mythouro_distill_tiny
  --device cpu --batch 8 --seq-len 512`, then rescale the training-time table by
  `(measured ÷ estimate)`. Confirm AMX is firing with `ONEDNN_VERBOSE=1` on a big
  bf16 GEMM (want tens of TFLOPS, fed by HBM — no 2-channel starvation this time).
- **Role / expectations:** dedicated trainer + capacity box for ≤3B that the
  12 GB 5070 can't hold; frees the work rig. The 5070 stays the **fast** card for
  ≤1B-that-fits. Training a 3B on the Max is still slow (CPU-AMX class, ~year
  scale) — for a *fast* 3B run, rented A100/H100. The Max = **fit + iterate +
  hold**, not raw speed.

#### Other accelerants

| If you get… | Then unlock |
|-------------|-------------|
| **Cloud A100/H100 hour budget** | 5–10× faster training; 20K-step runs become single overnights |
| **DeepSeek V3 / Llama 3.3 70B API access** | Stronger teacher → break past the Ouro-2.6B quality ceiling regardless of student size |

### Post-training pipeline stages (mostly hardware-independent)

Beyond pretraining + SFT, the full pipeline has more stages we haven't touched:

| Stage | Cost | What it unlocks |
|-------|------|-----------------|
| **Preference tuning** (DPO / ORPO / KTO) | ~1–2 nights at 1B scale | Smooth, helpful style; reduces refusals/over-refusals |
| **Reasoning RL** (GRPO-style with verifier rewards) | ~3–5 nights | Math/code/logic improvements; this is where Ouro-Thinking gets its edge |
| **Tool use / function calling** | ~1 night SFT | Capability expansion (API calling, code execution) |
| **Self-improvement loop** | open-ended | Diminishing returns, but a real research direction |

Each is realistic on the current hardware once we have a stable 1B base.

---

## Inference efficiency (deployment phase)

How to make a *trained, frozen* MythOuro run faster/cheaper. Planning captured
now; **not a near-term action** (see scale caveat).

### ⚠️ Scale caveat — read first
Quantization's payoff is **scale-dependent**, and it's **modest at 632M**:
- A 632M model is ~1.3 GB in bf16 — it **already fits trivially**, so the 2–4×
  *memory* win is moot at this size.
- Quant *speedup* comes mostly from reduced memory **bandwidth**, which dominates
  for **large, bandwidth-bound** models (7B+). A small model that's
  **compute-bound on the recurrent loops** sees little gain from weight quant.

So inference-efficiency work is a **large-model, deployment-phase** lever. Don't
burn a session quantizing the current model expecting big wins — the payoff
arrives at 7B+ scale and/or real serving load.

### QAT — when and when not (decided 2026-06-10)

**QAT does not improve training** — it inserts simulated quantization noise so
the *deployed quantized* model loses less accuracy, at the cost of ~10–30%
slower steps and a slightly worse full-precision model. Therefore: **no QAT for
any current-scale training** (≤1B isn't deployed quantized — nothing to gain).
The one planned use: the **3B → INT4 deployment path** — standard play is bf16
training → **short QAT finetune at the end** → quantize (never QAT-from-
scratch). The component-aware map below still governs what stays high-precision
(router, LTI A/B, ACT/uncertainty heads); `inference.py`'s existing
`quantization_aware_training_hooks` already skip those. Not to be confused with
8-bit Adam (optimizer-state quantization, already in use, unrelated to deployed
precision).

### The three approaches (for when it's worth doing)

| # | Approach | When | Notes |
|---|----------|------|-------|
| **A** | **torchao quantization in PyTorch** | first / always | No ONNX export — handles the dynamic ACT loop, MLA, MoE natively. Biggest lever, least friction. |
| **B** | Fixed-loop static export → ONNX Runtime / TensorRT | if more speed needed, model frozen | Must unroll ACT to a fixed K (lose adaptive depth); MLA/MoE need plugins or decomposition. ONNX RT over TensorRT initially (more op-tolerant). |
| **C** | Hybrid: static parts to a runtime, ACT/routing in PyTorch | best-of-both, more eng | Quantized matmuls + flexible dynamic control. |

### MythOuro export obstacles (why it's not "export and go")
- **ACT dynamic loop** — data-dependent termination doesn't export to a static
  ONNX graph cleanly. The biggest blocker for B/C. (Same problem the Rust
  ACT-compaction work solves — see below.)
- **MLA** — no fused TensorRT/ONNX kernel; decomposes (slow) or needs a plugin.
- **MoE dispatch** — exports inefficiently; may need a plugin.

### Component-aware quantization map (torchao, Approach A)
Mixed precision: quantize compute-heavy matmuls, **protect** stability-critical
parts. Use the *real* module names below (from `model.named_modules()`), not
guessed regexes.

| Component (real FQN) | Quantization | Why |
|----------------------|--------------|-----|
| `recurrent...routed_experts.{i}.{gate,up,down}` | INT4 weight-only (group 64) | Heavy, compute-dense, tolerant if calibrated |
| `...shared_experts.{i}.{gate,up,down}` | INT4 weight-only | Same |
| GQAttention/MLAttention projections (q/k/v/o, MLA latents) | INT4/INT8 weight-only | Matmul-heavy; watch outliers |
| prelude/coda block FFNs (`Expert`) | INT4/INT8 weight-only | Standard heavy matmuls, safe |
| `...router` (Linear) + `router_bias` | **keep BF16/INT8** | Low-bit destabilises expert selection |
| `LTIInjection` A/B | **keep BF16** | Quant can push ρ(A) ≥ 1 → divergence |
| `ACTHalting` head, `UncertaintyHead` | **keep BF16** | Halt/confidence fragile; errors compound across loops |
| KV cache (if long-context) | INT8/FP8 dynamic | Big win at long context |

Progression: INT8 baseline → measure (PPL, routing entropy, avg ACT depth,
ρ(A), expert utilisation, output KL) → push experts/attention to INT4 → if
quality dips, per-component sensitivity ablation or light QAT on the sensitive
heads. **torchao API + FQN patterns must be verified against the installed
version** (the API evolves; don't trust copy-pasted config dicts).

### Convergence with the Rust path
The ACT dynamic-loop obstacle for TensorRT is *the same* problem the Rust+candle
runtime solves via active-set compaction. So the **Rust deployment runtime
subsumes the TensorRT path** for this architecture — it handles ACT natively
where TensorRT struggles. If you build the Rust runtime, you likely skip
TensorRT. Quantization (torchao) is orthogonal and applies in either world.

---

## Deployment & language strategy

A settled decision, recorded so it isn't re-litigated. (Explored in depth
2026-06-06: Rust vs C++ vs Zig vs Jule for a faster MythOuro.)

### The core finding: host language is NOT the efficiency lever

Model **capability** lives in the weights (training-determined) and is
**language-independent** — a Rust/C++/Python MythOuro with the same weights
produces identical outputs. Model **execution efficiency** is what a language
choice can affect, but even there the host language is the *smallest* lever,
because all of them call the same GPU kernels (cuBLAS/cuDNN) and hit the same
ceiling. The real efficiency levers, in order:

1. **Quantization** (INT8/INT4) — 2–4×, the biggest, language-agnostic
2. **Inference compiler/runtime** (TensorRT / ONNX Runtime / TVM) — kernel fusion
3. **Custom kernels** (Triton/CUDA) for the architecture-specific hot paths
4. **Host language** (Rust/C++/Zig/Jule) — marginal (overhead only)

So: don't language-hunt for efficiency. For inference efficiency → quantize +
compiler. For the *training* bottleneck → rented GPU / faster card (a language
can't add TFLOPs).

### The plan (phased)

| Phase | Language | Rationale |
|-------|----------|-----------|
| **Research / now** | **Python + PyTorch** | Max iteration speed, mature ecosystem, HF teacher + datasets live here. Correct for the exploration phase — proven by the findings this project produced. |
| **Scale-up** | Python + PyTorch | Same stack, bigger model, rented/upgraded compute. No language change. |
| **Deployment (post-coherence)** | **Rust + candle, C++ FFI for custom kernels** | Once there's a coherent model worth serving *and* a serving need. |

### Deployment-phase specifics (when reached)

- **Rust + [candle](https://github.com/huggingface/candle)** — inference-shaped,
  direct tensor/memory control, flash-attn binding. Preferred over `burn`
  (don't need autodiff/training abstraction for inference).
- **C++ FFI only for the gaps** — candle covers standard ops (matmul, softmax,
  RoPE, RMSNorm, attention). Bridge to hand-written **CUDA C++ kernels** only
  for the architecture-specific hot paths: **ACT active-set compaction**
  (gather still-looping tokens, compact compute but not the KV cache) and
  **MoE sorted-dispatch**. Same FFI pattern the maintainer already proved on a
  prior slicer project. `cudarc` is an option to call kernels from Rust without
  a C++ layer.
- **The architecture-specific upside**: ACT variable-compute is a first-class
  citizen in Rust (compaction) rather than a masked-overcompute afterthought as
  in PyTorch — a genuine ~1.5–2× inference win on the recurrent block, *specific
  to this architecture*, not generic "Rust is fast."
- **Not C++/Zig/Jule**: same efficiency ceiling as Rust (kernels dominate);
  Rust gives memory safety + the candle ecosystem. Jule is pre-stable (v0.2.2,
  2026) with no ML/GPU ecosystem — interesting language, wrong tool here.

**Trigger to start this phase**: a coherent, deploy-worthy model (post scale-up)
+ an actual serving/latency/edge requirement. Not before — it optimizes
inference (not the current bottleneck) of a model that isn't coherent yet.

---

## Application layer (far horizon) — RAG / retrieval

The furthest-downstream thread: features built *around* a coherent MythOuro,
not improvements *to* the model. Captured so the three "later" threads stay
cleanly separated (they're easy to conflate).

### The three downstream threads, disambiguated

| Thread | Layer | What it touches | Phase |
|--------|-------|-----------------|-------|
| Inference efficiency (torchao / ONNX / TensorRT) | the model | makes the *model* run faster | deployment |
| Rust + candle runtime | the model | efficient *model* execution (ACT compaction) | deployment |
| **RAG / retrieval (this section)** | **around the model** | an *application feature*, retrieval over a corpus | **application — furthest out** |

### RAG and where turbovec fits

**RAG (Retrieval-Augmented Generation)** = MythOuro *generates*, a vector store
*retrieves* relevant documents to inject into the prompt. It makes a *coherent*
model more **factual/grounded**; it **cannot** make a scale-limited model
coherent. So it sits in the "deliberately NOT yet" bucket until there's a model
good enough to *use* retrieved context — bolting RAG onto a 632M gibberish model
does nothing.

**turbovec** ([RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)) —
a candidate retrieval backend for that eventual RAG layer:
- **What it is**: vector search / approximate-nearest-neighbor (ANN) tool.
  Compresses embedding vectors and does fast similarity search. Rust + Python
  bindings, MIT, production-ready (~7k stars). Uses Google Research's
  *TurboQuant* (data-oblivious quantizer, no codebook training).
- **Claims**: ~16× compression on 1536-dim vectors; beats FAISS IndexPQFastScan
  12–20% on ARM, matches/exceeds on x86; filtered search with no recall penalty.
- **Fits the project ethos**: local-first, private, no managed service / no data
  leaving the machine — aligns with the "best *local* LLM" goal.

### ⚠️ Naming trap: turbovec quant ≠ model quant
turbovec's "quantization" compresses **embedding vectors** (for search). The
torchao/TensorRT quantization in the Inference-efficiency section compresses
**model weights** (for faster LLM inference). **Same word, unrelated jobs,
different layers of the stack.** Do not conflate them — turbovec is *not* on the
inference-efficiency path; it's retrieval infrastructure for a RAG application.

### Trigger
A **coherent** MythOuro (post scale-up) **and** a use case where it answers from
a document corpus. Furthest-downstream item on the roadmap — past scale-up, past
basic deployment. Alternatives at that point: FAISS, or other ANN libraries;
turbovec is bookmarked as the local-first, high-compression option.

---

## Data roadmap

What data feeds which stage. The SFT stage is built; the rest are planned and
listed here so the data decision is made before the engineering, not during.

| Stage | Status | Datasets | Format / notes |
|-------|--------|----------|----------------|
| **Distillation** | ✓ done | FineWeb-Edu (40%) · open-web-math (40%) · codeparrot-clean (20%) | Raw text + teacher logits. Math-heavy to transfer Ouro-Thinking's reasoning. |
| **SFT** | ✓ done (v2–v4) | OpenHermes-2.5 (30%) · MetaMathQA (40%) · Magicoder-Evol-Instruct (30%) | ChatML, loss masked to response tokens. Use `seq_len≥1024` for OpenHermes (multi-turn) — at 512 it's ~95% rejected. |
| **Preference tuning (DPO/ORPO)** | 📋 planned | `HuggingFaceH4/ultrafeedback_binarized`, `Anthropic/hh-rlhf` | Needs `(prompt, chosen, rejected)` triples. ORPO folds it into one stage (no separate reward model) — simplest for a solo overnight setup. Start small (~10k pairs at this scale). |
| **Reasoning RL (GRPO)** | 📋 planned | GSM8K + MATH (train splits) with a **programmatic verifier** as the reward | The reward is a correctness checker (parse final answer, compare), not a learned reward model. This is where the recurrent-depth architecture should shine — more loops on harder problems. Memory-heavy (multiple rollouts/step); needs the 8-bit-Adam + small-batch budget. |
| **Tool use / function calling** | 📋 later | `glaiveai/glaive-function-calling-v2`, ToolBench subsets | SFT-style, but responses contain structured tool-call tokens. Define the call schema first. |
| **Long-context** | 📋 later | Curated long documents (books, repos) | Only meaningful once a model is coherent; the Ouro tokenizer supports 131k but nothing's been trained past 1k. |

### Planned domain expansion: science + medical (user goal, 2026-06-10)

Add science/medical as a data *kind* across the stages — per the variety
principle below, a new domain is exactly the kind of addition that has moved
behaviour at this scale. Candidates by stage (verify licenses at use time):

| Stage | Candidate datasets | Notes |
|-------|--------------------|-------|
| **Distillation / pretrain mix** | `allenai/peS2o` (open academic papers, clean license) · RedPajama **arXiv** subset · **PMC Open Access** (biomedical full text) · PubMed abstracts | Slot into `MixedDataset` as new sources with their own mix ratios — config work, no new code. For the 3B from-scratch run, include from day one rather than retrofitting. |
| **SFT** | `SciQ` (science QA) · `MedMCQA`, `PubMedQA`, `MedQA-USMLE` (medical QA → instruction format) · CAMEL science dialogues | Same ChatML / loss-mask pipeline as OpenHermes. **Provenance check per set** — several popular medical-chat sets (ChatDoctor, Medical Meadow variants) are GPT-derived → same OpenAI-ToS flag as OpenHermes (fine for research, a constraint if distributing). |
| **Eval** | MedQA / PubMedQA / SciQ accuracy | Add to `eval/metrics.py` alongside ARC — same cloze/log-likelihood pattern, small lift. |

**Safety note (medical specifically):** at proof-of-concept scale this is a
*domain-data experiment*, not a medical model — outputs are research artifacts,
never advice. If the domain survives to a coherent-scale model, revisit the
parked "safety alignment" roadmap item *before* anything medical-flavoured is
distributed or served.

**General principle observed so far:** at this scale, *data variety* moved
behaviour more than *data volume* or *parameter count*. Adding OpenHermes (v4)
unlocked the social-prompt register and recovered the confidence-halt that MoE
growth had blurred — a data change, not a scale change. Prefer adding a new
*kind* of data over more of the same.

---

## External eval baselines (context for the numbers)

Our metrics in isolation don't say whether they're good. Rough anchors for
small models at comparable scale (held-out web-text perplexity; exact numbers
vary by tokenizer/corpus, so treat as order-of-magnitude):

| Model | Params | Train tokens | Ballpark PPL | Note |
|-------|-------:|-------------:|-------------:|------|
| **MythOuro v1 (distill)** | 278M | ~20M | **37** | Ours — but distilled, so PPL is teacher-shaped, not from-scratch |
| GPT-2 small | 124M | ~40B | ~30–35 | ~2000× more tokens than ours |
| Pythia-410M | 410M | ~300B | ~12–15 | ~15000× more tokens |
| GPT-2 medium | 355M | ~40B | ~22–26 | — |

**The honest takeaway:** our PPL ~37 is *reasonable for the token budget* (we
trained on ~20M tokens vs. tens of billions for the others — distillation is why
it's even comparable). But coherent generation empirically needs both more
params (~1B+) and more tokens (~10B+) than the workstation can reach. The gap to
"usable" is **scale**, and these baselines quantify roughly how far: ~3× the
params and ~500× the tokens to reach Pythia-410M territory.

---

## Glossary

Project-specific terms used throughout the code and docs:

| Term | Meaning |
|------|---------|
| **RDT** | Recurrent-Depth Transformer — the core architecture (Prelude → looped Recurrent block → Coda) |
| **ACT** | Adaptive Computation Time — per-token learned halting; decides how many loops each token needs |
| **halt distribution** | Per-loop probability mass of where tokens stop looping; should be *spread*, not pinned to loop 1 (pinned = collapse) |
| **loop_efficiency** | avg_halt_depth / max_loops — how much of the available depth the model actually uses (~0.5 = genuinely adaptive) |
| **LTI / ρ(A)** | Linear Time-Invariant injection; ρ(A) is its spectral radius. Must stay < 1 or the recurrence diverges. |
| **MoE** | Mixture of Experts — routed + shared FFN experts; only top-k routed experts fire per token |
| **cv / max% / min%** | MoE utilisation stats: coefficient of variation across experts, and the most/least-used expert's share. Low cv = balanced. |
| **router_bias** | DeepSeek-V3 aux-loss-free load-balancing bias on router logits; nudged outside the optimizer toward uniform expert use |
| **sentinel-bias** | Large negative router_bias on newly-grown experts at promotion, decayed over N steps — keeps MoE expansion function-preserving |
| **depth-reg** | PonderNet × Ouro KL-to-uniform regulariser on the halt distribution; prevents ACT loop-collapse |
| **resp_frac** | Fraction of tokens in an SFT batch that contribute to the loss (response tokens, not prompt/padding) |
| **soft / hard loss** | Distillation: `soft` = distance to teacher logits, `hard` = CE against gold tokens |
| **ECE** | Expected Calibration Error — how well predicted confidence matches actual correctness (lower = better calibrated) |
| **UncertaintyHead** | Per-token uncertainty predictor; the `ConfidenceAwareGenerator` uses it to halt/refuse low-confidence generation |
| **h_K (emission)** | The final loop's hidden state — what the model emits at BOTH training and inference since P0.3. The old eval path emitted the ACT-weighted blend `h_out`, a never-trained mixture (removed). |
| **gnorm** | Gradient norm before clipping (clip caps actual updates at 1.0). Healthy: single digits. Hundreds–thousands = chaotic landscape; the step still has norm 1 but points in noise directions → learning stalls (the flatline-incident signature). |
| **LoopCurriculum / `--start-loops` / `--random-depth`** | Training depth schedule: hold at `start_loops`, ramp linearly to `max_loop_iters` by mid-run. `--random-depth` samples each step's depth uniformly in [start, current cap]. Caveat: depths below `start_loops` are never emission depths → uncertainty head uncalibrated there (the loop-0 finding, P0.5). |
| **warmup (LR)** | Steps before full learning rate. The proven from-scratch recipe needs 500 — 200 destabilised a fresh 4-loop recurrent model (see failure modes 2026-06-10). Distinct from the *curriculum* warmup and the *new-component* warmup. |
| **best-of-trajectory** | Inference experiment: score every loop's output with the UncertaintyHead, emit the most-confident depth per token. Diagnostic-grade (no KV cache); `min_loops=2` default excludes the miscalibrated loop 0. |
| **forced-depth probe (`--force-full-depth`)** | Measurement override suppressing ACT's early exits so all `n_loops` run — exposes the counterfactual loops ACT skips ("would deeper have helped?"). Pure measurement, weights untouched. |
| **per-loop calibration / best-exit teacher** | ECE measured per loop index (`tools/per_loop_calibration.py`). The "best exit" per token (lowest per-loop CE) is MoDr's supervision target — per-loop CE, NOT uncertainty-argmin (loop-0 miscalibration would poison it). |
| **MoDr** | Mixture-of-Depth routing (candidate direction): one learned router emitting expert choice AND per-token recurrence depth, trained teacher/student (forced-depth trajectory = teacher; cheap router = student). Gated behind the MoE-vs-dense ablation. |
| **matched-active (ablation)** | The dense arm's FFN width is sized so its params/FLOPs per token equal the MoE arm's *activated* FFN per token (`expert_dim·topk·(1+n_shared)`) — same compute, different total capacity. |
| **sidecar evals** | Per-run eval JSONs copied into the run's checkpoint dir immediately after a run — because `eval_results/` filenames collide across runs (caused a false alarm 2026-06-10). |
| **proven recipe** | v1's final successful hyperparameters (warmup 500, depth-reg 0.3, mb1/ga8), recorded in its MODEL_CARD provenance and now the `distill.py` defaults. Rule: diff against the model card's command, never trust old defaults. |
| **P0 / P1 / P2** | Review priorities: P0 = correctness (fix before training), P1 = perf/measurement validity, P2 = strategic. Tracker: `docs/review_action_plan.md`. |
| **XPU / `rope_real`** | `torch.xpu` = Intel GPU backend (native ≥ PyTorch 2.5); `mythouro/device.py` abstracts cuda/xpu/cpu. `rope_real` swaps the complex RoPE table for an equivalent cos/sin form for backends without complex-op support. |
| **tok/s (k-suffix!)** | The training log prints `2.5k tok/s` = 2,500. Misreading the k once spawned a multi-day "training is slow" investigation (it wasn't). |

---

## Decision rules

When to apply which technique:

| Symptom | Treatment |
|---------|-----------|
| VRAM at ceiling, want bigger model | `--use-8bit-adam` (working, ~2.5 GB, best MoE fit — quantization is architecture-agnostic) |
| 8-bit Adam not enough, single-card | GaLore/LoRA-Pre (pure PyTorch, installs clean) — but note: low-rank methods are *dense-optimized*; on MoE's many small expert matrices the benefit shrinks and per-matrix SVD overhead grows. Validate on a small model first. |
| Want >1B and 8-bit Adam maxed | FSDP across the rig (accept PCIe penalty) — or buy an Ampere+ card with more VRAM |
| Capacity unused, output still gibberish | Add data variety, not more params (the 278M/420M ceiling is param-count, not architecture) |
| Calibration drifts after promotion (e.g. confidence-halt lost) | More SFT with varied prompt types re-tightens it (v4 recovered all 3 halt mechanisms by adding OpenHermes) |
| Halt discipline gets fuzzier post-promotion | Threshold-retune the ConfidenceAwareGenerator OR more SFT steps |
| New experts stay idle after sentinel decay | Extend training, the bias updater needs time (v3 hit min% 1.1 by step ~2500) |
| bnb "cuda132 binary not found" | `_configure_bnb_cuda_version()` already auto-handles it (picks cuda130 for CUDA 13.2) — see failure modes |

---

## What's deliberately NOT on the roadmap (and why)

| Idea | Why parked |
|------|-----------|
| Rewriting in JAX/Triton | Massive engineering cost, marginal benefit for tonight-scale training |
| Building a custom CUDA kernel for MoE dispatch | We're using torch SDPA fallback anyway; FlashAttention2 unavailable on cuda_cc=(12,0) per our logs |
| Adding RAG | Sidesteps the model-quality problem rather than fixing it. Useful as a layer over the eventual best model, not a substitute |
| Constitutional AI / safety alignment | Premature at proof-of-concept scale; revisit when output is coherent |
| Multimodal extension (vision, audio) | Single problem at a time; pure-LM has to work first |
| Custom MoE routing (beyond DeepSeek-V3) | Current routing is working — `cv 0.19` is exceptional. Don't fix what isn't broken |
| Changing the activation (SwiGLU → ReGLU/GeGLU/etc.) | **SwiGLU/SiLU is the SOTA standard** (LLaMA/PaLM/Mistral) and isn't a bottleneck. The only architectural reason to switch would be an *idempotent* activation (ReGLU) to make Net2Deeper depth-growth function-preserving — but that trades away SwiGLU quality to unlock an axis we don't need (MoE expansion + Net2Wider already cover growth). Also can't be retrofit onto trained weights (would require from-scratch retraining). Considered, rejected. |

> **Note on activations & growth** (corrects an earlier doc error): MythOuro
> uses **SiLU inside SwiGLU** (`down(SiLU(gate(x)) · up(x))`). SiLU's
> non-idempotence only blocks **Net2Deeper (depth growth)**, not **Net2Wider
> (width growth)** — Net2Wider is function-preserving with any element-wise
> activation. So width growth toward ~1B *is* available without changing the
> activation. See [`docs/growth_design.md`](growth_design.md) for the corrected
> analysis.

---

## Licensing & data provenance (gate before distributing weights)

Full writeup in the README's "Licensing & data provenance" section. Short version:

- **Code**: MIT. **Teacher** (`ByteDance/Ouro-2.6B-Thinking`): **Apache 2.0** — clean for distillation/redistribution with attribution.
- **The gating issue**: the SFT datasets (OpenHermes 2.5, MetaMathQA, Magicoder-Evol-Instruct) all contain **OpenAI-generated data**, whose terms restrict training competing models. Fine for private research; a real constraint if distributing/commercialising the SFT'd checkpoints.
- **To get a cleanly-distributable checkpoint**: retrain SFT on non-OpenAI-provenance data (Dolly-15k, documented Tulu-3 subsets, or self-generated from an Apache/MIT model). Distillation data (FineWeb-Edu, OpenWebMath, CodeParrot) is open-provenance and fine.

---

## Active checkpoints (working set, not archived)

- `checkpoints_grown\step_0003500.pt` — same as v3 archive (kept for live use)
- `checkpoints_grown_v4\step_*.pt` — v4 SFT run (420M, 48 experts)
- `checkpoints_grown_v5\step_*.pt` — v5 2nd-expansion run (632M, 96 experts);
  last step 2887. Archived as `mythouro_distill_xl_grown_v5`; **expert-count
  ceiling** data point, not an improvement over v4.

## Sessions log

| Date | Session focus | Outcome |
|------|---------------|---------|
| Pre-2026-05-31 | Architecture build, debugging ACT collapse, distillation recipe | Pipeline working end-to-end |
| 2026-05-31 to 2026-06-01 | Full distillation 5K steps | v1 archived (PPL 37.4) |
| 2026-06-03 to 2026-06-04 | SFT pipeline build + 3K-step v2 SFT | v2 archived |
| 2026-06-04 | MoE expansion design + implementation + v3 grown run | v3 archived (cv 0.19, exceptional routing) |
| 2026-06-05 | OpenHermes re-add + 8-bit Adam wiring; v4 run at seq_len=768 (bnb blocked by CUDA 13.2) | v4 archived — all 3 halt mechanisms firing, 4/4 prompts halt cleanly |
| 2026-06-10/11 | Ablation arm 2 (dense, seed 0) trained overnight | **MoE-vs-dense seed-0 verdict: MoE wins 4.0×** (dense 22.66 vs MoE 5.72 final PPL; gap grows with training). MoE retained per pre-registered rule, pending seed 1. Dense still beats v1 — fixes+recipe lifted everything. |
| 2026-06-09/10 | External code review (Fable 5): P0.1–P0.5 + most P1s fixed, invariant tests added; v2 re-baseline (PPL 46.3→39.25 from P0.3 alone); per-loop calibration audit (loop 0 miscalibrated → MoDr target = per-loop CE); GPU smokes both training paths; first ablation attempt flatlined → root cause: script defaults ≠ v1's proven recipe (warmup 500/depth-reg 0.3) → defaults fixed | **Ablation arm 1 (MoE, seed 0) COMPLETE on fixed code + proven recipe: final PPL 5.72 vs v1's 37.4 (6.5× better in 1k fewer steps), loop_eff 0.500, ECE 0.015.** Likely mechanism: P0.1 (v1 trained with noise-injecting clobbered o_proj) + P0.2 (all-loop balancing). Caveat: FineWeb train/eval stream overlap may flatter absolute PPL — applies equally to v1, so the relative gain stands; don't quote 5.7 against external baselines. Evals archived in checkpoints_ablation_moe_s0/. Dense arm + seed-1 runs await user go. |
| 2026-06-06 | bnb fixed (cuda130 auto-detect); 2nd MoE expansion 48→96 (`distill_xl`, 632M); v5 run to step 2887; full docs/attribution/licensing pass; hardware analysis (A10 identified as best-fit upgrade) | v5 archived — **expert-count ceiling hit**: 2nd expansion net-comparable to v4 (7-prompt inspector), cv wouldn't tighten below ~0.5. Q#1 answered. Next lever: width/scale, not more experts. |

---

## Capability success criteria per milestone

What "succeeded" looks like at each checkpoint. Use these as the
yardstick when deciding whether to ship a reference checkpoint vs
keep iterating.

### v1 (distill_tiny — distillation only) — ACHIEVED ✓

- Held-out perplexity < 50 (achieved: **37.4**)
- ECE < 0.05 (achieved: **0.041**)
- Loop efficiency in 0.30–0.70 band (achieved: **0.50**)
- MoE cv < 0.8 (achieved: **0.34**)
- No ACT loop collapse during training

### v2 (distill_tiny — SFT'd) — ACHIEVED ✓

- All v1 criteria preserved (no degradation from SFT)
- Inspector test: at least 1 of 4 prompts produces `stop='eos'` (achieved: **1/4**, fibonacci)
- Inspector test: at least 1 of 4 prompts produces `stop='confidence'` (achieved: **1/4**, trivia)
- Different prompt types produce different response *registers* (achieved)
- `resp_frac` stays in 0.4–0.6 band during training (achieved)

### v3 (distill_small — MoE grown) — ACHIEVED ✓

- Function preservation at promotion: identical logits to v2 within fp tolerance (validated by tests)
- Loss does NOT spike at promotion or during sentinel decay (achieved: ce stayed ~1.4)
- All 48 experts get nonzero traffic by step 3000 (achieved: **min% 1.1, max% 3.0**)
- MoE cv after stabilization < 0.5 (achieved: **0.19** — better than v2)
- LTI ρ(A) stays < 1.0 (achieved: 0.34–0.39)
- Inspector test: math prompt produces `stop='eos'` (achieved)
- Inspector test: at least 1 prompt shows visibly more domain-relevant content than v2 (achieved on math + code)

### v4 (distill_small — OpenHermes-augmented SFT at seq_len=768) — ACHIEVED ✓

Ran at seq_len=768 (not the planned 1024) because bitsandbytes 8-bit Adam
couldn't load on this machine's CUDA 13.2 (no matching prebuilt binary).
Dropped the `--use-8bit-adam` flag and reduced seq_len to fit fp32 AdamW
in 12 GB. VRAM ran at ~10.7 GB the whole run — tight but stable.

Results vs the criteria set before the run:

- **OpenHermes acceptance ≥ 40%**: achieved (resp_frac 0.25–0.40 confirms general data flowing)
- **"Say hello" produces attempted social response**: achieved — v4 opens **"Sure,"** and halts on `confidence` (v3 ran 50 tokens of code-register gibberish)
- **Trivia re-triggers a non-`max_new_tokens` halt**: achieved — fires `stop='cycle'` (repetition detector), v3 ran out the clock
- **All v3 criteria preserved**: achieved — MoE cv 0.20, min% 1.4 (better than v3's 1.1), no loop collapse, ρ(A) 0.34–0.39

**Headline milestone — all three halt mechanisms now fire:**

| Prompt | v3 stop | v4 stop | Mechanism |
|--------|---------|---------|-----------|
| Math (2+2) | `eos` | `eos` | end-of-turn recognition |
| Code (fib) | `max_new_tokens` | `confidence` | UncertaintyHead guard |
| Trivia (France) | `max_new_tokens` | `cycle` | repetition detector |
| Hello | `max_new_tokens` | `confidence` | UncertaintyHead guard |

v4 halts cleanly on **4/4** prompts (v3: 1/4, v2: 2/4) and exercises all
three distinct halt mechanisms. The OpenHermes data variety re-tightened
the calibration that had drifted between v2 and v3, and generalized the
confidence-halt behaviour across every prompt type.

Unchanged ceiling: content is still gibberish at 420M params. Every
behavioural mechanism in the architecture is now demonstrably working;
the only barrier to usable output is parameter count (a compute problem,
not a design problem).

Archived as `archived_models/mythouro_distill_small_v4/`.

### v5 (distill_xl — 2nd MoE expansion, 48 → 96 experts) — DONE, ceiling hit ⚠️

Took **Path A** (stay single-card; promote v4's 48 experts to 96, ~632M params,
8-bit Adam). Ran to step 2887. Function preservation and post-promotion MoE
balance worked as designed — but the capacity **did not translate to inspector
improvement**:

- `cv` wouldn't tighten below ~0.5 (v3/v4 reached 0.19–0.20); min% stayed ~0.1–0.4
- 7-prompt inspector read came out **net-comparable to v4** (2 better, 2 worse on
  the standard 4) — correct register + clean halting, but scale-bound gibberish
- Conclusion: at 632M / ~20–40M tokens the model can't find distinct work for 96
  experts. **MoE expansion is tapped out** (Open research Q#1, answered 2026-06-06)

Archived as `archived_models/mythouro_distill_xl_grown_v5/`. **Do not do a third
expansion.** Next lever is width (Net2Wider) or scale, not more experts.

**Path B (NOT taken — recorded for reference): FSDP across both cards, direct 1B
distill from a stronger teacher**
- FSDP gradient sync overhead per step measurable but tolerable (~20–25 s/step at 1B)
- Teacher logits accessible (Llama 3.3 70B local quantized OR DeepSeek V3 API)
- Promoted student still benefits from MoE growth machinery (test: promote 1B → 1.5B)
- Calendar: 20K steps × 25s = 140 hours = ~17 overnights
- This is the "break past Ouro-2.6B quality ceiling" path. Now folded into the
  larger [Scale-up execution plan](#scale-up-execution-plan-the-destination)
  (from-scratch distilled 3B on rented compute), which supersedes it.

### Eventual full-pipeline target (~1B post-RL)

Far-horizon — only after a stable 1B base is shipped:

- DPO converges (KL-divergence to base bounded, reward score positive)
- Reasoning RL on GSM8K with verifier improves accuracy by >5 percentage points
- Multi-turn behaviour stable (no degenerative repetition over 5+ turns)
- ECE stays < 0.10 through post-training stages
- Safety + refusal calibration acceptable on red-team prompts (TBD on rubric)

---

## Failure modes encountered + recovery patterns

Institutional memory. If any of these come back in a future session,
the fix is already known — saves hours of re-debugging.

### Script defaults are NOT the proven recipe (2026-06-10)
- *Symptom*: both ablation arms (MoE AND dense) flatlined identically from
  scratch — hard CE stuck at ~7.6–8 after step ~20, eval PPL in the thousands
  (v1 was at 368 by step 1000), transient gradient-norm spikes (32→2743)
  concentrated on deep-loop batches, weights NOT diverged (max component
  growth 1.36×), data stream verified clean.
- *Root cause*: the runs used `training/distill.py`'s DEFAULTS (warmup 200,
  depth-reg 0.1, mb2/ga4) — but v1's "final successful run" used **warmup 500,
  depth-reg 0.3, mb1/ga8**, recorded only in the v1 MODEL_CARD provenance.
  Warmup 200 hits full 3e-4 LR at step 200 on a fresh 4-loop recurrent model →
  early instability → bad basin → flatline. The proven recipe lived in the
  model card, not the code.
- *Diagnosis chain that isolated it* (each step cheap): depth-sweep eval
  (killed "h_K eval artifact"), init-stats comparison (killed "dense FFN too
  hot"), checkpoint weight forensics (killed "runaway divergence"), 30-step
  resume probe (found the gnorm spikes + flatline), MoE control arm
  (reproduced the flatline → killed "dense arch"), data-stream decode (clean),
  v1 model-card command diff (the answer).
- *Fix*: `distill.py` defaults now ARE the proven recipe (warmup 500,
  depth-reg 0.3); ablation commands updated. **Rule: when reproducing a run,
  diff your command against the MODEL_CARD provenance command, never trust
  script defaults.**

### Code-review P0 fixes (2026-06-09)

External review (Fable 5), independently verified against the code and fixed —
all were **invariant violations the 303-test suite didn't catch** (it had no
zero-init / telemetry-coverage / emission-parity invariants; now added):

- **P0.1 — `_init_weights` clobbered deliberate zero-inits.** The blanket
  N(0,0.02) ran *after* submodules zero-init, so `CrossLoopAttention.o_proj`
  (identity residual) and `UncertaintyHead.net[-1]` (neutral 0.5) were random in
  all v1–v5 checkpoints. *Fix:* those layers mark `_skip_global_init=True`.
- **P0.2 — MoE router telemetry saw only the last loop.** The single recurrent
  MoEFFN is called n_loops×, but `_last_router_logits`/counts were overwritten
  each call → aux losses, the DeepSeek-V3 bias updater, and the cv/min%/max%
  metrics only constrained loop K−1. *Fix:* `_loop_body` returns the telemetry
  (checkpoint-safe, grad-tracked); `RecurrentBlock` accumulates across all loops.
- **P0.3 — eval emitted a never-trained path.** Training returns `h_K`; eval
  returned the ACT blend `h_out`, which under-summed for non-halting positions
  **and** was never seen by the coda/head in training. *Fix:* return `h_K` at
  inference too (ACT = early-exit criterion only). **Re-baseline (same v2
  weights, fixed code): PPL 46.3→39.25, ECE 0.058→0.042, loop_eff 0.500 unchanged
  — a free capability/calibration gain.** See
  [`reports/rebaseline_v2_after_p0_fixes.md`](../reports/rebaseline_v2_after_p0_fixes.md).
- **Also fixed:** `eval.harness` hardcoded `mythouro_1b` (couldn't load other
  checkpoints) and defaulted to a wrong-vocab tokenizer (gpt-neo, 50k) →
  out-of-range ids on the 49152-vocab models. Both corrected. *Open:* the eval
  metrics don't clamp inputs to `max_seq_len`.

**Caveats to existing conclusions:** the v3/v5 **cv numbers** (incl. open-Q#1's
"expert-count ceiling") were measured through P0.2 (last-loop-only), so treat
them as indicative, not exact. All archived eval numbers were measured through
P0.3's `h_out` path, so they were **pessimistic**. v1–v5 were *trained* under
P0.1+P0.2, so **a fresh run on the fixed code should improve further** — the
re-baseline only captures P0.3's emission gain on already-trained weights.

### Training / architecture

**ACT loop collapse — depth distribution pins to one bucket**
- *Symptom*: `depth` log metric drops to 0.000 sustained; inspector shows halt distribution `[1.0, 0, 0, 0]`. Loops are not used despite being configured.
- *Root cause*: ACT halt probabilities trained against the main task gradient. The optimizer learns to pin λ₀≈1.0 because the weighted-sum output `Σ w_t · h_t` lets it shortcut through one loop's output.
- *Fix*: **Return `h_K` (last loop's hidden state) during training instead of the ACT-weighted sum.** Inference still uses the weighted sum. See `RecurrentBlock.forward` in [mythouro/main.py](../mythouro/main.py); the `if self.training and kv_cache is None:` branch at the end.
- *Additional guard*: depth-reg coefficient on the PonderNet × Ouro KL-to-uniform regulariser. We use 0.1 as default in v3 onward.

**MoE expansion: new experts stay idle after sentinel decay**
- *Symptom*: post-promotion training shows `min% = 0.0` indefinitely; new experts never enter top-k.
- *Root cause*: sentinel decay completed but bias updater hasn't had time to nudge underused experts up.
- *Fix*: keep training. By step 2000+ post-promotion the DeepSeek-V3 updater pushes underused experts into the top-k pool. Verified in v3 training run.

**Loss spike at MoE promotion**
- *Symptom*: CE jumps by >0.5 immediately after grown checkpoint is loaded.
- *Root cause*: new experts' `down.weight` not zeroed, or sentinel bias not high enough to exclude from top-k.
- *Fix*: `_promote_state_dict` in [mythouro/grow.py](../mythouro/grow.py) zeroes the down projection AND sets `router_bias[new] = -100.0`. Tests assert this in `tests/test_grow.py`.

### Data pipeline

**HF streaming hangs silently for hours**
- *Symptom*: `load_dataset(..., streaming=True)` succeeds; first `next(iter(ds))` call hangs without timing out. No errors, no progress.
- *Root cause*: HF datasets' range-request iterator doesn't time out on stalled TCP connections. Happens on flaky / firewall-restricted home internet.
- *Fix*: use `streaming=False` (pre-download). Implemented in [mythouro/sft_data.py](../mythouro/sft_data.py) `_open_source`. Pre-download datasets once via `load_dataset(repo, split='train')` in a separate command.

**`apply_chat_template(..., tokenize=True)` returns `BatchEncoding`, not `list[int]`**
- *Symptom*: 100% of SFT samples rejected with `empty_or_shorter_response` reason. `len()` of returned object is 2 (number of BatchEncoding fields) rather than token count.
- *Root cause*: Ouro tokenizer returns a `BatchEncoding` wrapper; calling `len()` gives field count, not token count.
- *Fix*: render with `tokenize=False` to get text, then call `tokenizer.encode(text, add_special_tokens=False)`. See `_build_sft_example` in [mythouro/sft_data.py](../mythouro/sft_data.py).

**OpenHermes ~95% rejection rate at seq_len=512**
- *Symptom*: SFT diagnostic shows `general: 0/N (0.0% accept) [prompt_too_long=N]`. Training stalls waiting for valid samples.
- *Root cause*: OpenHermes-2.5 is mostly multi-turn — system + user + previous-assistant turns already exceed 512 tokens before the final assistant response can land in the loss-bearing region.
- *Fix*: either drop OpenHermes (v2 approach) OR bump to `seq_len=1024` (v4 approach). At 1024 acceptance jumps to ~60–70%.

**HF dataset `.shard(num_shards=N, index=i)` raises "list index out of range"**
- *Symptom*: streaming source crashes immediately on iterator open when `num_workers > num_shards`.
- *Root cause*: many instruction datasets ship as a single parquet file (`num_shards=1`); `.shard(num_shards=2, ...)` on those crashes inside the streaming library.
- *Fix*: only call `.shard()` when `total_shards > 1`. Use `num_workers=0` in DataLoader for single-process runs. See `MixedSFTDataset._open_source` in [mythouro/sft_data.py](../mythouro/sft_data.py).

**"Training is slow" — partly a misread (light runs), partly real (heavy runs)**
- *Symptom*: the per-step `tok/s` looks tiny (e.g. "11.4", "0.3"); seems to decline as training progresses; heavy runs take many hours.
- *The misread part (light SFT runs were fine)*:
  1. **The log prints `tok/s` with a `k` suffix** — "3.3k tok/s" is 3,300, not 3.3. The light SFT runs (278M, RTX 5070) ran a healthy **~2–3.3k tok/s**, verified against the on-disk logs.
  2. **The decline with steps is the `LoopCurriculum`** (`start_loops=2 → 4`): more loops = ~2× compute, so tok/s halving from 3.3k → ~1.7k across a run is *by design*.
  3. **Wall-clock conflates training with debugging** — restarts, the `0.0% accept / build_reject` BatchEncoding stalls above, and repeated 1.5M-example dataset reloads inflate the session time well beyond the clean training time.
- *The REAL part — the heavy MoE-grown runs (v3–v5) are genuinely slow.* Per-step
  time, **measured from checkpoint save-timestamps** (these runs have no cards/eval
  JSONs — reconstructed cards added 2026-06-08):

  | Run | Model | s/step | ~tok/s |
  |-----|-------|-------:|-------:|
  | v2 (light SFT) | 278M, 24 exp | ~4.4 | ~1,900 |
  | v3 | 420M, 48 exp | **~8** | ~1,500 |
  | v4 | 420M, fp32 Adam | **~13** | ~950 |
  | v5 | 632M, 96 exp | **~33** | **~370** |

  v5 ground **~8 h for ~900 steps**. The slowdown is **bigger model + 96-expert MoE
  dispatch + micro-batch 1 (forced by 12 GB)** — i.e. the heavy runs are
  **VRAM-gated**, dropping to the worst-case batch-1 utilisation regime.
- *The lever is hardware, on two independent axes*:
  - **More VRAM** → bigger batch → escape the micro-batch-1 strangulation that
    dominates the heavy runs. The 5070 at batch-2 sustains ~3k tok/s but
    `bench_step` at batch-8 hit **6,852** — ~½ the card is locked behind 12 GB.
  - **More dense-BF16 TFLOPS** → raises the compute ceiling (the 5070 is segmented
    to ~33.9). A faster card lifts the ~6,852 batch-8 ceiling itself.
  A card with *both* (4090 24 GB / 5090 32 GB) compounds them. Note this is a
  *new-card* purchase, not a free config win — the 5070 is already at its genuine
  limit for what fits 12 GB.

### Checkpoint / resume

**Grown checkpoint refuses to load — `KeyError: 'param_groups'`**
- *Symptom*: `load_checkpoint` crashes when resuming from a `tools/grow_checkpoint.py`-produced file.
- *Root cause*: grown checkpoints contain an empty `optimizer` dict (the source optimizer's tensor shapes don't match the promoted model). `optimizer.load_state_dict({})` blows up because `param_groups` is missing.
- *Fix*: `load_checkpoint` detects empty optimizer state and skips the load — caller's fresh optimizer is kept. See [mythouro/checkpointing.py](../mythouro/checkpointing.py).

**Tools script can't find `mythouro` module**
- *Symptom*: `python tools/grow_checkpoint.py ...` fails with `ModuleNotFoundError: No module named 'mythouro'`.
- *Root cause*: Python adds the script's directory to sys.path, not the cwd. From `tools/`, the project root isn't on the path.
- *Fix*: `sys.path.insert(0, project_root)` at top of `tools/grow_checkpoint.py`. Already implemented.

### Environment

**Pytest collection segfaults on Python 3.14 + Windows + pandas**
- *Symptom*: `python -m pytest tests/` segfaults during collection, before any test runs.
- *Root cause*: importing pandas (transitively pulled in by `datasets`) at top-level in a heavy training module triggers a crash on Py3.14+Windows.
- *Fix*: split checkpointing helpers into their own lightweight module so tests don't transitively import the training script's heavy deps. See [mythouro/checkpointing.py](../mythouro/checkpointing.py) module docstring.

**bitsandbytes "Configured CUDA binary not found at libbitsandbytes_cuda132.dll"**
- *Symptom*: `--use-8bit-adam` crashes at import — bnb looks for a `cuda132` binary that doesn't exist, even on the latest bnb (0.49.2).
- *Root cause*: torch is built for CUDA 13.2 (`cu132`), but bnb's prebuilt wheels only bundle binaries up to `cuda130`. There's no exact 13.2 binary in any release.
- *Fix*: **RESOLVED.** CUDA binaries are forward-compatible within a major version, so the `cuda130` binary runs fine against the 13.2 runtime. `training/sft.py` now calls `_configure_bnb_cuda_version()` before importing bnb — it scans the bundled `libbitsandbytes_cudaXXX` binaries, picks the highest ≤ the torch CUDA version, and sets `BNB_CUDA_VERSION` automatically. Verified: `AdamW8bit` constructs and runs on CUDA. No source build, no manual env var needed.
- *Manual override*: set `BNB_CUDA_VERSION` yourself before launching to force a specific binary; the helper respects an existing value.

**bitsandbytes not installed at all**
- *Symptom*: ImportError when `--use-8bit-adam` is passed and bnb isn't installed.
- *Fix*: import inside the `if args.use_8bit_adam:` branch with a clear `pip install bitsandbytes` error message. See [training/sft.py](../training/sft.py).

### Numerics / autocast

**`F.binary_cross_entropy` raises in bf16 autocast**
- *Symptom*: `RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast` during distillation.
- *Root cause*: BCE is on PyTorch's autocast-banned list because of fp16/bf16 stability issues.
- *Fix*: compute BCE manually as `-(target * log(p) + (1 - target) * log(1 - p))` and cast inputs to fp32. See `uncertainty_calibration_loss` in [mythouro/training_utils.py](../mythouro/training_utils.py).

**Cross-device tensor error during distillation**
- *Symptom*: `RuntimeError: Expected all tensors to be on the same device` when teacher is on cuda:0 and student is on cuda:2.
- *Root cause*: `teacher_logits()` was being passed input_ids on the student's device.
- *Fix*: `teacher_logits` internally moves input_ids to the teacher's device, and returns logits that the caller `.to(student_device)`. See [mythouro/training_utils.py](../mythouro/training_utils.py).

---

## Open research questions specific to MythOuro

These are MythOuro-specific questions where existing literature doesn't fully answer. Worth tracking as we learn more.

1. **Does MoE expansion compound across rounds?** **ANSWERED (2026-06-06), but SOFTENED (2026-06-09): No — it stops compounding by the second round at this scale, though the cv evidence is now suspect.** First round (24→48, v3) clearly compounded: recovered all 3 halt mechanisms, tightened cv to 0.19. Second round (48→96, v5) hit the **expert-count ceiling** — cv wouldn't tighten below ~0.5, min% stayed ~0.1–0.4, and a 7-prompt inspector read came out **net-comparable to v4**. **Caveat (P0.2):** every cv/min%/max% number here was measured through the last-loop-only telemetry bug, so the *cv* evidence isn't trustworthy — treat it as indicative. The **inspector read independently supports the ceiling**, so the conclusion likely holds, but it should be **re-confirmed after the P0.2 fix** (re-run a short expansion with all-loop routing balance). Conclusion (provisional): **don't do a third expansion; the next lever is width (Net2Wider) or scale, not more experts** — but verify the cv post-fix before treating it as settled.
2. **What's the right depth-reg coefficient post-promotion?** Currently kept at 0.1 across all runs. Empirically the halt distribution stays uniform; is there a coefficient that allows *more* adaptive depth at inference without re-triggering loop collapse?
3. **Does the ConfidenceAwareGenerator stop-threshold need per-checkpoint retuning?** v2 hit `stop='confidence'` on trivia; v3 didn't. Suggests the calibration shifted with the larger pool.
4. **Can we promote AND maintain a custom teacher in distillation?** I.e., distill the promoted v3 from Ouro-2.6B again, or do the new experts disrupt logit-matching?
5. **How does loop count interact with MoE specialization?** Different loops might want different expert mixes; we've never measured this.
6. **Would Ouro's per-step weighted loss beat our final-loop loss?** See the experiment below — Ouro trains the exit gates *via the task loss* (per-step LM loss weighted by exit probability); we use final-loop (`h_K`) loss + a decoupled depth regulariser. Ours works but theirs is arguably more principled.
7. **Should expert routing and recurrence-depth routing be one learned head (MoDr)?** Currently `MoEFFN.router` (which experts) and `ACTHalting` (how deep) are separate. Unifying them into one router that emits expert-choice + per-token depth could let depth and expert specialisation co-adapt — but it couples two collapse-prone parts. Gated behind the MoE-vs-dense ablation. See [MoDr — Mixture-of-Depth routing](#candidate-direction-modr--mixture-of-depth-routing-learned).

These are research-paper-sized questions individually; flagged here so we don't lose them.

---

## Candidate experiment: train at depth 6 (does the extrapolation headroom survive per-loop CE?)

**Question (user, 2026-06-11):** the forced-depth probe showed uncertainty still
falling at loops 4–7 on some prompts — should we train those depths?

**Why the probe doesn't settle it:** loops 4–7 were never emission loops, and
P0.5 proved the UncertaintyHead is unreliable exactly there (loop 0's readings
were off by ~0.2 for the same reason). The extrapolation signal was measured
with the proxy outside its calibrated range and was never checked against
per-loop CE. It may be real; it may be the loop-0 artifact's deep-end twin.

**Why the ablation runs were right to stay at 4:** the Ouro teacher computes
targets at exactly 4 recurrent passes (deeper student loops chase no deeper
signal during distillation); Ouro's own curve peaks at 3–4 loops; and +50%
recurrent compute mid-protocol would have been an uncontrolled variable.

**The experiment (queued, post-SFT):** one run at `max_loop_iters=6`
(curriculum ramp to 6), scored by **per-loop CE** via
`forward_trajectory(force_full_depth=True)` against the 4-loop arm — single
variable, ~6–7 h. Decision value: if depth 5–6 carries genuine trainable CE
gains, MoDr's depth policy has real headroom to allocate (and the depth-6
config earns a slot); if not, 4 is confirmed as the scale-appropriate depth
and the extrapolation findings get re-labelled as proxy artifacts.

## Candidate experiment: Ouro-style per-step weighted loop loss

A ready-to-run experiment, documented so it can be picked up in a future
overnight. Context and the full Ouro-vs-MythOuro comparison are in
[`docs/growth_design.md`](growth_design.md) ("Related design decision:
loop-loss supervision").

**Hypothesis.** Replacing the current *final-loop CE* with Ouro's *expected task
loss across loops* — `L = Σ_t pφ(t|x)·CE^(t)` (per-loop CE weighted by exit
probability) plus the existing entropy/depth regulariser — trains the ACT exit
gates directly and may improve loop_efficiency (currently ~0.5) and calibration
without re-triggering the loop collapse we fixed earlier.

**Why it might work where our first attempt failed.** Our original collapse came
from an ACT-*weighted-sum hidden state* `Σ wₜhₜ` fed to a single CE — the
optimiser collapsed `λ₀→1` to shortcut. Ouro keeps the losses *per step* (each
loop gets its own CE against the gold tokens) and only the *weighting* is the
exit probability, with entropy reg holding the distribution open. That's a
different gradient path: every loop is independently supervised to predict the
target, so no single loop can "absorb" the others.

**Implementation sketch** (~1 session, isolated behind a flag):

1. **Forward** — `RecurrentBlock.forward` already runs all loops during training
   (the `h_K` fix forces `K = n_loops`). Expose the *per-loop* hidden states
   `[h_1 … h_K]` (stash a list, like `last_halt_distribution` is stashed), not
   just `h_K`.
2. **Per-loop logits** — run the shared LM head on each `h_t` → `logits_t`.
   (Memory: K× the logit tensor; with `K≤4`, vocab 49152, seq 768, mb 1 — a few
   GB. May need `micro_batch=1` + 8-bit Adam, which we have.)
3. **Per-loop CE** — `CE^(t) = cross_entropy(logits_t, targets)` (SFT: masked to
   response tokens, reusing `masked_ce_loss`).
4. **Exit probabilities** — `pφ(t|x)` from the ACT halt head's per-loop λ values
   (already computed; currently only used by the depth-reg). Normalise to a
   proper distribution over `t ∈ [1..K]`.
5. **Expected loss** — `L_task = Σ_t pφ(t)·CE^(t)`. Keep the existing depth /
   entropy regulariser term as-is (Ouro uses both).
6. **Flag it** — `--loop-loss {final,per_step_weighted}` (default `final` to
   preserve current behaviour). New path only activates with the flag.
7. **Tests** — (a) per-loop logits shape; (b) `pφ` sums to 1 over loops;
   (c) with `K=1` the per-step loss reduces exactly to the final-loop loss;
   (d) a train-step smoke doesn't NaN.

**Validation plan.** Run a short SFT (~1500 steps) on the v4 checkpoint with
`--loop-loss per_step_weighted`, compare against the `final` baseline on:
loop_efficiency, ECE, halt-distribution spread, and inspector behaviour
(do more prompts hit `stop='eos'`/`stop='confidence'`?). Keep it if it improves
adaptivity/calibration without collapse; otherwise the flag stays off and the
finding is documented.

**Risk.** Low — it's flag-gated, default-off, falls back to current behaviour,
and the `K=1` equivalence test guards correctness. Worst case it doesn't help
and we keep `final`.

---

## Shipped: best-of-trajectory emission (inference)

An inference-side experiment for getting more out of the existing depth
machinery *without retraining* — runnable today against v4/v5.

**What it is.** Standard decoding emits the recurrent block's ACT-weighted blend
over loops. Best-of-trajectory instead scores *every* loop depth with the
UncertaintyHead and emits the logits from whichever loop the head is most
confident about — "keep the best step you saw" rather than "loop more, then undo
a bad one." It avoids the trap where extra loops legitimately *raise* entropy on
genuinely hard tokens before they resolve.

**Implementation** (all default-off, normal path byte-for-byte unchanged):
- `RecurrentBlock` gains an opt-in `collect_trajectory` flag that stashes the
  per-loop hidden states in `last_trajectory`.
- `MythOuro.forward_trajectory(input_ids, n_loops)` runs each captured loop
  state through Coda + LM head + UncertaintyHead and returns
  `(logits_traj (B,T,K,V), unc_traj (B,T,K))`.
- `inference.BestOfTrajectoryGenerator` / `best_of_trajectory_generate` — B=1
  greedy/sampled decode that selects the argmin-uncertainty depth per token,
  with a `min_loops` floor and a `chosen_loops` telemetry trace.
- 8 tests in `tests/test_inference.py::TestBestOfTrajectory`.

**How to validate.** Run it against v4/v5 in the inspector and compare to the
default generator: does `chosen_loops` actually diverge from "always deepest"?
Does inspector behaviour (halt reasons, register) improve? It's a measurement
tool — keep it if the trace shows the head is discriminating usefully across
depths; the gibberish ceiling at this scale may mask the effect until the model
is larger.

**Caveat (the code-level subtlety).** Training returns `h_K`, not the weighted
sum, to defuse ACT λ-collapse; inference uses the blend. Best-of-trajectory adds
a *third* emission rule that reads per-loop states — so it's an inference-only
overlay, deliberately not wired into training.

**First results (2026-06-08, v4 + v5, `reports/inspect_v{4,5}.txt`).**
- **ACT caps usable depth at ~3, not the configured 4.** At `n_loops=4`, ACT
  halts *all* positions by loop ~2, so only **3 loops actually run** and loop 3
  never executes — on every prompt, both checkpoints. (An early "100% diverged
  from the deepest loop" reading was an artifact of comparing against loop 3,
  which never runs; the per-loop dump caught it. Real divergence is **35–90%**.)
- **The uncertainty-by-depth curve is mostly monotonic, with genuine interior
  dips on some prompts.** v5 trends *deeper = more confident* (min at the
  deepest-run loop); v4 has prompts where *shallower = more confident* (min at
  loop 0) — the two checkpoints have differently-shaped depth/confidence
  profiles. A couple of prompts (v5 fib + Roman-Empire) show a real interior
  dip at loop 1, where best-of-trajectory does non-trivial selection.
- **Takeaway:** best-of-trajectory is *not* a no-op, but it's also not a big win
  at this scale — it's partly "take the most-confident endpoint." The louder
  signal is the **ACT depth-collapse to ~3**: the deepest configured loop is
  dead weight. That's a concrete data point for the MoDr / depth-policy work
  (the depth decision wants tuning) and for revisiting the ACT halt threshold.

**Forced-depth probe (`--force-full-depth`, `reports/inspect_v{4,5}_forced*.txt`).**
The ACT-respecting run above can only observe loops ACT chose to run, so it
couldn't tell "loop 3 genuinely hurts" (Hyp. A) from "loop 3 never ran"
(Hyp. B). The `--force-full-depth` knob suppresses ACT's convergence + halt-all
early-exit during trajectory capture (pure measurement — no weight change, normal
path untouched) so the loop runs the full `n_loops` and we can score the skipped
loops. An `[A/B]` line then compares ACT's learned halt depth (`halt_step_mean`)
to where uncertainty actually bottoms out. Findings:

- **The answer is prompt-dependent — both hypotheses are true, per input.** On a
  *subset* of prompts the skipped loops *do* lower uncertainty below ACT's
  stopping point (Hyp. B — ACT halts too early): v5 "recurrent-depth…" and "2+2"
  both bottom at **loop 3** (past ACT's ~2.0 cutoff); v4 "fibonacci" likewise. On
  others uncertainty rises past loop ~2 (Hyp. A — ACT justified). So a *single*
  global ACT threshold is structurally wrong: the right depth varies by token.
- **Depth-extrapolation partially works (v5, `n_loops=8`, 2× trained depth).**
  Curves are non-monotonic ("wavy"), but on "recurrent-depth transformer is"
  uncertainty reaches its **global minimum at loop 7** (0.50), well past the
  trained depth of 4 — concrete evidence the model *can* use more depth than it
  was trained on for some inputs. Other prompts degrade past loop 4 (off-
  distribution: loop-index embeddings + per-loop LoRA were only trained for
  loops 0–3). So extrapolation is real but input-specific, not free.
- **Confirmed noise-free (greedy, `--top-k 1`, 40 tokens,
  `reports/inspect_v{4,5}_forced_n*_greedy.txt`).** The temperature-0.7 curves
  carried run-to-run noise, so we reran deterministically. The headline result is
  *stronger* under greedy, not weaker: for "recurrent depth transformer is" (v5,
  `n=8`) uncertainty decreases **strictly monotonically** loop 0→7
  `[0.76, 0.64, 0.60, 0.54, 0.48, 0.41, 0.39, 0.39]` (every step negative,
  flattening to ≈0 by loop 7 — i.e. converging, not bottoming early). 3 of 4 v5
  prompts hit their minimum at loop 7. So depth-extrapolation to **2× the trained
  depth genuinely lowers uncertainty** for continuation-style prompts; it's a real
  effect, not a sampling artifact. (Short-answer / factual prompts still prefer
  shallow — the prompt-dependence also survives greedy.) The
  `--temperature` / `--top-k` inspector knobs were added for exactly this.
- **Direct implication for MoDr.** "Right depth is prompt-dependent, sometimes
  shallow, sometimes 3, occasionally 7" is precisely the case a single learned
  halt threshold can't serve and a **per-token learned depth router can** — this
  probe is the empirical motivation for the MoDr direction below.

---

## Gating experiment: MoE-vs-dense ablation

The single experiment that unblocks MoDr (and decides whether ~60% of the
architecture's complexity — router, `router_bias` updater, load-balance loss,
sparse-activation loss, expert-specialisation probe, growth machinery — earns its
keep). Spec'd here so it's ready to run when there's GPU time.

**Question.** At *matched compute* (same activated parameters / FLOPs per token),
does the MoE recurrent FFN beat a plain dense FFN on eval? If a dense model within
noise of MoE, sparsity isn't paying for itself at this scale and the routing
machinery is removable.

### SEED-0 RESULT (2026-06-11): MoE wins 4.0× — retained, pending seed 1

Both arms run (fixed code, proven recipe, seed 0, identical but the FFN):
dense final PPL **22.66** vs MoE **5.72** — and the gap GROWS with training
(1.0× @1k → 1.3× @2k → 3.0× @3k → 4.0× @4k), i.e. the sparse capacity is
progressively utilised. Per the pre-registered rule (>5–10% = keep MoE):
**MoE earns its complexity at this scale — decision pending seed-1
confirmation** (runs queued, user-gated). Consequences if seed 1 confirms:
MoDr proceeds as the full unified expert+depth router; the dense variant stays
as the ablation control. Nuance: dense (22.7) still beat v1's 37.4 — the code
fixes + recipe lifted both arms. Data: `docs/training_runs.md`.

### Matched-active (matched-compute) configs — the primary comparison

Per SwiGLU expert = `3·dim·expert_dim` params. For `distill_tiny`
(`dim=1280, expert_dim=1280, n_experts=24, n_shared=2, top-k=4`):

| | Active FFN / token | Total FFN params | Recurrent FFN |
|---|---:|---:|---|
| **MoE arm** (= `distill_tiny`) | ~59 M | ~157 M | `MoEFFN` (24 routed + 2 shared) |
| **Dense arm** (new) | ~59 M | ~59 M | one `Expert(1280, 15360)` |

Matched-active dense width: `d_ff = expert_dim · top_k · (1 + n_shared)
= 1280 · 4 · 3 = 15360`, so `3·dim·d_ff ≈ 59 M` = MoE's per-token active FFN.
Both arms do the **same FLOPs/token**; the MoE arm just carries ~98 M extra
*total* (sparsely-used) capacity. Whole-model totals: MoE ≈ 278 M, dense ≈ 180 M.

**Prerequisite code change — DONE (2026-06-08).** `MythOuroConfig` has
`recurrent_dense` + `recurrent_dense_ffn_dim`; `RecurrentBlock` builds a dense
`Expert(dim, d_ff)` recurrent FFN when set (auto width
`expert_dim·top_k·(1+n_shared)`); `mythouro_distill_tiny_dense()` is registered
in both training CLIs. Verified: MoE arm 278.9 M total / dense arm 180.5 M (98 M
idle capacity removed), dense recurrent FFN width 15360, and a test pins
`dense_FFN_params == MoE_active_FFN_params` exactly
(`tests/test_dense_ablation.py`). The MoE-only aux losses already no-op on a
dense model (they short-circuit to 0 when there are no MoE layers), so the dense
arm runs through the existing `distill.py` / `sft.py` unchanged.

**Run it (both arms, matched everything but the FFN):**
```powershell
# MoE arm (= the existing distill_tiny recipe)
python -m training.distill --trust-remote-code --teacher-device cuda:2 `
    --student-variant mythouro_distill_tiny `
    --warmup-steps 500 --depth-reg-coeff 0.3 --micro-batch 1 --grad-accum 8 `
    --start-loops 2 --random-depth --total-steps 4000 --seed 0 --eval --eval-every 1000 --ckpt-dir checkpoints_ablation_moe
# Dense arm (same flags, dense variant; MoE aux terms vanish to 0 automatically)
python -m training.distill --trust-remote-code --teacher-device cuda:2 `
    --student-variant mythouro_distill_tiny_dense `
    --warmup-steps 500 --depth-reg-coeff 0.3 --micro-batch 1 --grad-accum 8 `
    --start-loops 2 --random-depth --total-steps 4000 --seed 0 --eval --eval-every 1000 --ckpt-dir checkpoints_ablation_dense
```
Repeat each with `--seed 1` for the ≥2-seed requirement, then compare with the
eval harness / inspector.

**Teacher placement (measured 2026-06-09):** `--teacher-device cuda:0`
(cohabitation with the student on the 12 GB card) **OOMs at micro-batch 2** —
the v1 "fits in ~9 GB" sizing was theoretical; the real v1 run used two cards.
Put the teacher on a second GPU — **mind the index mapping, it is not
intuitive**: on this rig `cuda:0` = 5070, **`cuda:1` = 4060**, **`cuda:2` =
5060**. The documented role split wants the teacher on the **5060 → use
`cuda:2`** (the 4060 is reserved for parallel eval). Single-card fallback:
`--micro-batch 1`. Both 20-step GPU smokes (MoE + dense arms,
`reports/gpu_smoke_distill_*.txt`) ran clean two-card (teacher landed on the
4060 via cuda:1 — works, either 8 GB card fits the 5.2 GB teacher, but cuda:2
is the intended layout).

### Protocol

- **Identical everything except the recurrent FFN:** same `dim`, layers, loops,
  attn, vocab, tokenizer, **seed**, distillation corpus + teacher, total steps, LR
  schedule, warmup, seq_len, optimizer (8-bit AdamW), hardware.
- **Dense arm drops the MoE-only aux losses** (load-balance, sparse-activation,
  router-bias updater, expert-specialisation) — they don't apply. Keep CE +
  uncertainty-calibration + depth-reg identical across arms.
- **From-scratch distillation** at a matched step budget (~3–5 k steps, like v1) —
  cleaner than SFT-from-shared-base (no warm-start confound).
- Run **≥2 seeds per arm** — at 278 M / ~20–40 M tokens the variance is large
  enough that a one-seed gap can flip.

### Metrics (eval harness) + pre-registered decision rule

Primary: held-out **perplexity**. Secondary: ARC-Challenge, GSM8K,
loop_efficiency, ECE. Report alongside: tok/s, peak VRAM, total params, and
(MoE arm only) cv / min% / max%. Plus the 4-prompt inspector read.

Pre-registered rule (decide *before* seeing numbers, to avoid post-hoc
rationalising):
- **Dense within ~3% PPL of MoE** (and ARC/GSM8K within seed noise) → **drop MoE.**
  Remove routing machinery + growth path; MoDr degenerates to a depth-only router
  (simpler, still valuable). Biggest possible simplification.
- **MoE beats dense by a clear, cross-seed margin** (say >5–10% PPL) → **keep MoE.**
  Proceed to MoDr as the unified expert+depth router.
- **In between** → keep MoE provisionally, re-decide at scale-up.

### The caveat that matters most (read before acting on the result)

This ablation answers **"for the current 278 M / consumer-hardware regime,"** not
"forever." MoE's benefit typically *grows with scale and token budget*, and at
tiny scale experts are chronically under-trained — so a **dense win here does not
prove dense wins at 3 B.** Treat a dense result as "don't pay for MoE *yet*," and
**re-run the same ablation once on the scaled-up config** before locking the
architecture for the big run. A MoE win here is the stronger signal (sparsity
already paying off despite small scale).

**Effect on MoDr:** if MoE is dropped, MoDr isn't moot — it collapses to a
**depth-only `DepthRouter`** (the teacher/student depth policy below, minus the
expert-choice branch), which the forced-depth findings already justify on its own.
If MoE stays, MoDr is the full unified router.

---

## Candidate direction: MoDr — Mixture-of-Depth routing (learned)

The learned generalisation of everything above. Best-of-trajectory selects depth
*post-hoc* with a threshold/argmin on the uncertainty head; MoDr makes the depth
decision a **learned policy**, and unifies it with expert routing.

**The idea.** MythOuro already has two routing-shaped heads that are trained and
used separately:
- `MoEFFN.router` — picks which experts fire per token (DeepSeek-V3
  aux-loss-free bias balancing).
- `ACTHalting` — emits a per-position halt/continue probability per loop.

MoDr collapses these into **one router head that emits both** *expert choice*
**and** *per-token recurrence depth* (halt/continue logit per position per
iteration), supervised against a best-exit target rather than gated by a fixed
`act_threshold`. "Extend by 2 or 4 loops" stops being an inference heuristic and
becomes a trained policy — the model learns how deep each token needs to go,
jointly with which experts handle it.

**Why the depth probe motivates this (2026-06-08 findings).** The
`--force-full-depth` experiment established the two premises MoDr needs, both
under deterministic greedy decode (noise-free):
- *The right depth is per-token, not global.* Some prompts bottom out at loop 7
  (2× the trained depth), others at loop 0. A single `act_threshold` cannot serve
  both; a per-token policy can.
- *The useful depth exists, but ACT misses it.* ACT collapses to ~2–3 loops, yet
  uncertainty keeps dropping monotonically to loop 7 on continuation prompts — so
  the current halting policy leaves real signal on the table. That's a *policy*
  bug to fix, not a capacity limit.

**Teacher / student — the crucial design point (don't skip this).** "Run full
depth, then pick the best exit" — what best-of-trajectory does — is the right move
at *training* time and the **wrong** move at *inference* time. Running every loop
on every token and choosing afterwards destroys the entire purpose of adaptive
depth: you've already paid for the deepest loop before deciding you could have
stopped at loop 2. You can't pick an exit you haven't already run past — picking
is free, *reaching* the loop is the cost. So the pipeline splits:
- **Teacher (train time, can afford full depth):** run all `n_loops`, find each
  token's best exit — the loop minimising uncertainty (today's proxy) or, once the
  model is coherent, per-loop CE loss. This is exactly the `--force-full-depth`
  trajectory we already compute in `forward_trajectory`.
- **Student (inference, must be cheap):** the `DepthRouter` learns to *predict*
  that best exit per token from the current hidden state and halt there directly —
  loop 2 for an easy token, loop 7 for a hard one — without running the rest.
  Best-of-trajectory is the *supervision signal*, never the decode path.

So the one-line design rule: **let it run full depth at *training* time to
discover the best exit, and train a cheap per-token policy to hit that exit
directly at inference.** The forced-depth probe is the teacher; MoDr's
`DepthRouter` is the student.

**Why it's attractive.** The MoE router and the ACT halt head already share the
same routing primitive (a linear projection of the hidden state to a routing
logit). Unifying them is conceptually clean and could let depth and expert
specialisation co-adapt (e.g. some experts for shallow pattern-matching, others
for deep refinement — currently they can't coordinate).

**Why it's gated (honest caveat).** This is a *coupling* change to two of the
most stability-sensitive parts of the model (expert routing + ACT halting, both
of which have already caused collapse failures — see Failure modes). Adding that
coupling **before** the open **MoE-vs-dense ablation** ([Open research
questions](#open-research-questions-specific-to-mythouro), Q on whether MoE earns
its complexity) risks entrenching machinery that the ablation might say to
remove. **Sequencing rule: run the dense-vs-MoE ablation first.** If MoE stays,
MoDr is the natural next architecture step; if MoE goes, MoDr is moot.

**Rough build sketch** (when unblocked, flag-gated, default-off):
1. A `DepthRouter` head emitting `(expert_logits, halt_logit)` from the shared
   hidden state; `ACTHalting` becomes the `halt_logit` branch.
2. **Supervise the halt branch against the best-exit target from the forced-depth
   teacher** — per token, the loop that minimises uncertainty (proxy now) or
   per-loop CE (once coherent), computed via `forward_trajectory(force_full_depth
   =True)`. This is strictly better than supervising against the *convergence*
   depth (`‖h_{t+1}−h_t‖ < convergence_eps`): our probe shows convergence often
   lands *later* than the uncertainty-minimising exit, so convergence over-spends.
3. Keep the depth regulariser (KL-to-uniform) as the anti-collapse guard, and the
   `min_loops` floor so the student can't collapse onto loop 0.
4. Train the teacher target offline first (cache best-exit labels with
   `--force-full-depth`) so the student trains against fixed labels — cheaper and
   more stable than computing full-depth trajectories every step.
5. Tests: router emits both heads; `K=1` reduces to current behaviour; best-exit
   target matches `forward_trajectory` argmin; depth regulariser still fires;
   no-NaN train step.

**ANSWERED (2026-06-09, P0.5 audit): supervise MoDr with per-loop CE, NOT
uncertainty-argmin.** `tools/per_loop_calibration.py` measured per-loop ECE on
v2 and v4 (`reports/per_loop_calibration_p05.md`): the head is well-calibrated
at loops 1–3 (ECE 0.01–0.04) but **badly miscalibrated at loop 0** (ECE
0.17–0.22, error *understated* by ~0.2 — the loop curriculum starts at 2, so
loop 0 was never an emission loop and the head never saw it). An
uncertainty-argmin teacher would systematically over-select loop 0.
Consequences: per-loop CE is the mandated best-exit target;
`BestOfTrajectoryGenerator` now defaults `min_loops=2` (loop 0 excluded from
the argmin); the earlier "v4 prefers loop 0 on some prompts" reads were partly
a calibration artifact. To unlock all-loop uncertainty selection later: add a
per-loop calibration term in training (BCE against per-loop argmax error at
every loop), or start the curriculum at 1.

**Relation to prior art.** This is the project's own framing of Mixture-of-Depths
(Raposo et al.) adapted to a *recurrent* (weight-shared, looped) block rather
than a stack of distinct layers — depth here means loop count, not layer index.
