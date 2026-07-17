# Ideas backlog

Solo project (owner + Claude). This file exists so we can **collect ideas without
thrashing** — every promising idea gets logged here and waits its turn, instead of
interrupting the run in flight. Collecting ≠ implementing.

## The main thread (protect this)

The spine of the work, in order. Ideas *serve* this; they don't replace it.

1. **Fix the degeneration** — UPDATED 2026-06-16: it is **NOT hidden-state collapse**
   (collapse_metrics: reps healthy / high-rank / recurrence *diversifies* — see
   training_runs.md 06-16). It's **downstream**: exposure bias + forward-KL output
   degeneration (the MiniLLM mechanism). Cheapest on-target test: **`--divergence rev_kl`**
   (Tier-1, staged, one flag); then **on-policy/GKD**. *Demoted:* Huginn recipe / MeSH /
   recurrent_state_noise (target a collapse we don't have). Confirm regime with
   generation-time collapse_metrics.
2. **Token-curve test** — push 5–10× tokens, inspect every ~50M tokens.
   *Does capability keep growing with tokens?*
3. That curve is the **go/no-go for capital** (rent A100 / buy a rig) and the
   **proof artifact** for attracting a collaborator or funding.

## Triage rubric (apply before implementing ANY idea)

An idea moves from backlog → active only if it clears all three:

1. **Does it hit a current bottleneck?** (tokens, or generation collapse) — not a
   tangential nicety.
2. **Cheap to test locally?** — can we get signal on the current rig without capital.
3. **Reversible?** — can we back it out cleanly if it doesn't help.

Three yeses → do it. Otherwise → it stays in the backlog with a note on what would
change the answer. (Example: "grow the model" failed #1 — wrong lever while
token-starved. The noise knob passed all three.)

## Backlog

| Idea | Targets | Local test cost | Reversible | Status | Notes |
|------|---------|-----------------|-----------|--------|-------|
| **Sandwich norm + depth-aware init** (Huginn recipe; corroborated by Ouro) | **Scale-up stability** (NOT the current bug) | staged code, config-gated | Yes (default off) | **STAGED, RE-SCOPED 2026-06-17** — `--use-sandwich-norm --use-depth-aware-init`. NOT for the current exposure-bias spiral (reps healthy → demoted there). **Re-elevated for the SCALE-UP fresh distill.** | Ouro (teacher) + Huginn BOTH use sandwich norm for recurrent stability at scale; we're the pre-norm outlier. Adopt when scaling, not now. training_runs.md 06-17. |
| **recurrent_state_noise** (per-loop σ·RMS(h) anti-collapse regulariser) | Collapse | Low (knob, σ=0 = off) | Yes | Tested — marginal (1→2-word repeats). Our reverse-engineered guess; Huginn used norm/LR/init instead | The P0.1-noise replacement. May be redundant with the Huginn recipe. See training_runs.md 06-15/06-16. |
| **On-policy mode-seeking distillation (GKD + MiniLLM)** — student generates, teacher scores, reverse-KL/JSD divergence, teacher-mixed sampling | **Both** — exposure bias (collapse) + token efficiency | Medium–High (RLHF-like: student samples each on-policy step) | Yes (λ=0 + fwd-KL = current behaviour) | **Tier-1 IMPLEMENTED** (`--divergence {fwd_kl,rev_kl,jsd}` + `--jsd-beta`, verified 2026-06-16); Tier-2 (on-policy sampling + `--teacher-mix-alpha` + length-norm) still to stage | Highest-fit lever. Forward-KL (our current loss) may *cause* collapse in our small-student/big-teacher case → Tier-1 tests that cheaply on fixed data. MiniLLM's **teacher-mixed sampling (α≈0.2) dissolves the "needs a clean base" catch** for Tier-2. |
| **Multi-Token Prediction (MTP)** (DeepSeek-V3: aux multi-token heads in training, discarded at inference) | Token efficiency (extra training signal) | Medium (add heads + a distill run) | Yes (aux objective, removable) | Backlog | Fits our DeepSeek-V3 routing stack. Enhancement, *not* a collapse fix. From the NTP-alternatives survey (arXiv 2509.24435). |
| **Data quality (phi-style "textbooks")** (curate higher-quality distill corpus) | Coherence per token | Medium (corpus work) | Yes | Backlog | Small models get coherent on curated data at fewer tokens. Stay OpenAI-free (the clean-data constraint). |
| **Unlikelihood training** (Welleck et al. 2019) — loss term penalising high prob on repeated tokens | Collapse | Low (aux loss term) | Yes | Backlog — strong cheap lever | Surgical anti-repetition; could layer onto the current run. |
| **Teacher-generated synthetic data** — use Ouro to generate clean training tokens | **Token supply** (#1 bottleneck) | Medium (gen + curate) | Yes | Backlog — strong | Attacks token *supply*, not just efficiency; fully OpenAI-free. Pairs with phi-style quality. |
| **Sequence-level KD** (Kim & Rush 2016) — teacher generates sequences offline, student trains on them | Both (cheap exposure-bias help) | Low–Med (offline gen, no per-step sampling) | Yes | Backlog — Tier-2 stepping-stone | Cheaper precursor to full on-policy; no RL loop. |
| **Tokenizer graduation** (transplant vocab → re-match to a bigger teacher family) | **Capability ceiling** (unlocks bigger teachers) | High (transplant + heal) | Partially (new run, but gated) | **Future milestone** — gated on (a) Ouro saturated + (b) base structurally settled | The *bridge between two matched-KD regimes*: migrate the matched-vocab anchor from SmolLM2/Ouro (small teachers) to e.g. Qwen (big teachers). NOT cross-tokenizer KD (lossy, keeps vocab) — this re-matches so clean logit KD resumes. See deep-dive below + roadmap "Tokenizer graduation". |

## Parked (failed triage — recorded so we don't re-litigate)

| Idea | Why parked | What would un-park it |
|------|-----------|----------------------|
| Grow the model (more params via 8-bit Adam) | Fails #1 — token-starved (~170× undertrained); bigger model = hungrier, worse | Token-curve shows we've reached compute-optimal at current size |
| Pivot to diffusion / LCM / non-transformer (from the NTP survey) | Fails #2/#3 — frontier big-lab work; throws away working stack + teacher | A funded team + compute; not a solo move |
| Pause/think tokens (Goyal et al., arXiv 2310.02226) — extra *width* scratchpad before output | **Already covered** — we do adaptive "think before you speak" via recurrent *depth* + ACT (stronger: adaptive vs fixed-K). Combining width+depth is tangential complexity | A task that needs working-*space* (more hidden vectors), not just iteration-*time* |
| T-FREE tokenizer-free LLM (Deiseroth et al., arXiv 2406.19223) — hash/trigram sparse embeddings | **Breaks distillation** — no shared vocab → can't KL against Ouro's ~49k-token output space → forces train-from-scratch (= abandons the teacher, our token-efficiency lever). Tempting embedding savings (~⅓ of a 278M model) + 3× micro-batch don't outweigh losing the teacher | A *from-scratch* foundation-model project at real scale (not MythOuro) |
| Coconut / Chain-of-Continuous-Thought (Hao et al., arXiv 2412.06769) — latent reasoning ACROSS positions (feed hidden state back as next input) | **Right family, wrong phase.** Orthogonal axis to ours (across-position vs our across-depth latent reasoning) — *combining them is an exciting far-future direction* (cf. Huginn's "warm-start recurrent state from prev token"). But Coconut's Stage 0 needs a base that already does coherent *language* CoT; we're ~1000× tokens away from coherent text. Doesn't touch collapse/tokens; needs CoT data + n+1 sequential passes. | A coherent, CoT-capable base exists → then test depth×position latent reasoning |
| **Parallel looped paths** (owner's original idea) — N independent recurrent paths on the same input, per-token confidence arbitration via the uncertainty head | **Reframed (06-16): a quality / parallel-compute feature, NOT a collapse fix or a speedup.** Real systems insight: recurrent depth is *sequential* (can't parallelize across depth) → independent paths fill otherwise-idle parallel hardware at ~constant latency, arbitrated by the existing best-of-trajectory/uncertainty machinery. Quality-for-compute trade. **Crux: paths must be *deliberately diverse* (injection schedule / routing temp / loop budget / seed)** or they collapse to correlated copies and buy nothing. Plausibly-original combination for recurrent-depth. NOT for training throughput (2× compute = slower training). | A **coherent base** exists → test-time-compute / calibration lever. **Design note written: docs/parallel_loops.md** (incl. the saturate-stranded-decode-compute rationale + honest prior-art). |

**Architectural validation note:** three independent lines — Ouro (recurrent-depth
teacher), the NTP survey's "latent reasoning / internal loop" category, and pause
tokens — all converge on *adaptive compute per token before output*. MythOuro is a
sophisticated instance (adaptive depth via ACT). Good evidence the architecture bet
is sound; useful for the "is it worth it" case.

---

## Reference shelf (resources for later phases, not active ideas)

| Resource | Relevant when | Notes |
|----------|---------------|-------|
| [Awesome-Collection-Token-Reduction](https://github.com/ZLKong/Awesome-Collection-Token-Reduction) | **Deployment / throughput phase** (after we have a coherent model) | "Token *reduction*" = fewer tokens for *compute* efficiency (pruning/merging, KV-cache & prompt compression), mostly inference & vision/VLM. NOT "token *efficiency*" (capability-per-training-token = our bottleneck, see GKD). The KV-compression slice is largely already covered by our **MLA**. |
| **Huginn (Geiping 2025, arXiv 2502.05171; code seal-rg/recurrent-pretraining; model tomg-group-umd/huginn-0125)** | **MINED → see training_runs.md 06-16** | Closest cousin to MythOuro. Documents our exact collapse + the cure (LR↓, sandwich norm, depth-aware init). Token reality-check: 3.5B/800B tokens. Promoted to a backlog item (Huginn stability recipe). |
| **linear_cross_entropy_loss** (Geiping, github.com/JonasGeiping/linear_cross_entropy_loss) | Throughput/VRAM phase | Fused head+CE, avoids materialising logits → bigger batch. **Caveat:** distill KL needs full logits → useful for hard-CE/SFT terms, not the distill soft-loss. |
| **flash-attention** (Dao-AILab) | **Blocked now** | FA2 unsupported on cuda_cc (12,0) = Blackwell 5070; we're on SDPA fallback. Watch for FA3/Blackwell support. |
| **ml-engineering** (Stas Bekman) | Scale-up phase | Practical training-at-scale reference (throughput, multi-GPU, debugging). |
| **bpeasy** (gautierdag) — fast BPE training | **Future Rust port only** | Can't swap tokenizers while distilling from Ouro (vocab must match — same constraint as T-FREE). Post-distill / from-scratch tool. **See "Tokenizer graduation" deep-dive — the controlled time to swap.** |
| **Zett / FOCUS / WECHSEL** (zero-shot/heuristic tokenizer transfer) | **Tokenizer-graduation milestone** | Embedding+head re-init from sub-token pieces so the new vocab starts warm, not random; heal with short continued training. The mechanism for the graduation deep-dive below. |
| DistiLLM-2 (Ko et al. 2024–25) — skew-KL + adaptive scheduling | When building Tier-2 on-policy | A refinement *of* the on-policy mode-seeking lever (divergence/scheduling). |
| Neural text degeneration (Holtzman 2019) + contrastive decoding | Deployment / decode-time | Decode-side anti-repetition; we already have confidence/cycle stops. |
| Deep Equilibrium Models stability (Bai et al. 2019) | Background | Theory for why our contractive recurrence (ρ(A)<1) converges to a fixed point — informs the noise/ρ levers. |
| **Connected Papers** (connectedpapers.com) — citation-neighborhood discovery | Scouting | Seed from Huginn / GKD / MiniLLM / MoR to surface the recurrent-depth & distillation-efficiency clusters. |
| **Looped-model scouting cluster** (to mine for transferable fragments) | Ongoing | **Mixture-of-Recursions (2025)** — recursion + routing for adaptive depth (closest to our ACT+MoE; top pick); **Relaxed Recursive Transformers** (Bae, 2410.20672) — recursion + LoRA-per-loop (we do this) + **Continuous Depth-wise Batching** (2–3× decode throughput; ties to parallel_loops saturation); **CoTFormer** — adaptive per-token depth; **Universal Transformers** (Dehghani 2018) + **PonderNet** (Banino 2021) — ACT/halt foundations; **Saunshi 2025** — looped depth-extrapolation theory; **ETD** (2510.07358) — loop reasoning-critical subset added at *mid-training*, +28% GSM8K @1B; **Loop-NN** (2409.14199) — basic looped refinement (superseded by the above). |
| **MeSH — Memory-as-State-Highways for Recursive Transformers** (2510.07739, ICLR 2026) | **Collapse-relevant — deep-dive candidate** | Names our two failure modes: *uniform computation across iterations* (→ contractive collapse) and *single overloaded hidden state*. Fix: **external memory buffer + per-iteration routing for specialization**. Structural complement to the Huginn recipe (norm/LR/init). Matches bigger non-recursive @1.4B with 33% fewer params; code available. We have partial versions (loop-index embed + LoRA-per-loop). |
| Throughput levers for distillation (the "speed up training" thread) | Now-ish | Linear-CE (head+CE fuse; helps hard-CE not distill-KL); MTP (backlog); seq-level KD (backlog); **sequence packing** — distill data appears already packed (P1.9 "packed distillation data" note) so likely DONE; num_workers dataloader A/B (queued). |
| **SubQ-1.1-Small / Subquadratic Sparse Attention (SSA)** — Subquadratic AI (subq.ai), 2026 | **Far-future long-context phase** (post-coherence; the medical / whole-artifact RAG endgame) | Content-dependent **sparse attention with linear compute/memory** → 64.5× fewer attention FLOPs @1M ctx; trained @1M tokens, retrieval generalises to **12M** (RULER 99.12%). **Orthogonal to our recurrent core** — would swap attention (MLA→SSA), not the loop. Fit: reason over *whole* rare-disease literature / patient history in-context instead of chunking. **Tension to note:** SubQ's pitch is "stop chunk-and-retrieve, hold the artifact in context" — complement to, not replacement for, an external knowledge store (see RAG/continual-learning in roadmap; can't fit all PubMed in 12M tokens, and external = updatable/citable/safe, which medical needs). **⚠️ Competitive-use policy on subq.ai → REFERENCE ONLY** — read the report for ideas / benchmarks / recipe insights, but do NOT adopt their model/code/weights for a distributable product (same family as the OpenAI non-compete constraint we already avoid). **Implement the *technique* (subquadratic/sparse long-context attention) via clean-licensed alternatives, swapping MLA→sparse:** DeepSeek **NSA** (Native Sparse Attention, arXiv 2502.11789 — closest analog, fits our DeepSeek-aligned lineage; check for official code) · **MoBA** (Moonshot, github.com/MoonshotAI/MoBA — trainable block-sparse, *official open repo* = code in hand) · **Mamba/Mamba-2** (Apache 2.0 — true linear-time, but a bigger swap: SSM replacing attention). Verify each license at adoption. **Implementability (assessed 2026-06-21): a modular *swap*, NOT an architecture rewrite** — attention is already abstracted (`cfg.attn_type` = mla/gqa), so NSA/MoBA slot in as a third `attn_type` class with the recurrent spine / MoE / ACT / LTI untouched. *Real work* = integrating the sparse mechanism's KV/block-state with our recurrent **per-loop cache** (`recurrent_loop_*`) + sink + RoPE (contained to the attention layer). Mamba is more invasive (replaces QKV with an SSM sublayer). **Scale-gated:** ~zero benefit at our seq_len 1024 — only pays at long context, so it's a far-future feature that also implies enabling a long-context *regime* (data / seq_len / memory), not just the attention class. |
| **MemPalace** (github.com/mempalace/mempalace, **MIT**) — local-first AI memory | **RAG / external-memory tier** (roadmap Stage A retrieval; post-coherence) | The off-the-shelf "memory palace": **verbatim** storage + semantic retrieval + temporal **entity-relationship knowledge graph**, structured wings/rooms/drawers; pluggable vector backends (Chroma/SQLite/Qdrant/pgvector); Python, mature. **Pairs with SSA** as complementary layers (external store → retrieve a slice → SSA reasons over the whole slice in-context — not competitors). Strong **medical** fit: local = patient privacy, KG = disease/symptom/treatment relations, verbatim+citable = never-hallucinate-from-weights. Candidate to plug in *instead of* building `RetrievalAugmentedInjector` from scratch. Check at adoption: embedding-model dependency + model-agnostic integration. |
| DistiLLM-2 (contrastive / skew-KL) | When building Tier-2 on-policy | Already noted above — refinement of the on-policy mode-seeking lever. |
| **VibeThinker-3B** (WeiboAI 2026 — arXiv 2606.16140; github WeiboAI/VibeThinker; hf WeiboAI/VibeThinker-3B) | **Validation now; levers for the future on-policy/RL stage** | Dense **3B** (Qwen2.5-Coder-3B base) hitting *frontier reasoning* (AIME26 94.3, IMO-AnswerBench 76.4%, LiveCodeBench 80.2) at **~$7.8k** post-training. **Core validation:** its "**Parametric Compression-Coverage Hypothesis**" = our exact bet — *verifiable reasoning compresses into a small param core; knowledge needs coverage* → independent proof a small model can be a frontier **reasoner**, knowledge left to retrieval (our RDT + MemPalace split; medical reasoning = compressible core, drug-facts/guidelines = retrieval). **Borrow (future):** (a) "**Spectrum-to-Signal**" diversity-then-amplify + **MaxEnt-Guided Policy Optimization (MGPO)** = entropy-preserving anti-mode-collapse — the *principle* speaks to our collapse fight, but it's **RL**, so it slots into the future on-policy/GKD stage, not current distillation; (b) **offline self-distillation** (sample → filter good trajectories → distill back) = a concrete instance of our parked on-policy lever. **Does NOT transfer wholesale:** dense **not RDT** (architecture not liftable — we reach small-reasoner via recurrence); **RL-heavy** (new capability, gated past coherence); **crucially the RL needs *verifiable* rewards** (math/code auto-checkable) — **medical largely isn't auto-verifiable**, so the core recipe fits only the *quantitative/rule-based* medical sub-tasks (dosing math, interaction lookups, guideline adherence), **not open diagnosis**. **Net: strategic validation + future-stage pointers, not a change-now recipe.** (Read from abstract/README/card; full MGPO/CLR mechanics need the paper.) |
| **Nemotron-3-Ultra** (NVIDIA 2026 — research.nvidia.com/labs/nemotron/Nemotron-3-Ultra) | **Datasets to vet now; MOPD + Mamba-hybrid for later phases** | Frontier **550B / 55B-active** MoE **Hybrid Mamba-Attention** (+ MTP speculative-decode; 1M ctx, strong RULER@1M; 5.9× throughput vs GLM-5.1). Model itself **unusable** (hyperscaler scale; NVFP4 quant is Blackwell-native → *not* our Intel/NNCF INT4 path). **Useful pieces:** (a) **open datasets** — 173B GitHub code + synthetic factual/QA + Nemotron-Posttraining-v3 SFT/RL → candidate distill/SFT inputs; **vet via dataset_selection.md**, but *likely clean* (NVIDIA Open Model License permits output reuse — favorable vs the OpenAI-tainted MIRIAD we dropped); **none medical**; verify exact license at adoption. (b) **MOPD = Multi-teacher On-Policy Distillation** — validates our parked on-policy/exposure-bias cure, AND "multi-teacher" = the concrete mechanism to blend **Ouro (RDT signal) + a bigger dense teacher (capacity)** [ties to "distill from larger models once coherent"; vocab-match/converter per teacher still applies]. (c) **Mamba+attention+MoE hybrid** validates the SubQ/NSA/MoBA/Mamba long-context thread — refinement: **do a *hybrid* (Mamba + *some* attention), not pure-Mamba**. **Core caveat:** Nemotron is a standard *non-recurrent* stack → **silent on our recurrent-depth core**; the MoE/SSM/on-policy bits are orthogonal swap-ins, not recurrence guidance. (Read from the research page; layer ratio / token count not disclosed there.) |
| **Self-Harness** (arXiv 2606.09498, 2026) — autonomous agent-harness self-improvement | **Deployment / product phase** (far-future; post-coherence + an agentic product to optimize) | LLM auto-improves its own *harness* (prompts / tools / agent-loop / scaffolding), **not the weights**: mine weaknesses from execution traces → propose minimal harness edits tied to failures → regression-validate, keep only safe ones (Terminal-Bench-2.0: MiniMax 40→62%, Qwen 24→38%, GLM 43→57%). **Fit:** if MythOuro ships **agentic** (retrieval/MemPalace → reason → cite → flag uncertainty), this auto-tunes that scaffolding to *our* model-specific failures — apt because (a) harness design is model-specific and a custom small medical model needs custom scaffolding, (b) could optimize **how the harness acts on the uncertainty head** (when to retrieve more / decline / flag) = the operational side of the honest-specialist thesis. **Caveats:** not model-related (wrapper only, no architecture/training bearing); applies only if the deployment is *agentic/tool-using* (little value for plain Q&A); it's *LLM-driven* → keep the optimizer local/open for sovereignty; check for released code vs re-implement. |

---

## Deep dive: On-policy distillation / GKD (assessed 2026-06-15)

Source: Agarwal et al., *On-Policy Distillation of Language Models* (GKD), ICLR
2024, arXiv 2306.13649. Surfaced via the NTP-alternatives survey (2509.24435).

### What it is
Standard distillation (what we do) trains the student on a **fixed corpus** with
KL to the teacher — but the student is then evaluated on **its own** generations,
which it never trained on. That train/inference mismatch is **exposure bias**, and
it is *exactly* our collapse signature (great teacher-forced loss, degenerate free
generation). GKD closes the gap: with probability **λ**, the student **generates a
sequence itself**, and the teacher **scores those self-generated tokens**; with
prob 1−λ, use the fixed corpus. Loss:

> `L = (1−λ)·E_(x,y)~data[ D(p_T ‖ p_S)(y|x) ] + λ·E_x, y~p_S[ D(p_T ‖ p_S)(y|x) ]`

Key knobs:
- **λ** (on-policy fraction): 0 = standard KD (us today), 1 = fully on-policy.
- **Divergence D**: forward KL (mode-covering, our current Hinton KL), reverse KL
  (mode-**seeking** — chases the teacher's high-prob tokens, *avoids low-quality
  generations*), or **JSD(β)** interpolating between them (paper uses β=0.1/0.5/0.9).
- Do **not** backprop through the sampling — treat the student's samples as fixed
  context, only differentiate the divergence. Stable and simple.

### Why it's the highest-fit idea we have
- **Attacks our collapse at the root, not the symptom.** The `recurrent_state_noise`
  knob mechanically perturbs the fixed point; GKD removes the *cause* (the model
  trains on the distribution it actually produces). Complementary, not competing.
- **Reverse-KL / JSD are mode-seeking** → less mass on garbage tokens → less
  repetition/hallucination. Right medicine for a collapse-prone model.
- **Token-efficient — our #1 bottleneck.** Paper: *on-policy GKD on 5% of the data
  beats supervised KD on the full dataset.* Relative gains 1.7–2.1×; T5-base GSM8K
  **10.2% → 20.5%**. That is capability-per-token, exactly what we're starved for.
- **Fits our stack.** We already run teacher+student with a KL term in
  `training/distill.py`. GKD = (a) add a divergence option (reverse-KL / JSD(β)),
  (b) sometimes sample a continuation from the student, (c) score it with the
  teacher, (d) λ to mix. A generalization of what exists, not a rewrite.

### The catch (sequencing) and the cost
- **Needs a base that already generates *adequately*.** GKD assumes student samples
  are usable; the paper starts from SFT'd students, never random/collapsed. **Our
  current base samples pure garbage** (`is is is`), so fully-on-policy on it would
  just train on degenerate strings. → **Do the noise fix FIRST** (get a
  non-collapsing base), *then* GKD. Or bootstrap with λ≈0.25 (mostly fixed data +
  a little on-policy) and ramp λ up as the student improves.
- **Sampling overhead ~1.8–2.2×** in the paper (vanilla T5). Ours is **higher** —
  the student is recurrent-depth, so each sampled token runs the loop + ACT + KV
  cache, costlier than a plain forward. Real cost on a token-starved local rig, but
  the token-efficiency win (5% ≈ full) should more than pay for it.
- **Best divergence is task-dependent** (paper's words) → try JSD(0.5)/(0.9) and
  reverse-KL; tunable.

### Triage verdict
Passes all three (hits *both* bottlenecks; local-testable; reversible) — but with a
**hard dependency: noise fix → non-collapsing base → GKD.** This is the natural
**next big lever after the noise test**, and plausibly the thing that moves us from
"varied register-salad" toward coherence by training on-policy under teacher
correction. Implementation sketch: add `--divergence {fwd_kl,rev_kl,jsd}`,
`--jsd-beta`, `--onpolicy-lambda`, `--onpolicy-temp` to `distill.py`; reuse the
existing teacher-forward + KL path; gate sampling behind λ.

---

## Deep dive: MiniLLM / reverse-KL — UNIFIES with GKD (assessed 2026-06-16)

Source: Gu et al., *Knowledge Distillation of Large Language Models* (MiniLLM),
arXiv 2306.08543.

**Correction to earlier note:** MiniLLM is NOT a cheap drop-in loss swap. It is
**on-policy + policy-gradient** (student samples each step, PPO-style clipping +
variance fixes) — *same cost class as GKD*. The value is the **insight**, and it
**merges with the GKD plan** rather than being a separate, cheaper option.

### The core insight (directly implicates our setup)
**Mode-seeking (reverse KL) > mode-covering (forward KL)** for distillation,
*especially* small-student / big-teacher. Forward KL makes the student *cover* all
teacher modes; when it can't represent them, it dumps mass into "void regions" →
degenerate text. **We are a 278M student distilling a 2.6B teacher, currently on
forward KL** → forward KL may be *actively contributing to our collapse*, not just
failing to stop it. New suspect, and a reason to switch divergence.

### Three takeaways (folding into the GKD lever)
1. **Reverse-KL / JSD divergence** — the mode-seeking anti-collapse + token-efficiency
   win. Evidence: exposure-bias error *plateaus* (their Fig 6, vs accumulating for
   fwd-KL/SeqKD), ECE improves toward teacher, and **diversity preserved** (99%
   distinct-4-grams — refutes "mode-seeking kills variety").
2. **Teacher-mixed sampling (α=0.2)**: sample from `α·teacher + (1−α)·student`.
   **This dissolves GKD's "needs a non-collapsed base" blocker** — the teacher term
   pulls samples toward sense even on a collapse-prone student, so on-policy can
   start **on a base like our current one**. Also prevents reward-hacking (student
   gaming the teacher via repeated phrases). High value.
3. **Length normalization** — removes short-sequence bias.

### ✅ Empirical update (2026-06-24) — the rev-KL "ECE regression" was a DEPTH-MISMATCH ARTIFACT (RETRACTED)
A 06-23 read on `step_0003216` (~53M tok, *pre*-transition) measured ECE 0.20 and was logged as "offline
rev-KL regresses ECE." **The depth-matched verdict refutes it:** `step_0006675` (n_loops=4 trained +
inferenced, ~109M tok) → **ECE 0.0152**, right in the well-calibrated band. The 0.20 was the
n_loops=4-on-n_loops=2/3 mismatch, **not** a rev-KL cost. So **rev-KL does NOT hurt calibration**;
takeaway #1's "ECE improves" is *not* contradicted. **BUT generation still mode-collapses at 109M** —
calibration good, free-gen collapsed: *exposure bias is decoupled from calibration.* The on-policy point
below stands as the **generation** cure, not a calibration one.

**Calibration as its own axis — backlog levers (cheap, no on-policy needed):**
- **Bump `--unc-coeff`** — the `uncertainty_calibration_loss` is *already in the loss*; weight it up
  to pressure ECE directly, independent of the divergence. Cheapest test.
- **Post-hoc temperature scaling** at inference — the standard, near-free ECE fix (fit one scalar T on
  a held-out set; doesn't touch training).
- **On-policy** (above) — the paper's *actual* ECE lever; helps exposure-bias *and* calibration, but
  RLHF-cost. The deferred deep fix.

Cross-ref: roadmap "Current status" + differentiator-#1 tension; training_runs.md 06-23 eval.

### Results / scale / cost
- 1–6% ROUGE-L over SeqKD across **GPT-2 120M–760M, OPT 1.3–6.7B, LLaMA 7B** (our
  size range covered); larger gains OOD and on longer responses (≥6 tokens).
- **Cost: RLHF-like** (sample every step + teacher scoring) — same as GKD, not cheap.
- Needs white-box teacher (we have it — Ouro).
- Limitation: gains minimal on very short outputs (small output space → fwd≈rev KL).

### Net
**MiniLLM + GKD = one lever: "on-policy mode-seeking distillation"** = GKD framework
+ reverse-KL/JSD divergence + teacher-mixed sampling (α≈0.2) + length-norm. The
teacher-mixed sampling is the unlock that lets us try it *before* a perfectly clean
base. Updated impl sketch for `distill.py`: `--divergence {fwd_kl,rev_kl,jsd}`,
`--jsd-beta`, `--onpolicy-lambda`, `--teacher-mix-alpha 0.2`, `--length-norm`,
`--onpolicy-temp`; reuse teacher-forward path; gate sampling behind λ.

**⚠️ Pipeline gotcha — teacher-generation stop tokens (2026-06-24).** The moment on-policy *generates*
from the teacher (demonstrations, or scoring student rollouts), the teacher generation MUST use the stop
set **`[<|endoftext|>=0, <|im_end|>=2]`**. Ouro's tokenizer `eos` is `<|endoftext|>` (id 0), but its chat
template ends turns with `<|im_end|>` (id 2) — a *different* token. Stopping only on `eos` → the teacher
never halts between turns and **self-conversations into `user/assistant/user/assistant` cascades**
(observed in the Ouro chat launcher, 2026-06-24; fix = collect both ids and pass as `eos_token_id` to
`generate`). Harvesting that = cascade-garbage teacher data ("unnecessary tokens that confuse the student,
but worse"). **Offline distillation is UNAFFECTED** (forward-only on fixed data, never calls `.generate()`
on the teacher). The same stop-set applies to **MythOuro's own serving** later (shared SmolLM2 49152 vocab
→ id 2 = `<|im_end|>` for the student too). **Bake `[0, 2]` into any teacher-gen / student-serve path.**

---

## Deep dive: MeSH (Memory-as-State-Highways) — collapse-relevant (assessed 2026-06-16)

Source: Yu et al., "MeSH: Memory-as-State-Highways for Recursive Transformers,"
arXiv 2510.07739, ICLR 2026. Code available. Pythia 160M–1.4B, 250B tokens.

### Mechanism
Replaces the single overloaded recurrent state with **external memory + per-iteration
routers**:
- Buffer **M** = **B slots** (B = N_loops+3), each (L×D). Init: embeddings in slot 0,
  rest zero.
- Per-iteration **write** router: `m_b ← m_b + f_core(h) ⊙ softmax(Lin_write^t(h))_b`.
- Per-iteration **read** router: `h^{t+1} = Σ_b m_b ⊙ softmax(Lin_read^t(h))_b`.
- Persistent info → buffer; hidden state → transient workspace. Per-iteration routing
  ⇒ loops *specialize* instead of one universal transformation.

### Why it fights OUR collapse (with receipts)
- **Fig 5:** base recursive = "faster spectral decay into lower-dim subspaces" (= our
  token-correlation→1 / rank collapse); MeSH "maintains high-dimensional structure".
- **Fig 3:** naive recursion = "first loop dominates" (= Huginn Bad Run 2, loops
  ignored); MeSH = "all loops contribute". Fixes BOTH recurrent failure modes.
- Mechanism: externalized memory relieves the single state of persistence-vs-plasticity;
  per-iteration routing breaks the uniform contractive map → no fixed point.

### Convergence with Huginn (validation)
MeSH **independently uses depth-aware init** (output-proj std × 1/√(2·N_compute)) —
the SAME Takase/Huginn init we just staged (`--use-depth-aware-init`). Two independent
recursive-transformer papers both require it ⇒ strong confidence the staged fix is
right. The recipes are **complementary**: Huginn = optimization/normalization (sandwich
norm + low LR + init); MeSH = representational (decouple memory + specialize loops);
shared root = depth-aware init.

### Map to MythOuro
- Already approximate per-iteration variation: loop-index embed + LoRA-per-loop.
- Missing: the B-slot memory buffer + per-iteration read/write routers (modest params
  K×(D×B+B)). Would replace/augment the LTI-injection update.
- Moderate architecture change → needs a FRESH distill, config-gateable, reversible.

### Triage + sequencing
Targets collapse directly; modest overhead; reversible if gated → passes rubric.
**Sequence AFTER the Huginn recipe** (cheaper, staged). MeSH = the structural lever if
norm/LR/init alone don't de-collapse; can combine.

### CHEAP fragment usable NOW (no training)
MeSH's **diagnostics**: singular-value spectrum of recurrent states across loops +
per-loop contribution. Compute on EXISTING collapsed checkpoints (read-only inference)
to *quantify* the collapse (rank decay loop-by-loop; later-loop contribution) — the
analog of Huginn's "token correlation → 1". Gives a tracked metric for whether the
Huginn recipe / MeSH actually fix it. BUILT: tools/collapse_metrics.py (verified 2026-06-16).

---

## Deep dive: Tokenizer graduation (future milestone, assessed 2026-06-18)

**The question that surfaced it:** can we distill from *regular* (non-recurrent)
transformers, given how few recurrent/looped teachers exist? **Answer: yes** — logit
KD only touches the teacher's output distribution over the vocab, never its internals,
so a feedforward teacher is fine; the student learns the *function* and implements it
with its own recurrence. Architecture mismatch only breaks *feature/hidden-state* KD,
which we don't do. **The real gate is the tokenizer, not the architecture.**

### The constraint (why we can't just grab a bigger teacher today)
Our vocab is the **SmolLM2 tokenizer** (GPT-2-style BPE, **49152**; see
`mythouro/tokenizer.py`, default `ByteDance/Ouro-2.6B-Thinking`). Logit KD computes KL
token-for-token over a *shared* support, so the teacher must use this exact vocab.
- **Drop-in same-vocab teachers = the SmolLM2 family** (SmolLM2-1.7B-Instruct, etc.).
  But that's *lateral* scale to our 2.6B Ouro teacher — diversity, not a ceiling lift.
- **Qwen / Llama / Mistral / Gemma have different vocabs** → not drop-in. A bigger,
  smarter teacher therefore requires changing our vocab.

### The graduation move (the bridge between two matched-KD regimes)
Not "abandon matched KD" — **migrate the matched-vocab anchor** from the SmolLM2/Ouro
family (small teachers) to a bigger family (e.g. Qwen):
1. Squeeze Ouro dry on the current matched tokenizer (extract all it can teach).
2. **Transplant** the tokenizer (Zett / FOCUS / WECHSEL): swap embedding + LM head for
   the new vocab, init new tokens from the *old* embeddings of their sub-token pieces
   (warm start, not random), heal with short continued training. The transformer body
   and **the recurrent-depth reasoning machinery are vocab-agnostic and survive** —
   we're re-skinning I/O, not relearning to think. That's what makes it cheap vs.
   from-scratch.
3. Resume **clean matched-vocab logit KD**, now from Qwen2.5-7B/14B/32B.

Contrast with **cross-tokenizer KD** (ULD / MinED): keeps our vocab, aligns across
tokenizers — lossy *every* step. Graduation re-matches once, then KD is exact again.

### Bonus + cost
- **Bonus:** a bigger vocab (Qwen ~151k) compresses text better → fewer tokens/doc,
  more effective context, cheaper generation. Dual payoff beyond teacher access.
- **Cost:** bigger vocab = larger embedding/head + heavier softmax; on a *small* model
  the head is a large param fraction → the *target* vocab is itself a decision (49k↔151k),
  not "go biggest". Also watch **capacity gap** (small student / huge teacher distills
  *worse*; reverse-KL helps) → moderate teacher (7B–14B), not a giant.

### Timing / triage
Gated on: (a) Ouro saturated (student ≈ teacher, gains flatlined) AND (b) base
structurally settled. Do it earlier and you heal embeddings you'll disrupt again.
Relationship to the scale-up plan: the roadmap's "from-scratch distilled 3B with a
bigger teacher" path would just *start* on the new vocab; **graduation is the alternative
that preserves the trained base** instead of restarting — pick based on whether the
current base is worth carrying forward when compute arrives. Provenance rule still
applies (Apache-2.0 teachers; screen instruct-tunes for OpenAI-derived synthetic data).


<!-- ===== moved from docs/roadmap.md (2026-06-27 doc reorg) ===== -->

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



<!-- ===== moved from docs/roadmap.md (2026-06-27 doc reorg) ===== -->

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
   `--force**Relation to prior art.** This is the project's own framing of Mixture-of-Depths
(Raposo et al.) adapted to a *recurrent* (weight-shared, looped) block rather
than a deep spatial network.

---


## Decision (parked): preference-tuning flavor when we reach that stage = ORPO, not KTO

**Logged 2026-07-17** (asked during the fixed-rollout recovery run). Preference methods are
**≥2 stages away** — they tune *which plausible output a model prefers*, and α=0.0 doesn't
produce plausible outputs yet. Sequence stays: (1) token grind to fluency→meaning,
(2) **v6 clean-data SFT** (chat/QA stops being OOD; instruction-following appears),
(3) preference round — and there, **ORPO**:

- **No reference model.** KTO/DPO keep a frozen reference policy resident; ORPO folds the
  preference term into the SFT loss (odds-ratio penalty) — one model, one stage. On a
  one-card rig that just barely fits teacher+student, that's a hard constraint, not taste.
- **Our preference data will be synthetic PAIRS** — teacher generation = chosen, student
  generation = rejected, same prompt; the rollout infrastructure already emits both halves.
  KTO's headline advantage (unpaired thumbs-up/down) targets data we don't have and won't
  have without deployed users.
- **One-stage SFT+preference** rides along with the v6 SFT pass instead of adding a phase.

**Flip to KTO iff** the model is ever deployed and collecting real, unpaired, imbalanced
human feedback. Triage rubric: passes (2) cheap and (3) reversible; fails (1) *today* — hence
parked here, not active.
