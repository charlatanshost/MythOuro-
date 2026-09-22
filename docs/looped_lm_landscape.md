# Looped-LM landscape — adopt / fork / preserve
**Drawn 2026-07-04 · §0 added 2026-08-10.** §1-§5 are the original architecture/serving map; §0 covers the objective question that batch opened, which §1-§5 never asked.

**Purpose.** A solo-planner's map of the (fast-growing) looped / recurrent-depth LLM literature:
what each neighbor does, which techniques are worth folding into MythOuro's **future**
architecture, the one strategic fork the space is forcing, and what stays uniquely ours. This is a
**forward architecture-planning artifact**, not a training to-do — the current bottleneck is
coherence (tokens + the landed fixes) and nothing here moves that. But MythOuro is planned by one
person, so the future architecture has to be planned *now*; this externalizes it so it isn't all
in-head.

**Vetting caveat.** Entries are **abstract-level reads** — several are 2026 papers past the
Jan-2026 model cutoff, surfaced via AI summaries that mis-stated at least one (see Hyperloop below:
an overview called it "parallel paths"; the paper is within-trajectory residual enrichment).
Confirm each mechanism against the full paper before any writeup or adoption. The *conclusions* are
trusted because two independent passes vetted them (my read + owner's firsthand corrections, e.g.
Xe Link); the *paper internals* still want a full-text check.

## 0. UPDATE 2026-08-10 — what changed since this map was drawn

The 2026-07-04 map below is an **architecture / serving** map: it asks who loops
what, and how to spend stranded compute. Everything filed since asks a different
and, for us, more urgent question: **what do you TRAIN the loops against?**

Seven works arrived on 2026-08-10 (owner-supplied), plus two earlier in the day.
They do not change the §3 fork or the §4 moat. They open one question the
original map never asked, and it is now the deciding question for
`--loop-loss-weighting` (rung 3, docs/roadmap.md).

### 0.1 ⚔ THE LOOP-SUPERVISION CONTRADICTION

Sources disagree, directly and with evidence on both sides, about whether the LM
loss should be applied at every recurrent step or only the last.

| source | position | evidence |
|---|---|---|
| [duongtrongnguyen123/recurrent-depth-ttc](https://github.com/duongtrongnguyen123/recurrent-depth-ttc) (MIT, 2026) | **every loop** | iterative-target supervision extrapolates **up to 24x beyond trained depth**; **final-only supervision causes accuracy WALLS**. Same prelude→core→coda layout as ours. Small-scale, seed-pinned, synthetic chain tasks. |
| **Ouro** — *Scaling Latent Reasoning via Looped Language Models*, Zhu et al., [arXiv:2510.25741](https://arxiv.org/abs/2510.25741) | **every loop** | `L = Σ_t p_φ(t\|x)·L^(t)`, per-step loss weighted by exit probability. 1.4B/2.6B production models. **This is our teacher.** |
| **RLTT** — *Prioritize the Process, Not Just the Outcome*, [arXiv:2602.10520](https://arxiv.org/abs/2602.10520) | **every loop** | distributing credit across the latent trajectory beats terminal-only credit by **+5.8% (1.4B) / +10.9% (2.6B)** — measured on Ouro. |
| **Silent Thinking** — *Thinking Deeper, Not Longer*, Hung-Hsuan Chen (NCU), [arXiv:2603.21676](https://arxiv.org/abs/2603.21676) | **final only** | loss at the final recurrence step *"eliminating intermediate shortcuts"*; rejecting intermediate supervision is called **critical for genuine multi-step reasoning**. 20+ recurrence steps, compositional tasks, 1.6–1.75x OOD. |
| **MythOuro today** | **final only** (`h_K`) | not chosen on theory — chosen because the ACT-weighted-sum output `Σ wₜhₜ` gave the optimiser a direct lever to pin λ₀≈1 and collapse depth. See docs/growth_design.md and the roadmap failure modes. |

**Why this is load-bearing rather than academic.** Our 2026-07-31 `--n-loops`
sweep was FLAT at 4/6/8 (L4 0.0%, median rel_err 0.400 in all eight arms) and we
concluded depth is not a lever at current capability. `recurrent-depth-ttc`
offers a competing explanation of that same shape: **final-only supervision
produces an accuracy wall exactly at the trained depth.** If that transfers,
depth is not dead, it is walled, the wall is the objective, and rung 5 (grow
depth) treats the symptom while rung 3 treats the cause.

**The counter-case is ours already.** We are not speculating about
intermediate-supervision shortcuts — we hit one. That is the whole reason `h_K`
is returned during training. Ouro resolves the same tension by pairing per-step
weighting **with** entropy regularisation toward a uniform halt prior; we adopted
that regulariser (`depth_regularization_loss`) and not the weighting, so the pair
is only half-installed.

⇒ **Rung 3 is a genuine two-sided experiment.** Watch the halt distribution and
`loop_efficiency` next to the evals: depth collapsing toward loop 0 vindicates
Silent Thinking; the flat-depth wall lifting vindicates per-loop supervision.

### 0.2 New entries to the landscape table

| Work | Axis | Mechanism | Same as ours? |
|---|---|---|---|
| **Retrofitted Recurrence** (McLeish, Li, Kirchenbauer et al., [arXiv:2511.07384](https://arxiv.org/abs/2511.07384)) | training recipe | surgery of a pretrained fixed-depth model into **prelude / recurrent / coda** + a recurrence schedule | **⭐ CLOSEST MATCH FILED** — that is our exact layout and our `LoopCurriculum` |
| **RLTT** ([arXiv:2602.10520](https://arxiv.org/abs/2602.10520)) | objective | RL credit across the latent trajectory, weights from the exit PDF | same family as rung 3; RL half gated on pass rate |
| **Silent Thinking** ([arXiv:2603.21676](https://arxiv.org/abs/2603.21676)) | objective | final-step-only loss + LayerScale 1e-4 + identity-biased gate (bias −2.0, ~88% retention) | **opposes** rung 3 |
| **Beyond Memorization** (Rodkin et al., [arXiv:2508.16745](https://arxiv.org/abs/2508.16745)) | measurement | disjoint train/test rule sets on 1-D cellular automata | not ours — but the best available ceiling on ACT |
| **T2MLR** (Cai, Zhu, Dong, He, Arora, [arXiv:2607.15178](https://arxiv.org/abs/2607.15178)) | efficiency | recurrence over a middle **slice** of layers, across POSITIONS | **No** — across-position, ours is across-depth |
| **The Recurrent Transformer** (Oncescu, Morwani, Jelassi, Meterez, Kwun, Kakade, [arXiv:2604.21215](https://arxiv.org/abs/2604.21215)) | efficiency | layers attend to their own KV; tiling Θ(N²)→Θ(N log N) | **No** — layerwise memory, pretraining-scale change |
| **Loop as a Bridge** (Chen, Liu, Shao, [arXiv:2601.10242](https://arxiv.org/abs/2601.10242)) | interpretability | concept injection across loops, **measured on Ouro 1.4B/2.6B** | measurement of our teacher |
| [PrathibhaDevkar/rdt_transformer](https://github.com/PrathibhaDevkar/rdt_transformer) (MIT) | educational | from-scratch RDT, `h(t+1)=A·h(t)+B·e+Transformer(h(t),e)` | same LTI-injection shape; **7 commits, 0 stars — not a result, do not cite** |

### 0.3 Three numbers worth carrying

1. **ACT buys ≈ +1 reasoning step.** [arXiv:2508.16745](https://arxiv.org/abs/2508.16745): a 4-layer baseline holds ~95% at k=1 and falls below 25% at k≥2; ARMT recurrence reaches k=2; **ACT adds about one step**; GRPO reaches k=3 with no intermediate supervision; CoT exceeds 99% through k=4. Their summary — *"depth, not width, drives multi-step accuracy"*, with embedding width giving minimal gains. **Read as a ceiling on our halting work:** three efforts have gone into that axis, and the best case for ACT specifically is modest. Training METHOD moved depth further than architecture did, which agrees with everything measured here on 2026-08-10.
2. **Test recurrence 32.** [arXiv:2511.07384](https://arxiv.org/abs/2511.07384): TinyLlama-1.1B retrofitted to recurrence reaches **51.2% GSM8K vs 46.2%** for the post-trained non-recurrent baseline, evaluated at **32** recurrent steps — against our trained 4 and our shelved 4→8 growth. Also: pretrained init saves ~950B tokens versus random init.
3. **Intermediate loops are not linguistically accessible even when trained.** [arXiv:2601.10242](https://arxiv.org/abs/2601.10242), measured on Ouro — which *does* supervise every loop — finds models *"largely insensitive to injections during the intermediate loops… detection and identification only occur in the final loop"*. Tempers expectation 1 above: per-loop supervision may fix per-loop CALIBRATION without making intermediate exits good OUTPUTS.

### 0.4 What is newly adoptable

- **Muon instead of AdamW** ([arXiv:2511.07384](https://arxiv.org/abs/2511.07384) switched specifically for recurrent stability). Cheapest untested lever in this batch on a rig where stability has been the recurring failure. No architecture change, no fresh distil.
- **A data curriculum** — general "healing" data first, task data after ([arXiv:2511.07384](https://arxiv.org/abs/2511.07384)). We ramp loop depth but never data composition.
- **Identity-biased gated recurrence** (bias −2.0, ~88% state retention) and **LayerScale 1e-4** ([arXiv:2603.21676](https://arxiv.org/abs/2603.21676)). Same instinct as our `ρ(A)<1` tracking and depth-aware init; adopt-on-next-architecture-rev, like hyper-connections in §2.
- **Hardcoded halt rule** ([recurrent-depth-ttc](https://github.com/duongtrongnguyen123/recurrent-depth-ttc)): per-example inference depth configured **without a learned stopping head**. After three failed efforts on ours, this is the cheap fallback if the head stays unfixable.

**Unchanged by this batch:** the §3 EXPLOIT fork and the §4 moat. Nothing here does
per-token cross-path uncertainty arbitration, the medical two-lane, or
MoE-recurrent.

---

## 1. The landscape — who works which axis

Everyone is working the **same looped / weight-shared primitive**, but on **different axes**. None
does MythOuro's specific combination.

| Work | Axis | Mechanism (abstract-level) | Same as ours? |
|---|---|---|---|
| **Ouro** | (teacher) | recurrent-depth base we distil from | n/a — our teacher |
| **PLT / LoopCoder-v2** (2606.18023) | efficiency | parallelize/cheapen depth (CLP + shared-KV attn), "only loop once" | **No** — *opposite*: eliminates stranding |
| **Hyperloop** (2604.21254, Yoon Kim) | param-efficiency | matrix-valued residual streams via hyper-connections after each loop | **No** — within-trajectory residual enrichment, one output |
| **MELT** (per overview) | decode memory | share loop-1 KV across loops, selective update | **No** — memory trick; we already have KV machinery |
| **RRT / Continuous Depth-wise Batching** (2410.20672) | throughput | batch tokens at different loop depths through the shared block, ~2–3× | **No** — throughput; shares our *saturation premise* |
| self-consistency / best-of-N / deep ensembles | quality | sample N, aggregate | **Underpins** our Mode-B |
| MDASH (2026) | quality | structured multi-model disagreement | **Informs** our arbitration |
| DSpark (2026-06-27) | speed | speculative decoding (draft/verify) | **Our self-speculative child** |
| **MythOuro parallel-loops** | quality | **N *diverse* trajectories + per-token uncertainty arbitration** | — (ours) |

## 2. ADOPT — fold into the future model / serving

- **Hyper-connections (from Hyperloop) — the strongest fit for our core.** Enriches cross-loop
  residual flow (matrix-valued residual streams) for **minimal parameters** — exactly where our
  recurrent depth accumulates its computation, and squarely on our parameter-/token-efficiency
  thesis. Additive, orthogonal to the §3 fork. **Retrain-required → design into the next
  architecture rev.** Top adopt candidate. (Confirm the hyper-connections formulation, Xie et al.
  2026, against the full paper first.)
- **Continuous depth-wise batching (RRT) — throughput, no retrain.** Batch tokens/rollouts at
  different loop depths through the shared block to saturate the latency-bound card. The "go WIDE"
  principle made concrete for the **on-policy rollout generation at the Max phase**. Serving-time;
  adopt whenever, no architectural commitment. See
  [decode_kernel_optimization.md](decode_kernel_optimization.md).
- **MELT-style KV-sharing — memory, reference only.** Already handled (`CrossLoopKVCache`,
  `compress_kv_cache`); MELT's loop-1-share is an alternative to consult *if* we revisit decode
  memory. Not a gap.

## 3. FORK — the one strategic decision the space is forcing

The literature is bifurcating into two philosophies of the sequential-depth cost, and **our
designs sit on one side**:

- **Exploit the stranded compute** *(ours)* — keep depth sequential, monetize the idle compute:
  parallel diverse paths (quality — [parallel_loops.md](parallel_loops.md)), self-speculative
  depth (speed — [decode_kernel_optimization.md](decode_kernel_optimization.md) §6), depth-wise
  batching (throughput).
- **Eliminate the stranded compute** — PLT-style parallelized depth: make one trajectory fast,
  nothing left to monetize.

**These are in tension.** Adopt PLT-style parallel depth and the "diverse paths are ~free" premise
evaporates (the card is no longer idle). So this is a fork to pick *deliberately*, not drift into.

**Decision (2026-07-04): commit to EXPLOIT.** Rationale: the moat (§4) exists *only* on the
sequential-depth-with-exploitation side; going "eliminate" makes MythOuro a faster commodity
looped-LM and sheds every differentiator. Hyper-connections (§2) is adoptable regardless — it's
orthogonal to the fork. Revisit only if decode latency becomes a hard product blocker that
exploitation genuinely can't meet.

## 4. PRESERVE — the moat (in none of the above)

None of the surveyed work does these; they are what make MythOuro *itself*:

- **Per-token cross-path uncertainty arbitration** via a trained uncertainty head (parallel-loops).
- **The medical two-lane** design (calibrated uncertainty → the mission).
- **MoE-recurrent** combination (fine-grained routed experts inside the shared loop).
- The **stranded-compute framing** as an *original systems argument* for this architecture.

Keep these central. The space filling with looped-LM papers **validates the direction without
eroding the novelty** — treat neighbors as a **menu of base-primitive upgrades** (§2), a **fork to
hold** (§3), and a **moat to protect** (§4).

## 5. Integration into parallel-loops — which cost each kills

Parallel-loops ([parallel_loops.md](parallel_loops.md)) has two known costs (its §3): **N× compute**
and **N× memory**. Several neighbors are *precisely* the fixes — so they're the **efficiency
substrate** for the idea, not competitors. Most-solid first.

- **Depth-wise batching (RRT) → kills N× COMPUTE.** Paths are already batch rows; with ACT-budget
  diversity (shallow vs deep paths) they sit at *different loop depths*, which is exactly what RRT
  batches through the shared block. The throughput substrate for the depth-diverse variant —
  low-risk, breaks no diversity assumption.
- **KV-sharing (MELT + cross-path) → kills N× MEMORY.** MELT shares KV across loops *within* a
  path; and the N paths share the input prefix, so they can share KV *across* paths too —
  strongest in **Mode A** (paths aligned on the agreed token), weaker in Mode B (paths diverge).
  Attacks the "never free in memory" tax directly.
- **Self-speculative (DSpark) → fuses quality + speed via AGREEMENT.** Use cross-path agreement as
  the speculative-accept signal: agree → accept fast, skip deeper compute; disagree → spend compute
  + arbitrate. The §5-of-parallel_loops MDASH "agreement-as-signal" turned into a *speed* mechanism
  — arbitration-quality *and* speculative-speed from one confidence signal. **Heuristic (lossy)
  accept, not lossless rejection-sampling** — same family as the current `inference.py` early-exit.
- **Hyper-connections → orthogonal enrichment + a bolder bet.** Cleanly: enrich each path's block
  (better per-path representation, minimal params; stacks with all the above). Bolder: use the
  matrix-valued residual **lanes AS cheap micro-paths** and arbitrate across lanes instead of N
  full trajectories → sidesteps the N× cost entirely. **Rides on the parallel_loops §4 diversity
  crux** (do lanes stay decorrelated, or collapse to copies?) — big-if, big-payoff.
- **PLT → does NOT integrate.** The opposing fork (§3): parallelized depth eliminates the stranding
  the paths exploit, so folding it in is self-defeating (no idle compute → paths aren't free). Its
  KV-share *sub*-technique could be borrowed independently (overlaps MELT); its core cannot.

**Net:** RRT + MELT de-risk the two *known* weaknesses (compute, memory); self-speculative +
hyper-connections offer the quality benefit *more cheaply*. Parallel-loops stays the novel
differentiator; the neighborhood is the substrate that makes N-path ensembling **practical instead
of N×-expensive.** The two bolder syntheses (lanes-as-micro-paths, agreement-as-accept) ride on the
diversity-decorrelation question (parallel_loops §4) — validate that first. All stage-gated behind
coherence.


---

## 2026-09-22 — link filed: EmergentMind "Looped Language Model (LoopLM)" topic page

`https://www.emergentmind.com/topics/looped-language-model-looplm` — an
aggregator, not a paper. Read for what it adds to §1. Five of its eight
citations were not previously filed:

| paper | filed? | verdict |
|---|---|---|
| Saunshi et al. **2502.17416** *Reasoning with Latent Thoughts* | yes (references.md) | **the one that matters — see below** |
| Ng et al. **2409.14199** *Loop Neural Networks for Parameter Sharing* | yes | GPT2-81M looped ≈ deep baseline. Same claim as Saunshi, smaller scale. |
| Giannou et al. **2301.13196** *Looped Transformers as Programmable Computers* | **new** | theory: loops emulate iterative algorithms given "sufficient loops". Depth-as-computation, no training recipe. |
| Yang et al. **2311.12424** *Looped Transformers are Better at Learning Learning Algorithms* | **new** | parameter efficiency vs standard transformers. In-context learning, not LM. |
| Chen et al. **2410.11268** *Bypassing the Exponential Dependency* | **new** | L loops ≈ L² in-context examples. Theory, linear attention. |
| Gao et al. **2402.13572** *AlgoFormer* | **new** | **pre-transformer / looped core / post-transformer** — our exact prelude-recurrent-coda split, independently arrived at. Joins Retrofitted Recurrence (2511.07384) as prior art for the layout. |
| Wu et al. **2510.24824** *Parallel Loop Transformer* | **new** | cross-loop parallelism; efficiency, not capability. Same bucket as PLT/LoopCoder in §1. |

### The finding that speaks to the current question

Saunshi et al. (already filed, but only as "looped depth-extrapolation
theory") reportedly separates the two axes:

> **reasoning accuracy scales with computational depth (loop count), not
> parameter count — while MEMORISATION remains parameter-dependent.**
> "A shallow model looped multiple times rivals a deep unlooped model."

And the aggregator's summary of Ouro (2510.25741) lands on the same split:
*"reasoning benefits more than memorisation."*

**This is our result, from the other direction.** Measured here:

* **reasoning-ish (code L4): flat at ~4-5%** across every size and objective —
  and we have never increased loop count. 4 loops throughout, `grow_depth.py`
  shelved, the 2026-07-31 depth sweep tested UNTRAINED extrapolation (4/6/8)
  and correctly concluded nothing about trained deeper loops.
* **memorisation (medical fact recall): flat at 3-4/30** — and the roadmap's
  own capacity estimate is ~90-170M tokens of facts at 2-4 bits/param. That is
  parameter-bound, and we went 180M → 240M activated, which is nothing.

If Saunshi is right, we have spent the entire project moving the axis that
governs memorisation (params, tokens, width) and never once moved the axis
that governs reasoning (loops). Ouro trains at 4 and so do we — but Ouro is
**48 layers** inside its loop and we are **2+2 prelude/coda over a 1-layer
shared block**. Same loop count, an order of magnitude different depth *per*
loop.

⚠ **Read from an aggregator's summary, not the papers.** Before this becomes a
decision, read 2502.17416 directly: whether the depth/param separation holds at
LM scale or only on the synthetic reasoning tasks it was measured on decides
whether it applies to us at all. Filed as a hypothesis with a name, not a
finding.

### What it would change

It re-opens rung 5 (`grow_depth.py`, 4→8 loops) on a *different* argument than
before. It was shelved for "needs a base worth spending the nights on", and the
2026-09-18 ceiling measurement plus this split is a better reason to revisit it
than that gate was to close it. It is also the one architectural axis this
project has never tested, in a model whose entire premise is recurrent depth.


---

## 2026-09-22 — filed: a Gemini summary of the LoopLM landscape. Two new items; most of it was already here with the attributions scrambled.

Checked claim-by-claim against the existing docs before acting on any of it.

### Already filed, and in two cases the summary contradicts our own reading

* **Identity-biased gating (bias −2.0) + LayerScale 1e-4** — presented as
  generic "stabilization techniques". They are specifically **Silent Thinking,
  [2603.21676](https://arxiv.org/abs/2603.21676)**, already in §0 of this doc
  and flagged there as **opposing** rung 3 (it argues final-step-only loss and
  that intermediate supervision is harmful — we run `exit_pdf` on every loop).
* **Per-token fixed-point convergence** — already filed via **FPRM
  [2606.18206](https://arxiv.org/abs/2606.18206)** and **Two-Scale Latent
  Dynamics [2509.23314](https://arxiv.org/abs/2509.23314)**, and
  `references.md` records the opposite conclusion to the one implied: 2509.23314
  claims its step-size criterion **beats** KL-based early exit, and our own
  `UncertaintyHead` is "a third family, and the one that keeps failing here."
* Universal Transformers, RMT, Giannou/Geiping algorithmic looping — all filed.

### New, and worth having

* **[2607.14427](https://arxiv.org/abs/2607.14427) — *Per-Token Fixed-Point
  Convergence in Depth-Recurrent Transformers*, Joe Logan.** Read directly.
  135M-class model on FineWeb-Edu. Successive-output KL falls 3.9e-1 at loop 2
  to 8.5e-6 by loop 16. **Convergence depth is ordered by token type —
  whitespace shallowest, content words deepest. Median token converges by loop
  6; ~10% still updating at depth 8.**
  ⚠ **It does NOT claim more loops improve capability** — the opposite of how
  the summary framed it. Validation loss is FLAT past 8 loops; the result is an
  efficiency one (early-exit at 4.94 average loops, 38% fewer, same quality).
* **Loopie — MoE + looping**, 20B total / 2B active and 6B total / 0.6B active.
  Not in any of our docs. The only filed work combining our two axes (MoE and
  recurrence) at scale. Claim: small active footprint looped beats vanilla
  transformers at equal pretraining compute. **Unverified — no arXiv ID in the
  summary and not independently checked.** Find the paper before citing it.

### The one number that bears on our plateau

We run **K=4**. 2607.14427 measures the median token converging at **loop 6**,
content words deeper, ~10% still moving at 8 — on a 135M model, so if anything
a larger model needs more, not less.

That is consistent with what we see and it is NOT the same claim as "more loops
= more capability":

* fluent, grammatical, syntactically correct output — the token classes that
  converge by loop 2-4;
* wrong content — the classes the paper measures as converging deepest.

⚠ **Consistent-with is not evidence-for.** The paper's own capability finding is
that loss is flat past 8, and its 135M model is not ours. What it supplies is a
mechanism that would explain the failure shape, and a measurable prediction:
if K=4 truncates content tokens mid-convergence, then raising K at inference
should move content accuracy and leave syntax alone.

**That is testable on a checkpoint we already have, on the card, in minutes** —
`code_eval --n-loops` at 4 / 6 / 8 on `anneal@3000`. Not a training run. If
accuracy is flat across loop counts the hypothesis dies cheaply; the
2026-07-31 sweep did exactly this and found flat, but on a far weaker base and
before `exit_pdf` trained the halt gates.


---

## 2026-09-22 — 📐 LoopMoE (2606.04438) is our architecture, and it names a structural anomaly we have: we are 2-3.5x FFN-heavy

**LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language
Modeling** — Chen, Li, Huang, Yin, Shang, Qin. [arXiv:2606.04438](https://arxiv.org/abs/2606.04438).
Read via abstract, not the full paper.

> "MoE and looped architectures scale models along two orthogonal axes, namely
> parameter capacity and effective depth."

That is the framing of our entire plateau, from a paper on our exact
combination. It claims *"the first strictly controlled, head-to-head evaluation
of a looped MoE against a Vanilla MoE under identical total parameters,
per-token FLOPs, and active sublayer ratios"*, at 3B and 9B.

⚠ **Not "Loopie".** A prior summary named this work Loopie and attached
20B/2B-active figures to it. The paper is LoopMoE and the abstract gives no
parameter or token budgets. The same source produced two inconsistent
descriptions of it; treat both as unverified until the PDF is read.

### The mechanism worth taking: the active-parameter ratio

LoopMoE includes *"a capacity-balancing strategy that recovers the
attention-to-FFN active parameter ratio of well-tuned non-looped references."*
Measured on our own configs (active FFN = top-k experts only):

| config | attn | FFN (active) | **ffn:attn** |
|---|---|---|---|
| tiny 278M | 20.5M | 85.2M | **4.16** |
| **wide 436M** | 20.5M | 144.2M | **7.04** |
| large 922M | 136.3M | 427.8M | **3.14** |
| standard transformer (SwiGLU) | — | — | **2.00** |
| **Ouro-2.6B** (2048 / 5632) | 16.8M | 34.6M | **2.06** |

**We are 1.6-3.5x FFN-heavy against both a standard block and our own teacher.**
The wide model is worst because Net2Wider doubled `expert_dim` and touched
nothing else — all of that growth went to FFN while attention stayed frozen at
dim 1280. `large` is closest to normal at 3.14 because raising `dim` grew
attention too.

**Why this is a candidate and not just a curiosity:** attention is what moves
information BETWEEN positions — the operation that binds "ibuprofen" to "what
is it used to treat". A model starved of attention relative to FFN would be
expected to look fluent per-token and poor at relevance, which is the failure
we measure. None of the levers swept so far (tokens, width, α, LR, objective,
loop count) would have surfaced it.

⚠ **It is a parameter-count ratio, not a FLOP or capability measurement**, and
the causal claim is LoopMoE's, read from an abstract. What is ours is the
number: our ratio is off, by a lot, and nobody had looked.

### The other mechanism: IterAdaLN

*"Resolves weight-sharing symmetry via a modulation signal jointly conditioned
on the iteration index AND the per-token hidden state."* We condition per-loop
on the iteration index only (`InjectionScheduler`, a learned per-loop
log-scale; plus loop-index embedding). Conditioning on the hidden state too is
strictly richer — and "weight-sharing symmetry" is the exact failure mode the
expert-growth programme hit twice (twinned experts at ~90% similarity,
symmetric units that never separate, 2026-09-01).

### What to do with it

1. **Read the PDF.** Both mechanisms are described from an abstract here.
2. **The ratio is checkable and cheap**: a config with attention widened
   (more heads or larger head_dim) against FFN held, at matched total params,
   is a one-night A/B on the existing curve harness.
3. Do NOT retrofit `wide` — it is the 7.04 outlier and is being retired in
   favour of the fresh `large` lineage anyway.

## See also

- [parallel_loops.md](parallel_loops.md) — the parallel-paths design + §7 prior-art (PLT, RRT, Hyperloop).
- [decode_kernel_optimization.md](decode_kernel_optimization.md) — §6 self-speculative (speed child) + depth-wise batching.
- [ideas.md](ideas.md) — experiment shelf.
