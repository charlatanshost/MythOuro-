# References & Credits

Everything MythOuro builds on or drew ideas from. Credit where credit is due.

> **Note on accuracy:** arXiv IDs for papers we fetched and read directly are
> verified. Entries marked *(verify ID)* are cited from memory — confirm the exact
> identifier before any formal publication / model card.

---

## Foundation — base model & teacher

- **OpenMythos** — kyegomez. github.com/kyegomez/OpenMythos. *The fork MythOuro
  started from.*
- **Scaling Latent Reasoning via Looped Language Models** — Zhu et al. 2025.
  arXiv **2510.25741**. *The Ouro paper. Two results MythOuro depends on:
  (a) loop count is FIXED at 4 in training — they tried 8 and dropped back after
  loss spikes / gradient oscillations, which is why `max_loop_iters=4` is a sound
  inherited default and why growing past it is a warned-against path; (b) the
  loss is supervised at EVERY recurrent step, weighted by exit probability, so
  Ouro's exit gates are trained by the task loss while ours are shaped only by
  the depth regulariser. See docs/growth_design.md "loop-loss supervision".*
- **Ouro-2.6B-Thinking** — ByteDance. huggingface.co/ByteDance/Ouro-2.6B-Thinking.
  *The distillation teacher; a recurrent-depth model itself. MythOuro's student is
  vocab-aligned to it (logit-level KD requires shared vocabulary).*

---

- **Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs** —
  Pareja, Nayak, Wang, Killamsetty et al. (Red Hat AI Innovation / MIT-IBM Watson AI
  Lab / IBM Research), 2024-12. arXiv **2412.13337**. Models: Granite 3B & 7B,
  Mistral 7B, LLaMA 3.2 3B. Data: TULU, a taxonomy-driven set (825k samples), and a
  Math/Reasoning/Coding set. Evals: MMLU, MTBench, Open LLM Leaderboard v2.
  **Findings, in order of relevance to us:**
  1. **Batch size is the dominant lever.** "Larger batch sizes paired with lower
     learning rates lead to improved model performance." Tested 128 / 3,840 / 7,680
     samples per optimizer step; 7,680 beat 128 on MTBench 6.83 vs 6.41.
  2. **lr 2e-5 consistently best** across benchmarks — which is exactly what our SFT
     ran at, so the learning-rate hypothesis stays dead.
  3. "Early-stage training dynamics, such as **lower gradient norms and higher loss
     values**, are strong indicators of better final model performance."
  4. Stacked training beats phased; warmup steps and LR decay can be omitted without
     harm. (We use warmup 100; per this, harmless either way.)
  **Why it matters here — it names an untested confound in our SFT autopsy.** Our runs
  were at **effective batch 8 sequences** (`--micro-batch 1 --grad-accum 8`, the
  argparse defaults; inferred from 20.2k tok/s at ~0.40 s/step, seq-len 1024, since
  sft.py does not log its args). That is 16x below the WORST setting they tested and
  ~960x below their best — and with mean `resp_frac` 0.179, only ~1,460 tokens per
  step actually carried loss. Our loss also fell to 0.04 by step 150 with gnorm
  decaying 0.88 -> 0.15, which is the inverse of (3).
  **Caveat, stated plainly:** their batch-size effect is worth ~0.4 MTBench points on
  well-behaved 3B-7B instruct bases. Ours is 75% -> 0% code L3+. An effect that size
  does not explain a total collapse, so this is unlikely to be the whole mechanism —
  but batch is a re-allocation rather than extra compute (fixed throughput means a
  6.5h window is ~460k samples at any batch size), so it costs one night to rule out.
  ⇒ Test logged as the `checkpoints_v8_bigbatch` A/B at effective batch 256.
  **Does NOT address** exposure bias, on-policy learning, distillation or mode
  collapse — none appear in the paper.

### ⚔️ 2026-08-10 — THE LOOP-SUPERVISION CONTRADICTION (read before running rung 3)

Two sources in this batch make **opposite claims about the exact question
`--loop-loss-weighting` tests**, and a third (our teacher) sits on one side. This
is recorded as an open contradiction, not a settled point, because rung 3 is the
experiment that resolves it for our model.

| source | claim | evidence offered |
|---|---|---|
| **recurrent-depth-ttc** (repo, below) | **Supervise EVERY loop.** Iterative-target supervision extrapolates **up to 24x beyond trained depth**; **final-only supervision causes ACCURACY WALLS** | small-scale, seed-pinned chain tasks |
| **Ouro** (our teacher, 2510.25741) | per-step loss weighted by exit probability, `L = Σ_t p(t\|x)·L^(t)` | 1.4B/2.6B production models |
| **RLTT** (2602.10520) | credit to every loop beats terminal-only | +5.8%/+10.9% on Ouro |
| **"Thinking Deeper, Not Longer"** (2603.21676, below) | **Supervise ONLY the final step.** The "Silent Thinking Objective" computes loss at the final recurrence step *"eliminating intermediate shortcuts"*, and rejecting intermediate supervision is called **critical for genuine multi-step reasoning** | compositional tasks, 20+ recurrence steps |
| **MythOuro today** | final-only (`h_K`) | chosen to stop the λ₀→1 ACT collapse |

**Why this matters more than a literature disagreement.** Our own 2026-07-31
depth sweep was FLAT at 4/6/8 loops, and we read that as "depth is not a lever."
The `recurrent-depth-ttc` result offers a competing explanation: **final-only
supervision produces an accuracy wall at the trained depth**, which is precisely
the shape we measured. If that transfers, depth is not dead — it is walled, and
the wall is our objective, not our architecture. That would also mean rung 5
(grow depth) is attacking the symptom while rung 3 attacks the cause.

**The counter-case is equally concrete.** 2603.21676 argues intermediate
supervision lets the model take shortcuts rather than genuinely iterate — and we
have our OWN instance of that failure: the ACT-weighted-sum output gave the
optimiser a lever to pin λ₀≈1 and collapse depth, which is what returning `h_K`
fixed. So "supervise every loop" is not free; it is the thing that already broke
once here.

⇒ Run rung 3 (`--loop-loss-weighting uniform`) as a genuine two-sided experiment.
Watch the halt distribution and `loop_efficiency` alongside the evals: if depth
collapses toward loop 0 again, 2603.21676's shortcut objection is the live one;
if the flat-depth wall lifts, `recurrent-depth-ttc`'s is.

---

- **Teaching Pretrained Language Models to Think Deeper with Retrofitted
  Recurrence** — McLeish, Li, Kirchenbauer et al. (UMD / NYU / LLNL), arXiv
  **2511.07384**. *Same group as Huginn (2502.05171).* **⭐ CLOSEST TO OUR
  ARCHITECTURE OF ANYTHING FILED** — explicit **prelude / recurrent block / coda**
  surgery, which is our exact layout, and a **recurrence schedule** that ramps
  depth during training, which is our `LoopCurriculum`.
  Converts fixed-depth pretrained models (TinyLlama 1.1B, OLMo-2-1B,
  Llama-3.2-1B) into depth-recurrent ones by continued training on 50B tokens.
  TinyLlama recurrent hits **51.2% GSM8K vs 46.2% baseline at test
  recurrence = 32** — note 32, against our 4. Reports pretrained init needs
  ~950B fewer tokens than random init to reach parity.
  **Actionable for us:** (a) they switched **AdamW → Muon** specifically for
  recurrent stability — a cheap, untested lever on a rig where stability has been
  the recurring problem; (b) a *data* curriculum (general "healing" phase, then
  task data) which we do not do; (c) it is the strongest external evidence that
  test-time recurrence far beyond the trained depth can pay, which bears directly
  on rung 5. **Gate:** they retrofit strong pretrained bases; we distil a weak
  one, so the healing phase may not transfer.

- **Beyond Memorization: Extending Reasoning Depth with Recurrence, Memory and
  Test-Time Compute Scaling** — Rodkin et al. (MBZUAI / MIPT / AIRI), arXiv
  **2508.16745**. 1-D cellular automata with **disjoint train/test rule sets**, so
  it measures rule generalisation rather than memorisation. Compares
  Transformers, LSTM, Mamba, ARMT.
  **The number worth remembering: ACT buys about ONE extra reasoning step.**
  4-layer baseline ~95% at k=1 collapsing below 25% at k≥2; ARMT reaches k=2;
  **ACT ≈ +1 step**; GRPO reaches k=3 with no intermediate supervision; CoT
  exceeds 99% through k=4. Also: *"Depth — not width — drives multi-step
  accuracy"*, with embedding width giving minimal gains.
  **Why it matters here:** it is a sober prior on how much our halting work can
  possibly return. We have spent three efforts on the depth/halting axis; this
  says the ceiling for ACT specifically is ~1 step, and that *training method*
  (GRPO, CoT) moved depth further than architecture did. Consistent with our own
  finding that the objective, not the budget, is the binding constraint.

- **T2MLR: Transformer with Temporal Middle-Layer Recurrence** — Cai, Zhu, Dong,
  He, **Arora** (Princeton), arXiv **2607.15178v2**. Recurrence over a *middle
  slice* of layers, carrying cached representations from a deeper layer of the
  previous token into an earlier layer of the current one via gated fusion —
  i.e. recurrence **across positions**, where ours is across depth.
  135M/361M/1B on 10B FineWeb-Edu. Retrofit of SmolLM2-1.7B-Instruct:
  **GSM8K 35.78 → 39.88, MATH500 12.80 → 18.00**. Looping only ~20% of layers
  beat full recurrence on downstream reasoning, at **~8% inference overhead**
  rather than the multiplied cost of full looping.
  **Relevance:** "loop a SLICE, not everything" is a live architectural question
  for us — our recurrent block is the whole middle. Also the same family as
  Coconut in `ideas.md` (across-position latent reasoning), and the same "right
  family, wrong phase" gate applies: retrofitting an *instruct* model is not our
  situation. File as architecture-rev input, not a near-term lever.

- **The Recurrent Transformer: Greater Effective Depth and Efficient Decoding** —
  Oncescu, Morwani, Jelassi, Meterez, Kwun, **Kakade** (Harvard), arXiv
  **2604.21215**. Layers attend to KV pairs from *their own* activations rather
  than the previous layer's, giving layerwise recurrent memory at no extra
  decoding cost. Ships an exact tiling algorithm cutting HBM traffic
  **Θ(N²) → Θ(N log N)**. 150M/300M on C4: better CE than baseline with fewer
  layers, smaller KV cache, lower latency.
  **Relevance:** efficiency/architecture shelf. The tiling result is the
  interesting part for a latency-bound single card, but it is a pretraining-scale
  architectural change — not adoptable without a fresh distil. Same shelf as
  oFFN and linear-CE.

- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional
  Generalization** — Hung-Hsuan Chen (National Central University), arXiv
  **2603.21676**, 2026-03. "Vertical chain-of-thought": iterate a shared block in
  latent space instead of emitting reasoning tokens, at **20+ recurrence steps**.
  Three stabilisers: the **Silent Thinking Objective** (loss at the FINAL
  recurrence step only, *"eliminating intermediate shortcuts"*), **LayerScale
  init at 1e-4**, and an **identity-biased gated recurrence** (GRU-like gate,
  bias −2.0, ≈88% retention of the previous state). Graph reachability, nested
  boolean logic, relational composition; reports a sharp "computational frontier"
  and 1.6–1.75x OOD generalisation beyond trained depth. No ACT, no early exit.
  **⚔ THIS IS THE COUNTER-CASE to loop-weighted supervision — see the
  contradiction table above.** Two mechanisms are independently interesting
  regardless of which side wins: identity-biased gating (88% retention) is a
  principled way to make deep recurrence stable, and it is the same instinct as
  our `ρ(A)<1` contractivity tracking; LayerScale at 1e-4 is a close cousin of
  our depth-aware init.

- **duongtrongnguyen123/recurrent-depth-ttc** (GitHub, MIT, 2026) — controlled
  small-scale experiments on **prelude → core → coda** looped transformers, i.e.
  our layout. Seed-pinned, deliberately narrow. Three results:
  **(1) ⭐ iterative-target supervision (supervising EVERY loop) extrapolates up
  to 24x beyond trained depth, while final-only supervision causes ACCURACY
  WALLS** — the direct empirical case for `--loop-loss-weighting`, and a
  competing explanation for our flat 4/6/8 depth sweep;
  (2) after LoRA fine-tuning, inference depth is configurable per example via a
  **hardcoded halt rule** — adaptive test-time compute with **no learned stopping
  head at all**, which is a pointed counterpoint to three failed efforts on ours;
  (3) on real text at sub-1B, recurrent models showed **higher loss and sharper
  minima** than dense at the same token budget.
  **Caveat, stated plainly:** unreviewed, small-scale, synthetic chain tasks. The
  24x figure is not a claim about language. But (1) is testable on our stack this
  week and (2) suggests a cheap fallback if the halting head stays unfixable.

- **PrathibhaDevkar/rdt_transformer** (GitHub, MIT) — educational from-scratch RDT
  implementation in three phases (GPT baseline → recurrent core → finetune/CoT/
  tools), with the update rule `h(t+1) = A·h(t) + B·e + Transformer(h(t), e)`,
  the same LTI-injection shape as ours. **7 commits, 0 stars — teaching material,
  not a result.** Logged for completeness; nothing to adopt. Its benchmark claims
  are unvalidated and should not be cited.


- **Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories
  Improves Reasoning in Looped Language Models** — arXiv **2602.10520**, 2026-02
  (Princeton PLI talk listing). *Trained on **Ouro-1.4B and Ouro-2.6B-Thinking** — our
  exact teacher.* Code released.
  **Method (RLTT).** Standard RL on looped LMs (GRPO) "only considers the terminal
  latent state's distribution", so credit never reaches the intermediate loops that did
  the computation. RLTT instead forms the policy-gradient objective from the
  log-probabilities of **every loop iteration**, combined with weights `ω_t` that sum
  to 1, plus a KL term against a reference policy to protect language modelling. Three
  weightings tried: **uniform**, **progressive** (`∝ t^α`, later loops weigh more), and
  **exit PDF** (weights taken from the model's own **early-exit / halting
  distribution**). Drop-in replacement for GRPO, no external verifier or learned reward
  model; memory grows linearly in `T_max`.
  **Results vs GRPO:** +5.8% mean accuracy at 1.4B, +10.9% at 2.6B over MATH-500 /
  AIME24-26 / BeyondAIME. Per-benchmark gains reported at 2.6B: GSM8K +34.3,
  GPQA +18.7 (zero-shot, non-math), AIME24 +16.6, MATH-500 +14.4, BeyondAIME +10.0,
  MMLU-ST +3.5, MBPP +3.3, ARC-C +0.7. Trains to *shorter* responses, keeps Pass@k
  diversity, reaches higher reward earlier.

  **⇒ THE IMMEDIATELY ACTIONABLE PART IS NOT THE RL.** Our distillation loss has the
  identical defect the paper attacks. `training/distill.py:777-781` computes the
  divergence on `s_logits` from `student_fwd(x_in, n_loops=n_loops)` — **final-loop
  logits only** — so loops 0..K-2 receive gradient only indirectly through the
  recurrence. The paper's mechanism (weight the loss across all loop steps, `Σω_t = 1`,
  optionally with the halting distribution as the weights) transfers to a **KD
  objective with no rewards, no verifier and no RL at all**.
  **And it reframes "depth is dead" (2026-07-31, 2026-08-09).** Everything we killed was
  depth as an *inference-time exit-selection* lever — forced budgets, best-of-trajectory,
  a trained depth policy. We have **never** used depth as a *distribution over the
  training signal*. Those are different claims and only the first is measured. The exit
  PDF weighting also gives `ACTHalting` / `last_halt_distribution` a job that does not
  depend on the head being well-calibrated at shallow loops — which per-loop calibration
  says it is not (ECE 0.288 at loop 0 vs 0.013 at loop 3), and which is exactly why the
  exit-selection use failed. Corroborated by 2601.10242: intermediate loops are not
  linguistically accessible, so *reading* them fails while *training* them may not.

  **Gate on the RL half: a base that sometimes succeeds.** Policy gradient needs non-zero
  pass rate to produce gradient, and their base is Ouro-2.6B-Thinking, already strong on
  MATH-500. Ours is 5.0% math L3+ / 0.0% L4 — reward would be near-constant zero. RLTT
  proper un-parks only after the base can solve some problems; verifiable rewards
  themselves are available (OpenMathInstruct-2 ships `\boxed` answers).
  **Implementation note:** logits are `micro_batch × seq × 49,152`; at 8×1024 that is
  ~805 MB per loop in bf16, so four loops is ~3.2 GB if materialised together —
  accumulate the weighted loss loop-by-loop instead of stacking trajectories.

- **Loop as a Bridge: Can Looped Transformers Truly Link Representation Space and
  Natural Language Outputs?** — Chen, Liu, Shao (Shanghai AI Lab / SJTU), 2026-01.
  arXiv **2601.10242**. *Measured ON OURO (1.4B and 2.6B, 1-8 loop steps) — our
  own teacher family, so it transfers unusually directly.*
  **Core finding:** models are "largely insensitive to injections during the
  INTERMEDIATE loops — detection and identification only occur in the FINAL loop
  iteration." Semantics are committed at the final loop; intermediate loops are
  not doing interpretable work. They also report representation-probe accuracy
  DECLINING with iterations, so the narrowing gap between probe and textual
  verification reflects representational degradation rather than better
  self-verification.
  **Why it matters here — external corroboration of four of our own results:**
  * best-of-trajectory selecting intermediate loops was WORSE than a fixed final
    depth (CE 0.550 vs 0.261, 2026-08-06);
  * the UncertaintyHead is calibrated at the final loop (ECE 0.013) and badly
    miscalibrated shallower (0.288 at loop 0, 2026-08-09);
  * forcing depth beyond the halt point never helped — measured twice
    (2026-07-31 budget/forced sweeps);
  * the depth policy improved per-loop CE while making the TASK worse
    (2026-08-09).
  If intermediate representations are not semantically committed, selecting an
  intermediate exit is intrinsically lossy and per-loop CE at those depths
  measures something that does not transfer to generation. ⇒ Supports keeping
  `tools/grow_depth.py` and `training/train_depth_policy.py` SHELVED.
  **Does NOT address** SFT, exposure bias, teacher forcing or mode collapse — it
  is not a lead on the 2026-08-10 SFT-collapse problem.

## Hardware / inference efficiency

- **Efficient LLM inference solution on Intel GPU** — Wu et al., Intel, 2023-12
  (rev. 2024-06). arXiv **2401.05391**. *Intel engineers optimising LLM inference
  on our exact hardware class. Three contributions, and their relevance to
  MythOuro differs sharply:*
  * **Decoder layer fusion** (fuse data movement + element-wise ops to cut
    memory-access frequency) — the most relevant. Our recurrent block runs the
    SAME layer n_loops times per token, so any per-layer fusion win multiplies by
    loop count. But `docs/decode_kernel_optimization.md` already rules that
    hand-written mega-kernel tier OUT for us (bespoke per backend, throwaway on
    the Intel port, wrong scale at 278M/4 loops) and names `torch.compile` +
    graph capture as "≈90% of the win, ≈5% of the effort". Read this paper as
    evidence the fusion win is real on Intel silicon, not as a build order.
  * **Custom SDPA kernel** — low marginal value. We enabled torch's XPU SDPA
    2026-07-30: 8–14x on the attention kernel, ~0% end-to-end, because attention
    is not where this model spends its seconds (48 MoE experts x 4 loops is).
  * **Segment KV cache** (request/response KV in separate physical memory) —
    irrelevant to training, where rollouts run `use_kv_cache=False` deliberately
    (the ACT distribution-preserving finding), but relevant to DEPLOYMENT.
  *Headline 7x latency / 27x throughput is inference vs an unoptimised
  HuggingFace baseline — do not expect anything of that order here.*
  See also `docs/decode_kernel_optimization.md`, `docs/max1100_field_notes.md`.

## Architecture — recurrent depth / looped / latent reasoning

- **Huginn** — "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent
  Depth Approach," Geiping et al. 2025. arXiv **2502.05171**; code
  github.com/seal-rg/recurrent-pretraining; model tomg-group-umd/huginn-0125.
  *Closest published cousin. Documented our exact "Bad Run" collapse modes;
  source of the depth-aware-init + sandwich-norm + low-LR stability recipe, and
  the sobering 3.5B-params / 800B-tokens scale reference.*
- **MeSH** — "Memory-as-State-Highways for Recursive Transformers," Yu et al. 2025
  (ICLR 2026). arXiv **2510.07739**. *External memory + per-iteration routing to
  break uniform computation; gave us the rank/spectral collapse diagnostic
  (collapse_metrics.py) and independent confirmation of depth-aware init.*
- **Relaxed Recursive Transformers** — Bae et al. (ICLR 2025). arXiv **2410.20672**.
  *Recursion + layer-wise LoRA (we use LoRA-per-loop); Continuous Depth-wise
  Batching — prior art for the parallel-loops saturation idea.*
- **ETD** — "Encode, Think, Decode," Koishekenov, Lipani, Cancedda 2025. arXiv
  **2510.07358**. *Loop a reasoning-critical subset, added at mid-training.*
- **Loop Neural Networks for Parameter Sharing** — Ng & Wang 2024. arXiv
  **2409.14199**. *Basic looped refinement at GPT-2 scale.*
- **Coconut** — "Training LLMs to Reason in a Continuous Latent Space," Hao et al.
  2024. arXiv **2412.06769**. *Across-position latent reasoning (orthogonal axis to
  our across-depth); the warm-start fragment.*
- **Think before you speak: Pause Tokens** — Goyal et al. 2023. arXiv **2310.02226**.
  *Width-wise extra compute; we do the adaptive depth-wise version.*
- **Universal Transformers** — Dehghani et al. 2018. arXiv **1807.03819**.
  *Recurrent-in-depth transformer + ACT — a foundation of the design.*
- **Adaptive Computation Time (ACT)** — Graves 2016. arXiv **1603.08983**.
- **PonderNet** — Banino et al. 2021. arXiv **2107.05407**. *Halting / depth
  regularisation lineage.*
- **Reasoning with Latent Thoughts: On the Power of Looped Transformers** — Saunshi,
  Dikkala, Li, Kumar, Reddi 2025 (ICLR 2025). arXiv **2502.17416** (verified 2026-06-20).
  *Looped depth-extrapolation theory.*
- **Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers** —
  Kohli, Parthasarathy, Sun, Yao 2026. arXiv **2604.07822** (verified 2026-06-20). *Implicit
  multi-hop reasoning + compositional generalization in RDTs — the architecture-family
  citation in mythouro.md.*
- **Parcae: Scaling Laws for Stable Looped Language Models** — Prairie, Novack,
  Berg-Kirkpatrick, Fu 2026. arXiv **2604.12946** (verified 2026-06-20). *Stability of
  looped LMs at scale; theoretical backing cited for MythOuro's LTI-injection (ρ(A)<1)
  stable update (complements Deep Equilibrium Models for the contractive-recurrence theory).*
- **Mixture-of-Recursions (MoR)** — 2025. *(verify ID)* *Recursion + routing for
  adaptive depth — closest to our ACT+MoE combination.*
- **CoTFormer** — Mohtashami et al. 2024. arXiv **2310.10845** *(verify ID)*.
- **Deep Equilibrium Models** — Bai, Kolter, Koltun 2019. arXiv **1909.01377**.
  *Fixed-point/stability theory for the contractive recurrence.*

---

## Distillation & token efficiency

- **Knowledge Distillation (Hinton KD)** — Hinton, Vinyals, Dean 2015. arXiv
  **1503.02531**. *The soft-KL distillation objective we use.*
- **GKD — On-Policy Distillation of LLMs** — Agarwal et al. (ICLR 2024). arXiv
  **2306.13649**. *The on-policy mode-seeking lever — **VALIDATED as the collapse
  cure** (broke generation collapse domain-wide, 2026-06-28); now the core of the
  training recipe, not just a direction.*
- **MiniLLM — Knowledge Distillation of LLMs** — Gu et al. 2023. arXiv **2306.08543**.
  *Reverse-KL mode-seeking; teacher-mixed sampling; the "forward-KL degenerates a
  small student" insight that matches our exposure-bias finding.*
- **Sequence-Level KD** — Kim & Rush 2016. arXiv **1606.07947**. *Offline
  teacher-generated sequences — **implemented** as the teacher-corpus pipeline
  (`gen_teacher_corpus`); the R=0.2 A/B breached the plateau floor (2026-07-21).*
- **DistiLLM / DistiLLM-2** — Ko et al. 2024/2025. arXiv **2402.03898** (DistiLLM);
  DistiLLM-2 *(verify ID)*. *Skew-KL + adaptive scheduling refinements.*
- **DeepSeek-V3** — DeepSeek-AI 2024. arXiv **2412.19437**. *Aux-loss-free MoE
  routing (we use it) and Multi-Token Prediction.*

---

## Degeneration, decoding & exposure bias

- **The Curious Case of Neural Text Degeneration** — Holtzman et al. 2019. arXiv
  **1904.09751**. *Why models repeat; nucleus sampling. Our generation spiral is
  this phenomenon.*
- **Neural Text Generation with Unlikelihood Training** — Welleck et al. 2019. arXiv
  **1908.04319**. *Explicit anti-repetition training objective (backlog).*
- **Self-Consistency** — Wang et al. 2022. arXiv **2203.11171**. *Sequence-level
  best-of-N; the Mode-B arbitration in parallel_loops.md.*

---

## Initialization, data & misc

- **Takase et al. 2024** — depth-aware weight init (output-proj std scaled by
  effective depth). *(verify ID)* *Used by Huginn and MeSH; our
  `--use-depth-aware-init` implements it.*
- **Textbooks Are All You Need (phi)** — Gunasekar et al. 2023. arXiv **2306.11644**.
  *Data-quality → coherence at fewer tokens — the thesis the teacher-corpus A/B
  is now testing (2026-07).*
- **BeyondWeb** — scaling synthetic data for trillion-scale pretraining, 2025.
  arXiv **2508.10975**. *Rephrase-over-generate + frame diversity; the quality
  upgrade queued for the teacher-corpus generator (harvest_speedup_plan.md).*
- **Synthetic Continued Pretraining / EntiGraph** — Yang et al. 2024. arXiv
  **2409.07431**. *Squeezing small domain corpora; filed for the medical-domain
  phase (teacher_data_curriculum.md Rung 1).*
- **Clean SFT datasets** — Tulu-3, OASST2, OpenMathInstruct-2, NuminaMath-CoT,
  OpenCodeInstruct, MIRIAD, PubMedQA, ChemData700K. Full attribution/licences in
  `docs/clean_sft_datasets.md`. *OpenAI-free distillable SFT mix.*

---

## Surveys, discovery & industry reference

- **Alternatives To Next Token Prediction — A Survey** — Wyatt, Joshi, Salim (UNSW)
  2025. arXiv **2509.24435**. *Taxonomy; confirmed our latent-reasoning direction.*
- **Connected Papers** — connectedpapers.com. *Citation-neighborhood discovery.*
- **Awesome-Collection-Token-Reduction** — ZLKong.
  github.com/ZLKong/Awesome-Collection-Token-Reduction. *Deployment-phase reference.*
- **Microsoft MDASH** (2026 announcement) & **OpenRouter Fusion** — multi-model
  orchestration examples ("disagreement is information"); informed the parallel-loops
  arbitration discussion.
- **Xiaomi ~1000 tok/s MoE — mega-kernel / persistent-kernel serving** — YouTube
  `mdPIjy-1Q6g`. *Kernel-launch-overhead diagnosis (transfers to looped RDT decode) +
  the mega-kernel solution (rejected for our scale). Vetted writeup:
  `docs/decode_kernel_optimization.md` — proportionate fix is `torch.compile` + graph
  capture, not bespoke kernels.*
- **VibeThinker-3B** — WeiboAI 2026. arXiv **2606.16140**; github `WeiboAI/VibeThinker`;
  hf `WeiboAI/VibeThinker-3B`. *Dense 3B (Qwen2.5-Coder-3B) at frontier reasoning via
  "Spectrum-to-Signal" (curriculum SFT → MaxEnt RL/MGPO → offline self-distillation).
  **Validates the Compression-Coverage = reasoning-vs-knowledge split** that our
  RDT-reasoning + retrieval-knowledge bet depends on. Borrow/caveat analysis (RL needs
  verifiable rewards medical lacks; dense not RDT) in `docs/ideas.md` reference shelf.*
- **Nemotron-3-Ultra** — NVIDIA 2026. research.nvidia.com/labs/nemotron/Nemotron-3-Ultra.
  *Frontier 550B/55B-active MoE **Hybrid Mamba-Attention**; notable for open datasets
  (173B code + synthetic) + recipe. Relevant pieces: clean-licensed **data to vet**,
  **MOPD** (multi-teacher on-policy distillation) for the future on-policy stage,
  **Mamba-hybrid** long-context validation. NOT recurrent-depth (silent on our core);
  NVFP4 quant is Blackwell-only, not our Intel path. Borrow/caveat in `docs/ideas.md`.*

---

## Hardware — Intel GPU realization & benchmarks

- **Optimization of Ported CFD Kernels on Intel Data Center GPU Max 1550 using
  oneAPI ESIMD** — Zubair, Walden, Nastac, Nielsen, Bauinger, Zhu (ODU + NASA
  Langley + Intel), SC-W 2023. doi.org/10.1145/3624062.3624251. *Concrete evidence
  for the standard-vs-custom-kernel XMX-realization split: hand-written CFD kernels
  needed Intel-specific ESIMD (+ prefetch intrinsics, large-GRF, unreleased
  compilers) to reach ~67% of peak bandwidth / A100-class wall-clock; plain SYCL got
  31% and was up to 43× slower. A SINGLE TILE of the 1550 (≈ the 1100: same Xe-HPC
  block, 64 vs 56 Xe-cores, ~300W) matched the A100 on all 3 kernels after ESIMD.
  Caveat: FP64/FP32 CFD, not BF16 — informs the silicon ceiling + the custom-kernel
  effort, not the matmul path directly. Corrects a secondhand "within 10% of stated
  bandwidth" claim (actual: ~67%, hand-optimized).*
- **device-benchmarks** — chsasank. github.com/chsasank/device-benchmarks. *Pure
  matmul (2·n³/time, square a@a) + tensor-copy bandwidth microbenchmark. Source of
  the Max 1100 figures: 140 BF16 / 781 GB/s realized. Standard matmul via PyTorch/xpu
  (oneMKL/oneDNN) hits XMX out-of-box — the library does the ESIMD-level work the CFD
  paper had to do by hand.*
- **Intel Data Center GPU Max 1100 Datasheet** — Intel doc **817799**, Rev 1.0.
  *Primary source: 48GB (3 active 16GB HBM2e stacks), 300W TDP / programmable peak
  1.2–2.0× (default 1.52× ≈ 456W, max 600W), 12V-2x6 H++ connector, Xe Link X2/X4
  bridges (53 Gbps lanes).*

---

## Engineering tools & frameworks

- **PyTorch**; **HuggingFace Transformers / Datasets / Hub** (teacher + data).
- **FlashAttention** — Dao-AILab. github.com/Dao-AILab/flash-attention. *(currently
  unusable on the Blackwell 5070 — SDPA fallback.)*
- **linear_cross_entropy_loss** — Jonas Geiping.
  github.com/JonasGeiping/linear_cross_entropy_loss. *Memory-efficient head+CE.*
- **ml-engineering** — Stas Bekman. github.com/stas00/ml-engineering. *Scale-up
  engineering reference.*
- **bpeasy** — gautierdag. github.com/gautierdag/bpeasy. *Tokenizer training (future
  Rust port).*
- **bitsandbytes / torchao / DeepSpeed** — optimizer-state quantization / offload
  (VRAM levers; CUDA-dependent).

---

*Maintained alongside the work — add a line whenever a new source informs a
decision. See `docs/ideas.md` for how each was triaged and `docs/training_runs.md`
for where findings were applied.*
