# MythOuro Roadmap

Living document tracking what's been built, what's queued, and what's
deliberately out of scope. Updated as we make decisions.

**What this project is:** a **research project on recurrent-depth transformers
(RDTs)** — training dynamics, distillation efficiency, and calibrated honesty —
with the applied goal of a small, private, **local-first medical-information
model**. The architecture is forked from **OpenMythos** (`kyegomez/OpenMythos`,
credited) and the model is distilled from **ByteDance Ouro-2.6B-Thinking** (the
teacher). It is *not* OpenMythos (extended into a trained pipeline + research
program), *not* Ouro (that's the teacher), and *not* Claude Mythos (speculative
inspiration only — **credited lineage, not the focus**). Trained checkpoints are
278M–632M proof-of-concept models — they validate the architecture and recipe,
not deployable quality. Full lineage writeup: README "Project identity & lineage".

**Attribution:** the upstream architecture is Kye Gomez's work
(`kyegomez/OpenMythos`, MIT) and is credited with thanks. He has no involvement
in this fork beyond that foundation, and no responsibility for its direction or
current state. The teacher (`ByteDance/Ouro-2.6B-Thinking`) is Apache 2.0. See
the README "Acknowledgements" and "Licensing & data provenance" sections.

**Full research credits:** every paper, dataset, and tool that informed the design
is catalogued in `docs/references.md` (with how each was used).

**Current status (2026-06-17, updated 2026-07-21):** the generation-degeneration investigation is
complete — it is **exposure bias** (a learned repetition attractor), **not**
recurrent/hidden-state collapse (reps are healthy; verified with
`tools/collapse_metrics.py`). v4's old "edge" was train-time noise co-adaptation,
not capability. Cure = **on-policy/GKD + tokens**; decode/inference-noise band-aids
ruled out. Full chain in `docs/training_runs.md` (06-16 entries); prioritised
backlog in `docs/ideas.md`.

**UPDATE 2026-06-23 — training instability SOLVED (a separate axis from the exposure bias above).**
The divergence-collapse fight (rev-KL output collapse @90M, JSD rank→1 @65M) was root-caused to
**LR — a gnorm explosion at 3e-4**, not the objective or the recurrence (ρ(A) stayed healthy
throughout). The **stability recipe** (`rev_kl + lr 1e-4 + --use-sandwich-norm --use-depth-aware-init`)
fixes it: gnorm flat <1.0, MoE cv 0.18 — the healthiest run yet, and **still *improving* at 53M where
the old rev-KL was already dead.** The base now **trains stably**, which is what unlocks "pour tokens +
on-policy" without it tearing itself apart. **New tension (06-23 eval):** rev-KL trades **calibration**
— a 06-23 read suggested ECE 0.20, **but the depth-matched verdict (step 6675, 06-24) is ECE 0.0152 —
that 0.20 was a depth-mismatch artifact; calibration is FINE.** The *real* result: **pure rev-KL
mode-COLLAPSES generation by ~109M tokens** — best-ever PPL (1.759) + good ECE (0.0152) + loop_eff 0.500
+ stability solved + healthy reps, and free-gen *still* collapses to `is is is`. **Exposure bias is
decoupled from every formal metric**; the stability recipe fixed *optimization*, not the rev-KL divergence
problem (it hit the same collapse the hot-LR rev-KL did @90M). **Next:** **stable-JSD** (cheap hybrid test
on the proven stable footing) → then the deep cure, **on-policy/GKD** (the student must train on its own
rollouts; no offline divergence alone reaches coherence). Full chains: `docs/training_runs.md`
(06-21/06-23/**06-24**), `docs/generation_probe_tracker.md`.

**UPDATE 2026-06-27 — on-policy IMPLEMENTED + PARTIALLY VALIDATED (the blocker's first break).**
The offline-divergence avenue is *closed* (fwd/rev-KL + JSD all collapse — **stable-JSD deprioritized**;
no offline objective reaches coherence). On-policy/GKD is now built (`generate_rollout` + the on-policy
step in `training/distill.py`, MiniLLM teacher-mixed sampling α; design + run cmds in
`docs/onpolicy_plan.md`), and the first run (warm-start **6675 → 6771**, ~96 steps, λ=0.5 α=0.6) produced
**the first movement on the unaided (α=0.0) generation metric in the project's history:** the prose probe
seed un-collapsed (top_share 0.45→0.14, distinct1 0.15→0.66; `the the the` → varied sentences). **Partial**
— medical/code seeds still collapsed = **dose-limited** (prose is over-represented in the corpus, un-collapses
first), *not* a mechanism failure. The question flipped from "does on-policy work?" (✅ **yes**) to "how much
dose?" — a **throughput problem** (5.8 min/step cross-GPU on the 12 GB 5070, micro-batch 1). That makes the
**Max 1100 the concrete unlock** (48 GB → *batched rollouts* → far more on-policy tokens/night; the decode is
latency-bound so the win is batching+`torch.compile`, not raw BF16 TFLOPS — `docs/hardware_options.md`).
**Next:** continue from 6771 at **λ=0.7**, more dose, re-probe, watch medical/code follow prose. Full record:
`docs/training_runs.md` + `docs/generation_probe_tracker.md` (06-27).

**UPDATE 2026-06-28 — ✅✅ COLLAPSE BROKEN DOMAIN-WIDE (the thesis flip).** At step **6906**
(~231 on-policy steps), a 6-seed probe shows α=0.0 `top_share` low on **every** seed (0.06–0.31)
— no hard repetition attractor anywhere, not just prose. **The exposure-bias blocker is cured.**
Regime is now "varied but incoherent" = a *normal undertrained small model* → the remaining gap
is **coherence/capability = tokens + scale** (the lever tokens *actually* move, unlike the
attractor). Capability shows at α≥0.5 (diabetes → correct symptoms; fibonacci → real code). The
earlier "medical still collapsed" read was partly **single-sample RNG noise** (same checkpoint,
different seed-order → bacterial α=0.0 went 0.97→0.18; probe now multi-samples). **Next = pour
tokens on the un-collapsed base** — exactly what the Max 1100's batched-rollout throughput is for.
Full: `docs/generation_probe_tracker.md` (06-28).

**UPDATE 2026-07 (through 07-21) — fluency SOLVED; the frontier moved to MEANING (a data-quality
wall), and teacher-generated data is the first lever to breach it.** The month's arc, all in
`docs/generation_probe_tracker.md`:
- **Hardware:** migrated to native Ubuntu on a single 48 GB **Intel Max 1100** (`torch.xpu`, no
  IPEX); teacher+student co-fit one card. Standalone write-up: `docs/max1100_field_notes.md`.
- **Rollout infra + a caught bug:** batched/cached rollouts (11.7× effective), then the discovery
  that the **cached student decode was NOT distribution-preserving** under ACT early-exit (~1 nat
  skew; corrupted steps 9780→12000 before a probe caught it). Rollout generation pinned uncached;
  `--min-lr` floor added after two legs were found starved at the cosine tail.
- **The plateau (confirmed, confound-free):** an n=5, real-LR, clean-instrument test showed **more
  *web* tokens no longer move α=0.0** at 278M. Not starvation behaving normally — the wall is
  **data quality**, not quantity.
- **The break:** a **teacher-generated corpus** (Ouro writes clean text; seq-level KD —
  `docs/teacher_corpus_plan.md`) at R=0.2 became the **first intervention to push the mean below
  the plateau floor since the 06 regime shift** — salad→framing-prose on the seeds that moved.
  Result is a *lower bound* (trained on a v1 corpus later found ~10% license-boilerplate from a
  head-seeding bug, since fixed). Prose gained; code/math lagged (the boilerplate), motivating a
  clean v2 harvest + the confirming A/B.
- **Strategy crystallized:** `docs/teacher_data_curriculum.md` (new seed domains → grow student →
  new teacher, gated by a measured student↔teacher parity signal) and `docs/harvest_speedup_plan.md`
  (ranked throughput levers, benchmark-gated).
**Next:** clean v2 teacher corpus → confirming R=0.2 A/B → if teacher data keeps paying, lean in
(more domains, higher R); if it plateaus too, the wall is capacity → **grow the student (Path A)**.
The go/no-go is now measured, not guessed.

---

## North star — what we are actually building (2026-06-12)

**Goal: a local LLM worth *using* — not a science project, and not a benchmark
challenger to frontier models.** The product axis matters more than the
research axis. We are not trying to out-score Qwen/Llama on general benchmarks
(they trained on 10–18 *trillion* tokens over millions of GPU-hours; no
architecture cleverness closes that data gap). We win on axes they neglect.

**Methodology — validate cheap, then scale (this is how the labs actually do
it).** Every frontier model descends from small ablation runs that settle the
recipe before the expensive run. The current 278M–632M work is the **test
bench**, not the ceiling: find the bugs cheap (we found 5), prove the recipe
cheap (warmup 500 / depth-reg 0.3), settle architecture questions cheap. When
scale comes — better hardware over time, and rented compute when justified —
it scales a recipe that's already debugged and measured, so the money isn't
wasted. The cheap discipline now is *what makes the expensive run later pay
off.*

**The three differentiators — and they COMPOUND with scale, they're not a
small-model consolation:**
1. **Calibrated honesty.** Most small local models hallucinate with total
   confidence — their worst, most-complained-about flaw. MythOuro has a
   working calibrated uncertainty head (ECE 0.01–0.04, demonstrated). **✅ Tension
   RETRACTED (2026-06-24):** a 06-23 read suggested reverse-KL regressed ECE to ~0.20, but
   the **depth-matched** eval (step 6675, n_loops=4 trained + inferenced) puts it at **ECE
   0.0152** — the 0.20 was a *depth-mismatch artifact*, not a real cost. rev-KL does NOT
   hurt calibration; this differentiator holds. A model
   that reliably says "I don't know" is *more useful* than a same-size model
   that doesn't — and a 3B that knows its limits beats a 3B that doesn't.
2. **Adaptive compute.** Recurrent-depth + ACT spends more on hard tokens,
   less on easy ones (loop_eff converges to exactly 0.500 across every run —
   the machinery is rock-solid and architecture-independent). Nobody's local
   model does this.
3. **Domain specialization (the wedge).** A small model fine-tuned *hard* on
   one domain can beat a general 7B *on that domain*. The clean science/medical
   data (docs/clean_sft_datasets.md) is the first wedge.

**Positioning:** *a small, fast, honest, domain-specialized model that knows
its limits* — exactly what the big local LLMs are worse at, not just smaller
versions of.

**What this means for decisions:**
- **Stop optimizing architecture at small scale.** The MoE-vs-dense ablation
  came back inconclusive (seed variance > architecture effect at 278M;
  docs/training_runs.md). That does NOT close MoE — MoE's benefits are *known
  to emerge at scale*, which is why every frontier model uses it. **Re-ask the
  MoE question at the scale where it's designed to matter (≥1B); keep MoE as
  the live default for the scale-up.** Don't burn more small-scale GPU on it.
- **First-order levers are scale, data quality, and the wedge** — not the FFN
  type. Optimize those.
- **Next concrete milestone:** v6 (clean-data SFT on the best base) → does an
  honest specialist emerge? Then scale toward 1B on the single card, 3B on
  rented compute, differentiators intact.

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

**UPDATE 2026-06-29 — the v5 "ceiling" RE-DIAGNOSED: token dilution + collapse, NOT an
expert/parameter ceiling (so "tapped out" is RETRACTED).** Two confounds, neither an
architecture limit: (1) **token dilution** — 48→96 experts on a *fixed, already-insufficient*
token budget **halved the tokens each expert saw**, so v5's experts were *more* undertrained
than v4's (`cv` stuck ~0.5 = undertrained routing, not saturated capacity); the architecture
outran the *data*. (2) v5 was **pre-on-policy**, so the whole model was exposure-bias
**collapsed** — the "gibberish at this scale" was the *collapse* (cured 2026-06-27), not a
parameter ceiling; scaling a collapsed model can't help regardless. **MoE expansion is a valid
tool; the guardrail is just: expand experts only (a) on the un-collapsed base, and (b) with
tokens scaling *in step* with experts — never ahead.** v5 violated both. (Credit: owner's
tokens-per-expert read — and Ouro itself is *dense*, so there was never a teacher expert count
to match against.)

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

> **Don't re-run the 48 → 96 MoE expansion *the way v5 did*** (token-starved, on a
> *collapsed* model) — that hit the apparent ceiling (2026-06-06). Per the 06-29
> re-diagnosis above, it's **not a banned operation** — it's a valid expansion *when
> token-justified on the un-collapsed base*. Expand experts only in step with tokens.

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

**Master router** — every doc and what it owns (grouped by purpose). Each
category has one home; this file (the roadmap) is forward-plan + lineage only and
points here for everything else.

**Plan & status**
| File | What's in it |
|------|--------------|
| [docs/roadmap.md](roadmap.md) | This file. Forward plan, milestones, scale-up execution, checkpoint lineage, decision rules. **Start here when resuming.** |
| [docs/onpolicy_plan.md](onpolicy_plan.md) | **On-policy / GKD** — the live cure for generation collapse: design, run commands, status (✅ validated 2026-06-27). |
| [docs/training_commands.md](training_commands.md) | Copy-paste-ready training/eval command reference + flag table. **What to run.** |

**Results & diagnostics**
| File | What's in it |
|------|--------------|
| [docs/training_runs.md](training_runs.md) | Every run's eval stats (PPL/loop_eff/ECE), recipes, raw-data paths, + external eval baselines. Update after each run. |
| [docs/generation_probe_tracker.md](generation_probe_tracker.md) | Generation-collapse probes across checkpoints × categories + the probe prompt set. The "is it learning, where, how fast" scoreboard. |
| [docs/failure_modes.md](failure_modes.md) | **Failure modes + recovery patterns** — the debugging/lessons reference (every failure hit and its fix). |
| [docs/moe_vs_dense_ablation.md](moe_vs_dense_ablation.md) | The pre-registered MoE-vs-dense gating experiment — protocol, configs, results. |

**Architecture & design**
| File | What's in it |
|------|--------------|
| [docs/mythouro.md](mythouro.md) | Full `MythOuro` API reference + architecture (incl. best-of-trajectory emission). |
| [docs/growth_design.md](growth_design.md) | MoE expansion / model-growth design. Read before promoting a checkpoint. |
| [docs/decode_kernel_optimization.md](decode_kernel_optimization.md) | Recurrent-decode throughput (kernel-launch overhead, `torch.compile` + graph capture). |
| [docs/parallel_loops.md](parallel_loops.md) | Parallel-loop ensemble idea (deferred post-coherence). |
| [docs/looped_lm_landscape.md](looped_lm_landscape.md) | Looped-LM literature map + **adopt / fork / preserve** architecture-planning (PLT, Hyperloop, MELT, RRT); the "exploit vs eliminate stranded compute" fork. |
| [docs/ideas.md](ideas.md) | Research-idea / experiment shelf — candidate experiments (depth-6, per-step loss), MoDr direction, open questions, paper triage. |

**Hardware & deployment**
| File | What's in it |
|------|--------------|
| [docs/hardware_options.md](hardware_options.md) | Scale-up hardware decision (Max 1100 plan + operating principle) + the hardware-scaling analysis. |
| [docs/deployment.md](deployment.md) | Post-training: quantization/export, host-language strategy, RAG/retrieval layer. |

**Data**
| File | What's in it |
|------|--------------|
| [docs/datasets.md](datasets.md) | Corpora reference — what we use and how. |
| [docs/dataset_selection.md](dataset_selection.md) | Data vetting rubric + decision log + the **data roadmap** / domain-expansion plan. |
| [docs/clean_sft_datasets.md](clean_sft_datasets.md) | Clean-SFT dataset registry + verified provenance status. |

**Provenance & reference**
| File | What's in it |
|------|--------------|
| [docs/references.md](references.md) | References & credits — every paper/source the project draws on. |
| [docs/glossary.md](glossary.md) | Project terminology. |
| [docs/fork_vs_openmythos.md](fork_vs_openmythos.md) | Verified code-level diff vs upstream `kyegomez/OpenMythos`. |
| [docs/mythouro_code_review_findings.md](mythouro_code_review_findings.md) · [docs/review_action_plan.md](review_action_plan.md) | The external code review (Fable 5, 2026-06-09) + its P0–P2 status tracker. |
| `archived_models/<name>/MODEL_CARD.md` | Per-checkpoint provenance. One per shipped reference checkpoint. |
| `CHANGES.md` | Changelog at the codebase root. |

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

## Strategy: exhaust the local rig before spending capital (2026-06-14)

**Principle:** a new rig or rented compute add *exponential* cost; the current
rig is ~free (time + electricity). So **wring every signal out of local before
committing capital** — commit only once the evidence says scale will pay off.
Same "validate cheap before committing" discipline that's caught every wrong
turn.

**The decision criterion (what justifies the spend):** does generation
**improve monotonically** as local resources increase? **Positive sign already
on record (2026-06-14):** v4 (more cumulative SFT) generates varied,
domain-relevant word-salad with appropriate uncertainty; small_sft (one SFT
pass) mode-collapses to repetition — same size, same code. So *more SFT visibly
moves the needle.* If the trajectory keeps climbing (repetition → word-salad →
phrases → coherence) as SFT / experts / width increase → **scale will pay off,
commit.** If it plateaus at word-salad regardless → scale is *required* (and
confirmed not a recipe problem). Either outcome de-risks the spend.

**Local-exhaustion sequence (all ~free, no capital):**
1. **Continued 420M SFT** — does more SFT reach v4-class varied generation?
   (running 2026-06-14; small_sft was under-SFT'd — one 3k pass vs v4's ~6.5k).
2. **More SFT / boost chat-register data** — SFT is the demonstrated lever and
   is VRAM-cheap (~10 GB, no quantize); pour it in. v4 credits OpenHermes with
   "unlocking the social register" → if SFT plateaus, enrich the clean mix's
   chat ratio (Tulu) before anything else.
3. **Grow 48→96 (632M) + 8-bit Adam + heavy SFT** — re-test the v5 expert
   ceiling on FIXED code + proper SFT. Legitimate retest: v5's ceiling was
   measured under broken code (P0.1 noise, P0.2-corrupted cv) AND truncated SFT
   (stopped step 2,887). All three now improvable. Caveat: more experts = more
   capacity a token-starved model may not fill — genuinely uncertain, hence
   worth the local test. Fits ~11.7 GB (v5 precedent).
4. **Build Net2Wider** (`grow_width.py`, ~2 sessions, unbuilt) — width growth
   toward ~1B, the one growth axis never tried; pure dev time, no capital.

**VRAM ladder:** SFT on 420M ~10 GB (no quantize, lots of runway) → 632M needs
8-bit Adam (~11.7 GB) → ~1B = true local wall (even 8-bit Adam won't fit) =
where capital finally becomes necessary. "Quantize" here = **8-bit Adam
(optimizer state)**, a clean ~2 GB saver — NOT QAT (that's the separate
3B→INT4 deployment endgame).

**Capital is INCREMENTAL, never all-or-nothing (de-risks "what if scale is
pointless?").** The fear: spend big, still get gibberish, money wasted. The
safeguard: when local signs say "go," **rent a SMALL step first** (e.g. a few
$ of A100 time for ~100–200M tokens at a bigger size) and check the
monotonic-improvement criterion *continues at scale* before committing more.
You spend a little to confirm success keeps climbing — not a fortune to
discover failure. Why the evidence favors scale working (not gibberish
forever): the teacher (Ouro-2.6B) is coherent, the architecture is a proven
family (transformer + MoE both scale to coherence in published models),
LLM scaling laws are among ML's most robust findings, every local lever moves
generation the right way (PPL ↓ with tokens; v4 >> small_sft with more SFT),
and all health metrics are clean (no pathology — just undertraining). The real
risk isn't "broken design," it's "coherence needs more tokens than the budget"
— a budget question, answered incrementally, not an all-in gamble.

## Stage two: the token-volume scale-up (the path to first coherent text)

**The binding constraint is training tokens, not params or architecture**
(established 2026-06-13 by the v6 behavioural read — see `training_runs.md`).
moe_s0 saw ~16M distill tokens; coherent small models see ~2T (SmolLM2-135M is
*smaller* and fluent). The whole stage-one body of work (bug fixes, recipe,
ablation, clean data, calibration) is validated — what's missing is **data
volume**, and the lever is **throughput × time**.

**Step 1 — the cheap de-risk probe (free, ~1.5 days, no new code):** run the
current 278M distill ~10× longer → **~160M tokens** (`--total-steps ~40000`,
proven recipe), then run the test-prompt suite. Decision:
- gibberish → partial words → phrases as tokens climb ⇒ **tokens confirmed as
  the lever**; rent for the full run.
- still pure gibberish at 10× ⇒ important negative; investigate before spending.
This answers "do more tokens fix coherence?" for the price of patience, before
any rental.

**Step 2 — the real run = rented compute, NOT this rig.** Target ~1B+ distill
tokens. One A100/H100 on Linux gives NCCL, a single fast homogeneous device, no
teacher-memory squeeze — ~1B tokens in roughly a day for tens of dollars. That
single rented card beats the local 3-GPU option on every axis.

**Why multi-GPU data-parallel distillation is IMPOSSIBLE on the local rig
(tested empirically 2026-06-13 — not theory):** a direct memory test put a
teacher + trainable student + fp32 Adam on the 4060 → **OOM at 7.16 GB during
the first forward**, before finishing one step. Data-parallel needs a teacher
(5.2 GB) on *every* rank, and that won't fit alongside a training student in
8 GB — so only the 5070 (12 GB) can hold a full distillation replica, and you
can't data-parallel with one eligible card. 8-bit Adam reclaims only ~1 GB;
fragmentation still sinks it. Compounding factors even if memory allowed:
**no NCCL on Windows (gloo-only)** → gradient all-reduce over CPU sockets;
**heterogeneous cards** → runs at the slowest/smallest. The only local
multi-GPU path that *could* exist is a **shared teacher server** (teacher alone
on the 4060 serving logits to student replicas on 5070 + 5060) — a real build
(DDP + inter-process logit serving) on gloo for ~1.5×, not worth it vs one
rented A100. The data pipeline + MoE-bias all-reduce are already DDP-ready
(`MixedDataset` rank/world_size, `_maybe_all_reduce_counts`), so DDP is a future
**Linux + homogeneous-GPU** move (incl. rented multi-GPU). The 3 local cards'
real use is **parallel independent experiments** (config/seed sweeps), not
speeding one run.

**Why teacher/student GPU overlap doesn't rescue local distill throughput
(measured + tried 2026-06-13):** the distill step is sequential — teacher
forward (5060) then student step (5070) — so the student GPU is idle during
the teacher forward. Measured per-step (mb1, seq512): **teacher 541 ms vs
student 286 ms** (teacher 65%), so the theoretical overlap ceiling is ~1.53×.
But it's unreachable in-process: (1) **async prefetch is a no-op** — the Ouro
teacher's custom forward self-syncs (reads tensor values mid-forward), so the
call blocks; (2) **threaded prefetch is *slower*** (measured 0.4k vs 0.6k
tok/s) — the teacher forward is Python-heavy (recurrent, custom mask/cache
logic) and **holds the GIL**, starving the student thread. Only a separate
teacher *process* (no shared GIL, IPC the logits) could claim the 1.53×, and
that's a heavy build for a ceiling a rented GPU laps. Also measured: the
teacher scales **super-linearly** with batch (541→1464→2542 ms for mb 1/2/4 —
memory-bound on the 8 GB 5060), so bigger batch makes distill *worse*, not
better (the student loves batch — 286→328 ms for 4× work — but the teacher
swamps it). Net: **local distill throughput is teacher-bound and not
improvable by batch or in-process overlap; reverted both attempts.**

**Tokens before params** is softened (2026-06-13): the user's own v1→v4
history shows scaling params *did* increase coherency at this token budget, so
**both levers work** — params is the empirically-validated, locally-feasible
one (grow moe_s0 → ~650M, fits the 5070), tokens is the bigger theoretical gap
that needs rented throughput. Pursue params locally now; rent for tokens later.

### PRIORITY next-session experiment: distill throughput (workers + compile)

The real bottleneck is **data supply, not compute** (established 2026-06-13):
GPU util stayed low (~12%/35%) while tok/s *rose* late in the run → the GPUs
wait on data the whole time; tok/s tracks how fast the CPU pipeline feeds them.
Per-micro-step: ~827 ms compute (teacher 541 + student 286) vs ~1450 ms wall →
**~620 ms is data-supply stall**. And the rig is a **56-core Xeon 8480** running
the dataloader at `num_workers=2` (distill.py:322) — ~54 cores idle. Two stacked
levers, run as a quick A/B next session before the next long distill:

1. **`num_workers` A/B (the big one).** Bump 2 → 4 → 8 → 16, measure tok/s, find
   saturation (= where the GPU becomes the real bottleneck). Potential **~1.5–
   1.75×** by collapsing the 620 ms data stall toward the 827 ms compute floor.
   Watch for: (a) HF streaming "list index out of range" when workers exceed
   shards — distill data is multi-shard so likely fine, but that's the failure
   mode; (b) the 8480's 2/8-channel memory bandwidth may cap effective workers
   around 8–16, not 56. First lever all session that targets the TRUE bottleneck
   AND uses owned hardware.
2. **`torch.compile(teacher)` — lowers the compute floor the workers feed.** During
   the teacher forward the 5060 only hits ~35% util — it idles between kernels
   (recurrent + custom-Python Ouro forward, eager-mode launch overhead).
   `torch.compile(teacher)` fuses kernels / cuts launch overhead → could shrink
   the 541 ms teacher forward directly (distill is teacher-bound, so a faster
   teacher speeds every step). One line; fails fast if HF custom modeling won't
   compile. The two stack: workers kill the data stall, compile lowers the floor.

Lower-priority / fallback throughput levers (see the overlap finding below):
teacher-process server (~1.4× overlap, process-only due to GIL, Windows-fiddly);
rent (laps all of it). Original single-lever note:

**`torch.compile` the teacher.** The diagnosis that motivates it: during the
teacher forward the 5060 only hits ~35% util — it idles *between* kernels
because the Ouro teacher is recurrent + custom-Python and pays per-kernel
launch overhead (eager mode). `torch.compile(teacher)` fuses kernels and cuts
that Python launch overhead, which could shrink the ~541 ms teacher forward
*directly* — and since distill is teacher-bound, a faster teacher speeds every
step (potentially a bigger win than overlap, which only hides the student's
286 ms). Cheap to try (one line; fails fast if it doesn't compile). Risk: HF
custom modeling (Ouro's recurrent forward) doesn't always compile cleanly.
Order of throughput levers: (1) try torch.compile teacher [cheap, root-cause];
(2) teacher-process server [the overlap design — works only across processes,
not threads (GIL); ~1.4× ceiling; fiddly on Windows, no CUDA-IPC]; (3) rent
[laps both]. The threaded prefetch was already tried and was *slower* (GIL
contention) — see the overlap finding above.

---

## Planned capability (post-coherence): retrieval → continual learning

A future direction (user goal, 2026-06-13), sequenced AFTER the base produces
coherent text — retrieval amplifies a model that can read and synthesise; it
cannot create comprehension. Two stages, escalating in ambition and risk.

**Stage A — RAG (retrieval-augmented generation), the near-ish, safe step.**
Embed a corpus (the clean medical/science data first, or a web snapshot),
retrieve top-k relevant chunks per query, feed them into context. Inference-
time, no retraining. Why it fits the north star: it offloads *knowledge* to an
external store so the small model only has to *reason over* facts — a 1B+RAG
beats a 7B-without-RAG on factual tasks, and it's the *right* design for the
medical wedge (retrieve from PubMed + cite, never hallucinate from weights).
Scaffolding exists (`RetrievalAugmentedInjector` in inference.py — untested at
scale; standard prompt-context RAG is the likely real implementation).
Knowledge stays **external**: updatable, citable, safe.

**Stage B — continual learning from retrieval (the user's vision: "what it
searches adds to its training").** Fold retrieved/searched information back
into the weights so the model permanently learns it. Genuinely powerful, and
genuinely hard — the known failure modes are why it's a multi-stage research
goal, not a feature:
- *Catastrophic forgetting*: naive training on new data erodes old skills →
  needs replay/rehearsal or parameter-isolation.
- *Provenance/poisoning*: training on retrieved web text = training on possibly
  wrong/adversarial content → the model learns falsehoods. Especially dangerous
  for medical.
- *Model collapse*: training on self-selected / self-generated data amplifies
  the model's own errors (a documented phenomenon).
- *Architectural tension*: baking knowledge into weights LOSES RAG's citability
  + updatability and REINTRODUCES hallucination risk. The thing that makes
  knowledge safe (external, sourced) is exactly what Stage B gives up.

**The safe path to the vision — curated offline retrain loop, NOT live online
weight updates.** Don't update weights live from web search (poisoning + collapse
risk). Instead: RAG at inference (Stage A) + a *human-in-the-loop curation
step* where good retrieved/searched content is filtered into the training
corpus for the *next* training round. This reuses the data-quality machinery
already built — `data/contamination.py` (eval leakage), `data/dedup.py`
(near-dup), provenance discipline (clean_sft_datasets.md) — as the gate that
keeps poisoned/garbage data out. Periodic, curated, verifiable; the model
improves across training rounds from vetted retrieved knowledge, without the
live-online-learning landmines.

**Sequencing:** coherence (Stage Two) → Stage A RAG → curated-retrain loop →
(only if ever justified) live continual learning. Each step is a real project;
the first one that matters is still just reaching coherent text.

### Parallel-loop ensemble (evaluated 2026-06-13, deferred post-coherence)

Idea (from a Grok thread, vetted): run N parallel recurrent paths, each with
its own best-of-trajectory exit, then review the paths' best-exits against each
other — a two-level selection (within-path trajectory pick + across-path
ensemble). **Not redundant with best-of-trajectory** (it layers on top), and a
real technique (test-time compute / self-consistency). Verdict: **deferred,
because the version with real value is expensive and gated on coherence.**
- *Shared-weight paths* (one model, perturbed injection/routing/n_loops): paths
  are highly correlated → cross-path review rarely helps → K× compute, thin gain.
- *Different-seed paths*: genuinely decorrelated (our seed variance proves it:
  5.72 vs 22) → legitimate ensemble — but it's an **N-model ensemble = N× the
  distillation cost**, and you can't ensemble models that aren't *individually*
  coherent yet (the precondition).
- The across-path *referee* leans on the uncertainty head, which P0.5 measured
  as miscalibrated at some depths → fix calibration before any ensemble-by-
  uncertainty.
- Ensembling correlated-wrong outputs from a pre-coherent base = wrong;
  test-time-compute scaling helps *most* on capable bases, least at small scale.
Revisit only with: a coherent base + spare compute for N models + P0.5 fixed.
Caveat on AI-sourced suggestions: vet against the actual code/bottleneck — the
original sketch had bugs (same block object N× = identical paths, K× compute for
zero diversity) and cited nonexistent DDP support.

---

## Where we are (as of 2026-06-12)

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

### Code state (post-review, 2026-06-09/10; updated 2026-06-12)

External code review (Fable 5) found and we fixed **5 correctness bugs** —
notably P0.1 (v1–v5 all trained with noise injected via a clobbered zero-init)
and P0.3 (eval emitted a never-trained path) — plus most of the P1 perf items.
The moe_s0 run above is the proof the fixes matter. Trackers:
[review_action_plan.md](review_action_plan.md) ·
[mythouro_code_review_findings.md](mythouro_code_review_findings.md).

**P0.6 — OpenCodeInstruct adapter silently rejected 100% of code samples
(fixed 2026-06-12 12:46 ET).** The `_to_messages_opencode` adapter in
`mythouro/sft_data.py` checked `str(status).lower() not in ("pass", ...)`
against `tests_execution_status`, but the field is a **JSON-encoded list**
(e.g., `'["pass", "pass", "fail"]'`), not a scalar string. The stringified
list never matches any of the expected values, so **every `clean_code` sample
was dropped** — confirmed by diagnostic logs: `clean_code: 0/5790 (0.0%
accept)` across the entire v6-attempt-1 run (steps 1–1532). Fix: parse the
JSON list and accept only samples where ALL individual tests passed.
**Consequence:** v6-attempt-1 (steps 1–1532) trained with zero code data;
run stopped, checkpoint discarded, restart required with the fix.

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

### Re-introducing explicit reasoning (post-coherence) — the base→Thinking sequence (2026-06-24)

If we distill from **base Ouro-2.6B** (concise targets, easier for the tiny student — see the
base-teacher A/B) to reach coherence, how do we get the "thinking" back? It's a **deferred, well-defined
step, not a hard problem** — with one key nuance:

- **Latent reasoning is free.** MythOuro is recurrent-depth — *the loops ARE reasoning* (iterative latent
  refinement), present regardless of teacher. What `Ouro-2.6B-Thinking` adds is **explicit, token-level CoT**
  (`"let me figure out… but wait…"`). So "thinking" = **latent (always there) + explicit (addable)**. For
  **medical**, explicit reasoning is worth it (a clinician wants to *see* the chain — interpretability/trust),
  but you're adding a *visible* layer on top of latent reasoning you already have.
- **Staged curriculum (easy→hard):** Stage 1 = base → coherence; Stage 2 = re-add explicit reasoning, via
  any of (all vocab-compatible, same Ouro family): (a) **continue-distill from `-Thinking`** (a coherent
  student learns the harder thinking distribution far better than from-scratch — swap teacher-id, continue
  from the coherent ckpt); (b) **CoT SFT** (reasoning traces — clean datasets, or harvest `-Thinking` outputs
  *with the `[0,2]` stop set* or you re-poison with cascade garbage); (c) **on-policy with the `-Thinking`
  teacher**.
- **The gate (honest):** this assumes Stage 1 *reaches coherence* — and the 06-24 verdict says offline
  distillation of *any* teacher likely collapses (exposure bias); **on-policy is the real coherence cure**.
  So "get thinking back" folds into the **on-policy roadmap**: reach coherence via on-policy, *then* dial in
  reasoning by teacher choice (`-Thinking`) + CoT data. Elegant version = the **Nemotron MOPD pattern**
  (multi-teacher on-policy): use **both** Ouro variants (base = grounding, `-Thinking` = reasoning) →
  coherence + explicit reasoning in one stage.
- **Practically:** don't over-plan now — post-coherence with known easy levers. *Now* = base-teacher A/B +
  scope on-policy. *Post-coherence* = add explicit reasoning (continue-distill / CoT SFT / on-policy, ideally
  multi-teacher). Latent reasoning via the loops is yours throughout.

Resolve the exact teacher **closer to the run** — the open-model landscape moves
fast; there may be a clearly-best option by the time you rent compute.

### Tokenizer graduation — how to reach a bigger teacher *without* restarting (2026-06-18)

The "Teacher = student vocab" coupling above has a corollary: our current vocab is the
**SmolLM2 tokenizer** (49152; `mythouro/tokenizer.py`, default Ouro-2.6B-Thinking), and
logit KD forces the teacher to share it. The only *same-vocab* teachers are the SmolLM2
family — all lateral-or-smaller scale. So a bigger teacher means changing the vocab.
Two ways to do that, and they're genuinely different paths:

- **From-scratch at scale (the plan above):** start the big run *directly* on the new
  teacher's vocab (Llama/Qwen/Gemma). Clean, but throws away the trained base.
- **Tokenizer graduation (the preserve-the-base alternative):** keep the current
  MythOuro base, *transplant* its tokenizer to the bigger teacher's vocab, then resume
  clean matched-vocab logit KD from that teacher. It's a **bridge between two matched-KD
  regimes** — migrate the anchor from SmolLM2/Ouro (small) to e.g. Qwen (big) — not
  cross-tokenizer KD (which keeps the vocab and is lossy every step).

**Mechanism** (Zett / FOCUS / WECHSEL): swap embedding + LM head for the new vocab, init
new tokens from the *old* embeddings of their sub-token pieces (warm start), heal with
short continued training. The transformer body and **the recurrent-depth reasoning
machinery are vocab-agnostic and survive** — re-skinning I/O, not relearning to think,
which is why it's cheap vs. from-scratch.

**Gated on:** (a) Ouro saturated (gains flatlined) AND (b) base structurally settled —
do it earlier and you heal embeddings you'll disrupt again. **Decision at run time:**
graduate (carry the base forward) vs. from-scratch (restart on the new vocab) depending
on whether the current base is worth preserving when compute arrives. Bonus: a bigger
vocab (Qwen ~151k) also compresses text better (fewer tokens/doc). Cost: bigger
embedding/head + heavier softmax (large param fraction at small scale → the *target*
vocab is its own choice), and capacity gap (favour a 7B–14B teacher, not a giant;
reverse-KL helps). Full deep-dive + technique refs: `docs/ideas.md` "Tokenizer graduation".

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

> **→ Hardware-scaling analysis** (memory-bandwidth findings, per-device perf, scale math) lives in [`hardware_options.md`](hardware_options.md).
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

> **→ Inference efficiency, quantization/export, host-language strategy & the RAG/retrieval layer** live in [`deployment.md`](deployment.md).
> **→ Data roadmap & planned domain expansion** (science/medical) live in [`dataset_selection.md`](dataset_selection.md).
> **→ External eval baselines** (context for the numbers) live in [`training_runs.md`](training_runs.md).
> **→ Glossary** lives in [`glossary.md`](glossary.md).
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

**DECISION (2026-06-11): clean-provenance data is the DEFAULT for all future
SFT.** The user may open-source or productise MythOuro; no future checkpoint
should carry the OpenAI-ToS constraint. The OpenHermes/Magicoder/MetaMathQA
mix is retired for new runs (v1–v5 remain archived research artifacts with the
constraint documented). Source: [docs/clean_sft_datasets.md](clean_sft_datasets.md).
With distillation data (FineWeb-Edu etc.) and the teacher (Ouro, Apache 2.0)
already clean, **the entire forward pipeline is now distribution-safe.**
Accepted tradeoff: v6-vs-v4 behavioural comparisons conflate new-base ×
new-data (decided knowingly). If/when productised: revisit the naming/
disclaimer posture and unpark the safety-alignment item first.

Full writeup in the README's "Licensing & data provenance" section. Short version:

- **Code**: MIT. **Teacher** (`ByteDance/Ouro-2.6B-Thinking`): **Apache 2.0** — clean for distillation/redistribution with attribution.
- **The gating issue**: the SFT datasets (OpenHermes 2.5, MetaMathQA, Magicoder-Evol-Instruct) all contain **OpenAI-generated data**, whose terms restrict training competing models. Fine for private research; a real constraint if distributing/commercialising the SFT'd checkpoints.
- **To get a cleanly-distributable checkpoint**: retrain SFT on non-OpenAI-provenance data — **a vetted registry now exists: [docs/clean_sft_datasets.md](clean_sft_datasets.md)** (Tulu-3, OASST2, OpenMathInstruct-2, NuminaMath, OpenCodeInstruct, MIRIAD, PubMedQA, ChemData — chat, math, code, medical, hard sciences). Distillation data (FineWeb-Edu, OpenWebMath, CodeParrot) is open-provenance and fine.

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

> **→ Failure modes encountered + recovery patterns** (the debugging reference) live in [`failure_modes.md`](failure_modes.md).
> **→ Open research questions & candidate experiments** (depth-6, Ouro per-step loop loss) live in [`ideas.md`](ideas.md).
> **→ Best-of-trajectory emission** (inference feature) is documented in [`mythouro.md`](mythouro.md).
> **→ MoE-vs-dense ablation** (protocol, configs, results) lives in [`moe_vs_dense_ablation.md`](moe_vs_dense_ablation.md).
> **→ MoDr (Mixture-of-Depth routing) candidate direction** lives in [`ideas.md`](ideas.md).
> **→ Probe test prompts** live in [`generation_probe_tracker.md`](generation_probe_tracker.md).
