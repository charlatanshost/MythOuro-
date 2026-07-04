# Parallel Looped Paths — design note

**Status:** design note, not implemented. **Stage-gated:** only worth building on a
*coherent* base (see docs/training_runs.md — current base is collapse-fighting /
token-starved). This documents an original idea (owner's) so it's captured for the
future model card / writeup. Honest prior-art + scoping included on purpose.

**⚠ Naming clash — read first.** Distinct from **Parallel Loop Transformers** (PLT /
"LoopCoder-v2", arXiv 2606.18023, Jun 2026) despite the near-identical name. PLT *cheapens one
trajectory's depth* for **efficiency** ("only loop once"); this runs N **diverse** trajectories
for **quality**. Opposite attack on the same cost. Consider renaming (e.g. "parallel diverse
trajectories" / "multi-path loop ensemble") to avoid the collision. Full distinction in §7.

---

## 1. The core idea

Run the recurrent block as **N independent paths over the same input**, each going
through its own loop trajectory, then **arbitrate per-token across paths** using the
existing confidence/uncertainty machinery — emit, for each token, the
lowest-uncertainty candidate (or a confidence-weighted blend) across all paths and
depths.

```
            ┌── path A: e → loopA₁ → loopA₂ → … → ŷ_A  (uncertainty u_A) ──┐
input e ────┼── path B: e → loopB₁ → loopB₂ → … → ŷ_B  (uncertainty u_B) ──┼─► per-token
            └── path C: …                                                  ─┘   arbitrate
```

## 2. The motivation: saturate otherwise-wasted compute

This is the heart of *why* it fits **this** architecture specifically.

MythOuro's recurrent depth is **inherently sequential**: loop *t* needs loop *t−1*,
so depth **cannot be parallelized**. During **autoregressive decode** (batch = 1,
one token at a time, T = 1), each loop step is a *tiny* matmul (1 token × dim).
That regime is **memory-bandwidth-bound and severely compute-under-saturated** — the
GPU's SMs/cores sit mostly idle while we walk the loop chain serially. The compute
is *stranded*: we pay the latency of K sequential loops but use a small fraction of
the card.

**Parallel paths fill that idle compute.** Because the paths are independent, they
run concurrently — and the cleanest way to realize this is to make the N paths a
**batch dimension**: turn the under-utilized `batch=1` decode into a `batch=N`
decode where the N rows are *diverse trajectories of the same input*. The batch fills
the SMs the single path left empty. So you get **N trajectories' worth of "thinking"
per token at roughly constant wall-clock latency** — *up to the saturation point* —
arbitrated into one output by the uncertainty head.

This is the owner's insight, and it's correct **for the regime it targets**.

**This stranded-compute insight is the *parent* of a family — two ways to spend the idle budget.**
Filling the under-saturated decode by widening a parallel dimension has (at least) two distinct
payoffs, and this doc is only the *first*:
- **Quality spend — parallel paths (this doc):** widen the **batch** dimension with N *diverse
  paths* over the same token, arbitrate per-token → a *better token* at ~constant latency.
  **Lossy** (picks the best opinion across diverse trajectories).
- **Speed spend — self-speculative depth ([decode_kernel_optimization.md §6](decode_kernel_optimization.md)):**
  widen the **sequence** dimension — draft N *future* tokens with **few loops**, verify all N in one
  **deep** parallel pass, rejection-sample → the *same tokens*, fewer sequential deep passes.
  **Lossless** (DSpark-style; serving speedup).

Both reuse the shallow/deep loop primitive (§4 even lists "one shallow path, one deep path" as a
diversity axis) and the uncertainty head. Both cost **N× memory** and **compete for the same idle
SMs**, so you can't have both for free — doing both trades the saturation headroom between quality
and speed. Same premise, different parallel dimension, different payoff: *siblings, not subset.*

## 3. Honest scoping (where it helps, where it doesn't)

- **Helps: inference / decode.** `batch=1`, T=1 token generation is where the GPU is
  under-saturated and sequential depth strands compute. Parallel diverse paths use
  that idle capacity for quality at ~constant latency. **This is the use case.**
- **Does NOT help: training throughput.** Training already saturates the GPU via
  **batch size** (many *distinct* samples in parallel — the standard, correct
  saturation tool). Running N paths on the *same* input during training is redundant
  compute on one sample → it *slows* training, not speeds it. For "speed up
  training," increase batch / use the throughput levers (docs/ideas.md), not this.
- **"Free" has limits.** Compute is ~free only **up to the saturation point**: if
  `batch=1` decode uses ~X% of the card, roughly `100/X` paths fit before it stops
  being free and goes linear. And it is **never free in memory** — N paths = **N×
  KV-cache + activations** (VRAM cost). So: *compute-cheap until saturated, N× memory.*
- **Prefill** (processing the prompt, T = many) is already more saturated → less
  benefit there; the win is concentrated in the **decode** loop.

## 4. The crux: engineered diversity

N **identical** paths are correlated copies — they collapse to the same trajectory
and buy nothing (this is why naive ensembling fails; see docs/ideas.md). The whole
method lives or dies on making the paths **genuinely diverse**. Architecture-specific
diversity injectors available in MythOuro:

diversity injectors, **cheapest first** (all HYPOTHESES — to be validated, §8):

- **Sampling temperature** *(cheapest, no compute change)* — a temperature spread
  across paths (high = explore, low = exploit). Caveat: this diversifies the
  *sampling*, not the internal computation — temp-different paths share the same
  forward/hidden states and only diverge at token *selection*, then compound into
  **different token sequences**. So temperature is the natural fit for
  **sequence-level arbitration (Mode B)**, not per-token.
- **Injection schedule** — different `InjectionScheduler` magnitude per path (one
  anchors hard to input, one reasons more freely).
- **MoE routing temperature / bias** — perturb per-path routing so different
  **experts** fire → different *computation*, not just different sampling.
- **Loop budget / ACT threshold** — one shallow path, one deep path (depth diversity).
- **Initial-state / per-loop noise** — different seed for `recurrent_state_noise`
  (if enabled) or a small per-path perturbation of the initial recurrent state.

Temperature is the free starting axis; the *computational* injectors (routing /
injection / depth) decorrelate the actual computation and are what enable per-token
arbitration (Mode A) — and are the more novel part. Best design: layer temperature
on top of a computational injector.

## 5. Arbitration — reuse what already exists

MythOuro already has the machinery; this extends it from *one axis to two*:

- Today: **best-of-trajectory** emits, per token, the lowest-uncertainty *loop depth*
  **within one path** (inference.py `BestOfTrajectoryGenerator`), plus
  confidence-aware generation (cycle/confidence stops).
- Parallel loops: arbitrate over **paths × depths** — same uncertainty head, one
  extra axis. Two modes (theory; pick per validation):
  - **Mode A — per-token (paths stay aligned):** all paths agree on each emitted
    token; pick the minimum-uncertainty candidate across paths × depths per token.
    Requires *computational* diversity (routing/injection/depth) so paths differ
    while still agreeing on the token. The more novel mode.
  - **Mode B — sequence-level best-of-N (paths diverge freely):** each path generates
    a whole trajectory (e.g. at its own temperature), arbitrate at the end by
    aggregate confidence. This is essentially **self-consistency** (Wang et al.) —
    well-validated, but the less novel mode. Temperature diversity fits here.

**Beyond argmax-confidence — structured disagreement (lesson from MS MDASH, 2026).**
Picking the single most-confident path is *self-referential* and weak — an
under-trained model is hyper-confident about garbage (uncertainty ~0.01 on pure
repetition). Multi-model orchestration (e.g. MDASH) works because disagreement is
*structured and grounded*: **adversarial roles** (a path that argues *against*),
**agreement-as-signal** (paths converging = trust; diverging = spend more compute /
flag), and **external grounding** (tools / retrieval / execution — MDASH's "Prove"
stage). MythOuro analog, on a coherent base: add a critic path and use cross-path
(dis)agreement, not just per-path self-confidence. **Honest caveat:** MDASH's
constituents are *already-capable, genuinely different* models with *execution-grounded*
verification — conditions we lack (correlated copies of one model, self-confidence
only). It validates the *direction* and sets the bar for what makes multi-path pay
off; it is not evidence that same-model parallel loops alone will work.

## 6. Implementation sketch (when the time comes)

- **Paths as batch rows.** At decode, replicate the prompt into N rows, assign each a
  diversity config (injection/routing/budget), run the normal batched forward → the
  batch saturates the card. KV-cache is per-row (the N× memory cost).
- **Per-token arbitration** at emission: gather the N (logits, uncertainty) candidates,
  select/blend by confidence, append the chosen token to *all* rows (keep paths
  aligned on the agreed prefix), continue.
- **Knobs:** `--n-paths`, per-path diversity spec, arbitration mode
  (argmin-uncertainty | confidence-weighted), and a saturation-aware cap on N.
- Inference-only; no training changes; reuses the uncertainty head and the
  best-of-trajectory plumbing.

## 7. Prior art & novelty (honest)

> **Full survey + planning strategy:** [looped_lm_landscape.md](looped_lm_landscape.md) — the
> whole looped-LM neighborhood mapped by axis, with adopt / fork / preserve calls. The bullets
> below are the parallel-paths-specific slice.

- **Known / related:** best-of-N sampling, self-consistency (Wang et al.), parallel
  sampling, deep ensembles, mixture-of-experts. None of these is new on its own.
- **Same goal, different mechanism — note for honesty:** *Continuous Depth-wise
  Batching* (Relaxed Recursive Transformers, Bae et al., arXiv 2410.20672) already
  fills the under-saturated recurrent-depth decode by **batching tokens that are at
  different loop depths** together (shared block) for ~2–3× throughput. That is prior
  art for the *saturation* premise here, via batch-across-depth-states rather than
  N diverse parallel paths. Ours adds *diversity + per-token confidence arbitration*
  on top; theirs is pure throughput. Cite it; don't reinvent the batching.
- **Naming collision + OPPOSITE mechanism — cite explicitly:** *Parallel Loop Transformers*
  (PLT / "LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling", arXiv
  2606.18023, Jun 2026, 7B coders) share the **name** and the **starting problem** (sequential
  looping is latency/KV-cache-expensive) but take the *opposite* route. PLT **cheapens the
  looping itself** — cross-loop position offsets (CLP) + shared-KV gated sliding-window attention
  restructure *one* trajectory's depth so it isn't stranded — an **efficiency/architecture** move,
  single trajectory, **no diversity**. Ours **spends** the stranded compute on N *diverse*
  trajectories arbitrated per-token. So PLT is a **third response to the §2 stranded-compute
  root — it ELIMINATES the stranding rather than SPENDING it** (cf. the quality/speed children in
  [decode_kernel_optimization.md](decode_kernel_optimization.md) §6). They are in **tension, not
  complementary**: if PLT-style parallel depth saturates the card with one trajectory, our
  "diverse paths are ~free" premise weakens (the SMs are no longer idle). **A reviewer's first
  question will be "how is this different from PLT?"** — answer: different *mechanism* (diverse
  paths + trained-uncertainty arbitration vs position-offset attention), different *goal* (quality
  vs efficiency), *inference strategy on an existing model* vs *trained architecture*. (Abstract-
  level read, 2026-07-04 — confirm the exact mechanism against the full paper before the writeup.)
- **Plausibly original for MythOuro:** (a) framing parallel paths as a way to fill the
  **structural under-saturation of sequential recurrent-depth *decode*** (a systems
  argument specific to this architecture), (b) **per-token cross-path arbitration via a
  trained uncertainty head**, (c) the **architecture-specific structured diversity
  injectors** (injection schedule, MoE routing, ACT budget). The *combination* is the
  contribution. Do not claim "nobody has done this" without a literature check
  (seed Connected Papers from self-consistency + looped-transformers + deep ensembles).

## 8. Open questions to validate (cheap, on a coherent base)

1. **Profile the premise:** measure actual GPU occupancy at `batch=1` T=1 decode. Is
   it really under-saturated, and by how much? (Sets the "free up to N" number.)
2. Does **structured diversity** actually produce decorrelated trajectories (measure
   inter-path output disagreement)?
3. Quality gain of N-path arbitration **vs** single-path best-of-trajectory at matched
   latency — is the extra axis worth the N× memory?
4. Diminishing-returns curve in N; best diversity injector(s).

## 9. One-line summary

Use the **idle compute that sequential recurrent-depth decode leaves stranded** to run
**structurally-diverse parallel trajectories**, arbitrated per-token by the existing
uncertainty head — a quality-for-otherwise-wasted-compute feature for *inference* on a
coherent base, not a training speedup.

---

## See also

- [decode_kernel_optimization.md](decode_kernel_optimization.md) — the **orthogonal**
  decode lever from the *same* latency/dispatch-bound diagnosis: kernel-launch overhead
  in the looped block, fixed proportionately with `torch.compile` + graph capture (not
  mega-kernels). Parallel paths fill stranded *compute*; graph capture removes per-loop
  *launch overhead* — combinable, both stage-gated on a coherent base.
