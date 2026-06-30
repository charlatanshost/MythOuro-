# Decode-time kernel-launch overhead — design note

**Status:** design note, not implemented. **Stage-gated:** this is a *decode-speed*
optimization → only worth doing on a **coherent** model (faster degenerate output is
still degenerate). Sibling to [parallel_loops.md](parallel_loops.md) — same
latency/dispatch-bound decode regime, **orthogonal** fix. Captured from a vetted
external source so it's on the shelf for later, with the honest scoping that keeps us
from pulling it forward.

**Source (vetted):** analysis of Xiaomi's ~1000 tok/s **trillion-param MoE** serving
optimization via **mega-kernels / persistent kernels** (YouTube `mdPIjy-1Q6g`). The
*diagnosis* transfers to RDTs; the *solution* (hand-written mega-kernels) does not, for
us. This note records both halves so the distinction isn't lost.

---

## 1. The diagnosis (legitimate — transfers to RDTs)

At small compute-per-step, **kernel-launch overhead** dominates: the CPU must dispatch
each GPU kernel (norm, attention, matmul, MoE routing) as a discrete program. A forward
pass fires *hundreds* of launches; at sub-millisecond kernels, the setup/switch time
**dwarfs the actual compute**.

**Why it bites RDTs specifically:** a recurrent-depth block loops the *same* block K
times, so you pay the per-launch overhead **K times over** for the identical structure.
This is the same **latency / dispatch-bound, compute-under-saturated** regime documented
for sequential recurrent-depth decode in [parallel_loops.md](parallel_loops.md) §2. Real
observation, worth recording — it is *not* wrong.

### 1a. Empirical motivation — latency is CONFIRMED, and grows with scale

The "if latency matters" hedge is resolved: **it matters.** Running the teacher
**Ouro-2.6B-Thinking (`total_ut_steps=4`) decodes at ~3–8 tok/s** on a consumer card —
sluggish enough to feel. Decompose it:

- **Fundamental (the RDT floor):** K loop passes/token → an RDT decodes ~**K× slower**
  than a same-size dense model. Ouro pays ~4× the per-token compute of a 2.6B dense.
  Unavoidable — it's the architecture.
- **Fixable (most of the *felt* slowness):** Ouro runs the **worst-case path** — HF
  `trust_remote_code` **eager**, no `torch.compile`, no graph capture, SDPA fallback. A
  big chunk of 3–8 tok/s is unoptimized-eval overhead = exactly the K×-launch-overhead
  this note targets.

**Scaling makes this matter MORE, not less.** MythOuro is *not* staying at 278M — the Max
rig is for scaling up (continuous training on a dedicated box). As params grow, the K×
recurrence penalty rides on a bigger base, so a 1B+ MythOuro decodes more like Ouro than
like the current tiny model. **A scaled-up RDT walks straight into this latency wall** →
the cheap optimizations (compile + graph capture) are what keep it *usable* at size, and
**ACT halting** (easy tokens exit at 1–2 loops, cutting *average* depth) is the
architectural mitigation that scales with you.

## 2. The proportionate fix (≈90% of the win, ≈5% of the effort)

**NOT** mega-kernels. The standard tools capture most of the kernel-overhead win with
~zero bespoke-kernel code:

- **`torch.compile`** (CUDA) / **`torch.compile` on the XPU backend** (Intel Max) —
  automatic kernel *fusion* + launch reduction, ~one line. Already on the
  vetted-optimization list (hardware_options.md).
- **CUDA graphs / SYCL (Level-Zero) graphs** — capture a launch sequence once, replay it
  as a single unit. **The RDT loop is the ideal case:** the *same block every iteration*
  → capture the loop body **once**, replay it, killing the per-loop launch overhead —
  exactly the problem the video describes, *without* hand-written kernels.

`torch.compile` + graph capture is "address the plumbing" done **sanely** — it's what
production engines reach for *before* (and mostly *instead of*) mega-kernels.

## 3. The rejected tier (mega-kernels) — and why, for us

Fusing the whole layer + loop + routing into one persistent kernel is the hyperscaler
endgame. We do **not** pursue it:

- **Brutal to write** — hand-rolled memory barriers (thread races → *silent data
  corruption*), manual SRAM tiling, warp specialization. The source itself notes that
  major production engines avoid pure mega-kernels because they're so brutal to code.
- **Throwaway given the Intel move** — mega-kernels are bespoke *per backend*; CUDA
  mega-kernels would have to be rewritten in SYCL/Level-Zero on the Max port. Doubly
  not-now.
- **Wrong scale.** Xiaomi: 1T-param MoE, "20+ loops," serving-at-scale *economics*,
  target ~1000 tok/s. Us: **278M, max ~4 loops, single-user local** — "fast enough for
  one person," likely already met. The overhead-vs-compute ratio that makes mega-kernels
  pay off is the *opposite* of our regime.

## 3b. Combining with parallel paths (they stack — and synergize)

This and [parallel_loops.md](parallel_loops.md) are **orthogonal layers** and compose:
- **Parallel-paths** = *workload shape* — batch=1 → batch=N diverse trajectories
  (quality from otherwise-idle compute).
- **This note** = *execution plumbing* — fuse + replay the loop body (latency).

Stacked: **graph-capture the *batched* N-path loop body, replay per token.** They
synergize — the batched N-path forward is a *better* compile/capture target than thin
batch=1 (more regular, fills the kernels), and graph capture is *more* valuable when the
loop runs N× wide.

**The one design rule for combining:** *capture the static heavy compute; keep the
dynamic light control flow eager.* The N-path forward (fixed batch=N, fixed dims) is
static → capture/compile it. The **per-token arbitration** (gather N logits,
argmin-uncertainty, append) and the **ACT halt-count** are data-dependent → keep them
eager, *outside* the captured region. `torch.compile(model, mode="reduce-overhead")` does
compile **+** CUDA graphs in one knob and **graph-breaks gracefully** around those dynamic
bits, so it handles the split automatically. (ACT specifically: capture the *loop body*
once, eager-loop the *count* K — the counter is dynamic, the body is static.) **XPU
caveat:** the reduce-overhead / SYCL-graph path is less mature than CUDA — verify on the
Max port.

## 4. Sequencing

**Coherence first** (tokens + on-policy) → **then `torch.compile` + graph capture**
(latency is **confirmed** to matter — §1a — and grows as params scale up) → **bespoke
kernels basically never**, at our scale. Parallel-paths layers on top for quality once
latency is acceptable. Parked here on purpose; do not pull the kernel work forward of
coherence.

## 5. Already-done (don't re-derive)

The decode path's *memory* plumbing is handled: **depth-wise KV caching** (per-loop cache
keys), `compress_kv_cache` / `CrossLoopKVCache`, MLA latent caching, and the P1.6
snapshot/rollback fix (all in `mythouro/inference.py` + `main.py`). This note is about the
*launch-overhead* plumbing, which is **not** yet addressed — and is correctly deferred.

## 6. Algorithmic axis — self-speculative decoding via loop depth (DSpark-inspired, 2026-06-30)

A *different* decode-speedup axis from the kernel work above (1–4): **algorithmic**, not
launch-overhead. Parked as a **deployment-phase** item; revisit when serving a coherent model.

**Ref:** DeepSeek **DSpark** (2026-06-27) — *"Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation"* (60–85% faster *serving* on V4): a small **draft** proposes N
tokens, the big **target** verifies all N *in one parallel pass*, rejection-sampled → **lossless**.
Companion **DeepSpec** (MIT) trains draft models — *Qwen3/Gemma targets only*. Speed claims are
3rd-party-unverified; gains workload-dependent (structured/code > open chat). [VentureBeat /
MarkTechPost, 2026-06-27]

**Why parked, not adopted:** it's a *serving* optimization and the model isn't coherent yet
(training phase); DeepSpec targets standard archs, not our recurrent-MoE, so the *code* won't run
— mine the ideas, not the tooling (the IPEX pattern).

**The architecture-specific opportunity (why keep it):** recurrent depth makes us *uniquely* suited
to **self-speculative decoding with ONE model** — no separate draft to train/align:
- **Draft** N tokens with **few loops** (shallow), **verify** all N in *one parallel* pass at
  **full loops** (deep), rejection-sample → **lossless** *and* the parallel-multi-token speedup that
  DSpark's 60–85% actually comes from.
- DSpark's **confidence-scheduling** ≈ repurpose our **uncertainty / ACT gate** to decide
  draft-shallow vs verify-deep. "Semi-autoregressive" = draft multiple tokens per step.

**Relation to current code (verify before acting):** `inference.py` already has a loop-depth
draft/verify generator (`draft_loops` / `verify_loops`), BUT it *appears* to be the **lossy
early-exit / adaptive-depth** form (per-token "redo deep if uncertain"), NOT the lossless
parallel-multi-token form. So this is likely an **upgrade path**, not a from-scratch build:
early-exit → true lossless self-speculative decode. **TODO (deployment phase):** check whether the
existing generator is lossless or heuristic — decides "refine" vs "new decode path."

**Net:** our depth gives DSpark's win *without* a 2nd model — a differentiator, not a me-too.
Combines with the kernel work (§1–4) and [parallel_loops.md](parallel_loops.md); all gated behind
coherence.

## 7. See also

- [parallel_loops.md](parallel_loops.md) — same decode regime, orthogonal
  (quality-for-otherwise-wasted-compute) fix; can be combined with this.
- `mythouro/inference.py` — existing KV-cache machinery (the memory-plumbing half).
- [references.md](references.md) — Xiaomi mega-kernel video; Relaxed Recursive
  Transformers (continuous depth-wise batching, the prior-art batching lever).
