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

### 1b. Direct measurement — VTune confirms it's a tiny-kernel flood (2026-07-24)

Profiled the teacher decode loop under **Intel VTune 2026.2 `xpu-offload`**
(`tools.bench_harvest`, stock cache, batch 30, 128-tok). The launch-bound
diagnosis above is now measured, not inferred:

- **3,857,911 `zeCommandListAppendLaunchKernel` calls.** Of the GPU work, the
  `[Others]` bucket alone is **3,148,227 kernel instances at 0% SIMD** — i.e.
  **~80% of all launches are tiny scalar overhead kernels doing ~no vector
  work.** That flood of near-empty launches *is* the launch-bound tax.
- **The single hottest GPU task is `CatArrayBatchedCopy` — above every
  individual `gemm_kernel`.** That is the HF stock KV-cache `torch.cat` doubling
  every step. Direct proof of the pathology the prealloc cache exists to kill —
  and of *why* `--prealloc-cache` gives 2.06× at production length. (We profiled
  the STOCK path because the prealloc config segfaults under VTune — see below —
  so this `cat` cost is exactly what production already removes.)
- **The `gemm_kernel`s are 100% SIMD.** The teacher's real matmul work vectorises
  fully (XMX engages); the waste is entirely the *overhead around* the GEMMs —
  the `cat` plus the 3.15M-instance tiny-kernel tail. **That tail is precisely
  what `torch.compile` fuses**, which promotes §2's compile lever from "~+10%
  guess" to an evidence-backed target with visible headroom.

**Read counts/ratios, not absolute seconds:** VTune's tracing inflates wall time
(~464 s vs ~65 s native — tracing 3.86M launches is heavy), so the *structure*
(what launches, how many, which dominate) is the robust signal, not the timings.
Also: stock-not-prealloc, and 128-tok not the 768-tok production length (pattern
holds, counts scale). EU-occupancy/XMX% came back blank — `xpu-offload` traces
the Level-Zero API, not EU hardware counters; the EU/stall/idle breakdown needs a
**`gpu-hotspots`** run (which uses the perf sysctls below).

**Repro / gotchas (so future-you doesn't rediscover these):**
- VTune 2026 renamed the analysis: **`-collect gpu-offload` → `-collect xpu-offload`**.
- **`xpu-offload` GPU-page-faults on the prealloc-cache path** (`Segmentation
  fault from GPU ... NotPresent ... PDE Write`, NEO `drm_neo.cpp`). NOT a harvest
  bug — that config built the whole v2 corpus over many hours; it's profiler
  instrumentation perturbing the prealloc slice-write buffer. **Profile STOCK
  targets** (or a stock-only bench) until a lighter collection is found. The
  GPU recovers clean after the abort (`xpu-smi` → ~22 MiB); no zombie.
- GPU HW metrics need (sudo, runtime-only, resets on reboot):
  `sudo sysctl -w dev.i915.perf_stream_paranoid=0 kernel.perf_event_paranoid=1`.
- Working command:
  ```bash
  HF_HUB_OFFLINE=1 /opt/intel/oneapi/vtune/latest/bin64/vtune -collect xpu-offload \
    -r <fresh_result_dir> -- ../venv-xpu/bin/python -m tools.bench_harvest \
    --device xpu:0 --trust-remote-code --batches 30
  ```
  Finalisation continues (and salvages a usable result) even if the target
  crashes mid-run.

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

**Sibling of [parallel_loops.md](parallel_loops.md) — both children of its §2 *stranded-compute*
insight** (sequential recurrent-depth decode under-saturates the card; fill the idle SMs by widening
a parallel dimension). Parallel-loops spends that idle budget on **quality** (widen *batch* → N
diverse paths, *lossy*); this spends it on **speed** (widen *sequence* → draft-shallow / verify-deep
across future tokens, *lossless*). Same premise, different parallel dimension; they **compete for the
same idle-SM + N×-memory budget**, so they trade off rather than stack for free.

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
