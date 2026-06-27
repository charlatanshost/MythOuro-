# Deployment, inference efficiency & retrieval

> Post-training concerns — quantization/export, host-language strategy, and the RAG/retrieval application layer. Split out of `roadmap.md` (2026-06-27 doc reorg).


<!-- ===== moved from docs/roadmap.md (2026-06-27 doc reorg) ===== -->

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

