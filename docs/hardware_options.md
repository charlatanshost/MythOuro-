# Hardware Options — the scale-up decision

## ✅✅ REAL BENCHMARKS RESOLVE THE COMPUTE QUESTION (2026-06-18)

Measured Max 1100 via **PyTorch/xpu** (github.com/chsasank/device-benchmarks):
**BF16 140 TFLOPs**, FP16 140, FP32 21 (~95% of spec), INT8 221 TOPS, **bandwidth 781 GB/s**.

This **settles the floor-vs-ceiling question favorably:**
- **XMX engages through standard PyTorch/xpu** — 140 is *way* above the vector floor (~44),
  ~33% of the 419 peak. **No hand-patching needed for standard matmul** → milestone-2 kernel
  worry largely evaporates for the common path.
- **~4× the 5070** (140 ÷ 33.7 = 4.15×) → the compute+VRAM instinct is **confirmed measured**,
  not estimated. Uncapped-XMX-vs-capped-consumer holds up empirically.
- **781 GB/s realized** — latency-derated (~64% of spec, confirms C&C) but still **> 5070's
  ~672** → better decode/serving too.
- **Beats A100 on FP32** (21 vs 19); ~56% of A100 BF16 but more VRAM (48 vs 40) at a fraction
  of the price. Strong datacenter-class value.
- **Nuances:** 140 is standard GEMM; custom recurrent-depth ops may realize somewhat less →
  aggregate likely a bit below 140 but still ~3–4×. Single source, but same PyTorch-matmul
  metric as the 5070's 33.7 → fair comparison.
- **4-card scaled:** ~560 TFLOPs BF16 / 192 GB. Serious rig, cheap, bubble-insulated.

**→ Decision validated with real data, not just defensible.** The biggest unknown (XMX
realization) is resolved at ~4×. Still confirm on *our* workload at buy-time, but the risk
collapsed from "is it an upgrade" to "is it 3× or 4× on our specific ops."

## 📄 Datasheet-verified specs (Intel doc 817799, Max 1100 Datasheet Rev 1.0)

Pulled from the owner's copy of the official datasheet — primary source, supersedes earlier
web-relayed/assumed numbers.

- **Memory:** 48 GB = **three active 16 GB HBM2e stacks** (3.2 GT/s/stack). A *fourth* stack is
  physically present but **fused off** → sensor readings for the dead stack are normal, not a
  fault. (Datasheet lists ~1.6 TB/s aggregate theoretical; device-benchmark realized 781 GB/s.)
- **Power:** TDP **300 W**, idle ~50 W. **Programmable peak power 1.2–2.0× TDP, default 1.52×**
  → **~456 W default-peak**, up to **600 W at 2.0×**. PWRBRK# emergency floor 95 W.
  Connector: **CEM 5.1 `12V-2x6 H++`** (12VHPWR-style).
  - **Build implication:** it is NOT a 300 W card under load — size the PSU for ~456 W (default)
    from the GPU, not 300. With the 9462 (~350 W) that's ~800 W peak → **1000 W ATX 3.0/3.1 PSU
    with a 12V-2x6 cable** (1200 W if ever run at 2.0×). Don't buy a plain 8-pin-only unit.
- **Xe Link (interconnect — my earlier "OAM-only" claim was WRONG):** edge connector has **six
  53 Gbps** lanes; *"high-speed coherent unified fabric connecting multi-GPU."*
  - **X2 bridge:** 2 cards, **six** Xe Link connections.
  - **X4 bridge:** 4 cards, **two** Xe Link connections (thinner per-pair than X2, but a real
    4-card fabric).
  - Bridges are a **separate accessory** (Intel Xe Link Bridge Card Datasheet doc# 788941; part
    numbers in datasheet Table 7-1) — "supports it" ≠ "have it"; source the bridge separately and
    it doesn't count against the $3000 single-card build.
  - **vs A30:** A30 NVLink caps at a **2-card pair**; the Max 1100 does **2- and 4-card** fabric.
    Another point to the Max 1100 — though our workload (compute-bound, no FSDP, role-separated
    cards) doesn't need it today; it's future optionality (sharding a bigger model later).

**XMX optimization toolkit (Intel oneAPI GPU Opt Guide, 2026-06-18) — resources milestone 2.**
The 140 BF16 = XMX/DPAS (448 engines on the 1100) via **oneDNN/oneMKL** — the path PyTorch/xpu
already uses (so 140 is out-of-box, no custom code). Full documented self-optimize stack:
- **Out-of-box:** oneMKL (GEMM) / oneDNN (DL) → auto-XMX → 140. Already via PyTorch/xpu.
- **Custom kernels (recurrent-block ops):** `sycl::ext::oneapi::experimental::joint_matrix`
  → maps to DPAS; tiling/layout/precision (BF16/FP16) guidance in the guide.
- **Verify/diagnose:** `intel_gpu_top` / XPUManager (XMX util), VTune (hotspots), Intel Advisor
  (roofline). → can *measure* whether our training hits XMX + find gaps.
- **AOT-compile** for the Max 1100 target.
Caveats persist: 140 ≈ 33% of peak (good out-of-box; more needs tuning vs CUDA maturity);
some impls (llama.cpp SYCL) report full-XMX-util is hard → expect ~140 standard, tune customs.
**XMX/compute investigation CLOSED — measured + tooled. Remaining = buy-time execution:** run
MythOuro on card #1, monitor XMX util via intel_gpu_top, joint_matrix-optimize custom ops if
underutilized.

**What the "140" actually is — a clean-GEMM ceiling, NOT a full-model rate (2026-06-18,
source identified + Gemini writeup vetted).** The 140 comes from **chsasank/device-benchmarks**
(github.com/chsasank/device-benchmarks) — a **pure matmul microbenchmark**: `torch.matmul` on
square matrices (sizes 256–6888) per dtype, plus a tensor-copy for bandwidth. Author: *"I use
matrix multiplication to measure FLOPs... copy a large tensor to measure bandwidth — the two
most important metrics for LLM inference."* So three numbers, in descending order, that we must
NOT conflate:
- **419 TFLOPS** — theoretical XMX peak (ideal/marketing; never sustained on real work).
- **140 TFLOPS** — *realized clean-GEMM* (this benchmark). Methodology (`benchmark.py`):
  `tflops = 2·n³ / time` on a square `a@a`, swept **per-size** (n = 256…~8192), **eager
  `torch.matmul`, no warmup**, `synchronize()` around timing, **fp32 default** (bf16 needs
  `--dtype bfloat16`). So 140 is the *best-size* clean GEMM. = **33% of peak on an ideal
  matmul**, a *bit low* for a tuned GEMM (a good oneMKL GEMM should clear ~70% of peak). The
  gap is plausibly the eager path + **no-warmup first-call overhead** + no `torch.compile` —
  i.e. the GEMM number itself *might* tune upward (where Gemini's torch.compile/sizes advice
  legitimately applies). 419 is not the target; ~140 is the honest realized matmul figure today.
- **Real MythOuro training throughput** — an **MFU fraction *below* 140** (training is
  MFU × the matmul ceiling, and MFU < 100% — *lower* for our recurrent-loop + MoE + ACT
  architecture with its many small awkward matmuls). **We do not have this number yet** — only
  a card-#1 run of a real MythOuro step gives it. Expect noticeably under 140.

**Correction to the prior draft of this note:** an earlier version called 140 "~33% MFU,
normal for LLM training." That was wrong — 140 is a *matmul* microbench, not an MFU. 33% is
realized-GEMM-vs-peak, not model-FLOPs-utilization. The MFU is the separate, lower, unmeasured
training number.

**Does this dent the buy-case? No.** The **~4× vs the 5070 (140 ÷ 33.7) is matmul-to-matmul**
— both are GEMM/peak-class figures, so the *ratio* is fair even though neither is the actual
training rate. Our training throughput on the 1100 will be some MFU below 140, but the 5070's
is likewise below its 33.7, and the relative ~4× compute advantage (plus 4× VRAM) is what the
decision rests on.

**Vetted optimization list (3 valid, 1 misfire) — for the card-#1 *real-model* MFU bench:**
- ✅ **`torch.compile` (Triton-XPU fusion)** — real win on launch overhead. *Caveat:* Triton-XPU
  may refuse to compile our **custom recurrent block / Ouro teacher forward** (same risk as
  compiling the teacher) → verify it compiles, don't assume.
- ✅ **Tile alignment** — XMX wants matrix dims as multiples of 64/128 (odd dims waste the
  systolic array). Our **vocab 49152 = 768×64 is already aligned**; confirm hidden/head/FFN dims
  too.
- ✅ **BF16 AMP** — `autocast(device_type="xpu", dtype=torch.bfloat16)`, NOT manual `.to(fp16)`
  (which can hit slow fallback). Use **bf16 not fp16** (our native dtype; same XMX speed, no loss
  scaling).
- ❌ **Channels-last (NHWC)** — IGNORE. That's a **CNN/conv** optimization for 4D image tensors
  (N,C,H,W); transformers have no channels/spatial dims. Inapplicable to MythOuro (the classic
  AI-suggestion-out-of-context error — vet against our actual workload).

## ✅ DECISION (2026-06-17): Intel Max 1100, scaled incrementally toward 4 cards

Owner chose the **Max 1100** over the B70 front-runner — deciding factor: **4-card
scalability = up to 192 GB** (4×48), incrementally (buy 1 now, add as the project grows
/ budget allows). The B70's 32 GB caps out below that; the 1100 is the only path here to
*serious* VRAM on a tight incremental budget, fitting growth ambition + the open ethos.
Accepted trade-offs: EOL line, unverified XMX realization, multi-GPU software effort —
all mitigated by the open, self-patchable stack (the project's through-line).

**Execution checklist (make the 1100 + 4-card path succeed):**
1. **Verify Xe Link bridge on the *PCIe* 1100** (unconfirmed — may be OAM/1550-only). Load-bearing for multi-card.
2. **Multi-XPU = milestone 2** (oneCCL + distributed train/serve on XPU; vLLM-XPU does multi-GPU serving). Single-card first.
3. **Rig infra for 4×300W:** ~1500W+ PSU, cooling, 4× PCIe x16 (or bifurcation), chassis. Plan the host for the 4-card endgame.
4. **EOL availability risk:** 1100 is discontinued → buying #2–4 *later* is a supply gamble. Acquire as a batch while available, or accept a card-count cap.
5. **Benchmark XMX engagement on card #1 BEFORE buying more** (vector floor 1.4× vs XMX ceiling 3–6×; patch oneDNN if it falls back — open stack). Don't buy 4 if #1 reveals an unpatchable XMX problem.

**Optimization knowledge (oneAPI training, 2026-06-18):** Intel's official SYCL GPU-opt
notebooks give **Max 1100-specific** occupancy numbers — 64 threads/Xe-core, 56 cores =
3,584 threads, **~112 properly-sized work-groups → 100% occupancy**, WG sizes ×64
(128/256/512/1024), sub-group 32, 128KB SLM. Two takeaways: (a) **the C&C "can't saturate
even with 500M threads" was likely a *thread-mapping* problem (partial dispatch), not a
hardware wall** — full occupancy is achievable with proper WG sizing → reassuring + fixable;
(b) **concretely validates self-optimization** (Intel ships the training + Occupancy
Calculator + knobs). CAVEAT: this is *vector*-engine occupancy, **NOT XMX** (notebook
explicitly excludes matrix engines). So it's the **milestone-2 toolkit** for hand-optimizing
custom recurrent-block SYCL kernels if oneDNN underperforms — but the *standard-matmul XMX
realization* (floor-vs-ceiling) still needs the benchmark + oneDNN-layer work, not this.

Gated on the token-curve: buy card #1 when the model proves it scales. Rationale for all
other options (B70, NVIDIA, etc.) retained below for reference.

---


Decision doc for what hardware (if any) to add for the next phase. Companion to
the roadmap's measured hardware analysis (bench numbers, the 8480 memory-
bandwidth findings) — this file is the **forward-looking purchase/rent
decision**, kept separate so it doesn't get buried.

Owner context: solo, self-funded ($1.5k is a significant, planned purchase),
**prefers to stay local** (dedicated rig planned), **Linux is fine**.

---

## ⚠ CORE TENSION (2026-06-17): compute-bound, want compute+VRAM, on a budget

Owner correction: **we ARE compute-bound** — SFT already pins the 5070's BF16 (pure
student compute, no teacher), and scaling the model deepens it. Goal: buy compute AND
VRAM together to avoid the next bottleneck. (Earlier "not compute-bound" was only true
for the *teacher-bound distill* case; offline precompute still helps *that*, but doesn't
address SFT/scaling compute.)

**KEY CORRECTION (2026-06-17): Intel IS a real BF16-compute upgrade — earlier "not a
compute upgrade" was WRONG.** The 5070's BF16-dense is **33.7 TFLOPS** (in roadmap.md;
matches our tok/s) — and that's *artificially low* because NVIDIA **caps GeForce
FP32-accumulate tensor throughput** (datacenter protection). Training needs FP32 accumulate
→ we're stuck at the capped 33.7. **Intel XMX (Max + Arc Pro) is UNCAPPED.** Recomputed vs
33.7: **B70 ~90–130 realized ≈ 3–4×; Max 1100 ~105–210 ≈ 3–6×.** Even pessimistic realization
clears ~2–2.5×. So both Intel cards are genuine 3–6× BF16-*training* upgrades + VRAM — the
owner's compute+VRAM instinct was correct; the error was comparing a capped consumer rate to
uncapped Intel. (Residual: exact multiple wants a real benchmark — risk is "3× or 5×," not
"upgrade at all.")

This reframes cards by **realized BF16 (not peak)** — a *certainty spectrum*:
- **Max 1100:** biggest *potential* (419 peak) but **a gamble** — FP32-vector benched badly
  (latency); **BF16-*matrix* path UNMEASURED.** Could be 2–4× the 5070 or ≈ it. Cheap. *Its
  whole cheap-big-compute appeal rests on this one unmeasured number → BENCHMARK BF16-matmul
  on an actual 1100 before buying; don't bet on the unrealized peak. The A380 can't tell you
  (diff arch + op-coverage ≠ perf).*
- **B70:** modest, more *reliable* (monolithic realizes better, ~183 peak) — likely ≥ 5070
  but not a leap. 32GB, open, current-gen. ~$949.
- **NVIDIA 4090 24GB (~$1500) / 5090 32GB (~$2000) / used A100 40GB (~$2500+):** **KNOWN**
  big realized compute (mature CUDA) + VRAM. Priciest; gives up the open stack.

**The tension:** cheap compute+VRAM (Max 1100 dream) vs *known* compute+VRAM (NVIDIA,
pricier). The 1100 promised both cheaply but its compute is **unverified + at-risk**.
**Resolve by:** stretch budget for known NVIDIA compute, OR verify the 1100's real
BF16-matrix perf before gambling. B70 is the safe-but-modest open middle.

Gated on the token-curve: the compute need bites when we SCALE the model; current ≤420M
runs on the 5070 (saturated but workable ~4k tok/s SFT). Buy when scaling. Ruled out:
**Tenstorrent** (TT-Metalium, niche, poor BF16/$) + **Gaudi** (Habana/OAM) — non-XPU.

### "BUY ONCE, BUY RIGHT" (sole-funder constraint, 2026-06-17) — the deciding frame
Solo + sole funder → **can't re-buy.** Buying for the *current* bottleneck is false economy
if we outgrow it. The purchase must serve the **scaling trajectory** → buy for *headroom*
(real compute + VRAM that lasts), not cheapest-current-fix. This **demotes the cheap options
as false economy:** Max 1100 (EOL + unverified compute = two longevity strikes), B70/3090
(only modest compute over the 5070 → compute-limited as we scale → the re-buy we can't afford).

**Headroom card: RTX 5090 32GB — MSRP $1999 but STREET $3000–5000 in the AI bubble.**
A100 40GB also inflated (~$2.5–5k+). So the ideal "known compute+VRAM headroom" NVIDIA
option is **bubble-priced out of reach for a sole-funder right now.**

### AI-bubble pricing reality (2026-06-17) — strengthens DEFER to near-decisive
The bubble inflates the cards everyone wants (NVIDIA 5090/A100, 1.5–2.5×) and **insulates
the ones nobody fights over (Intel Max 1100, B70 — cheap *because* no AI demand).** So
"cheap because unwanted" flips into a *relative* strength in an inflated market.

**→ Plan: DON'T buy now (doubly justified).** (1) token-curve gates the spend → runway;
(2) **the gating runway is also a bubble-hedge** — wait the bubble out; NVIDIA headroom may
return near MSRP by the time we need to scale; (3) offline precompute stretches the 5070
meanwhile. Don't let the bubble force a bad buy (overpaying 2× for NVIDIA, or a compromise
Intel card before needed).

**If forced to buy *during* the bubble:** the insulated Intel cards are the pragmatic value —
**B70** ($950, 32GB, open, current-gen, modest compute) or **Max 1100** (cheap, 48GB, EOL +
unverified-compute gamble) — accepting the compute compromise *because NVIDIA headroom is
temporarily unaffordable*. There is **no cheap-and-perfect option in this bubble**; wait if
you can, insulated-Intel-with-compromises if you can't.

### CONVERGED CONCLUSION (2026-06-17, after the 5070-cap correction)
The 5070's BF16 is *consumer-capped* (33.7 TFLOPS) → Intel's *uncapped* XMX makes the **B70 a
genuine ~3–4× compute upgrade + 32GB + current-gen + open + ~$950 + bubble-insulated.** That
satisfies **all** constraints at once (compute-bound, buy-once headroom, budget, bubble) — so
the B70 is **NOT a false economy; it's the buy-once answer that actually fits.** The earlier
"save for the 5090" path is undercut: the 5090 is bubble-priced ($3–5k) **and also
consumer-capped** (smaller real advantage than its price implies).
- **→ Front-runner: Arc Pro B70 32GB (~$950)** — real compute + VRAM + current-gen + open +
  bubble-proof. The buy-once pick.
- **Max 1100** — cheaper, higher ceiling (~3–6× + 48GB) but EOL + unverified realization: the
  gamble alternative.
- **used 3090** — only ~capped-consumer compute + 24GB: now clearly *not* enough headroom.
Still gated on the token-curve; confirm the B70/1100's *realized* BF16 with a benchmark at
buy-time (risk is "3× or 5×," not "upgrade at all").

**FLOOR/CEILING correction (2026-06-17, from Tom's B70 specs):** Tom's lists **FP16 45.88
TFLOPS — that's the *vector/shader* rate, NOT XMX.** ML matmuls use **XMX (~183 BF16, from
367 TOPS INT8)**, a separate unlisted spec. So the B70 has two FP16 numbers ~4× apart:
**vector floor 45.88, XMX ceiling ~183.** Which you get depends on oneDNN routing matmuls
through XMX:
- XMX engaged (intended) → ~90–130 realized → **~3–4× the 5070's 33.7**
- vector fallback (immature kernels) → ~45.88 → **~1.4×**
So it's an upgrade *either way* (even the floor beats capped 33.7), but **3–4× vs 1.4× is the
make-or-break and is unverified** — it hinges on whether XMX engages for *our* matmuls. The
benchmark must answer "does XMX engage," not just "is it faster." (Same caveat amplifies for
the Max 1100: its ~419 is XMX peak; if PVC's latency/oneDNN starve XMX, it too falls toward
its vector floor.)

**Open-source mitigates the XMX risk (2026-06-17):** because oneDNN/the kernels are OPEN, XMX
engagement is a *diagnosable, fixable* problem, not a passive gamble — if a matmul falls to
vector you can inspect kernel selection and force/patch the XMX path (impossible on closed
CUDA). So the floor risk becomes "1.4× *until fixed*, then toward 3–4×." Caveats: (a) real
SYCL/oneDNN kernel effort (LLM-assisted, not free); (b) standard BF16 matmuls likely already
route through XMX by default → fixing is edge-cases (~last 20%, e.g. our custom recurrent
block), not from scratch; (c) light on the B70 (current-gen, actively optimized) vs heavier
on the 1100 (PVC/EOL/less-optimized + less community). Framing: NVIDIA = closed-but-mature
(can't fix gaps); Intel = open-but-less-mature (can fix gaps, esp. LLM-assisted) — XMX
engagement is therefore not a dealbreaker for an owner willing to do the engineering.

## What we're actually trying to buy past

Established 2026-06-13/14 (see `docs/training_runs.md`): **the bottleneck is
token volume → coherence, and the local rig can't reach it.** The 278M ceiling
is thoroughly mapped — more tokens lower PPL but don't buy coherence; behavior
needs ≥420M; content fluency needs the real scale-up (more params **and** ~1000×
more tokens than the ~32M we can grind locally). So the scale-up needs **a
faster / bigger-VRAM device + throughput**, not a recipe change.

Current rig & its bottlenecks:
- **RTX 5070, 12 GB** — the student/training card. VRAM-bound (caps batch/model size).
- **Xeon 8480 ES, 56 cores** — CPU; data pipeline runs here. Note: 2/8 DDR5
  channels = memory-bandwidth-starved (roadmap failure modes).
- **RTX 4060 / 5060, 8 GB each** — teacher host (5060) / spare (4060).
- Distillation is **teacher-bound + data-supply-bound** (not compute-bound);
  multi-GPU data-parallel distill is impossible locally (teacher+student OOMs
  8 GB — roadmap). Throughput lever: dataloader `num_workers` (queued).

---

## Rent vs buy

| | Rent (A100/H100, Linux cloud) | Buy local |
|---|---|---|
| Friction | **zero** — CUDA, everything works | setup + drivers + capital |
| Cost shape | ~$1–2/hr (~$tens for a 1B-token run) | one-time, then "free" |
| Fit | best for a **one-off** scale-up validation | best for **ongoing** local capacity |
| Owner pref | — | ✅ "stay local" |

For a *single* scale-up run, renting an A100 is the lowest-friction, likely
cheapest path. For *ongoing* local work (the stated preference), a cheap
big-VRAM card wins — **if** the ecosystem friction is cleared first.

---

## Card options

| Card | VRAM | Ecosystem | Rel. cost | Friction for THIS project |
|------|-----:|-----------|-----------|---------------------------|
| **Intel Max 1100** (Ponte Vecchio) | 48 GB HBM2e | XPU (IPEX/oneAPI) | **cheapest** | teacher-on-XPU untested; no bnb; Linux-only; driver/IPEX setup |
| **AMD MI210** (CDNA2) | 64 GB HBM2e | ROCm | mid | ROCm > XPU maturity but < CUDA; HF custom-modeling hit/miss |
| **NVIDIA A100** | 40/80 GB | CUDA | priciest | **none** — gold standard, everything works |
| Used **3090 / 4090** | 24 GB | CUDA | low–mid | **none** — drop-in; 24 GB < the others but zero friction |

---

## Intel Max / Ponte Vecchio (the card under consideration)

**Attractive:** 48 GB HBM2e for a fraction of an A100/MI210; the XPU port
already exists (`mythouro/device.py` cuda/xpu abstraction + `rope_real`
complex-op fallback) — built specifically so the **MythOuro model** runs on
Intel; good HBM bandwidth (unlike the bandwidth-starved 8480).

**The three frictions, and where they land:**
1. **The HF Ouro teacher WAS the gating risk — largely de-risked 2026-06-17 via
   vLLM.** Originally: distillation needs the Ouro-2.6B teacher (HF *custom
   modeling* code, untested on XPU) → make-or-break. **But vLLM serves Ouro**, and
   vLLM uses its *own* model implementation (not the HF `modeling_ouro.py`), so it
   runs through vLLM's **XPU backend** — sidestepping the custom-code-on-XPU risk
   entirely. Confirmed from the repo files:
   - Ouro README: vLLM runs the model at **full `total_ut_steps` (4 recurrent
     loops)** — it just lacks the adaptive early-exit, which is a *serving* optim
     and **irrelevant for a teacher** (we want full-depth logits anyway).
   - vLLM logprobs API: **`PromptLogprobs`** (per-position logprobs over the input =
     teacher-forcing) with **`num_logprobs=-1` → full-vocab** (top-k for efficiency;
     exact-KD fallback available). Exactly what KD needs.
   So the gate shrank from "will custom HF code run on XPU?" to "confirm vLLM-XPU
   throughput + logprob extraction on Intel silicon" — a standard, smaller check.
   Plus it enables **offline precompute** (cache top-k prompt_logprobs over the
   corpus once → train student off the cache → teacher hardware irrelevant + no
   teacher in the training loop). Needs a small `distillation_loss` adaptation to
   consume top-k logprobs.
   **Second de-risk (2026-06-17, from reading `modeling_ouro.py`):** even the raw
   HF teacher is **portable plain PyTorch** — no flash-attn/triton/CUDA-only deps,
   device-agnostic (`.to(device)`, no hardcoded `.cuda()`), **eager attention by
   default** + SDPA/flash optional, standard ops only (Linear/softmax/RoPE/RMSNorm),
   recurrence = a plain Python loop. So even *without* vLLM, the HF teacher should
   run on `torch.xpu` and expose logits via our *existing* distill path. **Teacher
   gate now de-risked two ways.** Residual = *perf* (want SDPA on XPU, not slow
   eager) + op-coverage edge cases — both Arc-testable, not "will it run."
2. **bitsandbytes is CUDA-only** (no 8-bit Adam on XPU). See alternatives
   below — and largely moot at 48 GB.
3. **Linux datacenter part** — no Windows support (fine per owner). Setup is
   **lighter than originally feared: `torch.xpu` is native upstream PyTorch (2.12)**
   — Intel GPU support no longer requires IPEX (now optional perf only), and the
   API mirrors `torch.cuda` ~1:1 (our `device.py` port maps mechanically). So it's
   **PyTorch + Intel GPU drivers (oneAPI runtime / Level Zero)**, not an IPEX build
   project. The **1100 is PCIe dual-slot** (workstation-viable); the **1550 is OAM**
   (needs a baseboard — not a workstation card).

Also: Intel's data-center GPU roadmap has been turbulent → some long-term
software-support uncertainty.

### 8-bit Adam on XPU
bitsandbytes won't run, but:
- **torchao low-bit optimizers** (`AdamW8bit/4bit`) — native-PyTorch-based, the
  most portable "8-bit Adam" equivalent; verify XPU coverage.
- **Adafactor** — the safe bet: factored second moment (~√ the state of Adam),
  pure PyTorch → runs anywhere incl. XPU. Needs LR tuning; battle-tested (T5).
- **DeepSpeed ZeRO-Offload** — optimizer state → CPU RAM (you have 64 GB);
  Intel contributes XPU support; *more* savings than 8-bit Adam.
- **Most importantly: 48 GB largely obviates 8-bit Adam.** It was a 12 GB-budget
  crutch. fp32 Adam state for a 1B model is ~8 GB; model + grads + activations
  on top still fit 48 GB. Just run plain fused AdamW. The VRAM *is* the fix.

---

## ⭐⭐ FRONT-RUNNER: Arc Pro B70 32GB (VERIFIED 2026-06-17)

**The B70 is real** (Intel product page SKU 245797; Tom's/Phoronix/igor'sLAB reviews;
**#1 best-seller in workstation GPUs on Newegg**; released 2026-03-25, $949 MSRP /
~$999 street). Specs: BMG-G31, 32 Xe2 cores, 256 XMX, **32GB GDDR6 / 256-bit / 608 GB/s**,
22.94 TFLOPS FP32, 367 TOPS INT8, 230W, PCIe 5.0 x16, 4× DP 2.1.

**Why it's the front-runner — it inverts the central Intel risk:** the Max 1100's core
problem was *low adoption* (→ thin support, dead cloud, EOL). The B70 is the opposite —
**current-gen, best-selling, explicitly adopted for local LLM inference** → healthy
support/community/software-optimization tail. The EOL/thin-support worry largely *evaporates*.

It answers every prior concern: **vs B60** — +8GB (32 unified), +bandwidth (608 vs 456),
~2× compute (solves the 24GB limit); **vs Max 1100** — current-gen + adopted + monolithic
(realizes perf, no PVC latency), less VRAM but *usable*; **vs used 3090** — +8GB, open
stack, current-gen, lower power (230 vs 350W), built for local LLM, ~$250 more.

**Value:** at $949/32GB it *wins* vs **professional** NVIDIA (beats RTX Pro 4000 on price +
AI per reviews; pro cards cost 2×+ for similar VRAM). The "mediocre compute/$" critique was
vs *consumer* used cards — and 32GB + adoption + current-gen + open stack justify it for a
train+serve rig (we're not compute-bound anyway). Caveats: still not a consumer-card compute/$
bargain (~0.19 TFLOPS/$ FP16); realized XMX/oneDNN ML perf wants a benchmark (but monolithic
Battlemage + strong adoption = far better software optimization than PVC).

**→ NEW RECOMMENDATION: Arc Pro B70 32GB (~$949–999) is the buy** if going Intel — current-gen,
adopted, open stack, 32GB unified, built for local LLM. Supersedes B60 and Max 1100.
Still gated on the token-curve. The only competing pick is a used 3090 (pure CUDA value),
which gives up the open ecosystem + 8GB + low power.

## Battlemage Arc Pro B50/B60 (2026-06-17) — superseded by the B70 for our use

Intel **Arc Pro B-series = current, SUPPORTED gen (Xe2 Battlemage)** — which dissolves
most of the Max 1100's EOL risk. Same XPU stack (torch.xpu/oneAPI/vLLM-XPU), so all the
software de-risking transfers.

| | Arc Pro B50 | **Arc Pro B60** | Dual-B60 | Max 1100 (ref) |
|---|---|---|---|---|
| VRAM | 16GB GDDR6 | **24GB GDDR6** | 48GB (2×24) | 48GB HBM2e |
| Bandwidth | 224 GB/s | 456 GB/s | 456/GPU | 1,229 GB/s |
| FP16 XMX (peak) | ~85 TOPS-class | ~98 | 2× | ~419 |
| Power | 70W (slot-only) | 120–200W | ~300W | 300W |
| Price | ~$349 | **~$500–600** | ~$1,200 | EOL/used |
| Generation | **current** | **current** | **current** | **EOL dead-end** |

**Why B60 likely beats the 1100 for our needs:**
- **Current/supported gen** → EOL risk mostly gone; Intel actively optimizes oneDNN/
  torch.xpu/vLLM-XPU for Battlemage; bigger Arc community than PVC.
- **Monolithic die → no PVC memory-latency pathology** (the thing that hobbled the 1100)
  → realizes more of its peak + better LLM-decode/serving latency.
- **Workstation card → display output built in** (no separate A380) + easy power.
- **24GB = 2.4× our ~10GB usage** → plenty for ≤420M train + small serve.

**Honest trade vs 1100:** lower peak compute (~4×) + bandwidth (456 vs 1229), and 48GB
only via dual-B60 (2×24, needs model-parallel for >24GB) vs 1100's unified 48GB. BUT we
don't need that peak (tiny models), and the 1100 can't realize its peak anyway (latency).

**RECOMMENDATION SHIFT: a single Arc Pro B60 24GB (~$500–600) is likely the better,
lower-risk buy than the Max 1100** for current bounded needs (current support, better
latency, display built-in, easy power). Reserve Max 1100 / dual-B60 / rented-A100 for IF
we scale to models needing unified 48GB + high compute. Gated on the token-curve regardless.

**⚠ Value caveat (the honest economics):** Battlemage *Pro* cards are **mediocre BF16
TFLOPS/$** — a workstation premium (B50 ~85 TFLOPS/$349; B60 ~98/$550). The tell:
*consumer* B580 (~$250, same Xe2) gives more compute/$ — so the Pro price buys VRAM +
low-power + display + support, NOT FLOPS. **Two honest reads:** (a) at ≤420M we're *not*
compute-bound, so the poor compute/$ doesn't bite our workload — we're buying VRAM +
open stack + support; (b) but if pure value matters, Intel isn't the bargain: **used
NVIDIA 3090 24GB (~$700) is the value champion** (realizable TFLOPS/$, CUDA zero-friction,
good resale) — at the cost of abandoning the open/XPU direction. **The decision is
values-vs-value:** open self-maintainable stack (Intel, value penalty we don't feel at
our scale) vs best dollars-per-FLOP + zero friction (used NVIDIA, gives up the open
ecosystem). Intel's case was always the open stack + alignment, never "best value."

**Street prices (2026-06-17) sharpen this:** B50 ~$470–500, B60 ~$660–800, B70/dual-B60
~$1000–1200 (above MSRP, supply-tight). **B60 24GB ≈ used 3090 24GB (~$700)** — at parity
price the **3090 wins on value** (more *realizable* compute via mature CUDA, zero friction,
resale). So at street prices Intel has **no value/compute edge** at 24GB; the B60's case
rests entirely on **open stack + low power (70–200W vs 3090's ~350W → real always-on
operating-cost/cooling savings) + current-gen support + display.** Pure values-vs-value, no
tiebreaker.

**Ruled out (2026-06-17):** **Gaudi 2** — different software stack (Habana SynapseAI, NOT
XPU/oneAPI → none of our de-risking transfers), OAM socket (not workstation-pluggable),
niche/less-documented, roadmap-uncertain. **All OAM cards** (Max 1550, Gaudi OAM) — need a
baseboard, not workstation-viable. **Viable field = PCIe only:** Max 1100, Battlemage Pro
(B50/B60), or NVIDIA (used 3090/4090).

## Perf estimate: Max 1100 via Arc extrapolation (2026-06-17)

Rig purpose = **train + serve** (not just training).

> **⚠ Correction (2026-06-17): this extrapolation is for rough orientation only —
> it does NOT reliably predict perf.** What's the same across all Intel GPUs is the
> *software stack* (torch.xpu/oneAPI/oneDNN), so **op-coverage / "does it run"
> transfers** — but **performance does NOT.** There are *three distinct Xe μarchs*:
> Alchemist=Xe-HPG (Arc A), Ponte Vecchio=Xe-HPC (Max), Battlemage=Xe2 (Arc B).
> Spec-ratio scaling across them assumes equal per-unit efficiency, which is false
> (different cache/memory/XMX). **The DIRECT Chips and Cheese 1100 benchmark is the
> authoritative perf source** (and it showed the 1100 *under*-performing its specs —
> exactly what cross-arch scaling failure looks like). No Arc card is a perf proxy
> for the 1100; a cheap Arc only validates op-coverage + display. The table below is
> kept as a loose spec map, not a perf prediction.

| Spec | Arc A380 | Arc A770 16GB | **Max 1100** | A100 80GB (ref) |
|---|---|---|---|---|
| FP16 (XMX) peak | ~39 TFLOPS | ~157 | **~419** | 312 (BF16) |
| FP32 | ~4 | ~19.7 | **22.2** | 19.5 |
| VRAM | 6 GB | 16 GB | **48 GB HBM2e** | 80 GB HBM2e |
| Bandwidth | ~186 GB/s | 560 GB/s | **1,229 GB/s** | 2,039 GB/s |

**Scaling:** A770→Max ≈ ×2.7 compute / ×2.2 bandwidth / ×3 VRAM. A380→Max ≈ ×11 / ×6.6 / ×8.
- Compute-bound (training/prefill) → ×2.7 from A770. Memory-bound (decode/serving) → ×2.2.

**Vs A100 (peak):** ~1.3× the A100 on *peak* compute, ~60% bandwidth, 48 GB —
"A100-class on paper." **REAL-WORLD CORRECTION (2026-06-17, Chips and Cheese tested
the actual 1100):** peak badly flatters it. Real-world FP32: MI210 beats it 47%, H100
3×; they place it **between NVIDIA P100 (2016) and V100 (2017)**, NOT A100. Killer is
**memory latency** (L2 ~286ns, VRAM ~600ns) → **can't saturate its 1.2 TB/s.**
- *Caveat (cuts the other way):* they benched FP32/FP64 *vector*, NOT the **XMX matrix
  path** ML uses (oneDNN, ~419 TFLOPS FP16 peak) — so transformer perf is *unmeasured*
  and could be better than the "P100/V100" verdict.
- *Caveat that transfers to ML:* the **memory-latency hit lands on LLM decode/serving**
  (memory-bound) → serving throughput below the bandwidth spec. Real concern for the
  "run LLMs" half.
- **BF16 nuance (raises the ML estimate back up):** the "P100/V100 tier" verdict is
  for FP32/FP64 *vector*. But **P100 has no matrix accel and V100 has no BF16** (BF16
  hardware = A100+). The 1100's **XMX does native BF16/FP16 matrix** — a path those
  cards *lack*. So for BF16 transformers the 1100 is NOT a "P100/V100" card; they're
  not comparable on the matrix axis. ML ceiling is much higher than the vector verdict.
- **Honest cap on that:** having BF16 XMX ≠ realizing peak — memory latency *starves
  the matrix engines* (must be fed past ~600ns VRAM latency) and oneDNN maturity gates
  achieved %. So real ML ≈ between "V100-FP16-ish if starved" and "near XMX peak if
  well-fed" — *unmeasured*.
- **Reframe: not an "A100 bargain," but NOT a P100/V100 either — a cheap 48 GB card
  with real BF16 matrix accel, perf gated by memory latency + software.** Adequate for
  ≤420M train + small-model serve; gap bites only at multi-B serving. True number needs
  a real BF16/XMX/oneDNN benchmark. [chipsandcheese.com Ponte Vecchio deep-dive]

**Buy implications:**
- The Max is **compute-only (no display output)** → a train+serve rig built on it
  needs a cheap display GPU. An **Arc A380 (~$120) doubles as display + the
  op-coverage validation** (note: A380 is a weak *perf* proxy — ×11 from Max, 6 GB
  can't hold the 2.6B teacher; perf comes from the spec extrapolation, not the A380).
- **Serving synergy:** vLLM-XPU is a *serving* engine, so "vLLM-XPU on the Max" is
  both the distill-teacher path AND how you'd *run* models on the rig. 48 GB HBM2e is
  genuinely useful for serving (model + KV cache) in a way it wasn't for ≤420M training.
- For bounded current needs (≤420M train, small-model serve), even the derated Max is
  ample at ~½ an A100's price — *if* you accept the EOL/adoption risk below.

## ⚠ Venue + adoption risk (2026-06-17): the Max line didn't sell

Owner reports **Intel Tiber Developer Cloud appears gone/restructured** (current
entry `console.cloud.intel.com`, unconfirmed whether Max/PVC instances remain) —
and pinpoints the root cause: **the Intel Max GPU (PVC) and Xeon Max (HBM) lines
had low market adoption**, so Intel wound down the cloud that hosted them. That's
the key strategic fact, and it cuts **both ways**:
- **Why it's cheap:** ~half the price of competitors *because it's unwanted
  inventory* on a low-adoption line. Real value for a budget buyer.
- **Why it's risky:** low adoption → **thin support, scarce availability, no easy
  rental to test, low resale, uncertain driver/software longevity** (Intel's
  data-center GPU roadmap pivoted away from PVC). You'd be buying into a likely
  end-of-line product.

**EOL is universal — DECISION (owner, 2026-06-17): accept it.** NVIDIA cards EOL too
(A100/Ampere is aging out for Blackwell; V100/Volta is effectively dead for modern
frameworks despite nominal driver listings). So EOL is *not* an Intel-specific
disqualifier. The real axis is **usable-tail length + resale**, where NVIDIA wins
(bigger ecosystem carries cards years past EOL; deep resale market) — but the owner
is **buying budget/EOL gear either way, not relying on resale or a long tail.** Under
that strategy, EOL parity holds → the Max gives **more VRAM/$ for the same EOL
acceptance.** The one genuinely-Intel-specific residual — *software-stack stagnation*
(torch.xpu/IPEX/vLLM-XPU optimization for PVC may stall faster than CUDA's tail) — is
**mitigated by version-pinning a frozen known-good stack** (a fixed research rig
needs no bleeding-edge updates; owned HW + pinned stack ignores the vendor roadmap).
**Net: the Intel-Max value buy is defensible; the EOL con is no longer weighted
against it.**

**Open-source EOL resilience (owner, 2026-06-17) — this *inverts* the EOL axis.**
The Intel stack is largely open: oneAPI/oneDNN, Level Zero, `intel-compute-runtime`,
the **Xe kernel driver upstream in Linux**, and **`torch.xpu` upstream in PyTorch
core**. So at EOL you're **not locked out** — freeze a working stack OR rebuild/patch
it yourself against newer software. Contrast CUDA/cuDNN (proprietary): at vendor EOL
(V100) you *cannot* patch → hard wall, frozen forever. So open-source EOL is
*survivable* where proprietary EOL is terminal — which makes the Intel tail risk
**better-mitigated than NVIDIA's for an owned, kept-long rig.** Bonus: torch.xpu
being upstream means the PyTorch community carries some maintenance, not just Intel.
Caveat: self-maintaining an open stack takes effort + Intel's compute community is
small — realistic plan is freeze/pin, with self-patch as the always-open fallback.
Also aligns with the project's open ethos (open data, OpenAI-free, open stack).
**LLM-assisted maintenance further lowers the bar (and stacks with open source):** an
LLM can read the open oneAPI/Level Zero/IPEX source, diagnose build failures, and
write patches *you can actually apply* — impossible with closed CUDA at EOL. (This
very feasibility analysis was LLM-assisted desk research.) Nuance: LLMs are trained
on far more CUDA than Intel XPU, so they're weaker on niche oneAPI/Level Zero
internals — but working *against the open source* makes that verifiable/self-correcting.
**EOL plan, net: freeze/pin for stability → LLM-assisted self-patch as fallback.**

This matters a lot:
- **Intel Max (PVC) is NOT on mainstream clouds** (GCP/AWS/Azure) — Intel's own
  cloud was the accessible place to rent one. If it's gone or lacks Max instances,
  **you may be unable to de-risk the card before buying** → purchase risk jumps.
- A vendor restructuring its developer cloud + scarce rental availability **compounds
  the ecosystem-volatility caveat** above.
- The Max 1100's appeal was conditional on three things; status now: (a) teacher
  runs on it — **SOLVED via vLLM ✅**; (b) cheap pre-buy validation — **NOW IN
  QUESTION ⚠**; (c) ecosystem stability — **WEAKER ⚠**. Two of three got worse.

**→ Tilts back toward zero-friction CUDA:** rent A100/H100 (abundant; Lambda/RunPod/
Vast cheap) for the one-off scale-up, or a used 24GB 3090/4090 for ongoing local.
**Decisive check:** log into `console.cloud.intel.com` — are Max 1100/PVC instances
still offered, and at what price? Available → de-risk as below; gone → lean CUDA.

## The de-risk gate (do BEFORE buying) — UPDATED 2026-06-17

The desk-research above (vLLM runs Ouro full-depth; PromptLogprobs full-vocab
available) already cleared the *conceptual* blockers. The one remaining test is
runtime-only:

**Rent a PVC hour on Intel's Tiber Developer Cloud and serve the Ouro teacher via
`vLLM` on XPU, requesting `prompt_logprobs` (top-k).** Confirm: (a) vLLM-XPU loads
and runs Ouro, (b) prompt_logprobs come back correctly, (c) throughput is usable.
- Works → the teacher gate is fully cleared; a 1100 purchase is justified (and you
  can even precompute teacher logprobs offline, decoupling teacher hardware).
- Breaks/crawls → fall back to a CUDA card or rent for the teacher precompute.

(The original "run raw HF `training.distill` on XPU" test is now the *fallback*
path, not the primary — vLLM is the lower-risk route.)

Same discipline that's served the project: validate cheap before committing
capital.

---

## Recommendation (updated 2026-06-17 — supersedes 06-14)

Net of the 06-17 research (vLLM teacher, portable Ouro code, native `torch.xpu`,
Tiber gone, Max line low-adoption):

- **One-off scale-up validation:** rent **A100/H100** from a cheap specialized
  cloud (Lambda/RunPod/Vast, ~$1–2/hr) — zero friction, the right first move for
  "does more tokens fix coherence." GCP works but pricier; **TPUs are off-limits**
  (JAX/XLA; custom Ouro won't port).
- **Intel Max 1100 — software case now SOLID, but a low-adoption / likely-EOL
  line.** Teacher runs (vLLM-XPU *and* portable HF code ✅), `torch.xpu` native
  upstream ✅, no SYCL/IPEX-build needed ✅. It's cheap (~½ price) **precisely
  because the Max/Xeon-Max line didn't sell** — same fact that makes it risky
  (thin support, no rental to test, low resale, uncertain longevity). A calculated
  **value buy for bounded needs** (≤420M models today), eyes-open on EOL risk —
  not a safe long-term platform bet.
- **De-risking now that Tiber's gone (don't burn $300 on a throwaway Arc):** use
  free signals first — an **Arc iGPU you may already own**, and **existing
  community XPU benchmarks** — then treat residual *perf* as a calculated bet.
  Cheapest hands-on option if truly needed: a ~$120 **Arc A380**, not the A770.
- **Lowest-friction local if Intel feels too risky:** a used **24 GB CUDA card
  (3090/4090)** — less VRAM than the 1100, but drop-in, abundant, everything
  works, good resale.
- **Gate everything on the token-curve.** No hardware spend until the model proves
  it scales with tokens — and the cheapest path to that proof is rented A100/H100,
  not any purchase.

> Caveat: assessment as of early-2026 knowledge. Verify current Intel-cloud
> (`console.cloud.intel.com`) Max availability, `torch.xpu` op-coverage/perf, and
> PVC pricing/roadmap before deciding — those move fast.
