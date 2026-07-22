# References & Credits

Everything MythOuro builds on or drew ideas from. Credit where credit is due.

> **Note on accuracy:** arXiv IDs for papers we fetched and read directly are
> verified. Entries marked *(verify ID)* are cited from memory — confirm the exact
> identifier before any formal publication / model card.

---

## Foundation — base model & teacher

- **OpenMythos** — kyegomez. github.com/kyegomez/OpenMythos. *The fork MythOuro
  started from.*
- **Ouro-2.6B-Thinking** — ByteDance. huggingface.co/ByteDance/Ouro-2.6B-Thinking.
  *The distillation teacher; a recurrent-depth model itself. MythOuro's student is
  vocab-aligned to it (logit-level KD requires shared vocabulary).*

---

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
