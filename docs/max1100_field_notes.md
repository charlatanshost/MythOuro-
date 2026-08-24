# Intel Data Center GPU Max 1100 — field notes & benchmarks

Real-world numbers from daily LLM training on a single Max 1100 (PVC, 48 GB HBM2e,
PCIe dual-slot, 300 W), measured 2026-07-12 → 07-18. Card is a **gray-market
engineering sample** running on **stock upstream drivers** — no vendor hand-holding
involved, which is rather the point: everything here is reproducible by anyone with
the card and a Linux box.

**Rig**: an all-Intel-datacenter-**sample** build (fitting — both are gray-market
on stock drivers): host is a **Xeon Platinum 8480+ (Sapphire Rapids, 56C/112T),
QYFS qualification sample** — reports as `Genuine Intel(R) CPU 0000%@`, the
telltale sample fingerprint. It matters here for two concrete reasons: it
provides the **PCIe Gen5 x16** the Max 1100 wants, and its 56 cores carry the
CPU-side harvest load (tokenization, the XPU CPU-sampling workaround, JSONL
writes). native Ubuntu (kernel 6.8, i915), `torch 2.13.0+xpu` from
`download.pytorch.org/whl/xpu` in a plain venv — **no IPEX** (it's EOL; Intel
upstreamed XPU support into PyTorch core, and `torch.xpu` mirrors `torch.cuda`
~1:1). Same box hosts an RTX 5070 (12 GB, `torch 2.13.0+cu130`) — all
head-to-heads below are same-rig, same-OS, same-PyTorch-version.

**Workload**: training a 278M-parameter recurrent-depth MoE student (custom
architecture: 4-loop weight-shared block, 24-expert MoE, ACT halting, MLA
attention) distilled from a 2.6B teacher (ByteDance Ouro) — teacher and student
resident on the one card. Not a cherry-picked GEMM benchmark; a worst-case-ish
real model full of small awkward kernels.

---

## Raw compute

| measurement | value | notes |
|---|---|---|
| bf16 GEMM, 4096², warmed | **224 TFLOPS** | ~54% of the 419 TFLOPS XMX marketing peak — healthy tuned-GEMM territory; XMX engages out of the box via oneDNN |
| same, eager / no warmup | 140 TFLOPS | the number you'll see first; warm the kernel before judging |
| memory | 48 GB HBM2e, ~real-world bandwidth not separately re-measured here | |

For scale: the RTX 5070's consumer-capped bf16-dense is 33.7 TFLOPS → ~6.6× on
paper. Real training does **not** see that ratio (below).

## Real training throughput (278M model, fwd+bwd, bf16, seq 512)

| batch | RTX 5070 (12 GB) | Max 1100 (48 GB) |
|---|---|---|
| 1 | **5,889 tok/s** | 3,333 |
| 8 | 10,662 (its ceiling) | 12,156 |
| 16 | OOM | — |
| 32 | OOM | 15,084 |
| 64 | OOM | 15,596 (plateau; 128 OOM) |
| 64 + `torch.compile` | — | **17,210** |

**The operating principle, confirmed on silicon: the Max only wins WIDE.** At
batch 1 the 5070 beats it ~1.8×; from batch 8 up the Max pulls ahead, and
best-vs-best it's ~1.6× — not the 6.6× matmul ratio, because this model's
per-loop small kernels (MoE experts, latent attention, per-loop LoRA) can't
saturate XMX. A vanilla dense transformer should land closer to the GEMM ratio.
The *qualitative* win is unconditional: batch 32–64 training and 2.6B-teacher
cohabitation simply don't exist on a 12 GB card.

## Sustained utilization on a real distillation workload

Measured from step timing on the actual training run (teacher fwd + student
fwd/bwd, effective batch 16 × seq 1024, ~6.1 s/step): **~60–65 TFLOPS blended ≈
25–30% of the realized GEMM ceiling**. ~85% of that is the teacher's big dense
forwards (which saturate well); the recurrent student alone draws only ~8 TFLOPS.
Lesson: PVC utilization is a function of your kernel sizes, not your enthusiasm.

## Decode / batch-1 inference (the weak spot)

Token-by-token decode is **kernel-launch-bound** on PVC — a 1-token forward
costs nearly what a 120-token one does. Consequences, measured:

- Greedy 32-token generations (batch 1): 5070 is ~2× faster than the Max.
- Autoregressive rollout generation (96-token, with 2.6B teacher in the mix):
  23 tok/s naive → **134 tok/s** by batching 32 sequences wide and caching —
  batch amortization is life.
- Long-form generation from the 2.6B (4-loop) teacher, 768-token continuations,
  KV-cached: **~56 tok/s at batch 24** — and batch 24 is the 48 GB memory cap
  at that length, because HF's KV-cache `torch.cat` pattern transiently
  *doubles* the cache each step. Wall-clock per decode step is ~flat in batch
  (launch-bound), so tokens/s scales with batch until memory says stop.
- **Preallocating the KV cache doubled long-form generation** (measured
  2026-07-23): the HF dynamic cache's `torch.cat` copy cost grows with sequence
  length — at 768-token generations it dominates. A preallocated slice-write
  cache + on-device sampling took the same workload from ~49 to **~101 accepted
  tok/s (2.06×)** and *reduced* peak memory at iso-batch (no cat transient),
  letting batch rise 24→30 in the same 48 GB. Caveat for benchmarkers: a
  128-token micro-bench showed only 1.4× — **cat-cost grows with length, so
  bench at production length.**

If your workload is single-stream chat inference, this is not your card. If
it's training or wide batch serving, carry on.

**Nuance worth knowing (measured 2026-07-21): batch-1 is only slow when SMALL
kernels dominate.** We ran the same batch-1 autoregressive probe two ways.
Student-only (278M) decode: the RTX 5070 is ~2× faster, as expected. But the
*same* decode loop with a 2.6B teacher mixed in per token: the **Max is ~2×
faster** (20 min vs 38 min, identical run). The teacher's large dense GEMMs
saturate XMX even at batch 1 and dominate the loop — so "batch-1 = weak" flips
to "batch-1 = strong" the moment a big model is the bottleneck. Match the card
to what dominates the inner loop, not to the batch size.

## `torch.compile` on XPU

Works (Inductor/Triton-XPU), compiled a nontrivial recurrent training step
end-to-end with zero graph breaks: **+10%** over eager. Two required
environment fixes and one anti-recommendation:

- `TRITON_DEFAULT_BACKEND=intel` — mandatory if an NVIDIA card is also visible
  (otherwise Triton dies with `2 active drivers`).
- `apt install intel-ocloc` — Triton shells out to it; missing = build failure.
- **Do NOT use `mode="max-autotune"`** — it replaces oneDNN's XMX GEMMs with
  Triton matmul templates, which *lose* on PVC (measured 14.9k vs 17.2k tok/s).
  Default mode only.

## Numerical fidelity

Cross-backend check: same checkpoint, same inputs, full forward on the Max
(with all workarounds below) vs the 5070 — max per-position KL divergence
**≤ 0.03 nats**, first-token distributions matching to ~0.01. The XPU stack's
outputs are trustworthy; bf16 near-ties flip greedy decode paths across
backends, exactly as they do between any two backends.

## The workaround list (all stable, none optional)

1. **PyTorch SDPA segfaults on XPU** inside HF models → force `eager`/manual
   attention (`attn_implementation="eager"`; custom models: bmm-based attention).
   - **⚠ PARTIALLY OBSOLETE as of torch 2.13.0+xpu (re-tested 2026-07-30).** For
     our *own* model's GQA path this is fixed and SDPA is now enabled
     (`mythouro/main.py`, commit b83d1dc). Re-probed every shape the model
     actually takes on a Max 1100 / i915 LTS 2523.x — including the exact
     `enable_gqa=True` call with UNEXPANDED K/V (16 q-heads vs 4 kv-heads),
     a *different kernel* from the pre-expanded one and the likely original
     culprit. No segfault, no exception, max rel err 0.0040 (bf16 tolerance):
     `enable_gqa B8 T1024` 0.72 ms vs 10.07 manual (**14.0x**);
     `B32 T80` 0.27 vs 2.16 (**8.0x**); decode shapes 7.8–14.3x.
     Reproduce with `tools/sdpa_probe.py` (one case per process, so a segfault
     is distinguishable from a wrong answer by exit code).
   - **MLA still uses manual — for SPEED, not safety.** With the K/V head-dim
     mismatch (Dk=80, Dv=48) SDPA loses its fused path: 25.40 ms vs 9.08 manual
     at B8 T1024 = **0.36x, ~3x SLOWER**. Gated separately as `sdpa_ok_for_mla`.
   - **The end-to-end gain was ZERO.** 8–14x on the attention kernel moved step
     time by ~1% (see `docs/training_throughput.md`) — attention is a small
     slice next to 48 MoE experts run 4x per forward. Kept because it is free
     and numerically gated (0.00200 nats KL), not because it bought throughput.
   - **The HF-teacher half of this item still stands** — untested, leave `eager`.
   - **FlashAttention-2 is not available on this hardware, ever.** `flash_attn`
     is hand-written CUDA for sm80+; there is no SYCL/XPU build and it will not
     compile. `_fa2_usable()` requires a CUDA compute capability, which is
     `None` here. **torch SDPA on XPU IS the fused-attention path for this card**
     — Intel's kernel filling the same role. Not a consolation prize, and per the
     point above it would not have mattered anyway.
2. **`topk`/`multinomial` sampling segfaults** → sample on CPU (negligible cost
   at batch sizes that matter).
3. **Complex-dtype ops unsupported** → real-valued RoPE variant.
4. **Cosmetic abort at process exit** (`terminate called without an active
   exception`, exit 134) after successful completion — harmless, checkpoint is
   already saved; don't chase it.
5. Env vars that should be in your profile: `SYCL_CACHE_PERSISTENT=1` (kills
   the brutal first-JIT on every launch), `PYTORCH_ALLOC_CONF=expandable_segments:True`
   (fragmentation under variable batch shapes).
6. **After ANY reboot**: if `xpu-smi` suddenly sees nothing / "XPU device count
   is zero" — your `/dev/dri/renderD*` access was via a session ACL that died
   with the reboot. Durable fix: `sudo usermod -aG render,video <user>` + relogin.
7. **An OOM'd job may not actually die.** On CUDA, a crashed process = driver
   reclaims VRAM, clean slate. Here we've observed a Level-Zero OOM
   (`UR_RESULT_ERROR_OUT_OF_RESOURCES`, surfaced as generic `RuntimeError`, so
   typed-OOM catch blocks miss it) followed by the SYCL runtime **deadlocking in
   teardown** — the process hangs half-dead holding its full allocation, and the
   next job OOMs at model load on a "free" card. Before any relaunch after a
   crash: `pgrep` the old job, `kill -9` if present, verify with
   `xpu-smi dump -d 0 -m 18` that memory actually returned. (Same runtime
   teardown quirk as item 4, landing on the harmful side.)

## Thermals (the passive-card tax)

It's a passive datacenter card expecting server airflow. In a tower case, stock:
**96 °C core / 92 °C memory with throttling** under training load. With a single
40 mm counter-rotating fan zip-tied to the shroud: **68–70 °C sustained**, no
throttle, no power cap needed. (Stopgap if cooling regresses:
`xpu-smi config -d 0 --powerlimit 225` — note it resets on reboot.) Multi-card
builds should skip the hacks and use a front-to-back server chassis.

## Software stack status (why EOL isn't doom)

Intel retired the Max GPU line and EOL'd IPEX — but **GPU support was upstreamed
into PyTorch core first**. `torch.xpu` ships in stock PyTorch wheels, and the whole
stack is open and self-patchable. **Correction (2026-07-24): the kernel driver is
NOT mainline/in-tree i915 for Ponte Vecchio** — the Max 1100/1550 need Intel's
**out-of-tree** i915 kernel module (`intel-i915-dkms`), Intel's own term; the
in-tree kernel i915 does not fully support PVC. That distinction is the whole
reason the driver was hard to find — see the note below. The card survives its
vendor's pivot. Multi-card: Xe Link bridging works on the
PCIe cards in pair topologies; standard DDP/FSDP backends exist for XPU
(untested by us so far — single card to date).

**⚠ The ONE genuinely hard install step — sourcing the driver (owner, 2026-07-24).**
The trap: PVC needs the **out-of-tree** i915 module (Intel's term), NOT the in-tree
mainline kernel driver that newer Intel GPUs use — so a stock Ubuntu kernel does
*not* fully drive the card, and searching for "i915" leads you to the wrong (in-tree)
path. The working config is a **specific out-of-tree LTS release, `intel-i915-dkms`
2523.x**, and it was **brutal to find — ~1 week.** The in-tree-vs-out-of-tree map is
Intel's supported-hardware page:
<https://dgpu-docs.intel.com/overview/supported-hardware/i915-driver-gpus.html>
(PVC = "Out-of-tree" for both initial and full support). Load-bearing for the
*entire* XPU stack (no driver → no `torch.xpu`, no harvest, no training); applies
identically to the incoming 1550 (same PVC/Xe-HPC, same out-of-tree path) and the
planned multi-card host. **Do not lose this pointer.**
**Source (recorded 2026-07-24):** Intel's official guide —
<https://dgpu-docs.intel.com/installation-guides/max-and-flex/installation.html>.
The whole "week to find" collapses to **one token in the apt repo line: `lts/2523`.**
Supported: Ubuntu 22.04 (5.15) / 24.04 (our host runs 6.8 — works). Verbatim:
```bash
# 1. GPG key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
# 2. Repo — the load-bearing bit is the "lts/2523" component
. /etc/os-release
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2523 unified" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list
sudo apt update
# 3. Kernel driver (the i915 LTS DKMS) + firmware + xpu-smi
sudo apt install -y linux-headers-$(uname -r) linux-modules-extra-$(uname -r) \
  flex bison intel-fw-gpu intel-i915-dkms xpu-smi
# 4. Compute runtime + Level Zero (what torch.xpu binds to)
sudo apt install -y intel-opencl-icd libze-intel-gpu1 libze1
```
`intel-i915-dkms` from the `lts/2523` component **is** the "i915 LTS 2523.x" driver.
2523.x supersedes the older 2350.x. Applies identically to the 1550 (same PVC path).
Full list (dev headers, media libs) in the linked guide. Mirror: memory
`[[intel-i915-lts-driver]]`.

## Why third-party benchmarks underreport this card

Treat published PVC numbers as a floor, not an estimate. Benchmarking the Max
the way reviewers benchmark NVIDIA cards — cold start, narrow batch, default
knobs — systematically understates it. Our own receipts:

- **Warmup**: the identical GEMM measures 140 TFLOPS cold and **224 warmed**
  (+60%). Most published figures are the cold one.
- **Batch width**: real-model training swings **3.3k → 17.2k tok/s (5×)**
  from batch 1 to 64. Any batch-1 benchmark reports the card's worst case.
- **The "obvious" knob is a trap**: `torch.compile` default mode gains +10%,
  but `max-autotune` — the setting a reviewer reaches for — *loses* 13%
  (Triton GEMM templates displace oneDNN's XMX kernels). A tuned-looking
  config can be slower than eager.

None of this is exotic tuning — warm the kernels, go wide, leave oneDNN in
charge — but defaults don't do it for you, and neither do most reviews.

*Planned addition: standard-model numbers (Llama-class via HF transformers,
vLLM-XPU batch serving, llama.cpp SYCL) measured on this same rig, for
apples-to-apples comparison with published reviews.*

## Honest buying guidance (2026 gray market)

- ES/QS samples circulate cheaply; ours runs on stock everything. Ask sellers
  for `sycl-ls` / `xpu-smi discovery` output before paying.
- The line is EOL — if you plan a multi-card build, buy as a batch; supply of
  card #N later is a gamble.
- Know your workload: **wide training / batch inference with big memory = the
  card's home turf** (48 GB at this price has no NVIDIA answer). Batch-1
  latency-sensitive serving = buy something else.

---

*Data from the MythOuro project (this repo) — bench tables and methodology in
`docs/hardware_options.md`, day-by-day validation history in the git log.
Corrections and reproductions welcome.*

## ⚠ GPU page fault at `--rollout-len` 128 (2026-08-24) — TRANSIENT, DID NOT REPRODUCE

`tools/bench_rollout.py` completed rollout-len 64 cleanly, then aborted entering 128:

```
Segmentation fault from GPU at 0xff0000007d3ac000, ctx_id: 1 (CCS)
type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.
Abort was called at 288 line in ./shared/source/os_interface/linux/drm_neo.cpp
```

A driver-level page fault from the Intel compute runtime, not a Python exception.
Config: step_0149500, batch 32, prompt 64, n_loops 4, uncached, teacher=None.
Sequence length at the crash was 64+128 = 192, far under the model's 1024
`max_seq_len`, so this is NOT a context overflow. Card recovered on its own —
`xpu-smi` read 22.49 MiB immediately after, nothing stuck, no reboot needed.

**UPDATE, same day: it did not reproduce.** A 9-config sweep (len 64/128/256 x
batch 8/16/32, each in its own subprocess) completed every cell — including the
exact len 128 / batch 32 config that aborted. So this was a transient fault, not
a length or memory ceiling, and `bench_rollout` is usable at all these sizes.

Cause still unknown. Worth knowing it can happen, and worth running sweeps under
a per-config subprocess driver (`run_bench_rollout.sh`) so a recurrence costs one
cell instead of the whole run.

**It did not block the decision it was run for** — see the rollout-cost note below.

## Rollout generation is ~29% of every training leg (2026-08-24, measured)

`generate_rollout` is hard-pinned UNCACHED (2026-07-16: cached decode runs FULL
depth because both early-exit paths require `kv_cache is None`, while uncached
early-exits at the measured ACT depth of 2.00/4 — a 2x depth gap worth 0.95 nats
of KL). Uncached means generating n tokens re-runs the forward over the whole
prefix each step: **O(n²), not O(n)**.

Measured on step_0149500, batch 32, prompt 64, n_loops 4:

### MEASURED (9 configs, `run_bench_rollout.sh`), hours are per 6,000-step leg

| len \ batch | 8 | 16 | 32 |
|---|---|---|---|
| **64** | 3.56 s → **0.7 h** | 5.67 s → **1.2 h** | 10.01 s → **2.1 h** ← current recipe |
| **128** | 8.12 s → **1.7 h** | 13.51 s → **2.8 h** | 24.70 s → **5.1 h** |
| **256** | 26.96 s → **5.6 h** | 36.60 s → **7.6 h** | 69.31 s → **14.4 h** |

**⚠ SCALING IS SUB-QUADRATIC: ~7x at 256, not the 16x O(n²) predicts** (7.6x /
6.5x / 6.9x at batch 8/16/32). The card absorbs much of the longer prefix. An
earlier note here projected 9.8 h for 128 and 39 h for 256 from the complexity
argument and concluded "not viable" — **that was wrong by more than 2x, in the
favourable direction.** Measure rollout cost; do not derive it.

**⇒ Raising `--rollout-len` IS viable.** Against the current 2.1 h:
* len 128 / batch 16 → **2.8 h** — double the window for +0.7 h
* len 256 / batch 8 → **5.6 h** — quadruple it; a leg goes ~8.5 h → ~12 h

⚠ Lower `--rollout-batch` is cheaper partly by generating FEWER on-policy
sequences per regeneration (32 → 8 quarters them). That is a real trade, not free
throughput. Note also that tok/s RISES with batch (144 → 205 at len 64), so the
big batch is more efficient per token even though it costs more wall-clock.

⚠ 256 still does not span a ~495-token think block. Extrapolating the measured
curve, len 512 / batch 8 lands near 13 h of rollout time — borderline. A partial
extension may still be worth trying, but "cover think + answer" is not yet cheap.
