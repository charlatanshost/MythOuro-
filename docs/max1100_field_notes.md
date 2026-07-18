# Intel Data Center GPU Max 1100 — field notes & benchmarks

Real-world numbers from daily LLM training on a single Max 1100 (PVC, 48 GB HBM2e,
PCIe dual-slot, 300 W), measured 2026-07-12 → 07-18. Card is a **gray-market
engineering sample** running on **stock upstream drivers** — no vendor hand-holding
involved, which is rather the point: everything here is reproducible by anyone with
the card and a Linux box.

**Rig**: native Ubuntu (kernel 6.8, i915), `torch 2.13.0+xpu` from
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

If your workload is single-stream chat inference, this is not your card. If
it's training or wide batch serving, carry on.

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

## Thermals (the passive-card tax)

It's a passive datacenter card expecting server airflow. In a tower case, stock:
**96 °C core / 92 °C memory with throttling** under training load. With a single
40 mm counter-rotating fan zip-tied to the shroud: **68–70 °C sustained**, no
throttle, no power cap needed. (Stopgap if cooling regresses:
`xpu-smi config -d 0 --powerlimit 225` — note it resets on reboot.) Multi-card
builds should skip the hacks and use a front-to-back server chassis.

## Software stack status (why EOL isn't doom)

Intel retired the Max GPU line and EOL'd IPEX — but **GPU support was upstreamed
into PyTorch core first**. `torch.xpu` ships in stock PyTorch wheels, the kernel
driver is mainline i915, and the whole stack is open and self-patchable. The
card survives its vendor's pivot. Multi-card: Xe Link bridging works on the
PCIe cards in pair topologies; standard DDP/FSDP backends exist for XPU
(untested by us so far — single card to date).

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
