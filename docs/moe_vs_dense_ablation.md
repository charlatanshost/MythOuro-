# MoE-vs-dense ablation

> The pre-registered MoE-vs-dense gating experiment — protocol, matched-compute configs, results, decision rule. Split out of `roadmap.md` (2026-06-27 doc reorg).


<!-- ===== moved from docs/roadmap.md (2026-06-27 doc reorg) ===== -->

## Gating experiment: MoE-vs-dense ablation

The single experiment that unblocks MoDr (and decides whether ~60% of the
architecture's complexity — router, `router_bias` updater, load-balance loss,
sparse-activation loss, expert-specialisation probe, growth machinery — earns its
keep). Spec'd here so it's ready to run when there's GPU time.

**Question.** At *matched compute* (same activated parameters / FLOPs per token),
does the MoE recurrent FFN beat a plain dense FFN on eval? If a dense model within
noise of MoE, sparsity isn't paying for itself at this scale and the routing
machinery is removable.

### SEED-0 RESULT (2026-06-11): MoE wins 4.0× — retained, pending seed 1

Both arms run (fixed code, proven recipe, seed 0, identical but the FFN):
dense final PPL **22.66** vs MoE **5.72** — and the gap GROWS with training
(1.0× @1k → 1.3× @2k → 3.0× @3k → 4.0× @4k), i.e. the sparse capacity is
progressively utilised. Per the pre-registered rule (>5–10% = keep MoE):
**MoE earns its complexity at this scale — decision pending seed-1
confirmation** (runs queued, user-gated). Consequences if seed 1 confirms:
MoDr proceeds as the full unified expert+depth router; the dense variant stays
as the ablation control. Nuance: dense (22.7) still beat v1's 37.4 — the code
fixes + recipe lifted both arms. Data: `docs/training_runs.md`.

### Matched-active (matched-compute) configs — the primary comparison

Per SwiGLU expert = `3·dim·expert_dim` params. For `distill_tiny`
(`dim=1280, expert_dim=1280, n_experts=24, n_shared=2, top-k=4`):

| | Active FFN / token | Total FFN params | Recurrent FFN |
|---|---:|---:|---|
| **MoE arm** (= `distill_tiny`) | ~59 M | ~157 M | `MoEFFN` (24 routed + 2 shared) |
| **Dense arm** (new) | ~59 M | ~59 M | one `Expert(1280, 15360)` |

Matched-active dense width: `d_ff = expert_dim · top_k · (1 + n_shared)
= 1280 · 4 · 3 = 15360`, so `3·dim·d_ff ≈ 59 M` = MoE's per-token active FFN.
Both arms do the **same FLOPs/token**; the MoE arm just carries ~98 M extra
*total* (sparsely-used) capacity. Whole-model totals: MoE ≈ 278 M, dense ≈ 180 M.

**Prerequisite code change — DONE (2026-06-08).** `MythOuroConfig` has
`recurrent_dense` + `recurrent_dense_ffn_dim`; `RecurrentBlock` builds a dense
`Expert(dim, d_ff)` recurrent FFN when set (auto width
`expert_dim·top_k·(1+n_shared)`); `mythouro_distill_tiny_dense()` is registered
in both training CLIs. Verified: MoE arm 278.9 M total / dense arm 180.5 M (98 M
idle capacity removed), dense recurrent FFN width 15360, and a test pins
`dense_FFN_params == MoE_active_FFN_params` exactly
(`tests/test_dense_ablation.py`). The MoE-only aux losses already no-op on a
dense model (they short-circuit to 0 when there are no MoE layers), so the dense
arm runs through the existing `distill.py` / `sft.py` unchanged.

**Run it (both arms, matched everything but the FFN):**
```powershell
# MoE arm (= the existing distill_tiny recipe)
python -m training.distill --trust-remote-code --teacher-device cuda:2 `
    --student-variant mythouro_distill_tiny `
    --warmup-steps 500 --depth-reg-coeff 0.3 --micro-batch 1 --grad-accum 8 `
    --start-loops 2 --random-depth --total-steps 4000 --seed 0 --eval --eval-every 1000 --ckpt-dir checkpoints_ablation_moe
# Dense arm (same flags, dense variant; MoE aux terms vanish to 0 automatically)
python -m training.distill --trust-remote-code --teacher-device cuda:2 `
    --student-variant mythouro_distill_tiny_dense `
    --warmup-steps 500 --depth-reg-coeff 0.3 --micro-batch 1 --grad-accum 8 `
    --start-loops 2 --random-depth --total-steps 4000 --seed 0 --eval --eval-every 1000 --ckpt-dir checkpoints_ablation_dense
```
Repeat each with `--seed 1` for the ≥2-seed requirement, then compare with the
eval harness / inspector.

**Teacher placement (measured 2026-06-09):** `--teacher-device cuda:0`
(cohabitation with the student on the 12 GB card) **OOMs at micro-batch 2** —
the v1 "fits in ~9 GB" sizing was theoretical; the real v1 run used two cards.
Put the teacher on a second GPU — **mind the index mapping, it is not
intuitive**: on this rig `cuda:0` = 5070, **`cuda:1` = 4060**, **`cuda:2` =
5060**. The documented role split wants the teacher on the **5060 → use
`cuda:2`** (the 4060 is reserved for parallel eval). Single-card fallback:
`--micro-batch 1`. Both 20-step GPU smokes (MoE + dense arms,
`reports/gpu_smoke_distill_*.txt`) ran clean two-card (teacher landed on the
4060 via cuda:1 — works, either 8 GB card fits the 5.2 GB teacher, but cuda:2
is the intended layout).

### Protocol

- **Identical everything except the recurrent FFN:** same `dim`, layers, loops,
  attn, vocab, tokenizer, **seed**, distillation corpus + teacher, total steps, LR
  schedule, warmup, seq_len, optimizer (8-bit AdamW), hardware.
- **Dense arm drops the MoE-only aux losses** (load-balance, sparse-activation,
  router-bias updater, expert-specialisation) — they don't apply. Keep CE +
  uncertainty-calibration + depth-reg identical across arms.
- **From-scratch distillation** at a matched step budget (~3–5 k steps, like v1) —
  cleaner than SFT-from-shared-base (no warm-start confound).
- Run **≥2 seeds per arm** — at 278 M / ~20–40 M tokens the variance is large
  enough that a one-seed gap can flip.

### Metrics (eval harness) + pre-registered decision rule

Primary: held-out **perplexity**. Secondary: ARC-Challenge, GSM8K,
loop_efficiency, ECE. Report alongside: tok/s, peak VRAM, total params, and
(MoE arm only) cv / min% / max%. Plus the 4-prompt inspector read.

Pre-registered rule (decide *before* seeing numbers, to avoid post-hoc
rationalising):
- **Dense within ~3% PPL of MoE** (and ARC/GSM8K within seed noise) → **drop MoE.**
  Remove routing machinery + growth path; MoDr degenerates to a depth-only router
  (simpler, still valuable). Biggest possible simplification.
- **MoE beats dense by a clear, cross-seed margin** (say >5–10% PPL) → **keep MoE.**
  Proceed to MoDr as the unified expert+depth router.
- **In between** → keep MoE provisionally, re-decide at scale-up.

### The caveat that matters most (read before acting on the result)

This ablation answers **"for the current 278 M / consumer-hardware regime,"** not
"forever." MoE's benefit typically *grows with scale and token budget*, and at
tiny scale experts are chronically under-trained — so a **dense win here does not
prove dense wins at 3 B.** Treat a dense result as "don't pay for MoE *yet*," and
**re-run the same ablation once on the scaled-up config** before locking the
architecture for the big run. A MoE win here is the stronger signal (sparsity
already paying off despite small scale).

**Effect on MoDr:** if MoE is dropped, MoDr isn't moot — it collapses to a
**depth-only `DepthRouter`** (the teacher/student depth policy below, minus the
expert-choice branch), which the forced-depth findings already justify on its own.
If MoE stays, MoDr is the full unified router.

---

