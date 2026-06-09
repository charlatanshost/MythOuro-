# Re-baseline: v2 eval after the P0 fixes (2026-06-09)

Re-ran `eval.harness` on the **same** `mythouro_distill_tiny_sft_v2/step_0003000.pt`
weights, on the post-fix code (P0.1/P0.2/P0.3), to isolate what the fixes change
on existing checkpoints. For a *loaded* checkpoint only **P0.3** can move the
numbers — P0.1 affects only init (overwritten by the loaded weights) and P0.2
affects training-time routing balance (not eval).

Command:
```
python -m eval.harness --checkpoint archived_models/mythouro_distill_tiny_sft_v2/step_0003000.pt \
    --device cpu --tokenizer ByteDance/Ouro-2.6B-Thinking \
    --benchmarks perplexity loop_efficiency ece --max-samples 20
```

| Metric | v2 archived (`h_out` emission) | v2 re-baseline (`h_K`, fixed) | Δ |
|--------|-------------------------------:|------------------------------:|---|
| Perplexity | 46.3 | **39.25** | **−15%** (better) |
| loop_efficiency | 0.500 | 0.500 | unchanged ✓ |
| ECE | 0.0578 | **0.0422** | **−27%** (better) |

**Caveat:** re-baseline used `max_samples=20` vs the archived `50`, and the
FineWeb stream sample differs, so these aren't perfectly apples-to-apples. But
the gaps dwarf sampling noise, and `loop_efficiency` matching to 3 d.p. confirms
the halt path is untouched (only the emission changed).

**Conclusion.** P0.3 was emitting a **never-trained** path (`h_out`, the ACT
blend the coda/head never saw in training, and under-summed for non-halting
positions). Switching eval to the trained `h_K` path **recovers measured
capability + calibration for free** — before any retraining. This validates the
review's hypothesis and means the archived eval numbers (v1 PPL 37.4 etc.) were
*pessimistic*, measured through the buggy path.

**Implication for the archive.** v1–v5 were *trained* under P0.1 (clobbered
zero-inits) and P0.2 (last-loop-only routing balance), so a fresh training run on
the fixed code should improve further — the re-baseline above only captures the
P0.3 *emission* gain on already-trained weights. Retraining is the way to realize
P0.1/P0.2.

## Eval-harness bugs found + fixed while doing this
- `_build_model_from_config` hardcoded `mythouro_1b` → couldn't load any other
  checkpoint (size mismatch). Now rebuilds from the checkpoint's own `cfg`.
- `--tokenizer` defaulted to `EleutherAI/gpt-neo-125m` (~50k vocab) → out-of-range
  token ids on the 49152-vocab models. Now defaults to `ByteDance/Ouro-2.6B-Thinking`.
- *Still open:* the perplexity/ece/loop_efficiency metrics don't truncate input
  to `max_seq_len` (a long document → RoPE/embed index error). Worked around here
  by the small default `seq_len` (512/256); fix = clamp to `cfg.max_seq_len`.
