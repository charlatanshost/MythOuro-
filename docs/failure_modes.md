# Failure modes encountered + recovery patterns

> The debugging / lessons-learned reference: every failure mode hit and how it was fixed. Split out of `roadmap.md` (2026-06-27 doc reorg).


<!-- ===== moved from docs/roadmap.md (2026-06-27 doc reorg) ===== -->

## Failure modes encountered + recovery patterns

Institutional memory. If any of these come back in a future session,
the fix is already known — saves hours of re-debugging.

### Script defaults are NOT the proven recipe (2026-06-10)
- *Symptom*: both ablation arms (MoE AND dense) flatlined identically from
  scratch — hard CE stuck at ~7.6–8 after step ~20, eval PPL in the thousands
  (v1 was at 368 by step 1000), transient gradient-norm spikes (32→2743)
  concentrated on deep-loop batches, weights NOT diverged (max component
  growth 1.36×), data stream verified clean.
- *Root cause*: the runs used `training/distill.py`'s DEFAULTS (warmup 200,
  depth-reg 0.1, mb2/ga4) — but v1's "final successful run" used **warmup 500,
  depth-reg 0.3, mb1/ga8**, recorded only in the v1 MODEL_CARD provenance.
  Warmup 200 hits full 3e-4 LR at step 200 on a fresh 4-loop recurrent model →
  early instability → bad basin → flatline. The proven recipe lived in the
  model card, not the code.
- *Diagnosis chain that isolated it* (each step cheap): depth-sweep eval
  (killed "h_K eval artifact"), init-stats comparison (killed "dense FFN too
  hot"), checkpoint weight forensics (killed "runaway divergence"), 30-step
  resume probe (found the gnorm spikes + flatline), MoE control arm
  (reproduced the flatline → killed "dense arch"), data-stream decode (clean),
  v1 model-card command diff (the answer).
- *Fix*: `distill.py` defaults now ARE the proven recipe (warmup 500,
  depth-reg 0.3); ablation commands updated. **Rule: when reproducing a run,
  diff your command against the MODEL_CARD provenance command, never trust
  script defaults.**

### Code-review P0 fixes (2026-06-09)

External review (Fable 5), independently verified against the code and fixed —
all were **invariant violations the 303-test suite didn't catch** (it had no
zero-init / telemetry-coverage / emission-parity invariants; now added):

- **P0.1 — `_init_weights` clobbered deliberate zero-inits.** The blanket
  N(0,0.02) ran *after* submodules zero-init, so `CrossLoopAttention.o_proj`
  (identity residual) and `UncertaintyHead.net[-1]` (neutral 0.5) were random in
  all v1–v5 checkpoints. *Fix:* those layers mark `_skip_global_init=True`.
- **P0.2 — MoE router telemetry saw only the last loop.** The single recurrent
  MoEFFN is called n_loops×, but `_last_router_logits`/counts were overwritten
  each call → aux losses, the DeepSeek-V3 bias updater, and the cv/min%/max%
  metrics only constrained loop K−1. *Fix:* `_loop_body` returns the telemetry
  (checkpoint-safe, grad-tracked); `RecurrentBlock` accumulates across all loops.
- **P0.3 — eval emitted a never-trained path.** Training returns `h_K`; eval
  returned the ACT blend `h_out`, which under-summed for non-halting positions
  **and** was never seen by the coda/head in training. *Fix:* return `h_K` at
  inference too (ACT = early-exit criterion only). **Re-baseline (same v2
  weights, fixed code): PPL 46.3→39.25, ECE 0.058→0.042, loop_eff 0.500 unchanged
  — a free capability/calibration gain.** See
  [`reports/rebaseline_v2_after_p0_fixes.md`](../reports/rebaseline_v2_after_p0_fixes.md).
- **Also fixed:** `eval.harness` hardcoded `mythouro_1b` (couldn't load other
  checkpoints) and defaulted to a wrong-vocab tokenizer (gpt-neo, 50k) →
  out-of-range ids on the 49152-vocab models. Both corrected. *Open:* the eval
  metrics don't clamp inputs to `max_seq_len`.

**Caveats to existing conclusions:** the v3/v5 **cv numbers** (incl. open-Q#1's
"expert-count ceiling") were measured through P0.2 (last-loop-only), so treat
them as indicative, not exact. All archived eval numbers were measured through
P0.3's `h_out` path, so they were **pessimistic**. v1–v5 were *trained* under
P0.1+P0.2, so **a fresh run on the fixed code should improve further** — the
re-baseline only captures P0.3's emission gain on already-trained weights.

### Training / architecture

**ACT loop collapse — depth distribution pins to one bucket**
- *Symptom*: `depth` log metric drops to 0.000 sustained; inspector shows halt distribution `[1.0, 0, 0, 0]`. Loops are not used despite being configured.
- *Root cause*: ACT halt probabilities trained against the main task gradient. The optimizer learns to pin λ₀≈1.0 because the weighted-sum output `Σ w_t · h_t` lets it shortcut through one loop's output.
- *Fix*: **Return `h_K` (last loop's hidden state) during training instead of the ACT-weighted sum.** Inference still uses the weighted sum. See `RecurrentBlock.forward` in [mythouro/main.py](../mythouro/main.py); the `if self.training and kv_cache is None:` branch at the end.
- *Additional guard*: depth-reg coefficient on the PonderNet × Ouro KL-to-uniform regulariser. We use 0.1 as default in v3 onward.

**MoE expansion: new experts stay idle after sentinel decay**
- *Symptom*: post-promotion training shows `min% = 0.0` indefinitely; new experts never enter top-k.
- *Root cause*: sentinel decay completed but bias updater hasn't had time to nudge underused experts up.
- *Fix*: keep training. By step 2000+ post-promotion the DeepSeek-V3 updater pushes underused experts into the top-k pool. Verified in v3 training run.

**Loss spike at MoE promotion**
- *Symptom*: CE jumps by >0.5 immediately after grown checkpoint is loaded.
- *Root cause*: new experts' `down.weight` not zeroed, or sentinel bias not high enough to exclude from top-k.
- *Fix*: `_promote_state_dict` in [mythouro/grow.py](../mythouro/grow.py) zeroes the down projection AND sets `router_bias[new] = -100.0`. Tests assert this in `tests/test_grow.py`.

### Data pipeline

**HF streaming hangs silently for hours**
- *Symptom*: `load_dataset(..., streaming=True)` succeeds; first `next(iter(ds))` call hangs without timing out. No errors, no progress.
- *Root cause*: HF datasets' range-request iterator doesn't time out on stalled TCP connections. Happens on flaky / firewall-restricted home internet.
- *Fix*: use `streaming=False` (pre-download). Implemented in [mythouro/sft_data.py](../mythouro/sft_data.py) `_open_source`. Pre-download datasets once via `load_dataset(repo, split='train')` in a separate command.

**`apply_chat_template(..., tokenize=True)` returns `BatchEncoding`, not `list[int]`**
- *Symptom*: 100% of SFT samples rejected with `empty_or_shorter_response` reason. `len()` of returned object is 2 (number of BatchEncoding fields) rather than token count.
- *Root cause*: Ouro tokenizer returns a `BatchEncoding` wrapper; calling `len()` gives field count, not token count.
- *Fix*: render with `tokenize=False` to get text, then call `tokenizer.encode(text, add_special_tokens=False)`. See `_build_sft_example` in [mythouro/sft_data.py](../mythouro/sft_data.py).

**OpenHermes ~95% rejection rate at seq_len=512**
- *Symptom*: SFT diagnostic shows `general: 0/N (0.0% accept) [prompt_too_long=N]`. Training stalls waiting for valid samples.
- *Root cause*: OpenHermes-2.5 is mostly multi-turn — system + user + previous-assistant turns already exceed 512 tokens before the final assistant response can land in the loss-bearing region.
- *Fix*: either drop OpenHermes (v2 approach) OR bump to `seq_len=1024` (v4 approach). At 1024 acceptance jumps to ~60–70%.

**HF dataset `.shard(num_shards=N, index=i)` raises "list index out of range"**
- *Symptom*: streaming source crashes immediately on iterator open when `num_workers > num_shards`.
- *Root cause*: many instruction datasets ship as a single parquet file (`num_shards=1`); `.shard(num_shards=2, ...)` on those crashes inside the streaming library.
- *Fix*: only call `.shard()` when `total_shards > 1`. Use `num_workers=0` in DataLoader for single-process runs. See `MixedSFTDataset._open_source` in [mythouro/sft_data.py](../mythouro/sft_data.py).

**"Training is slow" — partly a misread (light runs), partly real (heavy runs)**
- *Symptom*: the per-step `tok/s` looks tiny (e.g. "11.4", "0.3"); seems to decline as training progresses; heavy runs take many hours.
- *The misread part (light SFT runs were fine)*:
  1. **The log prints `tok/s` with a `k` suffix** — "3.3k tok/s" is 3,300, not 3.3. The light SFT runs (278M, RTX 5070) ran a healthy **~2–3.3k tok/s**, verified against the on-disk logs.
  2. **The decline with steps is the `LoopCurriculum`** (`start_loops=2 → 4`): more loops = ~2× compute, so tok/s halving from 3.3k → ~1.7k across a run is *by design*.
  3. **Wall-clock conflates training with debugging** — restarts, the `0.0% accept / build_reject` BatchEncoding stalls above, and repeated 1.5M-example dataset reloads inflate the session time well beyond the clean training time.
- *The REAL part — the heavy MoE-grown runs (v3–v5) are genuinely slow.* Per-step
  time, **measured from checkpoint save-timestamps** (these runs have no cards/eval
  JSONs — reconstructed cards added 2026-06-08):

  | Run | Model | s/step | ~tok/s |
  |-----|-------|-------:|-------:|
  | v2 (light SFT) | 278M, 24 exp | ~4.4 | ~1,900 |
  | v3 | 420M, 48 exp | **~8** | ~1,500 |
  | v4 | 420M, fp32 Adam | **~13** | ~950 |
  | v5 | 632M, 96 exp | **~33** | **~370** |

  v5 ground **~8 h for ~900 steps**. The slowdown is **bigger model + 96-expert MoE
  dispatch + micro-batch 1 (forced by 12 GB)** — i.e. the heavy runs are
  **VRAM-gated**, dropping to the worst-case batch-1 utilisation regime.
- *The lever is hardware, on two independent axes*:
  - **More VRAM** → bigger batch → escape the micro-batch-1 strangulation that
    dominates the heavy runs. The 5070 at batch-2 sustains ~3k tok/s but
    `bench_step` at batch-8 hit **6,852** — ~½ the card is locked behind 12 GB.
  - **More dense-BF16 TFLOPS** → raises the compute ceiling (the 5070 is segmented
    to ~33.9). A faster card lifts the ~6,852 batch-8 ceiling itself.
  A card with *both* (4090 24 GB / 5090 32 GB) compounds them. Note this is a
  *new-card* purchase, not a free config win — the 5070 is already at its genuine
  limit for what fits 12 GB.

### Checkpoint / resume

**Grown checkpoint refuses to load — `KeyError: 'param_groups'`**
- *Symptom*: `load_checkpoint` crashes when resuming from a `tools/grow_checkpoint.py`-produced file.
- *Root cause*: grown checkpoints contain an empty `optimizer` dict (the source optimizer's tensor shapes don't match the promoted model). `optimizer.load_state_dict({})` blows up because `param_groups` is missing.
- *Fix*: `load_checkpoint` detects empty optimizer state and skips the load — caller's fresh optimizer is kept. See [mythouro/checkpointing.py](../mythouro/checkpointing.py).

**Tools script can't find `mythouro` module**
- *Symptom*: `python tools/grow_checkpoint.py ...` fails with `ModuleNotFoundError: No module named 'mythouro'`.
- *Root cause*: Python adds the script's directory to sys.path, not the cwd. From `tools/`, the project root isn't on the path.
- *Fix*: `sys.path.insert(0, project_root)` at top of `tools/grow_checkpoint.py`. Already implemented.

### Environment

**Pytest collection segfaults on Python 3.14 + Windows + pandas**
- *Symptom*: `python -m pytest tests/` segfaults during collection, before any test runs.
- *Root cause*: importing pandas (transitively pulled in by `datasets`) at top-level in a heavy training module triggers a crash on Py3.14+Windows.
- *Fix*: split checkpointing helpers into their own lightweight module so tests don't transitively import the training script's heavy deps. See [mythouro/checkpointing.py](../mythouro/checkpointing.py) module docstring.

**bitsandbytes "Configured CUDA binary not found at libbitsandbytes_cuda132.dll"**
- *Symptom*: `--use-8bit-adam` crashes at import — bnb looks for a `cuda132` binary that doesn't exist, even on the latest bnb (0.49.2).
- *Root cause*: torch is built for CUDA 13.2 (`cu132`), but bnb's prebuilt wheels only bundle binaries up to `cuda130`. There's no exact 13.2 binary in any release.
- *Fix*: **RESOLVED.** CUDA binaries are forward-compatible within a major version, so the `cuda130` binary runs fine against the 13.2 runtime. `training/sft.py` now calls `_configure_bnb_cuda_version()` before importing bnb — it scans the bundled `libbitsandbytes_cudaXXX` binaries, picks the highest ≤ the torch CUDA version, and sets `BNB_CUDA_VERSION` automatically. Verified: `AdamW8bit` constructs and runs on CUDA. No source build, no manual env var needed.
- *Manual override*: set `BNB_CUDA_VERSION` yourself before launching to force a specific binary; the helper respects an existing value.

**bitsandbytes not installed at all**
- *Symptom*: ImportError when `--use-8bit-adam` is passed and bnb isn't installed.
- *Fix*: import inside the `if args.use_8bit_adam:` branch with a clear `pip install bitsandbytes` error message. See [training/sft.py](../training/sft.py).

### Numerics / autocast

**`F.binary_cross_entropy` raises in bf16 autocast**
- *Symptom*: `RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast` during distillation.
- *Root cause*: BCE is on PyTorch's autocast-banned list because of fp16/bf16 stability issues.
- *Fix*: compute BCE manually as `-(target * log(p) + (1 - target) * log(1 - p))` and cast inputs to fp32. See `uncertainty_calibration_loss` in [mythouro/training_utils.py](../mythouro/training_utils.py).

**Cross-device tensor error during distillation**
- *Symptom*: `RuntimeError: Expected all tensors to be on the same device` when teacher is on cuda:0 and student is on cuda:2.
- *Root cause*: `teacher_logits()` was being passed input_ids on the student's device.
- *Fix*: `teacher_logits` internally moves input_ids to the teacher's device, and returns logits that the caller `.to(student_device)`. See [mythouro/training_utils.py](../mythouro/training_utils.py).

---

