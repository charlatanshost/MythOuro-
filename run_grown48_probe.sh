#!/usr/bin/env bash
# ISOLATE THE 48-EXPERT SEGFAULT — is it rollout memory? ~10 min, saves nothing.
#
#   bash run_grown48_probe.sh
#
# WHAT IS ALREADY RULED OUT (bench_step, 2026-08-28):
#   tiny  278M/24-expert, batch 1  seq 512    139 ms/step   OK
#   small 397M/48-expert, batch 1  seq 512    230 ms/step   OK
#   small 397M/48-expert, batch 8  seq 1024   727 ms/step   OK
# The 48-expert architecture, forward AND backward, at full training geometry,
# is fine. The crash needs the teacher or the rollout.
#
# The real run segfaults (core dumped) right after the first forward, twice,
# deterministically. Rollout is the suspect: batch 32, uncached O(n^2), 4 loops,
# on top of a frozen 2.6B teacher on the same 48GB card. The 278M line peaked
# ~31GB; the student side here is 1.42x.
#
# UPDATE 2026-08-28 09:45. rollout-batch 8 ALSO segfaulted, so width is not it.
# The 278M CONTROL (run_control_278m.sh) ran the identical path CLEAN — logged a
# step, printed the profile, exited normally. So the 429 rate-limiting was a red
# herring and the fault is specific to 48 experts.
#
# The control also pins the window: 278M took 62s from the first-forward warning
# to its first logged step. The 48-expert runs die inside that window — the first
# COMPLETE training step (forward + teacher + backward + optimizer).
#
# Everything else at 48 experts passes: bench_step fwd+bwd at batch 8 seq 1024
# (727 ms/step), forward with router_bias[24:]=-100, generate_rollout uncached
# with ACT live, and teacher-resident rollout at alpha=0.45 (peak 21.4 GB of 48).
# NONE of those ran backward WITH the teacher resident AND an optimizer.
#
# UPDATE 2026-08-28 11:10. micro-batch 4 ALSO segfaulted, so it is not memory
# geometry. And tools/repro_step.py now runs a COMPLETE step at 48 experts —
# optimizer, teacher logits, rev_kl loss, backward, optimizer step, router-bias
# update, with the trainer's exact cfg mutations (max_seq_len 1024, sandwich
# norm, depth-aware init) and the sentinel applied — and it PASSES at 21.75 GB.
#
# So the model, the step and the config are all clean. What repro_step does NOT
# exercise is the trainer's scaffolding: the rollout BUFFER (mythouro/rollout.py
# — refill, reuse, accumulator), the extra heads (prm_head, esp_probe) and their
# optimizer groups, the grad-accum loop, and MixedDataset.
#
# THIS RUN sets --onpolicy-lambda 0, which removes rollout generation entirely:
# no buffer, no refill, no reuse. Pure offline distillation.
# UPDATE 2026-08-28 (late morning). Bisection has cleared: the 48-expert model
# fwd+bwd, sentinel routing, ACT generation, rollout width, memory (peak 21.75 GB
# of 48), a COMPLETE training step with the trainer's exact cfg mutations, and
# the environment (the 278M control runs this path clean). Six predictions about
# the location, six wrong.
#
# UPDATE (12:00). generate_rollout now PASSES standalone with EVERY trainer
# parameter matched — real promoted checkpoint, real teacher, autocast,
# seed (32,16), alpha 0.45, n_loops 4, max_new 64, T=1.0, top_k=50, uncached.
# Peak 8.46 GB. So the rollout call itself is exonerated.
#
# The faulthandler dump showed a LIVE HTTP THREAD during the fault:
#   urllib3 -> requests -> huggingface_hub -> datasets.read_with_retries
# and rollout.py's docstring calls this fault class "shape/timing-dependent",
# i.e. a RACE. That fits: intermittent at 24 experts (2026-07-12, one-off),
# deterministic at 48 where the timing differs.
#
# --teacher-data-ratio 1.0 draws ALL data from the local JSONL shards, so the HF
# base stream never runs. If the crash disappears, the race is confirmed and the
# fix is to cache the base corpus locally instead of streaming it — which also
# cures the 429 rate-limiting seen all morning.
# ⚠ ratio 1.0 is a DIAGNOSTIC, not a recipe: it removes the 80% base stream that
# the training mix depends on.
#
# UPDATE (12:20). --teacher-data-ratio 1.0 ALSO segfaulted, with ZERO http
# frames in the dump and an IDENTICAL stack. The HuggingFace streaming race is
# dead. So is train-vs-eval mode: standalone rollout on the REAL promoted
# checkpoint passes in BOTH modes.
#
# Standalone generate_rollout now passes with every trainer argument matched —
# real weights, real teacher, autocast, seed (32,16), alpha 0.45, n_loops 4,
# T=1.0, top_k=50, uncached, train or eval. The crash is therefore in the
# SCAFFOLDING around the call, not the call.
#
# --rollout-legacy removes exactly two things: the RolloutBuffer, and the
# seed-accumulation loop that calls next(data_iter) three more times mid-step to
# reach rollout_batch rows. That interleaving of data iteration with GPU work is
# the last structural difference from the passing standalone.
#   clean    -> the buffer/accumulation path is the fault, AND --rollout-legacy
#               is a usable workaround (costs the phase-5 throughput win).
#   crashes  -> the fault is in generate_rollout under trainer conditions that
#               the standalone still does not reproduce; next is bisecting the
#               trainer itself rather than rebuilding it from outside.
#
# UPDATE (14:00). The minimal repro turned out to be INVALID as a control: at
# 24 experts it also page-faults on iteration 2, because a hand-rolled full-vocab
# fp32 KL pushed it to 35.85 GB. Swapping in the trainer's own distillation_loss
# did not help — the repro is simply heavier than the trainer (~31 GB real peak).
#
# BUT the failure signature is the same one seen all day, including bench_rollout
# on 2026-08-24:
#   "Segmentation fault from GPU ... type: 0 (NotPresent), level: 1 (PDE),
#    access: 1 (Write), banned: 1"
# That is a write to an unmapped page. ⇒ THIS CARD PAGE-FAULTS INSTEAD OF RAISING
# A CLEAN OOM. Every "segfault" today is consistent with memory exhaustion.
#
# Memory was dismissed earlier on a forward-only measurement (21.4 GB). A COMPLETE
# step at batch 8 is 36.32 GB — only ~12 GB of headroom, and 48 experts costs just
# +0.47 GB over 24, i.e. it sits right at the margin.
#
# THE UNTESTED COMBINATION: both knobs reduced TOGETHER. The rollout peak and the
# training-forward peak are separate; cutting one leaves the other at full size.
# micro-batch 8->4 alone segfaulted. rollout-batch 32->8 alone segfaulted.
# This run does BOTH.
#
# UPDATE (15:35). micro-batch 4 + rollout-batch 8 together ALSO segfaulted, at
# distill.py:910 (the checkpointed student forward). Cutting the rollout peak
# moved the crash to the next peak, which is what a memory ceiling looks like —
# but every reduction so far has just relocated it.
#
# The one structural element in the stack that has never been removed is
# torch.utils.checkpoint at main.py:1794. --no-gradient-checkpointing (added
# today) drops it. ⚠ This RAISES activation memory — the recurrent loop goes from
# O(1) to O(n_loops)=4 — hence micro-batch 4.
#   clean   -> the fault is in the checkpoint path, and this is a workaround at a
#              memory cost we can afford (36 GB peak, ~12 GB spare).
#   crashes -> the last structural difference is gone and what remains is the MoE
#              forward itself under sustained load, which is an Intel-stack issue
#              to report rather than something to configure around.
#
# THIS RUN STOPS GUESSING. Config is back to the real one — we WANT the crash —
# with PYTHONFAULTHANDLER=1 so SIGSEGV dumps a Python traceback, and
# ZE_SERIALIZE=2 so async kernel launches do not misattribute the frame.
# Whatever line it names is the answer.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
# ── THE ONE VARIABLE NEVER TESTED (found 2026-08-28 16:00) ──────────────────
# Every failing run exported PYTORCH_ALLOC_CONF=expandable_segments:True; every
# passing standalone repro ran WITHOUT it (fresh shells / bare activate). The
# fault signature — write to a NotPresent PDE — is precisely what a kernel
# writing into a grown-but-not-yet-mapped allocator segment produces. Expandable
# segments is the component whose job is mapping those pages.
# This run drops ONLY that setting. Everything else identical to the last crash.
# ── LAST ENV VARIABLE STANDING (16:35) ───────────────────────────────────────
# expandable_segments removed -> still crashed. The only env difference left
# between every failing and every passing run is SYCL_CACHE_PERSISTENT=1 — the
# on-disk JIT kernel cache at ~/.cache/libsycl_cache (2.2 GB, 219 files, oldest
# entry dated 2026-07-12: the SAME DAY as rollout.py's documented one-off abort).
# Standalone repros compile fresh in memory; the trainer loads cached binaries.
# A stale or truncated cached kernel scribbling = write to a NotPresent page.
# This run disables the cache. The cache dir itself is NOT touched.
# ── ROOT CAUSE FOUND (16:54): the persistent SYCL kernel cache ──────────────
# With SYCL_CACHE_PERSISTENT unset, the exact config that crashed five times ran
# 3 clean profiled steps. The cache (~/.cache/libsycl_cache, oldest entry
# 2026-07-12 = the day of the original one-off abort) held a corrupt/stale
# kernel binary that only the 48-expert compile path loaded.
#
# THIS RUN is the verification pass: PRODUCTION flags restored (micro-batch 8,
# grad-accum 2, rollout-batch 32, gradient checkpointing ON — none of those were
# ever the problem), cache RE-ENABLED against a freshly quarantined cache dir.
#   clean  -> the cache FEATURE is fine, the old CONTENTS were bad. Keep
#             persistence (it saves ~minutes of JIT per launch) and run the leg.
#   crash  -> the feature itself mis-keys these kernels; run permanently with
#             the cache off and report to Intel.
export SYCL_CACHE_PERSISTENT=1 TRITON_DEFAULT_BACKEND=intel

# ── MAKE THE SEGFAULT NAME ITSELF ────────────────────────────────────────────
# Every crash so far printed only "Segmentation fault (core dumped)" with no
# stack, which is why six hypotheses had to be eliminated by bisection instead.
# faulthandler installs a SIGSEGV handler that dumps the Python traceback of
# every thread at the moment of the fault — it turns "somewhere in the trainer"
# into a file and line number.
export PYTHONFAULTHANDLER=1
# Unbuffered + line-buffered so nothing is lost when the process dies hard.
export PYTHONUNBUFFERED=1
# Serialise SYCL kernel launches. Async launches mean the reported Python frame
# is usually NOT the failing kernel — this makes the traceback point at the real
# call site, at a throughput cost that does not matter for a 3-step probe.
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2
mkdir -p logs
LOG="logs/grown48_probe_$(date +%Y%m%d_%H%M).log"

python -u -m training.distill \
  --student-variant mythouro_distill_small \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 32 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 \
  --teacher-data-files 'data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl' \
  --onpolicy-lambda 0.7 \
  --ckpt-dir checkpoints_grown48 \
  --num-workers 0 --trust-remote-code \
  --profile-warmup 2 --profile-steps 3 \
  --total-steps 6000 \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2) || true

echo
echo "  survived => rollout width was the problem; rerun the leg with a smaller"
echo "              --rollout-batch. segfaulted => try --onpolicy-lambda 0 next."
