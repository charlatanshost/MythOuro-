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
# THIS RUN STOPS GUESSING. Config is back to the real one — we WANT the crash —
# with PYTHONFAULTHANDLER=1 so SIGSEGV dumps a Python traceback, and
# ZE_SERIALIZE=2 so async kernel launches do not misattribute the frame.
# Whatever line it names is the answer.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

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
