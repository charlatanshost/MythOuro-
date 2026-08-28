# DRAFT — SYCL persistent cache serves stale device binaries across a compute-runtime upgrade; result is a GPU page fault, not a recompile

**Where to file:** `intel/llvm` (SYCL runtime owns `SYCL_CACHE_PERSISTENT`), cross-ref
`pytorch/pytorch` (label `module: xpu`). This is a runtime robustness issue, not a
Ponte Vecchio issue — the persistent cache is shared by every SYCL device.
**Status: DRAFT — finalize after the fresh-cache verification pass.**

## Summary

With `SYCL_CACHE_PERSISTENT=1` and a cache populated **before** a same-day
compute-runtime/driver upgrade, a PyTorch XPU training workload deterministically
dies with a GPU page fault on its first forward pass:

```
Segmentation fault from GPU at 0xff00000586c80000, ctx_id: 1 (CCS)
type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.
```

Unsetting `SYCL_CACHE_PERSISTENT` — changing nothing else — makes the identical
workload run correctly. Expected behavior: the cache detects the runtime mismatch
(or binary incompatibility) and falls back to JIT recompilation; observed
behavior: an incompatible cached binary is loaded and executed, corrupting GPU
memory.

## Environment

| component | version |
|---|---|
| torch | 2.13.0+xpu (native `torch.xpu`, no IPEX) |
| GPU | Intel Data Center GPU Max 1100 |
| level-zero driver | 1.6.33578+77 |
| compute runtime | intel-opencl-icd 25.18.33578.77-1146~24.04 |
| i915 DKMS | intel-i915-dkms 1.25.2.57.250224.65+i75-1 |
| kernel | 6.8.0-138-generic (Ubuntu 24.04) |

## The timeline that produced the stale cache (all one day, 2026-07-12)

```
14:22–16:11  64 cache entries compiled & written to ~/.cache/libsycl_cache
             (under the runtime installed at the time)
16:23        intel-i915-dkms 1.25.2.57 installed        (dpkg.log)
17:45        intel-opencl-icd 25.18.33578.77 installed  (dpkg.log)
```

Every subsequent process with `SYCL_CACHE_PERSISTENT=1` loaded those pre-upgrade
binaries. Cache keys evidently did not incorporate whatever changed.

## Why it stayed hidden for six weeks, then became deterministic

The original workload (a 278M-parameter MoE with 24 experts) ran for six weeks on
the stale binaries with a single unexplained abort (recorded 2026-07-12 — the day
itself). Promoting the model to **48 experts** (same kernel *sources*, therefore
the **same cache keys** — SYCL kernels are keyed by source/build options, not
tensor shapes) changed launch geometry enough that an incompatible binary faulted
on every run. The cache contains **no entries** from the 48-expert runs: every
kernel was a stale hit.

## Evidence

1. **One-variable flip:** 5 consecutive crashes with the cache enabled; 3 clean
   profiled training steps immediately after `unset SYCL_CACHE_PERSISTENT`.
   Same binary, model, weights, data, and all other env.
2. **Standalone-vs-in-situ split:** every standalone reproduction (fresh shells
   without the env var) passed all along, including a complete training step with
   identical arguments — the only difference was cache participation.
3. **Not memory:** peak 36.3 GB of 48; the 48-expert model costs +0.47 GB over
   the 24-expert one at identical batch.
4. **Structurally intact cache:** 73 entries, no orphaned `0.src`, no truncated
   `0.bin` — the incompatibility is content/keying-level, not a torn write.
5. The full quarantined cache directory is preserved and can be provided:
   `~/.cache/libsycl_cache.quarantined_20260828` (2.2 GB, 219 files).

## Reproduction (as observed; minimal repro available on request)

1. Populate the persistent cache under runtime A.
2. Upgrade compute runtime/driver to B without clearing `~/.cache/libsycl_cache`.
3. Run a workload whose kernels hit the cached entries under launch conditions
   differing from those the binaries were compiled for.
4. GPU page fault (`NotPresent`, PDE, Write) at first execution of a stale kernel.

## Ask

Persistent-cache keys should incorporate the compute-runtime/driver identity (or
the loader should validate binary compatibility) so a mismatch triggers
recompilation rather than execution of an incompatible binary.
