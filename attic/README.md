# attic — completed work, moved out of the way (2026-08-30)

Nothing here is deleted, and nothing here is broken. These are finished
experiments, superseded corpora and one-off scripts moved off the repo root so
the working set is visible. **Everything is one `mv` from being back.**

Restore anything with:

```bash
mv attic/checkpoints/<dir> .        # or attic/corpora/<dir>
git mv attic/scripts/<file> .       # scripts are tracked
```

## What is still at the root, and why

**Scripts (9)** — the active workflow only:

| script | why it stayed |
|---|---|
| `run_grown48.sh` | the current 397M leg |
| `run_grown48_broadmix.sh` | the broadened-corpus successor |
| `run_grown48_probe.sh` | diagnostic for the current model |
| `run_anneal_readout.sh` | prose eval — **called by `run_grown48.sh`'s tail** |
| `run_control_278m.sh` | referenced by `run_grown48_probe.sh` |
| `run_mathcode.sh`, `run_mathcode_read.sh` | the documented 278M fallback leg |
| `run_newmix_pour.sh` | the canonical pour, referenced by broadmix |
| `status.sh`, `stop_gpu_jobs.sh` | operating utilities |

**Checkpoints (8)** — every directory an active script resolves:
`checkpoints_base` (the 278M source of truth, holds `step_0157000.pt`),
`checkpoints_grown48` (the live 397M), `checkpoints_newmix`,
`checkpoints_mathcode`, `checkpoints_codemix`, `checkpoints_rollout256`,
`checkpoints_anneal040`, `checkpoints_anneal045`.

**Corpora (6)** — `data_teacher_{code,math,v2,med,chat,chat_clean}`. The first
two are what the live leg reads; `v2` + `med` are what `broadmix` adds.

`archived_models/` deliberately stayed at the root: it is already correctly
named and placed, and the roadmap and README link to it by that path.

## What moved

- **`scripts/` (40)** — one-off legs, probes, sweeps and A/Bs that have all
  reported. Their findings live in `docs/generation_probe_tracker.md` and
  `docs/roadmap.md`; the scripts are kept for reproducibility, not for running.
  Note some form chains (`run_leg_and_probe.sh` → `run_ab_confirm.sh`,
  `run_overnight_chain.sh` → `run_reuse8_ab.sh`) — they moved together, so the
  chains still resolve inside `attic/scripts/`.
- **`checkpoints/` (39, ~155 GB)** — dead experiment rotations. Includes the
  MoE-vs-dense ablation arms, the λ sweep, the SFT attempts, and the old
  `grown`/`grown_v4`/`grown_v5` lineage.
- **`corpora/` (3)** — `data_teacher` (retired v1, boilerplate-contaminated),
  `data_teacher_clean` (prose-skewed), and
  `data_teacher_chat_TRUNCATED_BAD` (the 1.22M-token harvest that passed every
  structural check and was still unusable — kept as the counter-example).
- **`misc/`** — `run_leg.log` (stale 2026-07-27), `example.py` and
  `try_mythos.py` (OpenMythos leftovers, zero references anywhere).

## Safety notes for whoever does this next

The move was made **while a training leg was running**. Before touching
anything, the trainer's open file descriptors were read from
`/proc/<pid>/fd` — it held only `data_teacher_code` and `data_teacher_math`,
both of which stayed. Every kept script was then re-checked for unresolved
`run_*.sh`, `checkpoints_*` and `data_teacher*` references. Do the same before
moving more: **`ls -l /proc/$(pgrep -f training.distill)/fd`**.

Moves were `mv` within one filesystem, so they are renames — instant, and no
data was copied or rewritten.
