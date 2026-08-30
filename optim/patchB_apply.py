#!/usr/bin/env python3
"""
OPTIMISATION B — precomputed top-K teacher logits for the OFFLINE path.

Stacks on top of OPT-A. A removes the teacher forward from on-policy
micro-steps (cached across rollout reuse); B removes it from offline
micro-steps (read from disk). Together the measured 78.4% teacher share
largely disappears.

MUST BE APPLIED AFTER patchA_apply.py.

⚠ B IS AN APPROXIMATION AND A IS NOT. A is numerically equivalent; B trains
against a coarsened teacher (top-K + one lumped tail bucket), which by the
data-processing inequality is a LOWER bound on the true KL. That is exactly why
these ship as two separate legs with an A/B between them — see README.md.

⚠ B ALSO FIXES THE OFFLINE CORPUS. Streaming is infinite; a cache is finite.
`TeacherLogitCache.epochs_for()` reports the re-read dose and the trainer logs
it at startup. Past ~1.35 epochs you are in the territory that cost 6.2pp in the
chat-mix post-mortem.
"""
import shutil, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

# ---- new modules -----------------------------------------------------------
for src, dst in (("sparse_kd.py", "mythouro/sparse_kd.py"),
                 ("logit_cache.py", "mythouro/logit_cache.py"),
                 ("precompute_teacher_logits.py",
                  "tools/precompute_teacher_logits.py")):
    (REPO / dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / src, REPO / dst)
    print(f"  installed {dst}")

p = REPO / "training/distill.py"
s = p.read_text()
if not pathlib.Path(str(p) + ".preB").exists():
    shutil.copy(p, str(p) + ".preB")   # pristine-only, see patchA_apply.py

def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"FAIL distill.py: expected {count} of\n{old[:150]!r}\ngot {n}")
    s = s.replace(old, new)

# 1. imports
sub("""from mythouro.grow import apply_sentinel_to_router_biases""",
"""from mythouro.grow import apply_sentinel_to_router_biases
# OPT-B: precomputed top-K teacher logits for the offline path. See
# tools/precompute_teacher_logits.py and docs/teacher_logit_cache.md.
from mythouro.sparse_kd import sparse_distillation_loss
from mythouro.logit_cache import TeacherLogitCache""")

# 2. CLI flag
sub("""    p.add_argument("--use-depth-aware-init", action="store_true",""",
"""    p.add_argument("--teacher-logit-cache", type=str, default=None,
                   help="OPT-B. Directory of precomputed top-K teacher logits "
                        "(tools/precompute_teacher_logits.py). When set, OFFLINE "
                        "micro-steps read the teacher's answer from disk instead "
                        "of running the 2.6B forward, which _StepProfiler "
                        "measures at 78.4% of a step. The cache supplies the "
                        "TOKENS too, so it replaces the corpus stream for those "
                        "steps. APPROXIMATE: the divergence is computed on a "
                        "top-K + lumped-tail coarsening, a lower bound on the "
                        "true KL. Must be built at the same --temperature.")
    p.add_argument("--use-depth-aware-init", action="store_true",""")

# 3. build the cache
sub("""    # Rollout reuse buffer (docs/onpolicy_plan.md phase 5): decouple the wide""",
"""    # OPT-B: offline teacher-logit cache. Replaces both the corpus batch and
    # the teacher forward on offline micro-steps.
    logit_cache = None
    logit_cache_iter = None
    if args.teacher_logit_cache:
        logit_cache = TeacherLogitCache(
            args.teacher_logit_cache,
            seq_len=args.seq_len,
            temperature=args.temperature,
            batch_size=args.micro_batch,
            device=str(device),
        )
        logit_cache_iter = iter(logit_cache)
        # Dose check, logged where it will be seen. The cache is FINITE; the
        # stream it replaces was not. 1.35 epochs cost 6.2pp of code L3+ in the
        # chat-mix post-mortem, 10.3 epochs cost 25pp and did not recover.
        off = max(0.0, 1.0 - args.onpolicy_lambda)
        ep = logit_cache.epochs_for(
            steps=max(1, args.total_steps - start_step),
            micro_per_step=max(1, int(round(args.grad_accum * off))),
        )
        msg = (f"distill: OPT-B logit cache — {len(logit_cache):,} rows, "
               f"~{ep:.2f} epochs over this leg")
        (logger.warning if ep > 1.35 else logger.info)(msg)
        if ep > 1.35:
            logger.warning(
                "distill: OPT-B cache will be RE-READ past the 1.35-epoch dose "
                "that measurably cost capability. Enlarge the cache."
            )

    # Rollout reuse buffer (docs/onpolicy_plan.md phase 5): decouple the wide""")

# 4. per-micro-step state
sub("""            # OPT-A: set by the rollout buffer when it carries teacher logits
            # for this micro-step; None means "run the teacher yourself".
            cached_t_logits = None""",
"""            # OPT-A: set by the rollout buffer when it carries teacher logits
            # for this micro-step; None means "run the teacher yourself".
            cached_t_logits = None
            # OPT-B: (topk_idx, topk_val, tail_lse) when this offline micro-step
            # is served from the precomputed cache; None means dense teacher.
            sparse_pack = None""")

# 5. offline path reads from the cache
sub("""            else:
                x_in, y_in = x, y
                distill_targets = y""",
"""            else:
                if logit_cache_iter is not None:
                    # OPT-B: the cache carries its own tokens. Taking them from
                    # here (rather than re-deriving them) is what makes
                    # misalignment structurally impossible.
                    x_in, y_in, _c_idx, _c_val, _c_tail = next(logit_cache_iter)
                    sparse_pack = (_c_idx, _c_val, _c_tail)
                else:
                    x_in, y_in = x, y
                distill_targets = y_in""")

# 6. skip the teacher entirely when the cache serves this step
sub("""                with prof.region("teacher_fwd"):
                    if cached_t_logits is not None:
                        # OPT-A hit: identical x_in was already sent to the
                        # teacher when this rollout was generated.
                        t_logits = cached_t_logits
                    else:
                        t_logits = teacher_logits(teacher_fwd, x_in).to(device)""",
"""                with prof.region("teacher_fwd"):
                    if sparse_pack is not None:
                        # OPT-B hit: the teacher answered this offline batch
                        # once, offline. Never run the 2.6B forward here.
                        t_logits = None
                    elif cached_t_logits is not None:
                        # OPT-A hit: identical x_in was already sent to the
                        # teacher when this rollout was generated.
                        t_logits = cached_t_logits
                    else:
                        t_logits = teacher_logits(teacher_fwd, x_in).to(device)""")

# 7. dispatch helper + call sites
sub("""class _StepProfiler:""",
'''def _kd(sparse_pack, student_logits, teacher_logits_dense, **kw):
    """OPT-B dispatch: cached sparse teacher, else the dense teacher forward.

    Routed through one helper so a call site cannot be missed — a site left on
    the dense path would dereference `t_logits=None` and crash immediately
    rather than train against a wrong target, but only if it is reached.
    """
    if sparse_pack is not None:
        return sparse_distillation_loss(student_logits, *sparse_pack, **kw)
    return distillation_loss(student_logits, teacher_logits_dense, **kw)


class _StepProfiler:''')

# Three call sites, each `X = distillation_loss(\n<indent>ARG, t_logits, ...`.
# Rewrite them by exact text so an indentation change fails loudly here rather
# than silently leaving a site on the dense path.
for old, new_ in (
    ("""                    distill_total, distill_metrics = distillation_loss(
                        s_logits, t_logits,""",
     """                    distill_total, distill_metrics = _kd(
                        sparse_pack, s_logits, t_logits,"""),
    ("""                            loss_k, metrics_k = distillation_loss(
                                logits_k, t_logits,""",
     """                            loss_k, metrics_k = _kd(
                                sparse_pack, logits_k, t_logits,"""),
    ("""                        distill_total, distill_metrics = distillation_loss(
                            s_logits, t_logits,""",
     """                        distill_total, distill_metrics = _kd(
                            sparse_pack, s_logits, t_logits,"""),
):
    sub(old, new_)

remaining = [ln for ln in s.splitlines()
             if "distillation_loss(" in ln
             and "def " not in ln and "sparse_" not in ln
             and "return distillation_loss" not in ln]
if remaining:
    raise SystemExit("FAIL: un-dispatched distillation_loss call sites:\n  "
                     + "\n  ".join(remaining))

p.write_text(s)
print("  patched training/distill.py")
print("  OPT-B applied.")
