"""
Grow a checkpoint's recurrent DEPTH: expand the per-loop parameters 4 -> N.

WHY THIS IS NEEDED AT ALL. Raising `max_loop_iters` is nearly a config flip —
99.988% of the model is loop-SHARED and loads unchanged at any depth, which is
the whole point of a recurrent-depth transformer. But four tensors are sized BY
the loop count, and they hold the per-loop adaptation the model spent 70,500
steps learning:

    recurrent.lora.B                          (4, 8, 1280)
    recurrent.lora.scale.weight               (4, 8)
    recurrent.ms_inject.blend                 (4, 3)
    recurrent.injection.scheduler.log_scale   (4,)

41,008 elements, 0.012% of 341.9M. Bumping the config without expanding them
makes every one shape-mismatch.

CORRECTED 2026-08-10 — this previously claimed `load_state_dict(strict=False)`
"DROPS mismatched keys SILENTLY" and that the run would start with the
adaptation freshly reinitialised with nothing in the logs. **That is wrong, and
believing it would misjudge the risk of growing depth.** Verified on torch
2.13.0+xpu: `strict=False` suppresses MISSING and UNEXPECTED keys, but a SHAPE
mismatch still raises `RuntimeError: size mismatch for <key>`. On top of that,
`mythouro/checkpointing.py` calls `_assert_clean_state_load` after every load,
which raises on any unexpected key and on any missing key beyond the RoPE
buffers (P1, 2026-07-01). So a config bump without this tool fails LOUDLY at
load, twice over. Growing depth is still a deliberate act — the per-loop
adaptation is real and must be expanded on purpose — but the failure mode is a
crash, not silent corruption.

WHY DEPTH IS WORTH GROWING. The @70,500 --n-loops budget sweep was flat at
4/6/8 (L4 0% throughout, median rel_err 0.400/0.400/0.400) while n_loops=2 was
clearly worse (0.500) — the signature of ACT halting at ~4, i.e. extra budget
going unused. Depth cannot be added by ASKING at inference; it has to be
TRAINED IN. And the student is 278M against a 2.6B teacher, so if the recurrent
thesis holds (compute substitutes for parameters) it plausibly needs MORE passes
than the teacher's 4, not the same. Cost is ~13.5% throughput: at reuse=8 the
student forward + backward is only 663ms of a 4,900ms step.

INITIALISATION, per tensor — this is the only real decision:
  lora.B      -> ZERO. LoRA's B is zero-init by convention, so new loops start as
                 the bare recurrent block with no loop-specific delta: "run the
                 block again, unmodified". Copying loop 4's B instead would apply
                 its adaptation four more times, which can compound.
  everything  -> COPY the last trained slot, so the new loops continue the
    else       deepest learned behaviour rather than jumping to an untrained one.

⚠ The ACT halting head is calibrated at 4 loops (and the P0.5 audit already
flagged it as badly calibrated at loop 0). It must relearn when to stop in a
deeper regime, so expect the first stretch after growing to look unsettled.

Usage
-----
    python -m tools.grow_depth -c checkpoints_reuse8/step_0070500.pt \\
        --loops 8 -o checkpoints_depth8/step_0070500.pt --verify
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# name-substring -> policy for the NEW slots
_INIT_POLICY = {
    "lora.B": "zero",
}
_DEFAULT_POLICY = "copy_last"


def expand_tensor(t: torch.Tensor, new_n: int, policy: str) -> torch.Tensor:
    """Grow dim 0 from n to new_n under `policy`."""
    n = t.shape[0]
    if new_n <= n:
        return t
    extra_shape = (new_n - n,) + tuple(t.shape[1:])
    if policy == "zero":
        extra = torch.zeros(extra_shape, dtype=t.dtype, device=t.device)
    else:                                              # copy_last
        extra = t[-1:].expand(extra_shape).clone()
    return torch.cat([t, extra], dim=0)


def policy_for(name: str) -> str:
    for frag, pol in _INIT_POLICY.items():
        if frag in name:
            return pol
    return _DEFAULT_POLICY


def grow(state: dict, old_loops: int, new_loops: int) -> "tuple[dict, list]":
    """Expand every loop-sized tensor. Returns (new_state, report rows)."""
    out, rows = {}, []
    for k, v in state.items():
        if (isinstance(v, torch.Tensor) and v.ndim >= 1
                and v.shape[0] == old_loops
                and k.startswith("recurrent.")):
            pol = policy_for(k)
            nv = expand_tensor(v, new_loops, pol)
            rows.append((k, tuple(v.shape), tuple(nv.shape), pol))
            out[k] = nv
        else:
            out[k] = v
    return out, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--loops", type=int, required=True, help="new max_loop_iters")
    p.add_argument("--verify", action="store_true",
                   help="Load the grown checkpoint and assert that a forward at "
                        "the ORIGINAL depth is bit-identical to the original "
                        "model's. This is the safety property: growing must not "
                        "change existing behaviour, only add unused capacity.")
    p.add_argument("--drop-optimizer", action="store_true",
                   help="Drop optimizer state. Its moments for the grown tensors "
                        "have the old shape and cannot be resumed; the trainer "
                        "would either crash or silently reset them. Default is "
                        "to drop ONLY the mismatched entries (see below).")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg")
    old = getattr(cfg, "max_loop_iters", None)
    if old is None:
        raise SystemExit("checkpoint has no cfg.max_loop_iters")
    if args.loops <= old:
        raise SystemExit(f"--loops {args.loops} must exceed current {old}")
    print(f"growing recurrent depth {old} -> {args.loops}  (step {ck.get('step')})")

    new_state, rows = grow(ck["model"], old, args.loops)
    if not rows:
        raise SystemExit(f"no loop-sized tensors found with dim0=={old} — aborting "
                         "rather than writing a checkpoint that changes nothing")
    print(f"\n  {'tensor':46} {'from':>16} {'to':>16}  init")
    for k, a, b, pol in rows:
        print(f"  {k:46} {str(a):>16} {str(b):>16}  {pol}")
    print(f"\n  {len(rows)} tensors expanded")

    new_cfg = copy.deepcopy(cfg)
    new_cfg.max_loop_iters = args.loops
    out = dict(ck)
    out["model"] = new_state
    out["cfg"] = new_cfg
    if isinstance(ck.get("cfg_dict"), dict):
        d = dict(ck["cfg_dict"])
        d["max_loop_iters"] = args.loops
        out["cfg_dict"] = d

    # Optimizer moments for the grown tensors still have the OLD shape. Resuming
    # would crash (shape mismatch) or silently reset. Dropping optimizer state
    # entirely costs one warmup of Adam moments, which is far cheaper than a
    # confusing half-restored state.
    if "optimizer" in out:
        out.pop("optimizer")
        print("\n  optimizer state DROPPED — its moments for the grown tensors "
              "have the old shape.\n  Adam re-warms in a few hundred steps; a "
              "half-restored optimizer is worse.")

    extra = dict(out.get("extra") or {})
    extra["grown_depth_from"] = old
    extra["grown_depth_to"] = args.loops
    extra["grown_from_ckpt"] = str(args.checkpoint)
    out["extra"] = extra

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\n  wrote {args.out}")

    if args.verify:
        _verify(args.checkpoint, args.out, old)


def _verify(orig_path: str, grown_path: str, old_loops: int) -> None:
    """A forward at the ORIGINAL depth must be unchanged by growing."""
    from mythouro import MythOuro
    from mythouro.variants import mythouro_distill_tiny

    print("\n  verifying: forward at the original depth must be UNCHANGED ...")
    outs = []
    for path in (orig_path, grown_path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        cfg = copy.deepcopy(cfg)
        cfg.max_seq_len = 128
        m = MythOuro(cfg)
        state = {k: v for k, v in ck["model"].items()
                 if k not in ("freqs_cis", "freqs_cis_mla")}
        missing, unexpected = m.load_state_dict(state, strict=False)
        bad = [k for k in missing if "freqs" not in k]
        if bad:
            raise SystemExit(f"  FAIL: {len(bad)} keys did not load, e.g. {bad[:3]}")
        m.eval()
        torch.manual_seed(0)
        x = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            o = m(x, n_loops=old_loops)
        outs.append((o[0] if isinstance(o, (tuple, list)) else o).float())
    d = (outs[0] - outs[1]).abs().max().item()
    print(f"    max abs logit difference at n_loops={old_loops}: {d:.3e}")
    if d > 1e-5:
        raise SystemExit("  FAIL: growing CHANGED behaviour at the original depth")
    print("    PASS — original-depth behaviour is preserved; the new loops are "
          "additional capacity, not a perturbation.")


if __name__ == "__main__":
    main()
