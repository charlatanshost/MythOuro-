#!/usr/bin/env python
"""
P0.6 migration — fold `LoRAAdapter.scale` into `B`, non-destructively.

Background
----------
`LoRAAdapter.scale` (an `nn.Embedding(max_loops, rank)`, ones-init) was being
clobbered by `MythOuro._init_weights`' blanket `nn.Embedding` → N(0, 0.02)
re-init in every checkpoint (P0.6, same class as P0.1). The code fix tags
`scale` with `_skip_global_init` so fresh models keep the ones-init — but any
checkpoint trained *before* that fix has a random-small `scale` with the trained
`B` co-adapted to it. Simply loading such a checkpoint into the fixed code and
resetting `scale`→ones would blow up the adapter output.

The fold is FUNCTION-PRESERVING. The forward is

    delta = (down(x) * scale[t]) @ B[t]  ==  down(x) @ (scale[t][:, None] * B[t])

so multiplying `scale` into `B` per (loop t, rank r) and setting `scale`→ones
leaves the forward mathematically identical (up to fp reassociation) while
future gradients into `B` use the intended unit scale.

    B'[t, r, :] = scale[t, r] * B[t, r, :]        # (max_loops, rank, dim)
    scale'      = ones_like(scale)                 # (max_loops, rank)

Optimizer state
---------------
The 8-bit Adam moments for `B` and `scale` were accumulated against the old
parameterization; the fold rescales `B` rows by ~0.02-magnitude values with
random signs, so those moments are wrong in scale AND direction. Surgically
dropping only the `B`/`scale` entries requires matching them to their integer
indices in the saved optimizer state — fragile against a bitsandbytes 8-bit
optimizer's quantized per-param buffers, and a wrong index silently corrupts a
*different* parameter's momentum.

This script takes the robust route instead: it drops the ENTIRE optimizer state
(`ckpt["optimizer"] = {}`), so `load_checkpoint` starts a fresh optimizer on
resume — exactly the pattern `mythouro/grow.py` already uses for weight surgery.
The `step` (and thus the LR schedule) is preserved; only Adam's moments reset
and re-warm over ~1k steps, a mild bounded transient — appropriate for a
one-time boundary migration, especially since the training regime is changing
here anyway (P2 doubles on-policy gradient, EOS separators shift the data).

Usage
-----
    python -m tools.fold_lora_scale <in_ckpt> <out_ckpt>

Then MANDATORY equivalence check before resuming (see printout at the end):
load pre- and post-fold checkpoints into two models (post-fold built from the
patched code with the _skip_global_init tag), run the same batch in fp32 eval,
assert torch.allclose(logits_pre, logits_post, atol=1e-4).

Idempotent: re-running on an already-folded checkpoint (scale already ones) is
a no-op for the fold and re-marks the flag.
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Fold LoRAAdapter.scale into B (P0.6).")
    p.add_argument("in_ckpt", help="path to the pre-fold checkpoint (.pt)")
    p.add_argument("out_ckpt", help="path to write the folded checkpoint (.pt)")
    p.add_argument(
        "--keep-optimizer", action="store_true",
        help="Do NOT drop the optimizer state. UNSAFE — the B/scale moments are "
             "invalid post-fold; only use if you have a verified surgical drop.",
    )
    args = p.parse_args()

    print(f"[fold] loading {args.in_ckpt}")
    ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    scale_keys = [k for k in sd if k.endswith("lora.scale.weight")]
    if not scale_keys:
        sys.exit("[fold] no '*lora.scale.weight' in state dict — wrong checkpoint?")

    folded_any = False
    for sk in scale_keys:
        bk = sk.replace("scale.weight", "B")
        if bk not in sd:
            sys.exit(f"[fold] found {sk!r} but no matching B key {bk!r}")
        scale, B = sd[sk], sd[bk]
        # scale: (max_loops, rank)   B: (max_loops, rank, dim)
        if scale.shape != B.shape[:2]:
            sys.exit(f"[fold] shape mismatch: {sk}={tuple(scale.shape)} vs "
                     f"{bk}={tuple(B.shape)} (expected scale == B[:2])")
        if torch.allclose(scale, torch.ones_like(scale)):
            print(f"[fold] {sk}: already all-ones → skipping (idempotent)")
            continue
        # Compute in float32 for precision, cast back to B's storage dtype.
        sd[bk] = (B.float() * scale.float().unsqueeze(-1)).to(B.dtype)
        sd[sk] = torch.ones_like(scale)
        folded_any = True
        print(f"[fold] folded {sk} → {bk} | scale was "
              f"mean={scale.float().mean():.4f} std={scale.float().std():.4f} "
              f"min={scale.float().min():.4f} max={scale.float().max():.4f}")

    # Optimizer: full reset (robust). B/scale moments are invalid post-fold and
    # a surgical index-drop is fragile against the bnb 8-bit state; load_checkpoint
    # treats an empty dict as "fresh optimizer" (same as grow.py). Step/LR-schedule
    # is preserved in ckpt["step"].
    if not args.keep_optimizer:
        had = bool(ckpt.get("optimizer")) and "param_groups" in (ckpt.get("optimizer") or {})
        ckpt["optimizer"] = {}
        print(f"[fold] optimizer state {'dropped (fresh on resume)' if had else 'was already empty'}")
    else:
        print("[fold] WARNING: --keep-optimizer set; B/scale Adam moments are "
              "STALE and will mis-scale the first updates. Only safe with a "
              "verified surgical drop, which this script does not perform.")

    ckpt["extra"] = {**ckpt.get("extra", {}), "lora_scale_folded": True}
    torch.save(ckpt, args.out_ckpt)

    step = ckpt.get("step", "?")
    print(f"[fold] wrote {args.out_ckpt} (step {step}, folded_any={folded_any})")
    print(
        "\n[fold] MANDATORY before resuming training:\n"
        "  1. Equivalence: load pre-fold and post-fold into two models (post-fold\n"
        "     built from the PATCHED code with scale._skip_global_init), run the\n"
        "     same batch in fp32 eval, assert allclose(logits_pre, logits_post,\n"
        "     atol=1e-4). Worse than 1e-4 in fp32 means the fold is wrong.\n"
        "  2. Ones check: post-fold model's scale.weight is exactly all-ones\n"
        "     (proves the _skip_global_init tag survives fresh construction).\n"
        "  3. First ~50 resumed steps: gnorm may rise (B's gradient wakes up by\n"
        "     design) but should stay bounded by clip(max_norm=1.0); watch for\n"
        "     instability and re-arm component warmup if noisy."
    )


if __name__ == "__main__":
    main()
