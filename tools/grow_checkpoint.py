#!/usr/bin/env python3
"""
CLI wrapper around `mythouro.grow.grow_moe_checkpoint`.

Promote a trained MoE checkpoint (e.g., `mythouro_distill_tiny` at 24 experts)
to a larger expert pool (`mythouro_distill_small` at 48 experts) in a
function-near-preserving way. See `docs/growth_design.md` for the algorithm
and `mythouro/grow.py` for the implementation.

Usage
-----
    python tools/grow_checkpoint.py \\
        --src archived_models/mythouro_distill_tiny_sft_v2/step_0003000.pt \\
        --dst checkpoints_grown/promoted_step_0.pt \\
        --expansion-factor 2

After running this, train with:

    python -m training.sft \\
        --resume checkpoints_grown/promoted_step_0.pt \\
        --student-variant mythouro_distill_small \\
        --device cuda:0 \\
        ...

The training script detects the `growth_metadata` in `extra` and applies the
sentinel-decay schedule automatically.
"""

from __future__ import annotations

import argparse
import os
import sys

# Add the project root (the directory containing `mythouro/`) to sys.path
# so this script works when invoked as `python tools/grow_checkpoint.py`
# from the project root, without needing `PYTHONPATH` or `python -m`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mythouro.grow import DEFAULT_SENTINEL_BIAS, grow_moe_checkpoint


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(
        description="Promote a MythOuro checkpoint to a larger n_experts variant.",
    )
    p.add_argument(
        "--src", required=True,
        help="Path to source step_*.pt checkpoint to promote.",
    )
    p.add_argument(
        "--dst", required=True,
        help="Path to write the promoted checkpoint to.",
    )
    p.add_argument(
        "--expansion-factor", type=int, default=2,
        help="Multiplier on n_experts (default 2 — doubles the routed pool).",
    )
    p.add_argument(
        "--sentinel-bias", type=float, default=DEFAULT_SENTINEL_BIAS,
        help=(
            "Initial router_bias for new experts. Large negative makes "
            "promotion function-preserving. Decayed to 0 over "
            "`--n-decay-steps` after promotion."
        ),
    )
    p.add_argument(
        "--perturb-scale", type=float, default=0.0,
        help=(
            "Gaussian noise sigma added to duplicated experts' gate/up "
            "weights. 0.0 (default) is fine — SGD noise breaks symmetry "
            "naturally. Raise to ~1e-3 to accelerate divergence."
        ),
    )
    p.add_argument(
        "--n-decay-steps", type=int, default=500,
        help=(
            "Number of post-promotion training steps over which the sentinel "
            "bias decays linearly to 0. The DeepSeek-V3 aux-loss-free updater "
            "then takes over."
        ),
    )
    args = p.parse_args(argv)

    metadata = grow_moe_checkpoint(
        src_path=args.src,
        dst_path=args.dst,
        expansion_factor=args.expansion_factor,
        sentinel_bias=args.sentinel_bias,
        perturb_scale=args.perturb_scale,
        n_decay_steps=args.n_decay_steps,
    )

    print("\nGrowth metadata embedded in promoted checkpoint:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print(
        "\nNext step:\n"
        f"  python -m training.sft --resume {args.dst} "
        f"--student-variant mythouro_distill_small --device cuda:0 ..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
