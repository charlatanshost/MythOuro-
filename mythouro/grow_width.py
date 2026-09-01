"""
MythOuro width growth (Net2Wider) — the axis that actually raises ACTIVATED params.

WHY THIS AND NOT MORE EXPERTS. Expert-count growth was measured out over
2026-08-28..09-01, twice, on a healthy base:

    activated params, 24 experts:  180,726,115
    activated params, 48 experts:  180,726,115     <- UNCHANGED

Only `n_experts_per_tok` (4) experts fire per token, so promoting adds STORAGE
the model cannot reach within a token. The new experts converged to ~90% twins of
their parents on an asymptotic curve (fit 0.893), router perturbation to cos
0.704 made differentiation *slower*, removing the load balancer moved it by
-0.003, and capability came out flat against the pre-growth base twice.

Widening `expert_dim` raises the inner dimension of the experts that DO fire, so
activated params scale with it. It is the only growth axis that moves the
quantity the capacity hypothesis identifies.

WHY IT IS FUNCTION-PRESERVING UNDER SwiGLU. `Expert` computes
`down(silu(gate(x)) * up(x))`. Duplicating hidden unit i as unit j copies BOTH
`gate[i]` and `up[i]`, so both units produce the identical activation a_i. Split
the outgoing weight in half and the sum is unchanged:

    down[:,i]*a_i                    ->  (down[:,i]/2)*a_i + (down[:,j]/2)*a_i
                                      =  down[:,i]*a_i

This holds for ANY element-wise activation — SiLU included. It is *depth* growth
(Net2Deeper) that SiLU blocks, because that applies the activation twice and
needs idempotency. See `docs/growth_design.md`.

⚠ NOISE IS MANDATORY HERE, NOT OPTIONAL — and this is the hard-won part.
A duplicated unit pair receives the SAME input (unlike routed experts, which at
least see different token subsets) and, after the even split, the SAME gradient.
Exactly symmetric units with exactly equal gradients stay exactly equal FOREVER;
SGD noise cannot separate them because there is no asymmetry to amplify. This is
strictly worse than the expert case, where routing differed and the twins still
failed to separate. `noise_scale` therefore defaults to a non-zero value, and
`grow_width_checkpoint` REFUSES noise_scale=0 unless explicitly forced.

The noise is applied to the DOWN projection of the duplicated pair in equal and
opposite amounts, which breaks the symmetry while keeping the sum exact:

    down[:,i] = down[:,i]/2 + eps
    down[:,j] = down[:,i]/2 - eps      (sum still down[:,i], function preserved)

So promotion stays bit-exact AND the pair is asymmetric from step 0. That is
something the expert-count promotion could not do — its symmetry breaking cost
function preservation, so it had to hide behind a sentinel and decay in.
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, is_dataclass

import torch
from loguru import logger

DEFAULT_NOISE_SCALE = 0.02


def grow_width_checkpoint(
    src_path: str,
    dst_path: str,
    *,
    expansion_factor: int = 2,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    allow_zero_noise: bool = False,
    seed: int = 1234,
) -> dict:
    """
    Promote a checkpoint to a larger `expert_dim`, function-preservingly.

    Args:
        src_path         -- source `step_*.pt`
        dst_path         -- destination checkpoint
        expansion_factor -- multiplier on expert_dim. 2 -> double.
        noise_scale      -- symmetry-breaking noise on the duplicated pair's
                            down-projection, RELATIVE to that column's std.
                            Applied equal-and-opposite so the sum is exact.
                            **Do not set to 0** — see the module docstring.
        allow_zero_noise -- required to permit noise_scale=0.0.
        seed             -- RNG seed for the noise.
    """
    if expansion_factor < 2 or int(expansion_factor) != expansion_factor:
        raise ValueError(f"expansion_factor must be an integer >= 2, got {expansion_factor!r}")
    if noise_scale == 0.0 and not allow_zero_noise:
        raise ValueError(
            "noise_scale=0.0 produces EXACTLY symmetric unit pairs that receive "
            "exactly equal gradients and can never separate. The expert-count "
            "promotion failed for a weaker version of this reason. Pass "
            "allow_zero_noise=True only to reproduce that failure deliberately."
        )
    expansion_factor = int(expansion_factor)

    logger.info(f"grow_width: loading {src_path}")
    ck = torch.load(src_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg")
    if cfg is None:
        from mythouro.main import MythOuroConfig
        cfg = MythOuroConfig(**ck["cfg_dict"])

    h_src = int(cfg.expert_dim)
    h_tgt = h_src * expansion_factor
    logger.info(f"grow_width: expert_dim {h_src} -> {h_tgt}")

    tgt_cfg = copy.deepcopy(cfg)
    tgt_cfg.expert_dim = h_tgt

    g = torch.Generator().manual_seed(seed)
    state = _widen_state_dict(ck["model"], expansion_factor, noise_scale, g)

    meta = {
        "source_path": os.path.abspath(src_path),
        "source_step": int(ck.get("step", 0)),
        "source_expert_dim": h_src,
        "target_expert_dim": h_tgt,
        "expansion_factor": expansion_factor,
        "noise_scale": float(noise_scale),
        "seed": int(seed),
        "method": "net2wider_v1",
    }
    extra = dict(ck.get("extra") or {})
    extra["width_growth_metadata"] = meta

    out = {
        "checkpoint_version": int(ck.get("checkpoint_version", 2)),
        "step": 0,
        "model": state,
        "optimizer": {},                 # shapes changed; fresh optimizer required
        "cfg": tgt_cfg,
        "cfg_dict": _cfg_to_dict(tgt_cfg),
        "vocab_size": ck.get("vocab_size"),
        "rng_state": ck.get("rng_state"),
        "scaler_state": None,
        "extra": extra,
    }
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    tmp = dst_path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, dst_path)
    logger.success(f"grow_width: wrote {dst_path}")
    return meta


def _widen_state_dict(src, factor: int, noise_scale: float, g) -> dict:
    """Widen every Expert triple (gate, up, down); pass everything else through.

    An Expert is detected by the presence of `<prefix>.down.weight` alongside
    `<prefix>.gate.weight` and `<prefix>.up.weight`. This catches routed experts,
    shared experts, AND the dense prelude/coda FFNs, which are the same module.
    """
    src = dict(src)
    prefixes = [
        k[: -len(".gate.weight")] for k in src
        if k.endswith(".gate.weight")
        and f"{k[:-len('.gate.weight')]}.up.weight" in src
        and f"{k[:-len('.gate.weight')]}.down.weight" in src
    ]
    logger.info(f"grow_width: {len(prefixes)} SwiGLU block(s) to widen")

    out = dict(src)
    for p in prefixes:
        gate, up = src[f"{p}.gate.weight"], src[f"{p}.up.weight"]
        down = src[f"{p}.down.weight"]                      # (dim, H)
        H = gate.shape[0]
        idx = torch.arange(H).repeat(factor - 1)            # which units to copy

        out[f"{p}.gate.weight"] = torch.cat([gate, gate[idx]], dim=0).contiguous()
        out[f"{p}.up.weight"] = torch.cat([up, up[idx]], dim=0).contiguous()

        # Split the outgoing weight evenly, then perturb equal-and-opposite so
        # the SUM is untouched (function-preserving) but the pair is asymmetric.
        share = down / float(factor)                        # (dim, H)
        cols = [share.clone() for _ in range(factor)]
        if noise_scale > 0.0:
            sd = down.float().std()
            for c in range(factor - 1):
                eps = torch.randn(share.shape, generator=g, dtype=share.dtype) * (noise_scale * sd)
                cols[c] = cols[c] + eps
                cols[-1] = cols[-1] - eps                   # keep the sum exact
        out[f"{p}.down.weight"] = torch.cat(cols, dim=1).contiguous()
    return out


def _cfg_to_dict(cfg) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(vars(cfg))
