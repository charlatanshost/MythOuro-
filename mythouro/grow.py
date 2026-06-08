"""
MythOuro MoE expansion (model growth).

Promotes a trained `mythouro_distill_tiny` checkpoint (24 routed + 2 shared
experts) into the larger `mythouro_distill_small` config (48 routed + 2 shared
experts), in a way that is *byte-exactly function-preserving* at the moment of
promotion.

Why MoE expansion first (vs Net2Wider / Net2Deeper)
---------------------------------------------------
MythOuro uses SiLU (inside SwiGLU). Idempotency of the activation only matters
for *depth* growth, not width:

* Net2Deeper inserts an identity-init layer that applies the activation twice
  (`φ(φ(Wx))`), which preserves function only if `φ(φ(z)) = φ(z)`. ReLU is
  idempotent; SiLU is NOT (`SiLU(SiLU(x)) ≠ SiLU(x)`), so depth growth would
  introduce a recoverable loss spike — or need an idempotent activation.
* Net2Wider duplicates a unit and splits its outgoing weight; this is
  function-preserving for ANY element-wise activation, SiLU included. So width
  growth is also a valid function-preserving axis for MythOuro.

MoE expansion is chosen as the *first* growth axis because it's the lowest-risk
(strictly loss-preserving via sentinel routing) — by gating new experts'
contribution to zero at promotion (via both a zeroed `down` projection AND a
sentinel router_bias), the promoted model
computes the exact same function as the source. See `docs/growth_design.md`
for the full discussion.

The promotion recipe (MoE-LPR style, adapted for DeepSeek-V3 routing)
--------------------------------------------------------------------
For each `MoEFFN` instance:

1. Copy source experts into the first `E_src` target slots, byte-identical.
2. Duplicate each source expert into a new target slot (round-robin), then
   zero the new expert's `down.weight`. The new expert is "alive" (gradient
   flows back) but its output is zero, so it contributes nothing at promotion.
3. Copy the source router weight rows into target router rows `[0:E_src]`.
4. Tile the source router weight rows into the new rows `[E_src:E_tgt]` so
   each new expert has the same routing direction as its parent. This means
   the bias decides which one of (parent, child) gets picked.
5. Copy the source `router_bias` buffer into the first `E_src` entries.
6. Set new entries `[E_src:E_tgt]` to a large negative *sentinel*. This
   guarantees that `(logits + router_bias).topk(K)` never selects a new
   expert at step 0, so promotion is bit-exact.

The sentinel decays toward 0 over the first `N_decay` training steps after
promotion (handled by `growth_sentinel_factor` — caller schedules). When the
sentinel reaches 0 the DeepSeek-V3 aux-loss-free bias updater
(`mythouro.training_utils.update_router_bias_from_counts`) takes over and
rebalances the larger pool naturally.

Limitations
-----------
* Only supports integer `expansion_factor`. `E_tgt = expansion_factor * E_src`.
  Non-integer ratios complicate the tile-source-router-rows step.
* Only grows routed experts. Shared experts stay at `n_shared_experts`
  unchanged.
* Only grows `n_experts`. Other shape fields (`dim`, `n_layers`, `expert_dim`,
  …) MUST match between source and target cfgs.
* Single-process only (no FSDP). The grown checkpoint can then be loaded
  under FSDP by the training script.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, is_dataclass

import torch
from loguru import logger


# Large-negative initial bias for new experts. At promotion the router computes
# `topk(logits + router_bias)`; with -100 on new experts, the pre-topk score for
# them is ~ -90..-110 vs a few units for established experts — they cannot enter
# the top-k. -1e9 would be even safer but introduces NaN risk in some autograd
# corner cases (subtractions in router stats). -100 is comfortably below any
# real logit on the existing router (which sees softmaxed values in [0, 1]
# before bias is added).
DEFAULT_SENTINEL_BIAS = -100.0


# ---------------------------------------------------------------------------
# Top-level public API
# ---------------------------------------------------------------------------


def grow_moe_checkpoint(
    src_path: str,
    dst_path: str,
    *,
    expansion_factor: int = 2,
    sentinel_bias: float = DEFAULT_SENTINEL_BIAS,
    perturb_scale: float = 0.0,
    n_decay_steps: int = 500,
) -> dict:
    """
    Promote a checkpoint on disk to a larger n_experts variant.

    Args:
        src_path           -- path to source `step_*.pt`
        dst_path           -- where to write the promoted checkpoint
        expansion_factor   -- multiplier on n_experts. 2 → double.
                              Only integer values supported.
        sentinel_bias      -- initial router_bias for new experts; large
                              negative makes promotion bit-exact. Decayed
                              over `n_decay_steps` post-promotion.
        perturb_scale      -- σ of Gaussian noise added to duplicated experts'
                              gate/up weights. 0.0 (default) is fine because
                              SGD noise breaks symmetry naturally; raise to
                              e.g. 1e-3 to accelerate divergence.
        n_decay_steps      -- step count over which the sentinel decays to 0.
                              Stored in checkpoint metadata; training script
                              reads it and applies the schedule.

    Returns:
        Dict of growth metadata that was embedded in the promoted checkpoint
        under `extra["growth_metadata"]`. Useful for callers that want to log
        the operation without re-reading the file.

    Side effects:
        Writes a new checkpoint to `dst_path`.
    """
    if expansion_factor < 2 or int(expansion_factor) != expansion_factor:
        raise ValueError(
            f"expansion_factor must be an integer >= 2, got {expansion_factor!r}"
        )
    expansion_factor = int(expansion_factor)

    logger.info(f"grow: loading source checkpoint {src_path}")
    src_ckpt = torch.load(src_path, map_location="cpu", weights_only=False)

    src_cfg = src_ckpt.get("cfg")
    if src_cfg is None:
        # Fall back to cfg_dict reconstruction for older checkpoints.
        from mythouro.main import MythOuroConfig
        src_cfg = MythOuroConfig(**src_ckpt["cfg_dict"])

    e_src = int(src_cfg.n_experts)
    e_tgt = e_src * expansion_factor
    logger.info(
        f"grow: source n_experts={e_src} → target n_experts={e_tgt} "
        f"(expansion_factor={expansion_factor})"
    )

    # Build the target cfg by copying and overriding n_experts.
    tgt_cfg = _clone_cfg_with_n_experts(src_cfg, e_tgt)

    # Promote the model state dict.
    src_state = src_ckpt["model"]
    tgt_state = _promote_state_dict(
        src_state,
        e_src=e_src,
        e_tgt=e_tgt,
        sentinel_bias=sentinel_bias,
        perturb_scale=perturb_scale,
    )

    # Build growth metadata so the training script can apply the
    # sentinel-decay schedule and so the lineage is recoverable from
    # the checkpoint alone.
    growth_metadata = {
        "source_path": os.path.abspath(src_path),
        "source_step": int(src_ckpt.get("step", 0)),
        "source_n_experts": e_src,
        "target_n_experts": e_tgt,
        "expansion_factor": expansion_factor,
        "sentinel_bias": float(sentinel_bias),
        "perturb_scale": float(perturb_scale),
        "n_decay_steps": int(n_decay_steps),
        "method": "moe_expansion_v1",
    }

    # Compose the destination checkpoint. Reuse most metadata from the source
    # (vocab_size, rng_state, scaler_state) — those carry forward unchanged.
    # Optimizer state is intentionally NOT copied because its shapes are
    # for the source model; the training script must build a fresh optimizer.
    extra = dict(src_ckpt.get("extra") or {})
    extra["growth_metadata"] = growth_metadata

    dst_ckpt = {
        "checkpoint_version": int(src_ckpt.get("checkpoint_version", 2)),
        "step": 0,                                       # start of new training
        "model": tgt_state,
        "optimizer": {},                                 # fresh optimizer required
        "cfg": tgt_cfg,
        "cfg_dict": _cfg_to_dict(tgt_cfg),
        "vocab_size": src_ckpt.get("vocab_size"),
        "rng_state": src_ckpt.get("rng_state"),
        "scaler_state": None,                            # fresh scaler
        "extra": extra,
    }

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    tmp_path = dst_path + ".tmp"
    torch.save(dst_ckpt, tmp_path)
    os.replace(tmp_path, dst_path)
    logger.success(f"grow: wrote promoted checkpoint → {dst_path}")

    return growth_metadata


def get_sentinel_decay_factor(step: int, growth_metadata: dict) -> float:
    """
    Linear decay schedule for the sentinel router_bias.

    Returns 1.0 at step 0 (full sentinel value applied) decaying linearly to
    0.0 at `n_decay_steps`. After `n_decay_steps`, returns 0.0 forever — the
    DeepSeek-V3 aux-loss-free updater takes over.

    Training scripts call this every step on a promoted checkpoint and
    multiply the result by `growth_metadata["sentinel_bias"]` to get the
    current bias delta to apply to the new-expert slots.

    Args:
        step              -- current training step relative to promotion
                             (i.e., the training-script step counter, which
                             starts at 0 for a grown checkpoint)
        growth_metadata   -- dict from `extra["growth_metadata"]`

    Returns:
        Decay factor in [0.0, 1.0].
    """
    n_decay = int(growth_metadata.get("n_decay_steps", 500))
    if n_decay <= 0 or step >= n_decay:
        return 0.0
    return 1.0 - (float(step) / float(n_decay))


def apply_sentinel_to_router_biases(
    model,
    growth_metadata: dict,
    step: int,
) -> None:
    """
    For each `MoEFFN` in the model, force the new-expert entries of
    `router_bias` to (sentinel * decay_factor(step)). Called every step on
    promoted checkpoints.

    This is the runtime side of the sentinel scheme. The DeepSeek-V3
    updater otherwise drifts the new-expert biases according to traffic
    counts, but during the decay window we WANT those biases to follow the
    schedule rather than be data-driven.

    Args:
        model               -- the MythOuro module being trained
        growth_metadata     -- from `extra["growth_metadata"]`
        step                -- current training step (post-promotion)
    """
    from mythouro.main import MoEFFN

    factor = get_sentinel_decay_factor(step, growth_metadata)
    if factor == 0.0:
        return  # decay finished; let the regular updater run unimpeded

    sentinel = float(growth_metadata["sentinel_bias"])
    e_src = int(growth_metadata["source_n_experts"])
    bias_value = sentinel * factor

    with torch.no_grad():
        for mod in model.modules():
            if not isinstance(mod, MoEFFN):
                continue
            mod.router_bias[e_src:] = bias_value


# ---------------------------------------------------------------------------
# Internals — state_dict promotion
# ---------------------------------------------------------------------------


def _promote_state_dict(
    src_state: "dict[str, torch.Tensor]",
    *,
    e_src: int,
    e_tgt: int,
    sentinel_bias: float,
    perturb_scale: float,
) -> "dict[str, torch.Tensor]":
    """
    Walk the source state dict, promote any MoEFFN sub-dicts, leave everything
    else byte-identical.

    The state dict has keys like:
        "<prefix>.router.weight"            (E_src, dim)
        "<prefix>.router_bias"              (E_src,)
        "<prefix>.routed_experts.<i>.gate.weight"
        "<prefix>.routed_experts.<i>.up.weight"
        "<prefix>.routed_experts.<i>.down.weight"

    We detect MoEFFN modules by looking for the router_bias buffer key — any
    `<prefix>.router_bias` is an MoEFFN to promote. Everything outside of
    `<prefix>.{router,router_bias,routed_experts}` passes through unchanged.

    `shared_experts` are NOT touched (we don't grow the shared pool).
    """
    src_state = dict(src_state)  # shallow copy of keys

    # Find all MoEFFN prefixes by locating router_bias keys.
    moe_prefixes = [
        k[: -len(".router_bias")]
        for k in src_state.keys()
        if k.endswith(".router_bias")
    ]
    logger.info(f"grow: detected {len(moe_prefixes)} MoEFFN layer(s) to promote")

    promoted_keys: "set[str]" = set()
    new_entries: "dict[str, torch.Tensor]" = {}

    for prefix in moe_prefixes:
        # 1. Router weight: (E_tgt, dim) tiled from (E_src, dim)
        rk = f"{prefix}.router.weight"
        w_src = src_state[rk]                                   # (E_src, dim)
        if w_src.shape[0] != e_src:
            raise RuntimeError(
                f"{rk}: expected first dim {e_src}, got {tuple(w_src.shape)}"
            )
        w_tgt = w_src.repeat(e_tgt // e_src, 1).contiguous()    # tile
        new_entries[rk] = w_tgt
        promoted_keys.add(rk)

        # 2. router_bias: (E_tgt,). First E_src entries identical; rest sentinel.
        bk = f"{prefix}.router_bias"
        b_src = src_state[bk]                                   # (E_src,)
        b_tgt = torch.full(
            (e_tgt,), float(sentinel_bias),
            dtype=b_src.dtype, device=b_src.device,
        )
        b_tgt[:e_src] = b_src
        new_entries[bk] = b_tgt
        promoted_keys.add(bk)

        # 3. Routed experts: copy [0:E_src] verbatim, tile new slots with
        #    zeroed-down promotion.
        expert_prefix = f"{prefix}.routed_experts."
        for i in range(e_src):
            for piece in ("gate.weight", "up.weight", "down.weight"):
                k = f"{expert_prefix}{i}.{piece}"
                new_entries[k] = src_state[k]                   # byte-identical
                promoted_keys.add(k)

        for i in range(e_src, e_tgt):
            parent = i % e_src
            for piece in ("gate.weight", "up.weight", "down.weight"):
                k_new = f"{expert_prefix}{i}.{piece}"
                k_src = f"{expert_prefix}{parent}.{piece}"
                w = src_state[k_src].clone()
                if piece == "down.weight":
                    # Zero out — this is the key trick that makes promotion
                    # function-preserving. Without it, even with the sentinel
                    # bias blocking selection, an autograd graph anomaly or
                    # router-bias decay step could leak the expert into the
                    # forward and shift the output.
                    w.zero_()
                elif perturb_scale > 0.0:
                    w = w + perturb_scale * torch.randn_like(w)
                new_entries[k_new] = w
                promoted_keys.add(k_new)

    # Pass everything else through unchanged.
    tgt_state: "dict[str, torch.Tensor]" = {}
    for k, v in src_state.items():
        if k in promoted_keys:
            tgt_state[k] = new_entries[k]
        else:
            tgt_state[k] = v
    # Add any new keys that didn't exist in the source (the expanded experts
    # past index E_src). promoted_keys covered them via new_entries; we just
    # need to merge anything not yet written.
    for k, v in new_entries.items():
        tgt_state.setdefault(k, v)

    return tgt_state


def _clone_cfg_with_n_experts(src_cfg, n_experts: int):
    """
    Return a copy of `src_cfg` with `n_experts` set to the new value. Works
    for dataclass-style cfgs (which is what MythOuroConfig is).
    """
    tgt_cfg = copy.deepcopy(src_cfg)
    tgt_cfg.n_experts = n_experts
    return tgt_cfg


def _cfg_to_dict(cfg) -> dict:
    """Mirror of `mythouro.checkpointing._cfg_to_dict` to keep grow.py
    importable without circular dependencies."""
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(vars(cfg))
