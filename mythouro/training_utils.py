"""
Training utilities for MythOuro (Part 1).

Components
----------
collect_router_logits   — walk a (possibly FSDP-wrapped) model and gather the
                          per-step router logits stashed by MoEFFN on every
                          forward pass. Used by load_balance_loss and any
                          downstream MoE-aware diagnostic.

load_balance_loss       — classic Switch/DeepSeekMoE auxiliary loss that pulls
                          expert utilisation toward uniform. Complements (does
                          not replace) DeepSeek-V3's aux-loss-free routing-bias
                          mechanism already in MoEFFN.

consistency_loss        — KL self-distillation across loop depths. Forces the
                          shallow-loop prediction to track the deep-loop one,
                          so easy tokens converge fast and the ACT halting
                          signal acquires meaningful discriminative power.

uncertainty_calibration_loss — BCE between the UncertaintyHead's per-token
                          score and the realised error mask. Trains the
                          uncertainty score to be a *calibrated* probability
                          of error rather than a raw entropy proxy.

LoopCurriculum          — linear ramp of n_loops from `start` to `max_loops`
                          across the run. Early steps train cheap shallow
                          loops; later steps engage the full depth.

MixedDataset            — streaming IterableDataset that interleaves three HF
                          corpora at user-tunable ratios:
                            40% HuggingFaceFW/fineweb-edu     (general)
                            40% open-web-math/open-web-math   (math)
                            20% codeparrot/codeparrot-clean   (code)

combined_loss           — single-call master that returns (total_loss, metrics)
                          combining CE + load-balance + uncertainty
                          calibration. Pass `step` to enable schedule-aware
                          coefficients.

log_spectral_radius     — diagnostic: prints ρ(A) of every LTIInjection found
                          inside the model. Cheap (one buffer access per
                          module), call every few hundred steps.

All loss helpers return torch.Tensor scalars. None of them require new
parameters on the model; the only state outside parameters is the
`_last_router_logits` attribute stashed by MoEFFN.forward (added in Part 1A.2).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

# Importing inside functions where datasets is needed keeps the rest of this
# module importable from environments that don't have `datasets` installed
# (CI, lint, smoke tests). Same trick for the MoEFFN / LTIInjection type
# checks below.


# ---------------------------------------------------------------------------
# Router-logit collection
# ---------------------------------------------------------------------------


def collect_router_logits(model: nn.Module) -> list[torch.Tensor]:
    """
    Gather the `_last_router_logits` tensor that MoEFFN.forward stashes on
    every call. Returns one tensor per MoE layer in the model in module order.

    Robust to FSDP wrapping: walks `model.modules()` rather than a hardcoded
    path, so the same code works for unwrapped, DDP, and FSDP models alike.

    Returns an empty list if no MoE layers are found or none have been
    forwarded yet (the attribute is set in-place by the forward pass).
    """
    from mythouro.main import MoEFFN

    buf: list[torch.Tensor] = []
    for mod in model.modules():
        if not isinstance(mod, MoEFFN):
            continue
        # P0.2: prefer the per-forward buffer holding ALL loops' router logits
        # (one (N, E) tensor per recurrent loop). Fall back to the single
        # last-call stash for contexts that didn't loop (e.g. a bare MoEFFN
        # forward outside RecurrentBlock).
        per_loop = getattr(mod, "_router_logits_buf", None)
        if per_loop:
            buf.extend(per_loop)
        elif getattr(mod, "_last_router_logits", None) is not None:
            buf.append(mod._last_router_logits)
    return buf


# ---------------------------------------------------------------------------
# Aux-loss-free MoE routing: bias update + utilisation logging
# ---------------------------------------------------------------------------


def collect_expert_counts(model: nn.Module) -> "dict[str, torch.Tensor]":
    """
    Snapshot per-expert dispatch counts (shape `(n_experts,)`) from every
    MoEFFN, keyed by qualified name so multiple MoE layers update independently.

    P0.2: returns counts **summed across all recurrent loops** (`_expert_counts_sum`),
    so the DeepSeek-V3 bias updater balances total per-forward expert usage, not
    just the last loop's. Falls back to the single last-call stash if the sum
    isn't populated (non-recurrent context).

    Called once per micro-step; the trainer accumulates across grad-accum
    micro-steps before calling `update_router_bias_from_counts`.
    """
    from mythouro.main import MoEFFN

    out: "dict[str, torch.Tensor]" = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, MoEFFN):
            continue
        counts = getattr(mod, "_expert_counts_sum", None)
        if counts is None:
            counts = getattr(mod, "_last_expert_counts", None)
        if counts is not None:
            out[name] = counts.detach().clone()
    return out


def _maybe_all_reduce_counts(
    counts_by_layer: "dict[str, torch.Tensor]",
    ddp: bool,
) -> "dict[str, torch.Tensor]":
    """
    Sum expert counts across all distributed ranks.

    DeepSeek-V3's aux-loss-free routing keeps `router_bias` in sync across
    ranks because they're all driven by the same global counts. Without
    this all-reduce each rank sees only its local tokens and the buffers
    drift, defeating the balancing mechanism.
    """
    if not ddp:
        return counts_by_layer
    import torch.distributed as dist
    out: "dict[str, torch.Tensor]" = {}
    for name, c in counts_by_layer.items():
        # all_reduce is in-place; clone so we don't mutate the caller's tensor.
        c = c.clone()
        dist.all_reduce(c, op=dist.ReduceOp.SUM)
        out[name] = c
    return out


def update_router_bias_from_counts(
    model: nn.Module,
    counts_by_layer: "dict[str, torch.Tensor]",
    bias_lr: float = 1e-3,
    ddp: bool = False,
) -> "dict[str, dict]":
    """
    Apply the aux-loss-free DeepSeek-V3 routing bias update.

    For each MoEFFN with accumulated counts `c[i]` for expert i:
        target = (Σ_i c[i]) / n_experts            # uniform target
        bias[i] += bias_lr * sign(target - c[i])   # nudge underused up

    The bias is a non-gradient buffer, so this update happens *outside*
    the optimizer (call after `optimizer.step()`). Under DDP/FSDP the
    counts are all-reduced first so every rank applies the same update
    and the buffer stays in sync.

    Returns per-layer utilisation diagnostics:
        {layer_name: {"cv": float, "min_frac": float, "max_frac": float,
                       "n_tokens": int}}
    suitable for logging every N steps.
    """
    from mythouro.main import MoEFFN

    counts_by_layer = _maybe_all_reduce_counts(counts_by_layer, ddp)
    name_to_mod = dict(model.named_modules())

    stats: "dict[str, dict]" = {}
    for name, counts in counts_by_layer.items():
        mod = name_to_mod.get(name)
        if not isinstance(mod, MoEFFN):
            continue
        n_experts = mod.n_experts
        total = float(counts.sum().item())
        if total <= 0:
            continue
        target = total / n_experts
        # sign(target - counts): +1 where expert is underused, -1 where overused
        delta = torch.sign(target - counts.to(mod.router_bias.dtype))
        mod.router_bias.add_(bias_lr * delta)

        fracs = counts.float() / total
        stats[name] = {
            "cv": float(fracs.std().item() / (fracs.mean().item() + 1e-12)),
            "min_frac": float(fracs.min().item()),
            "max_frac": float(fracs.max().item()),
            "n_tokens": int(total),
            "bias_l2": float(mod.router_bias.detach().norm().item()),
        }
    return stats


# ---------------------------------------------------------------------------
# §3 New-component LR warmup
# ---------------------------------------------------------------------------
#
# Why this exists (re-scoped from AGENT_TASKS.md):
#
# The original spec listed every "new component" added in Parts 1+2 as a
# warmup target. That's wrong — most of them are zero-output-initialised
# (CrossLoopAttention's `o_proj`, UncertaintyHead's final Linear,
# ProcessRewardHead's final Linear) and self-warm by virtue of having
# nothing to contribute at step 0.
#
# The real risk surface is:
#     - InjectionScheduler.log_scale (cosine-init, immediately scales B(t)·e)
#     - LoRAAdapter.down            (std=0.02 init, gradient leaks into the
#                                    recurrent block before B is trained up)
#     - MultiScaleInjection         (std=0.02 projections + non-uniform
#                                    blend weights, immediately rewrites e)
#
# These three start contributing *real* signal at step 0; warming them
# gives the base block 2k steps to settle before they intervene.


_WARMUP_COMPONENT_NAMES = ("InjectionScheduler", "LoRAAdapter", "MultiScaleInjection")


def _collect_new_component_param_ids(model: nn.Module) -> set:
    """
    Walk `model.modules()` and return the `id()` of every parameter belonging
    to a risk-prone new component.

    For `LoRAAdapter` we deliberately include ONLY the `down` projection —
    `B` is zero-init (self-warms) and `scale` only multiplies that zero,
    so warming them is unnecessary churn. For the other two, every owned
    parameter goes in the group.
    """
    from mythouro.main import (
        InjectionScheduler, LoRAAdapter, MultiScaleInjection,
    )
    ids: set = set()
    for mod in model.modules():
        if isinstance(mod, InjectionScheduler):
            ids.add(id(mod.log_scale))
        elif isinstance(mod, LoRAAdapter):
            ids.update(id(p) for p in mod.down.parameters())
        elif isinstance(mod, MultiScaleInjection):
            ids.update(id(p) for p in mod.parameters())
    return ids


def get_optimizer_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.0,
    extra_base_params=(),
) -> "list[dict]":
    """
    Split the model's parameters into two optimizer groups for separate
    LR scheduling:

        Group "base"          — everything else (full LR every step).
        Group "new_component" — InjectionScheduler / LoRAAdapter.down /
                                MultiScaleInjection. Its LR is multiplied
                                by `ComponentWarmup.factor(step)` at every
                                training step via `apply_component_warmup`.

    `extra_base_params` is for auxiliary heads constructed outside the
    main model (e.g. ProcessRewardHead, ExpertSpecializationProbe). They
    go into the base group at full LR — they're zero-output-init and
    self-warm.

    Returns a list of dicts suitable for `torch.optim.AdamW(...)`.
    """
    new_ids = _collect_new_component_param_ids(model)
    base_params = []
    new_params = []
    for p in model.parameters():
        (new_params if id(p) in new_ids else base_params).append(p)
    base_params.extend(extra_base_params)
    return [
        {
            "params": base_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "base",
        },
        {
            "params": new_params,
            "lr": base_lr,           # start at full; warmup factor applied per step
            "weight_decay": weight_decay,
            "name": "new_component",
        },
    ]


@dataclass
class ComponentWarmup:
    """
    Linear LR ramp 0 → 1 over `warmup_steps`. Stateless on step, like
    `LoopCurriculum` and `LoopDepthAnnealer` — checkpoints don't need
    to save anything to resume the schedule.
    """
    warmup_steps: int

    def factor(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        if step >= self.warmup_steps:
            return 1.0
        return max(0.0, step) / self.warmup_steps


def apply_component_warmup(
    optimizer,
    base_lr: float,
    step: int,
    warmup_steps: int,
) -> float:
    """
    Set per-group LR on `optimizer` for this step. Replaces the canonical
    `for g in optimizer.param_groups: g["lr"] = cur_lr` line in the
    training loop.

    Base group sees `base_lr`; new_component group sees
    `base_lr * factor(step)`. Returns the factor for logging.
    """
    factor = ComponentWarmup(warmup_steps).factor(step)
    for g in optimizer.param_groups:
        if g.get("name") == "new_component":
            g["lr"] = base_lr * factor
        else:
            g["lr"] = base_lr
    return factor


def log_expert_utilization(stats: "dict[str, dict]", step: int) -> None:
    """
    Pretty-print MoE utilisation diagnostics.

    `cv` (coefficient of variation) is the single most informative
    scalar: 0 ↔ perfectly uniform, grows as routing skews. Spikes
    above ~0.5 indicate the bias updater is falling behind and the
    loss is starting to depend on a handful of dominant experts.
    """
    if not stats:
        return
    parts = []
    for name, s in stats.items():
        short = name.rsplit(".", 1)[-1] or name
        parts.append(
            f"{short}: cv={s['cv']:.3f} "
            f"min={s['min_frac']*100:.1f}% max={s['max_frac']*100:.1f}% "
            f"bias|·|₂={s['bias_l2']:.3f}"
        )
    logger.info(f"step {step:>7d} | MoE util: " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Load balance loss (Switch / DeepSeekMoE auxiliary)
# ---------------------------------------------------------------------------


def load_balance_loss(
    router_logits_buf: list[torch.Tensor],
    topk: int,
) -> torch.Tensor:
    """
    Standard MoE load-balancing auxiliary loss.

    For each layer's router logits (N, E):
        f_i = fraction of (token, slot) pairs whose top-k router selected expert i
        P_i = mean unbiased softmax probability assigned to expert i across tokens
        L_layer = E * Σ_i f_i * P_i

    Minimum 1.0 (perfectly uniform routing); deviations push it above 1.
    Multiply by ~1e-2 when adding to the main objective.

    The expert count E is read from each tensor's last dim — no need to
    pass it explicitly. Returned as a single mean across all MoE layers.
    """
    if not router_logits_buf:
        return torch.tensor(0.0)

    losses = []
    for logits in router_logits_buf:                          # (N, E)
        N, E = logits.shape
        probs = F.softmax(logits, dim=-1)                     # (N, E)
        # f_i: fraction of (token, slot) pairs assigned to expert i across
        # the topk selection. Matches the routing path used in MoEFFN.forward.
        _, topk_idx = probs.topk(topk, dim=-1)                # (N, topk)
        one_hot = F.one_hot(topk_idx, num_classes=E).float()  # (N, topk, E)
        f_i = one_hot.sum(dim=(0, 1)) / (N * topk)            # (E,)
        P_i = probs.mean(dim=0)                               # (E,)
        losses.append((f_i * P_i).sum() * E)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Cross-depth consistency (KL self-distillation)
# ---------------------------------------------------------------------------


def consistency_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_loops_low: int,
    n_loops_high: int,
    detach_teacher: bool = True,
) -> torch.Tensor:
    """
    KL self-distillation across loop depths.

    Runs the model at both loop counts and minimises
        KL(p_deep || p_shallow)
    so the shallow prediction is pulled toward the deep one. This trains the
    early loops to commit quickly when they already match the answer the deep
    loops would arrive at — sharpening the ACT halting signal in the process.

    `detach_teacher=True` (default): only the shallow distribution receives
    gradient. Standard self-distillation choice — prevents the deep network
    from racing the shallow one to a degenerate match.

    Cost: one extra forward at `n_loops_high`. Run on a fraction of steps
    (e.g. every 4th) rather than every step to amortise.
    """
    # Deep (teacher) forward
    if detach_teacher:
        with torch.no_grad():
            logits_deep, _ = model(input_ids, n_loops=n_loops_high)
    else:
        logits_deep, _ = model(input_ids, n_loops=n_loops_high)

    # Shallow (student) forward — gradient flows here
    logits_shallow, _ = model(input_ids, n_loops=n_loops_low)

    # KL(p_deep || p_shallow) — uses log_target=False, i.e. computes
    # p_deep * (log p_deep - log p_shallow), summed over vocab and
    # averaged over tokens.
    p_deep = F.softmax(logits_deep.float(), dim=-1)
    log_shallow = F.log_softmax(logits_shallow.float(), dim=-1)
    return F.kl_div(log_shallow, p_deep, reduction="batchmean")


# ---------------------------------------------------------------------------
# Uncertainty calibration
# ---------------------------------------------------------------------------


def uncertainty_calibration_loss(
    logits: torch.Tensor,
    uncertainty: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy between predicted uncertainty and realised error.

    Target = 1 if the model's argmax at position t differs from the gold
    token, 0 otherwise. The UncertaintyHead is therefore trained to predict
    "probability that I will be wrong at this position" — directly
    interpretable as a confidence score.

    Logits are detached from the graph: we don't want the LM head to be
    optimised against the uncertainty signal, only the uncertainty head
    itself.
    """
    with torch.no_grad():
        pred = logits.argmax(dim=-1)               # (B, T)
        target_unc = (pred != targets).float()     # 1 = wrong, 0 = right

    # UncertaintyHead already outputs probabilities in (0, 1) via sigmoid
    # in its own forward — clamp for numerical safety in float16/bfloat16.
    p = uncertainty.float().clamp(1e-6, 1 - 1e-6)
    # Manual closed-form BCE rather than F.binary_cross_entropy: torch's
    # BCE is autocast-banned (it asks callers to use
    # binary_cross_entropy_with_logits instead, but we don't have the
    # pre-sigmoid logit here — UncertaintyHead already passed it through
    # sigmoid). The math is the same, just spelled out:
    #     BCE(p, t) = − [ t · log p  +  (1 − t) · log(1 − p) ]
    # Computed in fp32 (p was `.float()`-cast above) so it stays safe
    # under bf16/fp16 autocast.
    bce = -(target_unc * torch.log(p) + (1.0 - target_unc) * torch.log(1.0 - p))
    return bce.mean()


# ---------------------------------------------------------------------------
# Loop curriculum
# ---------------------------------------------------------------------------


@dataclass
class LoopCurriculum:
    """
    Linear ramp of recurrent loop depth across training.

    Schedule (in optimiser steps):
        step < warmup_steps        → start_loops
        warmup_steps ≤ step < end  → linear ramp start_loops → max_loops
        step ≥ end                 → max_loops

    Use a low `start_loops` (1–2) for the first ~5% of training to amortise
    the cost while parameters are still moving rapidly; ramp to
    `cfg.max_loop_iters` by mid-training so the model spends most of the run
    learning at full depth.
    """

    start_loops: int
    max_loops: int
    warmup_steps: int
    total_steps: int

    def get(self, step: int) -> int:
        if step < self.warmup_steps:
            return self.start_loops
        denom = max(1, self.total_steps - self.warmup_steps)
        frac = min(1.0, (step - self.warmup_steps) / denom)
        return int(self.start_loops + frac * (self.max_loops - self.start_loops))

    def get_sampled(self, step: int, rng: "random.Random | None" = None) -> int:
        """
        Return a *random* unroll depth uniformly drawn from
        ``[start_loops, get(step)]``.

        Why expose this in addition to `get`:
            Standard `LoopCurriculum.get(step)` returns a single depth per
            step, so the model only ever trains at one depth at a time
            (incrementing as training progresses). That couples the model's
            performance to whichever depth the curriculum happens to be on.

            Sampling uniformly in ``[start_loops, get(step)]`` per batch
            forces the model to be robust *across* the range it has been
            curriculum-ramped to so far. The upper bound still tracks
            curriculum progress; it just samples below that bound on
            each call.

        Pass `--random-depth` on the training scripts to enable; otherwise
        the default behaviour is unchanged.

        Determinism:
            Pass a seeded `random.Random` instance for reproducible
            sampling. Without it we use the global `random` module —
            fine for production training, not for the test suite.
        """
        upper = self.get(step)
        if upper <= self.start_loops:
            return upper
        if rng is None:
            import random as _random
            return _random.randint(self.start_loops, upper)
        return rng.randint(self.start_loops, upper)


# ---------------------------------------------------------------------------
# Mixed multi-corpus dataset
# ---------------------------------------------------------------------------


# Per-corpus blending weights (must sum to 1.0). Tuned for a small-LLM
# pretraining run that wants reasonable code+math competence in addition to
# general web text. Adjust here, not via a flag — these ratios are a strong
# determinant of what the model can actually do and deserve to be reviewed.
_MIX_RATIOS = {
    "general": 0.40,
    "math":    0.40,
    "code":    0.20,
}


_DATASET_SPECS = [
    # (key, repo, config, split, text_field)
    ("general", "HuggingFaceFW/fineweb-edu",    "sample-10BT", "train", "text"),
    ("math",    "open-web-math/open-web-math",  None,          "train", "text"),
    ("code",    "codeparrot/codeparrot-clean",  None,          "train", "content"),
]


class MixedDataset(IterableDataset):
    """
    Interleaves three streaming HF datasets at fixed proportions.

    Sharding model
    --------------
    Identical to FineWebEduDataset: a `(rank, worker_id)` pair deterministically
    owns one slice of the global stream of each corpus. Cross-shard coordination
    is unnecessary — every rank reads a disjoint shard of every source.

    Per-step source selection
    -------------------------
    On every yielded chunk, we draw a source uniformly weighted by
    `_MIX_RATIOS`. This gives the empirical token mix the user's ratios
    *over many steps*, not exactly within a single batch — which is the
    correct interpretation for SGD-style mixing.

    Robustness
    ----------
    Any individual dataset failing to load (404, auth, network) is logged
    and skipped — the remaining sources are renormalised on the fly. This
    keeps training runnable when a single mirror is flaky.

    Yields:
        (input_ids, target_ids) — both `torch.long` of shape `(seq_len,)`,
        shifted by one for next-token prediction.
    """

    def __init__(
        self,
        encoding,
        seq_len: int,
        rank: int,
        world_size: int,
        mix_ratios: Optional[dict] = None,
        seed: int = 0,
    ):
        self.encoding = encoding
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.ratios = mix_ratios or _MIX_RATIOS
        self.seed = seed

    def _open_source(
        self,
        repo: str,
        config: Optional[str],
        split: str,
        total_shards: int,
        shard_index: int,
    ) -> Optional[Iterator]:
        """Open one streaming, sharded dataset. Returns None on failure."""
        from datasets import load_dataset

        try:
            ds = load_dataset(
                repo, name=config, split=split, streaming=True,
            ).shard(num_shards=total_shards, index=shard_index)
            return iter(ds)
        except Exception as exc:                                 # noqa: BLE001
            logger.warning(
                f"MixedDataset: skipping {repo!r} — failed to open ({exc})"
            )
            return None

    def _extract_text(self, sample: dict, field: Optional[str]) -> str:
        return sample.get(field, "") if field else ""

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        # Open each source independently so a one-stream restart doesn't
        # touch the others. We keep the spec alongside the iterator so we
        # can re-open exactly that source on StopIteration.
        active: list[dict] = []
        for key, repo, config, split, field in _DATASET_SPECS:
            if self.ratios.get(key, 0.0) <= 0:
                continue
            it = self._open_source(repo, config, split, total_shards, shard_index)
            if it is None:
                continue
            active.append({
                "key": key, "repo": repo, "config": config, "split": split,
                "field": field, "iter": it, "weight": self.ratios[key],
            })

        if not active:
            raise RuntimeError(
                "MixedDataset: no sources opened successfully. "
                "Check network access and HuggingFace dataset availability."
            )

        weights = torch.tensor([s["weight"] for s in active], dtype=torch.float)
        weights = (weights / weights.sum()).tolist()        # renormalise after drops
        rng = random.Random(self.seed + shard_index)

        buf: list[int] = []

        while True:
            idx = rng.choices(range(len(active)), weights=weights)[0]
            src = active[idx]

            try:
                sample = next(src["iter"])
            except StopIteration:
                # Re-open just this source; streaming datasets restart from
                # the shard's beginning. Acceptable at pretraining scale.
                new_iter = self._open_source(
                    src["repo"], src["config"], src["split"],
                    total_shards, shard_index,
                )
                if new_iter is None:
                    # Source has become unreachable; drop it from the rotation.
                    del active[idx]
                    if not active:
                        return
                    weights = [active[i]["weight"] for i in range(len(active))]
                    s = sum(weights)
                    weights = [w / s for w in weights]
                    continue
                src["iter"] = new_iter
                continue
            except Exception as exc:                                 # noqa: BLE001
                logger.warning(
                    f"MixedDataset: {src['key']} stream error ({exc}); skipping batch"
                )
                continue

            text = self._extract_text(sample, src["field"])
            if not text:
                continue

            buf.extend(self.encoding.encode(text))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


def combined_loss(
    model: nn.Module,
    logits: torch.Tensor,
    uncertainty: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    topk: int,
    lb_coeff: float = 1e-2,
    unc_coeff: float = 5e-2,
    depth_reg_coeff: float = 0.0,
) -> "tuple[torch.Tensor, dict]":
    """
    Single-call master objective combining:
        - cross-entropy on next-token prediction (the main task)
        - MoE load-balancing auxiliary loss
        - uncertainty-head calibration loss
        - depth regulariser (PonderNet × Ouro uniform-prior KL) — OFF
          BY DEFAULT (depth_reg_coeff=0). Pass `cfg.depth_reg_coeff` to
          turn it on; recommended 1e-3 to 1e-2 when enabling.

    Returns `(total_loss, metrics)` where `metrics` is a dict of float
    component values suitable for logging without re-syncing GPU state
    (each is `.item()`-d immediately).

    `topk` must match `cfg.n_experts_per_tok` so the load-balance loss
    measures the same routing fan-out the model actually uses.

    Note: consistency_loss is intentionally *not* folded in here because it
    runs an extra full forward pass and should be applied only on a fraction
    of training steps (every 4th, every 8th, etc.) — handled at the call
    site, not inside this helper.
    """
    ce = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
    )

    router_buf = collect_router_logits(model)
    lb = load_balance_loss(router_buf, topk=topk).to(ce.device)

    unc = uncertainty_calibration_loss(logits.detach(), uncertainty, targets)

    total = ce + lb_coeff * lb + unc_coeff * unc

    metrics = {
        "ce":  ce.detach().float().item(),
        "lb":  lb.detach().float().item(),
        "unc": unc.detach().float().item(),
    }

    # Depth regulariser — opt-in via depth_reg_coeff > 0. When off we
    # skip the collection walk + KL math entirely so the default
    # combined_loss has zero new cost vs. before this addition.
    if depth_reg_coeff > 0.0:
        depth = depth_regularization_loss(
            model, prior="uniform", coeff=1.0,
        ).to(ce.device)
        total = total + depth_reg_coeff * depth
        metrics["depth"] = depth.detach().float().item()
    else:
        metrics["depth"] = 0.0

    return total, metrics


# ---------------------------------------------------------------------------
# Spectral-radius diagnostic
# ---------------------------------------------------------------------------


def log_spectral_radius(model: nn.Module, step: int) -> None:
    """
    Walk the model and print ρ(A) of every LTIInjection encountered.

    A spectral radius drifting toward 1.0 across training signals that the
    stability guarantee (`A < 1` strictly) is degrading — usually because
    `log_dt + log_A` has been pushed against the lower clamp by aggressive
    gradients. Triggering this telemetry early lets us catch the failure
    before it manifests as loss instability.
    """
    from mythouro.main import LTIInjection

    for name, mod in model.named_modules():
        if isinstance(mod, LTIInjection):
            with torch.no_grad():
                A = mod.get_A()
                rho_max = A.max().item()
                rho_min = A.min().item()
            logger.info(
                f"step {step:>7d} | {name}: ρ(A) ∈ [{rho_min:.6f}, {rho_max:.6f}]"
            )


# ===========================================================================
# Part 2 training utilities
# ===========================================================================
#
# Each of the following targets a specific failure mode that the Part 1
# objectives don't address:
#
#   contrastive_loop_loss      — ACT halting often degenerates because every
#                                token receives the same "deep" answer
#                                regardless of difficulty. Easy tokens should
#                                produce small hidden-state deltas across
#                                loops, hard ones should produce large ones.
#   ProcessRewardHead +        — RLVR rewards only the final answer. In a
#   process_reward_loss          looped model each loop IS a reasoning step,
#                                so we can reward the quality of intermediate
#                                states without generating CoT tokens.
#   LoopDepthAnnealer          — depth-extrapolated reasoning works best when
#                                the model has been pushed beyond its
#                                training depth in the final phase of the run.
#   sparse_activation_loss     — load-balancing alone can produce a
#                                "balanced but indecisive" router. L1 on the
#                                routing probabilities pushes decisive expert
#                                selection.
#   ExpertSpecializationProbe  — provides interpretability and a weak
#                                supervision signal: predict the input
#                                domain (code / math / language / instruction)
#                                from per-expert routing probabilities.
#   build_fsdp_model           — HYBRID_SHARD FSDP for NVLink-paired clusters:
#                                shard within NVLink groups, replicate across.
# ===========================================================================


# ---------------------------------------------------------------------------
# Contrastive loop regularisation
# ---------------------------------------------------------------------------


def contrastive_loop_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    n_loops_low: int,
    n_loops_high: int,
    margin: float = 1.0,
    coeff: float = 0.05,
) -> torch.Tensor:
    """
    Force the ACT halting signal to discriminate easy vs hard tokens.

    Idea:
        Easy tokens (correctly predicted at shallow loops) should have
        small hidden-state distance between shallow and deep — they
        converge fast. Hard tokens (wrong at shallow) should have large
        distance — they're still being updated.

    Per token:
        easy:   penalise large ‖h_high - h_low‖
        hard:   penalise small ‖h_high - h_low‖ via a hinge with `margin`

    Implementation runs two forwards (shallow and deep) and reads the
    hidden state via the RecurrentBlock's forward output. Run sparingly
    (every N steps) because of the double forward cost.

    Returns a coefficient-scaled loss tensor (multiply ready).
    """
    hidden_buf: dict[str, torch.Tensor] = {}

    def _hook_factory(name):
        def _hook(_mod, _inp, out):
            hidden_buf[name] = out.detach() if name == "low" else out
        return _hook

    # Identify the recurrent block (works under FSDP / DDP / unwrapped)
    from mythouro.main import RecurrentBlock
    recurrent = None
    for mod in model.modules():
        if isinstance(mod, RecurrentBlock):
            recurrent = mod
            break
    if recurrent is None:
        return torch.tensor(0.0, device=input_ids.device)

    # ─── shallow pass (no grad on hidden, only logits used for easy/hard mask)
    handle = recurrent.register_forward_hook(_hook_factory("low"))
    with torch.no_grad():
        logits_low, _ = model(input_ids, n_loops=n_loops_low)
    handle.remove()
    h_low = hidden_buf.get("low")

    # ─── deep pass (grad-tracked through recurrent output)
    handle = recurrent.register_forward_hook(_hook_factory("high"))
    logits_high, _ = model(input_ids, n_loops=n_loops_high)
    handle.remove()
    h_high = hidden_buf.get("high")

    if h_low is None or h_high is None:
        return torch.tensor(0.0, device=input_ids.device)

    # RecurrentBlock returns hidden states including the n_sink prepended
    # positions; the logits / targets have been sink-stripped. Match the
    # token axis by trimming the leading sink positions from the hiddens.
    T = logits_low.shape[1]
    h_low = h_low[:, -T:]
    h_high = h_high[:, -T:]

    # Difficulty mask: shallow prediction correct ↔ easy
    with torch.no_grad():
        pred_low = logits_low.argmax(dim=-1)                # (B, T)
        easy = (pred_low == targets).float()
        hard = 1.0 - easy

    # Hidden-state distance per token
    dist = (h_high - h_low).norm(dim=-1)                    # (B, T)

    easy_loss = easy * dist                                  # large dist on easy = bad
    hard_loss = hard * F.relu(margin - dist)                 # small dist on hard = bad
    return coeff * (easy_loss + hard_loss).mean()


# ---------------------------------------------------------------------------
# Process reward head + loss
# ---------------------------------------------------------------------------


class ProcessRewardHead(nn.Module):
    """
    Small MLP head over the final hidden state predicting "will this answer be
    correct?". Trained with binary cross-entropy against realised correctness.

    At inference the sigmoid of its output is a scalar confidence per
    sequence — usable as a halting gate, a verification signal for
    speculative decoding, or input to an RL value baseline.

    Architecture: dim → dim/4 → dim/16 → 1, SiLU activations, zero-init
    output layer so the initial reward logit is 0 (probability 0.5) and
    training begins neutrally.
    """

    def __init__(self, dim: int):
        super().__init__()
        h1 = max(dim // 4, 32)
        h2 = max(dim // 16, 16)
        self.head = nn.Sequential(
            nn.Linear(dim, h1, bias=False),
            nn.SiLU(),
            nn.Linear(h1, h2, bias=False),
            nn.SiLU(),
            nn.Linear(h2, 1, bias=False),
        )
        nn.init.zeros_(self.head[-1].weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden : (B, T, dim) — mean-pooled over T to produce one reward
                     logit per sequence.
        Returns:
            (B,) reward logit (pre-sigmoid).
        """
        pooled = hidden.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def process_reward_loss(
    prm_head: ProcessRewardHead,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    coeff: float = 0.1,
) -> torch.Tensor:
    """
    BCE between PRM prediction and realised correctness on the batch.

    The "correctness label" is the per-sequence fraction of tokens the
    model predicted correctly — a continuous signal in [0, 1] rather
    than a binary right/wrong (BCEWithLogitsLoss handles soft targets
    correctly). The hidden state is taken at the pre-LM-head position
    (already-normed `normed` from MythOuro.forward).
    """
    with torch.no_grad():
        pred = logits.argmax(dim=-1)                         # (B, T)
        is_correct = (pred == targets).float().mean(dim=1)   # (B,) ∈ [0, 1]

    reward_logit = prm_head(hidden)                          # (B,)
    return coeff * F.binary_cross_entropy_with_logits(
        reward_logit, is_correct, reduction="mean"
    )


# ---------------------------------------------------------------------------
# Loop depth annealer
# ---------------------------------------------------------------------------


@dataclass
class LoopDepthAnnealer:
    """
    Linear ramp of effective n_loops *above* the training maximum in the
    final phase of training.

    Looped transformers depth-extrapolate: a model trained at N loops
    typically produces a usable answer at N+k loops, often a *better*
    one on hard problems. To make this property robust, the last phase
    of training is run at progressively deeper loop counts so the model
    must learn representations that work at extended depth.

    Schedule:
        step < anneal_start    → base_loops
        anneal_start ≤ step    → linear ramp toward max_extra_loops
        step ≥ total_steps     → max_extra_loops

    Use alongside LoopCurriculum: LoopCurriculum ramps shallow → base
    in the first half of training; LoopDepthAnnealer pushes base →
    extended in the final ~15%.
    """

    base_loops: int
    max_extra_loops: int
    anneal_start: int
    total_steps: int

    def get(self, step: int) -> int:
        if step < self.anneal_start:
            return self.base_loops
        denom = max(1, self.total_steps - self.anneal_start)
        frac = min(1.0, (step - self.anneal_start) / denom)
        return int(self.base_loops + frac * (self.max_extra_loops - self.base_loops))


# ---------------------------------------------------------------------------
# Sparse activation regularisation
# ---------------------------------------------------------------------------


def sparse_activation_loss(
    router_logits_buf: list[torch.Tensor],
    coeff: float = 1e-3,
) -> torch.Tensor:
    """
    Per-token entropy of MoE routing probabilities — encourages decisive
    (peaked) routing.

    Why not just L1? Earlier versions of this loss used `probs.abs().mean()`
    under the assumption that L1 would pull probabilities toward zero. For
    a softmax distribution the L1 norm is identically 1 (probabilities are
    non-negative and sum to 1), so the L1 mean is exactly 1/E for any
    routing pattern — perfectly uniform or perfectly collapsed alike. The
    "loss" had zero gradient and no effect on training.

    Entropy actually distinguishes the two regimes:
        H(uniform)   = log(E)   (the maximum)
        H(one-hot)   = 0        (the minimum)
    Minimising entropy pulls the router toward decisive (one-hot-ish)
    decisions while leaving the gating weights for those decisions free
    to differ across tokens.

    Complements `load_balance_loss`: load-balance prevents collapse
    (forces *which* expert wins to differ across tokens); sparse_activation
    encourages decisiveness (the winner takes more of the mass).
    """
    if not router_logits_buf:
        return torch.tensor(0.0)

    all_logits = torch.cat(router_logits_buf, dim=0)         # (sum_N, E)
    log_probs = F.log_softmax(all_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return coeff * entropy


# ---------------------------------------------------------------------------
# Expert specialisation probe
# ---------------------------------------------------------------------------


class ExpertSpecializationProbe(nn.Module):
    """
    Auxiliary classifier predicting input domain from per-expert routing.

    Trains an interpretability-and-supervision-signal pair: each expert
    gets a tiny classifier that, given the routing probability assigned
    to it, predicts the input domain (language / code / math / instruction).
    The loss propagates back through the router, encouraging the model
    to specialise its experts along recognisable axes.

    At inference, querying `predict_expert_domains()` returns the
    classifier-implied primary domain for each expert — useful for
    sanity-checking that specialisation actually emerged.

    Architecture is intentionally minimal (one Linear per expert from
    scalar routing prob → domain logits) because the signal should live
    in the routing decision itself, not in a deep classifier head.
    """

    DOMAIN_NAMES = ("language", "code", "math", "instruction")

    def __init__(self, n_experts: int, n_domains: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.n_domains = n_domains
        self.classifiers = nn.ModuleList([
            nn.Linear(1, n_domains, bias=True) for _ in range(n_experts)
        ])

    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            router_logits : (N, E)
        Returns:
            (N, E, n_domains) domain logits per (token, expert)
        """
        probs = F.softmax(router_logits, dim=-1)              # (N, E)
        return torch.stack(
            [self.classifiers[e](probs[:, e : e + 1]) for e in range(self.n_experts)],
            dim=1,
        )                                                      # (N, E, n_domains)

    def loss(
        self,
        router_logits_buf: list[torch.Tensor],
        domain_labels: torch.Tensor,        # (B,) or (sum_N,) integer
        coeff: float = 0.05,
    ) -> torch.Tensor:
        """
        Cross-entropy: each expert predicts the input's domain.

        domain_labels are typically per-document; we broadcast them along
        the token dimension to match the flattened routing logits.
        """
        if not router_logits_buf:
            return torch.tensor(0.0)

        all_logits = torch.cat(router_logits_buf, dim=0)      # (sum_N, E)
        N = all_logits.shape[0]

        if domain_labels.shape[0] != N:
            # Expand per-sequence labels to per-token by tiling.
            import math
            reps = math.ceil(N / domain_labels.shape[0])
            labels = domain_labels.repeat(reps)[:N]
        else:
            labels = domain_labels

        domain_logits = self(all_logits)                       # (N, E, D)
        total = torch.tensor(0.0, device=all_logits.device)
        for e in range(self.n_experts):
            total = total + F.cross_entropy(
                domain_logits[:, e, :], labels, reduction="mean"
            )
        return coeff * total / self.n_experts

    @torch.no_grad()
    def predict_expert_domains(self) -> dict[int, str]:
        """Per-expert primary domain implied by the classifier bias."""
        return {
            e: self.DOMAIN_NAMES[clf.bias.argmax().item() % len(self.DOMAIN_NAMES)]
            for e, clf in enumerate(self.classifiers)
        }


# Heuristic keyword-based domain labeller — good enough as a supervision
# signal for ExpertSpecializationProbe. Replace with a learned classifier
# for higher-quality labels.

_CODE_KW = ("def ", "class ", "import ", "```", "function", "var ", "const ")
_MATH_KW = ("theorem", "proof", "equation", "\\frac", "\\sum", "integral")
_INST_KW = ("###", "User:", "Assistant:", "Human:", "<|im_start|>", "<|user|>")


def get_domain_labels(texts: list, device: torch.device) -> torch.Tensor:
    """
    Cheap keyword-based per-document domain labeller.

    Domain index:
        0 = general language
        1 = code
        2 = math
        3 = instruction
    """
    labels = []
    for text in texts:
        if any(k in text for k in _CODE_KW):
            labels.append(1)
        elif any(k in text.lower() for k in _MATH_KW):
            labels.append(2)
        elif any(k in text for k in _INST_KW):
            labels.append(3)
        else:
            labels.append(0)
    return torch.tensor(labels, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# HYBRID_SHARD FSDP builder (NVLink-aware)
# ---------------------------------------------------------------------------


def build_fsdp_model(
    model: nn.Module,
    local_rank: int,
    amp_dtype: torch.dtype,
):
    """
    FSDP HYBRID_SHARD wrapping for NVLink-paired GPU setups.

    FULL_SHARD shards parameters across all GPUs globally — every gradient
    all-reduce crosses the inter-node fabric. HYBRID_SHARD shards within
    NVLink groups and replicates across groups, so within-group reductions
    use NVLink bandwidth (≈300 GB/s on V100 SXM2) and cross-group
    synchronisation only happens at gradient-averaging time.

    Group size is read from the `NVLINK_GROUP_SIZE` environment variable
    (default 2 — matches a typical NVLink pair topology). Falls back to
    FULL_SHARD on torch versions without HYBRID_SHARD.

    Returns the FSDP-wrapped model. Caller must construct the optimizer
    *after* this call so it tracks the sharded parameters.
    """
    import os
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    from mythouro.main import TransformerBlock, RecurrentBlock

    mp_policy = MixedPrecision(
        param_dtype=amp_dtype,
        reduce_dtype=amp_dtype,
        buffer_dtype=amp_dtype,
    )
    wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})

    # NVLink topology hint (used by the user / launcher to validate that
    # the group size matches the physical NVLink pair structure).
    group_size = int(os.environ.get("NVLINK_GROUP_SIZE", "2"))
    try:
        sharding = ShardingStrategy.HYBRID_SHARD
        logger.info(
            f"FSDP: HYBRID_SHARD with NVLink group size {group_size}"
        )
    except AttributeError:
        sharding = ShardingStrategy.FULL_SHARD
        logger.warning(
            "FSDP: HYBRID_SHARD unavailable on this torch version — "
            "falling back to FULL_SHARD."
        )

    return FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
    )


# ===========================================================================
# Knowledge distillation (Ouro-as-teacher pipeline)
# ===========================================================================
#
# Why this lives here:
#
# Ouro (Zhu et al. 2025) released 1.4B and 2.6B Looped-LM checkpoints trained
# on 7.7T tokens. Distilling MythOuro from Ouro lets us inherit that scale's
# benefits — the loop-allocation behaviour, the calibrated next-token
# distribution — at orders of magnitude less compute than from-scratch
# pretraining. The "RL only surfaces existing capacity" finding (Yue et al.
# 2024) sharpens this: the base model's ceiling is set by what the base
# distribution encodes, so starting from a higher ceiling matters more than
# adding post-training tricks.
#
# Two pieces ship here:
#   1. `distillation_loss` — temperature-scaled KL(teacher || student) plus
#      an optional CE term against gold labels (Hinton et al. 2015).
#   2. `load_distillation_teacher` — frozen `AutoModelForCausalLM` wrapper
#      that enforces the tokenizer-alignment precondition (logit distillation
#      across different vocabularies is mathematically meaningless).
#
# Hidden-state distillation (a project-of-student-hidden vs project-of-
# teacher-hidden L2) is intentionally NOT supported here — it would dilute
# the design surface for the small minority of cases where vocab alignment
# isn't feasible, and add a projection head to maintain. Caller should
# arrange vocab alignment instead (re-init MythOuro's embed/head against
# the teacher's tokenizer) when starting a distillation run.


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: "Optional[torch.Tensor]" = None,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
    divergence: str = "fwd_kl",
    jsd_beta: float = 0.5,
) -> "tuple[torch.Tensor, dict]":
    """
    Hinton-style knowledge distillation with an optional gold-label CE blend.

    Math:
        soft_loss = T² · KL( softmax(t/T) || softmax(s/T) )
        hard_loss = CE(s, targets)             # only if `targets` is given
        total     = alpha · soft_loss  +  (1 − alpha) · hard_loss

    The T² scaling factor restores the gradient magnitude of the soft loss
    to what it would be at T=1, which is the standard convention from the
    original Hinton paper. Without it, soft-loss gradients shrink as T²
    and the hard-loss term silently dominates.

    Args:
        student_logits  -- (B, T, V) — gradient flows through these.
        teacher_logits  -- (B, T, V) — assumed already detached / no-grad.
        targets         -- (B, T) gold labels for the hard-CE term. Pass
                            None to use pure distillation (alpha is
                            ignored in that case and we return only the
                            soft loss).
        temperature     -- softmax temperature for both teacher and student.
                            Higher T (~4) reveals more "dark knowledge" from
                            the teacher's secondary modes; T=1 collapses to
                            argmax-style. T=2 is a reasonable default.
        alpha           -- weight on the soft (distillation) term. 0.0 →
                            pure CE; 1.0 → pure distillation.
        ignore_index    -- CE ignore-index for padding / masked positions.

    Returns:
        (total_loss, metrics) where `metrics` has scalar floats:
            "soft" — the KL term (before alpha scaling, after T² scaling).
            "hard" — the CE term (or 0.0 if targets is None).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"distillation_loss: student / teacher logit shapes differ "
            f"({tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}). "
            "Both must share batch, sequence, and vocabulary dimensions — "
            "if they don't, the tokenisers are misaligned and logit-level "
            "distillation is mathematically meaningless."
        )

    T = float(temperature)
    if T <= 0:
        raise ValueError(f"temperature must be > 0; got {T}")

    # Compute in fp32 to keep the divergence stable under bf16/fp16 training.
    V = student_logits.shape[-1]
    s_log_probs = F.log_softmax(student_logits.float() / T, dim=-1)
    with torch.no_grad():
        t_log_probs = F.log_softmax(teacher_logits.float() / T, dim=-1)
        t_probs = t_log_probs.exp()

    # Per-position divergence between teacher p_T and student p_S:
    #   fwd_kl = KL(p_T ‖ p_S)  — Hinton, mode-COVERING (the original default).
    #            Identical to the previous F.kl_div(s_log_probs, t_probs).
    #   rev_kl = KL(p_S ‖ p_T)  — MiniLLM, mode-SEEKING: concentrates on the
    #            teacher's dominant modes, puts little mass on its "void
    #            regions" → less degenerate generation for a small student
    #            distilling a much larger teacher. arXiv 2306.08543.
    #   jsd(β) = β·KL(p_T‖M) + (1−β)·KL(p_S‖M),  M = β·p_T + (1−β)·p_S — the
    #            interpolation (β→0 ≈ fwd, β→1 ≈ rev). GKD, arXiv 2306.13649.
    # rev_kl / jsd flow gradient through p_S in BOTH the probs and log-probs;
    # fwd_kl only needs the log-probs (teacher side is detached above).
    if divergence == "fwd_kl":
        div_rows = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
    elif divergence == "rev_kl":
        s_probs = s_log_probs.exp()
        div_rows = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
    elif divergence == "jsd":
        b = float(jsd_beta)
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"jsd_beta must be in [0, 1]; got {b}")
        s_probs = s_log_probs.exp()
        m_log = (b * t_probs + (1.0 - b) * s_probs).clamp_min(1e-9).log()
        div_rows = (
            b * (t_probs * (t_log_probs - m_log)).sum(dim=-1)
            + (1.0 - b) * (s_probs * (s_log_probs - m_log)).sum(dim=-1)
        )
    else:
        raise ValueError(
            f"divergence must be 'fwd_kl', 'rev_kl', or 'jsd'; got {divergence!r}"
        )
    div_rows = div_rows.view(-1)                        # (B·T,) per-position

    # P1.9: the hard CE respects ignore_index but the soft term previously
    # averaged over ALL positions — harmless on packed distillation data (no
    # padding) but a silent footgun for blended phases with padded/masked rows.
    # Mask the divergence to the same positions the CE trains on.
    if targets is not None:
        valid = (targets.view(-1) != ignore_index)
        n_valid = int(valid.sum())
        if n_valid == 0:
            soft = div_rows.sum() * 0.0                 # keep graph, zero loss
        else:
            soft = div_rows[valid].mean() * (T * T)
    else:
        soft = div_rows.mean() * (T * T)

    if targets is None:
        return alpha * soft, {"soft": float(soft.item()), "hard": 0.0}

    hard = F.cross_entropy(
        student_logits.view(-1, V),
        targets.view(-1),
        ignore_index=ignore_index,
    )
    total = alpha * soft + (1.0 - alpha) * hard
    return total, {"soft": float(soft.item()), "hard": float(hard.item())}


# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------


def load_distillation_teacher(
    model_id: str,
    student_vocab_size: int,
    *,
    device: "str | torch.device" = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
):
    """
    Load a frozen `AutoModelForCausalLM` to use as a distillation teacher.

    Enforces three preconditions before returning:

        1. The teacher's `vocab_size` equals the student's. Logit
           distillation across mismatched vocabularies is not a thing —
           the per-position softmax is over different supports.
        2. All teacher parameters have `requires_grad = False`. We never
           want to backpropagate into the teacher.
        3. The teacher is moved to `device` in `dtype`. bf16 is the right
           default for inference on Ampere+ / Blackwell; halves the VRAM
           cost vs fp32 with no accuracy loss for forward-only use.

    On failure to load (network, auth, model id typo) the function logs
    the error and returns `None`. Callers should treat `None` as "skip
    distillation; fall back to pure CE" rather than crash.

    Args:
        model_id            -- HuggingFace model id (e.g. "ouro-llm/Ouro-1.4B")
                                or a local path.
        student_vocab_size  -- vocab dim of the MythOuro student, used to
                                enforce the tokenizer-alignment precondition.
        device              -- where to load the teacher's weights.
        dtype               -- inference dtype; bf16 is the right default.
        trust_remote_code   -- forwarded to `AutoModelForCausalLM`. Set True
                                only if the teacher repo has custom modeling
                                code (Ouro probably does — it's a non-
                                standard architecture).

    Returns:
        A `nn.Module` with `.forward(input_ids) → logits` semantics, OR
        `None` on load failure.
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error(
            "load_distillation_teacher: `transformers` is required; "
            "install with `pip install transformers`"
        )
        return None

    try:
        from transformers import AutoConfig

        # Load the config first and backfill `pad_token_id` if the teacher omits
        # it. Some Ouro configs (notably the *base* Ouro-2.6B's `OuroConfig`) do
        # not set it, which trips loading with "'OuroConfig' object has no
        # attribute 'pad_token_id'". Forward-only distillation never pads or
        # generates, so any valid id is fine — default to eos (or 0).
        cfg = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        if getattr(cfg, "pad_token_id", None) is None:
            eos = getattr(cfg, "eos_token_id", None)
            if isinstance(eos, (list, tuple)):
                eos = eos[0] if eos else None
            cfg.pad_token_id = eos if eos is not None else 0

        teacher = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=cfg,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:                                # noqa: BLE001
        logger.error(
            f"load_distillation_teacher: failed to load {model_id!r} ({exc}). "
            "Distillation will be disabled for this run."
        )
        return None

    # Precondition 1: vocab alignment.
    teacher_vocab = getattr(teacher.config, "vocab_size", None)
    if teacher_vocab != student_vocab_size:
        logger.error(
            f"load_distillation_teacher: vocab mismatch — teacher={teacher_vocab} "
            f"student={student_vocab_size}. Logit distillation requires "
            "matched tokenizers. Either re-init the student with the teacher's "
            "tokenizer, or skip distillation."
        )
        return None

    # Precondition 2: freeze every parameter.
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # Precondition 3: move to device + dtype.
    teacher = teacher.to(device=device, dtype=dtype)
    logger.success(
        f"load_distillation_teacher: loaded {model_id} (vocab={teacher_vocab}, "
        f"dtype={dtype}, device={device}). Teacher frozen."
    )
    return teacher


# ===========================================================================
# Depth regulariser (PonderNet × Ouro)
# ===========================================================================
#
# Pulls each RecurrentBlock's per-token halt distribution P(halt at step n)
# toward a prior — uniform by default, matching Ouro's empirical finding
# that a geometric prior (PonderNet's original choice) under-trains the
# late loops and that uniform is better at LLM scale.
#
# The distribution itself is computed inside RecurrentBlock.forward and
# exposed as `last_halt_distribution` (B, T, K) where K is the number of
# loops that actually ran. The math there:
#
#     P(halt at step n) = λ_n · ∏_{i<n}(1 − λ_i)
#     P(halt at step K) absorbs any residual mass        (force normalisation)
#
# This module just walks the model, collects those distributions, and
# computes KL(P || prior). Disabled by default (cfg.depth_reg_coeff=0).


def collect_halt_distributions(model: nn.Module) -> list[torch.Tensor]:
    """
    Gather `last_halt_distribution` from every RecurrentBlock in the model.

    Each entry is shape (B, T, K) with K = number of loops actually run on
    the latest forward (may differ from `cfg.max_loop_iters` when ACT
    short-circuits or convergence-detection fires). Returns an empty list
    if no RecurrentBlock has been forwarded yet on the current step.
    """
    from mythouro.main import RecurrentBlock

    out: list[torch.Tensor] = []
    for mod in model.modules():
        if isinstance(mod, RecurrentBlock):
            dist = getattr(mod, "last_halt_distribution", None)
            if dist is not None:
                out.append(dist)
    return out


def depth_regularization_loss(
    model: nn.Module,
    *,
    prior: str = "uniform",
    coeff: float = 1e-2,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    KL divergence between each block's halt distribution and a prior.

    Args:
        model  : (possibly FSDP-wrapped) model — walked for RecurrentBlocks.
        prior  : currently only `"uniform"` is supported. Geometric prior
                 (PonderNet's original) is intentionally not implemented;
                 Ouro's evidence is that uniform outperforms geometric at
                 LLM scale, and we don't want to ship an option whose
                 use we haven't yet validated.
        coeff  : multiplicative coefficient on the loss. The caller scales
                 it again via `cfg.depth_reg_coeff` when blending into
                 `combined_loss`; `coeff` here lets you debug with a
                 stand-alone larger weight without changing the cfg.
        eps    : log-stability floor for the KL clamp.

    Returns: scalar Tensor with `coeff` already applied. Returns 0 (no
    grad) when no halt distributions are present (the model hasn't been
    forwarded yet, or all RecurrentBlocks short-circuited before the
    first ACT call).
    """
    import math

    dists = collect_halt_distributions(model)
    if not dists:
        return torch.tensor(0.0)

    if prior != "uniform":
        raise NotImplementedError(
            f"depth_regularization_loss: prior={prior!r} not implemented. "
            "Only 'uniform' is supported (matches Ouro's empirical choice; "
            "geometric prior under-trains late loops per their ablations)."
        )

    total = torch.zeros((), device=dists[0].device, dtype=dists[0].dtype)
    for dist in dists:
        # dist: (B, T, K) — already a proper distribution per (B, T) row.
        K = dist.shape[-1]
        # KL(P || U) = Σ_k P_k · log(K · P_k)
        #            = Σ_k P_k · log P_k  +  log(K)
        log_p = torch.log(dist.clamp_min(eps))
        kl_per_token = (dist * log_p).sum(dim=-1) + math.log(K)
        # Mean over batch + sequence positions. Each position contributes
        # equally; the depth distribution is conceptually independent
        # across tokens so we average rather than sum.
        total = total + kl_per_token.mean()

    return coeff * total / len(dists)


def teacher_logits(
    teacher: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Run the frozen teacher to get next-token logits. Wrapper handles
    several real-world HF causal-LM quirks observed against the Ouro
    teacher and similar custom-modeling models:

    * Return type can be `CausalLMOutputWithPast` (has `.logits`) or a
      bare Tensor.
    * Some custom modeling code crashes during forward when called
      without `model.generate`'s kwargs in place — specifically the
      "cache_position is an int" failure in Ouro's `get_mask_sizes`.
      Passing ``use_cache=False`` + ``past_key_values=None`` keeps us
      on the no-cache code path that all HF causal LMs share.
    * Cross-device setups (teacher on CPU, student on CUDA — common
      when the teacher is too big to cohabit on a single GPU): we
      move `input_ids` to the teacher's device before the forward.
      The training loop is expected to `.to(student_device)` on the
      returned logits, so we don't move them back here.
    * Gradients must never flow through the teacher; we wrap the call
      in ``no_grad`` and detach the result so a caller that forgets
      can't accidentally update the frozen teacher.
    """
    # Find the teacher's device. `next(teacher.parameters()).device` is
    # the canonical way; works for nn.Module and parameter-bearing
    # custom-arch models. Stub teachers in tests have parameters too.
    try:
        teacher_device = next(teacher.parameters()).device
    except StopIteration:
        teacher_device = input_ids.device          # parameterless stub

    if input_ids.device != teacher_device:
        input_ids = input_ids.to(teacher_device)

    with torch.no_grad():
        try:
            out = teacher(
                input_ids=input_ids,
                use_cache=False,
                past_key_values=None,
            )
        except TypeError:
            # Some stubs (used in tests) don't accept the extra kwargs.
            out = teacher(input_ids=input_ids)
        if hasattr(out, "logits"):
            return out.logits.detach()
        return out.detach()
