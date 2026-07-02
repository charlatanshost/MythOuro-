"""
Checkpoint utilities for MythOuro training (§4 from AGENT_TASKS.md).

What's here
-----------
* `save_checkpoint` / `load_checkpoint` — full state round-trip including
  model, optimizer, RNG, optional GradScaler, and the cfg used at save.
* `CHECKPOINT_VERSION` — schema version bumped on every incompatible
  change to the on-disk format.
* `_check_cfg_compat` — shape-fields-must-match guard; benign drift
  (LR, ratios, dropout) is allowed so staged training can re-tune them.
* `ShutdownHandler` — cooperative SIGINT / SIGTERM / SIGBREAK handler.
  Sets a flag instead of killing the process so the training loop can
  flush a final checkpoint at a safe point.

Why this is its own module
--------------------------
Originally the helpers lived inside `training/3b_fine_web_edu.py`. That
script's top-level imports pull in `datasets` → `pandas`, which on
Python 3.14 + Windows segfaults during test collection. Splitting the
checkpoint code out lets the tests import a clean, lightweight module.

FSDP semantics
--------------
Under DDP/FSDP, the save path gathers a `FULL_STATE_DICT` on rank 0
only; non-master ranks participate in the gather collective but exit
before writing. The load path mirrors that contract — every rank reads
the file so its state is materialised on each rank's shards.

The RNG round-trip is *single-process only* for now. Under DDP, each
rank has its own RNG state; restoring rank-0's RNG on every rank would
desync them. Pass `restore_rng=False` (or accept the default behaviour
under `ddp=True`) to skip RNG restore in distributed runs.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Optional

import torch
from loguru import logger


# Bump when the on-disk schema changes incompatibly (new required keys,
# renamed sections, semantically different state). The loader compares
# this against the file's recorded version and refuses to resume on
# mismatch rather than silently corrupt state.
#   v1 — model + optimizer + cfg + vocab_size + step
#   v2 — adds rng_state, scaler_state, cfg_dict, checkpoint_version, extra
CHECKPOINT_VERSION = 2


# Fields whose change between save and resume invalidates the checkpoint
# because the parameter shapes themselves depend on them. Other fields
# (LR, dropout, ratios, eval cadences) are intentionally allowed to
# change between stages.
_SHAPE_FIELDS = (
    "vocab_size", "dim", "n_heads", "n_kv_heads",
    "prelude_layers", "coda_layers", "attn_type",
    "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim",
    "qk_nope_head_dim", "v_head_dim",
    "n_experts", "n_shared_experts", "n_experts_per_tok", "expert_dim",
    "lora_rank", "n_sink_tokens", "max_loop_iters",
)
# NOTE: `max_seq_len` deliberately omitted. It's a runtime upper bound on
# input length, not a weight-shape parameter. RoPE frequencies are computed
# on the fly per forward (see `precompute_freqs_cis` callers), so a
# checkpoint trained at max_seq_len=512 loads cleanly into a model built
# with max_seq_len=1024 (or any other value). If the model.load_state_dict
# ever surfaces a real buffer-shape mismatch tied to max_seq_len, it will
# fail loudly with a clearer error than the cfg-compat check provides.


# ---------------------------------------------------------------------------
# RNG state helpers
# ---------------------------------------------------------------------------


def _capture_rng_state() -> dict:
    """
    Snapshot all RNG sources we touch during training so a resume reproduces
    the same data stream + dropout / sampling decisions from the same step
    onward.

    Single-process only. Multi-rank RNG round-tripping is intentionally
    deferred — see module docstring.
    """
    state = {
        "torch_cpu": torch.get_rng_state(),
        "python":    __import__("random").getstate(),
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """Inverse of `_capture_rng_state`. Missing keys are tolerated so
    checkpoints written on a CUDA box can be resumed on CPU."""
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "python" in state:
        __import__("random").setstate(state["python"])
    if "numpy" in state:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except RuntimeError as exc:
            # Different GPU count on resume — log and continue.
            logger.warning(f"Could not restore CUDA RNG state ({exc})")


# ---------------------------------------------------------------------------
# Config compatibility
# ---------------------------------------------------------------------------


def _cfg_to_dict(cfg) -> dict:
    """
    Flatten a dataclass-style cfg into a plain dict so the saved
    representation survives changes to the dataclass definition.
    """
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(vars(cfg))


def _check_cfg_compat(saved: dict, current_cfg) -> None:
    """
    Raise on shape-affecting mismatches between the checkpoint's cfg and
    the current cfg. Logs (but does not fail) on benign drift.

    Staged training requires shape-fixed; LR / curriculum / loss
    coefficients are free to change between stages.
    """
    current = _cfg_to_dict(current_cfg)
    incompatible: list[tuple[str, object, object]] = []
    benign: list[tuple[str, object, object]] = []
    for k, sv in saved.items():
        if k not in current:
            continue
        cv = current[k]
        if sv == cv:
            continue
        if k in _SHAPE_FIELDS:
            incompatible.append((k, sv, cv))
        else:
            benign.append((k, sv, cv))
    if incompatible:
        msg = "; ".join(f"{k}: saved={sv!r} current={cv!r}" for k, sv, cv in incompatible)
        raise RuntimeError(
            "Checkpoint cfg is shape-incompatible with current cfg — "
            "weights cannot be loaded. Mismatched fields: " + msg
        )
    for k, sv, cv in benign:
        logger.info(f"resume: cfg change {k}: {sv!r} → {cv!r} (allowed)")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def list_ckpts(ckpt_dir: str) -> list[str]:
    """
    Return checkpoint paths in `ckpt_dir` sorted oldest → newest.

    Relies on the zero-padded `step_{0000000}.pt` filename convention so
    lexicographic sort matches chronological order. Changing the filename
    format elsewhere without updating the pad width would silently break
    both `keep_last` pruning and resume-latest on startup.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
    scaler=None,
    extra: Optional[dict] = None,
) -> None:
    """
    Gather full model + optimizer state, write atomically, prune old files.

    Atomic write: temp file + `os.replace` so a kill mid-save leaves
    the previous checkpoint intact instead of a truncated `.pt`.

    Args:
        model       -- FSDP-wrapped (ddp=True) or raw (ddp=False) model
        optimizer   -- optimizer whose state must round-trip with the model
        step        -- global step number; encoded zero-padded into the filename
        cfg         -- model config dataclass; persisted both as-is (legacy)
                       and as a plain dict (for compatibility checks)
        vocab_size  -- saved for sanity-check against a future tokenizer
        ckpt_dir    -- directory to write into; created if missing
        ddp         -- True if FSDP path; False for single-GPU / CPU
        master      -- whether this rank writes to disk (rank 0 only)
        keep_last   -- number of most-recent checkpoints to retain
        scaler      -- optional `torch.amp.GradScaler` whose state must
                       round-trip when training with fp16. Ignored if None.
        extra       -- optional dict of arbitrary picklable side-state
    """
    if ddp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": step,
        "model": model_state,
        "optimizer": optim_state,
        "cfg": cfg,                              # legacy / convenience
        "cfg_dict": _cfg_to_dict(cfg),           # canonical for resume checks
        "vocab_size": vocab_size,
        "rng_state": _capture_rng_state(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)

    for old in list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved → {final_path}")


_ALLOWED_MISSING = {"freqs_cis", "freqs_cis_mla"}


def _assert_clean_state_load(missing, unexpected) -> None:
    """
    Guard against silent partial loads (P1, 2026-07-01).

    `strict=False` is required only to tolerate the RoPE `freqs_cis` /
    `freqs_cis_mla` buffers we intentionally drop above (they're re-sized for
    the current max_seq_len). ANY other missing key means a renamed module,
    architecture-flag mismatch, or truncated file silently loaded random-init
    weights for that piece; ANY unexpected key means the file carries params
    the model lacks. Both are silent corruption without this check — and it
    also catches a botched fold migration (tools/fold_lora_scale.py).
    """
    extra_missing = [k for k in missing if k not in _ALLOWED_MISSING]
    if extra_missing or list(unexpected):
        raise RuntimeError(
            "load_checkpoint: unclean state-dict load — "
            f"unexpected={list(unexpected)}, "
            f"missing (beyond RoPE freqs)={extra_missing}. Usually an "
            "architecture/flag mismatch between the checkpoint and the model "
            "config; refusing to train on a silently partial load."
        )


def load_checkpoint(
    model,
    optimizer,
    path: str,
    ddp: bool,
    *,
    current_cfg=None,
    scaler=None,
    restore_rng: bool = True,
    allow_version_mismatch: bool = False,
) -> "tuple[int, dict]":
    """
    Restore model + optimizer (+ optionally RNG + scaler) from disk.

    Returns:
        (step, extra) tuple. `extra` is the dict the caller stored at save
        time (empty dict if absent).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Version + cfg compatibility checks. Run before touching model state so
    # an incompatible file fails loudly with an empty model rather than
    # half-loading and crashing mid-train.
    saved_version = int(ckpt.get("checkpoint_version", 1))
    if saved_version != CHECKPOINT_VERSION:
        msg = (f"Checkpoint version mismatch: file={saved_version} "
               f"current={CHECKPOINT_VERSION}")
        if allow_version_mismatch:
            logger.warning(msg + " — proceeding because allow_version_mismatch=True")
        else:
            raise RuntimeError(
                msg + " — pass allow_version_mismatch=True to override "
                "or migrate the checkpoint manually."
            )

    if current_cfg is not None and "cfg_dict" in ckpt:
        _check_cfg_compat(ckpt["cfg_dict"], current_cfg)

    # Optimizer state is treated as optional. `tools/grow_checkpoint.py`
    # writes an empty dict here because the source optimizer's tensor
    # shapes don't match the promoted model — the caller MUST build a
    # fresh optimizer in that case. We detect "no usable state" by
    # checking for the `param_groups` key that every real optimizer
    # state dict has, and skip the load if it's missing or the dict is
    # empty. The freshly-constructed optimizer passed in keeps its
    # default (zero-step) state, which is exactly what a grown
    # checkpoint wants.
    optim_state = ckpt.get("optimizer") or {}
    has_optim_state = bool(optim_state) and "param_groups" in optim_state

    # RoPE precomputed frequency buffers (`freqs_cis`, `freqs_cis_mla`)
    # are sized to `max_seq_len + 4`. They're deterministic functions of
    # cfg — not learned parameters — so when max_seq_len drifts between
    # checkpoint save and reload, the saved buffer has a different shape
    # than the freshly-built model expects. The fresh model's already-
    # constructed buffers are *correct* for the new cfg, so we simply drop
    # the saved versions and let the fresh ones stay in place.
    #
    # Anything OTHER than these RoPE buffers would still be caught by
    # `load_state_dict` with strict=True (the default), so this is a
    # narrow, named exception rather than a blanket strict=False.
    model_state = dict(ckpt["model"])
    _SAFE_RECOMPUTE_BUFFERS = ("freqs_cis", "freqs_cis_mla")
    for key in _SAFE_RECOMPUTE_BUFFERS:
        if key in model_state:
            model_state.pop(key)
            logger.info(
                f"load_checkpoint: dropping saved buffer '{key}' from state "
                f"dict; the freshly-built model has it sized for the current "
                f"max_seq_len ({getattr(current_cfg, 'max_seq_len', '?')})."
            )

    if ddp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            _missing, _unexpected = model.load_state_dict(model_state, strict=False)
            _assert_clean_state_load(_missing, _unexpected)
            if has_optim_state:
                fsdp_state = FSDP.optim_state_dict_to_load(
                    model=model,
                    optim=optimizer,
                    optim_state_dict=optim_state,
                )
                optimizer.load_state_dict(fsdp_state)
            else:
                logger.info(
                    "load_checkpoint: no optimizer state present in file — "
                    "keeping the fresh optimizer. This is expected for "
                    "grown / promoted checkpoints."
                )
    else:
        _missing, _unexpected = model.load_state_dict(model_state, strict=False)
        _assert_clean_state_load(_missing, _unexpected)
        if has_optim_state:
            optimizer.load_state_dict(optim_state)
        else:
            logger.info(
                "load_checkpoint: no optimizer state present in file — "
                "keeping the fresh optimizer. This is expected for "
                "grown / promoted checkpoints."
            )

    # GradScaler: restore only if the caller supplied one AND the file
    # contains state. Either side missing is a benign no-op (bf16 paths
    # never need a scaler).
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    # RNG: only restore in single-process mode by default — see module docstring.
    if restore_rng and not ddp and "rng_state" in ckpt:
        _restore_rng_state(ckpt["rng_state"])

    return int(ckpt["step"]), ckpt.get("extra", {})


# ---------------------------------------------------------------------------
# Cooperative shutdown
# ---------------------------------------------------------------------------


class ShutdownHandler:
    """
    Cooperative interrupt handler for long-running training jobs.

    Installs SIGINT, SIGTERM, and (Windows) SIGBREAK handlers that set
    a flag instead of immediately killing the process. The training
    loop polls `handler.requested` after each step and, when set, flushes
    a final checkpoint and exits cleanly.

    Why cooperative: Ctrl+C mid-`optimizer.step()` under FSDP frequently
    leaves param shards inconsistent, and the half-written `.pt.tmp`
    from a partial save confuses the next resume. A flag-and-poll
    design gives a deterministic save-then-exit window.

    Sending the same signal twice forces an immediate `KeyboardInterrupt`
    for the case where the loop is stuck and you actually want to kill it.
    """

    def __init__(self):
        self.requested: bool = False
        self._signal_name: Optional[str] = None
        self._installed: list[int] = []

    def install(self) -> None:
        """Register handlers for SIGINT, SIGTERM, (and SIGBREAK on Windows)."""
        import signal
        targets = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            targets.append(signal.SIGBREAK)
        for sig in targets:
            try:
                signal.signal(sig, self._on_signal)
                self._installed.append(sig)
            except (ValueError, OSError):
                # `signal()` only works in the main thread on some platforms;
                # under torchrun's worker threads it can raise. Best effort.
                pass

    def _on_signal(self, signum, _frame):
        import signal
        try:
            self._signal_name = signal.Signals(signum).name
        except ValueError:
            self._signal_name = str(signum)
        if self.requested:
            raise KeyboardInterrupt(
                f"Received second {self._signal_name} — forcing exit."
            )
        self.requested = True
        logger.warning(
            f"Received {self._signal_name} — will flush checkpoint and exit "
            "after the current step. Send again to force-exit."
        )
