"""
Device abstraction for CUDA / Intel XPU / CPU.

`torch.xpu` (native since PyTorch 2.5, no IPEX needed) mirrors the `torch.cuda`
API — `is_available`, `synchronize`, `is_bf16_supported`, `get_device_capability`,
`device_count`, RNG state, etc. (verified against the 2.12 docs). This module maps
the handful of accelerator calls the training / eval / bench scripts make across
all three backends, so `--device xpu` runs on Intel Arc / Battlemage (e.g. Arc
Pro B70) with no other code changes. See docs/roadmap.md (Intel B70 / XPU port).

Autocast is the generic `torch.amp.autocast(device_type=...)` — under `torch.amp`,
not `torch.<backend>` — so `autocast_type()` just returns the backend name.
"""

from __future__ import annotations

import torch


def backend(device: str) -> str:
    """'cuda' | 'xpu' | 'cpu' from a device string ('cuda:0' / 'xpu' / 'cpu')."""
    d = str(device).lower()
    if d.startswith("cuda"):
        return "cuda"
    if d.startswith("xpu"):
        return "xpu"
    return "cpu"


def _mod(device: str):
    """The torch accelerator submodule for `device`, or None for cpu / missing."""
    b = backend(device)
    return getattr(torch, b, None) if b != "cpu" else None


def pick_device(prefer: "str | None" = None) -> str:
    """Resolve a device: explicit `prefer`, else cuda:0, else xpu, else cpu."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda:0"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return "cpu"


def is_accelerator(device: str) -> bool:
    """True for cuda / xpu (i.e. autocast applies); False for cpu."""
    return backend(device) in ("cuda", "xpu")


def autocast_type(device: str) -> str:
    """`device_type` arg for torch.amp.autocast: 'cuda' | 'xpu' | 'cpu'."""
    return backend(device)


def is_available(device: str) -> bool:
    m = _mod(device)
    try:
        return bool(m is not None and m.is_available())
    except Exception:                                            # noqa: BLE001
        return False


def bf16_supported(device: str) -> bool:
    m = _mod(device)
    try:
        return bool(m is not None and m.is_available() and m.is_bf16_supported())
    except Exception:                                            # noqa: BLE001
        return False


def fused_adam_supported(device: str) -> bool:
    """torch's fused AdamW kernel is CUDA-only; off for xpu/cpu."""
    return backend(device) == "cuda"


def device_count(device: str) -> int:
    m = _mod(device)
    try:
        return int(m.device_count()) if m is not None else 0
    except Exception:                                            # noqa: BLE001
        return 0


def synchronize(device: str) -> None:
    """Block until all kernels on `device` finish (no-op on cpu)."""
    m = _mod(device)
    if m is not None:
        try:
            m.synchronize()
        except Exception:                                        # noqa: BLE001
            pass
