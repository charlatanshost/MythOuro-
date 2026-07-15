"""
Batched on-policy rollout serving (docs/onpolicy_plan.md phase 5).

The Max 1100 (and latency-bound accelerators generally) only reach their
throughput wide: batch-1 sequential decode is their worst workload, and
that's exactly what per-micro-step inline rollout generation was. The fix
is to decouple GENERATION batch from TRAINING micro-batch:

  * generate rollouts in one wide `generate_rollout` call
    (`rollout_batch` sequences at once — the decode cost is amortised),
  * store them in a `RolloutBuffer`,
  * serve `micro_batch`-sized slices to the training loop's on-policy
    branch, letting each stored rollout be consumed a bounded number of
    times (`reuse`) before a forced refill.

Reuse is deliberate mild off-policyness: GKD-style distillation tolerates
slightly stale self-generated data, and each reuse multiplies the
on-policy dose obtained per decode-second. Staleness is capped twice —
by the reuse budget and by `max_age_steps` optimizer steps — so the data
can never drift far behind the current policy.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from loguru import logger


class RolloutBuffer:
    """
    Fixed-size store of student-generated rollouts, sliced out per micro-step.

    Lifecycle: `needs_refill()` → (caller generates a wide batch) → `fill()`
    → repeated `draw()` until the reuse budget or age cap trips
    `needs_refill()` again. The buffer never generates anything itself; it
    only tracks freshness, so the caller keeps full control of model / RNG /
    autocast context at the generation site.
    """

    def __init__(
        self,
        rollout_batch: int,
        micro_batch: int,
        *,
        reuse: int = 2,
        max_age_steps: int = 50,
    ) -> None:
        if rollout_batch < micro_batch:
            raise ValueError(
                f"rollout_batch ({rollout_batch}) must be >= micro_batch "
                f"({micro_batch})"
            )
        if rollout_batch % micro_batch:
            adjusted = (rollout_batch // micro_batch) * micro_batch
            logger.warning(
                f"RolloutBuffer: rollout_batch {rollout_batch} is not a "
                f"multiple of micro_batch {micro_batch}; using {adjusted}."
            )
            rollout_batch = adjusted
        self.rollout_batch = rollout_batch
        self.micro_batch = micro_batch
        self.reuse = max(1, reuse)
        self.max_age_steps = max_age_steps

        self._data: Optional[torch.Tensor] = None      # (rollout_batch, L)
        self._cursor = 0
        self._draws_left = 0
        self._birth_step = -1

    def needs_refill(self, current_step: int) -> bool:
        """True when empty, reuse budget spent, or older than max_age_steps."""
        if self._data is None or self._draws_left <= 0:
            return True
        return (current_step - self._birth_step) >= self.max_age_steps

    def fill(self, rollouts: torch.Tensor, current_step: int) -> None:
        """Store a fresh (B, L) token batch; resets cursor, budget, and age."""
        if rollouts.shape[0] < self.rollout_batch:
            # Tolerate a short fill (e.g. tail of a dataset) — just scale the
            # reuse budget to the rows actually available.
            logger.warning(
                f"RolloutBuffer.fill: got {rollouts.shape[0]} rows, expected "
                f"{self.rollout_batch}; serving what we have."
            )
        self._data = rollouts.detach()
        self._cursor = 0
        self._draws_left = self.reuse * max(
            1, self._data.shape[0] // self.micro_batch
        )
        self._birth_step = current_step

    def draw(self) -> torch.Tensor:
        """Next (micro_batch, L) slice, wrapping around the stored batch."""
        if self._data is None or self._draws_left <= 0:
            raise RuntimeError(
                "RolloutBuffer.draw() on an empty/exhausted buffer — call "
                "needs_refill()/fill() first."
            )
        B = self._data.shape[0]
        start, end = self._cursor, self._cursor + self.micro_batch
        if end <= B:
            out = self._data[start:end]
            self._cursor = end % B
        else:                                   # wrap around the store
            out = torch.cat([self._data[start:], self._data[: end - B]], dim=0)
            self._cursor = end - B
        self._draws_left -= 1
        return out


def rollout_with_retry(fn: Callable, *args, **kwargs):
    """
    Run a wide rollout call, retrying ONCE on RuntimeError.

    A shape/timing-dependent abort was observed once at rollout start on XPU
    (2026-07-12; the identical rerun succeeded). A hard segfault can't be
    caught from Python, but op-level failures surface as RuntimeError — for
    those, one retry with identical inputs is cheap insurance on overnight
    runs. A second consecutive failure is real and re-raises.
    """
    try:
        return fn(*args, **kwargs)
    except RuntimeError as exc:
        logger.warning(f"rollout generation failed ({exc}); retrying once.")
        return fn(*args, **kwargs)
