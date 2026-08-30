#!/usr/bin/env python3
"""
OPTIMISATION A — teacher-logit reuse cache for the on-policy path.

MEASURED PROBLEM (tools' own _StepProfiler, 2026-08-28, 48-expert config):

    teacher_fwd   5077.8 ms/step   78.4%   4 calls/step
    backward       901.6 ms/step   13.9%
    student_fwd    404.4 ms/step    6.2%
    rollout          0.1 ms/step    0.0%   <- already free via reuse

The teacher forward is 78.4% of a step. RolloutBuffer sets
    _draws_left = reuse * (rollout_batch // micro_batch) = 8 * (8//4) = 16
and draw() wraps a cursor over an 8-row store in 4-row slices, so each slice is
served EIGHT TIMES, byte-identical. distill.py nevertheless recomputes
`teacher_logits(teacher_fwd, x_in)` on every micro-step. On the on-policy path
the teacher is asked the same question 8 times and returns the same answer.

FIX: run the teacher ONCE over the wide rollout at fill() time and slice the
cached logits alongside the tokens.

EXPECTED: on-policy micro-steps are 72.6% of the total (measured from the
`op N/4` log lines). Saving = 0.784 * 0.726 * 7/8 = 49.8% of step time,
11.5 -> ~5.8 s/step, ~2x the steps per night.

NOT BIT-IDENTICAL, and this is the honest caveat: the teacher now sees a batch
of 8 rows instead of two batches of 4, so kernel reduction order differs and
logits can move in the last bits. It is numerically equivalent, not bitwise.
The A/B gate below is what establishes that it does not matter.

MEMORY: one (rollout_batch, L-1, vocab) bf16 tensor held per fill —
~88 MB at L=112, ~805 MB at L=1024. Sized in the header of run_optA_ab.sh.
"""
import os, shutil, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

def patch(path, subs, backup=True):
    p = REPO / path
    s = p.read_text()
    # Only ever back up the PRISTINE file. Re-running a half-applied patch
    # would otherwise overwrite the backup with the patched text and destroy
    # the only way back.
    if backup and not os.path.exists(str(p) + ".preA"):
        shutil.copy(p, str(p) + ".preA")
    for old, new, count in subs:
        n = s.count(old)
        if n != count:
            raise SystemExit(f"FAIL {path}: expected {count} of:\n{old[:120]!r}\ngot {n}")
        s = s.replace(old, new)
    p.write_text(s)
    print(f"  patched {path}")

# ---------------------------------------------------------------- rollout.py
patch("mythouro/rollout.py", [
(
"""        self._data: Optional[torch.Tensor] = None      # (rollout_batch, L)
        self._cursor = 0""",
"""        self._data: Optional[torch.Tensor] = None      # (rollout_batch, L)
        # OPT-A: teacher logits for `_data[:, :-1]`, computed ONCE per fill.
        # Each 4-row slice is drawn `reuse` times (8) and the teacher's answer
        # is identical every time, so recomputing it per micro-step burned
        # ~50% of the step. None when the caller did not supply them.
        self._tlogits: Optional[torch.Tensor] = None   # (rollout_batch, L-1, V)
        self._cursor = 0"""
, 1),
(
"""    def fill(self, rollouts: torch.Tensor, current_step: int) -> None:
        \"\"\"Store a fresh (B, L) token batch; resets cursor, budget, and age.\"\"\"""",
"""    def fill(
        self,
        rollouts: torch.Tensor,
        current_step: int,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> None:
        \"\"\"Store a fresh (B, L) token batch; resets cursor, budget, and age.

        `teacher_logits` (OPT-A) is the teacher's (B, L-1, V) output for
        `rollouts[:, :-1]`, computed once by the caller. It is sliced in
        lockstep with the tokens by `draw_pair()`. Pass None to keep the old
        recompute-every-micro-step behaviour.
        \"\"\"
        if teacher_logits is not None:
            if teacher_logits.shape[0] != rollouts.shape[0]:
                raise ValueError(
                    f"teacher_logits rows ({teacher_logits.shape[0]}) must match "
                    f"rollouts rows ({rollouts.shape[0]})"
                )
            if teacher_logits.shape[1] != rollouts.shape[1] - 1:
                raise ValueError(
                    f"teacher_logits length ({teacher_logits.shape[1]}) must be "
                    f"rollouts length - 1 ({rollouts.shape[1] - 1}) — it is the "
                    f"teacher's output for rollouts[:, :-1]"
                )"""
, 1),
(
"""        self._data = rollouts.detach()
        self._cursor = 0""",
"""        self._data = rollouts.detach()
        self._tlogits = None if teacher_logits is None else teacher_logits.detach()
        self._cursor = 0"""
, 1),
(
"""    def draw(self) -> torch.Tensor:
        \"\"\"Next (micro_batch, L) slice, wrapping around the stored batch.\"\"\"
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
        return out""",
"""    @staticmethod
    def _wrap_slice(buf: torch.Tensor, start: int, end: int, B: int) -> torch.Tensor:
        \"\"\"Rows [start:end) of `buf`, wrapping around at B.\"\"\"
        if end <= B:
            return buf[start:end]
        return torch.cat([buf[start:], buf[: end - B]], dim=0)

    def draw(self) -> torch.Tensor:
        \"\"\"Next (micro_batch, L) slice, wrapping around the stored batch.\"\"\"
        return self.draw_pair()[0]

    def draw_pair(self) -> "tuple[torch.Tensor, Optional[torch.Tensor]]":
        \"\"\"Next (tokens, teacher_logits) slice pair.

        OPT-A. The logits half is None unless `fill()` was given them; when
        present it is the SAME rows of the cached teacher output, so the caller
        can skip the teacher forward entirely for this micro-step.
        \"\"\"
        if self._data is None or self._draws_left <= 0:
            raise RuntimeError(
                "RolloutBuffer.draw() on an empty/exhausted buffer — call "
                "needs_refill()/fill() first."
            )
        B = self._data.shape[0]
        start, end = self._cursor, self._cursor + self.micro_batch
        out = self._wrap_slice(self._data, start, end, B)
        tl = (
            None if self._tlogits is None
            else self._wrap_slice(self._tlogits, start, end, B)
        )
        self._cursor = end % B if end <= B else end - B
        self._draws_left -= 1
        return out, tl"""
, 1),
])

# ---------------------------------------------------------------- distill.py
patch("training/distill.py", [
(
"""                        rollout_buffer.fill(wide, step)
                    rollout = rollout_buffer.draw()
                x_in, y_in = rollout[:, :-1], rollout[:, 1:]""",
"""                        # ── OPT-A: TEACHER-LOGIT REUSE CACHE ──────────────
                        # This wide rollout will be drawn `reuse *
                        # (rollout_batch // micro_batch)` times (16 at the
                        # current recipe) in slices that REPEAT — each 4-row
                        # slice is served 8 times, byte-identical. The teacher
                        # forward is 78.4% of a step (measured), so running it
                        # once here instead of per micro-step is worth ~50% of
                        # wall clock. Numerically equivalent, not bitwise: the
                        # teacher now sees 8 rows at once rather than two 4-row
                        # batches, so reduction order differs in the last bits.
                        wide_tl = None
                        if teacher is not None:
                            with amp_ctx:
                                with prof.region("teacher_fwd"):
                                    wide_tl = teacher_logits(
                                        teacher_fwd, wide[:, :-1]
                                    ).to(device)
                        rollout_buffer.fill(wide, step, teacher_logits=wide_tl)
                    rollout, cached_t_logits = rollout_buffer.draw_pair()
                x_in, y_in = rollout[:, :-1], rollout[:, 1:]"""
, 1),
(
"""            )
            if is_onpolicy:
                prof.start("rollout")""",
"""            )
            # OPT-A: set by the rollout buffer when it carries teacher logits
            # for this micro-step; None means "run the teacher yourself".
            cached_t_logits = None
            if is_onpolicy:
                prof.start("rollout")"""
, 1),
(
"""                with prof.region("teacher_fwd"):
                    t_logits = teacher_logits(teacher_fwd, x_in).to(device)""",
"""                with prof.region("teacher_fwd"):
                    if cached_t_logits is not None:
                        # OPT-A hit: identical x_in was already sent to the
                        # teacher when this rollout was generated.
                        t_logits = cached_t_logits
                    else:
                        t_logits = teacher_logits(teacher_fwd, x_in).to(device)"""
, 1),
])
print("  OPT-A applied.")
