"""OPT-A correctness: cached teacher logits must track the tokens exactly."""
import sys, torch, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mythouro.rollout import RolloutBuffer
import pytest as _pt
if not hasattr(RolloutBuffer, "draw_pair"):      # OPT-A not applied yet
    _pt.skip("OPT-A not applied (optim/apply.sh A)", allow_module_level=True)

def test_backward_compatible_draw():
    b = RolloutBuffer(8, 2, reuse=2)
    b.fill(torch.arange(8 * 4).reshape(8, 4), current_step=0)
    assert b.draw().shape == (2, 4)          # still returns a bare tensor
    assert b.draw_pair()[1] is None          # no logits supplied -> None
    print("  ok: draw() unchanged when no logits given")

def test_logits_track_tokens_through_wraparound():
    """The real invariant. 8 rows / micro 4 / reuse 8 = 16 draws, wrapping."""
    B, L, V = 8, 5, 7
    toks = torch.arange(B * L).reshape(B, L)
    # make logits identifiable per row: row i is filled with value i
    tl = torch.arange(B).float().view(B, 1, 1).expand(B, L - 1, V).contiguous()
    b = RolloutBuffer(B, 4, reuse=8)
    b.fill(toks, current_step=0, teacher_logits=tl)
    n = 0
    while not b.needs_refill(0):
        t, g = b.draw_pair()
        assert g is not None
        # every row of g must carry the id of the token row it came with
        row_ids = (t[:, 0] // L).float()
        assert torch.equal(g[:, 0, 0], row_ids), f"draw {n}: {g[:,0,0]} vs {row_ids}"
        n += 1
    assert n == 16, f"expected 16 draws, got {n}"
    print(f"  ok: {n} draws, logits tracked tokens through every wraparound")

def test_each_slice_served_eight_times():
    """Proves the redundancy the patch removes."""
    B, L = 8, 5
    b = RolloutBuffer(B, 4, reuse=8)
    b.fill(torch.arange(B * L).reshape(B, L), current_step=0)
    from collections import Counter
    c = Counter()
    while not b.needs_refill(0):
        c[int(b.draw()[0, 0])] += 1
    assert set(c.values()) == {8}, c
    print(f"  ok: {len(c)} distinct slices, each served {set(c.values())} times "
          f"-> 7/8 of teacher forwards were redundant")

def test_shape_guards():
    b = RolloutBuffer(8, 4, reuse=2)
    for bad, why in (
        (torch.zeros(4, 4, 7), "wrong row count"),
        (torch.zeros(8, 9, 7), "wrong length"),
    ):
        try:
            b.fill(torch.zeros(8, 5, dtype=torch.long), 0, teacher_logits=bad)
        except ValueError:
            continue
        raise AssertionError(f"guard missed: {why}")
    print("  ok: shape guards reject mismatched logits")

for f in (test_backward_compatible_draw, test_logits_track_tokens_through_wraparound,
          test_each_slice_served_eight_times, test_shape_guards):
    f()
print("\n  OPT-A: all tests pass")
