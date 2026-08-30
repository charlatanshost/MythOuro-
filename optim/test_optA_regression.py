"""The four existing TestRolloutBuffer cases, run against the OPT-A file."""
import sys, pathlib, torch, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mythouro.rollout import RolloutBuffer
import pytest as _pt
if not hasattr(RolloutBuffer, "draw_pair"):      # OPT-A not applied yet
    _pt.skip("OPT-A not applied (optim/apply.sh A)", allow_module_level=True)

def test_reuse_budget_and_refill_cycle():
    buf = RolloutBuffer(8, 2, reuse=2, max_age_steps=100)
    assert buf.needs_refill(0)
    buf.fill(torch.arange(8 * 4).reshape(8, 4), current_step=0)
    for _ in range(8):
        assert not buf.needs_refill(1)
        assert buf.draw().shape == (2, 4)
    assert buf.needs_refill(1)
    with pytest.raises(RuntimeError):
        buf.draw()

def test_slices_cycle_through_all_rows():
    buf = RolloutBuffer(4, 2, reuse=1, max_age_steps=100)
    data = torch.arange(4 * 3).reshape(4, 3)
    buf.fill(data, current_step=0)
    seen = torch.cat([buf.draw(), buf.draw()], dim=0)
    assert torch.equal(seen, data)

def test_staleness_cap_forces_refill():
    buf = RolloutBuffer(4, 2, reuse=100, max_age_steps=10)
    buf.fill(torch.zeros(4, 3, dtype=torch.long), current_step=5)
    assert not buf.needs_refill(14)
    assert buf.needs_refill(15)

def test_rollout_batch_rounded_down_to_micro_multiple():
    buf = RolloutBuffer(7, 2, reuse=1)
    assert buf.rollout_batch == 6
