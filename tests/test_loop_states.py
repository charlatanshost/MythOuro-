"""forward_loop_states must return ALL loops during TRAINING.

Regression test for the 2026-09-01 defect: `collect` was gated on
`not self.training`, so forward_loop_states got last_trajectory=None in train
mode, fell through to `traj = e.unsqueeze(2)` (labelled "n_loops == 0"), and
supervised coda(prelude(x)) with the recurrent block bypassed. Every
--loop-loss-weighting mode was affected; only exit_pdf raised. Rung 3's
"uniform destroys the model" was measuring this.
"""
import torch
from mythouro.main import MythOuro
from mythouro.variants import mythouro_distill_tiny


def _tiny():
    cfg = mythouro_distill_tiny()
    cfg.vocab_size, cfg.max_seq_len = 128, 32
    cfg.dim, cfg.n_experts, cfg.expert_dim = 64, 4, 32
    cfg.n_experts_per_tok = 2
    return MythOuro(cfg), cfg


def test_all_loops_captured_in_train_mode():
    m, cfg = _tiny(); m.train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    states, halt = m.forward_loop_states(x, n_loops=4)
    assert states.shape[2] == 4, f"expected K=4, got {states.shape[2]}"
    assert halt is not None, "halt distribution rejected — shapes drifted"
    assert halt.shape == (2, 8, 4)


def test_halt_is_a_distribution_and_aligned():
    m, cfg = _tiny(); m.train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    states, halt = m.forward_loop_states(x, n_loops=4)
    assert halt.shape[:2] == states.shape[:2], "halt not aligned to states"
    assert torch.allclose(halt.sum(-1), torch.ones_like(halt.sum(-1)), atol=1e-5)


def test_states_carry_gradient():
    """Without the graph the recurrent block gets no gradient at all."""
    m, cfg = _tiny(); m.train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    states, _ = m.forward_loop_states(x, n_loops=4)
    assert states.requires_grad
    states.sum().backward()
    got = [n for n, p in m.recurrent.named_parameters()
           if p.grad is not None and p.grad.abs().sum() > 0]
    assert got, "recurrent block received NO gradient — the loop objective would train only the head"


def test_default_forward_unaffected():
    """Collection must stay off unless forward_loop_states opts in."""
    m, cfg = _tiny(); m.train()
    assert m.recurrent.collect_trajectory is False
    assert m.recurrent.trajectory_requires_grad is False
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = m(x)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == (2, 8, cfg.vocab_size)
    assert m.recurrent.last_trajectory is None
