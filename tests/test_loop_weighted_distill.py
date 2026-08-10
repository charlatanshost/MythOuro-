"""
Loop-weighted distillation — the objective that closes our divergence from Ouro.

BACKGROUND. `training/distill.py` has always distilled against `h_K`, the final
recurrent loop, while Ouro trains `L = Σ_t p(t|x)·L^(t)`: a per-step loss
weighted by exit probability. docs/growth_design.md ("loop-loss supervision")
records the divergence and calls closing it "the highest-value untried change to
the recurrent machinery"; arXiv 2602.10520 (RLTT) later measured the same idea on
Ouro-1.4B/2.6B and reported +5.8%/+10.9% over GRPO.

WHAT THESE TESTS PROTECT. Three failure modes, each of which would look like a
working run while training something other than intended:

  1. `--loop-loss-weighting off` drifting from the old behaviour. It must be
     BIT-identical, or every pre-2026-08-10 result becomes incomparable.
  2. The trajectory arriving DETACHED. `RecurrentBlock` detaches
     `last_trajectory` for its inference consumers; if that leaks into the
     training path the loss is finite, the run looks healthy, and only the LM
     head trains. Caught here by asserting gradient reaches the recurrent body.
  3. Weights that do not sum to 1 across loops, which silently rescales the
     whole objective relative to every other loss term.
"""
import sys
import pytest
import torch

sys.path.insert(0, "tests")
from test_depth_regulariser import _tiny_cfg          # noqa: E402

from mythouro.main import MythOuro                    # noqa: E402
from mythouro.training_utils import (                  # noqa: E402
    distillation_loss,
    uncertainty_calibration_loss,
)
from training.distill import _loop_weights            # noqa: E402

B, T, K = 2, 8, 4


@pytest.fixture
def model():
    torch.manual_seed(0)
    # act_threshold=99 suppresses early halting so K is deterministic.
    return MythOuro(_tiny_cfg(max_loop_iters=K, act_threshold=99.0)).eval()


@pytest.fixture
def ids(model):
    return torch.randint(0, model.cfg.vocab_size, (B, T))


class TestTokenWeights:
    """`token_weights` on distillation_loss — default None must change nothing."""

    @pytest.mark.parametrize("divergence", ["fwd_kl", "rev_kl", "jsd"])
    @pytest.mark.parametrize("with_targets", [True, False])
    def test_none_is_bit_identical(self, divergence, with_targets):
        torch.manual_seed(0)
        s = torch.randn(B, T, 23, requires_grad=True)
        t = torch.randn(B, T, 23)
        y = torch.randint(0, 23, (B, T)) if with_targets else None
        if y is not None:
            y[0, 3] = -100                        # exercise ignore_index
        kw = dict(targets=y, divergence=divergence, temperature=2.0, alpha=0.5)
        assert torch.equal(
            distillation_loss(s, t, **kw)[0],
            distillation_loss(s, t, token_weights=None, **kw)[0],
        )

    @pytest.mark.parametrize("divergence", ["fwd_kl", "rev_kl", "jsd"])
    @pytest.mark.parametrize("n_loops", [2, 4, 8])
    def test_uniform_weights_reconstruct_the_plain_mean(self, divergence, n_loops):
        """
        K uniformly-weighted copies of ONE loop must sum to that loop's own
        unweighted loss. This is what makes the weighted objective comparable in
        scale to the final-loop-only objective it replaces — without it the
        loop loss would silently run at K× or 1/K× strength against every other
        term in `loss`.
        """
        torch.manual_seed(0)
        s = torch.randn(B, T, 23, requires_grad=True)
        t = torch.randn(B, T, 23)
        y = torch.randint(0, 23, (B, T))
        y[0, 3] = -100
        kw = dict(targets=y, divergence=divergence, temperature=2.0, alpha=0.5)
        base = distillation_loss(s, t, **kw)[0]
        w = torch.full((B, T, n_loops), 1.0 / n_loops)
        total = sum(
            distillation_loss(s, t, token_weights=w[..., k], **kw)[0]
            for k in range(n_loops)
        )
        assert torch.allclose(total, base, atol=1e-5)

    def test_weighting_actually_bites(self):
        """A weight that isn't uniform must move the loss — guards a no-op."""
        torch.manual_seed(0)
        s, t = torch.randn(B, T, 23), torch.randn(B, T, 23)
        y = torch.randint(0, 23, (B, T))
        w = torch.rand(B, T, 2)
        w = w / w.sum(-1, keepdim=True)
        base = distillation_loss(s, t, targets=y)[0]
        parts = [distillation_loss(s, t, targets=y, token_weights=w[..., k])[0]
                 for k in range(2)]
        assert torch.allclose(sum(parts), base, atol=1e-5)
        assert not torch.allclose(parts[0], base, atol=1e-3)

    def test_shape_mismatch_raises_rather_than_broadcasts(self):
        s, t = torch.randn(B, T, 23), torch.randn(B, T, 23)
        with pytest.raises(ValueError, match="token_weights"):
            distillation_loss(s, t, targets=torch.randint(0, 23, (B, T)),
                              token_weights=torch.ones(B, T + 1))


class TestLoopWeights:
    def test_all_modes_sum_to_one_per_position(self):
        halt = torch.rand(B, T, K)
        halt = halt / halt.sum(-1, keepdim=True)
        for mode in ("uniform", "progressive", "exit_pdf"):
            w = _loop_weights(mode, K, halt, alpha=1.0,
                              shape=(B, T), device=torch.device("cpu"))
            assert w.shape == (B, T, K), mode
            assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5), mode

    def test_progressive_favours_later_loops(self):
        w = _loop_weights("progressive", K, None, alpha=1.0,
                          shape=(B, T), device=torch.device("cpu"))
        per_loop = w[0, 0]
        assert torch.all(per_loop[1:] > per_loop[:-1])

    def test_exit_pdf_without_halt_raises_instead_of_degrading(self):
        """
        Silently falling back to uniform would make the exit_pdf-vs-uniform A/B
        measure nothing — the single most important property of this flag.
        """
        with pytest.raises(RuntimeError, match="exit_pdf"):
            _loop_weights("exit_pdf", K, None, alpha=1.0,
                          shape=(B, T), device=torch.device("cpu"))


class TestForwardLoopStates:
    def test_shapes_and_halt_normalisation(self, model, ids):
        states, halt = model.forward_loop_states(ids, n_loops=K)
        assert states.shape == (B, T, K, model.cfg.dim)
        assert halt.shape == (B, T, K)
        assert torch.allclose(halt.sum(-1), torch.ones(B, T), atol=1e-5)

    def test_final_loop_equals_forward(self, model, ids):
        """
        THE EQUIVALENCE GATE. states[..., -1, :] fed through head/uncertainty
        must reproduce `forward()` exactly. If it drifts, the loop-weighted run
        is not training the model that inference will run.
        """
        with torch.no_grad():
            ref_logits, ref_unc = model(ids, n_loops=K)
            states, _ = model.forward_loop_states(ids, n_loops=K)
            assert torch.allclose(model.head(states[..., -1, :]), ref_logits,
                                  atol=1e-5)
            assert torch.allclose(model.uncertainty(states[..., -1, :]), ref_unc,
                                  atol=1e-5)

    def test_gradient_reaches_the_recurrent_body(self, model, ids):
        """
        Regression test for the detached-trajectory bug. `RecurrentBlock`
        detaches `last_trajectory` for its inference consumers; before
        `trajectory_requires_grad` existed, this path trained the HEAD ONLY —
        11 parameters instead of 51 — while producing a perfectly plausible
        loss curve.
        """
        states, _ = model.forward_loop_states(ids, n_loops=K)
        model.head(states[..., -1, :]).square().mean().backward()
        got = {n for n, p in model.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0}
        for part in ("recurrent", "prelude", "coda", "embed"):
            assert any(part in n for n in got), f"no gradient reached {part}"

    def test_inference_trajectory_stays_detached(self, model, ids):
        """The opt-in must not leak into the inference inspector."""
        model.forward_loop_states(ids, n_loops=K)          # sets, then resets
        assert model.recurrent.trajectory_requires_grad is False
        assert model.recurrent.collect_trajectory is False
        logits_traj, _ = model.forward_trajectory(ids, n_loops=K)
        assert logits_traj.requires_grad is False


class TestUncertaintyTokenWeights:
    """
    Per-loop calibration of the UncertaintyHead.

    The head is trained only on final-loop states today, which is why per-loop
    calibration reads ECE 0.013 at loop 3 and 0.288 at loop 0 — loop 3 is the
    only depth it has ever seen. Everything that consumes shallow-loop
    confidence inherits that: ACT halting, BestOfTrajectoryGenerator (which
    preferred loops 1-2 and RAISED the copy rate 41.2% -> 50.0%), and exit_pdf
    weighting itself.
    """

    def test_none_is_bit_identical(self):
        torch.manual_seed(0)
        logits = torch.randn(B, T, 23)
        unc = torch.rand(B, T)
        y = torch.randint(0, 23, (B, T))
        assert torch.equal(
            uncertainty_calibration_loss(logits, unc, y),
            uncertainty_calibration_loss(logits, unc, y, token_weights=None),
        )

    @pytest.mark.parametrize("n_loops", [2, 4])
    def test_uniform_weights_reconstruct_the_plain_mean(self, n_loops):
        torch.manual_seed(0)
        logits = torch.randn(B, T, 23)
        unc = torch.rand(B, T)
        y = torch.randint(0, 23, (B, T))
        base = uncertainty_calibration_loss(logits, unc, y)
        w = torch.full((B, T, n_loops), 1.0 / n_loops)
        total = sum(
            uncertainty_calibration_loss(logits, unc, y, token_weights=w[..., k])
            for k in range(n_loops)
        )
        assert torch.allclose(total, base, atol=1e-5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="token_weights"):
            uncertainty_calibration_loss(
                torch.randn(B, T, 23), torch.rand(B, T),
                torch.randint(0, 23, (B, T)),
                token_weights=torch.ones(B, T + 1),
            )

    def test_error_target_follows_the_logits_it_is_given(self):
        """
        The point of the whole exercise: loop k's error target must come from
        loop k's OWN logits. If a shallow loop were scored against the final
        loop's correctness it would keep inheriting confidence it hasn't earned,
        which is exactly the miscalibration being fixed.

        Asserted directly, with logits constructed to be right vs wrong, rather
        than via the model — on an untrained model every loop is wrong at every
        position, so real per-loop error rates are all 1.0 and could not
        distinguish "used loop k" from "used loop K-1".
        """
        y = torch.randint(1, 20, (B, T))
        right = torch.zeros(B, T, 23).scatter_(-1, y.unsqueeze(-1), 10.0)
        wrong = torch.zeros(B, T, 23).scatter_(-1, ((y + 1) % 23).unsqueeze(-1), 10.0)
        confident = torch.full((B, T), 0.01)      # "I will be correct"

        # Confident + right = low loss; confident + wrong = high loss. If the
        # target ignored the logits these would be equal.
        assert (uncertainty_calibration_loss(right, confident, y)
                < uncertainty_calibration_loss(wrong, confident, y))

    def test_per_loop_states_are_distinct(self, model, ids):
        """A per-loop target is only meaningful if the loops actually differ."""
        states, _ = model.forward_loop_states(ids, n_loops=K)
        for k in range(K - 1):
            assert not torch.allclose(states[..., k, :], states[..., k + 1, :]), (
                f"loop {k} and {k+1} produced identical states"
            )


class TestEndToEnd:
    @pytest.mark.parametrize("mode", ["uniform", "progressive", "exit_pdf"])
    def test_weighted_objective_trains(self, model, ids, mode):
        """Mimics distill.py's loop-weighted block: finite loss, real gradient."""
        teacher = torch.randn(B, T, model.cfg.vocab_size)
        states, halt = model.forward_loop_states(ids, n_loops=K)
        w = _loop_weights(mode, states.shape[2], halt, alpha=1.0,
                          shape=states.shape[:2], device=states.device)
        total = None
        for k in range(states.shape[2]):
            loss_k, _ = distillation_loss(
                model.head(states[..., k, :]), teacher, targets=ids,
                token_weights=w[..., k],
            )
            total = loss_k if total is None else total + loss_k
        assert torch.isfinite(total)
        total.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if "recurrent" in n)
