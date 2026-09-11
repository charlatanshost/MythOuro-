"""
Sparse (top-K + lumped tail) knowledge distillation — OPTIMISATION B.

WHY. `_StepProfiler` measured the teacher forward at **78.4% of every training
step** (2026-08-28, 48-expert config). OPT-A removes the on-policy share by
caching logits across rollout reuse. This removes the OFFLINE share by
precomputing the teacher's answer to disk once, so training never runs the
teacher on corpus batches at all.

THE APPROXIMATION, STATED PLAINLY. Storing a full (B, T, 49152) teacher
distribution per token is not affordable on disk. We store the top-K logits plus
ONE number summarising everything else: `tail_lse = logsumexp(tail_logits / T)`.
The divergence is then computed on a COARSENED event space — K explicit symbols
plus a single "everything else" bucket — for both teacher and student.

By the data-processing inequality, KL on a coarsening is a LOWER BOUND on the
true KL. So this trains against a slightly softened objective, and the gap
shrinks as K grows. That is a real approximation, not a free lunch, and it is
why OPT-B ships with an A/B against OPT-A rather than as a drop-in.

TEMPERATURE IS BAKED IN. `logsumexp(z/T)` over the tail cannot be recovered from
`logsumexp(z)`, so `tail_lse` is only valid at the temperature it was computed
at. The manifest records it and the loader refuses a mismatch. The top-K values
are stored RAW (T=1) and rescaled at train time, so only the tail term is
temperature-locked.

EXACTNESS GATE. With K = vocab_size the tail is empty (`tail_lse = -inf`) and
this function must reproduce `distillation_loss` to float tolerance. That
equivalence is the test that keeps the sparse path honest.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_distillation_loss(
    student_logits: torch.Tensor,     # (B, T, V) — gradient flows here
    topk_idx: torch.Tensor,           # (B, T, K) long  — teacher's top-K vocab ids
    topk_val: torch.Tensor,           # (B, T, K) float — RAW teacher logits (T=1)
    tail_lse: torch.Tensor,           # (B, T)    float — logsumexp(tail/T), -inf if none
    targets: "torch.Tensor | None" = None,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
    divergence: str = "fwd_kl",
    jsd_beta: float = 0.5,
    token_weights: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, dict]":
    T = float(temperature)
    if T <= 0:
        raise ValueError(f"temperature must be > 0; got {T}")
    if topk_idx.shape != topk_val.shape:
        raise ValueError(
            f"topk_idx {tuple(topk_idx.shape)} and topk_val "
            f"{tuple(topk_val.shape)} must match"
        )
    if topk_idx.shape[:2] != student_logits.shape[:2]:
        raise ValueError(
            f"cache is (B,T)={tuple(topk_idx.shape[:2])} but student is "
            f"{tuple(student_logits.shape[:2])} — the cache is misaligned with "
            "the batch, which would distil against the wrong tokens."
        )

    s_all = F.log_softmax(student_logits.float() / T, dim=-1)     # (B,T,V)
    s_logp_k = s_all.gather(-1, topk_idx)                          # (B,T,K)

    with torch.no_grad():
        t_top = topk_val.float() / T                               # (B,T,K)
        tail = tail_lse.float()                                    # (B,T)
        has_tail = torch.isfinite(tail)
        # logZ over K explicit symbols + the tail bucket
        cat = torch.cat([t_top, torch.where(has_tail, tail, tail.new_full((), -1e30))
                         .unsqueeze(-1)], dim=-1)
        logZ = torch.logsumexp(cat, dim=-1)                        # (B,T)
        t_logp_k = t_top - logZ.unsqueeze(-1)
        t_p_k = t_logp_k.exp()
        t_logp_tail = tail - logZ
        t_p_tail = torch.where(has_tail, t_logp_tail.exp(),
                               torch.zeros_like(logZ))

    s_p_k = s_logp_k.exp()
    # student mass on the tail bucket = 1 - mass on the top-K
    s_p_tail = (1.0 - s_p_k.sum(dim=-1)).clamp_min(1e-9)
    s_logp_tail = s_p_tail.log()

    def _tail_term(pa, log_pa, log_pb):
        """pa * (log_pa - log_pb), zeroed where the tail bucket is empty."""
        raw = pa * (log_pa - log_pb)
        return torch.where(has_tail, raw, torch.zeros_like(raw))

    if divergence == "fwd_kl":
        div_rows = (t_p_k * (t_logp_k - s_logp_k)).sum(dim=-1)
        div_rows = div_rows + _tail_term(t_p_tail, t_logp_tail, s_logp_tail)
    elif divergence == "rev_kl":
        div_rows = (s_p_k * (s_logp_k - t_logp_k)).sum(dim=-1)
        div_rows = div_rows + _tail_term(s_p_tail, s_logp_tail, t_logp_tail)
    elif divergence == "jsd":
        b = float(jsd_beta)
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"jsd_beta must be in [0, 1]; got {b}")
        m_k = (b * t_p_k + (1.0 - b) * s_p_k).clamp_min(1e-9).log()
        m_tail = (b * t_p_tail + (1.0 - b) * s_p_tail).clamp_min(1e-9).log()
        div_rows = (
            b * (t_p_k * (t_logp_k - m_k)).sum(dim=-1)
            + (1.0 - b) * (s_p_k * (s_logp_k - m_k)).sum(dim=-1)
            + b * _tail_term(t_p_tail, t_logp_tail, m_tail)
            + (1.0 - b) * _tail_term(s_p_tail, s_logp_tail, m_tail)
        )
    else:
        raise ValueError(
            f"divergence must be 'fwd_kl', 'rev_kl', or 'jsd'; got {divergence!r}"
        )

    div_rows = div_rows.view(-1)

    w_rows = None
    if token_weights is not None:
        w_rows = token_weights.reshape(-1).to(div_rows.dtype)
        if w_rows.shape != div_rows.shape:
            raise ValueError(
                f"token_weights gives {tuple(w_rows.shape)} positions but the "
                f"logits give {tuple(div_rows.shape)}"
            )

    if targets is not None:
        valid = (targets.view(-1) != ignore_index)
        n_valid = int(valid.sum())
        if n_valid == 0:
            zero = student_logits.sum() * 0.0
            return zero, {"soft": 0.0, "hard": 0.0}
        rows = div_rows * valid
        if w_rows is not None:
            rows = rows * w_rows
        soft = rows.sum() / n_valid
    else:
        rows = div_rows if w_rows is None else div_rows * w_rows
        soft = rows.mean()

    soft = soft * (T ** 2)

    if targets is None:
        return soft, {"soft": float(soft.detach()), "hard": 0.0}

    hard = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    total = alpha * soft + (1.0 - alpha) * hard
    return total, {"soft": float(soft.detach()), "hard": float(hard.detach())}
