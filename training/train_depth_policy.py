"""
Teach the halt decision from a best-exit target — the depth-policy trainer.

THE PIPELINE, per docs/ideas.md "MoDr":
    "Let it run full depth at TRAINING time to discover the best exit, and train
     a cheap per-token policy to hit that exit directly at inference. The
     forced-depth probe is the TEACHER; the depth policy is the STUDENT."

So the teacher here is the model's OWN forced-depth trajectory, not Ouro — no
second model forward is needed, which is why this is cheap enough to run on the
same card between distillation legs.

  TEACHER (no_grad): forward_trajectory(force_full_depth=True) runs every loop.
      Per-loop CE against the gold next token gives, for each position, which
      depth was ACTUALLY best. That argmin is the label.
  STUDENT (grad): the UncertaintyHead learns to rank the loops so its argmin
      matches the teacher's. Only this head trains; everything else is frozen.

WHY THIS EXISTS — the head currently optimises the wrong thing. Measured
2026-07-31 on step 70,500:
  * `BestOfTrajectoryGenerator` picked loop 1 or 2 for 98% of 3,161 tokens, never
    past loop 3 with a budget of 8, and drove the prompt-copy rate UP from 41.2%
    to 50.0%. Argmin-over-uncertainty is an ECHO-SEEKING objective, because
    copying a token just seen is the most confident prediction available.
  * ACT's halt fired at a constant depth — 2.00 in all 24 (domain x alpha) cells,
    120 measurements, zero variance.
Two different consumers of one head, both settling shallow while answers stay
wrong ⇒ the defect is the SIGNAL. Nothing has ever tied confidence to
correctness. `tools/per_loop_calibration.py` has printed the fix as its verdict
since June: "use per-loop CE, not uncertainty-argmin, as the best-exit target."

⚠ SCOPE — this is NOT MoDr. MoDr unifies expert routing and depth routing into
one head, and docs/ideas.md gates that behind the MoE-vs-dense ablation
("sequencing rule: run the dense-vs-MoE ablation first"). This trains the
EXISTING halting head against a better target and touches no routing, so it is
not gated by that decision.

⚠ PRECONDITION — run `tools/best_exit_probe.py` first. If the best exit is the
same loop for every token, there is no policy to learn and this trainer is
pointless; if the CE headroom between the oracle exit and the final loop is ~0,
a perfect policy buys nothing. The probe reports both.

⚠ NOT USEFUL YET at 0% task correctness: when every loop is wrong, "which loop
was least wrong" is a weak label. This queues BEHIND the corrected-data pour.

Usage
-----
    python -m training.train_depth_policy \\
        -c checkpoints_newmix/step_0093000.pt -o checkpoints_depthpolicy \\
        --device xpu:0 --steps 2000 --n-loops 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_checkpoint import _load_model                      # noqa: E402


def per_loop_ce(model, states: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
    """(B,T,K) CE of each loop's logits against the gold next token.

    Computed one loop at a time and discarded: stacking (B,T,K,V) is ~10GB in
    bf16 at training shape, which does not fit beside the rest of the run.
    """
    B, T, K, _ = states.shape
    ce = torch.empty(B, T, K, device=states.device, dtype=torch.float32)
    for k in range(K):
        logits = model.head(states[:, :, k, :])
        ce[:, :, k] = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1), reduction="none",
        ).view(B, T)
    return ce


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("-o", "--out-dir", required=True)
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--n-loops", type=int, default=8,
                   help="Depth scored by the teacher. ABOVE the trained 4 on "
                        "purpose: the policy should be free to discover that a "
                        "token wanted depth the training curriculum never used. "
                        "Costs teacher time only — the student head is tiny.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Higher than distillation LR: one small head trains from "
                        "a target it has never seen, and nothing else moves.")
    p.add_argument("--temp", type=float, default=1.0,
                   help="Softmax temperature on -uncertainty when forming the "
                        "ranking loss.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    from mythouro.tokenizer import MythOuroTokenizer
    from mythouro.training_utils import MixedDataset

    enc = MythOuroTokenizer(args.tokenizer)
    model, cfg, step0 = _load_model(args.checkpoint, args.device)
    model.eval()                       # frozen everywhere except the head below

    # FREEZE EVERYTHING, then unfreeze only the uncertainty head. This trainer
    # must not move the LM: the teacher labels come from the model's own
    # trajectory, so letting the body drift would move the target while chasing
    # it. Only the halt/uncertainty branch learns.
    for prm in model.parameters():
        prm.requires_grad_(False)
    head = model.uncertainty
    for prm in head.parameters():
        prm.requires_grad_(True)
    head.train()
    n_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    logger.info(f"training UncertaintyHead only: {n_train:,} / {n_all:,} params "
                f"({100 * n_train / n_all:.3f}%)")

    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)

    ds = MixedDataset(encoding=enc, seq_len=args.seq_len, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.micro_batch, num_workers=0)
    it = iter(loader)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run_loss = run_agree = run_gap = 0.0
    seen = 0

    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(args.device)
        y = y.to(args.device)

        # ── TEACHER: full-depth trajectory + per-loop CE -> best exit ──
        with torch.no_grad():
            _, _, states = model.forward_trajectory(
                x, n_loops=args.n_loops, force_full_depth=True,
                return_states=True,
            )
            ce = per_loop_ce(model, states, y)          # (B,T,K)
            best_k = ce.argmin(dim=-1)                  # (B,T) the label
            ce_oracle = ce.gather(-1, best_k.unsqueeze(-1)).mean()
            ce_final = ce[:, :, -1].mean()

        # ── STUDENT: rank the loops so argmin(unc) == argmin(CE) ──
        # states are detached (no_grad above), so gradient reaches ONLY the head.
        # Logits are -uncertainty: the head emits confidence-of-ERROR, so the
        # best loop is the LOWEST one, and negating turns argmin into argmax.
        # UncertaintyHead returns (B,T) — already squeezed. An earlier
        # `.squeeze(-1)` here was a no-op for T>1 but would have eaten the TIME
        # dimension at T==1, so it is deliberately absent.
        unc = torch.stack(
            [head(states[:, :, k, :].detach())
             for k in range(states.shape[2])], dim=-1)      # (B,T,K)
        loss = F.cross_entropy(
            (-unc / args.temp).reshape(-1, unc.shape[-1]),
            best_k.reshape(-1),
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            agree = (unc.argmin(-1) == best_k).float().mean()
        run_loss += float(loss.detach())
        run_agree += float(agree)
        run_gap += float(ce_final - ce_oracle)
        seen += 1

        if step % args.log_every == 0:
            dt = time.perf_counter() - t0
            logger.info(
                f"step {step:5d}/{args.steps} | loss {run_loss / seen:.4f} "
                f"| agree {100 * run_agree / seen:5.1f}% "
                f"| oracle headroom {run_gap / seen:.4f} nats "
                f"| {dt / step:.2f}s/step"
            )
            run_loss = run_agree = run_gap = 0.0
            seen = 0

        if step % args.save_every == 0 or step == args.steps:
            ck = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)
            sd = model.state_dict()
            ck["model"] = {k: v.detach().cpu() for k, v in sd.items()}
            ck.pop("optimizer", None)          # LM optimizer state is now stale
            extra = dict(ck.get("extra") or {})
            extra["depth_policy_steps"] = step
            extra["depth_policy_from"] = str(args.checkpoint)
            ck["extra"] = extra
            path = out / f"step_{step0 + step:07d}.pt"
            torch.save(ck, path)
            logger.info(f"saved {path}")

    logger.info("done. Compare agreement against the pre-training baseline from "
                "tools/best_exit_probe.py, then re-run tools/math_eval with "
                "--decoder best_of_trajectory to see if better exits produce "
                "better answers.")


if __name__ == "__main__":
    main()
