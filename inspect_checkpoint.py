"""
Inspect a MythOuro checkpoint — load it, generate from prompts, surface
the architecture-specific diagnostics that distinguish MythOuro from a
generic transformer.

What you get per prompt:
    1. Plain greedy generation                              — what tokens fall out
    2. ConfidenceAwareGenerator output + stop reason         — does the model "know
                                                              when to shut up"?
    3. Per-generated-token uncertainty trace                 — UncertaintyHead readout
    4. Mean halt depth across the generated tokens           — does ACT spread or collapse?
    5. MoE routing utilization snapshot (CV / min / max)     — is the router healthy?

Usage:
    # Inspect the most recent checkpoint in checkpoints_distill/
    python inspect_checkpoint.py

    # Specific checkpoint, specific prompt
    python inspect_checkpoint.py --checkpoint checkpoints_distill/step_0000200.pt \\
        --prompt "The recurrent depth transformer is"

    # Interactive: read prompts from stdin until Ctrl+D / 'exit'
    python inspect_checkpoint.py --interactive

Honest expectations:
    A 200-step distillation checkpoint is NOT going to produce coherent
    text. The generation will likely be repetitive, drift, or output
    weird high-frequency tokens. The diagnostics (uncertainty trace,
    halt distribution, MoE utilization) are the load-bearing part — they
    tell you whether the model's internal mechanisms are healthy even
    when the output isn't yet useful.

    Compare across checkpoints to see which numbers move.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import torch

from mythouro.main import MythOuro, MythOuroConfig
from mythouro.tokenizer import MythOuroTokenizer
from mythouro.inference import ConfidenceAwareGenerator


# Default prompt set — short, diverse, designed to exercise different
# capability axes (factual recall / continuation / instruction format /
# math / code). Substitute your own via --prompt.
_DEFAULT_PROMPTS = [
    "The recurrent depth transformer is",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "def fibonacci(n):",
    "Q: Roughly what year was the Roman Empire founded?\nA:",
]


# ---------------------------------------------------------------------------
# Checkpoint discovery + loading
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return the path to the most recent `step_*.pt` in `ckpt_dir`, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    return paths[-1] if paths else None


def _load_model(checkpoint_path: str, device: str) -> "tuple[MythOuro, MythOuroConfig, int]":
    """
    Rebuild the student from a checkpoint and move it to `device`.

    Doesn't touch optimizer state — we're forward-only here. Uses the
    pickled `cfg` to reconstruct the exact architecture the checkpoint
    was trained with; cfg_dict mismatch with the current dataclass would
    fail load_state_dict at the layer-name level, which is the correct
    behaviour.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    step = int(ckpt.get("step", -1))
    model = MythOuro(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, cfg, step


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _diagnostic_forward(model: MythOuro, ids: torch.Tensor, n_loops: int) -> dict:
    """
    Run one forward pass and pull every architecture-specific signal we
    care about. Used as a snapshot, not part of generation — the
    generator paths below handle generation themselves.
    """
    logits, unc = model(ids, n_loops=n_loops)
    out = {
        "logits_shape": tuple(logits.shape),
        "uncertainty_mean": float(unc.mean().item()),
        "uncertainty_max":  float(unc.max().item()),
        "uncertainty_min":  float(unc.min().item()),
    }
    halt = getattr(model.recurrent, "last_halt_step", None)
    if halt is not None:
        out["halt_step_mean"] = float(halt.float().mean().item())
        out["halt_step_max"]  = int(halt.max().item())
    dist = getattr(model.recurrent, "last_halt_distribution", None)
    if dist is not None:
        # Mean over (B, T): K-vector of average halt-probability per depth.
        mean_dist = dist.float().mean(dim=(0, 1))
        out["halt_distribution"] = [round(float(x), 3) for x in mean_dist.tolist()]
    # MoE health snapshot via stashed `_last_expert_counts` if present.
    from mythouro.main import MoEFFN
    for mod in model.modules():
        if isinstance(mod, MoEFFN):
            counts = getattr(mod, "_last_expert_counts", None)
            if counts is not None and counts.sum() > 0:
                fracs = counts.float() / counts.sum().clamp_min(1)
                out["moe_router_min_pct"] = round(100 * fracs.min().item(), 2)
                out["moe_router_max_pct"] = round(100 * fracs.max().item(), 2)
                # Coefficient of variation: 0 = perfectly uniform, grows
                # with skew. The DeepSeek-V3 bias updater is targeting
                # this implicitly.
                mean = fracs.mean().item()
                if mean > 0:
                    out["moe_router_cv"] = round(
                        (fracs.std().item() / mean), 3
                    )
            break
    return out


# ---------------------------------------------------------------------------
# Per-prompt inspection
# ---------------------------------------------------------------------------


def _inspect_prompt(
    model: MythOuro,
    tokenizer: MythOuroTokenizer,
    prompt: str,
    device: str,
    *,
    max_new_tokens: int,
    n_loops: int,
) -> None:
    print()
    print("=" * 80)
    print(f"prompt: {prompt!r}")
    print("-" * 80)

    ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device,
    )
    print(f"prompt tokens : {ids.shape[1]}")

    # ── Diagnostic forward (no generation) ─────────────────────────────
    diag = _diagnostic_forward(model, ids, n_loops=n_loops)
    print(f"diagnostic    : uncertainty mean={diag['uncertainty_mean']:.3f} "
          f"max={diag['uncertainty_max']:.3f}")
    if "halt_distribution" in diag:
        print(f"                halt distribution per loop = {diag['halt_distribution']}")
    if "halt_step_mean" in diag:
        print(f"                halt step mean = {diag['halt_step_mean']:.2f}  "
              f"(0 = halted immediately, {n_loops} = ran full depth)")
    if "moe_router_cv" in diag:
        print(f"                MoE router CV={diag['moe_router_cv']:.3f}  "
              f"(min={diag['moe_router_min_pct']:.1f}%  "
              f"max={diag['moe_router_max_pct']:.1f}%)")

    # ── Greedy generation via model.generate (the simplest path) ──────
    out = model.generate(
        ids, max_new_tokens=max_new_tokens,
        n_loops=n_loops, temperature=0.7, top_k=40,
    )
    new_tokens = out[0, ids.shape[1] :].tolist()
    decoded = tokenizer.decode(new_tokens)
    print()
    print("greedy generation (T=0.7 top_k=40):")
    print(f"  {decoded!r}")

    # ── Confidence-aware generation — shows stop_reason + uncertainty trace
    cag = ConfidenceAwareGenerator(
        model,
        n_loops=n_loops,
        eos_token_id=tokenizer.eos_token_id,
        # Natural breaks: encode period, newline, EOS. Best-effort — if a
        # tokenizer doesn't have these as standalone tokens, an empty
        # break_token_ids set disables the confidence stop entirely.
        break_token_ids=_break_token_candidates(tokenizer),
        confidence_window=4,
        confidence_threshold=0.4,
        cycle_min_len=3,
        cycle_window=12,
    )
    result = cag.generate(
        ids, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40,
    )
    cag_new = result["sequences"][0, ids.shape[1] :].tolist()
    print()
    print(f"confidence-aware generation  (stop={result['stop_reason']!r}, "
          f"{len(cag_new)} tokens emitted):")
    print(f"  {tokenizer.decode(cag_new)!r}")
    trace = result["uncertainty_trace"]
    if trace:
        print(f"  uncertainty trace: "
              f"mean={sum(trace)/len(trace):.3f}  "
              f"min={min(trace):.3f}  max={max(trace):.3f}")


def _break_token_candidates(tokenizer: MythOuroTokenizer) -> "list[int] | None":
    """
    Best-effort sentence-break token ids for the Ouro/ChatML tokenizer.
    The exact ids vary by tokenizer; we encode common terminators and
    de-duplicate. Returns None if nothing meaningful resolves, which
    disables the confidence stop (fail-closed default).
    """
    candidates: "set[int]" = set()
    for s in (".", "?", "!", "\n", "\n\n", "<|im_end|>", "<|endoftext|>"):
        try:
            ids = tokenizer.encode(s, add_special_tokens=True)
        except TypeError:
            ids = tokenizer.encode(s)
        if ids:
            candidates.update(ids)
    eos = tokenizer.eos_token_id
    if eos is not None:
        candidates.add(eos)
    return list(candidates) if candidates else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect a MythOuro checkpoint by generating from prompts "
                    "and surfacing the architecture-specific diagnostics.",
    )
    p.add_argument(
        "--checkpoint", "-c", default=None,
        help="Path to a step_*.pt file. Default: latest checkpoint in "
             "checkpoints_distill/, or checkpoints/ as fallback.",
    )
    p.add_argument(
        "--device", default=None,
        help="cuda:N / cpu. Default: cuda:0 if available else cpu.",
    )
    p.add_argument(
        "--tokenizer", default="ByteDance/Ouro-2.6B-Thinking",
        help="HF tokenizer id. Should match what the checkpoint was trained on.",
    )
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument(
        "--n-loops", type=int, default=None,
        help="Recurrent loop depth at inference. Default: cfg.max_loop_iters.",
    )
    p.add_argument(
        "--prompt", "-p", default=None,
        help="Single prompt to inspect. Default: a built-in 4-prompt set.",
    )
    p.add_argument(
        "--interactive", "-i", action="store_true",
        help="Read prompts from stdin until EOF or 'exit'.",
    )
    return p.parse_args(argv)


def main():
    args = _parse_args()

    # ── Locate the checkpoint ────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = (
            _find_latest_checkpoint("checkpoints_distill")
            or _find_latest_checkpoint("checkpoints")
        )
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("error: no checkpoint found. Pass --checkpoint or train one first.",
              file=sys.stderr)
        sys.exit(1)

    # ── Pick the device ──────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"loading: {ckpt_path}")
    print(f"device:  {device}")

    # ── Load model + tokenizer ───────────────────────────────────────
    model, cfg, step = _load_model(ckpt_path, device)
    print(f"step:    {step}")
    print(f"params:  {sum(p.numel() for p in model.parameters()):,}")
    print(f"vocab:   {cfg.vocab_size}  max_loops={cfg.max_loop_iters}")

    tokenizer = MythOuroTokenizer(args.tokenizer)
    if tokenizer.vocab_size != cfg.vocab_size:
        print(
            f"WARNING: tokenizer vocab ({tokenizer.vocab_size}) ≠ model "
            f"vocab ({cfg.vocab_size}). Generation will likely be garbled."
        )

    n_loops = args.n_loops if args.n_loops is not None else cfg.max_loop_iters

    # ── Run inspection ───────────────────────────────────────────────
    if args.interactive:
        print("\nInteractive mode. Type prompts, Ctrl+D / 'exit' to quit.")
        while True:
            try:
                prompt = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt or prompt.lower() in {"exit", "quit"}:
                break
            _inspect_prompt(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_new_tokens, n_loops=n_loops,
            )
    else:
        prompts = [args.prompt] if args.prompt else _DEFAULT_PROMPTS
        for prompt in prompts:
            _inspect_prompt(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_new_tokens, n_loops=n_loops,
            )


if __name__ == "__main__":
    main()
