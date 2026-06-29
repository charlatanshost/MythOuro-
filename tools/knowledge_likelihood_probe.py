#!/usr/bin/env python
"""
Knowledge LIKELIHOOD probe — does the student *know* the fact, decoupled from
whether it can *generate* it?

Free generation can't surface knowledge at low fluency (it has to land a long
correct token chain by luck). This instead **teacher-forces** the student over a
documented fact and reads the log-prob it assigns to the **correct** entity vs
plausible **distractors** in the same context. If the fact made it into the
weights — even faintly — the correct entity gets a lower NLL (the model finds it
*less surprising*), regardless of fluency. No teacher, no generation; student-only.

Motivated by `B104` (a real ibuprofen/PPARγ neuronal cell line) appearing once at
α=0.0 — is the association in the weights, or coincidence? This answers it.

A `sanity` fact (capital of France) checks the probe + that the model knows
*anything*; a `control` (wrong-drug) checks we're not just rewarding fluent words.

    python -m tools.knowledge_likelihood_probe --ckpt-dir checkpoints_onpolicy \
        --student-device cuda:0
"""
from __future__ import annotations

import argparse

import torch

from mythouro import MythOuro
from mythouro.checkpointing import list_ckpts
from mythouro.tokenizer import MythOuroTokenizer
from mythouro.variants import (
    mythouro_distill_small,
    mythouro_distill_tiny,
    mythouro_distill_tiny_dense,
    mythouro_distill_xl,
)

_VARIANTS = {
    "mythouro_distill_tiny": mythouro_distill_tiny,
    "mythouro_distill_tiny_dense": mythouro_distill_tiny_dense,
    "mythouro_distill_small": mythouro_distill_small,
    "mythouro_distill_xl": mythouro_distill_xl,
}

# (label, context, correct, [distractors]) — distractors are plausible & real,
# wrong only for THIS context, so a hit means association not generic fluency.
_FACTS = [
    ("sanity:capital", "The capital of France is", " Paris", [" London", " Berlin", " Madrid"]),
    ("ibuprofen:receptor", "Ibuprofen activates the nuclear receptor PPAR", "gamma", ["alpha", "beta", "delta"]),
    ("ibuprofen:cell_pc12", "Ibuprofen activates PPARgamma in the neuron-like cell line", " PC12", [" HEK293", " HeLa", " Jurkat"]),
    ("ibuprofen:cell_b104", "Ibuprofen activates PPARgamma in neuron-like PC12 and", " B104", [" HEK293", " HeLa", " Jurkat"]),
    ("ibuprofen:rhoa", "Activation of PPARgamma by ibuprofen mimics the inhibition of", " RhoA", [" mTOR", " p53", " EGFR"]),
    ("control:wrongdrug", "Metformin activates PPARgamma in neuron-like PC12 and", " B104", [" HEK293", " HeLa", " Jurkat"]),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Knowledge likelihood (cloze) probe.")
    p.add_argument("--ckpt-dir", default="checkpoints_onpolicy")
    p.add_argument("--student-variant", default="mythouro_distill_tiny", choices=list(_VARIANTS))
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--no-sandwich-norm", action="store_true")
    args = p.parse_args()

    dev = args.student_device
    tok = MythOuroTokenizer(args.tokenizer)
    cfg = _VARIANTS[args.student_variant]()
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = args.seq_len
    cfg.use_sandwich_norm = not args.no_sandwich_norm
    cfg.recurrent_state_noise = 0.0
    student = MythOuro(cfg).to(dev)
    ckpts = list_ckpts(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpt_dir!r}")
    ckpt = torch.load(ckpts[-1], map_location=dev, weights_only=False)
    state = {k: v for k, v in ckpt["model"].items()
             if k not in ("freqs_cis", "freqs_cis_mla")}
    student.load_state_dict(state, strict=False)
    student.eval()
    print(f"[lprobe] loaded {ckpts[-1]} (step {ckpt.get('step', '?')}) on {dev}\n")

    @torch.no_grad()
    def nll(context: str, completion: str) -> float:
        """Mean per-token NLL of `completion` given `context` (lower = preferred)."""
        ctx = tok.encode(context)
        comp = tok.encode(completion)
        if not comp:
            return float("inf")
        ids = torch.tensor([ctx + comp], device=dev)
        logits, _ = student(ids[:, :-1], n_loops=args.n_loops)
        logp = torch.log_softmax(logits[0].float(), dim=-1)          # (T-1, V)
        tgt = ids[0, 1:]                                              # (T-1,)
        comp_lp = logp[-len(comp):].gather(-1, tgt[-len(comp):, None]).squeeze(-1)
        return float(-comp_lp.mean())

    hits = 0
    scored = 0
    for (label, ctx, correct, distractors) in _FACTS:
        cands = [(correct, nll(ctx, correct))] + [(d, nll(ctx, d)) for d in distractors]
        cands.sort(key=lambda x: x[1])                               # lowest NLL first
        rank = [c[0] for c in cands].index(correct) + 1
        win = "✅" if rank == 1 else "  "
        is_test = not label.startswith(("sanity", "control"))
        if is_test:
            scored += 1
            hits += (rank == 1)
        print(f"{win} {label:<22} correct={correct.strip()!r:<10} rank {rank}/{len(cands)} "
              f"| NLL {cands[0][1]:.2f}..{cands[-1][1]:.2f}")
        print(f"     order: " + "  ".join(f"{c.strip()}={v:.2f}" for c, v in cands))

    print(f"\nIbuprofen facts the student ranked correctly: {hits}/{scored}")
    print("READ: correct entity ranked #1 (lower NLL than real distractors) = the "
          "association is in the weights. Check `sanity` ranks Paris #1 (probe works) "
          "and `control` (Metformin) does NOT prefer B104 (not just rewarding the word).")


if __name__ == "__main__":
    main()
