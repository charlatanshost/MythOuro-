#!/usr/bin/env python
"""
Knowledge probe — does the STUDENT carry correct domain knowledge, separate from
whether it's *fluent*?

Generates many **pure-student (α=0.0)** rollouts for TOPIC prompts vs CONTROL
prompts and counts target-entity hits in each. A real learned association shows
up as the entities **clustering in the topic and ~absent in control**; a
coincidence shows up as **uniform** hits across both.

Motivated by the model emitting `B104` — a real rat neuroblastoma line tied to
ibuprofen/PPARγ/RhoA neurite-outgrowth research — at α=0.0 inside otherwise
incoherent text. Is that retained knowledge or a lucky alphanumeric? This tells us.

α=0.0 needs no teacher, so it's student-only and can run on a spare card (e.g.
`--student-device cuda:1`) *while training continues* on cuda:0/cuda:2.

Example:
    python -m tools.knowledge_probe --ckpt-dir checkpoints_onpolicy \
        --student-device cuda:1 --samples 10
"""
from __future__ import annotations

import argparse

import torch

from mythouro import MythOuro
from mythouro.checkpointing import list_ckpts
from mythouro.tokenizer import MythOuroTokenizer
from mythouro.training_utils import generate_rollout, load_distillation_teacher
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

# default cluster: ibuprofen / PPARγ pharmacology
_TOPIC = [
    "Ibuprofen is a nonsteroidal anti-inflammatory drug that works by",
    "In neuronal cells, ibuprofen has been shown to activate",
    "Ibuprofen and other NSAIDs can promote neurite outgrowth by",
]
_CONTROL = [
    "In the morning the weather was clear, so we decided to",
    "The history of the old city began when",
    "My favorite recipe for dinner starts with",
]
# lowercase substrings counted as "correct domain entities"; the SPECIFIC ones
# (b104/pc12/ppar/rhoa) are the discriminators — they should be ~0 in control.
_ENTITIES = ["ppar", "b104", "pc12", "rhoa", "prostagl", "prosto",
             "cyclo", "neurite", "agonist", "nsaid"]


def main() -> None:
    p = argparse.ArgumentParser(description="Knowledge probe (entity surfacing, pure student).")
    p.add_argument("--ckpt-dir", default="checkpoints_onpolicy")
    p.add_argument("--student-variant", default="mythouro_distill_tiny", choices=list(_VARIANTS))
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--teacher-device", default="cuda:2")
    p.add_argument("--teacher-id", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--tokenizer", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Teacher-mix. 0.0 = pure student (tests the STUDENT; teacher not "
                        "loaded). >0 brings the teacher in and contaminates the knowledge test.")
    p.add_argument("--samples", type=int, default=10, help="Rollouts per prompt.")
    p.add_argument("--rollout-len", type=int, default=96)
    p.add_argument("--seed-len", type=int, default=20)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--no-sandwich-norm", action="store_true")
    p.add_argument("--topic-seeds", nargs="+", default=_TOPIC)
    p.add_argument("--control-seeds", nargs="+", default=_CONTROL)
    p.add_argument("--entities", nargs="+", default=_ENTITIES)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    sdev = args.student_device
    tok = MythOuroTokenizer(args.tokenizer)

    cfg = _VARIANTS[args.student_variant]()
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = args.seq_len
    cfg.use_sandwich_norm = not args.no_sandwich_norm
    cfg.recurrent_state_noise = 0.0
    student = MythOuro(cfg).to(sdev)
    ckpts = list_ckpts(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpt_dir!r}")
    ckpt = torch.load(ckpts[-1], map_location=sdev, weights_only=False)
    state = {k: v for k, v in ckpt["model"].items()
             if k not in ("freqs_cis", "freqs_cis_mla")}
    student.load_state_dict(state, strict=False)
    student.eval()
    print(f"[kprobe] loaded {ckpts[-1]} (step {ckpt.get('step', '?')}) on {sdev}")

    teacher = None
    if args.alpha > 0.0:
        teacher = load_distillation_teacher(
            args.teacher_id, student_vocab_size=tok.vocab_size,
            device=args.teacher_device, dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code)
        if teacher is None:
            raise SystemExit("teacher failed to load")
    mode = "pure student" if args.alpha == 0 else f"teacher-mixed α={args.alpha}"
    print(f"[kprobe] {mode}, {args.samples} samples/prompt, entities={args.entities}\n")

    def run(seeds):
        counts = {e: 0 for e in args.entities}
        examples, total = [], 0
        for seed_text in seeds:
            ids = tok.encode(seed_text)[: args.seed_len]
            prompt = torch.tensor([ids], device=sdev)
            for _ in range(args.samples):
                roll = generate_rollout(
                    student, teacher, prompt, n_loops=args.n_loops,
                    max_new_tokens=args.rollout_len, teacher_mix_alpha=args.alpha,
                    temperature=args.temp, top_k=args.top_k)
                text = tok.decode(roll[0, len(ids):].tolist())
                hits = [e for e in args.entities if e in text.lower()]
                for e in hits:
                    counts[e] += 1
                if hits:
                    examples.append((seed_text, hits, text))
                total += 1
        return total, counts, examples

    print("running TOPIC ...")
    t_n, t_counts, t_ex = run(args.topic_seeds)
    print("running CONTROL ...")
    c_n, c_counts, c_ex = run(args.control_seeds)

    print(f"\n=== entity hit counts (α={args.alpha}) ===")
    print(f"{'entity':<12} {'topic':>10} {'control':>10}")
    for e in args.entities:
        print(f"{e:<12} {f'{t_counts[e]}/{t_n}':>10} {f'{c_counts[e]}/{c_n}':>10}")
    print(f"{'TOTAL':<12} {f'{sum(t_counts.values())}/{t_n}':>10} "
          f"{f'{sum(c_counts.values())}/{c_n}':>10}")

    print(f"\n=== TOPIC rollouts that surfaced an entity ({len(t_ex)}) ===")
    for (seed_text, hits, text) in t_ex[:10]:
        print(f"\n[{','.join(hits)}] {seed_text!r}\n  {text!r}")
    if c_ex:
        print(f"\n=== CONTROL hits ({len(c_ex)}) — these would argue COINCIDENCE ===")
        for (seed_text, hits, text) in c_ex[:6]:
            print(f"\n[{','.join(hits)}] {seed_text!r}\n  {text!r}")

    print("\nREAD: specific entities (ppar/b104/pc12/rhoa) in TOPIC and ~absent in CONTROL "
          "= real learned association. Uniform across both = coincidental spam.")


if __name__ == "__main__":
    main()
