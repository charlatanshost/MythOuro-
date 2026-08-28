"""
Minimal repro of a COMPLETE training step at 24 vs 48 experts — no dataset.

Everything else at 48 experts passes (2026-08-28):
  bench_step fwd+bwd batch 8 seq 1024 ....... OK, 727 ms/step
  forward with router_bias[24:] = -100 ...... OK
  generate_rollout uncached, ACT live ....... OK
  teacher resident + rollout alpha=0.45 ..... OK, peak 21.4 GB of 48
  rollout-batch 32 -> 8 ..................... still segfaults
  micro-batch 8 -> 4 ........................ still segfaults
  278M control, identical path .............. RUNS CLEAN

So the fault is in the first COMPLETE step and is specific to 48 experts. This
builds that step directly — optimizer, teacher logits, distillation loss,
backward, optimizer step, router-bias update — and prints a marker before each
stage so the segfault names its own location.

    python -m tools.repro_step --device xpu:0 --experts 24   # control
    python -m tools.repro_step --device xpu:0 --experts 48   # the failing case
"""
from __future__ import annotations
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mythouro.variants import mythouro_distill_tiny, mythouro_distill_small  # noqa: E402
from mythouro.main import MythOuro, MoEFFN                                    # noqa: E402
from mythouro.training_utils import (                                         # noqa: E402
    load_distillation_teacher, update_router_bias_from_counts)


def mark(s):
    print(f"  >> {s}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--experts", type=int, choices=(24, 48), default=48)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--sandwich", action="store_true",
                   help="cfg.use_sandwich_norm = True, as distill.py sets it.")
    p.add_argument("--depth-aware-init", action="store_true",
                   help="cfg.use_depth_aware_init = True, as distill.py sets it.")
    p.add_argument("--sentinel", action="store_true",
                   help="Apply router_bias[24:]=-100 as a promoted ckpt would.")
    a = p.parse_args()
    dev = a.device

    cfg = mythouro_distill_tiny() if a.experts == 24 else mythouro_distill_small()
    # distill.py MUTATES cfg from CLI flags before building. Every earlier repro
    # and bench_step built the RAW variant, i.e. a different model than trains.
    cfg.max_seq_len = a.seq_len
    cfg.use_sandwich_norm = a.sandwich
    cfg.use_depth_aware_init = a.depth_aware_init
    mark(f"build {cfg.n_experts}-expert model "
         f"(max_seq_len={cfg.max_seq_len} sandwich={cfg.use_sandwich_norm} "
         f"depth_aware_init={cfg.use_depth_aware_init})")
    model = MythOuro(cfg).to(dev)
    model.train()

    if a.sentinel and a.experts == 48:
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, MoEFFN):
                    m.router_bias[24:] = -100.0
        mark("applied sentinel router_bias[24:] = -100")

    mark("build optimizer (Adam over all params)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    mark("load teacher")
    teacher = load_distillation_teacher(
        "ByteDance/Ouro-2.6B-Thinking", student_vocab_size=cfg.vocab_size,
        device=dev, dtype=torch.bfloat16, trust_remote_code=True)

    ids = torch.randint(0, cfg.vocab_size, (a.batch, a.seq_len), device=dev)

    mark("teacher forward")
    with torch.no_grad():
        t_out = teacher(ids)
        t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
    mark(f"teacher logits {tuple(t_logits.shape)}")

    mark("student forward (train mode, grad on)")
    with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
        s_out = model(ids)
        s_logits = s_out[0] if isinstance(s_out, tuple) else s_out
    mark(f"student logits {tuple(s_logits.shape)}")

    mark("distillation loss (rev_kl)")
    lp_s = torch.log_softmax(s_logits.float(), -1)
    lp_t = torch.log_softmax(t_logits.float(), -1)
    loss = (lp_s.exp() * (lp_s - lp_t)).sum(-1).mean()
    mark(f"loss {float(loss):.4f}")

    mark("backward")
    loss.backward()

    mark("optimizer step")
    opt.step(); opt.zero_grad(set_to_none=True)

    mark("router bias update")
    counts = {}
    for name, m in model.named_modules():
        if isinstance(m, MoEFFN) and getattr(m, "_last_expert_counts", None) is not None:
            counts[name] = m._last_expert_counts
    if counts:
        update_router_bias_from_counts(model, counts, bias_lr=1e-3, ddp=False)
    mark(f"updated {len(counts)} MoE layer(s)")

    if dev.startswith("xpu"):
        torch.xpu.synchronize()
        mark(f"peak {torch.xpu.max_memory_allocated()/1e9:.2f} GB")
    print(f"\n  COMPLETE STEP OK at {cfg.n_experts} experts", flush=True)


if __name__ == "__main__":
    main()
