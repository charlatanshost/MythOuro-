"""
Minimal repro: does a 48-expert MythOuro forward segfault when router_bias
gates half the experts off?

The failing training runs and the passing bench_step runs differ in exactly one
way that reaches the MoE: bench_step builds a RANDOM model (router_bias all
zeros, so all 48 experts receive traffic), while the promoted checkpoint carries
router_bias[24:] = -100.0, so top-k NEVER selects experts 24-47 and they receive
zero tokens.

This builds the same variant bench_step does, then runs one forward with each
bias setting. No checkpoint, no teacher, no dataset — if the crash reproduces
here, it is the routing, and everything else tonight is a red herring.

    python -m tools.repro_sentinel --device xpu:0
"""
from __future__ import annotations
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mythouro.variants import mythouro_distill_small          # noqa: E402
from mythouro.main import MythOuro, MoEFFN                     # noqa: E402


def run(model, dev, tag, seq, batch):
    ids = torch.randint(0, 1000, (batch, seq), device=dev)
    print(f"  [{tag}] forward batch={batch} seq={seq} ...", flush=True)
    with torch.no_grad(), torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
        model(ids)
    if dev.startswith("xpu"):
        torch.xpu.synchronize()
    print(f"  [{tag}] OK", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--with-teacher", action="store_true",
                   help="Load Ouro-2.6B alongside and report memory. The three\n"
                        "student paths pass at 48 experts; the crash needs this.")
    a = p.parse_args()

    cfg = mythouro_distill_small()
    print(f"  building {cfg.n_experts}-expert model on {a.device} ...", flush=True)
    model = MythOuro(cfg).to(a.device).eval()

    n_moe = sum(1 for m in model.modules() if isinstance(m, MoEFFN))
    print(f"  MoEFFN modules: {n_moe}", flush=True)

    # 1. control — bias all zeros, every expert reachable (what bench_step tests)
    run(model, a.device, "bias=0 (all 48 reachable)", a.seq_len, a.batch)

    # 2. the promoted-checkpoint condition — half the experts gated off
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MoEFFN):
                m.router_bias[cfg.n_experts // 2:] = -100.0
    run(model, a.device, "bias=-100 on experts 24-47", a.seq_len, a.batch)

    # 3. THE PATH NEITHER bench_step NOR THE FORWARDS ABOVE COVER.
    #    bench_step runs TRAIN mode (it does a backward), and ACT early-exit is
    #    gated on `not self.training` — so it never fires there. The forwards
    #    above are eval mode but single, fixed-length. Rollout generation is eval
    #    mode, AUTOREGRESSIVE, uncached, with ACT deciding depth per token as the
    #    sequence grows. That is where the training runs spent 11 minutes before
    #    dying, and it is the only major path left untested at 48 experts.
    from mythouro.training_utils import generate_rollout
    prompt = torch.randint(0, 1000, (8, 64), device=a.device)
    print("  [rollout] generate_rollout batch=8 len=64, uncached, ACT live ...", flush=True)
    with torch.no_grad():
        generate_rollout(model, None, prompt, n_loops=4, max_new_tokens=64,
                         teacher_mix_alpha=0.0, temperature=1.0, top_k=50,
                         use_kv_cache=False)
    if a.device.startswith("xpu"):
        torch.xpu.synchronize()
    print("  [rollout] OK", flush=True)

    if not a.with_teacher:
        print("\n  ALL STUDENT PATHS PASSED. Re-run with --with-teacher.", flush=True)
        return

    def mem():
        if a.device.startswith("xpu"):
            return torch.xpu.memory_allocated() / 1e9
        return 0.0

    print(f"\n  student-only allocated: {mem():.2f} GB", flush=True)
    from mythouro.training_utils import load_distillation_teacher
    teacher = load_distillation_teacher(
        "ByteDance/Ouro-2.6B-Thinking", student_vocab_size=cfg.vocab_size,
        device=a.device, dtype=torch.bfloat16, trust_remote_code=True)
    print(f"  + teacher resident:      {mem():.2f} GB", flush=True)

    # teacher forward at the training micro-batch shape
    ids = torch.randint(0, 1000, (a.batch, a.seq_len), device=a.device)
    print("  [teacher] forward at training shape ...", flush=True)
    with torch.no_grad():
        teacher(ids)
    if a.device.startswith("xpu"):
        torch.xpu.synchronize()
    print(f"  [teacher] OK — peak {torch.xpu.max_memory_allocated()/1e9:.2f} GB", flush=True)

    # and the combination the trainer actually does: rollout with teacher mixing
    print("  [rollout+teacher] alpha=0.45, the training path ...", flush=True)
    prompt = torch.randint(0, 1000, (8, 64), device=a.device)
    with torch.no_grad():
        generate_rollout(model, teacher, prompt, n_loops=4, max_new_tokens=64,
                         teacher_mix_alpha=0.45, temperature=1.0, top_k=50,
                         use_kv_cache=False)
    if a.device.startswith("xpu"):
        torch.xpu.synchronize()
    print(f"  [rollout+teacher] OK — peak {torch.xpu.max_memory_allocated()/1e9:.2f} GB", flush=True)
    print("\n  ALL PASSED WITH TEACHER — memory is not the limit; look elsewhere.", flush=True)


if __name__ == "__main__":
    main()
