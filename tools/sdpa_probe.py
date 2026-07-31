"""Probe the EXACT SDPA call GQAttention makes (main.py:481) on XPU.

The first probe used pre-expanded K/V with equal head counts and passed. The
real call passes UNEXPANDED K/V (4 kv-heads vs 16 q-heads) with enable_gqa=True
— a different kernel, and the likely subject of the "Intel's SDPA kernel
segfaults even with manual KV expansion" note at main.py:48.

  python sdpa_probe2.py <case>    case = native_gqa | expanded | decode
Exit 0 ok, 2 numeric mismatch, 3 exception, 139 segfault (parent decodes).
"""
import sys, time, math
import torch
import torch.nn.functional as F

CASE = sys.argv[1]
dev = "xpu:0"
torch.manual_seed(0)

H, KVH, D = 16, 4, 80                       # mythouro_distill_tiny
GROUPS = H // KVH

# (B, T) — training prefill, and the rollout decode shapes (no KV cache, so the
# sequence grows 16 -> 80 and every step recomputes the whole thing).
SHAPES = {
    "native_gqa": [(8, 1024), (32, 80)],
    "expanded":   [(8, 1024), (32, 80)],
    "decode":     [(32, 16), (32, 48), (32, 80)],
}[CASE]


def manual(q, k, v, causal):
    k_e = k.repeat_interleave(GROUPS, dim=1)
    v_e = v.repeat_interleave(GROUPS, dim=1)
    scale = 1.0 / math.sqrt(q.shape[-1])
    att = (q.float() @ k_e.float().transpose(-2, -1)) * scale
    if causal:
        L = q.shape[-2]
        att = att.masked_fill(
            torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), 1),
            float("-inf"))
    return (att.softmax(-1) @ v_e.float()).to(v.dtype)


def bench(fn, n=8):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


for (B, T) in SHAPES:
    q = torch.randn(B, H, T, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, KVH, T, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, KVH, T, D, device=dev, dtype=torch.bfloat16)
    causal = True
    tag = f"{CASE} B{B} T{T}"

    if CASE == "expanded":
        k_use = k.repeat_interleave(GROUPS, dim=1)
        v_use = v.repeat_interleave(GROUPS, dim=1)
        call = lambda: F.scaled_dot_product_attention(     # noqa: E731
            q, k_use, v_use, dropout_p=0.0, is_causal=causal)
    else:
        call = lambda: F.scaled_dot_product_attention(     # noqa: E731
            q, k, v, dropout_p=0.0, is_causal=causal, enable_gqa=True)

    try:
        out = call()
        torch.xpu.synchronize()
    except Exception as e:                                        # noqa: BLE001
        print(f"  {tag}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")
        sys.exit(3)

    ref = manual(q, k, v, causal)
    rel = ((out.float() - ref.float()).abs().max()
           / ref.float().abs().max().clamp(min=1e-6)).item()
    t_s = bench(call)
    t_m = bench(lambda: manual(q, k, v, causal))
    print(f"  {tag}: SDPA {t_s:7.3f} ms | manual {t_m:7.3f} ms | "
          f"speedup {t_m / t_s:5.2f}x | rel_err {rel:.4f} "
          f"{'OK' if rel < 0.05 else 'MISMATCH'}")
    if rel >= 0.05:
        sys.exit(2)

print(f"  {CASE}: PASS")
