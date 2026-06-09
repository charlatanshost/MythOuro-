from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN_IMPORT = True
except ImportError:
    flash_attn_func = None
    _HAS_FLASH_ATTN_IMPORT = False


class _Capabilities:
    """
    Module-level probe of attention-kernel availability.

    Three tiers, checked in order at every attention call:

        1. **flash_attn_func** — fastest, but Ampere+ only (CC ≥ 8.0).
           Volta (V100, CC 7.0) imports flash-attn fine but its kernels
           crash at launch, so a CC check is required in addition to
           the import test.
        2. **torch.nn.functional.scaled_dot_product_attention** —
           PyTorch's fused kernel. Works on all CUDA hardware and CPU,
           handles GQA's KV-head expansion natively, and has a clean
           causal-mask flag.
        3. **manual matmul** — the portable fallback for old PyTorch
           builds. Slow but always correct.

    `warn_once(key, msg)` emits each unique key at WARNING level exactly
    once over the process lifetime — keeps the cascade audible without
    drowning the per-step log line every forward pass.
    """

    def __init__(self):
        self.has_flash_attn_import: bool = _HAS_FLASH_ATTN_IMPORT
        self.has_sdpa: bool = hasattr(F, "scaled_dot_product_attention")
        self.cuda_cc: "tuple[int, int] | None" = self._probe_cuda_cc()
        self.fa2_usable: bool = self._fa2_usable()
        self._warned: set[str] = set()

    @staticmethod
    def _probe_cuda_cc() -> "tuple[int, int] | None":
        if not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.get_device_capability(torch.cuda.current_device())
        except Exception:                                       # noqa: BLE001
            return None

    def _fa2_usable(self) -> bool:
        # Three conditions must all hold:
        #   1. flash-attn imported cleanly,
        #   2. CUDA is available (FA2 is GPU-only),
        #   3. compute capability ≥ 8.0 (Ampere — A100, H100, Ada,
        #      Blackwell). Volta / Turing import but crash at launch.
        if not self.has_flash_attn_import:
            return False
        if self.cuda_cc is None:
            return False
        return self.cuda_cc[0] >= 8

    def warn_once(self, key: str, msg: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(msg)

    # Test seam: pytest swaps these to drive each branch deterministically.
    def reset_warnings(self) -> None:
        self._warned.clear()


CAPABILITIES = _Capabilities()
# Backwards-compatible alias — older diagnostics imported _HAS_FLASH_ATTN
# directly. Now derives from the live capability flag so a unit test that
# patches CAPABILITIES is also reflected here.
_HAS_FLASH_ATTN = CAPABILITIES.fa2_usable


@dataclass
class MythOuroConfig:
    """
    Hyperparameter configuration for MythOuro.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    # T — recurrent depth at inference. Ouro (Zhu et al. 2025, 1.4B/2.6B
    # looped LMs trained on 7.7T tokens) found peak accuracy on hard
    # reasoning benchmarks at 3-4 loops, with measurable *degradation*
    # past ~8. Default lowered from 16 → 6 to reflect that finding;
    # raise per-config only if you have a specific reason.
    max_loop_iters: int = 6
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0

    # --- Part 1 additions -----------------------------------------------------
    # Attention sink — learnable register tokens prepended before prelude and
    # stripped after coda. Stabilises attention distributions in deep recurrent
    # stacks (Xiao et al., 2024).
    n_sink_tokens: int = 4
    # Gradient checkpointing inside the recurrent loop — trades a second
    # transformer forward for O(1) activation memory in n_loops.
    gradient_checkpointing: bool = True
    # Convergence-detection early exit. Stops looping once ‖h_{t+1} − h_t‖
    # drops below this threshold at every position (inference only, since
    # training needs deterministic depth for grad-checkpoint replay).
    convergence_eps: float = 1e-4
    # --- Part 2 features (enabled by default) ---
    # Multi-scale injection: blend fine/coarse/global views of the encoded
    # input across loop depth. Pays one extra Linear per scale per loop.
    use_multiscale_injection: bool = True
    ms_window_size: int = 8
    # Cross-loop attention: small attention over a buffer of prior loop
    # states, giving the model an explicit handle on its reasoning history.
    use_cross_loop_attention: bool = True
    cross_loop_store_every: int = 4
    # Initial shape of the InjectionScheduler magnitude per loop.
    injection_decay: str = "cosine"  # "cosine" | "linear" | "constant"
    # DeepSeek-V3 aux-loss-free routing: per-step bias increment magnitude.
    # Bias is nudged ± this value per expert per macro-step toward uniform
    # utilisation. Too high → routing oscillates; too low → bias never
    # catches up. 1e-3 matches the DeepSeek-V3 paper.
    router_bias_lr: float = 1e-3
    # §3 new-component warmup: linear LR ramp 0→1 over this many steps for
    # parameters that are non-zero-init at step 0 (InjectionScheduler,
    # LoRAAdapter.down, MultiScaleInjection) so they don't destabilise the
    # base block before they've had a chance to converge to a useful state.
    # Zero-output-init components (CrossLoopAttention, UncertaintyHead,
    # ProcessRewardHead) self-warm and stay in the base LR group.
    new_component_warmup_steps: int = 2000
    # PonderNet-style halt-distribution regulariser. Pulls the per-token
    # P(halt at step n) distribution toward a uniform prior over depths
    # via KL — matches Ouro's "entropy-regularized depth allocation" when
    # the prior is uniform (Ouro found geometric prior under-trains late
    # loops, picked uniform). Set to 0.0 to disable (default; the current
    # ACT mechanism still drives halting on its own). Recommended range
    # when enabling: 1e-3 to 1e-2.
    depth_reg_coeff: float = 0.0

    # Ablation: replace the recurrent block's MoE FFN with a single dense
    # SwiGLU FFN, to test whether sparsity earns its keep at matched compute.
    # When True, the recurrent FFN is `Expert(dim, recurrent_dense_ffn_dim)`
    # instead of `MoEFFN`. Default width (when recurrent_dense_ffn_dim == 0) is
    # `expert_dim * n_experts_per_tok * (1 + n_shared_experts)`, which makes the
    # dense FFN's parameters/FLOPs per token equal to the MoE arm's *activated*
    # FFN per token — the matched-compute comparison spec'd in
    # docs/roadmap.md ("Gating experiment: MoE-vs-dense ablation").
    recurrent_dense: bool = False
    recurrent_dense_ffn_dim: int = 0

    # Use a real-valued (cos/sin) RoPE table instead of the default complex one,
    # for backends without complex-tensor op support (e.g. some Intel XPU
    # coverage). Mathematically identical rotation — safe to flip on a checkpoint
    # trained the other way. See precompute_rope_freqs / apply_rope.
    rope_real: bool = False


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim -- feature dimension to normalize over
            eps -- small constant added before sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by self.weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0, real: bool = False
) -> torch.Tensor:
    """
    Precompute RoPE rotation tables for positions 0..max_len-1.

    Each position gets a phasor e^{i·m·θ_k} for each frequency pair k.

    Args:
        dim     -- head dimension (must be even); frequencies are computed for dim//2 pairs
        max_len -- maximum sequence length to precompute
        theta   -- RoPE base (higher = slower frequency decay; 500k is the LLaMA-3 default)
        real    -- if False (default), return a **complex64** tensor of shape
                   (max_len, dim//2) — rotation is a single complex multiply.
                   If True, return a **real** tensor of shape (max_len, dim//2, 2)
                   holding (cos, sin), for backends without complex-tensor support
                   (e.g. some Intel XPU op coverage). The two encode the *same*
                   rotation — `apply_rope` dispatches on the tensor's dtype, so a
                   checkpoint trained one way runs identically the other way.

    Returns:
        complex64 (max_len, dim//2)  OR  float32 (max_len, dim//2, 2) if real.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    if real:
        return torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Interprets each pair of adjacent features as a 2D rotation and rotates it by
    the precomputed phasor for that position, without changing its norm.

    Two backends, selected by `freqs_cis`'s dtype (see `precompute_rope_freqs`):
      - **complex** `freqs_cis` (T, head_dim//2): a single complex multiply.
      - **real** `freqs_cis` (T, head_dim//2, 2) holding (cos, sin): the same
        rotation in pure real arithmetic — no `view_as_complex` / `polar`, for
        backends where complex ops are unsupported or slow (Intel XPU).

    Args:
        x         -- tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis -- precomputed rotation table, already sliced to the positions
                     being processed (caller handles the start_pos offset)

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    if freqs_cis.is_complex():
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return (
            torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
            .flatten(-2)
            .to(x.dtype)
        )
    # Real path: freqs_cis[..., 0] = cos, [..., 1] = sin, shape (T, head_dim//2).
    xr = x.float().reshape(*x.shape[:-1], -1, 2)          # (B, T, H, D//2, 2)
    x_even, x_odd = xr[..., 0], xr[..., 1]                # (B, T, H, D//2)
    cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(2)     # (1, T, 1, D//2)
    sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(2)
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023) with Flash Attention 2 (Dao et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    When flash-attn is installed, uses flash_attn_func which handles GQA natively
    (no KV head expansion needed) and is IO-bound-optimal. Inputs are cast to
    bfloat16 for flash_attn and restored to the original dtype afterward.
    Falls back to manual scaled dot-product attention when flash-attn is absent.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.
    """

    def __init__(self, cfg: MythOuroConfig):
        """
        Args:
            cfg -- MythOuroConfig; uses dim, n_heads, n_kv_heads
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        # Three-tier attention cascade. The capability probe at module
        # load time picked the best available kernel; here we dispatch
        # accordingly and emit a warn-once telling the operator which
        # path is hot so a misconfigured environment is visible at
        # startup rather than buried in throughput numbers.
        dropout_p = self.dropout_p if self.training else 0.0

        if CAPABILITIES.fa2_usable:
            # flash_attn_func expects (B, T, H, head_dim) — GQA is handled
            # natively (n_kv_heads < n_heads needs no repeat_interleave).
            # causal=True when a mask is present (prefill/training);
            # causal=False for single-token decode where T=1 / mask is None.
            orig_dtype = q.dtype
            q_f = q.to(torch.bfloat16)
            k_f = k.to(torch.bfloat16)
            v_f = v.to(torch.bfloat16)
            out = flash_attn_func(
                q_f, k_f, v_f,
                dropout_p=dropout_p,
                causal=(mask is not None),
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        elif CAPABILITIES.has_sdpa:
            # PyTorch fused SDPA — works everywhere FA2 doesn't (Volta,
            # Turing, CPU). Native GQA support via `enable_gqa=True`
            # avoids the repeat_interleave overhead when the build is
            # new enough; older builds fall through to expansion below.
            CAPABILITIES.warn_once(
                "gqa_sdpa",
                "GQAttention: using torch SDPA fallback "
                f"(FA2 unavailable; cuda_cc={CAPABILITIES.cuda_cc})."
            )
            q_t = q.transpose(1, 2)                                   # (B, H, T, head_dim)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            try:
                out = F.scaled_dot_product_attention(
                    q_t, k_t, v_t,
                    dropout_p=dropout_p,
                    is_causal=(mask is not None),
                    enable_gqa=True,
                )
            except TypeError:
                # Pre-2.3 torch: no enable_gqa kwarg → expand KV heads.
                k_t = k_t.repeat_interleave(self.groups, dim=1)
                v_t = v_t.repeat_interleave(self.groups, dim=1)
                out = F.scaled_dot_product_attention(
                    q_t, k_t, v_t,
                    dropout_p=dropout_p,
                    is_causal=(mask is not None),
                )
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        else:
            # Pure-PyTorch manual SDPA — slowest but always correct,
            # required for very old torch builds without fused SDPA.
            CAPABILITIES.warn_once(
                "gqa_manual",
                "GQAttention: using manual SDPA fallback "
                "(neither FA2 nor torch SDPA available)."
            )
            k_e = k.repeat_interleave(self.groups, dim=2)
            v_e = v.repeat_interleave(self.groups, dim=2)
            q_t = q.transpose(1, 2)                                   # (B, H, T, head_dim)
            k_t = k_e.transpose(1, 2)
            v_t = v_e.transpose(1, 2)
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = F.dropout(
                F.softmax(attn, dim=-1),
                p=self.dropout_p, training=self.training,
            )
            out = torch.matmul(attn, v_t)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.
    """

    def __init__(self, cfg: MythOuroConfig):
        """
        Args:
            cfg -- MythOuroConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # MLA cascade. Skips FA2 because MLA's nope+rope key concatenation
        # doesn't map onto flash_attn's API as cleanly as it does on
        # SDPA, where head_dim mismatch between K and V is supported.
        # Two-tier: SDPA → manual.
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        dropout_p = self.attn_drop.p if self.training else 0.0
        if CAPABILITIES.has_sdpa:
            # SDPA handles the K/V head-dim mismatch (Q,K share q_head_dim,
            # V has its own v_head_dim) and applies the causal mask via
            # its own kernel when `is_causal=True`. The per-call cost is
            # lower than the manual matmul path at non-trivial seq lens.
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=(mask is not None),
            )
        else:
            CAPABILITIES.warn_once(
                "mla_manual",
                "MLAttention: using manual SDPA fallback "
                "(torch SDPA unavailable)."
            )
            scale = self.q_head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = self.attn_drop(F.softmax(attn, dim=-1))
            out = torch.matmul(attn, v)  # (B, H, T, v_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
        """
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K of them
      via a learned router. A per-expert bias on router logits is updated during
      training to keep load balanced across experts without distorting the loss.
    - Shared experts: n_shared_experts larger FFNs always activated for every token,
      absorbing common cross-domain patterns (syntax, basic reasoning) that would
      otherwise be redundantly learned by many routed experts.

    Total activated parameters per token ≈ topk/n_experts of routed + all shared,
    keeping compute sparse while the total parameter count stays large.
    """

    def __init__(self, cfg: MythOuroConfig):
        """
        Args:
            cfg -- MythOuroConfig; uses n_experts, n_shared_experts, n_experts_per_tok,
                   dim, expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # load-balancing bias adjusted externally during training; not a gradient param
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

        # Per-call (overwritten) telemetry, read by `_loop_body`.
        self._last_router_logits: "Optional[torch.Tensor]" = None
        self._last_expert_counts: "Optional[torch.Tensor]" = None
        # Per-forward accumulators populated by RecurrentBlock across ALL loops
        # (P0.2). The router logits buffer keeps gradient (aux losses); the
        # expert-count sum is detached. Read by collect_router_logits /
        # collect_expert_counts.
        self._router_logits_buf: "list[torch.Tensor]" = []
        self._expert_counts_sum: "Optional[torch.Tensor]" = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (B, T, dim)
        Returns:
            Tensor of shape (B, T, dim); shared expert outputs are summed on top
            of the weighted routed expert outputs

        Implementation notes:
            Vectorised scatter dispatch — each expert runs exactly once per
            forward (the previous nested loop called each expert up to `topk`
            times). Tokens routed to the same expert across multiple top-k
            slots are batched into a single expert call and combined via
            `index_add_`, which is fused and gradient-safe.

            Router logits are stashed on `self._last_router_logits` so that
            auxiliary objectives (load balancing, sparse activation,
            specialisation probing) can read them without re-running the
            router or re-plumbing the forward signature.
        """
        B, T, D = x.shape
        N = B * T
        flat = x.view(N, D)

        # Aux-loss-free load balancing (DeepSeek-V3): the bias shifts only the
        # selection of which experts fire so underused experts are picked more,
        # but the gating weights come from unbiased softmax scores so the bias
        # never shows up in the gradient.
        logits = self.router(flat)                                # (N, E)
        self._last_router_logits = logits                         # exposed for aux losses
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renorm

        # Per-expert dispatch counts, computed once from the topk decision.
        # Exposed so the trainer can drive the aux-loss-free bias updater
        # — without it the `router_bias` buffer above is dead code and
        # this layer's "aux-loss-free balancing" claim is fiction.
        self._last_expert_counts = (
            torch.bincount(topk_idx.flatten(), minlength=self.n_experts)
            .detach()
        )

        # Flatten the top-k dim so dispatch becomes a single 1-D scatter.
        # (token i, slot k) → flat position i*K + k.
        K = self.topk
        flat_expert_ids = topk_idx.reshape(-1)                    # (N*K,)
        flat_gate       = topk_scores.reshape(-1).unsqueeze(-1)   # (N*K, 1)
        token_index = (
            torch.arange(N, device=x.device).repeat_interleave(K)
        )                                                          # (N*K,)

        out = torch.zeros_like(flat)
        for eid in range(self.n_experts):
            sel = flat_expert_ids == eid
            if not sel.any():
                continue
            tok = token_index[sel]                                # (M,)
            gate = flat_gate[sel]                                 # (M, 1)
            expert_out = self.routed_experts[eid](flat[tok]) * gate
            # `.to(out.dtype)` keeps the scatter dtype-consistent under
            # autocast (where the expert Linear emits bf16 but `out` may be the
            # fp32 input activation). No-op when dtypes already match — i.e. on
            # the native-bf16 / fp32 paths.
            out.index_add_(0, tok, expert_out.to(out.dtype))

        # shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat).to(out.dtype)

        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Loop-index RoPE (differentiates recurrent block across iterations)
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on. Adding the loop index lets
    the same parameters implement functionally distinct operations per iteration.

    Args:
        h        -- hidden state tensor of shape (B, T, dim)
        loop_t   -- current loop iteration index (0-based)
        loop_dim -- number of leading channels to receive the embedding (must be even)
        theta    -- sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels; same shape
    """
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter (per loop iteration)
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block, v2 — per-loop B matrix.

    Earlier versions tied the up-projection B across loops and modulated only a
    per-loop scale vector, which forces every iteration to project into the
    same output subspace and limits the kind of work the loop can specialise
    each iteration to do.

    v2 keeps the A (down-projection) shared so the bottleneck stays small,
    but learns a distinct B[t] for every loop iteration. This gives each
    loop its own output direction without paying the full cost of a
    per-loop adapter.

    delta(x, t) = down(x) * scale[t] @ B[t]

    Parameter cost (rank=16, dim=2048, max_loops=16):
        v1:  rank*dim  (shared B)            +  max_loops*rank   (scales) ≈   33k
        v2:  max_loops*rank*dim (per-loop B) +  rank*dim         (shared A) ≈ 537k
    The increase is the price of genuine per-loop specialisation.
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim       -- model hidden dimension (input and output size)
            rank      -- low-rank bottleneck dimension
            max_loops -- maximum number of loop iterations (determines per-loop table sizes)
        """
        super().__init__()
        self.rank = rank
        self.max_loops = max_loops

        self.down = nn.Linear(dim, rank, bias=False)              # shared A: dim → rank
        # Per-loop up-projection. Zero-init so the adapter starts as an
        # identity perturbation and the recurrent block can train freely
        # before LoRA contributes anything.
        self.B = nn.Parameter(torch.zeros(max_loops, rank, dim))
        # Per-loop scale, kept for fine-grained magnitude control. ones-init
        # so the adapter's effective transform at step 0 is `down @ B[t]`.
        self.scale = nn.Embedding(max_loops, rank)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.ones_(self.scale.weight)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x      -- input tensor of shape (B, T, dim)
            loop_t -- current loop index; clamped to max_loops-1 for depth
                      extrapolation (inference can run more loops than training).

        Returns:
            Delta tensor of shape (B, T, dim) to be added to the block output.
        """
        t_idx = min(loop_t, self.max_loops - 1)
        s = self.scale(torch.tensor(t_idx, device=x.device))       # (rank,)
        down = self.down(x) * s                                    # (B, T, rank)
        B_t = self.B[t_idx]                                        # (rank, dim)
        return down @ B_t                                          # (B, T, dim)


# ---------------------------------------------------------------------------
# Single Transformer Block (shared across recurrent loops)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(
        self,
        cfg: MythOuroConfig,
        use_moe: bool = False,
        dense_ffn_dim: Optional[int] = None,
    ):
        """
        Args:
            cfg           -- MythOuroConfig; attn_type selects the attention class
            use_moe       -- if True, use MoEFFN; otherwise use a dense Expert FFN
            dense_ffn_dim -- inner width of the dense Expert FFN when use_moe is
                             False. Defaults to the prelude/coda width
                             `dim * 4 // 3`; the recurrent dense ablation passes
                             the matched-active width explicitly.
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        if use_moe:
            self.ffn = MoEFFN(cfg)
        else:
            self.ffn = Expert(cfg.dim, dense_ffn_dim or (cfg.dim * 4 // 3))
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- precomputed RoPE frequencies
            mask      -- additive causal mask or None
            kv_cache  -- cache dict mutated in-place by the attention layer
            cache_key -- key identifying this layer in the cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# LTI-stable injection parameters  (spectral radius < 1 by construction)
# ---------------------------------------------------------------------------


class InjectionScheduler(nn.Module):
    """
    Per-loop scalar magnitude schedule for input injection.

    Replaces the constant B scalar in LTIInjection with a learned positive
    scalar per loop iteration. Early loops typically benefit from strong
    injection (anchor to input); late loops benefit from weak injection
    (reason freely). Rather than hardcode that, we initialise to a
    cosine decay and let the model fit its own shape during training.

    Implementation: a learned per-loop log-scale, exponentiated to stay
    positive. Out-of-range loop indices clamp to the last learned value
    so depth extrapolation reuses the deepest schedule entry.
    """

    SHAPES = {"cosine", "linear", "constant"}

    def __init__(self, max_loops: int, init_decay: str = "cosine"):
        super().__init__()
        if init_decay not in self.SHAPES:
            raise ValueError(
                f"init_decay must be one of {self.SHAPES}, got {init_decay!r}"
            )
        self.max_loops = max_loops
        init = self._make_init(max_loops, init_decay)
        # Storing in log space keeps the schedule positive after any
        # gradient step without needing a constraint or clamp at runtime.
        self.log_scale = nn.Parameter(torch.log(init.clamp_min(1e-8)))

    @staticmethod
    def _make_init(n: int, mode: str) -> torch.Tensor:
        t = torch.linspace(0, 1, max(n, 1))
        if mode == "cosine":
            # 1.0 → 0.1 along the loop axis with a smooth half-cosine.
            return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.pi * t))
        if mode == "linear":
            return 1.0 - 0.9 * t
        return torch.ones(n)

    def forward(self, loop_t: int) -> torch.Tensor:
        """Return the (positive) injection magnitude scalar for `loop_t`."""
        t_idx = min(loop_t, self.max_loops - 1)
        return torch.exp(self.log_scale[t_idx])


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent update rule (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t  +  B(t) · e  +  Transformer(h_t, e)

    where e is the encoded input injected at every loop step to prevent drift.
    Without constraints, A can develop spectral radius ≥ 1, causing the hidden
    state to explode across loop iterations and destabilise training.

    This class guarantees ρ(A) < 1 by construction via a ZOH discretisation:
        A_continuous = Diag(-exp(log_A))       always negative diagonal
        A_discrete   = exp(Δt · A_continuous)  element-wise, values in (0, 1)

    Part 2 update: the injection coefficient B is now loop-dependent. A
    learned `B_dir` vector sets the per-channel direction, and an
    `InjectionScheduler` provides a learned scalar magnitude per loop
    iteration. This lets early loops anchor strongly to the input while
    later loops reason more freely, without sacrificing the LTI
    stability guarantee (which only constrains A).
    """

    def __init__(
        self,
        dim: int,
        max_loops: int = 16,
        injection_decay: str = "cosine",
    ):
        """
        Args:
            dim              -- hidden state dimension; one scalar per channel for A
            max_loops        -- maximum loop iterations; determines scheduler table size
            injection_decay  -- "cosine" | "linear" | "constant" initialisation for
                                the per-loop magnitude schedule
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))   # log of A_continuous magnitude
        self.log_dt = nn.Parameter(torch.zeros(1))    # log of discretisation step Δt
        # Per-channel direction (signed); per-loop scalar magnitude lives in scheduler.
        self.B_dir = nn.Parameter(torch.ones(dim) * 0.1)
        self.scheduler = InjectionScheduler(max_loops, init_decay=injection_decay)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discretised diagonal state matrix A_discrete.

        Returns:
            1-D tensor of shape (dim,) with all values strictly in (0, 1),
            guaranteeing ρ(A) < 1 regardless of learned parameter values.
        """
        # Compute in log space to avoid 0 * inf = NaN when log_dt → -∞, log_A → +∞.
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # Lower clamp at -15 (not -20) so that exp(-exp(-15)) ≈ 1 - 3e-7 stays
        # strictly representable below 1.0 in float32 — at -20 the result
        # saturates to exactly 1.0 (1-2e-9 < float32 epsilon) and the stability
        # guarantee silently breaks after large gradient steps.
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-15, 20)))

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
        loop_t: int = 0,
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B(t)·e + transformer_out.

        Args:
            h               -- current hidden state (B, T, dim)
            e               -- encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out -- output of the recurrent TransformerBlock at this step (B, T, dim)
            loop_t          -- current loop index; feeds the scheduler for B's magnitude.

        Returns:
            Updated hidden state of shape (B, T, dim)
        """
        A = self.get_A()
        B_mag = self.scheduler(loop_t)                  # scalar > 0
        B = self.B_dir * B_mag                          # (dim,)
        return A * h + B * e + transformer_out


# ---------------------------------------------------------------------------
# Multi-scale injection (hierarchical compression of e at later loops)
# ---------------------------------------------------------------------------


class MultiScaleInjection(nn.Module):
    """
    Hierarchical input injection — three views of e blended per loop iteration.

    Injecting full-resolution Prelude encoding `e` at every loop forces all
    iterations to reason at the same granularity. Early loops should match
    token-level patterns; late loops should reason over compressed, abstract
    representations. This module supplies three views of `e`:

        e_fine   : original Prelude output                — token-level
        e_coarse : local window mean pool                 — phrase-level
        e_global : global mean pool (broadcast)           — document-level

    A learned per-loop blend (3 weights, softmax-normalised) decides the
    mix at each iteration. Init: early loops favour fine, late loops shift
    toward coarse + global. Training is free to learn a different schedule.
    """

    def __init__(self, dim: int, max_loops: int, window_size: int = 8):
        super().__init__()
        self.max_loops = max_loops
        self.window_size = window_size

        # Separate projection heads per scale — the model can learn to use
        # the same e through three different transformations.
        self.proj_fine = nn.Linear(dim, dim, bias=False)
        self.proj_coarse = nn.Linear(dim, dim, bias=False)
        self.proj_global = nn.Linear(dim, dim, bias=False)
        for proj in (self.proj_fine, self.proj_coarse, self.proj_global):
            nn.init.normal_(proj.weight, std=0.02)

        # Per-loop blend logits (softmax-normalised in forward). Init makes
        # early loops fine-heavy and late loops coarse+global-heavy.
        self.blend = nn.Parameter(torch.zeros(max_loops, 3))
        with torch.no_grad():
            for t in range(max_loops):
                frac = t / max(max_loops - 1, 1)
                self.blend[t] = torch.tensor([
                    1.0 - frac,            # fine    ↘
                    frac * 0.7,            # coarse  ↗
                    frac * 0.3,            # global  ↗ (slower)
                ])

    def _coarse(self, e: torch.Tensor) -> torch.Tensor:
        """Local-window mean pool, broadcast back to T positions."""
        B, T, D = e.shape
        w = self.window_size
        pad = (w - T % w) % w
        if pad:
            e = F.pad(e, (0, 0, 0, pad))
        pooled = e.reshape(B, -1, w, D).mean(dim=2)            # (B, T/w, D)
        # Repeat each window's mean across its w positions, then trim
        # padding so output length matches the input.
        out = pooled.repeat_interleave(w, dim=1)[:, :T]
        return out

    @staticmethod
    def _global(e: torch.Tensor) -> torch.Tensor:
        """Global mean pool broadcast to every position."""
        return e.mean(dim=1, keepdim=True).expand_as(e)

    def forward(self, e: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            e      : (B, T, dim) — Prelude output
            loop_t : current loop index

        Returns:
            (B, T, dim) blended injection signal for this loop.
        """
        t_idx = min(loop_t, self.max_loops - 1)
        w = F.softmax(self.blend[t_idx], dim=0)                # (3,)
        e_fine = self.proj_fine(e)
        e_coarse = self.proj_coarse(self._coarse(e))
        e_global = self.proj_global(self._global(e))
        return w[0] * e_fine + w[1] * e_coarse + w[2] * e_global


# ---------------------------------------------------------------------------
# Cross-loop attention (attend across past loop iteration states)
# ---------------------------------------------------------------------------


class CrossLoopAttention(nn.Module):
    """
    Lightweight attention over stored hidden states from earlier loop iterations.

    The recurrent block updates `h` each loop but otherwise never looks back at
    what earlier iterations produced. Cross-loop attention gives the model
    explicit access to its own reasoning history within a single forward pass:
    it can compare current state to earlier states (backtracking signal),
    detect oscillation, or aggregate information from multiple reasoning
    stages.

    Implementation: keep a small buffer of past hidden states (`store_every`
    sets the sampling rate), then run a single multi-head cross-attention
    over the buffer at each loop. Cost is O(L · T²) for L buffer entries,
    cheap compared to a full attention layer.

    The output projection is zero-initialised so the module starts as an
    identity residual and learns to contribute gradually.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        store_every: int = 4,
        buffer_cap: int = 4,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.store_every = store_every
        self.buffer_cap = buffer_cap

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        for p in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(p.weight, std=0.02)
        nn.init.zeros_(self.o_proj.weight)     # identity residual at init
        # Protect this zero-init from MythOuro._init_weights' blanket N(0,0.02)
        # re-init (else the identity-residual property is silently destroyed).
        self.o_proj._skip_global_init = True

        self.norm = RMSNorm(dim)

    def _maybe_snapshot(
        self,
        h: torch.Tensor,
        loop_t: int,
        loop_state_buffer: list,
    ) -> None:
        """
        Append `h` (detached) to the buffer on every `store_every`-th loop and
        trim to `buffer_cap`. Split out from `forward` so callers that manage
        row alignment themselves (ContinuousDepthwiseBatcher, which must store
        full-batch snapshots while attending on an active subset — P0.4) can
        snapshot and attend independently.

        NOTE: stores `h.detach()`, which shares storage with `h`. Callers that
        mutate their hidden state in place afterwards (the batcher's row splice)
        must pass a clone.
        """
        if loop_t % self.store_every == 0:
            loop_state_buffer.append(h.detach())
            while len(loop_state_buffer) > self.buffer_cap:
                loop_state_buffer.pop(0)

    def forward(
        self,
        h: torch.Tensor,                  # (B, T, dim)
        loop_t: int,
        loop_state_buffer: list,          # mutable, modified in-place
    ) -> torch.Tensor:
        """
        Args:
            h                 -- current hidden state (B, T, dim)
            loop_t            -- current loop index
            loop_state_buffer -- list of past (B, T, dim) hidden states; this
                                 method appends to it on every `store_every`-th
                                 loop and trims to `buffer_cap`.

        Returns:
            h + cross-loop attention residual, same shape as h.
        """
        self._maybe_snapshot(h, loop_t, loop_state_buffer)
        return self._attend(h, loop_state_buffer)

    def _attend(self, h: torch.Tensor, loop_state_buffer: list) -> torch.Tensor:
        """
        Attend over the buffered history. Every buffer entry must have the SAME
        batch dim as `h` and row i of every entry must be the same sequence as
        row i of `h` — attention is per-batch-row, so misaligned rows silently
        attend to another sequence's history.
        """
        B, T, D = h.shape

        # Nothing to attend to yet → identity.
        if not loop_state_buffer:
            return h

        h_norm = self.norm(h)
        q = (
            self.q_proj(h_norm)
            .view(B, T, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )                                                # (B, H, T, head_dim)

        past = torch.cat(loop_state_buffer, dim=1)       # (B, L*T, dim)
        L_total = past.shape[1]
        k = (
            self.k_proj(past)
            .view(B, L_total, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )                                                # (B, H, L*T, head_dim)
        v = (
            self.v_proj(past)
            .view(B, L_total, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                      # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return h + self.o_proj(out)


# ---------------------------------------------------------------------------
# ACT halting (Adaptive Computation Time)
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration. Positions
    where the hidden state has converged (high cumulative halting probability)
    stop accumulating updates, while positions still being refined continue.
    This lets easy tokens halt early and hard tokens receive more computation,
    all within the same batch. Also makes the model Turing-complete under
    certain assumptions about the expressiveness of the transformer block.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; input to the halting scalar predictor
        """
        super().__init__()
        self.halt = nn.Linear(dim, 1)
        # Bias initialised so sigmoid(bias) ≈ 0.1 — at init the model
        # "wants to keep looping" rather than halt immediately. Without
        # this, default bias=0 gives λ≈0.5 per step, which collapses to
        # K=1 within a few hundred steps as the task gradient drives λ→1.
        # Starting low gives the depth regulariser room to defend.
        nn.init.constant_(self.halt.bias, -2.2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability from the current hidden state.

        Args:
            h -- hidden state of shape (B, T, dim)

        Returns:
            Halting probability tensor of shape (B, T), values in (0, 1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block (one set of weights, looped T times)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    The core recurrent block of MythOuro — a single TransformerBlock looped T times.

    At each loop iteration t, the hidden state h is updated via:
        1. loop_index_embedding: inject sinusoidal loop-index signal into h
        2. TransformerBlock:     compute attention + MoE FFN on normalized (h + e)
        3. LoRAAdapter:          apply depth-wise LoRA delta to transformer output
        4. LTIInjection:         stable update h = A·h + B·e + transformer_out
        5. ACTHalting:           accumulate per-position halting probabilities;
                                  positions that have converged stop contributing

    The encoded input e (output of the Prelude) is injected at every step to keep
    the original input signal alive across arbitrary loop depth, preventing drift.
    The ACT mechanism produces a weighted sum of hidden states across iterations,
    where the weights reflect when each position converged.

    More loop iterations at inference = deeper reasoning chains, following the
    depth-extrapolation property of looped transformers (Saunshi et al., 2025).
    """

    def __init__(self, cfg: MythOuroConfig):
        """
        Args:
            cfg -- MythOuroConfig; uses dim, lora_rank, max_loop_iters,
                   act_threshold, gradient_checkpointing, convergence_eps
        """
        super().__init__()
        self.cfg = cfg
        if getattr(cfg, "recurrent_dense", False):
            # Ablation arm: dense SwiGLU FFN at the matched-active width so the
            # recurrent block does the same FLOPs/token as the MoE arm. See
            # docs/roadmap.md "Gating experiment: MoE-vs-dense ablation".
            d_ff = cfg.recurrent_dense_ffn_dim or (
                cfg.expert_dim * cfg.n_experts_per_tok * (1 + cfg.n_shared_experts)
            )
            self.block = TransformerBlock(cfg, use_moe=False, dense_ffn_dim=d_ff)
        else:
            self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(
            cfg.dim,
            max_loops=cfg.max_loop_iters,
            injection_decay=getattr(cfg, "injection_decay", "cosine"),
        )
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = (
            cfg.dim // 8
        )  # fraction of channels receiving loop-index embedding
        self.use_ckpt = getattr(cfg, "gradient_checkpointing", False)
        self.conv_eps = getattr(cfg, "convergence_eps", 1e-4)

        # Best-of-trajectory support (inference experiment, off by default).
        # When a generator sets `collect_trajectory`, each loop's committed
        # hidden state is stashed in `last_trajectory` so the caller can score
        # every depth with the uncertainty head and emit the most-confident one
        # — see MythOuro.forward_trajectory and
        # inference.BestOfTrajectoryGenerator. When False the normal path is
        # byte-for-byte unchanged and pays nothing.
        self.collect_trajectory = False
        self.last_trajectory: Optional[torch.Tensor] = None

        # Measurement override (off by default). When set alongside
        # `collect_trajectory`, the two inference early-exit breaks (convergence
        # and ACT halt-all) are suppressed so the loop runs the full n_loops.
        # This is what lets `forward_trajectory` observe the *counterfactual*
        # loops ACT would otherwise skip — answering "would deeper loops have
        # lowered uncertainty?" rather than just "where did ACT choose to stop?"
        # Pure measurement: changes no weights and never affects the normal
        # forward/generate path (which leaves this False).
        self.force_full_depth = False

        # Part 2: multi-scale injection (hierarchical fine/coarse/global e)
        self.use_ms = getattr(cfg, "use_multiscale_injection", False)
        if self.use_ms:
            self.ms_inject = MultiScaleInjection(
                cfg.dim,
                cfg.max_loop_iters,
                window_size=getattr(cfg, "ms_window_size", 8),
            )

        # Part 2: cross-loop attention (queryable history of prior loop states)
        self.use_cross = getattr(cfg, "use_cross_loop_attention", False)
        if self.use_cross:
            # Cross-loop heads default to a quarter of the main attention head
            # count — this attention is cheap (small buffer) and doesn't need
            # the full head expressiveness.
            self.cross_loop_attn = CrossLoopAttention(
                cfg.dim,
                n_heads=max(1, cfg.n_heads // 4),
                store_every=getattr(cfg, "cross_loop_store_every", 4),
            )

    def _loop_body(
        self,
        h: torch.Tensor,
        e_inject: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_cache: Optional[dict],
        t: int,
    ) -> "tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]":
        """
        One loop iteration. Factored out so it can be wrapped by
        torch.utils.checkpoint to recompute activations on the backward pass.

        `e_inject` is the per-loop injection signal — either the original
        Prelude output (single-scale) or the multi-scale blend produced by
        MultiScaleInjection for this loop index.

        Returns `(h_new, router_logits, expert_counts)`. The MoE telemetry is
        *returned* (not just stashed on the FFN) so it flows through the
        checkpoint boundary and is grad-tracked / recompute-safe — this is what
        lets `RecurrentBlock.forward` accumulate per-loop routing across ALL
        loops (P0.2) instead of seeing only the last one. `None` for the dense
        recurrent FFN.
        """
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        combined = self.norm(h_loop + e_inject)
        cache_key = f"recurrent_loop_{t}"
        trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
        trans_out = trans_out + self.lora(trans_out, t)
        h_new = self.injection(h, e_inject, trans_out, loop_t=t)
        ffn = self.block.ffn
        rlogits = getattr(ffn, "_last_router_logits", None)   # grad-tracked
        counts = getattr(ffn, "_last_expert_counts", None)    # detached
        return h_new, rlogits, counts

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h        -- initial hidden state from the Prelude, shape (B, T, dim)
            e        -- encoded input frozen for injection each step, shape (B, T, dim)
            freqs_cis-- precomputed RoPE frequencies
            mask     -- additive causal mask or None
            n_loops  -- number of loop iterations; defaults to cfg.max_loop_iters.
                        Can be increased at inference for deeper reasoning (depth extrapolation).
            kv_cache -- cache dict passed through to the inner TransformerBlock;
                        each loop iteration uses a separate cache key

        Returns:
            Hidden state of shape (B, T, dim). In training mode (no
            kv_cache), the final loop's hidden state `h_K`. In eval mode
            or with kv_cache, the ACT-weighted sum Σ_t w_t · h_t.

        Memory / compute notes:
            - When cfg.gradient_checkpointing is True and the model is in
              training mode AND no kv_cache is provided, the loop body is
              wrapped by torch.utils.checkpoint so activations from every
              iteration except the current one are released. This makes
              activation memory O(1) in n_loops at the cost of one extra
              forward per backward.
            - When kv_cache is provided, checkpointing is disabled because
              the cache mutates non-functionally and breaks the recompute
              contract.
            - At inference, ‖h_{t+1} - h_t‖ is monitored; once it drops
              below cfg.convergence_eps everywhere the remaining ACT mass
              is committed to the final state and the loop short-circuits.
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)
        h_prev: Optional[torch.Tensor] = None
        loop_state_buf: list[torch.Tensor] = []          # for cross-loop attention

        # Per-position halt-depth telemetry. Sentinel = n_loops means
        # "never halted before max depth". Eval harness reads this via
        # `recurrent.last_halt_step` after a forward pass.
        halt_step = torch.full(
            (B, T), n_loops, device=h.device, dtype=torch.long,
        )

        # Per-loop halt probabilities, captured as the loop runs. After the
        # loop completes, these are post-processed into a proper PonderNet-
        # style probability distribution P(halt at step n) and exposed via
        # `self.last_halt_distribution` for the depth regulariser to read.
        # Lambdas stay in the autograd graph so the depth-reg loss can
        # flow gradient back into ACTHalting → LoRA → recurrent block.
        loop_halt_probs: list[torch.Tensor] = []

        # Best-of-trajectory capture: committed per-loop hidden states, used by
        # MythOuro.forward_trajectory. Only active when a generator opted in via
        # `collect_trajectory`, and never during training (the training return
        # is h_K and must stay on its own gradient path — see the comment at the
        # end of this method).
        collect = self.collect_trajectory and kv_cache is None and not self.training
        traj_states: list[torch.Tensor] = []

        # P0.2: accumulate MoE routing telemetry across ALL loops (not just the
        # last). Reset the FFN's per-forward buffers here; append the returned
        # (grad-tracked) logits and (detached) counts each loop below. The
        # appends run in this forward loop only — torch.utils.checkpoint's
        # backward recompute re-runs `_loop_body` internally but NOT this loop,
        # so there's no double-counting.
        ffn = self.block.ffn
        ffn_is_moe = isinstance(ffn, MoEFFN)
        if ffn_is_moe:
            ffn._router_logits_buf = []
            ffn._expert_counts_sum = None

        use_ckpt = self.use_ckpt and self.training and kv_cache is None

        for t in range(n_loops):
            # Multi-scale injection: blend fine / coarse / global views of e
            # with learned per-loop weights. Falls back to plain e if disabled.
            e_inject = self.ms_inject(e, t) if self.use_ms else e

            if use_ckpt:
                h_new, rlogits, counts = ckpt_util.checkpoint(
                    self._loop_body,
                    h, e_inject, freqs_cis, mask, None, t,
                    use_reentrant=False,
                )
            else:
                h_new, rlogits, counts = self._loop_body(
                    h, e_inject, freqs_cis, mask, kv_cache, t
                )

            if ffn_is_moe and rlogits is not None:
                ffn._router_logits_buf.append(rlogits)
                ffn._expert_counts_sum = (
                    counts if ffn._expert_counts_sum is None
                    else ffn._expert_counts_sum + counts
                )

            # Cross-loop attention residual. Kept outside the checkpoint
            # because it mutates `loop_state_buf` in-place — checkpointed
            # functions must be free of side effects.
            if self.use_cross:
                h_new = self.cross_loop_attn(h_new, t, loop_state_buf)

            # Convergence-detection early exit (inference only; training needs
            # deterministic depth for checkpoint replay, and kv_cache decode
            # must populate every cache_key).
            if (
                h_prev is not None
                and not self.training
                and kv_cache is None
                and not self.force_full_depth
            ):
                delta = (h_new - h_prev).norm(dim=-1)              # (B, T)
                if (delta < self.conv_eps).all():
                    remainder = (1.0 - cumulative_p).clamp(min=0)
                    h_out = h_out + remainder.unsqueeze(-1) * h_new
                    # Any positions that hadn't already crossed the ACT
                    # threshold get assigned this loop as their halt step
                    # — they "converged" rather than "halted", but for
                    # the loop-efficiency metric the depth used is the
                    # same value.
                    still_unhalted = halt_step == n_loops
                    halt_step = torch.where(
                        still_unhalted,
                        torch.tensor(t, device=h.device, dtype=halt_step.dtype),
                        halt_step,
                    )
                    # Commit the converged state so the emitted h_K is the
                    # deepest-run hidden state (P0.3), not the prior loop's.
                    h = h_new
                    if collect:
                        traj_states.append(h_new)
                    break

            h_prev = h_new.detach()
            h = h_new
            if collect:
                traj_states.append(h)

            p = self.act(h)  # (B, T) — per-step halt prob (PonderNet's λ_t)
            # Track the lambda for the depth regulariser. We keep gradient
            # here intentionally — the regulariser drives ACTHalting.
            loop_halt_probs.append(p)
            still_running = ~halted

            # ACT remainder trick: once cumulative_p + p crosses threshold,
            # assign the remaining probability mass as the final weight.
            # Gate by still_running so halted positions contribute exactly
            # once (on the halting step) and zero thereafter — otherwise
            # threshold<1 leaves a non-zero remainder that leaks every step.
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            newly_halted = (cumulative_p >= self.cfg.act_threshold) & ~halted
            halt_step = torch.where(
                newly_halted,
                torch.tensor(t, device=h.device, dtype=halt_step.dtype),
                halt_step,
            )
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # Only short-circuit at inference (or KV-decoding requires depth
            # equality across calls). During training we keep every loop
            # running even when ACT wants to halt, so:
            #   1. the PonderNet halt distribution always has K = n_loops
            #      buckets, giving `depth_regularization_loss` a meaningful
            #      KL-to-uniform signal. A K=1 distribution is trivially δ
            #      and has KL=0, so an early break here would silence the
            #      depth regulariser and let ACT collapse onto K=1 — exactly
            #      the failure mode we observed empirically;
            #   2. gradients flow through every λ_t, not just the ones that
            #      survived to the break point.
            # Inference still short-circuits for latency — unless force_full_depth
            # is set for a counterfactual depth measurement.
            if (
                halted.all()
                and kv_cache is None
                and not self.training
                and not self.force_full_depth
            ):
                break

        # Detach so downstream diagnostics never accidentally extend the
        # autograd graph through this telemetry tensor.
        self.last_halt_step = halt_step.detach()

        # ── PonderNet-style halt distribution over actually-run depths ──
        # For each (batch, token) position, compute
        #     P(halt at step n) = λ_n · ∏_{i<n}(1 − λ_i)
        # with the last step taking the residual mass so the row sums to 1.
        # This gives `depth_regularization_loss` a clean probability
        # distribution to pull toward the uniform prior, without changing
        # the actual halt CRITERION (Graves cumulative-threshold continues
        # to drive when the loop short-circuits).
        if loop_halt_probs:
            # (B, T, K) — K is the number of loops we actually ran
            lambdas = torch.stack(loop_halt_probs, dim=-1)
            one_minus = (1.0 - lambdas).clamp(min=0.0, max=1.0)
            ones = torch.ones_like(lambdas[..., :1])
            # survival_before[n] = ∏_{i<n}(1 − λ_i); survival_before[0] = 1
            survival_before = torch.cat(
                [ones, one_minus[..., :-1]], dim=-1
            ).cumprod(dim=-1)
            halt_dist = lambdas * survival_before
            # Force normalisation: the final step absorbs any residual mass
            # left over from λ_K < 1. Without this the distribution may
            # under-sum (model hadn't "decided" to halt at the cap).
            residual = (
                1.0 - halt_dist[..., :-1].sum(dim=-1, keepdim=True)
            ).clamp(min=0.0)
            halt_dist = torch.cat(
                [halt_dist[..., :-1], residual], dim=-1
            )
            self.last_halt_distribution = halt_dist
        else:
            self.last_halt_distribution = None

        # Stash the per-loop trajectory for best-of-trajectory emission.
        # (B, T, K, D) where K is the number of loops actually run; None when
        # capture wasn't requested.
        if collect:
            self.last_trajectory = (
                torch.stack(traj_states, dim=2).detach() if traj_states else None
            )

        # Always return the FINAL loop's hidden state h_K — at training AND
        # inference (P0.3). Rationale:
        #   - Training returns h_K because the ACT-weighted sum `Σ w_t·h_t` gave
        #     the optimizer a knob to pin λ₀≈1.0 (output=h_0), collapsing depth
        #     regardless of the depth regulariser. Returning h_K severs that path
        #     so the task loss flows through every loop.
        #   - Inference previously returned `h_out` (the weighted sum), but (a)
        #     it under-summed for positions that never crossed act_threshold (no
        #     final remainder commit) and (b) the coda/head were ONLY ever trained
        #     on h_K, so h_out was a never-trained emission path. Returning h_K
        #     here makes inference consistent with training and removes both bugs.
        # ACT is now purely an early-exit CRITERION (drives the halt-all break and
        # `last_halt_step` telemetry), not an output blender. `h_out` is retained
        # only as vestigial accounting and is slated for removal (P1.8).
        return h


# ---------------------------------------------------------------------------
# Attention Sink (learnable register tokens prepended to every forward)
# ---------------------------------------------------------------------------


class AttentionSink(nn.Module):
    """
    Learnable "register" tokens prepended to the input sequence (Xiao et al., 2024).

    Why: deep recurrent stacks tend to concentrate disproportionate attention
    mass on the first few token positions, regardless of their content — a
    failure mode that destabilises long-context generation and limits the
    effective depth at which the loop can reason. Reserving a handful of
    dedicated sink positions gives the attention distribution a designated
    "trash bin", freeing real content positions from absorbing that load.

    Mechanics:
        prepend(x) → concatenates n_sink_tokens learnable embeddings to the
                     front of x along the sequence dimension.
        strip(h)   → removes the first n_sink_tokens positions from h before
                     the LM head sees it.

    The sink tokens occupy real RoPE positions 0..n_sink-1, so the rest of
    the sequence is shifted forward by n_sink_tokens. The caller is
    responsible for offsetting `start_pos` accordingly during incremental
    decoding (the sink is already in the KV cache after the first call).
    """

    def __init__(self, n_tokens: int, dim: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.tokens = nn.Parameter(torch.zeros(n_tokens, dim))
        nn.init.normal_(self.tokens, std=0.02)

    def prepend(self, x: torch.Tensor) -> "tuple[torch.Tensor, int]":
        """
        Args:
            x -- (B, T, dim)
        Returns:
            (B, n_sink + T, dim), n_sink
        """
        if self.n_tokens == 0:
            return x, 0
        B = x.shape[0]
        sink = self.tokens.unsqueeze(0).expand(B, -1, -1).to(dtype=x.dtype)
        return torch.cat([sink, x], dim=1), self.n_tokens

    def strip(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_sink + T, dim) → (B, T, dim)"""
        if self.n_tokens == 0:
            return x
        return x[:, self.n_tokens :]


# ---------------------------------------------------------------------------
# Uncertainty head (per-token confidence)
# ---------------------------------------------------------------------------


class UncertaintyHead(nn.Module):
    """
    Per-token uncertainty estimator over the final hidden state.

    Outputs a scalar in (0, 1) per position interpretable as predicted
    probability that the model's argmax token is wrong. Trained via the
    `uncertainty_calibration_loss` against the realised error signal
    (target = 1 - is_correct), making the score a learned, calibrated
    confidence reading rather than a raw entropy proxy.

    Used by:
        - UncertaintyGatedGenerator     (allocates more loop budget to
                                          high-uncertainty tokens at inference)
        - RetrievalAugmentedInjector    (adaptive retrieval trigger)
        - SpeculativeDecoder            (verification gate)

    Architecture: two-layer MLP with SiLU; zero-initialised output so the
    initial score is 0.5 (logit 0) and gradients are well-behaved at the
    start of training.
    """

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or max(dim // 4, 32)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Protect this zero-init from MythOuro._init_weights' blanket re-init,
        # so the head starts at sigmoid(0)=0.5 (calibrated neutral) as intended.
        self.net[-1]._skip_global_init = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h -- normalised hidden state, (B, T, dim)
        Returns:
            Uncertainty in (0, 1), shape (B, T)
        """
        return torch.sigmoid(self.net(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class MythOuro(nn.Module):
    """
    MythOuro — Recurrent-Depth Transformer language model.

    Implements the hypothesized Claude Mythos architecture as a Recurrent-Depth
    Transformer (RDT). The model divides computation into three functional blocks:

        Input tokens
             ↓
        [Prelude]          — prelude_layers standard transformer blocks, run once
             ↓
        [Recurrent Block]  — one transformer block looped T times with input injection
             ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
             ↓
        [Coda]             — coda_layers standard transformer blocks, run once
             ↓
        Output logits

    Key properties:
    - Same weights, more loops → deeper reasoning, no parameter growth
    - Depth extrapolation: train on N loops, test on N+k loops (emergent)
    - ACT halting: variable compute per position within a batch
    - MoE FFN in the recurrent block: breadth across domains
    - LTI-stable injection: spectral radius < 1 guaranteed by construction
    - Supports both GQA and MLA attention (set via cfg.attn_type)
    """

    def __init__(self, cfg: MythOuroConfig):
        """
        Args:
            cfg -- MythOuroConfig specifying all architecture hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # GQA uses full head_dim for RoPE; MLA uses only qk_rope_head_dim (decoupled).
        # RoPE table is sized to max_seq_len + n_sink_tokens so sink positions
        # plus the longest real sequence stay within precomputed frequencies.
        rope_len = cfg.max_seq_len + cfg.n_sink_tokens
        rope_real = getattr(cfg, "rope_real", False)
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, rope_len, cfg.rope_theta, real=rope_real
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, rope_len, cfg.rope_theta, real=rope_real
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.sink = AttentionSink(cfg.n_sink_tokens, cfg.dim)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self.uncertainty = UncertaintyHead(cfg.dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize linear and embedding weights with N(0, 0.02).

        Skips modules that self-initialize and mark themselves with
        `_skip_global_init = True` (e.g. `CrossLoopAttention.o_proj` and
        `UncertaintyHead.net[-1]`, which are deliberately zero-init). Without
        this guard the blanket re-init silently overwrites those zero-inits —
        the bug that made cross-loop attention inject noise from step 0 and the
        uncertainty head start off-neutral in all v1–v5 checkpoints (P0.1).
        """
        for m in self.modules():
            if getattr(m, "_skip_global_init", False):
                continue
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build an additive causal mask: 0 on and below the diagonal, -inf above.

        Args:
            seq_len -- sequence length
            device  -- target device
            dtype   -- tensor dtype (must match activation dtype so the additive
                       mask doesn't upcast the attention logits in the fallback
                       attention path — e.g. bf16 weights with an fp32 mask
                       promotes attn to fp32 and then breaks the fp32-vs-bf16
                       matmul against V)

        Returns:
            Tensor of shape (1, 1, seq_len, seq_len) broadcastable over (B, H, T, S)
        """
        mask = torch.full(
            (1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype
        )
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Forward pass through Sink → Prelude → Recurrent Block → Coda → Sink-strip.

        Args:
            input_ids -- token indices of shape (B, T)
            n_loops   -- recurrent loop depth; defaults to cfg.max_loop_iters.
                         Increase at inference to extrapolate to harder problems.
            kv_cache  -- dict mutated in-place for autoregressive KV caching;
                         pass an empty dict {} and reuse across decode steps
            start_pos -- index of the first token in input_ids within the full
                         sequence (caller-visible position; sink offset is
                         handled internally). 0 for prefill, prompt_len for
                         the second decode step, etc.

        Returns:
            (logits, uncertainty) tuple:
              logits      -- (B, T, vocab_size)
              uncertainty -- (B, T) per-token confidence-of-error in (0, 1)
        """
        device = input_ids.device

        x = self.embed(input_ids)

        # Sink: prepend learnable register tokens once on the prefill call.
        # On subsequent decode steps the sink is already in the KV cache, so
        # we only shift the RoPE offset by sink_len rather than re-prepending.
        if start_pos == 0:
            x, sink_len = self.sink.prepend(x)
            rope_start = 0
        else:
            sink_len = 0
            rope_start = self.sink.n_tokens + start_pos

        T_ext = x.shape[1]
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[rope_start : rope_start + T_ext]
        mask = self._causal_mask(T_ext, device, x.dtype) if T_ext > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x  # encoded input frozen for injection every loop
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        # Strip sink positions before LM head (sink positions exist only in
        # the activation tensor on the prefill call — decode steps pass T=1
        # and never carry sink positions in `x`).
        if sink_len:
            x = self.sink.strip(x)

        normed = self.norm(x)
        logits = self.head(normed)
        unc = self.uncertainty(normed)
        return logits, unc

    @torch.no_grad()
    def forward_trajectory(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        force_full_depth: bool = False,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Inference-only forward returning a per-recurrent-loop output trajectory.

        For each loop the recurrent block runs, this pushes that loop's hidden
        state through the Coda + LM head + UncertaintyHead, so every depth is
        scored as a *full model output* (not a raw recurrent state). Used by
        `inference.BestOfTrajectoryGenerator` to emit the lowest-uncertainty
        depth across the trajectory rather than the ACT-weighted blend.

        Runs without a KV cache (full recompute); the Coda is re-run once per
        captured loop, so cost is O(K) Codas. Fine at inspector / experiment
        scale — not a fast-decode path. The normal `forward` is untouched.

        Args:
            input_ids        -- token indices of shape (B, T)
            n_loops          -- recurrent depth; defaults to cfg.max_loop_iters.
                                May be raised above the trained value
                                (depth extrapolation).
            force_full_depth -- when True, suppress ACT's convergence/halt-all
                                early-exit so the loop runs the *full* n_loops.
                                Lets the trajectory observe the loops ACT would
                                otherwise skip — the counterfactual needed to
                                tell "loop k genuinely hurts" from "loop k never
                                ran". Pure measurement; weights untouched.

        Returns:
            (logits_traj, unc_traj):
              logits_traj -- (B, T, K, vocab_size)
              unc_traj    -- (B, T, K) per-loop confidence-of-error in (0, 1)
            where K is the number of loops actually run. With
            force_full_depth=True, K == n_loops; otherwise K ≤ n_loops (fewer
            when ACT halts all positions or the state converges early).
        """
        device = input_ids.device
        x = self.embed(input_ids)
        x, sink_len = self.sink.prepend(x)

        T_ext = x.shape[1]
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[:T_ext]
        mask = self._causal_mask(T_ext, device, x.dtype) if T_ext > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, None, cache_key=f"prelude_{i}")

        e = x  # injected each loop
        self.recurrent.collect_trajectory = True
        self.recurrent.force_full_depth = force_full_depth
        try:
            self.recurrent(x, e, freqs_cis, mask, n_loops, None)
            traj = self.recurrent.last_trajectory  # (B, T_ext, K, D) or None
        finally:
            self.recurrent.collect_trajectory = False
            self.recurrent.force_full_depth = False
            self.recurrent.last_trajectory = None

        # No loop ran (n_loops == 0): fall back to a single Coda pass on the
        # Prelude output so the trajectory still has exactly one entry.
        if traj is None:
            traj = e.unsqueeze(2)

        K = traj.shape[2]
        logits_steps: list[torch.Tensor] = []
        unc_steps: list[torch.Tensor] = []
        for k in range(K):
            h = traj[..., k, :]
            for i, layer in enumerate(self.coda):
                h = layer(h, freqs_cis, mask, None, cache_key=f"coda_{i}")
            if sink_len:
                h = self.sink.strip(h)
            normed = self.norm(h)
            logits_steps.append(self.head(normed))
            unc_steps.append(self.uncertainty(normed))

        logits_traj = torch.stack(logits_steps, dim=2)  # (B, T, K, V)
        unc_traj = torch.stack(unc_steps, dim=2)          # (B, T, K)
        return logits_traj, unc_traj

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV caching.

        On step 0 the full prompt is processed. On subsequent steps only the
        last generated token is passed, with all previous keys and values
        retrieved from kv_cache. This keeps decode cost proportional to one
        token per step rather than the full growing sequence.

        n_loops can be set higher than the training value to extrapolate to
        harder problems at inference time (depth extrapolation property).

        Args:
            input_ids      -- prompt token indices of shape (B, T)
            max_new_tokens -- number of tokens to generate
            n_loops        -- recurrent loop depth for each decode step
            temperature    -- softmax temperature; lower = more greedy
            top_k          -- restrict sampling to top-K logits (0 = disabled)

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
            logits, _ = self.forward(
                cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos
            )
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids
