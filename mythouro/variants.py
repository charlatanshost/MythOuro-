from mythouro.main import MythOuroConfig

# Parameter budget breakdown per variant:
#   total ≈ embed + prelude/coda dense blocks + recurrent MLA + MoE
#   MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
# expert_dim is solved from the residual budget after all other terms.
#
# `max_loop_iters` choice:
#   Ouro (Zhu et al. 2025) trained looped LMs at 1.4B/2.6B on 7.7T tokens and
#   measured peak accuracy on hard reasoning benchmarks at 3–4 loops, with
#   measurable degradation past ~8 (overlooping hurts). The 1B/3B configs
#   here track Ouro's tested range. Larger variants extrapolate cautiously
#   (8–12); no empirical data exists above 2.6B for this question, so don't
#   read those numbers as validated — they're informed guesses.


def mythouro_distill_tiny() -> MythOuroConfig:
    """
    ~240M parameter student tuned to fit alongside the bf16 Ouro-2.6B
    teacher on a single 12 GB GPU.

    Why this exists
    ---------------
    Logit distillation needs both teacher and student forward-active in
    the same memory budget. ByteDance/Ouro-2.6B-Thinking in bf16 takes
    ~5.2 GB. AdamW for ~240M params is ~2 GB of optimiser state, plus
    ~480 MB grads + ~480 MB weights + ~200 MB activations at
    `micro_batch=1, seq_len=2048`. Total fits comfortably under 12 GB
    with headroom for routing spikes.

    Vocab is Ouro's 49152 (see `MythOuroTokenizer` default) — this is the
    non-negotiable alignment requirement for logit distillation to be
    mathematically meaningful.

    Loop depth is 4 to match Ouro's `total_ut_steps=4`. Going beyond
    that on the student side has no clear benefit during distillation
    (the teacher's targets are already shaped by 4-loop computation)
    and increases per-step compute proportionally.

    For 8 GB cards (5060 / 4060): the student fits, but you must keep
    the teacher on CPU — see the `--teacher-device cpu` flag on
    `mythouro-distill`. Distillation throughput drops noticeably
    because the teacher forward has to round-trip through PCIe.
    """
    return MythOuroConfig(
        vocab_size=49152,           # MUST match Ouro tokenizer
        dim=1280,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=2048,
        max_loop_iters=4,           # match Ouro's total_ut_steps
        prelude_layers=2,
        coda_layers=2,
        attn_type="gqa",            # simpler than MLA; less risky for first run
        # GQA-relevant fields only; MLA fields kept at the dataclass defaults
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=48,
        v_head_dim=48,
        # MoE: moderate width, enough sparsity to specialise
        n_experts=24,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=1280,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythouro_distill_small() -> MythOuroConfig:
    """
    ~420M target for MoE expansion from `mythouro_distill_tiny` checkpoints.

    Identical to `mythouro_distill_tiny` in every dimension except `n_experts`,
    which is doubled (24 → 48). This is the function-near-preserving promotion
    target documented in `docs/growth_design.md`:

      * Active params/token unchanged (n_experts_per_tok = 4 unchanged), so
        inference compute and memory at decode time don't grow with the larger
        pool.
      * Routed expert storage doubles, lifting total params from ~278M to
        ~420M.
      * Router weight matrix gains 24 new rows; new experts are tied to
        their parent for first-pass selection, then bias-decayed in.

    Used by `tools/grow_checkpoint.py` and by `--student-variant
    mythouro_distill_small` when resuming a promoted checkpoint. NOT a
    from-scratch training target — train at this size only after promoting
    from a trained tiny checkpoint.
    """
    return MythOuroConfig(
        vocab_size=49152,
        dim=1280,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=2048,
        max_loop_iters=4,
        prelude_layers=2,
        coda_layers=2,
        attn_type="gqa",
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=48,
        v_head_dim=48,
        # Doubled routed expert pool; everything else identical to tiny.
        n_experts=48,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=1280,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythouro_distill_xl() -> MythOuroConfig:
    """
    ~630M target for the second MoE-expansion round (v5 Path A).

    Identical to `mythouro_distill_small` in every dimension except `n_experts`,
    which is doubled again (48 → 96). Promotion target of a trained
    `mythouro_distill_small` checkpoint via `tools/grow_checkpoint.py
    --expansion-factor 2`. See `docs/growth_design.md`.

      * Active params/token unchanged (n_experts_per_tok = 4), so decode-time
        compute/memory don't grow with the larger pool.
      * Each of the 96 experts is ~4.9M params (3 × dim × expert_dim);
        96 routed + 2 shared lifts the total to ~630M.

    NOT a from-scratch training target — train at this size only after
    promoting from a trained `distill_small` (48-expert) checkpoint.
    """
    return MythOuroConfig(
        vocab_size=49152,
        dim=1280,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=2048,
        max_loop_iters=4,
        prelude_layers=2,
        coda_layers=2,
        attn_type="gqa",
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=48,
        v_head_dim=48,
        # Doubled again: 48 → 96 routed experts. Everything else identical.
        n_experts=96,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=1280,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythouro_1b() -> MythOuroConfig:
    """1B parameter config. Small research/fine-tuning model. dim=2048, 64 experts, 6 loop iters, 4k context."""
    return MythOuroConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=6,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythouro_3b() -> MythOuroConfig:
    """3B parameter config. Compact inference model. dim=3072, 64 experts, 6 loop iters, 4k context."""
    return MythOuroConfig(
        vocab_size=32000,
        dim=3072,
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        max_loop_iters=6,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384,
        q_lora_rank=768,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        v_head_dim=96,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=4096,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythouro_10b() -> MythOuroConfig:
    """10B parameter config. Mid-scale general model. dim=4096, 128 experts, 8 loop iters, 8k context."""
    return MythOuroConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=8,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=128,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=5632,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=16,
    )


def mythouro_50b() -> MythOuroConfig:
    """50B parameter config. Large reasoning model. dim=6144, 256 experts, 8 loop iters, 8k context."""
    return MythOuroConfig(
        vocab_size=32000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=8,
        prelude_layers=3,
        coda_layers=3,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=9728,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=32,
    )


def mythouro_100b() -> MythOuroConfig:
    """100B parameter config. Frontier-class model. dim=8192, 256 experts, 12 loop iters, 1M context, 128k output."""
    return MythOuroConfig(
        vocab_size=32000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=1000000,
        max_loop_iters=12,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=8,
        expert_dim=13568,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=64,
        max_output_tokens=131072,
    )


def mythouro_500b() -> MythOuroConfig:
    """500B parameter config. Ultra-scale MoE model. dim=12288, 512 experts, 12 loop iters, 1M context, 128k output."""
    return MythOuroConfig(
        vocab_size=100000,
        dim=12288,
        n_heads=96,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=12,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=3072,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=23040,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=128,
        max_output_tokens=131072,
    )


def mythouro_1t() -> MythOuroConfig:
    """1T parameter config. Maximum scale. dim=16384, 512 experts, 12 loop iters, 1M context, 128k output."""
    return MythOuroConfig(
        vocab_size=100000,
        dim=16384,
        n_heads=128,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=12,
        prelude_layers=6,
        coda_layers=6,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=4096,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=34560,
        act_threshold=0.99,
        rope_theta=2000000.0,
        lora_rank=256,
        max_output_tokens=131072,
    )
