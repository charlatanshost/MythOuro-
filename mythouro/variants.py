from dataclasses import replace

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


def mythouro_distill_tiny_dense() -> MythOuroConfig:
    """
    Dense twin of `mythouro_distill_tiny` for the MoE-vs-dense ablation.

    Identical in every respect except the recurrent block's FFN: the MoE
    (24 routed + 2 shared experts, top-4) is replaced by a single dense SwiGLU
    FFN of width `expert_dim * n_experts_per_tok * (1 + n_shared_experts)`
    = 1280 * 4 * 3 = 15360. That width makes the dense FFN's parameters /
    FLOPs per token equal to the MoE arm's *activated* FFN per token — the
    matched-compute comparison. Same compute, ~98M fewer total params (the
    idle routed experts), and none of the routing machinery.

    Use as the dense arm of the ablation in docs/roadmap.md ("Gating
    experiment: MoE-vs-dense ablation"). The training run must drop the
    MoE-only aux losses (load-balance / sparse-activation / router-bias) for
    this variant — there are no MoE layers for them to read.
    """
    return replace(
        mythouro_distill_tiny(),
        recurrent_dense=True,
        recurrent_dense_ffn_dim=15360,
    )


def mythouro_distill_mid() -> MythOuroConfig:
    """
    ~633M total / ~460M ACTIVATED — 2.55x the activated params of
    `mythouro_distill_tiny`. The growth target chosen on evidence, 2026-09-15.

    Why this shape and not more experts
    -----------------------------------
    `mythouro_distill_small` / `_xl` grow by EXPERT COUNT at dim 1280, top-4.
    That axis was run (24 -> 48, 2026-08-27..09-01) and it failed for a
    measured reason: only top-k experts fire per token, so activated params
    stayed at 180.6M while total went 278M -> 397M. The new experts were
    dormant capacity the router never trained; masking them back OUT improved
    the model. Capability tracks ACTIVATED parameters.

    So this variant moves the three things that change activated params and
    leaves expert count alone:
      * dim 1280 -> 1792 (and expert_dim with it)   attention + every expert wider
      * prelude/coda 2 -> 4                          more non-recurrent depth
      * n_experts_per_tok 4 -> 6                     more experts firing per token

    Same 24 routed + 2 shared experts, same 4 loops, same Ouro vocab.

    Why now
    -------
    The token curve (main-thread #2, run 2026-09-13..15) was flat across three
    points at 278M — L3+ 79.4 / 78.8 / 75.0, distinct1 0.576 / 0.562 / 0.556 —
    at ~16 tokens per activated parameter. That is the un-park condition
    `ideas.md` set for growth in June, met on the instruments it named.

    Cost is NOT yet measured. With the teacher out of rollout (alpha=0), most
    of the 7.2 s/step at 278M scales with the student. `run_scale_profile.sh`
    measures it; do not plan nights on arithmetic.
    """
    return replace(
        mythouro_distill_tiny(),
        dim=1792,
        expert_dim=1792,
        prelude_layers=4,
        coda_layers=4,
        n_experts_per_tok=6,
    )


def mythouro_distill_wide() -> MythOuroConfig:
    """
    ~436M total / ~240M ACTIVATED — `mythouro_distill_tiny` with `expert_dim`
    doubled (1280 -> 2560). The Net2Wider promotion target, 2026-09-15.

    This is the config a checkpoint produced by `grow_width.py` carries, so it
    is what `--student-variant` must name to resume one; `_check_cfg_compat`
    refuses any shape mismatch. Every other field is identical to tiny —
    including `n_experts_per_tok = 4`, which CANNOT change in the same step:
    shared experts are `Expert(dim, expert_dim * n_experts_per_tok)`, and a
    top-k change resizes them by a non-integer factor Net2Wider cannot produce.

    Why this and not `mythouro_distill_mid`: mid (460M activated) is the
    from-scratch reference shape, and from-scratch costs ~150 nights just to
    reach the plateau the trained 278M model is already at. Widening the trained
    model is function-preserving (verified on curve@9000: max |logit delta|
    1.5e-5, argmax agreement 100%), so nothing is lost and the curve continues.
    Widen again (2560 -> 5120) when this size goes flat.
    """
    return replace(mythouro_distill_tiny(), expert_dim=2560)


def mythouro_distill_large() -> MythOuroConfig:
    """
    ~922M total / ~695M ACTIVATED — 3.9x the 278M student's activated params.
    Sized for the Max 1100's 48 GB, not the 5070's 12 GB the tiny was built
    for (2026-09-21). dim 2048, prelude/coda 6, top-k 6, same 24 experts, same
    4 loops, same Ouro vocab. From-scratch; no promotion path. Estimated
    ~34 GB static+activations at mb2/seq1024 — the scale profile measures it.
    """
    return replace(
        mythouro_distill_tiny(),
        dim=2048, expert_dim=2048,
        prelude_layers=6, coda_layers=6,
        n_experts_per_tok=6,
    )


def mythouro_distill_xlarge() -> MythOuroConfig:
    """
    ~1.15B total / ~866M ACTIVATED — 4.8x. dim 2304, prelude/coda 6, top-k 6.
    The largest config with plausible headroom on 48 GB (~9 GB estimated).
    Profile before committing a night; if it fits and runs under ~13 s/step it
    is the target, otherwise `mythouro_distill_large` is.
    """
    return replace(
        mythouro_distill_tiny(),
        dim=2304, expert_dim=2304,
        prelude_layers=6, coda_layers=6,
        n_experts_per_tok=6,
    )


def mythouro_distill_large_bal() -> MythOuroConfig:
    """
    `mythouro_distill_large` with the ATTENTION-TO-FFN ACTIVE RATIO REBALANCED.
    The B arm of the 2026-09-23 A/B. Two knobs move, both directly on the ratio:

        n_kv_heads   4 -> 8       (more KV heads = more attention params)
        expert_dim   2048 -> 1664 (less active FFN)

    ⚠ n_kv_heads must DIVIDE n_heads (16): valid values are 1/2/4/8/16. A first
    draft used 12 and died in scaled_dot_product_attention — "Number of heads in
    key and value must divide the number of heads in query". 2026-09-23.

    Everything else is identical to `large`: dim 2048, 16 heads, prelude/coda 6,
    24 experts, top-6, 4 loops, Ouro vocab.

    | | activated | attn | ffn(active) | **ffn:attn** |
    |---|---|---|---|---|
    | large | 695M | 136M | 428M | **3.14** |
    | large_bal | 680M | 177M | 418M | **2.36** |
    | standard SwiGLU block | | | | 2.00 |
    | Ouro-2.6B (2048/5632) | | | | 2.06 |

    **Activated params match to 2.2%**, so the arms cost about the same per
    token and the comparison is fair on a fixed time budget.

    ⚠ 2.36 does not reach 2.00. With prelude/coda held at 6 and GQA restricted
    to divisors of 16, nothing lands on the target: the alternatives overshoot
    to ~1.62-1.68. 2.36 is the closest available to Ouro's own 2.06 and moves
    68% of the way from our 3.14. If the axis matters, that should be visible;
    if it is not, a larger swing is the follow-up, not the first test.

    Why: measured 2026-09-22, our configs run 3.14-7.04 against 2.00 for a
    standard block and 2.06 for our own teacher. LoopMoE (arXiv 2606.04438)
    includes a capacity-balancing strategy specifically to recover this ratio in
    looped MoE models. Attention is what moves information BETWEEN positions —
    what binds a question to its answer — so an attention-starved model would be
    expected to look fluent per-token and poor at relevance, which is the
    failure we measure across every lineage.

    ⚠ The causal claim is LoopMoE's, read from an abstract. What is ours is the
    number: our ratio is off by 1.6-3.5x and nobody had looked.
    """
    return replace(
        mythouro_distill_large(),
        n_kv_heads=8,
        expert_dim=1664,
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
