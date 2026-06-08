# MythOuro Model Growth — Design Notes

Concrete plan for promoting a trained MythOuro checkpoint into a wider /
deeper / more-expert variant *in a function-near-preserving way*, then
continuing training on the larger model. The reference target for this
document is **MoE expansion** because it has the lowest risk profile
for MythOuro's architecture and the cleanest precedent (MoE-LPR, 2024).

This is a design doc, not committed code. Build the implementation against
the contracts here.

---

## Motivation

After distill v1 (5K steps) + SFT v2 (3K steps), the MythOuro reference
artifacts prove the recipe works at 278M parameters. The bottleneck for
producing usable text is parameter count, not training quality. The
options for crossing into useful capability:

1. **Train a larger variant from scratch** — Path 2 from the post-SFT
   discussion. Predictable but ~14 nights of overnight compute on this
   hardware to hit a useful 1B / 20K-step run.
2. **Promote the trained 278M into a larger architecture** — Path 3.
   Reuses the 278M training as a warm-start, cuts overnight-count to
   ~4–5 for an equivalent quality jump. Less predictable but faster.

This doc designs Path 3 in concrete terms.

---

## Why MoE expansion first

MythOuro has three orthogonal growth axes:

| Axis | Risk | Function preservation |
|------|------|-----------------------|
| **MoE width** (more experts) | **Low** | Strictly possible with sentinel routing |
| **Hidden/expert width** (Net2Wider) | Low–Medium | **Function-preserving even with SiLU** — Net2Wider duplicates units + splits the downstream weight, which holds for any element-wise activation |
| **Layer depth** (Net2Deeper) | Medium | **Blocked by SiLU** — needs an idempotent activation |
| **Loop depth** (`max_loop_iters`) | Trivial | Already free at inference |

**Correction (activation idempotency only blocks *depth*, not *width*).**
An earlier version of this doc claimed SiLU's non-idempotence breaks both
Net2Wider and Net2Deeper. That's wrong:

- **Net2Wider** duplicates a unit and splits its outgoing weight in half so
  the sum is unchanged. The activation is applied per-unit and identically to
  both copies, so this is function-preserving for **any** element-wise
  activation — **including SiLU/SwiGLU**. Widening an expert's inner dim (or,
  with more bookkeeping, the model dim) is on the table for us with no
  activation change.
- **Net2Deeper** inserts an identity-initialised layer, which applies the
  activation *twice*: `φ(φ(Wx))`. That equals `φ(Wx)` only if `φ(φ(z)) = φ(z)`
  — **idempotency**. ReLU has it; **SiLU does not** (`SiLU(SiLU(x)) ≠ SiLU(x)`
  because SiLU's output goes negative). So depth growth would need an
  idempotent activation (e.g. ReGLU) or accept a recoverable loss spike.

MoE expansion remains the lowest-risk axis (strictly loss-preserving via
sentinel routing), but **Net2Wider is a genuine second option** — it's also
function-preserving and reaches scale on a different axis (width). The growth
order is: **MoE expansion first (proven), then Net2Wider for width toward ~1B**;
Net2Deeper is the only axis that the SiLU/SwiGLU choice actually blocks.

---

## Target promotion: 278M (24+2 experts) → ~420M (48+2 experts)

| Property | Before (current `mythouro_distill_tiny`) | After (proposed `mythouro_distill_small`) |
|----------|------------------------------------------|-------------------------------------------|
| `dim` (hidden) | 1280 | 1280 (unchanged) |
| `n_layers` | per variant | unchanged |
| `n_experts` (routed) | 24 | **48** |
| `n_shared_experts` | 2 | 2 (unchanged) |
| `n_experts_per_tok` (topk) | 6 | 6 (unchanged) |
| `expert_dim` | per variant | unchanged |
| Param count | 278M | ~420M |
| Active params/token | ~80M | ~80M (unchanged, topk fixed) |

Total params grow by ~50% but active compute per token is *unchanged*
because top-k routing fires the same number of experts regardless of
pool size. The new experts are pure storage — they only activate for
tokens the router decides to send them.

---

## Affected MythOuro code

Four code surfaces change shape when `n_experts` grows. Each needs a
deterministic remap from the smaller checkpoint to the larger one.

### 1. `MoEFFN.router` — `nn.Linear(dim, n_experts, bias=False)`

Reference: `mythouro/main.py:672`.

Weight shape: `(n_experts, dim)`. Promotion appends new rows for the new
expert indices.

### 2. `MoEFFN.router_bias` — `register_buffer(torch.zeros(n_experts))`

Reference: `mythouro/main.py:674`.

Buffer shape: `(n_experts,)`. The DeepSeek-V3 aux-loss-free balancer at
`mythouro/training_utils.py:143` updates this *outside* the optimizer,
so it survives checkpoint loading as long as we resize the buffer
during promotion.

### 3. `MoEFFN.routed_experts` — `nn.ModuleList([Expert(...)] * n_experts)`

Reference: `mythouro/main.py:676`.

Each `Expert` (`mythouro/main.py:615`) has three weight matrices:
`gate`, `up`, `down`. Promotion appends new `Expert` modules. Shared
experts (`MoEFFN.shared_experts`) are unchanged — promotion only grows
the routed pool.

### 4. `ExpertSpecializationProbe.classifiers` — `nn.ModuleList([Linear] * n_experts)`

Reference: `mythouro/training_utils.py:1118`.

One classifier per expert. Promotion appends new ones, initialised
fresh because there's no meaningful prior over new experts' domains.

---

## Promotion algorithm

Given:
- Source checkpoint `src_ckpt` (e.g., `step_0003000.pt`) with `n_experts = E_src`
- Target `n_experts = E_tgt` where `E_tgt = E_src * k` for an integer `k`
  (the doubling case `k=2` is the cleanest; non-integer ratios complicate
  the "split downstream weights" math we want for partial loss
  preservation, so reject non-integer ratios).
- Source `MoEFFN` with router weights `W_src ∈ ℝ^{E_src × dim}` and
  experts `[E_0, ..., E_{E_src-1}]`.

### Step 1 — Allocate target `MoEFFN`

Build a fresh `MoEFFN` with `cfg.n_experts = E_tgt`. All buffers and
modules are freshly constructed with default init.

### Step 2 — Copy source experts verbatim into the first `E_src` slots

```python
for i in range(E_src):
    tgt_moe.routed_experts[i].load_state_dict(
        src_moe.routed_experts[i].state_dict()
    )
```

The first `E_src` target experts compute exactly the source experts'
function. No perturbation, no scaling.

### Step 3 — Duplicate-and-perturb the new expert slots

For `i in range(E_src, E_tgt)`:

```python
parent = (i % E_src)   # round-robin source
tgt_moe.routed_experts[i].load_state_dict(
    src_moe.routed_experts[parent].state_dict()
)
# Symmetry-breaking perturbation on the down projection.
with torch.no_grad():
    tgt_moe.routed_experts[i].down.weight.mul_(0.0)
```

**Zeroing the `down` projection** is the key trick — it makes the new
expert's *contribution* zero regardless of what `gate` and `up` do.
The expert is "alive" (gradient flows back through it) but its output
is zero at promotion, so the model's function is exactly preserved
through these slots.

Alternative (more aggressive): perturb `gate` and `up` weights by
`+ 1e-3 * randn_like(...)` to break exact symmetry between parent and
child experts. Optional — gradient noise during training breaks
symmetry anyway, but explicit perturbation accelerates divergence.

### Step 4 — Resize router weights

Source router: `W_src ∈ ℝ^{E_src × dim}` (note: PyTorch `nn.Linear`
stores weight as `(out_features, in_features)` so each row is one
expert's routing direction).

```python
W_tgt = torch.zeros(E_tgt, dim)
W_tgt[:E_src] = W_src
W_tgt[E_src:] = W_src.repeat((E_tgt // E_src) - 1, 1)   # tile
```

New router rows are copied from their parent expert's routing
direction. This means a token that would have routed to expert `j` in
the source has equal pre-bias logit for target expert `j` and target
expert `j + E_src` (the child). The bias buffer then breaks the tie.

### Step 5 — Resize router_bias buffer with sentinel value

This is the load-bearing piece. The source bias has spent the entire
training run balancing 24 experts. We want to:

- Preserve the source's balanced state for the first `E_src` experts
- Set the new experts' bias such that they are *not* selected initially
  (so output is identical to source at promotion)

```python
b_tgt = torch.zeros(E_tgt)
b_tgt[:E_src] = src_moe.router_bias.clone()
# Large negative sentinel — new experts never enter top-k initially.
b_tgt[E_src:] = SENTINEL_BIAS  # e.g., -1e9
```

With `b_tgt[new] = -1e9`, `(logits + router_bias).topk(K)` never
selects a new expert because their effective score is `-inf`. The
model's forward pass at promotion is **exactly** identical to the
source.

### Step 6 — Decay the sentinel over time

The sentinel-based exact-preservation is only useful as a starting
point — if new experts stay at `-1e9` forever, they never train.

Two strategies:

**6a. Linear sentinel decay** (recommended)
- Start `b_tgt[new] = -100.0` (large but not `-inf`)
- Over the first `N_decay` training steps (e.g., 500), linearly
  interpolate `b_tgt[new]` toward 0
- After `N_decay`, the bias is 0 and the regular DeepSeek-V3 updater
  takes over

**6b. Hand off to the DeepSeek-V3 updater immediately**
- Start `b_tgt[new] = 0`
- New experts get traffic immediately
- Loss spike is larger but training is simpler
- Updater's sign-of-difference rule still keeps everything balanced

The choice depends on tolerable loss-spike size. 6a gives strictly
function-preserving promotion with a smooth handoff; 6b is simpler but
spikes loss by perhaps 0.2–0.5 nats at promotion.

### Step 7 — Resize the specialization probe

```python
new_probe = ExpertSpecializationProbe(n_experts=E_tgt)
for i in range(E_src):
    new_probe.classifiers[i].load_state_dict(
        src_probe.classifiers[i].state_dict()
    )
# Probe classifiers for new experts stay at their fresh init —
# domain labels are unknown until the new experts specialize.
```

### Step 8 — Save the promoted checkpoint

Write a new checkpoint dict with:
- `model_state_dict` containing the resized weights
- `cfg_dict` reflecting `n_experts = E_tgt`
- `step = 0` (the promoted model starts a fresh training counter)
- All other metadata from source (vocab_size, rng_state, etc.)

---

## Training contract after promotion

The promoted model is NOT just a checkpoint resume — it's a *new model*
that should start its own training run. Specific requirements:

### 1. Fresh optimizer state

The source AdamW moments are sized `(E_src, dim)` for the router and
similarly for each Expert. Loading them into the target optimizer
would either crash (shape mismatch) or apply stale moments to fresh
weights (worse than fresh init). **Build a new AdamW from the
promoted model and do NOT load the source optimizer state.**

### 2. Fresh LR schedule

Whatever LR the source ended at is not appropriate for the promoted
model. Start a new cosine warmup → decay cycle.

Recommended starting LR after promotion: `lr_promoted = lr_source_peak * 0.3`.
The 0.3 multiplier reflects that the source weights are already useful
and we don't want to perturb them as aggressively as a from-scratch
run. Warmup over 200 steps. Decay to `lr_promoted * 0.1` over the full
target training budget.

### 3. Optional: freeze source experts for warm-in

For the first 500–1000 steps of post-promotion training, freeze the
parameters of the first `E_src` experts. This mirrors the MoE-LPR
approach: let the new experts find a niche while the source experts
hold the existing knowledge in place.

```python
for i in range(E_src):
    for p in model.recurrent_block.moe.routed_experts[i].parameters():
        p.requires_grad = False
```

After step 500, unfreeze and resume joint training. Optional — not
strictly needed because the sentinel-bias trick already minimises
interference, but the MoE-LPR paper finds it helps preserve original
language capabilities.

### 4. Continue with the same depth regulariser

`depth_reg_coeff = 0.1` (the v1/v2 value). No changes — the ACT halt
distribution is a model-level property, not router-specific.

### 5. Watch the diagnostic counters

The MoE utilisation log lines (`cv`, `max%`, `min%`, `bias|·|₂`) are
the canary for whether promotion succeeded. Healthy trajectory:

- **Step 0** (promotion): cv low (~0.3), max% modest (~7%), min% effectively 0
  for new experts because they have sentinel-biased zero traffic
- **Step 500** (sentinel decayed): min% climbs as new experts start
  receiving traffic
- **Step 2000**: routing rebalances to roughly uniform across all 48
  experts. cv stays in the 0.3–0.5 band. max% in 4–6% range
  (target is `~1/n_experts ≈ 2.1%` but with topk=6 you'd expect ~4%
  for a balanced 48-expert pool).
- **Step 5000**: utilisation similar to a from-scratch 48-expert
  MoE, but with a head-start advantage from the warm initialisation.

Failure modes to watch for:

- `max% > 20%` sustained after sentinel decay → expansion failed to
  diversify; new experts collapsed back into the source's distribution.
  Recovery: increase symmetry-breaking perturbation in step 3, retry.
- `cv > 1.5` sustained → router can't balance the larger pool; some
  experts are dead. Recovery: longer sentinel decay, increase `bias_lr`
  in the updater.

---

## Code sketch

New module `mythouro/grow.py`. Approximate signature:

```python
def grow_moe(
    src_state_dict: dict,
    src_cfg: MythOuroConfig,
    expansion_factor: int = 2,
    *,
    sentinel_bias: float = -100.0,
    perturb_scale: float = 0.0,
) -> tuple[dict, MythOuroConfig]:
    """
    Promote a MythOuro state dict to a larger n_experts variant.

    Args:
        src_state_dict     -- weights from a trained checkpoint
        src_cfg            -- cfg the source was trained with
        expansion_factor   -- multiplier on n_experts (2 → double)
        sentinel_bias      -- initial bias for new experts; large
                              negative makes promotion function-preserving
        perturb_scale      -- gaussian noise on duplicated experts'
                              gate/up weights to break symmetry. 0.0 is fine.

    Returns:
        (new_state_dict, new_cfg) — ready to load into a fresh
        MythOuro instance built from new_cfg.
    """
```

A standalone CLI script `tools/grow_checkpoint.py`:

```python
python tools/grow_checkpoint.py \
    --src archived_models/mythouro_distill_tiny_sft_v2/step_0003000.pt \
    --dst checkpoints_grown/promoted_step_0.pt \
    --expansion-factor 2
```

Then a normal resume of training/distill or training/sft against the
grown checkpoint, with `--ckpt-dir checkpoints_grown` so the new training
state stays separate from the source.

---

## Tests we'll need

1. **Function preservation at promotion**: build a tiny `MoEFFN` with
   `n_experts=4`, promote to `n_experts=8`, assert that
   `tgt(x) == src(x)` for random inputs (within float tolerance) when
   `sentinel_bias = -1e9` and `perturb_scale = 0`.

2. **Router weight shape correctness**: assert that the promoted
   `router.weight` has shape `(E_tgt, dim)` and that the first `E_src`
   rows are byte-identical to the source.

3. **Expert weight identity**: assert that target experts `[0:E_src]`
   are byte-identical to source experts.

4. **Down projection zeroed for new experts**: assert that experts
   `[E_src:E_tgt]` have `down.weight` zeros.

5. **Sentinel bias placement**: assert that `router_bias[:E_src]`
   equals source bias and `router_bias[E_src:]` equals `sentinel_bias`.

6. **End-to-end: promoted model produces same logits as source**:
   given the same input, the promoted MythOuro's forward output
   should match the source MythOuro's forward output exactly (within
   floating point tolerance).

7. **Train step works after promotion**: a single training step on
   the promoted model should not raise (catches shape mismatches in
   the optimizer / router_bias buffer / specialization probe).

8. **Bias updater handles new n_experts**: pass synthetic counts to
   `update_router_bias_from_counts` against a promoted module and
   verify it updates the buffer in the expected direction.

---

## Open questions to resolve before coding

1. **Non-integer expansion ratios** — do we support `E_tgt = 1.5 * E_src`?
   Recommend NO for v1 — the "tile router weights" math is cleaner with
   integer ratios. Defer until later.

2. **Width growth (Net2Wider) on the same checkpoint** — should
   `grow.py` support hidden-dim growth too, or be MoE-only?
   Recommend MoE-only for v1 to keep scope tight. Width growth has the
   SiLU-non-idempotency caveat and deserves its own design pass.

3. **Should the depth regulariser be relaxed during the post-promotion
   recovery window?** New experts cause router output entropy to shift,
   which might transiently affect ACT λ values. Recommend leaving at
   0.1 and watching — easier to add a transient annealing if needed
   than to debug a regulariser change in the same session as a
   promotion.

4. **FSDP compatibility** — the buffer resize for `router_bias` happens
   on a single-process target. Under FSDP-wrapped training the buffer
   needs to be all-reduced consistently. Defer FSDP support for grow.py
   v1 (single-process is fine for tonight-scale runs at ~420M).

5. **Checkpoint version field** — `mythouro/checkpointing.py` already
   has a version field. Bump it on promoted checkpoints and add a
   `growth_history` entry recording the promotion (source path,
   expansion factor, date) so the lineage is recoverable from the
   checkpoint metadata.

---

## Effort estimate

| Task | Time |
|------|------|
| Write `mythouro/grow.py` with the algorithm above | ~2h |
| Write `tools/grow_checkpoint.py` CLI | ~30min |
| Write 8 tests above | ~1h |
| Run tests + iterate to green | ~1h |
| Promote v2 checkpoint, run 500-step smoke | ~3h overnight |
| Inspect promoted checkpoint, decide on full run | ~30min |

**Total**: one design + implementation session (~4h hands-on), one
overnight to validate. If both go well, the next overnight starts a
real training run on the promoted model.

---

## What this design does NOT cover

- **Hidden-dim growth (Net2Wider)**: separate doc when we're ready
- **Layer-count growth (Net2Deeper)**: same
- **Loop-count growth**: trivial, just `cfg.max_loop_iters += K`
- **Cross-tokenizer expansion**: bigger vocabulary, requires
  embedding resize
- **Pruning the source experts during promotion**: e.g., consolidating
  underused experts before promoting. Out of scope.

These can all be additional growth axes once MoE expansion is proven.

---

## Related design decision: loop-loss supervision (divergence from Ouro)

Not a growth topic, but a recurrent-depth design decision documented here
because it's the same "how do we shape per-loop behaviour" problem the growth
machinery interacts with (the depth regulariser, the halt distribution).

### What Ouro actually does (from the paper, arXiv:2510.25741)

Verified against *Scaling Latent Reasoning via Looped Language Models* (Zhu
et al., 2025):

- **Loop count is FIXED at 4 during training** (they tried 8, dropped to 4 in
  Stage 1b after loss spikes / gradient oscillations). **No curriculum, no
  linear loop progression.** What progresses linearly in Ouro is *sequence
  length* (4K→16K→64K→32K), not loop count.
- **Loss is supervised at every recurrent step**, combined as an *expected task
  loss* weighted by exit probability:
  `L = Σ_t pφ(t|x) · L^(t)`, with an exit gate computing an LM loss at each
  step `t ≤ Tmax`.
- **Adaptive depth via entropy regularisation toward a uniform prior**:
  `−β·H(pφ(·|x))` penalises collapse to the deepest step; uniform prior
  `1/Tmax` keeps exit decisions depth-unbiased.

### What MythOuro does, and where it diverges

| Aspect | Ouro | MythOuro | Match? |
|--------|------|----------|--------|
| Loop count in training | fixed 4 | **linear curriculum 2→4** (`LoopCurriculum`) + `--random-depth` sampling | ✗ our addition |
| Entropy / uniform-prior depth reg | `−βH(pφ)` | "PonderNet × Ouro KL-to-uniform" depth-reg (`depth_regularization_loss`) | ✓ faithful |
| Loss across loops | **per-step LM loss weighted by exit prob** | **final-loop loss only** (`h_K`); halting decoupled into ACT + depth-reg | ✗ **diverged** |

The substantive divergence is the **loss across loops**. Background: our first
implementation used an ACT-weighted-sum output `Σ wₜ hₜ` and computed the loss
on that — which gave the optimiser a direct lever to collapse to a single loop
(`λ₀→1`). We fixed collapse by returning `h_K` (final loop) during training and
letting the depth regulariser alone shape the halt distribution. See the
"ACT loop collapse" entry in the roadmap failure modes.

Ouro solves the *same* collapse problem a third way we never tried: keep a
per-step loss, but weight each step's loss by its exit probability **and** add
entropy regularisation. Their exit gates are trained *by the task loss itself*;
ours are shaped *only* by the regulariser. We adopted Ouro's entropy-reg half
(faithfully) but not its per-step-weighted-supervision half.

### Status: deliberate-enough, but an open optimisation

We did not consciously reject Ouro's per-step weighted loss — we simply solved
collapse differently (h_K + depth-reg) before checking their exact mechanism.
Our approach **works**: halt distributions stay spread, loop_efficiency ~0.5,
all three halt mechanisms fire. So this is an *optimisation to consider*, not a
bug. Ouro's per-step weighted loss is arguably more principled (it trains the
exit gates directly), and is a candidate future experiment — see the
implementation sketch in the roadmap's research-questions / experiments area.
