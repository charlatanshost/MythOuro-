# Training runs — comparison table

Living comparison of every training session's eval stats, built from the raw
eval JSONs where they exist (paths in the last column). Update after each run.

> **Comparability caveats** (read before drawing conclusions):
> 1. **Eval emission path changed 2026-06-09 (P0.3):** v1–v5 numbers were
>    measured through the old under-summed `h_out` blend (pessimistic — fixing
>    it alone moved v2 from PPL 46.3→39.25 on the same weights). Runs from
>    `moe_s0` onward use the trained `h_K` path. Cross-era comparisons are
>    directional, not exact.
> 2. **Absolute PPL is flattered by FineWeb train/eval stream overlap** —
>    applies to all runs equally, so within-protocol comparisons stand; don't
>    quote against external baselines.
> 3. v2's PPL rise vs v1 is expected (SFT specialises away from raw web text).

## Run overview

| Run | Date | Arch / params | Method + recipe deltas | Steps | Final PPL | loop_eff | ECE | Verdict |
|-----|------|---------------|------------------------|------:|----------:|---------:|----:|---------|
| **v1** | 06-01 | MoE 24exp / 278M | distill from scratch; warmup 500, depth-reg 0.3, mb1/ga8 (**the proven recipe**); pre-fix code (P0.1 noise injection active) | 5000 | 37.42 | 0.50 | 0.041 | archived |
| **v2** | 06-04 | = v1 | SFT on v1 (math+code) | 3000 | 46.27¹ | 0.50 | 0.058 | archived |
| **v3** | 06-04/05 | MoE 48exp / 420M | MoE-grown + SFT | 3500 | n/a² | — | — | archived |
| **v4** | 06-05 | = v3 | OpenHermes SFT, fp32 | 3000 | n/a² | — | — | archived; best PoC behaviour |
| **v5** | 06-06 | MoE 96exp / 632M | 2nd MoE growth + SFT | 2887 | n/a² | — | — | archived; expert-ceiling data point |
| *(flatline)* | 06-10 | dense + MoE, both | ablation attempt on **script defaults** (warmup 200, depth-reg 0.1) | ~2000 | 4,462–8,301 | 1.0 (no halt) | — | **dead** — see failure modes; root-caused to recipe defaults |
| **moe_s0** | 06-10 | MoE 24exp / 278M | ablation arm 1: distill from scratch; proven recipe; **post-fix code** | 4000 | **5.72** | 0.500 | 0.015 | ✅ **6.5× better than v1 final in 1k fewer steps** |
| **dense_s0** | 06-10/11 | dense / 180M | ablation arm 2: same recipe/seed as moe_s0, FFN only difference | 4000 | **22.66** | 0.500 | 0.026 | ✅ complete — **MoE wins 4.0× at seed 0**; note dense still beats v1's 37.4 |
| **moe_s1** | 06-11 | MoE 24exp / 278M | ablation arm 3: seed-1 repeat of moe_s0 | 4000 | **22.23** | 0.500 | 0.025 | ✅ complete — **~4× seed spread vs moe_s0 (5.72)**; ≈ dense_s0 (22.66). dense_s1 now decisive |
| **dense_s1** | 06-12 | dense / 180M | ablation arm 4: seed-1 repeat of dense_s0 | 4000 | **20.83** | 0.500 | 0.041 | ✅ complete — **dense BEATS MoE at seed 1** (20.83 < moe_s1 22.23); seeds disagree |
| **v6** | 06-13 | MoE 24exp / 278M | clean-data SFT on moe_s0 (no OpenAI provenance; +medical/chem/code) | 3000 | 5.74 | 0.500 | 0.023 | ✅ complete — PPL flat vs base (broad-mix, near-zero general-competence cost), calibration/depth held. **BUT generation still degenerate (scale/token ceiling — see read)** |

¹ SFT specialises toward chat; web-text PPL rises by design.
² v3–v5 predate per-run eval archiving — only inspector/behavioural results in
their MODEL_CARDs (reconstructed 2026-06-08).

## PPL trajectory (per 1k-step eval)

| Step | v1 (old eval) | **moe_s0** | **dense_s0** | **moe_s1** | **dense_s1** |
|-----:|------:|------:|------:|------:|------:|
| 1000 | 368.4 | 559.9 | 578.0 | 475.4 | 525.9 |
| 2000 | 178.6 | 112.1 | 150.3 | 122.9 | 132.1 |
| 3000 | 81.7 | **11.1** | 33.6 | 34.0 | 31.3 |
| 4000 | 51.8 | **5.72** | 22.7 | 22.2 | 20.8 |

## ABLATION VERDICT (2026-06-12): inconclusive — no robust MoE advantage; seed variance dominates

Complete 2×2 (within-seed is the only valid MoE-vs-dense comparison):

| | seed 0 | seed 1 |
|---|---:|---:|
| **MoE** | 5.72 | 22.23 |
| **dense** | 22.66 | 20.83 |
| winner | MoE 4.0× | **dense 1.07×** |

**The seeds disagree on direction.** Three of four runs cluster at 20.8–22.7;
only `moe_s0` (5.72) is an outlier. Applying the pre-registered rule (keep MoE
only if >5–10% better *across* seeds): **NOT satisfied** — seed 1 favours dense.
Honest conclusion: **`moe_s0`'s 5.72 was a favourable-seed outlier, not an
architecture win; MoE shows no robust advantage at this scale/seed-count.**

The real finding is **seed variance**: a 4× PPL swing from seed alone means
4000-step / ~16M-token runs are *underpowered* to separate architecture from
noise — the comparison is premature, not settled in dense's favour either.
Constant across all four fixed-code runs: loop_eff converges to exactly 0.500.

**Consequences:**
- **MoDr's gate is NOT cleanly passed** — do not commit to the unified
  expert+depth router on this evidence.
- To actually resolve it: (a) more seeds (≥3–5) and/or (b) longer runs (token
  count is the variance lever) before claiming either way; or (c) adopt dense
  for simplicity, since MoE's extra complexity is currently unjustified by data.
- `moe_s0` (5.72) remains the **best single checkpoint** and the right SFT base
  (v6) regardless — "best checkpoint" and "architecture verdict" are separate
  questions. Its luck doesn't make it worse; it just isn't *evidence for MoE*.

Dense sidecars: `checkpoints_ablation_dense_s{0,1}/`.

Note the crossover: moe_s0 starts *slower* (560 vs 368 at step 1000 — longer
warmup in effect) then collapses past v1 between steps 2000–3000, exactly as
the full-depth curriculum phase (step 2000+) kicks in.

## loop_efficiency / ECE trajectory (moe_s0)

| Step | loop_eff | ECE |
|-----:|---------:|----:|
| 1000 | 0.926 (full depth, ACT untrained) | 0.145 |
| 2000 | 0.572 | 0.096 |
| 3000 | 0.496 | 0.011 |
| 4000 | **0.500** (design-band center) | 0.015 |

## Continuation (moe_s0 → 32M tokens) — the definitive local-coherence test (2026-06-13)

Continued moe_s0 (warm restart, lr 1e-4) from step 4000 → 8000 (~16M → ~32M
distill tokens). **PPL trajectory: 5.72 → 4.04 → 3.52 → 3.18 → 3.057** (each
+~4M tokens; drops halving → plateau ~3.0). ECE 0.0052 (best ever), loop_eff
0.500 throughout. Best checkpoint the project has: `checkpoints_distill_cont/
step_0008000.pt`.

**But generation is STILL degenerate** (`reports/inspect_cont8000.txt`): every
prompt → pure repetition (`a a point point`, `is is is`, `R R R`), same class
as the moe_s0 base, low uncertainty on its own garbage. **2× the tokens halved
PPL but bought zero coherence.**

**Verdict — the local-coherence question is settled (negative):**
- PPL (teacher-forced next-token objective) and free-generation coherence are
  **decoupled** at this scale — the model mode-collapses in autoregression
  regardless of how low PPL goes. Low PPL ≠ can talk.
- 32M tokens is ~60,000× short of coherent small models (~2T); 2× was never
  going to cross it. Incremental local distill **lowers PPL but does not buy
  coherence** — now proven, not theorized.
- Combined with v4 (632M, also sub-coherent), **coherent text is not reachable
  at the params/data budget the local rig provides.** It needs the real
  scale-up (more params AND vastly more tokens) = rented compute. The local rig
  validates pipeline + recipe; it cannot produce a coherent model.
- SFT on this base (the next planned experiment) will likely give a v4-class
  result: better *behavioral* surface (register/format/halting) but not content
  coherence — SFT styles fluency the base has, and this base lacks it.

## cont_sft (SFT on the 3.06 distill base) — 278M floor confirmed (2026-06-14)

Clean-mix SFT on the continuation base (278M, PPL 3.06 → 3.38 post-SFT, ECE
0.0077). **Generation still degenerate** (`reports/inspect_cont_sft_gpu.txt`):
`is is is`, newlines, faint code-indent echo on the fib prompt — same class as
v6. Does NOT halt on `<|im_end|>` (just newlines).

**The better base did NOT make SFT's behavioral gains take at 278M.** Three
independent confirmations now that **278M is below the behavioral-coherence
floor regardless of tokens/base/data/code**: continuation (more tokens), v6
(clean SFT on 5.72 base), cont_sft (clean SFT on 3.06 base). The only config
that ever showed behavioral coherence was **v4 at 420M** — and that card's
claims were likely optimistic (v2, also 278M, almost certainly looked like
this; its halting/guard write-up was cherry-picked).

**Honesty-thesis caveat:** uncertainty traces are 0.005–0.044 (very low) on
the degenerate output — the model is *confident in its garbage*. The eval ECE
(0.0077) measures next-token calibration on REAL text; it does NOT transfer to
the model judging its own free generation. "Honest specialist" behaviour is
not present at this scale.

**Decision implication:** next behavioral test = grow 24→48 → SFT (reach the
v4 420M size) — best-justified shot but not guaranteed (v4 was broken-code +
OpenAI data). The 278M ceiling is now thoroughly mapped; coherence (behavioral
or content) needs the rented scale-up.

## small_sft (420M, fair test) — local scaling CONCLUSIVELY closed (2026-06-14)

Grew the 3.06 continuation base 24→48 experts (~420M, function-preserving) →
clean-mix SFT. **The training-time MoE cv confirms a FAIR test:** cv rode the
sentinel decay (1.21 @100 → 0.83 @500) then tightened to **0.223 @800 (min
1.3%)** — all 48 experts integrated, balanced as well as v3's best (the P0.2
fix working). Final eval: PPL ~3.05–3.36, ECE 0.017, loop_eff 0.500, training
CE down to 0.4–0.7.

**Every metric healthy — generation still degenerate** (`is is is`, newlines,
`R R R`; `reports/inspect_small_sft_gpu.txt`). This is the conclusive result:
**at this scale, metrics ≠ capability.** The model nails teacher-forced
next-token prediction (low loss/PPL) but mode-collapses in free generation
(exposure-bias collapse in a radically undertrained model) — independent of
params, routing health, or code correctness.

**Local scaling is now exhaustively closed, fair tests at every lever:**
| lever | result |
|---|---|
| more tokens (continuation, 278M) | degenerate |
| SFT (v6 / cont_sft, 278M) | degenerate |
| +params, experts integrated (small_sft, 420M) | degenerate |

The bottleneck is **training scale (token volume)**, confirmed not-fixable by
params/data/code/recipe locally. v4's "best PoC behavior" is almost certainly a
cherry-picked read — small_sft is its fixed-code/clean-data/integrated
equivalent and it's degenerate. **The engineering is done and validated; only
scale remains → the rented scale-up is the sole remaining path, now fully
de-risked.**

## v4 vs small_sft head-to-head (2026-06-14) — RETRACTION: local path NOT closed

Ran the same prompts on archived **v4** (it still exists) vs small_sft, same
current code (`reports/inspect_v4_compare.txt`). **v4 is categorically better:**
- math prompt → `"product of 2&9... ≈9) ox1$"` (attempts a math answer, LaTeX);
  code → `"console()... Solution"`; history → `"City... Date (1000)"`.
- **Does NOT mode-collapse** (varied tokens, never `is is is`), attempts
  domain-relevant content, and uncertainty is high+appropriate (0.47–0.56, knows
  it's unsure) vs small_sft's confidently-wrong 0.03–0.06.

It's word-salad, not coherent — but it's genuinely *"more than what we have,"*
exactly as the user said. **The prior "local scaling conclusively closed" /
"metrics ≠ capability is the whole story" conclusion was WRONG and is retracted.**

Why v4 > small_sft (NOT the emission — both ran identical h_K code, so it's the
weights/training):
- **Cumulative SFT:** v4 had ~6,500 SFT steps *on 420M* (v3 3,500 + v4 3,000) +
  3,000 at 278M. **small_sft had ONE 3,000-step pass.** ~½–⅓ the exposure → the
  prime suspect: small_sft is **under-SFT'd**.
- **Chat data:** v4's card credits OpenHermes with "unlocking the social
  register"; small_sft's clean mix may carry weaker chat-register data.

**Action:** continue SFT-ing small_sft toward v4's exposure (~6,500+ 420M-SFT
steps) and watch whether it crosses from repetition into v4-style varied
generation. The local path has more to give — small_sft was just under-trained.

## Test prompt suite

Run with `python inspect_checkpoint.py --checkpoint <ckpt> --device cpu`
(per-prompt: generated text, per-token uncertainty trace, ACT halt
distribution, MoE utilisation). Keep the **canonical 4** fixed run-to-run so
behavioural reads are comparable; add the **v6+ extension** once a checkpoint
has had clean-data SFT, to probe the north-star differentiators.

**Canonical 4** (`_DEFAULT_PROMPTS` in `inspect_checkpoint.py` — the comparability anchor):

| # | Prompt | Tests |
|---|--------|-------|
| 1 | `The recurrent depth transformer is` | open continuation / fluency |
| 2 | `<\|im_start\|>user\nWhat is 2+2?<\|im_end\|>\n<\|im_start\|>assistant\n` | ChatML instruction-following + halting |
| 3 | `def fibonacci(n):` | code structure |
| 4 | `Q: Roughly what year was the Roman Empire founded?\nA:` | factual recall |

**v6+ extension** (the wedge — pass via `--prompt`; only meaningful post-clean-SFT):

| # | Prompt | Differentiator probed |
|---|--------|-----------------------|
| 5 | `<\|im_start\|>user\nWhat are the common symptoms of iron-deficiency anemia?<\|im_end\|>\n<\|im_start\|>assistant\n` | medical domain (MIRIAD/PubMedQA) |
| 6 | `<\|im_start\|>user\nWhat does the SMILES string CCO represent?<\|im_end\|>\n<\|im_start\|>assistant\n` | chemistry domain (ChemData) |
| 7 | `<\|im_start\|>user\nWrite a Python function that checks if a number is prime.<\|im_end\|>\n<\|im_start\|>assistant\n` | verified-code domain (OpenCodeInstruct) |
| 8 | `<\|im_start\|>user\nWhat is the capital of the fictional country of Zambonia?<\|im_end\|>\n<\|im_start\|>assistant\n` | **honesty / calibration** — should show HIGH uncertainty and ideally decline, not confabulate (the key differentiator) |

Prompt 8 is the most important for the product thesis: the *uncertainty trace*
matters as much as the text. A model that flags "I don't know" on an
unanswerable prompt is the honest-specialist edge in action.

## Behavioural read: v6 clean-SFT test prompts (2026-06-13) — the scale ceiling, seen directly

Ran the canonical 4 + extension (medical, honesty probe) on v6's final
checkpoint. **Generation is still degenerate** — same repetition collapse as
the moe_s0 base, across every prompt and domain:

| Prompt | v6 output | uncertainty |
|--------|-----------|-------------|
| "recurrent depth transformer is" | `response response response…` | low (0.07) |
| "What is 2+2?" (ChatML) | `\n\n\n\n…` | low (0.08) |
| `def fibonacci(n):` | `"""init__(((xxxx…` | low (0.05) |
| Roman Empire year | `R R R R…` | 0.18 |
| iron-deficiency anemia (medical) | `is is is is…` | low (0.08) |
| Zambonia (honesty probe) | `\n\n\n\n…` | low (0.13) |

**Diagnosis — token volume, not params/recipe/architecture:**
- moe_s0 saw **~16M distill tokens**. Coherent small models see **~2T**
  (SmolLM2-**135M**, *smaller* params, is fluent) — a **~120,000× gap**. The
  model is radically undertrained on *data volume*, not under-parameterized.
- SFT can't conjure coherence the base lacks capacity/exposure for — it teaches
  format/behaviour, not fundamental fluency. So flat PPL + working calibration
  + degenerate generation is the **expected** signature at this token budget,
  exactly the "parameter-count ceiling" the README/roadmap always flagged
  (more precisely: a *token-count* ceiling).
- The good PPL (5.74) is partly the FineWeb stream-overlap caveat; the
  degenerate generation shows it does NOT imply generalization at 16M tokens.

**What this run DID validate (the actual goal of this stage):** the full
clean-data pipeline end-to-end (7 sources, contamination guard, loss mask),
near-zero general-competence cost from broad-mix SFT, calibration holding
(ECE 0.02) and depth machinery perfect (0.500) through a domain shift. The
*recipe* is sound; the *scale* is the bottleneck.

**Strategic implication (see roadmap North Star):** the binding constraint is
**training tokens**, and the lever is **throughput × time** (≈278 GPU-h / ~12
days for 1B tokens on the 5070) — which is precisely what rented compute buys.
**Tokens before params:** a longer distill run on the *current* 278M (toward
~1B tokens) likely buys more coherence than jumping to 1B params on the same
16M-token budget. First coherent text is a data-volume milestone, not an
architecture one.

## Behavioural read: moe_s0 test prompts (2026-06-11)

Inspector on the PPL-5.72 checkpoint (4-prompt set, T=0.7/top_k=40, raw output:
`reports/inspect_moe_s0.txt`). **Context: distill-only, no SFT** — the fair
comparison is v1 (also distill-only), not the SFT'd v2/v4.

| Prompt | Output character | Stop | Notes |
|--------|------------------|------|-------|
| "The recurrent depth transformer is" | degenerate repetition ("compared"×12) | `cycle` ✅ | guard fired |
| ChatML "What is 2+2?" | newline/digit loops | `cycle` ✅ | no chat structure (expected pre-SFT) |
| `def fibonacci(n):` | **correctly indented `
    """` docstring opening**, then collapse | `cycle` ✅ | genuine learned code convention — the one bright spot |
| Roman Empire trivia | "R R R R…" | `cycle` ✅ | guard fired |

**Findings:**
1. Real *structure* learning is visible (the docstring), but open-ended
   generation still collapses into repetition attractors — classic small-LM
   degeneration; halt depth steady at 2.0; cycle guard 4/4.
2. **The PPL-vs-generation gap reinforces the stream-overlap caveat** on the
   absolute 5.72 — hold the absolute number loosely; relative comparisons
   (vs v1, vs dense) remain methodologically sound.
3. Behavioural quality at this scale historically arrives with **SFT** (v4's
   registers/halts all came from it) — this read strengthens the case for
   SFT-ing this base as the next pipeline step.

## Where the raw data lives

| Run | Eval JSONs |
|-----|-----------|
| v1 | `archived_models/mythouro_distill_tiny_v1/distill_step_*.json` |
| v2 | `archived_models/mythouro_distill_tiny_sft_v2/sft_step_*.json` |
| moe_s0 | `checkpoints_ablation_moe_s0/distill_step_*.json` (sidecars) |
| flatline | numbers preserved in roadmap failure-mode entry only |

**Convention going forward:** copy each run's eval JSONs into its checkpoint
dir as sidecars immediately after the run (`eval_results/` filenames collide
across runs — that footgun produced a false alarm on 2026-06-10; a per-run
eval-output path is on the action-plan list).

---

# 2026-06-15 — ROOT CAUSE of the post-fix generation collapse: P0.1 noise was load-bearing

The biggest diagnosis since the build. **Every post-fix checkpoint mode-collapses
in free generation (single-token repetition: `is is is`, `\n\n\n`, `R R R`) while
posting healthy PPL/cv/ECE.** v4 (pre-fix lineage) does not. Today's bisection
located the cause and it is **not** data, growth, SFT, or inference — it is the
P0.1 fix itself.

## Evidence chain (all verified 2026-06-15, GPU, raw outputs in `reports/`)

| Checkpoint | Lineage | Generation | File |
|---|---|---|---|
| v4 step_3000 | pre-fix (noise present) | **varied, domain-relevant** | `inspect_v4_compare.txt` |
| distill base step_8000 (PPL 3.06, 24exp) | fixed code | **collapsed** | `inspect_distill_base.txt` |
| grown base (48exp, pre-SFT) | fixed code | collapsed (inherited) | `inspect_grown_base.txt` |
| chat-heavy clean SFT (60% Tulu) | fixed code | collapsed | `inspect_chat_sft.txt` |
| 6k clean SFT / small_sft | fixed code | collapsed | earlier reports |

- **All run the identical current inference path.** v4 is varied through it →
  inference is exonerated; the *weights* are collapsed.
- **Three data mixes** (structured-heavy clean, chat-heavy clean, OpenHermes-era)
  → data has no bearing. The collapse is in recurrence *dynamics*, not content.
  The chat-heavy-clean experiment (the diversity hypothesis) was **refuted**.
- Bisection puts the collapse **upstream of growth and SFT**, in the fixed-code
  **distill base**.

## Mechanism

P0.1 (CHANGES.md): `_init_weights` clobbered the identity-init of
`CrossLoopAttention.o_proj`, so **v1–v5 trained with noise injected into the
hidden state every loop.** The fix made it a clean identity residual. CHANGES.md
flagged this as "the prime suspect for why the post-fix run trains so much
better" — and that is exactly the cost: with the noise gone, the **contractive**
recurrent update `h_{t+1}=A·h_t+B·e+trans_out` (`ρ(A) ∈ [0.32, 0.36]`, measured
in every inspect) converges to a **degenerate fixed point** under free-running
autoregression. Lower teacher-forced PPL, collapsed free generation — two faces
of one change. The accidental noise was a **load-bearing anti-collapse
regulariser.**

## Why this is the good outcome

- **Not a scaling wall.** v4 proves this architecture at this size generates
  varied text. Renting/buying would have **reproduced the collapse bigger** —
  local-first testing prevented exactly the wasted capital we feared.
- The fix is a **tunable code knob**, found for free, locally.

## The fix (staged 2026-06-15, OFF by default)

New config `recurrent_state_noise: float = 0.0` (`mythouro/main.py`). When >0 and
training, perturbs the committed loop state by `σ·RMS(h)·N(0,1)` each loop
(RMS-relative; inference untouched; σ=0 byte-identical to current). The
principled, controlled replacement for the P0.1 accident. CLI: `--recurrent-state-noise`
on both `training/distill.py` and `training/sft.py`. Suggested σ: 0.02–0.1.
Verified: default 0.0, fires only in train+σ>0, both flags parse.

## Confirmation test (pending user go — DO NOT auto-launch)

Continue distillation from the collapsed base with noise on, then inspect:
seed a fresh `--ckpt-dir` with `checkpoints_distill_cont/step_0008000.pt`, run
~2000 steps with `--recurrent-state-noise 0.05` (proven recipe otherwise),
inspect the result. Generation de-collapses → confirmed + fixed. (Faster but
weaker alternative: SFT the grown base with `--recurrent-state-noise 0.05`.)

---

# 2026-06-16 — distill-stage noise test (step 11k) + the capital-gate metric

First test of `recurrent_state_noise` at the **real LR** (distill, lr 3e-4),
resuming the 24-expert base (`mythouro_distill_tiny`) from step 8000 with σ=0.05,
teacher on cuda:2 (5060). Inspected step 11000 (3000 noise steps in).
Raw: `reports/inspect_noise_distill_11k.txt`.

## Result: marginal de-collapse, NOT a clean base, NOT coherent

- **Real but small improvement:** now emits brief *actual English* before
  collapsing — "This would make the…", "In a…" — and repetition unit grew from
  **1 token → 2+ words** ("remove remove", "replace replace"). The pre-noise base
  produced only symbol-salad (`""" """`, `::::`). So the noise mechanism is
  **directionally confirmed.**
- **Still collapsing hard:** every prompt degenerates into repetition within 2–4
  tokens; model is hyper-confident about the garbage (uncertainty ~0.01). Far short
  of v4's sustained variety; not a clean non-collapsing base.
- Depth machinery healthy: uniform halt dist [0.25×4], loops run 3.5–4.0,
  uncertainty falls monotonically with depth.

## Methodological caveat (important)

This run **varied noise AND tokens together** (added σ=0.05 *and* 3000 steps), so
the improvement is **confounded** — it does NOT, on its own, prove "token
starvation." It's consistent with that hypothesis, not a demonstration of it.

## The capital-gate test: the span-length token-curve

The single experiment that converts belief → evidence (and gates renting/buying):
**keep training (more tokens, noise on) and track the coherent-span length before
collapse** as the token count climbs.

- **Metric:** longest coherent (grammatical, non-repeating) prefix the model emits
  before falling into a repetition loop. At 11k it's ~3–4 tokens ("This would make
  the"). Repetition-unit size is a secondary proxy (1→2 words so far).
- **Protocol:** inspect every few thousand steps (≈ every ~50M tokens); record the
  span at each. Same 4-prompt set for comparability.
- **GREEN LIGHT (token starvation, proven → capital justified):** span grows
  monotonically with tokens (4 → 8 → 12 …).
- **RED FLAG (not just tokens → don't spend):** span stuck at ~3–4 regardless of
  how many tokens are fed → look at recipe / σ / data, not a bigger rig.

## Where we are on the curve

11000 steps × 16,384 tok/step ≈ **~180M token-exposures** for a 278M model ≈
**~0.65 tokens/param, ~30× undertrained** (Chinchilla-optimal ~20). Incoherence is
expected here; the curve answers whether *more* closes it.

Optional knob: σ→0.1 for a cleaner (less collapse-prone) base to read the curve on.
Next lever after a non-collapsing base: GKD (see docs/ideas.md).

---

# 2026-06-16 — Huginn recipe → MythOuro stability gaps (the validated collapse cure)

Source: Geiping et al. 2025, "Scaling Test-Time Compute with Latent Reasoning: A
Recurrent-Depth Approach" (Huginn / nebel-raven-3.5b). arXiv 2502.05171; code
github.com/seal-rg/recurrent-pretraining; model tomg-group-umd/huginn-0125.
**This is the closest published cousin to MythOuro and it documents our exact
collapse.**

## Huginn's "Bad Run 1" IS our collapse

Quote: "Hidden state collapse. Token correlation quickly reached 1.0 — the model
predicted identical hidden states for all tokens. The recurrence operation
inherently increased token correlation until complete collapse." = our `is is is`
degeneration. They flag it as an INHERENT failure mode of recurrent-depth training
and list the fixes that produced their successful run. We are missing several.

## Stability-gap table (most actionable item on the board)

| Huginn fix | MythOuro status | Action / priority |
|---|---|---|
| **Peak LR 4e-4 → 5e-5** to stop recurrence oscillating into collapse | **We use 3e-4** (their "bad" range) | **Drop to ~5e-5 — cheapest, highest-value test, one flag** |
| **Sandwich norm** (RMSNorm before AND after each sublayer; "required to train recurrence at scale") | **Pre-norm only** (TransformerBlock: attn_norm/ffn_norm, no post-norms) | Add post-norms to recurrent block — architectural, high value |
| **Depth-aware init** σ_out²=1/(5h·l), l = effective recurrent depth | **Blanket N(0,0.02)** (the P0.1 _init_weights) | Adopt depth-scaled output-proj init |
| **Embedding scale γ√h** (prevents early representation collapse) | likely absent — verify | check/add |
| Learned concat adapter [s,e]→h (best at scale vs addition) | LTIInjection (A·h+B·e+trans_out) | different approach; note, not urgent |
| Truncated backprop through last k=8 steps | full-loop grad checkpointing | consider — VRAM + stability |
| s_0 ~ truncated normal σ²=2/5; randomized depth log-normal-Poisson **mean 32** | h init = prelude output; LoopCurriculum/random-depth, max ~4 | note (we're far smaller; depth 4 vs 32) |

## Reframe: the noise knob was our guess; THIS is the validated cure

`recurrent_state_noise` was reverse-engineered from the P0.1 accident and gave only
marginal improvement (1→2-word repeats). **Huginn did NOT use noise** — they used
**normalization + low LR + depth-aware init**. Pivot: test the **LR drop (3e-4→5e-5)
and sandwich norm** before leaning further on the noise hack. The LR change is a
one-line test and is the specific intervention they credit with stopping collapse.

## Token reality-check (settles the capital question)

Huginn: **3.5B params, 800B tokens** (baseline run 180B), 4096 MI250X on Frontier.
**We are at ~180M tokens — ~1000× below even their smallest run.** The closest prior
art needed ~800B tokens for a working recurrent-depth model. This CONFIRMS the
token-starvation diagnosis at the most brutal scale: architecture works; coherence
is firmly token-gated; that is exactly what the rented-compute scale-up buys.

## Next experiment priority (revised by this finding)

1. **LR drop to 5e-5** (one flag) — cheapest collapse test, prior-art-validated.
2. **Sandwich norm** in the recurrent block — the fix they say is "required at scale".
3. Then the token-curve / reverse-KL / on-policy levers on the more-stable base.

---

# 2026-06-16 — collapse_metrics result OVERTURNS the recurrent-collapse hypothesis

Ran tools/collapse_metrics.py on the noise base (checkpoints_noise_test/step_0011000,
state_noise=0.05). Result across all 4 prompts:

| prompt (T tokens) | final-loop token_corr | final-loop eff_rank (max=T) | loop trend |
|---|---|---|---|
| recurrent-depth (T=5)  | 0.138 | 4.73 / 5  | recurrence DIVERSIFIES |
| 2+2 ChatML (T=16)      | 0.169 | 14.55 / 16| recurrence DIVERSIFIES |
| fibonacci (T=6)        | 0.152 | 5.84 / 6  | recurrence DIVERSIFIES |
| Roman Empire (T=15)    | 0.146 | 14.08 / 15| recurrence DIVERSIFIES |

Summary: token_corr=0.151, eff_rank=9.80 → **verdict: healthy.**

## What this means

**MythOuro does NOT have hidden-state / representation collapse.** Reps are
high-rank, decorrelated, and the recurrence *increases* rank / *decreases*
correlation across loops — the OPPOSITE of Huginn's "Bad Run 1" (corr→1). So the
recurrent-collapse framing (Huginn sandwich-norm/depth-aware-init, MeSH,
recurrent_state_noise) targets a failure mode we don't have. Explains why the
noise knob only helped marginally (perturbing healthy states).

## Where the degeneration actually is

Healthy reps + degenerate generation (`is is is`) ⇒ the problem is **downstream**:
**exposure bias** (teacher-forced fine, free-running degenerates) and/or the
**output distribution** (forward-KL → small student puts mass on degenerate
continuations — the MiniLLM mechanism). Two threads, same conclusion.

## Priority reorder (from this data)

- DOWN: Huginn recipe, MeSH, recurrent_state_noise (target hidden-state collapse
  we don't have; keep depth-aware init only as scaling hygiene).
- UP: **mode-seeking divergence (reverse-KL/JSD — Tier-1 already staged, one flag)**
  and **on-policy/GKD** (exposure bias). Cheapest on-target test: `--divergence rev_kl`.

## Caveat + next diagnostic

Measured the teacher-forced PROMPT regime (healthy reps expected there). Confirm by
measuring the metrics DURING autoregressive generation: reps stay healthy + output
degenerate ⇒ output/exposure-bias path; reps degrade as self-fed ⇒ free-running
dynamics. Either way downstream of the recurrence. (Extend collapse_metrics.py.)

---

# 2026-06-16 — generation-time diagnostic: it's EXPOSURE BIAS (free-running degeneration)

Ran collapse_metrics.py --generate (greedy, 32 tok) on the noise base
(step_0011000). The entropy trajectory is the smoking gun.

| prompt | generated | entropy: start -> final | gen rep (corr/rank) |
|---|---|---|---|
| recurrent-depth | `is is is…` | 5.44 -> 0.40 (top_prob 0.95) | 0.965 / 3.71 |
| 2+2 | `\n####…` | start high -> ~1.2 | 0.912 / 5.14 |
| fibonacci | repeats whole `def fibonacci(n):` line | 4.30 -> 0.05 (top_prob 0.99) | 0.198 / 12.23 |
| Roman Empire | `R: R: R:…` | 4.35 -> 0.05 (top_prob 0.99) | 0.531 / 5.39 |

## Diagnosis (confirmed)

Every prompt **starts with high-entropy, healthy distributions** (4–6 nats) and
**spirals into a confident repetition attractor within ~5–7 tokens** (entropy→~0,
top_prob→~0.99). This is **exposure bias / neural text degeneration** (Holtzman
2019), not static collapse. Reps degrade as a *consequence* of emitting repeated
tokens. Confirms: not the recurrence, not hidden-state collapse.

## Metric nuance

Phrase-level loops (fibonacci) keep token-corr LOW (0.20) yet clearly degenerate —
the **entropy trajectory** catches it where rep-corr doesn't. Entropy is the more
robust degeneration signal.

## Confirmed priorities

- **Training fix (real cure):** on-policy / GKD (learn to not spiral) + reverse-KL /
  mode-seeking + unlikelihood (anti-repetition on the output distribution). All
  target exposure bias directly.
- **Cheap inference band-aid:** early entropy is HIGH → greedy locks the spiral →
  sampling (temp/top-p) + repetition penalty taps the available diversity (validates
  the temperature-diversity intuition). Mitigation, not cure.
- **Underneath:** still token-starved — exposure bias is what undertrained small
  models do; more tokens + on-policy is the deep fix.
- OFF-TARGET (confirmed): Huginn recipe / MeSH / recurrent_state_noise.

---

# 2026-06-16 — temperature test: sampling does NOT escape the spiral (v4 theory refined)

collapse_metrics.py --generate --temperature 0.8 --top-k 40 on step_0011000:
still degenerates (`which which`, `29999`, `getget`, `44455555`); entropy still
collapses to ~0.04-0.5, top_prob -> 0.93-0.99. Sampling delayed the spiral by ~1
token, no more.

## Conclusion
- "Greedy was the problem" = REFUTED. Once the distribution collapses to ~0.99 on
  the repeat, output-level sampling (temperature/top-k) can't escape.
- **v4 theory refined:** v4's noise was *representation-level* (perturbs the hidden
  state each loop → changes the distribution itself), which is strictly stronger
  than *output-level* temperature for escaping a collapsed distribution. That's the
  likely reason v4 stayed varied where sampling fails.
- **Implication:** the repetition attractor is baked into the LEARNED DISTRIBUTION
  → strengthens the case for on-policy/GKD training (decode tricks won't fix it).
- Confirmation test (no code change): run collapse_metrics --generate on the real
  v4 checkpoint (archived_models/mythouro_distill_small_v4/step_0003000.pt). If v4
  doesn't spiral, its representation noise is the cause (data already ruled out).
- Optional isolation: add an inference-time recurrent_state_noise path and test on
  the fixed model.

---

# 2026-06-16 — v4 mystery SOLVED: representation noise prevents the spiral

Ran collapse_metrics.py --generate (greedy) on the real v4 checkpoint
(archived_models/mythouro_distill_small_v4/step_0003000.pt). Decisive vs the fixed
model under IDENTICAL greedy decoding:

| | mean entropy | gen rep rank | gen corr | output |
|---|---|---|---|---|
| v4 (P0.1 noise in weights) | 3.8-4.3 nats | 23-26 (~max) | 0.25-0.31 | varied, grammatical |
| fixed step_11000 | 0.4-1.8 | 3.7-9 | 0.86-0.97 | `is is is` spiral |

## Conclusion (airtight)
v4 does NOT spiral: holds high entropy, healthy near-full-rank generated reps,
varied output — same decoder, same scale. Only difference: v4's **representation
noise** (P0.1 random o_proj, baked into weights; cfg state_noise=0.0). This is the
cause of v4's "better results." Refined theory CONFIRMED: representation-level noise
escapes the exposure-bias repetition attractor where output-level temperature cannot.

Causal chain: reps healthy → fixed model free-gen spirals (exposure bias) → v4's
rep-noise jitters the distribution each step → escapes the attractor.

## Caveats
- v4 is varied but NOT coherent (register-salad); one prompt still drifted into a
  late phrase loop. Noise resists, doesn't cure; v4 always undertrained.
- So the "regression" vs v4 = removal of accidental rep-noise. Neither is coherent.

## Cures (ranked)
1. Real: on-policy/GKD (retrain the distribution to not spiral) + more tokens.
2. Band-aid (v4-proven): representation noise at INFERENCE (recurrent_state_noise is
   currently training-only → add an inference path). Gives v4-like variety on demand
   from the fixed model. Optional isolation test: confirms noise mechanism vs v4's
   data/SFT confounds.
3. Weak: output-level sampling / repetition penalty (can't escape a collapsed dist).

---

# 2026-06-16 — inference-noise band-aid FAILS → v4 needed train-time co-adaptation

collapse_metrics --generate --inference-noise 0.05 on step_0011000: near-identical
to no-noise greedy (still `is is is` / repeated `def fibonacci(n):` / `A: A:`,
entropy still collapses). The post-hoc inference-noise band-aid does NOTHING.

## Why (the deeper, final v4 insight)
The repetition attractor is a **learned, SHARP** feature of the fixed model's
distribution (0.99 confidence). A 5%-RMS perturbation can't move a 0.99 logit off
its token; escaping needs huge noise → which yields garbage, not v4's grammatical
variety. So v4's secret was **training with always-on noise** → distribution stayed
diffuse → never learned the sharp attractor. You CANNOT bolt v4's behavior onto a
model that already learned the attractor.

## Consequence (path forward, reinforced)
- No decode-time trick (sampling OR inference noise) escapes a learned sharp
  attractor. Ruled out with evidence.
- The fix MUST be at training time. On-policy/GKD is the principled cure (retrains
  the distribution so the model learns to recover instead of spiraling). v4-style
  always-on-noise training would also avoid the attractor but only yields salad.
- Inference-noise path left in (main.py RecurrentBlock.inference_noise, off by
  default) as a diagnostic knob; not a usable fix.

## v4 story — COMPLETE
reps healthy → fixed model learns a sharp repetition attractor (exposure bias) →
spirals under any decoding → v4 avoided learning it via always-on training noise
(diffuse dist) → neither is coherent (undertrained). Cure = on-policy/GKD + tokens.

---

# 2026-06-17 — Ouro pipeline note: sandwich-norm convergence + teacher scale

From Ouro's model page (the teacher): "standard decoder-only Transformer with RoPE,
SwiGLU, and **sandwich normalization for enhanced training stability with deep
recurrent computation**." Training: **7.7T tokens** (3T + 3T phases, 1.4T CoT
annealing, 20B LongCT, 300B mid-training, reasoning SFT).

## Sandwich norm — refine the earlier demotion
Two independent recurrent-depth models now converge on sandwich norm for recurrent
stability: **Ouro** (teacher, total_ut_steps=4) and **Huginn** ("required to train
recurrence at scale"). **MythOuro is the pre-norm outlier.**
- **Current bug (exposure-bias spiral): sandwich norm is still NOT the fix** —
  collapse_metrics shows reps are healthy at our tiny scale; cure stays on-policy/
  reverse-KL. Demotion holds for *now*.
- **Scale-up: RE-ELEVATE.** No instability at ~180M tokens ≠ none at billions; the
  teacher (7.7T) and Huginn (800B) both needed it at scale. So the **scale-up fresh
  distill should include `--use-sandwich-norm` (+ `--use-depth-aware-init`) from the
  start** — match the architectures that work at scale. (Already staged; this sets
  the *timing*: scale-up, not the current small runs.)

## Teacher scale (7.7T tokens)
Rich distillation signal (good for token-efficiency — we inherit some of it cheaply),
but reinforces the gap: teacher 7.7T vs student ~180M tokens. Distillation narrows it;
the token-curve scale-up still closes it.

---

# 2026-06-18 — fresh reverse-KL run, PRELIMINARY read @ step 1500 (mechanism confirmed)

Fresh distill from random init with `--divergence rev_kl` (mythouro_distill_tiny, the
"teach it right from the start" run). Inspected step 1500 (~25M tokens — early).
Raw: reports/collapse_freshrevkl_1500.txt + _1500_T08.txt.

## Greedy: still repeats, BUT distribution diffuse on 2/4 (unlike fwd-KL)
| prompt | final entropy | final top_prob |
|---|---|---|
| recurrent-depth | 2.38 | 0.43 |
| Roman | **7.38** | **0.08** |
| 2+2 | 0.37 | 0.97 (still sharp-collapse) |
| fibonacci | 0.19 | 0.98 (still sharp-collapse) |

vs forward-KL baseline which collapsed ALL prompts to entropy ~0.1 / top_prob ~0.95–0.99.
So reverse-KL keeps the distribution **diffuse** (shallow attractor) on the diffuse prompts.

## T=0.8 sampling: CONFIRMS the mechanism
- **Roman (most diffuse) → sampling ESCAPED the spiral → varied output** ("next true wrong
  possible basic optimal easy legal proper actual…"). Forward-KL+sampling could NEVER do this
  (0.99 spike left nothing to sample). **Reverse-KL → diffuse → sampling-recoverable: CONFIRMED.**
- recurrent-depth: partial (few varied tokens then repeats).
- 2+2, fibonacci: still collapse even sampled (distributions still sharp at 1500).

## Honest verdict
- ✅ **Mechanism confirmed:** reverse-KL reshapes toward diffuse/recoverable — a real
  qualitative difference from fwd-KL's hard collapse. The signal we wanted.
- ⚠️ **Preliminary/uneven:** only ~1–2 of 4 prompts so far; still word-salad (expected at 25M
  tokens); only 1500 steps. Prompt reps corr 0.5–0.85 = still undertrained (will sharpen).
- **Verdict pending the ~3k read:** does diffuse/recoverable **spread to all 4 prompts + deepen**
  (longer varied span, less repetition) as tokens grow? That calls Tier-1-sufficient vs need-Tier-2.
- Leaning **on-track** (right direction, mechanism confirmed) — NOT yet a "Tier-1 wins" call.

---

# 2026-06-20 — fresh reverse-KL run: REGRESSION at 5500 → pure rev-KL mode-collapses

**Negative result that overturns the 3k optimism.** Tracked the run across three checkpoints
with the categorised probe (`collapse_metrics --probe-set all`, greedy + T=0.8). Full reports:
`reports/collapse_freshrevkl_{3000,5500}_full*.txt`.

## The trajectory (escape rate under T=0.8 sampling, "not clearly degenerate")
| step | ~tokens | greedy character | T=0.8 escape | call |
|---|---|---|---|---|
| 1500 | ~25M | diffuse on 2/4 | 1/4 | transient diffuse |
| 3000 | ~50M | **domain-aware** repeats (code→`def`/`self`, math→`x`/`r`, prose words) | **~12/19** | looked encouraging |
| 5500 | ~90M | **newline/digit collapse** (`\n\n\n…`, `12`, `222`) across ALL categories | **0/19** | **regressed** |

3000→5500: domain-aware structure → degenerate newline/digit attractor; generated rep-corr
back up to 0.7–0.99 (escapers were 0.31–0.60 at 3000). More tokens made it **worse**.

## Diagnosis: pure reverse-KL mode collapse
Reverse-KL is **mode-seeking** → with continued training it over-concentrates onto the
unconditionally-dominant mode (newline/digits, ubiquitous in code/math/web). The 3000 diffuse
phase was the model passing *through*; continued training drove it into the dominant-mode
attractor. **This is the documented failure of *pure* reverse-KL** — exactly why GKD uses **JSD**
(interpolated fwd/rev) and MiniLLM adds **teacher-mixed sampling** (see ideas.md deep-dives).
The 3000 "Tier-1 working, keep tokens" call was **premature** — two improving points (1500→3000)
were a transient, not a trend.

## Consequence / next
- **Pure `--divergence rev_kl` is NOT sufficient — it mode-collapses by ~90M tokens.** Cheap,
  valuable negative result (found for ~$0, before any rented compute).
- **Next test: JSD** — `--divergence jsd --jsd-beta 0.5`, fresh from random init (attractor
  entrenches; can't un-teach a collapsed ckpt — cf. revkl-continue @10k). Balances mode-covering
  + mode-seeking to avoid the pure-mode-seeking collapse.
- If JSD insufficient → full **Tier-2** (teacher-mixed sampling α≈0.2 + on-policy), the MiniLLM/GKD
  recipe built for exactly this.
- The 8000 checkpoint from this run is not worth reading (collapse will deepen, not recover).

---

# 2026-06-21 — fresh JSD run: REPRESENTATION COLLAPSE (rank→1) at 4000

JSD (`--divergence jsd --jsd-beta 0.5`, fresh from random init, `checkpoints_freshjsd`)
to test whether balancing fwd+rev divergence avoids pure rev-KL's mode collapse. Read
@ step 4000 (~65M tokens), `--probe-set all`, on the free 4060. Reports:
`reports/collapse_freshjsd_4000_full*.txt`.

## Result: worse than rev-KL — and a NEW failure mode
- **Greedy:** all 19 prompts → `..............` (just periods).
- **T=0.8:** random punctuation/digit soup (flat dist; entropy ~7–8, top_prob ~0.05–0.12).
- **Decisive:** token-corr ≈ **0.99–1.00**, effective rank ≈ **1.0–1.6** on ALL prompts
  (prompt *and* generated) → genuine **hidden-state / representation collapse** (every
  token → ~one vector). Earlier (65M) than rev-KL's collapse (90M), and at the
  *representation* level, not just the output.

## Why it matters: recurrent collapse FINALLY appeared
fwd-KL and rev-KL produced *output* collapse with **healthy reps** (rank 5–19) — hence
the 06-16 diagnosis "exposure bias, NOT hidden-state collapse," and the demotion of the
Huginn/MeSH stability fixes. **That held for fwd/rev-KL — but JSD breaks it:** it induced
the rank→1 recurrent collapse those fixes target. Refined diagnosis: *both output-collapse
(fwd/rev-KL) and representation-collapse (JSD) are reachable; the divergence selects which.*

## Divergence sweep
| objective | failure | reps |
|---|---|---|
| forward-KL | output collapse (hard repetition) | healthy (rank 5–19) |
| reverse-KL | output collapse (escaped to 12/19 @50M, collapsed @90M) | healthy |
| JSD | **representation collapse (rank→1) @65M** | **collapsed** |

→ No divergence alone is the fix; rev-KL is least-bad; **JSD is worst.**

## Mechanism (from the training log) — optimization instability, NOT the objective per se
The loss log reveals *how* it collapsed:
- **Healthy early** (~step 10–700): loss 6.17→2.6, hard CE 10.9→4.3, **gnorm calm ~0.8–8**.
- **Fast fit** (~700–800): loss ~1.4, hard CE ~2.
- **Destabilizes from ~step 1100**: loss bounces (1.3↔2.7) and **gnorm explodes** — 15.9
  (1280), 26.6 (1380), 29.8 (1510), 30.1 (1620) vs ~1–8 early. MoE balance degrades in
  step (cv 0.37→0.89; router bias‖·‖₂ 0.15→1.06).
- **ρ(A) stays healthy** (0.35→0.32, ≪1 throughout) → the **recurrence is NOT exploding**;
  the instability is in the **gradients/optimization**, not the LTI fixed point.
- **soft (JSD) term plateaus** (~0.85–0.95) while hard CE plummets → model fit the data,
  the optimization tore itself up doing it, reps collapsed (rank→1) by 4000.

→ The rank→1 collapse is the **endpoint of an optimization that went unstable (gnorm
explosion)**, not a mysterious objective failure. **LR 3e-4 is too hot for this config.**

## Pivot (next experiment) — evidence-backed recipe
Target the observed instability directly: **rev-KL, fresh from random init, plus**
- **Lower LR** 3e-4 → **1e-4–1.5e-4** (the gnorm-30 spikes demand it).
- **Tight gradient clipping** (~1.0) — cap the spikes so they can't wreck the reps.
- **`--use-sandwich-norm --use-depth-aware-init`** — architectural conditioning (Huginn).

Likely helps **both** failure modes (rev-KL output collapse + JSD rep collapse) if both
were driven by the same hot-LR instability. Optionally read an earlier JSD ckpt (1000/1500)
to localize collapse onset. Caveat: one *probe* data point, but the **training log
corroborates** (gnorm explosion) → high confidence this is optimization instability, not a
step-1 bug (JSD-impl interaction still not fully ruled out).

---

# 2026-06-21 — JSD COMPLETE POST-MORTEM (full log to step 4056): n_loops 2→3 was the knockout

The 4000-probe entry above was written before the full training log was reviewed. The log to
the stop point (step 4056, clean SIGINT) completes the failure story and **refines the
mechanism**. Raw: freshjsd console log (lr 3e-4, **no** stability flags, `checkpoints_freshjsd`,
278,871,411 params).

## Full gnorm trajectory — four phases

| phase | steps | gnorm | n_loops | state |
|---|---|---|---|---|
| calm | ≤1000 | 0.8–8 | 2 | healthy, fitting fast (loss→1.4 by step 750) |
| chronic instability | 1000–3000 | 15–55 (routinely) | 2 | never settles; loss bounces 1.3↔2.7; cv/bias climbing |
| **true explosion** | 3100–3400 | **214 (3160), 336 (3380), 1593 (3390)** | 2 | runaway; loss *rising* (→4.1 by 3500) |
| **catastrophe** | 3510 (`n_loops 2→3`) | **4271 → 1,000,446 (3750)** | 3 | knockout; `unc` breaks to 4.28; `lb`→5+; cv→2.18 |
| wreckage | 3510–4056 | clip-bounces 1–8000 | 3 | loss stuck **4.5–4.9** (was 1.5); never recovers |

## The decisive refinement: instability is at n_loops=2; loop-3 is the amplifier, NOT the cause
- The runaway gnorm (214→1593) at steps 3100–3400 happened **entirely at n_loops 2**, *before* the
  curriculum added the 3rd loop at 3510. → **loop-depth is not the root cause** (rules out the
  "3rd-loop-triggered" hypothesis; the loop curriculum here lands at ~3500, not ~1500).
- BUT the `n_loops 2→3` transition at 3510 was the **catastrophic finishing blow** on an
  already-diverging model: gnorm 173 (3500, n=2) → **4271** (3510, n=3) → **1,000,446** (3750).
  Adding loop-depth to an already-unstable optimization = detonation.
- Net: **hot-LR optimization instability is the disease; the loop-depth increase is what turned a
  bad run into an unrecoverable one.** (User caught this from the loop-timing — confirmed by the log.)

## End state (completes the 4000 probe's rank→1 finding)
- **Permanent loss regression:** ~1.5 (step 2000) → **~4.5–4.9** (3500+), hard CE 2→7–11. Destroyed,
  not merely degraded — consistent with the probe's rank→1 representation collapse.
- **Expert collapse:** cv→**2.18** (step 3600), router bias‖·‖₂ 0.15→3.0+ — routing concentrated onto
  ~one expert (the MoE face of the rank→1 collapse).
- **ρ(A) stayed healthy** (0.30–0.34) to the very end → the recurrence never exploded; the LTI fixed
  point was fine throughout. The blowup was **100% gradients/optimization.**

## Confirmed conclusion
**LR 3e-4 is too hot for this config**, full stop. Textbook optimization divergence (chronic high
gnorm → runaway → detonation at the depth-curriculum step), not an objective-specific or
recurrence-specific pathology. Direct motivation for the stability run below.

---

# 2026-06-21 — rev-KL STABILITY run, IN PROGRESS (the evidence-backed fix)

Fresh from random init, the recipe the post-mortem points to: **rev-KL + lr 1e-4 +
`--use-sandwich-norm --use-depth-aware-init`**, grad-clip 1.0, `checkpoints_revkl_stable`,
**278,884,211 params** (sandwich-norm adds ~13k vs JSD's 278,871,411). Paused at step 585 (clean
SIGINT save → `step_0000585.pt`); resuming overnight from the same command (auto-resume).

## Early read (to step 585) — calm, but NOT yet past the danger zone
- **gnorm:** high at init (18–20, steps 10–50 — the depth-aware-init signature under near-zero-LR
  warmup; harmless big-grad-tiny-step), **settles to ~1–2.3 by step 60 and holds there through 585**
  at peak LR (1e-4 reached ~step 510).
- **Honest caveat: at matched steps, JSD was *also* calm here** (JSD gnorm ~1.5 at step 580). Step 585
  is *before* the zone that discriminates → **no verdict yet.**
- **Distillation progressing:** rev-KL soft term **decreasing 6.5→3.7** (student converging toward
  teacher) — contrast JSD's soft *plateau* (~0.9). MoE cv healthier than JSD at matched steps
  (0.32–0.78 vs JSD's later 0.89+). ρ(A) 0.36.
- **Slower-but-stable tradeoff (expected):** loss higher than JSD at matched steps (lower LR learns
  slower); the 12k-step budget absorbs it. Stable-and-slower > fast-and-collapses.

## The verdict zones (watch the gnorm column on resume)
| zone | JSD did (lr 3e-4) | "passing" |
|---|---|---|
| 1000–1500 | crept to 15–40 | stays ~1–3 → first real evidence |
| 3100–3400 | **exploded 100–1600** (still n_loops 2) | stays calm → strong |
| **~3510 (`n_loops 2→3`)** | **4271 → 1,000,000, model destroyed** | single/double-digit gnorm → **the win** |

If gnorm clears the `n_loops 2→3` transition without detonating, the hot-LR diagnosis is confirmed
and the fix holds. Then `--probe-set all` read at the next checkpoint past ~3600 to confirm the model
is **diffuse/healthy**, not stable-but-degenerate.

## Teacher note (settled 2026-06-21)
Ouro-2.6B-Thinking is **the only trained RDT with open weights** (others found are untrained
architecture releases / Ouro forks). The student tokenizer was *built to match* it for clean
logit distillation. So the teacher is a **fixed constant**, not a variable — and it's not the
instability (distillation consumes only the teacher's output logits; the blowup is student-side, ρ(A)
healthy). Ceiling leverage lives on the **student side** (tokens, on-policy/GKD, feature distillation
— the RDT→RDT match enables the latter). A teacher swap, if ever needed, goes via a **tokenizer
converter** (cross-tokenizer/ULD alignment), not a re-foundation.

---

# 2026-06-23 — rev-KL stability EVAL @3216: stability won, but formal stats favor fwd-KL (the calibration tax)

First eval-harness run on the stability checkpoint (`step_0003216.pt`, ~53M tokens, **pre**-`n_loops
2→3` transition; `eval/harness.py`, n=500):

| metric | rev-KL-stable @3216 | moe_s0 (fwd-KL) | continuation | read |
|---|---|---|---|---|
| **PPL** | **8.21** | 5.72 | 3.06 | mid-pack |
| **loop_eff** | 0.483 (depth 1.93/4) | 0.500 | 0.500 | healthy |
| **ECE** | **0.1997** | 0.015 | 0.0052 | **~10–40× worse** |
| **ARC** | 0.234 | — | — | ≈ random (0.25) |
| GSM8K | (pending; ~0 expected) | — | — | no math at this scale |

## Two findings
1. **PPL 8.21 mid-pack — but NOT comparable to the fwd-KL runs.** rev-KL is mode-*seeking* → trades
   mode-covering PPL for concentration on teacher modes; 8.21 vs moe_s0's 5.72 is the **expected
   objective tradeoff, not a regression.** (Also confirms the earlier caution: eval 8.21 ≫ the
   training-stream ~2 — low training loss never meant "best version.")
2. **ECE 0.1997 is the real flag — a calibration regression.** Every prior run was 0.005–0.04; this
   is badly **overconfident**, consistent with rev-KL mode-seeking (sharpens onto modes → poor
   calibration). **Genuine tension with the honest-specialist / medical thesis**, which depends on a
   *well-calibrated* uncertainty head: the recipe that trains stably + escapes repetition (rev-KL) is
   at odds with the recipe that calibrates well (fwd-KL). Caveat: mid-training, pre-transition,
   n_loops=4-on-n_loops=2 eval → watch whether it improves; but 0.20 is a flag, not noise.

## Honest verdict
**Stability = solved; formal stats = fwd-KL still wins (PPL *and* ECE).** rev-KL-stable's edge is
**stability + generation trajectory + diffuse-not-collapsed**, NOT the formal metrics — the eval
crowns moe_s0/continuation on the numbers. Different axes; PPL/ECE measure what fwd-KL is good at.

## Next lever — the "cover all areas" (hybrid) question
The fwd/rev hybrid **already exists**: `--divergence jsd --jsd-beta` (Jensen-Shannon interpolates
fwd+rev). We tried JSD once and it rank→1 collapsed — **but that was LR-instability (gnorm explosion
at 3e-4), NOT JSD** (see 06-21 post-mortem). So **stable-JSD is the untested, well-motivated
experiment:** `--divergence jsd --jsd-beta 0.5 --lr 1e-4 --use-sandwich-norm --use-depth-aware-init`
fresh — JSD on the same stable footing that just tamed rev-KL. Could give fwd calibration/PPL + rev
diffuseness; `jsd-beta` dials the balance. **Honest caveat:** interpolations can *compromise*
(mediocre-at-both) rather than *combine* (great-at-both) — no guarantee.

**Calibration specifically may be better fixed by targeted levers than the divergence:** (a) bump
**`--unc-coeff`** (the existing `uncertainty_calibration_loss` directly pressures ECE); (b) post-hoc
**temperature scaling** at inference (cheap, standard ECE fix); (c) **on-policy** (helps exposure-bias
*and* calibration). So "cover all areas" is likely a **stack of per-axis tools** (stable-JSD or rev-KL
for stability/diffuseness + unc-coeff / temp-scaling for calibration), not one magic divergence.

---

# 2026-06-24 — rev-KL stability DEPTH-MATCHED VERDICT @6675 (~109M tokens): stability won, pure rev-KL collapses, calibration-tension was an artifact

Ran the full curriculum to **step 6675 (fully n_loops=4-trained)** — survived **both** loop transitions
(2→3 *and* 3→4), gnorm calm throughout (the healthiest trace in the project; transient 5-6 spikes that
recovered, ρ(A) stayed 0.30–0.33). Then the **depth-matched** probe + eval (n_loops=4 trained *and*
inferenced — no more pessimism caveat). Reports: `reports/collapse_revkl_stable_6675_full*.txt`,
`checkpoints_revkl_stable/eval_step_6675.json`.

## Eval — every formal metric excellent
| metric | @6675 (depth-matched) | @3216 (mismatched) | lineage best |
|---|---|---|---|
| **PPL** | **1.759** (best ever) | 8.21 | continuation 3.06 |
| **loop_eff** | 0.500 (avg_depth 2.0/4) | 0.483 | 0.500 (all) |
| **ECE** | **0.0152** | 0.1997 | 0.0052–0.04 |

## Two findings — one a CORRECTION
1. **CALIBRATION-TENSION RETRACTED — the ECE 0.20 was a depth-mismatch ARTIFACT.** Depth-matched ECE is
   **0.0152**, right in the well-calibrated band. The 06-21/06-23 "rev-KL trades calibration / tension
   with the honest-specialist thesis" framing was **largely spurious** (driven by the n_loops=4-on-
   n_loops=2/3 read). **The honest-specialist thesis is intact**; rev-KL does NOT ruin calibration.
   (The 06-23 entry's caveat — "watch whether it improves; depth-mismatch could distort it" — is now
   confirmed: it was the artifact.) Roadmap differentiator-#1 + ideas.md ECE note corrected accordingly.
2. **Generation: HARD MODE-COLLAPSE.** Greedy ~3/19, T=0.8 ~2/19 "not degenerate" — dominated by sharp
   `is is is` / `# # #` / `A A A A` (top_prob 0.9–0.99, sampling can't escape). **The step_4000 "varied
   salad" (11–15/19) was untrained-4th-loop NOISE, confirmed** — once the 4th loop is *trained*, the
   model converges to a sharp repetition attractor. More training at the right depth made generation
   *worse*. Reps stayed healthy (rank 4.6–21, **not** rank→1) → exposure-bias *output* collapse, not
   representation collapse.

## The key result (bigger than this checkpoint)
**Stable rev-KL @109M reached the SAME mode-collapse the hot-LR rev-KL hit @90M** (06-20). The
stability recipe fixed the **optimization** (gnorm/rank→1); it did **not** fix the rev-KL **divergence**
problem. **Pure rev-KL is insufficient — stable or not — it mode-collapses with tokens.**

And the cleanest statement of the project's core finding: **best-ever PPL (1.759) + good calibration
(0.0152) + loop_eff 0.500 + stability solved + reps healthy → and free generation still hard-collapses.**
Exposure bias is **decoupled from every formal metric** (loss, calibration, stability, representation
health). Low PPL and the collapse are two faces of one thing (tight teacher-forced fit → sharp →
free-run collapse). **The cure is on-policy — not any formal-metric or divergence lever.**

## Verdict + next
- ✅ **Stability recipe = keeper** (lr 1e-4 + sandwich-norm + depth-aware-init): use it for every future run.
- ✅ **Calibration fine** (tension was spurious).
- ❌ **Pure rev-KL = insufficient** (collapses with tokens).
- **Next:** (a) **stable-JSD** — cheap test on the proven stable footing; the hybrid *might* resist the
  pure-rev-KL collapse, but (b) the *deep* cure is **on-policy/GKD** — rev-KL collapsing here is strong
  evidence no offline divergence alone reaches coherence; the student must train on its own rollouts
  under teacher correction. Stable-JSD informs; on-policy is the destination.

---

## 2026-06-27 — ✅ ON-POLICY VALIDATED (partial): first movement on the generation blocker

First on-policy/GKD run. Warm-started from rev-KL-stable **6675**, `--onpolicy-lambda 0.5
--teacher-mix-alpha 0.6 --rollout-len 64`, cross-GPU (student cuda:0 / teacher cuda:2),
seq 1024. Ran **6675 → 6771 (~96 steps, ~9.3 h, ~5.8 min/step, ~50–100 tok/s)**. Stable
throughout (gnorm 0.37–1.33, mostly <1 — on-policy didn't destabilise), on-policy fired
(`op` 6–11/16 ≈ λ=0.5), MoE balanced (cv 0.109). Loss noisy ~1.0 — a non-signal, as always.

**The result is in the probe, not the metrics** (coherence is decoupled from every formal
metric — the project's core lesson). α=0.0 rows (pure student, no teacher-mix = the real
success metric), baseline 6675 → 6771:

| Seed | 6675 α=0.0 | 6771 α=0.0 | |
|---|---|---|---|
| Weather (prose) | top_share 0.45, distinct1 0.15 (`this this was was`) | **0.14 / 0.66** (varied sentences) | **un-collapsed** |
| Bacterial (medical) | 0.89 / 0.06 (`the the the`) | 0.90 / 0.09 | no movement |
| fibonacci (code) | 0.27 / 0.12 (number spam) | 0.14 / 0.16 | marginal |

**First time ANY lever has moved the unaided (α=0.0) generation metric** the project has been
blocked on. The prose shift (top_share −0.31, distinct1 ×4.4; stuck attractor → real
sentences) is large and real — **on-policy as the exposure-bias cure is empirically validated**,
not just theorised. (`op` confirms it's the on-policy steps doing it, not the offline half.)

**Uneven = dose, not mechanism:** corpus is general web/math/code, so prose got the most
on-policy reps and un-collapsed first; medical/code are sparser → lag. Same medicine, the
under-represented domains need more of it. (Medical is the mission seed *and* the laggard —
expected from dose; medical capability is a later SFT/retrieval stage regardless.)

**Verdict:** the question flipped from *"does on-policy work?"* (✅ yes) to *"how much dose to
propagate past prose?"* — now a **throughput/dose problem**. At 5.8 min/step the dose is slow;
strongest justification yet for the Max 1100 (48 GB → **batched rollouts** → far more on-policy
tokens/night; the decode is latency-bound, so the win is batching+`torch.compile`, not raw
BF16 TFLOPS — see hardware_options.md). **Next:** continue from 6771, **bump λ→0.7** (gnorm had
headroom), more steps; re-probe and watch medical/code follow prose.

**UPDATE 2026-06-28 (step 6906, ~231 on-policy steps):** ✅✅ **collapse broken DOMAIN-WIDE.**
6-seed probe → α=0.0 `top_share` low on *every* seed (0.06–0.31); no hard attractor anywhere.
Regime flipped from exposure-bias collapse → "varied but incoherent" (normal undertrained small
model) = now **tokens/scale-bound**, not blocker-bound. Capability present at α≥0.5 (diabetes →
correct symptoms; fibonacci → real code). The earlier "medical still collapsed" read was partly
single-sample RNG noise (probe now multi-samples). **Next = pour tokens on the un-collapsed base
→ the Max 1100 throughput.** Full read: generation_probe_tracker.md 2026-06-28.


<!-- ===== moved from docs/roadmap.md (2026-06-27 doc reorg) ===== -->

## External eval baselines (context for the numbers)

Our metrics in isolation don't say whether they're good. Rough anchors for
small models at comparable scale (held-out web-text perplexity; exact numbers
vary by tokenizer/corpus, so treat as order-of-magnitude):

| Model | Params | Train tokens | Ballpark PPL | Note |
|-------|-------:|-------------:|-------------:|------|
| **MythOuro v1 (distill)** | 278M | ~20M | **37** | Ours — but distilled, so PPL is teacher-shaped, not from-scratch |
| GPT-2 small | 124M | ~40B | ~30–35 | ~2000× more tokens than ours |
| Pythia-410M | 410M | ~300B | ~12–15 | ~15000× more tokens |
| GPT-2 medium | 355M | ~40B | ~22–26 | — |

**The honest takeaway:** our PPL ~37 is *reasonable for the token budget* (we
trained on ~20M tokens vs. tens of billions for the others — distillation is why
it's even comparable). But coherent generation empirically needs both more
params (~1B+) and more tokens (~10B+) than the workstation can reach. The gap to
"usable" is **scale**, and these baselines quantify roughly how far: ~3× the
params and ~500× the tokens to reach Pythia-410M territory.

---

