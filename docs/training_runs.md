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
