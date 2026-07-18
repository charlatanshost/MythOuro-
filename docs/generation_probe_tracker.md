# Generation-probe tracker (cross-comparison)

**Purpose.** A longitudinal cross-comparison of the generation-time collapse
diagnostics (`tools/collapse_metrics.py --probe-set all`) across checkpoints and
prompt categories, cross-referenced against the roadmap's hypotheses. This is the
"*is it learning, where, and how fast*" scoreboard.

Distinct from the neighbouring docs:
- `docs/training_runs.md` — per-run **eval stats** (PPL / ECE / loop_eff) + recipe, chronological.
- `docs/roadmap.md` — **strategy / plan** and the standing hypotheses.
- **This file** — the **generation-quality** picture per category over checkpoints, and whether it confirms or contradicts the roadmap claims.

Update after each probe run on a new checkpoint.

---

## Protocol

- Tool: `python tools/collapse_metrics.py -c <ckpt> --device cuda:0 --generate --probe-set all` (greedy), then again with `--temperature 0.8 --top-k 40` (sampled).
- **Categories and what each tests** (the distill mix is FineWeb-Edu 40% / open-web-math 40% / codeparrot 20%, `_MIX_RATIOS`):
  - `prose` / `code` / `math` — **in-format** for the distill corpus (fair test of "is it learning the data it sees").
  - `chat` (ChatML) / `qa` (`Q:/A:`) — **OOD formats** until SFT introduces them; expected to lag.
- **Greedy read** = argmax / attractor depth + whether the locked token is *domain-appropriate*.
- **T=0.8 read** = the **escape test**: does sampling break the spiral? The tool flags each prompt `inconclusive / not clearly degenerate` (= **escaped**) or `FREE-RUNNING rep degradation` (= **locked**). Escape-under-sampling is the **Tier-1 (reverse-KL) success signal** — forward-KL collapses so hard there is nothing left to sample.

---

## Cross-comparison — escape rate under T=0.8 sampling (escaped / total)

| checkpoint | ~tokens | prose | code | math | chat | qa | overall | raw report |
|---|---|---|---|---|---|---|---|---|
| step_1500 | ~25M | — | — | — | — | 1/4\* | 1/4 (orig 4-set) | `reports/collapse_freshrevkl_1500*.txt` |
| step_3000 | ~50M | 3/4 | 3/4 | 1/4 | 3/3 | 2/4 | **~12/19** | `reports/collapse_freshrevkl_3000_full*.txt` |
| step_5500 | ~90M | 0/4 | 0/4 | 0/4 | 0/3 | 0/4 | **0/19** ⬇ | `reports/collapse_freshrevkl_5500_full*.txt` |
| **JSD** step_4000 | ~65M | 0/4 | 0/4 | 0/4 | 0/3 | 0/4 | **0/19 + rank→1** ❌❌ | `reports/collapse_freshjsd_4000_full*.txt` |
| **rev-KL-STABLE** step_3216 | ~53M | 0/4 | 0/4 | 0/4 | 0/3 | 0/4 | **0/19 strict — but ✅ healthy reps + domain lock-ons + improving (NOT collapse)** | console 2026-06-23 (save to `reports/`) |
| **rev-KL-STABLE** step_6675 **(DEPTH-MATCHED)** | ~109M | 0/4 | 1/4 | 0/4 | 0/3 | 1/4 | **~2/19 — MODE-COLLAPSED** (`is is is`, sharp; sampling can't escape) ❌ | `reports/collapse_revkl_stable_6675_full*.txt` |

> **DEPTH-MATCHED VERDICT (2026-06-24): pure rev-KL MODE-COLLAPSES; the step_4000 "varied salad" was an
> ARTIFACT.** At full n_loops=4 training (109M tok), greedy ~3/19 and T=0.8 ~2/19 — hard `is is is` with
> sharp distributions. The step_4000 11–15/19 was **untrained-4th-loop noise**, not capability (confirmed
> by training the 4th loop → generation got *worse*). This is the **same collapse the hot-LR rev-KL hit
> @90M** → the stability recipe fixed *optimization*, not the rev-KL *divergence* problem. Reps stayed
> healthy (rank 4.6–21, NOT rank→1) = exposure-bias *output* collapse. Eval @6675: **PPL 1.759 (best ever),
> loop_eff 0.500, ECE 0.0152** (the 3216 ECE 0.20 was a depth artifact — calibration is FINE). **Best-ever
> formal metrics + collapsed generation = exposure bias decoupled from all metrics → cure is on-policy.**
> Next: stable-JSD (cheap), then on-policy/GKD (the real fix). See training_runs.md 06-24.

> **⚠️ The three "0/19" rows mean OPPOSITE things — do not read them as equivalent:**
> `freshrevkl@5500` = 0/19 because it **regressed into mode-collapse** (got worse). `JSD@4000` =
> 0/19 because of **rank→1 representation collapse** (catastrophic). `rev-KL-stable@3216` = 0/19
> only by the strict escape flag, but reps are **healthy (rank 2–13)**, it shows **domain-appropriate
> lock-ons + sentence-like fragments**, and it is **still *improving* with tokens, not collapsing** —
> the *recoverable* exposure-bias regime, the best 0/19 we've had. Escape-rate alone is misleading
> here; see the result note below.

\* step_1500 used the original 4-prompt set (recurrent-depth / 2+2 / fibonacci / Roman), not the categorised set, so only partially comparable. At 1500 only Roman escaped; 2+2 and fibonacci were hard-locked. By 3000 those two freed up and Roman regressed.

**⬇ REGRESSION (2026-06-20): the curve went DOWN, not up.** 3000→5500 (50M→90M): domain-aware
repeats → **newline/digit collapse** across all categories; escape ~12/19 → **0/19**. Diagnosis:
**pure reverse-KL mode-collapses** (mode-seeking → over-concentrates onto the dominant token,
newline/digits) as training continues. The 3000 "Tier-1 working" read was a **transient diffuse
phase**, not a trend. See training_runs.md 06-20. → pure `rev_kl` insufficient; **next test JSD**.

**JSD result (2026-06-21): WORSE — representation collapse.** JSD @4000 (~65M) → all 19
prompts output `...` (greedy) / punctuation soup (T=0.8), with **token-corr ≈ 1.0,
effective rank ≈ 1.0–1.6** = genuine **hidden-state collapse** (the rank→1 mode fwd/rev-KL
never hit — their reps stayed rank 5–19). So: forward-KL → output collapse; reverse-KL →
output collapse (least-bad, escaped to 12/19 @50M first); **JSD → representation collapse,
earlier and worse.** No divergence alone is the fix. **Next lever is NOT another divergence —
it's the architectural stability recipe** (`--use-sandwich-norm --use-depth-aware-init`, ±
lower LR) paired with rev-KL, now justified because JSD finally surfaced the rank-collapse
those fixes target. See training_runs.md 06-21.

**JSD full-log post-mortem (2026-06-21): the rank→1 collapse was the endpoint of a gnorm
explosion, and the trigger is LR.** Reviewing the complete training log to step 4056: gnorm was
chronically unstable from ~step 1000 (15–55 at n_loops 2), truly exploded at 3100–3400 (**214→1593,
still n_loops 2**), then detonated when the loop curriculum added the 3rd loop at step 3510
(**4271 → 1,000,446**), permanently wrecking the model (loss 1.5→4.5, cv→2.18 expert collapse).
**ρ(A) stayed healthy (0.30–0.34) throughout** → not the recurrence. Two conclusions: (1) **LR 3e-4
is too hot** (the disease); (2) the **`n_loops 2→3` transition is an amplifier, not the cause** —
instability ran away *before* it, ruling out the loop-depth-trigger hypothesis. This is the direct
basis for the rev-KL stability run below.

**rev-KL STABILITY result @3216 (2026-06-23, ~53M tokens): the "0/19" that means the OPPOSITE of
the collapse rows — and the first run on a healthy trajectory.** Probed the stability run
(lr 1e-4 + sandwich-norm + depth-aware-init) at step 3216 (*pre-`n_loops 2→3` transition* — paused
when home). Training metrics rock-solid: gnorm flat <1.0 *through the 3100–3200 zone where JSD hit
214* (3160: 0.91 vs JSD 214), hard CE **0.7**, MoE **cv 0.18**. Generation:
- **Greedy:** lateral vs the 1655 probe — still degenerate (the standing "teacher-forced loss ≠
  free-gen coherence" gap; hard CE 0.7 did *not* buy coherence).
- **T=0.8 (the meaningful read — rev-KL's edge is under sampling):** **0/19 strict escape**, BUT
  qualitatively the best yet — **domain-appropriate lock-ons** (`primary`→"primary colors",
  `np`→numpy) and **sentence-like fragments** (`the first: but because a`, `If the second… The
  entire second:`, `They were in`) before collapse. More contextual relevance + grammatical
  structure than the **1655** probe (same run, earlier — a fair within-run comparison).
- **Reps healthy** (generated rank 2–13, *not* JSD's rank→1) → **exposure-bias output degeneration**
  (recoverable), **not** representation collapse.

**Honest comparison caveat vs freshrevkl@3000 — do NOT oversell this as a clean win.** On the *binary
escape metric*, this run @3216 (**0/19**) reads *below* the old pure-rev-KL @3000 (**12/19**) at a
similar ~50M tokens. That gap is confounded three ways: (a) this probe is `n_loops=4` inference on
`n_loops=2` training (pessimistic, untrained depth) whereas the shorter freshrevkl run was likely
depth-matched at 3000; (b) escape is binary and misses the *character* gains (lock-ons/fragments);
(c) **critically, the old 12/19 was a transient that collapsed to 0/19 by 90M**, while this run's
reps are healthy and *still improving*. Net: "better" here is about **trajectory + rep-health**, NOT
the escape number — the headline metric actually favors the old (doomed) run. The honest,
depth-matched verdict needs a **post-transition probe** (n_loops 3–4 trained *and* inferenced).

**The decisive contrast:** pure freshrevkl was **dead (mode-collapsed) by 90M**; this stable-recipe
run is **still *improving* at 53M, not collapsing.** That is the stability fix paying off — it can
now keep learning productively *past* where the old run tore itself apart. **Stability = solved
(pending the loop-transition confirmation); coherence = still token-gated/exposure-bias (expected,
unchanged) — but for the first time the trajectory is *accumulating, not dying*.**
**Caveat:** probe ran at `n_loops=4` on an `n_loops=2`-trained checkpoint (pre-transition) → inferenced
deeper than trained → **pessimistic read**; should improve once the loop curriculum trains depth 3–4.

---

## Roadmap-hypothesis cross-reference

| Roadmap / ideas claim | Probe evidence @ step_3000 | Verdict |
|---|---|---|
| Degeneration is **exposure bias, not architecture** (roadmap "Current status") | A pure *objective* swap (fwd-KL → rev-KL), zero architecture change, moved escape 1/4 → ~12/19 | ✓ confirmed |
| **Reverse-KL (Tier-1)** is the cure (ideas.md "main thread") | ~12/19 prompts escape under sampling vs forward-KL's 0 | ✓ working — keep tokens, don't build Tier-2 yet |
| **Binding constraint is tokens** (roadmap "Stage two") | No correct fact/computation anywhere (no Jupiter / Paris / 4 / 55); facts surface only as weak argmax ("7"), gone under sampling | ✓ knowledge gap = token volume |
| **In-format > OOD until SFT** | Greedy: in-format diffuse + domain-appropriate; OOD chat → markdown, qa → WH-word echo | ◑ true for *distribution health & greedy*, but under sampling OOD **chat escapes 3/3** (shallow lock) |
| Recurrence is healthy / not collapsing (training_runs 06-16) | Generated-token rep-corr drops to 0.31–0.60 on escapers; ranks stay high | ✓ consistent |

---

## Per-checkpoint detail

### step_3000 — ~50M tokens, 2026-06-19 (fresh reverse-KL run)

**Domain-aware repetition (greedy) — the key qualitative gain.** What it repeats is now *context-correct*:
- code → `def`, `def module`, `self`, `init`, `(self):`, indentation
- math → `x`, `r`, `3`, digits
- prose → prose words; chat → markdown `#`/`###`; qa → the question word ("what"/"where")

vs. the old forward-KL failure where *everything* collapsed to `is is is` regardless of context. So the model has learned the **per-context conditional token distributions** — it knows *which* tokens belong in code vs math vs prose. What it hasn't learned: long-range coherence and **stopping**.

**Escape under T=0.8:** chat 3/3, prose 3/4, code 3/4, qa 2/4, math 1/4 (~12/19).
- **Surprise 1 — chat escapes despite OOD.** The ChatML markdown lock is shallow; sampling breaks it (incoherent, not stuck).
- **Surprise 2 — math is the *worst* (1/4) despite in-format.** Counting/listing prompts form *deep* attractors ("first first first", digit loops) because the model loops on plausible numbers/list-words when it can't actually compute. In-format helps the *distribution* but not the deep counting loops.
- Best single result: prose `recurrent-depth` (generated rep-corr **0.315**, most-decorrelated of all).
- Deepest residual locks: Roman-qa (0.93), France-qa (0.82), sum-math (0.88) — the fact-recall + computation gaps.

**Knowledge probe (qa set).** Under sampling none of the facts surface (no Paris/blue; "7" smears to digit-soup), confirming the specific world-facts are genuinely *not learned yet* — a coverage/token-volume gap, not "present but hidden". When it lacks a fact it falls back to echoing the prompt's question word.

**Noise caveat.** Per-prompt escape is high-variance at this token budget: Roman was the *best* prompt at 1500 and the *worst* at 3000. Treat single-prompt swings as noise; trust the category-level aggregate and its trend across checkpoints.

---

## Version history (pre-freshrevkl lineage) — what changed and what it did to generation

The arc that motivated the fresh reverse-KL run. Three behaviour classes recur:
**varied salad** (diverse but incoherent — no lock), **hard collapse** (single-token
attractor), **diffuse/escaping** (wide distributions sampling can break out of).

> **Decode caveat:** pre-fix rows are from `inspect_checkpoint.py` (T=0.7 greedy +
> best-of-trajectory); freshrevkl rows are from `collapse_metrics.py` (T=0 greedy /
> T=0.8). Not perfectly apples-to-apples on *decode settings*, but the behaviour
> *class* is robust to that. Raw reports cited per row.

| version | params | key change | `recurrent-depth` greedy | Roman greedy | class | raw |
|---|---|---|---|---|---|---|
| **v4** (`grown_v4`) | 397M | OpenHermes SFT — **P0.1 noise still active** | `188\cdot$ package (using std` | `double list was the city.get[::-[]` | **varied salad** (noise-driven) | `inspect_v4.txt` |
| **moe_s0** | 279M | **P0.1 fixed (noise removed)** + clean recipe, fwd-KL distill | `not quite "DDDDDDDD&&` | `R R R R II IIionsions` | **hard collapse** | `inspect_moe_s0.txt` |
| **noise_distill_11k** | 279M | + `recurrent_state_noise` σ (replace the lost P0.1 noise) | `is is is ( ( (` | `The The The … Two Two` | collapse — **marginal** (1→2-word) | `inspect_noise_distill_11k.txt` |
| **revkl_10k** | 278M | reverse-KL distill **continued** from the collapsed 24-expert base | `is is is …` | `R:\n\nThe The The` | collapse persists (can't un-teach) | `collapse_revkl_10k.txt` |
| **freshrevkl @1500** | 278M | reverse-KL from **random init** ("teach it right from the start") | `correct the the the` (diffuse) | diffuse (esc. @T=0.8) | **diffuse, 1/4 escape** | `collapse_freshrevkl_1500*.txt` |
| **freshrevkl @3000** | 278M | + ~25M tokens | domain-aware repeats, diffuse | locked (regressed) | **diffuse, ~12/19 escape** | `collapse_freshrevkl_3000_full*.txt` |
| **freshrevkl @5500** | 278M | + ~40M more tokens | `\n\n\n…12…` newline/digit collapse | `\n\n\n…12…` | **REGRESSED — mode collapse, 0/19** | `collapse_freshrevkl_5500_full*.txt` |

**The narrative these rows tell (cross-ref: training_runs.md 06-15/06-16, roadmap "Current status"):**
1. **v4's "variety" was an artifact, not capability.** It looked best because **P0.1's
   accidental noise** (a clobbered zero-init injecting noise into the hidden state) kept
   it out of the repetition attractor at inference. Real, but accidental.
2. **The P0.1 fix removed that noise → exposed the underlying exposure-bias collapse**
   (moe_s0: hard `DDDD` / `R R R R`). The fix didn't *cause* a regression; it revealed
   the true free-running behaviour the noise had been masking.
3. **Decode/inference band-aids failed.** The `recurrent_state_noise` σ knob (the
   principled replacement for P0.1's noise) only nudged it from 1-word to 2-word repeats.
4. **Reverse-KL *continued* on an already-collapsed base failed too** (revkl_10k) — the
   attractor was entrenched; you can't un-teach it.
5. **Reverse-KL *fresh* escaped the attractor early (1500→3000) but then COLLAPSED**
   (5500). Pure reverse-KL is mode-seeking → over-concentrates onto the dominant token
   (newline/digits) with continued training. So "teach it right from the start" got *past*
   the exposure-bias attractor, only to fall into a *different* one (mode collapse). Pure
   rev-KL is not the answer on its own.

## Standing conclusions / what to watch — UPDATED 2026-06-20 (verdict flipped)

1. **Pure `--divergence rev_kl` is NOT sufficient — it mode-collapses by ~90M tokens.** The
   earlier "Tier-1 validated, keep pouring tokens" call (based on 1500→3000) was premature: two
   improving points were a *transient diffuse phase*, not a trend. 5500 regressed to 0/19. Cheap,
   valuable negative result (found for ~$0, pre-rented-compute).
2. **Next test: JSD** — `--divergence jsd --jsd-beta 0.5`, **fresh from random init** (attractor
   entrenches; can't un-teach a collapsed ckpt — cf. revkl_10k). JSD interpolates mode-covering
   (fwd) + mode-seeking (rev) → should avoid the pure-mode-seeking collapse. If insufficient →
   full **Tier-2** (teacher-mixed sampling α≈0.2 + on-policy; the MiniLLM/GKD recipe for exactly
   this).
3. **Lesson for reading the curve:** don't call a trend from two points. Require a *third*
   checkpoint before declaring direction — the 1500→3000→5500 arc (up, then down) is the case study.
4. **Still open (carry to the JSD run):** (a) does any divergence setting reach *correct answers*
   (real knowledge) rather than just diffuse-vs-collapsed? (b) does the in-format vs OOD-format gap
   (prose/code/math vs chat/qa) hold under a non-collapsing objective?

## IN PROGRESS — rev-KL stability run (2026-06-21): the LR fix on trial

Fresh from random init: **rev-KL + lr 1e-4 + `--use-sandwich-norm --use-depth-aware-init`**
(`checkpoints_revkl_stable`). The first run aimed squarely at the gnorm-explosion diagnosis, not
the divergence. Paused at step 585 (resuming overnight). **No probe read yet** — too early; the
discriminating zones are deeper.

- **Training-metric read so far (to 585):** gnorm settles ~1–2.3 at peak LR (vs JSD heading to 15+);
  rev-KL soft term *decreasing* 6.5→3.7 (converging, not plateauing); cv healthier than JSD at
  matched steps. **But JSD looked equally calm at 585** — verdict requires clearing the danger zone.
- **The gnorm verdict zones (the whole test):** 1000–1500 (JSD crept to 15–40), 3100–3400 (JSD
  exploded 100–1600 at n_loops 2), and especially **~3510, `n_loops 2→3`** (JSD went 4271→1,000,000
  and died). Staying single/double-digit through that transition = the fix confirmed.
- **First probe read DONE @3216** (~53M, *pre*-`n_loops 2→3` transition — paused when home): 0/19
  strict escape but **healthy reps + domain lock-ons + sentence fragments + improving-not-collapsing**
  — the recoverable exposure-bias regime, best 0/19 yet. See the rev-KL stability result note above
  and the cross-comparison row. **Still pending:** (a) the `n_loops 2→3` transition (~3510) gnorm
  verdict; (b) a *post-transition* probe at `n_loops=4`-matched depth (this read was `n_loops=4`
  inference on `n_loops=2` training → pessimistic).

---

## 2026-06-27 — ✅ ON-POLICY @6771: first α=0.0 un-collapse (partial, dose-limited)

Probe of `checkpoints_onpolicy/step_0006771.pt` (96 on-policy steps off 6675, λ=0.5 α=0.6).
Tool: `tools/onpolicy_rollout_probe.py` (α = 0.0/0.25/0.5/0.7 × 3 seeds). **Read the α=0.0
rows** — pure student, no teacher-mix = the real success metric; α>0 is teacher-assisted and
doesn't isolate the student's own gain. `top_share / distinct1`, baseline 6675 → 6771:

| Seed | 6675 α=0.0 | 6771 α=0.0 | verdict |
|---|---|---|---|
| Weather (prose) | 0.45 / 0.15 — `this this was was` | **0.14 / 0.66** — varied sentences | **un-collapsed** |
| Bacterial (medical) | 0.89 / 0.06 — `the the the` | 0.90 / 0.09 — `the the the` | no movement |
| fibonacci (code) | 0.27 / 0.12 — numbers | 0.14 / 0.16 — numbers | marginal |

The prose un-collapse is **large and real — first movement on the unaided-generation blocker
in the project's history** (top_share nearly thirded, distinct1 ×4.4, stuck attractor → real
sentences). Uneven = **dose** (prose over-represented in the corpus un-collapses first;
medical/code sparser → lag, need more on-policy tokens), NOT a mechanism failure. α=0.5/0.7
look reasonable across seeds but that's the teacher-mix carrying them — α=0.0 is what counts.
**Mechanism validated; on-policy is the cure, confirmed empirically.** Caveat on α=0.25: it
*regressed* on some seeds (seed-1 0.98) — too weak to break the attractor *and* perturbs the
student; ignore, the bracketing α=0.0 and α=0.5+ are the signal. **Next:** continue from 6771,
λ→0.7, re-probe; full verdict in training_runs.md 2026-06-27.

---

## 2026-06-28 — ✅✅ ON-POLICY BROKE THE COLLAPSE DOMAIN-WIDE (now tokens-bound)

Probe of **step 6906** (~231 on-policy steps off 6675; +135 at λ=0.7 over the 6771 run),
6-seed set (prose / 3 medical / code / math). **Headline: the hard repetition attractor is
GONE on every seed.** α=0.0 `top_share` across all six: **0.11 / 0.18 / 0.11 / 0.23 / 0.31 /
0.06** — all low; no `is is is` / `the the the` anywhere. The collapse that blocked the
project for months is broken **domain-wide**, not just prose.

**Methodological correction (important):** the 6906 *3-seed* probe showed bacterial α=0.0 at
`top_share 0.97` (`the the the`) — but the *6-seed* run, **same checkpoint**, gave bacterial
α=0.0 at **0.18 (varied)**. Only difference: seed *order* → RNG state. So a **single sampled
rollout per (seed,α) is high-variance**; that "medical hard-collapsed" read was a noisy draw,
not a real attractor. → probe now **multi-samples** (`--samples`, default 3, reports
mean [min-max]) so one unlucky draw can't mislead.

**Robust re-probe (n=3, step 6906):** α=0.0 mean `top_share` per seed (prose / bacterial /
diabetes / ibuprofen / code / math): **0.15 / 0.11 / 0.12 / 0.18 / 0.22 / 0.20** — all low,
tight ranges (bacterial 0.10–0.12; the 0.97 was definitively noise). Gradient: prose + medical
cleanest; **math weakest** (dash/digit spam). Capability at α≥0.5: fibonacci wrote *real* Python
(`fib_sequence = [0, 1]; while …`), ibuprofen → "pain, inflammation, fever", diabetes →
"increased thirst and urination". Conclusion unchanged, now confirmed robust.

---

## 2026-06-29 — 🧠 KNOWLEDGE PROBE: real but COARSE domain-cluster knowledge at 110M tokens

A domain-expert catch — `B104`, a real ibuprofen/PPARγ neuronal cell line, surfaced at α=0.0
inside incoherent text — prompted a **knowledge-vs-fluency** test (step 7024).

**Generation can't measure knowledge here** (`tools/knowledge_probe.py`): 45 ibuprofen α=0.0
rollouts surfaced **0** diagnostic entities — but so did common terms (`nsaid`/`agonist`), so
it's the *fluency floor*, not absent knowledge. Free generation needs a long lucky token chain
to land a fact; at this fluency it can't.

**Likelihood (cloze) test** (`tools/knowledge_likelihood_probe.py`) — teacher-force the fact,
read which entity gets the lowest NLL (no generation → fluency-independent). v1 result
(ubiquitous distractors): **`B104` ranked #1, beating `HEK293`** (a corpus-ubiquitous line) in
the ibuprofen/PPARγ context; `PC12`/`RhoA` crushed the wrong-context distractors
(`Jurkat`/`HeLa`, `mTOR`/`EGFR`) by ~3–6 NLL. The strict `1/4` headline was a **frequency
confound** (obscure-correct vs ubiquitous-distractor), not weak knowledge.

**Verdict: the student carries REAL but COARSE domain-cluster knowledge.** It learned the
*co-occurrence cluster* (PPARγ neuronal research → PC12 / B104 / RhoA) — context sorts "belongs"
from "doesn't" decisively — but NOT the fine facts: PPAR-γ vs -α/-β undifferentiated; `B104`
also wins for *Metformin* (it keys on the local "PC12 and ___" pairing, not the distant drug,
at this under-training). So: **correct semantic scaffold, no precise causal/drug structure yet.**

**Why it matters:** (1) forming the *right* cluster at 110M tok / 278M params (brutal
under-training) is the **token-efficiency** the recurrent-depth bet predicts — not generic
small-model noise. (2) It's exactly the **retrieval-paired medical design** — weights hold the
domain scaffold, retrieval supplies precision; the B104 catch validates both halves. Probe
refined 2026-06-29 (frequency-matched obscure distractors + a non-bio control); clean re-run
pending. Reading lesson stacks: top_share noise → single-sample noise → *generation can't probe
knowledge; use likelihood, and frequency-match the distractors.*

**⚠ CORRECTION (2026-06-29, same day) — the refined probe DEFLATED the knowledge claim;
the above "real domain-cluster knowledge" is RETRACTED.** Added frequency-matched obscure
distractors (B35/B50/NG108, Rho-family) + a **non-bio control** ("In the morning the weather
was clear and we saw a ___"). Result: **`B104` ranks #1 in the *weather* context too** (and
for Metformin) — it beats the distractors *regardless of context*. So B104's lead is **token
frequency, not an ibuprofen association**; the v1 "B104 beats HEK293" was the same frequency
effect (HEK293 is merely even more common). The only *real* signal left is **coarse slot/type
priming** — the bio context lowers *all* cell-line NLLs (B104 1.56 bio vs 2.42 weather; B35
1.87 vs 2.77), i.e. the model learned "a cell-line-shaped token belongs after 'PC12 and ___'"
— the *shape of the slot*, NOT the *fact that fills it*. That's distributional learning, not
medical knowledge. **Unaffected:** (1) the coherence-climb result (separate, still real);
(2) the retrieval-paired design — specific facts were always retrieval's job, never the 278M
weights, so "no parametric facts" is the *expected* division of labor, not a loss. **Lesson:
even a likelihood probe needs a non-bio control to separate knowledge from token frequency —
and watch for over-reading an exciting single catch (the B104 generation + the confounded v1
both pointed the wrong way).**

**Regime shift (the whole point):** "sharp repetition attractor" (exposure bias) →
**"varied but incoherent word-salad"** — the *normal* regime of a small, undertrained model.
The exposure-bias **blocker is cured**; what remains is coherence/capability = **tokens +
scale**, the lever tokens *actually* move (unlike the attractor, which they worsened).
Capability is present at α≥0.5: diabetes α=0.7 gave the **correct symptoms** (thirst /
frequent urination / fatigue / blurred vision); ibuprofen → pain + long-term side effects;
fibonacci → real code + test reasoning. Knowledge/structure is there; unaided (α=0.0) fluency
is what's missing.

**Verdict:** on-policy converted a tokens-*proof* attractor into a tokens-*responsive*
undertrained model — the thesis flip the project was chasing. **Next = pour tokens on the
un-collapsed base** (throughput → the Max 1100). Full context: training_runs.md / roadmap.md
(2026-06-28).


## 2026-06-30 — ⏸ α=0.0 PLATEAU at fixed α=0.6 → start the α-anneal

Probe of **step 7242** (~218 steps past 7024, all at fixed λ=0.7 / **α=0.6**).
`onpolicy_rollout_probe`, n=3, 6 seeds, α=0.0/0.25/0.5/0.7.

**α=0.0 (pure student) — FLAT vs 7024, still varied-but-incoherent.** top_share across the six:
prose 0.17 / bacterial 0.30 / diabetes 0.16 / ibuprofen 0.12 / fib 0.19 / quad 0.16 — no coherence
jump, no movement on the 7024 read. Matches the **loss plateau** (~1.5 soft / ~0.85 over 190 steps;
an earlier "loss dropping" read was noise off a lucky 7030 sample). Bacterial α=0.0 spiked to **0.47
on one sample** (LaTeX-symbol attractor) → un-collapse holds but is **fragile** on the symbol/number
seeds.

**α≥0.5 — capability clearly PRESENT (teacher-assisted):** bacterial α=0.7 = correct
antibiotic/antifungal/antiviral/antiparasitic taxonomy ("antibiotics target bacteria, which are
prokaryotic microorganisms…"); diabetes α=0.7 "increased thirst and urination"; ibuprofen α=0.5
"pain, fever and inflammation", α=0.7 real brand names (Advil/Motrin).

**Diagnosis:** capability present but **NOT internalized into α=0.0.** Fixed α=0.6 keeps 60% of each
rollout teacher-driven → the student rarely recovers from its *own* errors → the exposure-bias gap
doesn't close by token-grinding alone. **Decision → start the documented α-anneal: 0.6 → 0.5**
(tonight's run, from 7242). Hypothesis + what-to-watch (loss may rise = expected/good; watch
fragile-seed re-collapse): **onpolicy_plan.md 2026-06-30**.


## 2026-07-01 — 🔽 α-ANNEAL VERDICT (0.6→0.5): SAFE + metric moved, text still salad

Probe of **step 7458** (~216 steps at **α=0.5**, off 7242). Same tool/config (n=3, 6 seeds).
This is the anneal experiment's read: did dropping α 0.6→0.5 convert tokens to unaided coherence
faster than the flat-0.6 grind (which left α=0.0 flat)?

**α=0.0 top_share, 7242 → 7458:** weather 0.17→**0.09** · bacterial 0.30(max0.47)→**0.13**(max0.18,
**de-fragilized**) · diabetes 0.16→**0.11** · ibuprofen 0.12→0.11 · fib 0.19→**0.13** · quad
0.16→0.16. **Mean 0.18 → 0.12 (~⅓ down), 4/6 seeds improved.**

**Two solid takeaways:** (1) **No re-collapse** — the risk we watched for didn't happen; the
fragile bacterial/LaTeX seed *de-fragilized* (0.47→0.18). Anneal to 0.5 is **safe**. (2) The
**distribution moved** — more than the flat-0.6 grind (which was flat) at ~equal step count. So on
the *metric*, the anneal beat the flat grind.

**BUT — α=0.0 text is still incoherent salad** (weather "get the 1112. But as an interesting task
for the 10,300…"; ibuprofen "the and a bit. *c) A The number of A(969.56)…"). Lower top_share alone
is **ambiguous** (less-repetitive-toward-coherence vs just more-random); the text says **no
coherence jump yet**. Claim only what's clean: **safe + no re-collapse + distribution nudged right.**

**α≥0.5:** capability stable (bacterial α=0.7 antibiotic/antifungal/antiviral taxonomy; diabetes
α=0.7 "increased thirst and urination"). Factual wobble: ibuprofen α=0.7 called it a "proton pump
inhibitor" (**wrong** — NSAID/COX inhibitor; knowledge gap = tokens/scale, not a health issue).

**Chinese chars** on fib α=0.25 = **α=0.25-only noise** (α=0.0 fib is English code-salad, no
Chinese): untrained multilingual Ouro vocab reached in the awkward middle-mix on the weakest seed.
Not in the read.

**Decision: HOLD α=0.5, grind tokens.** 216 steps is too few to judge conversion, and the bottleneck
is **token volume, not α** — stepping α every session just adds unattributable noise. Give 0.5 a
real dose (~1,000+ steps across sessions; the Max makes this cheap), then re-probe for a read that
can separate "toward coherence" from "toward random." Decision context: onpolicy_plan.md 2026-06-30/07-01.


## 2026-07-06 — 🚀 SECOND REGIME SHIFT: salad → rambling-grammatical English (de-tax worked)

Probe of **step 8668** (~1,210 steps de-taxed off 7458: **full-strength on-policy (A1 fix) + EOS
(A2)**, λ=0.7, α=0.5). The big weekend verdict.

**⚠ METHODOLOGICAL CATCH — top_share INVERTED here; read the TEXT.** Mean α=0.0 top_share *rose*
0.12→0.16, which naively reads as "flat/worse." It's not — **fluent English repeats "the/of/a" far
more than random number-salad**, so climbing salad→sentences *raises* top_share even as coherence
improves. On the 4 improved seeds top_share is flat (~0.12); the 2 laggards pull the mean up.
`distinct1` is the honest metric (0.46→0.50; bacterial 0.43→0.54, quad 0.35→0.45). **Lesson: at the
salad→fluency transition, top_share is misleading-to-inverted — the text is the only real read.**

**α=0.0 evidence (7458 → 8668):** weather `"get the 1112. But as an interesting task…"` → *"the
number of people who had been able to take… What are your new of a long-time"* (connected grammatical
clauses); bacterial `"the low the bacterial, because…"` → *"The study of the proposed research showed
a highly detailed review of… various treatment"* (real English, on-topic); fibonacci symbol-salad →
**actual Python syntax** (`def test_n_r(self,c,a)`, `if not self.is_id_f_a_line():`, `for i in
self.new_info():`). **The fluency floor lifted** — clearest α=0.0 movement of the project.

**Boundary — fluent, NOT yet meaningful.** Grammar + on-topic vocab, but doesn't convey correct
info/reasoning — grammatical rambling, not thought. Real progress (fluency = prerequisite for
meaning), not "it works." Still ~120M tokens, deeply undertrained.

**Laggards (not uniform):** ibuprofen α=0.0 still stuck in the symbol/LaTeX salad; fibonacci α=0.0
high-variance (1/3 samples repetitive, top_share 0.62). Watch both.

**α≥0.5 stronger too:** diabetes α=0.7 = genuinely good clinical prose (*"increased thirst, frequent
urination or urination at night, but some of these symptoms can be vague, which leads people to
dismiss them to less severe conditions"*) with correct symptoms; ibuprofen pharmacology; antibiotic
taxonomy **with reasoning**; real fibonacci-structured code.

**Verdict: the de-tax WORKED.** α=0.0 moved far more than the flat taxed run (7458) did — full-strength
on-policy converted the big token dose into visible coherence. The A1/A2 fixes paid off (Opus
under-predicted this as "faint, still salad"). Progression now: collapsed (`is is is`) → varied-salad
(6906) → **rambling-grammatical English (8668)**. **Decision:** α=0.5 is *working* → **hold it, pour
TOKENS (the Max)** to push fluency→meaning; a gentle 0.45 anneal is optional/secondary — don't tweak
a setup that just delivered. Context: onpolicy_plan.md 2026-07-06.


## 2026-07-15 — ✅ XPU workaround stack A/B-VALIDATED (no behavioral drift) + α drift caught

**Purpose: cross-backend consistency check, not quality measurement.** The Max 1100 stack changed
the numerics under the model (rope_real instead of complex RoPE, manual bmm attention instead of
SDPA, CPU sampling). Before attributing any future probe movement to training, we needed to know
the workarounds themselves don't shift behavior. The exact-replay plan (re-probe step 8668 on XPU
vs its recorded 5070 outputs) was impossible — **step_0008668.pt rotated away** (oldest survivors:
9774 in `checkpoints_onpolicy`, 9838 in `checkpoints_onpolicy_xpu`). Ran a stronger same-checkpoint
two-backend A/B on **step_0009881.pt** instead: 5070/`cuda:0` (original numerics: torch SDPA,
complex RoPE) vs Max/`xpu:0` (full workaround stack), same venv-per-backend rig.

**Raw-logit A/B (6 prompts, identical inputs, text-free):** max per-position KL **≤ 0.03 nats**
(against 2–7 nats of distribution entropy), mean |Δlogit| 0.02–0.05, max |Δlogit| ~1.05. Greedy
argmax agreement 100% on 4/6 prompts, 83–86% on the rest — mismatches only at near-tie positions.

**Greedy 19-prompt probe (`collapse_metrics --probe-set all`), both backends:** first-token
distributions match to ~0.01 top_prob on all 19; texts fork mid-sequence on 15/19 but only where a
near-tie flips, then chaos-amplify. Same degeneration character on both sides. **Verdict: ordinary
bf16 cross-backend noise, NOT drift — the segfault-workaround stack is behaviorally faithful.**
Corollary: greedy text is NOT diffable across backends; diff distributions/metrics, or sampled
n=3 aggregates. Raw: `reports/collapse_onpolicy_xpu_9881_greedy_{xpu,cuda5070}.txt`.

**⚠ α drift caught in the doc command.** The XPU main-run block in `training_commands.md` carried
`--teacher-mix-alpha 0.6`, copied from the pre-anneal 2026-06-27 command — but the validated
decision (7458 → 8668) is **hold α=0.5**. The ~100 XPU steps of 2026-07-14 (9780→9881) and the
smoke tests likely ran at 0.6. **Decision (2026-07-15): α=0.5 for all future runs**; doc fixed.
Config-attribution caveat for the next entry: 8668→9881 spans ~1,200 steps at mixed/uncertain α.

**Baseline for the first Max token-dose (6-seed rollout probe, n=3, step 9881, XPU) — ⚠ α=0.0
REGRESSED vs 8668.** Per-seed α=0.0 `top_share` (mean of 3): weather **0.33** / bacterial **0.38**
/ diabetes **0.40** / ibuprofen **0.39** / fibonacci **0.56** / quadratic **0.41** → **mean ~0.41
vs 8668's 0.16**; `distinct1` mean **~0.22 vs 8668's 0.50**. The text agrees with the metrics this
time (no top_share inversion): α=0.0 is back to fragmented salad with repetition bursts (`protein
protein protein…`, `in in in in`) — the 8668 "rambling-grammatical English" is gone at α=0.0.
High-α capability persists (α=0.5 bacterial: *"antibiotic therapy… The correct answer is d)"*;
α=0.7 quadratic: genuine CoT-style *"Okay, I have to solve the quadratic equation… Hmm, first
thing I'm thinking…"*), so the regression is in the student's own trajectory, not capability —
the exposure-bias signature again. **Prime suspect: the ~1,200 post-8668 steps run from the doc
command at un-annealed α=0.6** (exactly the flat-0.6 regime that left α=0.0 flat pre-anneal);
backend is exonerated by the A/B above. Cannot fully rule out other causes — the overnight α=0.5
dose diffs against THIS baseline and should arbitrate. Raw:
`reports/onpolicy_rollout_probe_9881_xpu.txt`.


## 2026-07-16 — 🔴 PHASE-5 CACHED ROLLOUTS NOT DISTRIBUTION-PRESERVING; probe instrument split (cached vs uncached)

**The α=0.5 arbitration dose (9881→12000, ~2,100 steps overnight) did NOT recover α=0.0 — the
cached-instrument probe got WORSE** (top_share mean 0.41→0.59, distinct1 0.22→0.12, all 6 seeds,
n=3), with a new **token-doubling signature at every α** ("go go on on", "of of of") — evidence
of learning from corrupted rollout text, not exposure bias. α-drift hypothesis REJECTED as main
driver. Raw: `reports/onpolicy_rollout_probe_12000_xpu_cached.txt`.

**Root cause found & measured — the phase-5 cached student decode samples OFF-DISTRIBUTION.**
Cached vs uncached student logits on the SAME device/weights/trajectory (step-12000 ckpt, XPU):
max |Δlogit| **5.5**, KL up to **0.95 nats** — ~30× the entire cross-backend gap (≤0.03) validated
2026-07-15. Mechanism: the recurrent block's convergence/ACT early-exit needs *every position in
the current forward* to converge — the whole sequence uncached, but only the single new token
cached → per-token effective loop depth differs → different ACT-weighted output. Greedy argmax
survives (why `test_greedy_sequences_*` stayed green) but temp-1.0 sampling draws from a shifted
distribution. `generate_rollout` defaults `use_kv_cache=True`, so **all phase-5 training rollouts
AND the 07-15/07-16 probes used the shifted path**; every probe entry ≤8668 is the uncached
instrument. The two instruments are NOT comparable — probe entries must state which they used.
(`--no-kv-cache` flag added to `onpolicy_rollout_probe` for the legacy instrument.)

**Dose–response, cached instrument (apples-to-apples):** α=0.0 top_share mean 9780 (0 phase-5
steps) **0.34** → 9881 (~100) **0.41** → 12000 (~2,200) **0.59**; distinct1 0.28→0.22→0.12.
Monotone in phase-5 steps. Raw: `reports/onpolicy_rollout_probe_9780_xpu_cached.txt`.

**True damage is real but ~half instrument artifact — 12000 re-probed UNCACHED:** α=0.0
top_share mean **0.28** (not 0.59), distinct1 **0.37** (not 0.12), and the text is still
recognizably the 8668 rambling-grammatical-English regime (weather: *"This question is a very
long, but actually the time here is the same at the right…"*), NOT hard salad. vs 8668 (0.16 /
0.50): a **moderate real regression**, consistent with ~2,200 steps of training on
off-distribution rollout text at tail LR. Raw:
`reports/onpolicy_rollout_probe_12000_xpu_uncached.txt`.

**Standing lessons:** (1) any cached/incremental decode path in an ACT/early-exit architecture
needs a DISTRIBUTION-level (KL) equivalence gate, like the teacher cache already has — greedy
sequence tests are insufficient by construction; (2) never change training path and measurement
instrument in the same window. **Next:** fix rollout generation (wide-batch uncached, or repair
cache semantics), restart on-policy from a pre-phase-5 checkpoint (9780) with the LR schedule
extended, re-probe uncached against the 8668 baseline.


## 2026-07-17 — ✅✅ FULL RECOVERY at 13,944 — fixed rollouts un-did the phase-5 damage

Mid-run probe of `checkpoints_onpolicy_fixed/step_0013944.pt` (~4,160 fixed-rollout steps past
the clean 9780 restart, α=0.5, λ=0.7, real LR). **Instrument: uncached, 5070/CUDA** — the SAME
instrument and backend as the 8668 baseline, cleanest comparison since the migration. (venv-cuda
had drifted to transformers 5.13.1, which breaks Ouro's custom code — the known `<5` pin; fixed
to 4.57.6. Probe ran on the idle 5070 while training continued on the Max, zero interference.)

**α=0.0 per seed (top_share / distinct1):** weather **0.15/0.56** · bacterial **0.10/0.60** ·
diabetes **0.14/0.58** · ibuprofen 0.38/0.40 · fibonacci **0.10/0.60** · quadratic **0.12/0.34**
→ **mean 0.17 / 0.51 — the 8668 level (0.16/0.50) exactly**, from the tainted-12000 read of
0.28/0.37. **5/6 seeds individually at-or-better than 8668**; ibuprofen remains the documented
laggard (it lagged at 8668 too). distinct2 mostly 0.7–0.9. The doubling signature is GONE.
Text agrees: connected clauses with emerging Q&A structure (bacterial: *"the use of an
alternative method that causes the amount of the body; What is the relationship between an
immune and is the most effective disease…"*), python-comment-structured code on fibonacci,
real algebraic manipulation attempts on quadratic.

**Verdict: the cached-rollout defect was THE cause of the 9780→12000 damage, and clean
on-policy tokens repair it fast** — ~4.2k fixed steps fully reversed ~2.2k corrupted steps and
re-reached the pre-damage frontier. The coherence-scales-with-clean-on-policy-tokens thesis
survives with its strongest evidence yet. Run continues to 18,000 (ETA ~19:30 tonight);
final probe then, same instrument (`--no-kv-cache`, 5070). Watch whether 18,000 pushes PAST
8668 (first new frontier since 07-06) and whether ibuprofen de-lags.
Raw: `reports/onpolicy_rollout_probe_13944_cuda_uncached.txt`.


## 2026-07-17 (evening) — 18,000 final: PARITY HELD, NO ADVANCE — but the frontier test ran at tail LR

Run complete (9780→18000, all fixed rollouts, α=0.5, λ=0.7). Final probe, same instrument
(uncached, 5070). **α=0.0 means: top_share 0.225 / distinct1 0.385** vs 13,944's 0.17/0.51 and
8668's 0.16/0.50 — numerically flat-to-slightly-down, and the per-sample brackets are huge
(weather d1 [0.14–0.65]: one number-runaway draw drags the mean). **Text says regime UNCHANGED**:
bacterial still produces connected self-questioning English (*"What is the relationship between
an immune and is the most effective disease…"*), quadratic still algebra-shaped, fibonacci emits
runnable-looking `print(...)` calls; diabetes drew an initials-salad sample. Same
grammatical-rambling regime as 13,944, noisier draws. **Ibuprofen did NOT de-lag** (0.44 —
three probes running; candidate for the unlikelihood lever, ideas.md).

**Mid/high α improved slightly** (α=0.5 mean top_share 0.113→0.095, d1 0.58→0.59; α=0.7
healthy) — the training-distribution band is fine; no re-collapse anywhere.

**⚠ The confound that keeps this from being a clean negative: steps 13,944→18,000 ran at
LR ~1.2e-5 → 0** (cosine tail, again). The recovery phase (9780→13,944) had real LR
(~4.5e-5→1.5e-5); the frontier phase got a dying schedule. So "tokens don't move the frontier"
has NOT actually been tested at full signal — what's been shown is: clean on-policy tokens
REPAIR damage fast at real LR, and do ~nothing at near-zero LR (which is what near-zero LR does).

**Decision options for the next leg (pick before more compute):** (a) **extend
`--total-steps` to 30,000** — zero-code, puts step-18k LR at ~3.6e-5 (real), ~34 h for the
12k-step leg; (b) same but shorter legs with re-extension each time (keeps LR floor higher,
more probe points); (c) add a constant-LR / min-LR floor option to the schedule (small code).
The teacher-corpus build (docs/teacher_corpus_plan.md) was gated on "probe passes" — status is
**inconclusive-due-to-LR**, so the gate decision passes to the owner; the generator's value case
(token supply for exactly these longer legs) is unchanged.
Raw: `reports/onpolicy_rollout_probe_18000_cuda_uncached.txt`.


<!-- ===== moved from docs/roadmap.md (2026-06-27 doc reorg) ===== -->

## Test Prompts

Use these prompts with `inspect_checkpoint.py` to test the model's capabilities across the different domains in the clean SFT mix. 

> **Note on PowerShell:** Using angle brackets like `<ckpt_path>` in PowerShell will cause a `ParserError`. The examples below use a real path (`checkpoints_v6_clean_sft/step_0003000.pt`). If your checkpoint is named differently, just replace the path.

### Code Generation (`clean_code`)
```bash
python inspect_checkpoint.py --ckpt checkpoints_v6_clean_sft/step_0003000.pt --device cuda:0 --prompt "Write a Python function to find the longest common subsequence of two strings. Include type hints and comments explaining the dynamic programming matrix."
```

### Math & Reasoning (`clean_math`, `clean_numina`)
```bash
python inspect_checkpoint.py --ckpt checkpoints_v6_clean_sft/step_0003000.pt --device cuda:0 --prompt "A train leaves Chicago at 2 PM traveling at 60 mph. Another train leaves at 3 PM traveling in the same direction at 80 mph. What time will the second train catch up to the first?"
```

### Medical/Science QA (`clean_pubmedqa`, `clean_chem`)
```bash
python inspect_checkpoint.py --ckpt checkpoints_v6_clean_sft/step_0003000.pt --device cuda:0 --prompt "What are the common symptoms and recommended treatments for acute bronchitis? Please provide a structured answer."
```

### General Instruction Following (`clean_general`, `clean_miriad`)
```bash
python inspect_checkpoint.py --ckpt checkpoints_v6_clean_sft/step_0003000.pt --device cuda:0 --prompt "Explain the concept of 'entropy' in thermodynamics to a high school student, using an everyday analogy."
```ds; `K=1` reduces to current behaviour; best-exit
   target matches `forward_trajectory` argmin; depth regulariser still fires;
   no-NaN train step.

**ANSWERED (2026-06-09, P0.5 audit): supervise MoDr with per-loop CE, NOT
uncertainty-argmin.** `tools/per_loop_calibration.py` measured per-loop ECE on
v2 and v4 (`reports/per_loop_calibration_p05.md`): the head is well-calibrated
at loops 1–3 (ECE 0.01–0.04) but **badly miscalibrated at loop 0** (ECE
0.17–0.22, error *understated* by ~0.2 — the loop curriculum starts at 2, so
loop 0 was never an emission loop and the head never saw it). An
uncertainty-argmin teacher would systematically over-select loop 0.
Consequences: per-loop CE is the mandated best-exit target;
`BestOfTrajectoryGenerator` now defaults `min_loops=2` (loop 0 excluded from
the argmin); the earlier "v4 prefers loop 0 on some prompts" reads were partly
a calibration artifact. To unlock all-loop uncertainty selection later: add a
per-loop calibration term in training (BCE against per-loop argmax error at
every loop), or start the curriculum at 1.

**Relation to prior art.** This is the project's own framing of Mixture-of-Depths
(Raposo et al.) adapted to a *recurrent* (weight-shared, looped) block rather
than a stack of distinct layers — depth here means loop count, not layer index.
