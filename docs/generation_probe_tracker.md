# Generation-probe tracker (cross-comparison)

> ## ⚠️ READ FIRST — the metric is a proxy; the TEXT is ground truth.
> `top_share` / `distinct1` are cheap degeneracy *proxies*, not the thing we
> care about. **When the number and the output text disagree, the text wins.**
> This is a standing principle we'd invoked verbally for months but never wrote
> down until it nearly cost us a wrong call (2026-07-25, the @46,000 verdict):
> - **top_share is actively MISLEADING on code.** Valid code repeats tokens
>   (`if`, `return`, `def`, indentation, `0.0`) → *high* top_share. A degenerate
>   giant-float-literal has all-unique digits → *low* top_share. So the metric
>   **rewards the garbage and penalises real structure.** At @46,000 fibonacci's
>   top_share "regressed" 0.11→0.31 while the *text* went from a v1 float-blob to
>   actual Python `def`/`return`/`if-else` — a **win** the number reported as a loss.
> - **A high mean with a wide range is variance, not collapse.** n=5 with
>   `[0.09–0.98]` = one collapsed sample dragging the mean; the typical/`e.g.`
>   sample usually reads fine. Always read the `e.g.` text before trusting a
>   mean-top_share jump.
> - **Procedure:** never call a regression (or a win) from top_share alone.
>   Pull the `e.g.` outputs for the moved seeds and read them; the metric only
>   flags *where to look*. (This is the "exposure bias is decoupled from every
>   formal metric" lesson, made operational.)

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


## 2026-07-18 — mid-leg probe @24,010 (real LR, half dose): starvation-dip recovered, frontier plateau holds

Probe of `step_0024010`-era checkpoint (~6,000 min-lr steps / ~35M tokens past 18k at LR
5.5e-5→3.7e-5), uncached/5070. **α=0.0 means: top_share 0.207 / distinct1 0.45** — per seed
(ts/d1): weather 0.13/0.54 · bacterial 0.13/0.60 · diabetes 0.25/0.32 · ibuprofen 0.40/0.31 ·
fibonacci 0.10/0.48 · quadratic 0.23/0.45. **Read:** the starved-18k dip (0.225/0.385) lifted
once real LR resumed, but the metric sits AT the 8668/13,944 parity band (0.16–0.17/0.50–0.51),
not past it. Four probes now cluster in a **plateau around the frontier** (0.17 / 0.225 / 0.207
vs 8668's 0.16) with n=3 brackets wide enough (diabetes ts [0.10–0.51]) that within-plateau
ordering is noise. Text regime unchanged-to-slightly-sharper: quadratic drew its most math-shaped
sample yet (*"Which is the quadratic function f(x + 7) = 23? … we can use the method to solve
this equation"*); bacterial holds academic register; **ibuprofen laggard: 4th consecutive probe**.

**Verdict: no breakout at half dose; no regression; second half runs (nothing better to do
with the card before Sunday).** If the 30k final also lands in the plateau band, that's the
first honest evidence that tokens-alone plateau at this size/data — plan-B experiments
(teacher-corpus ratio A/B per docs/teacher_corpus_plan.md; unlikelihood for the laggard) are
next, not more extension. **Methodology for the 30k referendum: run `--samples 5`** — the
verdict probe should not inherit n=3 error bars.
Raw: `reports/onpolicy_rollout_probe_24010_cuda_uncached.txt`.


## 2026-07-19 — ⚖️ THE 30k REFERENDUM (n=5): aggregate PLATEAU CONFIRMED — but the laggard fixed and structure improved underneath

Leg complete: 18,000→30,000, all at real LR (5.5e-5→3e-5 floor), fixed rollouts, α=0.5, λ=0.7 —
~70M tokens, the first properly-powered frontier test. Final probe: **n=5** (tighter brackets,
as pre-registered), uncached/5070. **α=0.0 means: top_share 0.180 / distinct1 0.492.**

Ladder: 8668 **0.16/0.50** → 13,944 **0.17/0.51** → 24,010 **0.207/0.45** → 30,000
**0.180/0.49**. Five probes, one band. **The aggregate verdict is in: at 278M on this corpus,
clean on-policy tokens alone do NOT push the mean past the 8668 frontier — they hold and
polish it.** This is the honest negative the whole week was built to make trustworthy: no
cache defect, no α drift, no LR starvation, n=5 — the plateau is real, not an artifact.

**But the composition under the flat mean moved substantially:**
- **Ibuprofen DE-LAGGED: 0.13/0.61 — best-ever, after four probes stuck at 0.38–0.44.** The
  last stuck-attractor seed is unstuck; the unlikelihood lever loses its motivating case.
- **The "awkward middle" α=0.25 band is uniformly healthy for the first time** (all seeds
  0.10–0.12 top_share) — historically where fragile seeds fell into untrained-vocab salad.
- **Structure gains in text**: fibonacci's best code sample of the project (multi-line Python
  with conditionals + an apt comment); quadratic writes connected math prose ("This is part of
  the quadratic equation… The first one must be found, so we can write…").
- Failure mode shifted from *stuck seeds* to *occasional runaway draws* (weather mean 0.26 is
  one 0.88 sample; 4/5 draws fine).

**Reading:** more same-distribution tokens now buy within-regime polish, not regime change.
Fluency is a solved problem; **meaning does not emerge from more of this corpus at this
size.** Exactly the fork the roadmap's token-curve was built to detect.

**Decision → PLAN-B, as pre-registered:** the teacher-corpus mix A/B
(docs/teacher_corpus_plan.md). Sequence: (1) **harvest** teacher tokens on the now-free Max
(gen_teacher_corpus; ~day-scale for ~30M); (2) continue from 30,000 with
`--teacher-data-ratio 0.2`, total-steps extended, min-lr floor — ONE variable changes (R);
(3) probe after ~8–9k steps vs THIS n=5 baseline. If teacher-text also plateaus, the next
conversations are data curation (phi-style) and the v6 SFT milestone, not more tokens.
Raw: `reports/onpolicy_rollout_probe_30000_cuda_uncached_n5.txt`.


## 2026-07-21 — 🟢 A/B TRIPWIRE @34,500 (half dose): no harm, and the SALAD MODE BROKE on prose seeds

Mid-leg probe of the R=0.2 teacher-corpus A/B (~4,500 steps past the 30k baseline, ~26M
tokens of which ~5M teacher-sourced; LR 3.9e-5, uncached/5070, **n=3** — a tripwire, not a
verdict). Run continues; owner stops it ~16:45 for the n=5 verdict probe.

**Purpose was harm-detection** (the plan doc's flagged risk: teacher text *narrowing* the
distribution). **No harm found** — no homogenization across seeds, no metric blowout.

**Unexpectedly, the metrics moved the right way.** Matched per-seed vs the 30k n=5 baseline
(ts/d1): weather **0.26/0.47 → 0.09/0.57** · bacterial 0.10/0.59 → 0.14/0.57 (only regression)
· diabetes **0.26/0.40 → 0.10/0.51** · ibuprofen 0.13/0.61 → 0.13/0.57 · fibonacci 0.16/0.47 →
0.15/0.48 · quadratic **0.17/0.41 → 0.12/0.38**. **Mean 0.180/0.492 → 0.122/0.513** (top_share
−32%). 3 improved / 2 flat / 1 worse.

**The text is the real find — the FAILURE MODE changed on the two big movers**, which is why
this reads as signal rather than lucky draws (regression-to-the-mean would give different junk,
not different *kinds* of output):
- weather: baseline was **pure digit salad** (`get the 111283667680740138678...`) → now
  connected clauses (*"get the same place, how do we make sure we have the first three
  sections, and I'll give the next year…"*).
- diabetes: baseline was **initials salad** (`C. M.H.P., A. C.A.A.A. / R. C.A.A.S.I.A.A.`) →
  now multi-sentence medical prose (*"These are used for the case of a severe and chronic
  illness. However, many of these problems of increased mental diseases include 34 symptoms…
  The two main factors are the major factors of the patient's health."*).
- bacterial opens well then degrades to `B-P-P-P-P` late (~flat); ibuprofen a modest win
  (baseline dumped code fragments `CQB7 / FLA33 / C17`, now prose-shaped).

**⚠ CORRECTION on full text review (same day): CODE/MATH SEEDS ARGUABLY REGRESSED — and the
metric hid it (top_share inversion, the 8668 lesson in reverse).**
- fibonacci: baseline had recognizable control flow (`if n > 0: … return None … # check if it
  is a linear representation`) → now a degenerate string-concat `print()` + empty docstring.
  Metric flat (0.16→0.15), text worse.
- quadratic: baseline produced equations **with English explanation** (*"This is part of the
  quadratic equation that is obtained by using a linear equation. The first one must be
  found, so we can write the first two…"*) → now pure equation soup, **the prose vanished**.
  Metric "improved" 0.17→0.12 *because* dropping the English lowered repetition.

**Mechanism hypothesis:** the harvest seeds at 40/40/20 general/math/code but the teacher
writes **prose-flavored continuations regardless of seed domain** → an R=0.2 diet may enrich
prose while *diluting* genuine math/code signal. If the verdict probe confirms this split, the
fix is **source-conditional mixing** (teacher text into the general slice only; keep real
documents for math/code) — cheap, since `_MIX_RATIOS` already keys by source.

**Caveats:** n=3 vs an n=5 baseline; half the planned dose. **First α=0.0 text movement since
2026-07-06 on prose seeds — with a possible code/math cost.** The n=5 verdict probe on the
stop-point checkpoint (~36,200) settles both halves.
Raw: `reports/onpolicy_rollout_probe_34500_cuda_uncached.txt`.


## 2026-07-21 (evening) — ✅ A/B VERDICT @36,658 (n=5): teacher data HELPS — and it's a LOWER BOUND

Full R=0.2 dose (30,000 → 36,658; ~6,658 steps, ~39M tokens, ~7M teacher-sourced from the
CONTAMINATED v1 corpus — ~10% license spam, so this result *understates* clean teacher data).
n=5, uncached. **Instrument note: run on the Max (xpu:0), not the 5070** — empirically ~2× faster
for the teacher-in-loop rollout probe (bench: XPU n=3 ~20min vs CUDA ~38min; the teacher's dense
GEMMs are the Max's strength). Backends A/B-validated to ≤0.03 nats (07-15), well inside band
noise, so comparison to the 5070 30k baseline holds.

**α=0.0 per seed (ts/d1): mean 0.180/0.492 → 0.130/0.500.**
- weather **0.26/0.47 → 0.09/0.59 IMPROVED** (digit-salad → *"start building the water at once.
  We had to work on the sun's surface and we can make an event which is the result of…"*)
- fibonacci **0.16/0.47 → 0.11/0.41 IMPROVED (metric)** — but TEXT is worse: baseline had control
  flow (`if n>0: … return None`), now a giant float literal. top_share inversion again.
- quadratic **0.17/0.41 → 0.10/0.47 IMPROVED** — and text genuinely better: baseline was equation
  soup, now *"The problem is presented in this equation: … it is used to solve the following
  problem: 1) The first equation…"* (framing prose returned).
- bacterial / diabetes / ibuprofen: **flat**. Diabetes 0.10 at the half-dose tripwire → 0.27 here
  (wide [0.11–0.49]) — the tripwire gain was an n=3 noise draw, now corrected. (Lesson re-confirmed:
  n=3 within-band ordering is noise; the n=5 is why we run it.)

**Verdict: net positive, modest, real.** Mean top_share −28% (0.180→0.130), distinct1 flat-up.
3 seeds improved on metric / 3 flat / 0 regressed on metric; on TEXT, weather + quadratic are
clear wins, fibonacci is a metric-only artifact (code still degrades — consistent with the v1
corpus's code slice being 57% license spam). **This is the first intervention to move the mean
below the 8668/13,944 plateau floor (0.165) since 07-06 — and it did so handicapped by a
corpus that was ~10% legalese and code-starved.**

**Decision → the data-quality wall is REAL; lean in.** Next leg (owner's call on timing):
re-harvest with the FIXED generator (random-window seeding + boilerplate filter, committed
a7ab0f4) for a clean, properly-code-balanced v2 corpus; OR quick interim test on
`data_teacher_clean/` (5.56M, boilerplate stripped but prose-skewed) at R=0.2 from 36,658.
The clean-corpus A/B is the real confirmation — if code seeds stop regressing with balanced
teacher code, the whole plateau is a data-quality story, not a size story (defers the v6-SFT /
scale-up pivot). Raw: `reports/onpolicy_rollout_probe_36658_xpu_uncached_n5.txt`.


## 2026-07-25 — 🟢 CLEAN-v2 A/B TRIPWIRE @40,002 (~⅓ dose, n=5): healthy, diabetes stabilized, code still dose-limited

The confirming A/B is running: resumed 36,658 on the **clean v2 corpus** (8.65M tok, boilerplate-
fixed + mix-corrected to 40/40/20 + cross-session-shuffled — the three fixes of 07-23/24) at
R=0.2 via `run_ab_confirm.sh`. Stopped mid-leg at **40,002** (~3,344 of 9,342 steps) for the
routine halfway tripwire (catch bugs / divergence before spending the rest of the leg — cf. the
07-16 cached-decode corruption that a probe caught). n=5 uncached, Max, vs the 36,658 baseline.

**α=0.0 (student-alone — the load-bearing signal), new vs 36,658:**
| seed | top_share | distinct1 | read |
|---|---|---|---|
| weather (prose) | 0.09→**0.07** | 0.59→**0.68** | improved |
| bacterial (med) | 0.11→0.13 | 0.58→0.62 | flat/better |
| **diabetes (med)** | **0.27→0.11** | **0.38→0.60** | **big win + range collapsed** (was the noisiest seed) |
| ibuprofen (med) | 0.10→0.09 | 0.57→0.57 | flat-good |
| fibonacci (code) | 0.11→**0.17** | 0.41→0.40 | ts up BUT d1 floor 0.11→**0.27** (near-collapse tail gone) |
| quadratic (math) | 0.10→0.12 | 0.47→0.46 | flat |

**Tripwire = PASS.** Every top_share is well under the ~0.4+ attractor zone; distinct1 healthy
(0.40–0.68) everywhere; no NaN/degeneration/divergence. Diabetes — the baseline's worst,
noisiest seed — is now clean and *stable* (range 0.11–0.49 → 0.06–0.14): a real gain, not noise.

**Code is the dose-limited laggard, NOT a regression.** fibonacci α=0.0 mean ts ticked up
(0.11→0.17) but (a) it's inside the n=5 bands, (b) its distinct1 floor rose 0.11→0.27 (no sample
near-collapses anymore), (c) at α=0.7 fibonacci *improved* (ts 0.11→0.07, d1 0.45→0.57). Classic
dose-limited pattern (prose/medical un-collapse first). At ⅓ dose it's stabilizing, not yet
clearly winning — **the fibonacci α=0.0 mean is THE number the final code-regression verdict
rides on**, and it needs the fuller dose to settle.

**α=0.7 note:** teacher-dominated (least load-bearing) signal; only soft spot is diabetes (range
to 0.31), within noise. Not a concern.

**📊 CONTEXT — best behavioral run on record, at ⅓ dose.** Aggregate α=0.0 top_share across all
6 seeds: **0.115** (40,002) vs 0.130 (36,658 baseline) vs the **~0.165 plateau floor** that stood
since the June regime shift, and below the prior best (8668 "rambling-grammatical English", ~0.16).
**A project low — and the leg isn't done.** Two honesty caveats: (a) part of the drop is the
diabetes seed regressing to mean off a noisy-high baseline draw (0.27, range 0.11–0.49) — though
its range-collapse to 0.06–0.14 is real stabilization, not just a lucky draw; (b) "best aggregate
behavior" ≠ "code regression fixed" — the fibonacci α=0.0 mean is the one dimension still pending
dose. **Strategic read:** the three corpus fixes (boilerplate + mix-correction + cross-session
shuffle) produced the cleanest-behaving run yet — the empirical case that the plateau is a
data-quality wall → lean in.

**Decision → continue the leg tonight toward ~46k, re-probe near the end for the code verdict.**
Nothing says stop. Raw: `reports/onpolicy_rollout_probe_40002_xpu_uncached_n5.txt`.


## 2026-07-25 — ✅ CLEAN-v2 A/B @46,000 (full dose, n=5): CODE REGRESSION FIXED (read the text, not the metric)

Leg finished at 46,000. **The metric first read as a regression — then the text flipped it to a win.**
Recorded honestly because the mis-read nearly sent us on a checkpoint-recovery mission.

**What the top_share numbers said (α=0.0 mean):** 0.130 (36,658) → 0.115 (40,002) → **0.195 (46,000)** —
looked like over-training, with weather/ibuprofen/fibonacci sprouting collapse tails (ranges to
0.76 / 0.73 / **0.98**). I called "over-trained, peak lost." **Wrong read — see the READ-FIRST banner.**

**What the TEXT said (the fibonacci α=0.0 `e.g.`, the actual code verdict):**
- **36,658 (dirty v1):** `if n == 855384299668536778086114999946217867630189635.994633…` — a **giant float literal**, zero structure. The v1 code degeneration the 07-21 verdict flagged.
- **46,000 (clean v2):** `return n` … `def generate_and_add_re_sub_c(n):` … `def calc_sub_str(v): return f"~" + str(` — **real Python: function defs, returns, f-strings.** Nonsense logic, legitimate *structure*.

**Verdict: clean v2 FIXED the code regression** — float-blob → structured code, visible in the text at
both 40,002 and 46,000. The top_share "regression" on fibonacci is the metric penalising valid keyword
repetition (see banner); the wide-range seeds are variance (one collapsed sample of five), not systematic
decay — the typical `e.g.` text at 46,000 reads comparably to 40,002 (ibuprofen ~equal garble, fibonacci
*more* code-like; weather slightly worse, still readable). **"Over-training" is NOT supported by the text.**
The user caught this by reading the outputs against the metric — the exact discipline the banner now mandates.

**Strategic read: the data-quality thesis holds — clean v2 produced structured code where dirty v1
produced float-blobs. Lean in.** (Assisted levels α=0.5/0.7 stayed strong throughout — the student also
generates well *with* the teacher; the α=0.0-vs-assisted gap is the internalisation signal to keep tracking.)

**Process failure logged:** the 40,002 checkpoint (and the 36,658 baseline) were **rotated away** —
`save_checkpoint` kept only `keep_last=3`, and a 9,342-step leg checkpointing every 15 min ate the mid-leg
history. Same bug that lost step 8668. **Fixed** (2026-07-25): `--ckpt-milestone-every` permanently keeps
every Nth-step checkpoint + `--keep-last` raised to 5. 36,658 recoverable from the nvme backup if a re-run
is ever wanted. Raw: `reports/onpolicy_rollout_probe_46000_xpu_uncached_n5.txt`.


## 2026-08-10 — 🔴 SFT CANNOT WORK ON A STRONG BASE: more dose SHARPENS the attractor, exactly as documented

Continuation to **36,202 cumulative SFT steps** — 5.6x v4's known-good ~6.5k, the dose the
roadmap identified as the fix. It is not the fix.

| scored in each model's own format | base (raw) | SFT 3k | SFT 36.2k |
|---|---|---|---|
| code per-sample L3+ @ pen 1.15 | **75.0%** | 0.0% | 0.0% |
| math per-sample L3+ | 5.0% | 1.2% | **0.0%** |
| math copied-from-prompt | 30/80 | 2/80 | 1/80 |

**Monotonically worse with dose**, and the generations regressed rather than progressed on the
same `def add_two(a, b):` prompt:

```
SFT 3k      def def __init__(self, self):   (then doubled quote-blocks and doubled prose)
SFT 36.2k   - - - - - - - - : : : : : | | | = = = conclude:
```

At 3k it still emitted Python-shaped tokens; at 36.2k, punctuation spam. That is the OPPOSITE
of the roadmap's repetition → word-salad → phrases → coherence trajectory. More dose did not
walk it up that ladder; it walked it further down.

### 🔑 THE ANSWER WAS IN `docs/onpolicy_plan.md` THE WHOLE TIME

> *"Every offline divergence we swept **mode-collapses with tokens** into a sharp `is is is`
> attractor. The cause is **exposure bias**: offline distillation only ever sees TEACHER-FORCED
> sequences, so the student never learns to recover from its own trajectories. It is
> **decoupled from every formal metric** (PPL 1.759, ECE 0.0152, stability, reps all healthy at
> the collapsed checkpoint). **More offline tokens — continued OR fresh — only sharpen the
> attractor. The cure is a different objective, not more data.**"*

Every clause replicated tonight:

| documented | observed 2026-08-10 |
|---|---|
| decoupled from every formal metric | training `ce` mean 0.269, `gnorm` 0.15 — loss looks converged |
| mode-collapses into a repetition attractor | `def def def` → `- - - : : :` |
| **more tokens only sharpen it** | 3k → 36.2k strictly worse on both evals |
| the cure is a different objective | ⇒ SFT needs the on-policy treatment distillation already got |

**SFT is a teacher-forced offline objective.** It therefore has precisely the exposure-bias
failure this project already diagnosed and solved for distillation with on-policy/GKD. More SFT
was never going to work, and the document says so in as many words.

**Why June's `v4` was different:** it reached word-salad with more cumulative SFT on a MUCH
weaker base — one with no strong attractor to sharpen. On a base at 98% next-token accuracy and
75% code L3+, extra dose reinforces the attractor instead of building structure. The roadmap's
"more SFT moves the needle" is conditional on base strength. That condition was never stated;
it is now measured.

### The objective is NOT broken — verified, and my alarm was wrong

I flagged the final logged `ce 0.0130` as impossible and predicted a broken loss. Checked on
real batches from the live iterator against the trained checkpoint:

* alignment at masked positions: **0 misaligned** (`target[i] == input[i+1]` throughout)
* independently measured masked CE: **0.534**
* logged `ce`, last 200 steps: mean 0.269, median 0.190, **range 0.010–2.21**

`0.0130` was the *final* value in a distribution spanning two orders of magnitude. I read one
log line as representative of the run. The objective, the masking and the alignment are all
correct — which leaves the documented exposure-bias explanation as the only one still standing.

**One real (minor) bug found:** 32 masked targets are PAD tokens — 0.27% of loss-bearing
positions. Padding should never enter the loss. Far too small to explain anything here.

### Six hypotheses died this session, all to measurement

learning rate · format shock · truncated stop tokens (owner's — killed at 2.9%) · target
off-by-one (the doubling looked exactly like predicting the current token; verified false) ·
**dose** (5.6x made it worse) · broken loss (mine; measured 0.534, inside the logged range).
The surviving answer came from the docs both times the owner pointed at them — the standing
lesson, now on its third instance.

### ⚠️ ONE CONFOUND I DID NOT TEST — effective batch size (added 2026-08-10, arXiv 2412.13337)

The conclusion below was written before reading *"Unveiling the Secret Recipe: A Guide For
Supervised Fine-Tuning Small LLMs"* (Red Hat AI / MIT-IBM, arXiv 2412.13337). Its headline
finding is that **batch size is the dominant SFT lever** — "larger batch sizes paired with
lower learning rates lead to improved model performance", tested at 128 / 3,840 / 7,680
samples per optimizer step.

**Both of our SFT runs were at effective batch 8.** `--micro-batch 1 --grad-accum 8` — the
argparse defaults, never overridden. `sft.py` does not log its args, so this was inferred
from the log: 20.2k tok/s at ~0.40 s/step and seq-len 1024 ⇒ ~8 sequences per optimizer
step. That is **16x below the worst setting the paper tested and ~960x below its best**, and
with mean `resp_frac` **0.179** across 724 logged steps, only **~1,460 tokens per step
actually carried loss**. A very noisy gradient, and micro-batch 1 also badly underuses 48GB.

Their diagnostic points the same way: *"lower gradient norms and higher loss values are
strong indicators of better final model performance."* Ours went the other way — loss fell to
0.04 by step 150 and `gnorm` decayed 0.88 → 0.15.

Confirmed dead by the same paper: **lr 2e-5 consistently outperformed higher rates**, and
2e-5 is what we ran.

**Why this is not a retraction.** The paper's batch-size effect is worth ~0.4 MTBench points
(6.41 → 6.83) on well-behaved 3B–7B instruct bases. Ours is 75% → 0%. An effect that size
does not explain a total collapse, and neither the dose monotonicity nor the text *regressing*
from Python-shaped tokens to punctuation spam is what gradient noise alone predicts. The
exposure-bias reading still fits the evidence better.

**Why it is still worth a night.** Batch is a re-allocation, not extra compute: at fixed
throughput a 6.5h window is ~460k samples regardless of batch, so batch only changes how many
optimizer steps they are sliced into. ⇒ `checkpoints_v8_bigbatch`, effective batch 256
(`--micro-batch 8 --grad-accum 32`), same lr/data, ~2,500 steps. If code L3+ is still 0.0%,
exposure bias stands with batch ruled out. If it moves at all, SFT is salvageable without the
on-policy rebuild. Early read: `ce` staying HIGHER than last night's 0.04–0.11 with moderate
`gnorm` is the paper's better-final-model signature.

### Consequence

**SFT as implemented cannot be used on this base.** It needs on-policy treatment — generate
from the student, correct against the target — the same change that fixed offline distillation.
That is real work, not a flag. The base `checkpoints_newmix/step_0108471.pt` is untouched and
remains the line; the pour resumes there.

Two data defects found on the way, worth fixing before any future attempt:
`clean_code` rejects ~52% of OpenCodeInstruct (the adapter requires EVERY unit test to pass;
median score is 0.90), and sampling ratios are not gradient ratios — loss-bearing token fraction
is 65.9% for `clean_general` vs 12.7% math and 8.2% pubmedqa, so general receives ~5x the
gradient of math at a near-identical sample ratio.

Raw: `reports/code_sft_cont_36202_pen1.15.json`, `reports/math_sft_cont_36202.json`,
`reports/code_sft_cont_34000_pen1.15.json`.


## 2026-08-10 — ❌ SFT @3k COLLAPSED THE MODEL — and the roadmap already said it would

First SFT on the current line. `checkpoints_newmix/step_0108471.pt` → 3,000 steps,
`--data-mix clean`, `--seq-len 1024`. Result: **catastrophic**.

| scored in the format each model was TRAINED on | base (raw) | SFT (chat) |
|---|---|---|
| code per-sample L3+ @ pen 1.15 | **75.0%** | **0.0%** |
| math per-sample L3+ | 5.0% | 1.2% |
| math copied-from-prompt | 30/80 | 1/80 |

Output is token-doubled degenerate prose — `def def __init__(self, self)`,
`The The the the the`, `def def def def def def` — on a prompt of `def add_two(a, b):`.
Code generation is simply gone.

### 🔑 THE CAUSE WAS ALREADY IN THE ROADMAP

> *"v4 (more cumulative SFT) generates varied, domain-relevant word-salad;
> **`small_sft` (one SFT pass) mode-collapses to repetition** — same size, same code."*
> *"`small_sft` was **under-SFT'd — one 3k pass vs v4's ~6.5k**."*

**Tonight was one 3k pass.** That is the exact configuration recorded as mode-collapsing,
and "mode-collapses to repetition" describes `def def def def` precisely. Owner identified
this from memory ("SFT has worked fine in the past… I believe we poured a lot more SFT tokens
than we did tonight"); it was then confirmed verbatim in `docs/roadmap.md`.

Checkpoint sweep 2,000 / 2,500 / 3,000 is **flat at 0.0%** — the damage is present by step
2,000, so there is no good checkpoint hiding inside the first pass. Consistent with the whole
first pass sitting inside the collapsed regime.

### Hypotheses killed by measurement (three of them, none survived)

* **learning rate** (mine) — 2e-5 is the tuned default that produced usable June checkpoints.
* **format shock** (mine) — plausible, but not what the record says.
* **truncated stop tokens** (owner's) — genuinely good hypothesis, since the builder truncates
  at `seq_len` and the `<|im_end|>` teaches the model to halt. **Measured: only 2.9% of
  examples lose their terminator at seq_len 1024** (0.5% at 2048). Not the cause; and it
  removed the reason to stack RoPE extrapolation onto a dose experiment.
* **off-by-one in the SFT targets** (mine) — the doubling looked exactly like predicting the
  current token. **Verified false**: every loss-bearing position teaches next-token correctly
  (`sees 'The' → predict ' answer'`). Good thing to have checked rather than reported.

### Two real defects found on the way, neither the primary cause

1. **`clean_code` rejects ~52% of OpenCodeInstruct.** The adapter requires EVERY unit test to
   pass; median `average_test_score` is 0.90, so most samples have one failure. Confirmed live
   by the run's own diagnostic: `clean_code: 631/1316 (47.9% accept)` against 97.7% / 99.8% /
   100% for the other three. Effective code share **0.22 → ~0.10**. Accepting
   `average_test_score >= 0.9` would roughly double it. **Not fixed mid-experiment.**
2. **Sampling ratios are NOT gradient ratios.** Loss-bearing token fraction differs wildly by
   source — clean_general **65.9%**, clean_math 12.7%, clean_pubmedqa 8.2% — so gradient share
   is ~0.22 / 0.04 / 0.01 against nominal ratios of 0.34 / 0.32 / 0.12. **General gets ~5x the
   gradient of math despite a near-identical sample ratio.** A model trained overwhelmingly on
   Tulu-3 instruction-following is what answers `def add_two(a, b):` with chatty English.

### ⚠ INSTRUMENT: `--chat-template` was REQUIRED to measure this at all

SFT wraps every example with `apply_chat_template`, so the model answers
`<|im_start|>user …<|im_end|><|im_start|>assistant`. Every probe fed raw continuations.
Scoring an SFT checkpoint with raw prompts measures the format it was trained AWAY from —
the same trap as the code framing test. Added to `math_eval` and `code_eval`; must stay OFF
for base checkpoints, which never saw the template.

### In flight

Continuation from `step_0003000` → 40,000 steps (Ctrl-C in the morning), `--micro-batch 8
--grad-accum 1`. **That batch change alone gave 2.2x — 20.4k tok/s vs 9.2k at the same
depth** — because `micro_batch 1` was sized for a 12GB card, not a 48GB one. Same
tokens/step, same math. Expect ~29–33k cumulative SFT steps by 05:30, i.e. 4–5x v4's 6.5k.
Read the spread, not the endpoint: the trajectory to look for is
**repetition → word-salad → phrases → coherence**.


## 2026-08-09 — 📊 FULL VALIDATION BATTERY @108,471: next-token accuracy DOUBLED, code capability 49% → 75%

First run of `run_full_validation.sh` — all eleven runs across nine instruments on one
checkpoint, so the comparisons are matched rather than assembled from partial reads.

### 🔑 THE HEADLINE IS IN THE CALIBRATION TABLE, not the evals

| loop | accuracy @108,471 | was (P0.5 audit, v2, 2026-06-09) |
|---|---|---|
| 0 | **0.696** | 0.140 |
| 1 | **0.896** | 0.500 |
| 2 | **0.962** | 0.510 |
| 3 | **0.980** | 0.499 |

**98% next-token accuracy at the trained depth.** Confirmed independently by `best_exit_probe`:
CE at the trained depth **0.261 → 0.211**, oracle **0.168 → 0.153**. The LM itself improved
enormously; the task evals had not been showing this because they measure multi-token
generation, not next-token prediction.

### Code — the capability number moved a lot

| @108,471 | pen 1.0 (degeneracy) | pen 1.15 (capability) |
|---|---|---|
| per-sample L3+ | 48.8% | **75.0%** |
| char-degenerate | 67/80 | **0/80** |
| looped | 42/80 | **1/80** |

75% runnable at the capability setting vs **49% at 70,500**, with degeneracy fully removed by
the penalty. Per the protocol established 2026-07-30 (report BOTH), this is the number that
counts, and it is the largest capability move in the project so far.

### Math — oscillating, not climbing

| step | L3+ | copied from prompt |
|---|---|---|
| 70,500 | 1.2% | 41.2% |
| 82,711 | 3.8% | **16.2%** |
| 100,000 | 11.2% | 32.5% |
| 108,471 | **5.0%** | **37.5%** |

⚠ **Correction to the 08-06 entry.** The 82,711 copy-rate drop was reported there as the
mix change landing. Across four points it now reads as an OUTLIER, not a trend — copied has
drifted back to 37.5%, near the 41.2% baseline. Math is oscillating around a modest
improvement. The claim "the mechanism metric moved" was made on two points and does not
survive four.

### Depth headroom SHRANK as the model improved

`best_exit_probe`: headroom vs the trained depth **0.093 → 0.058 nats** (agreement 25.3% →
22.5%). A better LM leaves less for a depth router to recover — which independently reinforces
the 08-09 decision to shelve `train_depth_policy` after it made the task worse.

### The head is now WILDLY overconfident at shallow loops

ECE at loop 0 went **0.217 → 0.288**: it reports **1.7% error where the truth is 30.4%**. The
final loop is excellent (0.013). So the direction of the P0.5 finding holds — calibrated where
trained, badly miscalibrated shallower — but the magnitude at loop 0 worsened. Anything that
consumes this signal at shallow depths is consuming noise.

### Instrument/method notes

* **Verify by ARTIFACT, not exit code — the battery scored 2/11 FAILED that had fully
  succeeded.** Both hit the XPU teardown crash (`PyGILState_Release`, field-notes workaround
  #4) *after* writing their JSON and printing their verdict. The training scripts already
  avoid this; the validation script did not. Fixed.
* I printed per-loop accuracy with `r.get('acc', 0)` — the key is `accuracy` — and nearly
  reported **0.000 accuracy at every loop**, which would have inverted the single most
  positive result in the battery. Read the record's keys before reading its values.


## 2026-08-09 — ⚖️ DEPTH POLICY: succeeded on its objective, made the TASK worse (the label is teacher-forced)

`training/train_depth_policy.py`, 6,000 steps head-only on `step_0100000.pt`.
Both halting heads trained from one best-exit label — the loop with the lowest per-loop CE,
taken from the model's own forced-depth trajectory. 411k of 216M params (0.19%); LM body frozen
and verified unchanged.

### It cleared the bar it was given

| | @100,000 | after policy |
|---|---|---|
| head agrees with oracle | 25.3% | **42.4%** |
| CE at the head's choice | 0.5496 | **0.2220** |
| CE at fixed depth 3 *(the bar)* | 0.2606 | 0.2628 |
| CE oracle | 0.1680 | 0.1676 |

The head went from **worse than a constant** to **beating it**, capturing **43% of the
available headroom** ((0.2628−0.2220)/(0.2628−0.1676)). Selection became sensible: it used to
pick loop 7 — the WORST loop, CE 0.72 — 17.1% of the time; now 0.5%, with the mass on loops 3
and 4, the two lowest-CE depths.

**ACTHalting moved too:** mean halt depth **2.00 → 3.60**. The branch that was pinned at
exactly 2.00 across 120 measurements (6 domains × 4 α × 5 samples, zero variance) is adaptive.

### And the task got worse

| | @100,000 | after policy |
|---|---|---|
| math per-sample L3+ | 11.2% | **5.0%** |
| copied from prompt | 26/80 | 28/80 |

⚠ 9 samples vs 4 of 80 — the CIs overlap, so directionally consistent with the mechanism change
rather than conclusive alone.

### 🔑 ROOT CAUSE — the label is TEACHER-FORCED, deployment is AUTOREGRESSIVE

Per-loop CE is measured on gold tokens with a correct prefix. Generation runs on the model's
own output. So the policy was trained to answer *"which loop best predicts the next GOLD token
given a CORRECT prefix"* and then deployed where neither holds. **That is the exposure-bias gap
this whole project exists to close, and the depth label walked into it.**

It also fits the forced-depth null result (2026-07-31): making the model run deeper never
helped, and this made it actually run deeper (2.00 → 3.60) and the task got worse. Two
independent routes to the same conclusion — **depth is not where the capability is**.

⇒ **The fix is a label redesign, not a tuning change:** the best exit must be scored under
GENERATION (roll out from each candidate exit, score the continuation), not teacher-forced CE.
That is real work and is not queued.

⇒ **Pour continues from `checkpoints_newmix/step_0100000.pt`.** `checkpoints_depthpolicy` is
kept as a branch — a genuine result, not a discard. And note `distill.py` retrains BOTH heads
every step (`uncertainty_calibration_loss`, `depth_regularization_loss`) with the old
objectives, so pouring from the policy checkpoint on the standard recipe would overwrite it
within a few hundred steps regardless.

**Method note:** the trainer ran at 0.33–0.40 s/step, ~40 min for 6,000 steps — I had sized the
overnight expecting a head-only step to be slow because of the 8-loop teacher pass. It is not.
Also `| tail -40` in the wrapper buffers until exit, so the run appeared to produce no output
for 40 minutes; only the `--log-file` safeguard made it observable.


## 2026-08-06 — ✅ CORRECTED-MIX POUR @100,000: math capability 9x, and "container before content" is the ACQUISITION ORDER

**483M tokens on the corrected mix** (70,500 → 100,000; lifetime 1.64B, 5.9 tok/param) after
three data fixes landed together on 2026-07-31: medical promoted from harvest-only to 0.20,
math split 0.26 → 0.13 raw + 0.13 OpenMathInstruct-2, code split 0.27 → 0.135 raw + 0.135
OpenCodeInstruct. Verdict: **the mix change worked.** Do not lower the medical ratio.

### Math — the mechanism metric moved, then capability followed

| step | per-sample L4 | per-sample L3+ | copied from prompt | within 10% |
|---|---|---|---|---|
| 70,500 (pre-mix) | 0.0% | 1.2% | **41.2%** | 8/80 |
| 82,711 | 0.0% | 3.8% | **16.2%** | 9/80 |
| 100,000 | 1.2%* | **11.2%** | 32.5% | **16/80** |

`copied_from_prompt` was built to detect the exact @70,500 failure — `12 + 7 =` → `12`,
`10% of 250 is` → `100%`, the quadratic → `0` on all 8 samples. Slot-filling by echo. Adding
task-shaped math **more than halved it**, and L3+ (the correct answer *appears*) is now **9x
baseline** with near-misses doubled. The model stopped echoing and started attempting
arithmetic — wrong arithmetic, but computed.

**\* the L4 is NOT genuine.** Prompt `10% of 250 is`, answer 25, completion
`" 25% of 250. So 10% of 250 is 250."` — it wrote *25 percent*, the scorer took the leading
number. Same coincidence class as the earlier `len(items)` hit. **Genuine math L4 remains 0.**

### Code — genuinely correct implementations, but the primary metric dipped

| code | 70,500 | 100,000 |
|---|---|---|
| per-sample L3+ | **53.8%** | 46.2% |
| task-level L4 | 1/10 | **2/10** |
| char-degenerate | 64/80 | 59/80 |

`double_it` produced `return n * 2` — correct, first statement, no salvage. Both `add_two`
hits are real too. But per-sample L3+ went DOWN; code is roughly flat.

### 🔑 "CONTAINER BEFORE CONTENT" IS THE ACQUISITION ORDER, not a failure

Degeneracy at α=0.0 rose to its worst of the series (15/120 total, mean top_share
0.122 → 0.201). Reading it shows **why**, and it is not collapse:

    x^3 - 4x + 7.   **Step-by-Step Explanation for x: x^3 - 5x + 7.
                    **Step-by-Step Explanation for x + 5x + 7.
    14.9999...  25. Find the general solution y = x y - x.  11. Give...

That is **OpenMathInstruct's format** — step-by-step headers, numbered problem sequences,
LaTeX. The model learned the SHAPE of a worked solution and loops on it. Richer structure to
repeat ⇒ higher `top_share`. The degeneracy rise is a side effect of ACQUIRING the format.

This is the third instance of the same pattern: math answers (learned `So the answer is:` then
empty `\[ \]`), medical abstracts (register + correct associations, invented dosages), now
solution formatting. ⇒ **A format-degeneracy spike is a LEADING indicator that a domain is
landing**, not a warning. And it is the cheap problem: a repetition penalty took looping from
51/80 to 1/80 with L4 unmoved (2026-07-30), so this is decode-time fixable while capability is not.

### Medical — the 82,711 scare fully resolved

| α=0.0 degenerate | 70,650 | 82,711 | 90,499 | 100,000 |
|---|---|---|---|---|
| bacterial infection | 0/5 | **3/5** | 1/5 | **0/5** |
| type 2 diabetes | 1/5 | 3/5 | 2/5 | 3/5 |

Bacterial is at its **best reading in the series**. The 82,711 spike was mid-integration, and
the phrase attractor `"a combination of an infection, infection, or a viral..."` appears at BOTH
70,500 and 90,351 — pre-existing, not introduced by real abstracts.

⚠ **METHOD NOTE — four reversals in one session, all coverage failures.** Reading the
degeneracy COUNT before the text; reading α=0.0 only (30 of 120 samples); comparing one
remembered 70,500 line against a full 5-sample set; and calling a spike a regression from two
points. Each was corrected by widening coverage, never by thinking harder. Full 4-point
side-by-side: `reports/text_diff_pour_a00.md` (1,755 lines).

### Depth policy — precondition STRENGTHENED

`best_exit_probe` @100,000: best exits spread across all 8 depths; headroom vs the trained
depth **26.8% → 35.6%** (0.093 nats, CE 0.261 → oracle 0.168); head agreement flat at
**25.3%** vs 12.5% chance; and the head's own selection (CE 0.550) remains **worse than a
fixed depth of 3**. ⇒ `training/train_depth_policy.py` is a go, and the bar is fixed-depth-3,
not the broken head.


## 2026-07-30 — 🧪 CODE EVAL, FIRST CAPABILITY SERIES (46k→66k): the attractor hid STRUCTURE, not CORRECTNESS

**The headline: L4 (correct code) is ZERO everywhere, and a repetition penalty proves that is NOT a decoding
artifact.** New instrument `tools/code_eval.py` — generate a function, RUN it, check the answer. No judge
model, no interpretation. Graded ladder L0 nothing / L1 syntax / L2 defines / L3 runs-but-wrong / L4 correct,
scored best-of-n per task (pass@k semantics). Raw: `reports/code_eval_*.json`.

**⚠ Instrument correction: L1 and L2 are the same rung by construction.** L1+ equalled L2+ at all 11
checkpoints, and again at n=8 — per-sample L1 was literally **0/80**. Cause is structural, not behavioural:
the *prompt* supplies the `def fname(...):` line, so any parseable prefix already defines the function, while
a prefix short enough to lose the def line does not parse at all (bare `def f(a):` is a SyntaxError) and
scores L0. **Read the ladder as L0 / L1-2 / L3 / L4 and track L3 and L4.** Documented in the tool docstring
rather than silently renumbering, because existing reports use the old labels.

### The trend, and the trap in it

| step | L3+ (n=8) | looped frac | salvage % |
|---|---|---|---|
| 46,000 | 3/10 | 0.81 | 100% |
| 50,000 | 3/10 | 0.89 | 100% |
| 54,000 | 4/10 | 0.80 | 86% |
| 58,000 | 10/10 | 0.56 | 62% |
| 62,000 | 9/10 | 0.70 | 92% |
| 64,000 | 10/10 | 0.69 | 82% |

**The trap: `_truncate_to_parseable` SALVAGES a runnable stub out of a looping generation.** `if n == 0:` /
`return 0` repeated forever truncates to a function that defines, executes and returns the wrong answer —
i.e. scores L3 without the model ever finishing a function. At 64k, **77–87% of all L3 scores were salvage**
(`max_line_repeat >= 3`), and per-sample **62.5% of generations produced nothing parseable at all**. A first
reading of the n=3 sweep as "+2.7 L3-points/10k steps, the first positive capability trend" was **wrong in
kind**: salvage fraction averages 0.87 across the series and only drifts −0.11/10k steps.

**What the series does support:** looping falls (0.81 → 0.69) and salvage falls (1.00 → 0.82), so the L3 climb
is not purely a truncator artifact. ⇒ **tokens are buying FLUENCY and STRUCTURE; correctness has not moved.**
Same shape as the medical verdict below — vocabulary/format generalises, facts and logic do not.

### The decisive experiment: repetition penalty @64,000 (n=8, T=0.4)

| penalty | looped | median rep | per-sample L0 | per-sample L3 | **L4** |
|---|---|---|---|---|---|
| 1.0 (off) | 51/80 | 5 | 70% | 20% | **0/80** |
| 1.15 | **1/80** | 1 | 22% | 49% | **0/80** |
| 1.3 | **0/80** | 1 | 30% | 44% | **0/80** |

The penalty does exactly what it should — looping goes to ~zero, unparseable output drops 70%→22%, runnable
code more than doubles. **And L4 stays at zero at every setting: 240 generations with the loop provably
removed, not one correct function.** ⇒ **The capability is not hiding behind decoding. The L4 wall is
capacity/training, not sampling.**

**The single L4 in the entire study is itself an artifact.** 62k `count_items` — a *looped* generation
(`return len(items)` ×4, `max_line_repeat=4`) that passes only because `len(items)` happens to be the correct
answer for that task. Not capability.

### Consequences

- **❌ Unlikelihood training / DiverseKD comes OFF the queue.** It was scheduled to attack the attractor. It
  would work — and buy nothing for correctness, because a free *inference-time* penalty already removes the
  looping and L4 does not move. Don't spend training compute there.
- **✅ Strengthens the token pour AND `grow_width.py`.** L4 is a capacity wall; both queued items attack it.
- **Eval protocol going forward: report BOTH** — penalty 1.15 as the capability number, penalty 1.0 as the
  degeneracy number. A penalised score is *not* comparable to the unpenalised 46k→66k series.
- **NEW observation — chat-template leakage on bare-code prompts.** With looping suppressed, failures look
  like `if not a:\n\treturn True\nelse:\n\tprint(a)` then `<|im_end|>` and English meta-commentary. The model
  drifts into *chat mode* on a raw-completion prompt. Harvest-formatting question, cheap to check against the
  corpus; not yet investigated.

### Instrument notes (method, so the numbers stay readable later)

- **Best-of-3 is too noisy to read checkpoint-to-checkpoint.** Residual sd about the n=3 fit was 1.73 vs 1.55
  expected from pure task sampling — i.e. the scatter is almost entirely sampling. **A 66,000 reading of L3+
  3/10 vs 64,000's 7/10 looked like a medical-mix regression and was NOT**: at n=8 the same pair reads 8/10 vs
  9/10, differing on one task. **The medical mix is exonerated; that hypothesis is withdrawn.** Use n=8.
- **Repetition penalty is COUNT-SCALED, and penalises generated tokens ONLY.** Plain CTRL (flat divisor,
  Keskar et al.) measurably fails against a *sustained* loop — verified on a fake model with argmax pinned to
  one token: it breaks the loop for exactly one step then reverts, because once the alternative is also seen
  both are divided equally and the attractor still wins. Scaling by occurrence count makes pressure escalate.
  Penalising the *prompt* would suppress `a`/`b`/`n` — the very tokens a correct body must reuse — and
  manufacture a failure we would misread as model weakness.
- **Completions are now saved to the JSON.** The first version kept only the rung and 110 truncated chars, so
  every new question cost another card run. All re-analysis above was done offline from saved output.


## 2026-07-30 — 🩺 MEDICAL VERDICT @64,000 (full dose): buys FLUENCY, not FACTS — as the capacity math predicted

Full 6,000-step leg with medical in the teacher stream (29.4% of the pool, ~5.9% of
training tokens). **The mirror image of the 58k leg:**

| domain group | 58k → 64k (α=0.0) |
|---|---|
| **medical** | 0.210 → **0.120** ⬇ big improvement |
| trained (general/math/code) | 0.110 → **0.137** ⬆ worse |
| mean α=0.0 | 0.147 → **0.127** ⬇ |
| mean α=0.25 | 0.115 → **0.137** ⬆ (the 60,154 record of 0.102 did NOT hold) |

**Structure improved on every medical seed** — the degenerate modes are gone:
- bacterial: 58k letter-soup (`A. C. B. / F / 1. C. F. C.`) → grammatical prose with
  real register: *"a common type of **pathogen-based antibiotic** … may be one of the
  better treatment"*
- diabetes: 58k `disease…disease…disease` loop → structured numbered questions
- ibuprofen: still opens with invented chemical IDs, then drifts to meta-commentary

**But facts are absent, and confidently wrong where present.** Bacterial says *"the most
common type of **cancer**"* for an infection prompt. Diabetes **asks questions** instead
of listing symptoms (the prompt is "symptoms include___"). Ibuprofen never states what
it treats.

**⇒ VERDICT: medical data removed the medical degeneracy modes; it did not add medical
knowledge.** This is the empirical confirmation of the capacity analysis (roadmap): at
278M the memorisation bound is ~90–170M tokens' worth, against ~3–4M medical tokens
actually seen. **Facts do not fit in these weights at any token budget** — that is an
architecture fact, and it is the quantitative case for the retrieval tier. Further
medical harvest buys diminishing returns versus spending the tokens elsewhere.

Caveat: the domain-level swap (medical up, trained down) mirrors the 58k leg in the
opposite direction, which at n=5 with ±0.01–0.02 scatter is partly noise rather than a
real trade-off. Raw: `reports/onpolicy_rollout_probe_64000_xpu_uncached_n5.txt` ·
`reports/collapse_metrics_64000_xpu_aligned.txt`.

## 2026-07-30 — ❌ λ SWEEP: λ is NOT a throughput lever (~4%, not the predicted 45%)

Ran λ=0.7 (control, complete 64,000→66,000) vs λ=0.4 (test, stopped at 64,454) from an
identical 64,000 branch point, differing only in `--onpolicy-lambda`.

**Premise was wrong.** The hypothesis: offline steps run ~4.7× faster than on-policy
ones, so lowering λ should buy throughput (predicted λ=0.4 → 1.45×, cutting a 3B run
from 31 to 21 days). **Measured: ~4%** (5.90 → 6.16 steps/min over comparable
training-only windows, seed-copy checkpoints excluded).

**Why the model was wrong — two mechanisms, both λ-independent:**
1. **The rollout buffer refills on a STEP SCHEDULE** (`rollout_buffer.needs_refill(step)`),
   not per-λ-draw. λ gates which *loss* a step uses; it does not gate the expensive
   generation, which keeps running at its own cadence.
2. **Checkpoint I/O**: 3.3 GB written every 15 minutes, entirely independent of λ.

**⚠ AND A MEASUREMENT TRAP WORTH KEEPING — the logged `k tok/s` is NOT wall-clock
progress.** It is computed as `micro_batch*grad_accum*seq_len*log_every / dt` over the
**last 5 steps only** — instantaneous stepping throughput, excluding buffer refills and
checkpoint writes. It therefore reads ~50% high. Under λ=0.4 the log looks *much*
better (steady 2.5–2.8k vs λ=0.7's swinging 1.0–4.7k) because the slow on-policy dips
disappear — but total progress barely moves. **Compare arms by steps/min from
checkpoint timestamps, never by the logged tok/s.** (Three wrong speedup figures were
produced before this: 1.45× from the bad model, then 1.01× and 1.21× from bad
measurement windows — one using only the last two checkpoints, one including the
*copied* seed checkpoint whose mtime is the `cp` time, not a training time.)

**⇒ Do not lower λ for throughput.** 4% does not justify trading away on-policy dose,
which is what cured the exposure bias. **The real, untested throughput levers are
`--rollout-reuse`** (currently 2 — raising it directly halves generation work) **and
`--ckpt-every-mins`** (3.3 GB every 15 min). Both are λ-independent.

*Salvage: `reports/onpolicy_rollout_probe_66000_lambda07_n5.txt` is a clean, fully
probed 2,000-step λ=0.7 baseline at 66,000, usable for any future comparison.*

## 2026-07-29 — 🟢 MEDICAL-BLEND TRIPWIRE @60,154 (~⅓ dose, n=5): healthy, and α=0.25 is an ALL-TIME LOW

First leg with medical in the teacher stream (blend: 8.65M general/math/code +
3.61M medical = **29.4% medical**, ~5.9% of training tokens at R=0.2). Stopped at
60,154 of the 64,000 target for the routine mid-leg tripwire.

**α=0.25 hit the lowest top_share on record** — 0.102, beating the previous best
(40,002's 0.107) across the entire history back to 30,000, with `distinct1` holding
at 0.55 (not bought by going repetitive). Every seed improved or held.

| α | 58,000 → 60,154 |
|---|---|
| 0.0 | 0.147 → 0.142 (flat) |
| **0.25** | 0.115 → **0.102 ← all-time low** |
| 0.5 | 0.092 → 0.118 (worse — likely noise; both neighbours improved) |
| 0.7 | 0.112 → 0.092 |

**The α-SHAPE is the finding, not the α=0.0 number.** A *quarter* of teacher
steering now reaches near the model's best; going to 0.5 adds little. That is the
precise form of "needs less help": **not** that α=0.0 caught up (the α=0.0-vs-α=0.7
gap is still ~+0.04 and NOT closing), but that the required nudge shrank.

**selfrep resolved a metric disagreement.** The new `tools/relevance_probe.py`
showed mean 4-gram self-repetition RISING (52k 0.000 → 58k 0.017 → 60k 0.063),
apparently contradicting the α=0.25 record. Per-seed breakdown settles it — it is
**one seed**: diabetes 0.246, everything else ≤0.043. And the looping *moved*
seeds (58k it was quadratic at 0.087; now quadratic is clean at 0.036 and diabetes
broke). So: five of six seeds clean, one acute marker-spam failure.

**Trained domains look structurally richer in text despite flat metrics:**
- **fibonacci:** 58k = one arithmetic blob → 60k emits a `return`, a `# comment`,
  a *second* `def f(n)`, a markdown fence, and explanatory prose.
- **quadratic:** 58k's confused "roots of the first line" → 60k produces real
  algebra (`4x^2 - 6x + 6 = ...`) plus process language ("first step in simplifying").

**Medical unchanged at this dose** — still fluent vocabulary with no facts, and
diabetes actively degraded. Expected: ~1.2M medical tokens had been seen. The
medical question needs the full leg.

**Method note — a reading correction worth keeping.** The weather α=0.0 output was
scored three different ways in one session: "better (on topic)", then "worse (mere
prompt echo)", then finally the owner's reading — **it performs the deliberative
function the prompt asks for** (propose → compare → justify → reconsider: *"This
might be better than… but you have to… because… But no, if…"*). The prompt is
*"we decided to___"*, so weather words are incidental; doing decision-work is the
signal. ⇒ **The right unit of analysis is DISCOURSE FUNCTION, not topic keywords
and not degeneracy metrics.** Also: "the text wins over the metric" is NOT
unconditional — it holds where the metric is structurally blind (valid code
repeats keywords), but not when the metric is measuring the very thing that is
wrong (repetition). Raw: `reports/onpolicy_rollout_probe_60154_xpu_uncached_n5.txt`.

## 2026-07-29 — 🧪 TEACHER-EXHAUSTION GAUGE, first real series (46k→58k): rises, but CONFOUNDED

`tools/kd_exhaustion.py` measures the curriculum's most direct Rung-3 gate — soft-KL
on a fixed held-out sample — continuously and without the `top_share` inversion problem.
First real run (every 2000, 8×4×512 = 16,384 held-out tok, cached so the series is comparable):

| step | soft-KL | Δ |
|---|---|---|
| 46,000 | 2.2451 | |
| 48,000 | 2.2180 | −0.0271 |
| 50,000 | 2.2100 | −0.0080 |
| **52,000** | **2.2687** | **+0.0587** ← mix change lands here |
| 54,000 | 2.3165 | +0.0478 |
| 56,000 | 2.2636 | −0.0529 |
| 58,000 | 2.3216 | +0.0580 |

mean **2.2634**, half-to-half drift **+0.0682**, step-noise (rms) **0.0460**.

**Reading: soft-KL RISES at ~1.5× the scatter — but that is NOT evidence of teacher
exhaustion.** Two confounds survive, and both must be cleared before this number
informs anything:
1. **Mix change inside the series.** Values *fall* 46k→50k, then jump **+0.0587 at
   52,000 and stay elevated** — exactly where the uniform mix took effect (code
   20%→33%, general 40%→34%). The only clearly non-random feature sits on a
   training-distribution change. **Compare within one mix regime.**
2. **The on-policy caveat (the important one, unresolved).** At λ=0.7, ~70% of steps
   train on the student's OWN rollouts, not offline web text — so drifting away from
   the teacher's *web-text* distribution is EXPECTED, not exhaustion. The gauge may
   simply be pointed at the wrong distribution for an on-policy run.

**Discriminating test, not yet done:** measure soft-KL over **student rollouts** instead
of web text. Rollout-KL falling while web-KL rises ⇒ caveat 2 confirmed, re-point the tool.

**Verdict logic corrected in the same pass.** The first version printed a confident
"FLATTENING → GROW" off two half-series slopes; on this data that fired on scatter plus a
confound. It now compares half-to-half drift against point-to-point noise, prints
INCONCLUSIVE when |drift| < noise, and on a rise explicitly demands the two confounds be
ruled out. *Under-claiming is the right failure mode for a signal feeding a capital
decision.* **Do not use this alone for grow-vs-graduate** — cross-check the α=0.0 vs α=0.7
gap, which at 58k is still NOT closing (~+0.03–0.04), the capacity-limited signature and
currently the more trustworthy of the two. Raw: `reports/kd_exhaustion_46k_58k.txt`.

*(Process note: the run also confirmed the artifact-based orchestration fix — the gauge hit
the XPU interpreter-teardown crash AFTER writing all seven measurements, and the chained
medical harvest started anyway. Under the old exit-code check that crash would have
silently skipped phase 2.)*

## 2026-07-28 — ✅ @58,000 UNIFORM-MIX DOSE TEST: code improves, and the win is DISTRIBUTIONAL not attractor-removal

Leg 52,000 → 58,000 with `_MIX_RATIOS` made **uniform** (code 20%→33%, general 40→34,
math 40→33) to test the owner's hypothesis that code was the *dose-limited* laggard.

**Rollout probe (n=5, sampled) — the hypothesis holds, and the split by DIET is the story:**
| domain group | share of diet | 52k → 58k α=0.0 mean |
|---|---|---|
| general / math / code | 100% | **0.127 → 0.110 ✅** |
| medical | **0%** | 0.150 → **0.183 ❌** |

The headline mean *rose* (0.138 → 0.147) — **entirely dragged by untrained medical.** Split by
diet and the picture inverts. Standouts: **fibonacci 0.12 → 0.10 with the range collapsing
[0.06–0.27] → [0.09–0.11]** (the bimodal blow-up tail gone — the number-blob mode stopped
firing), and **quadratic 0.18 → 0.12 despite LOSING share** (40%→33%).

**⚠ THE KEY REFINEMENT — greedy reconciles it: this is DISTRIBUTIONAL improvement, NOT attractor
removal.** The greedy sweep at 58k shows the attractors *unchanged*: `sorted(` still collapses to
`list(list(list(…` byte-identically, hard-collapse totals flat (9 vs 8), fibonacci arguably
*worse* under greedy (`# # # #` comment spam vs 52k's `def n(n)` repetition). Meanwhile two
prompts genuinely improved — `import numpy` went `get_num_of_num_of_num…` → **`def
get_data_from_data_file(data_file):` + a real docstring**, and `binary_search` went
docstring-only → **`if not isinstance(val, str): return False`** (real control flow).
**Both readings are true: the extra dose moved probability mass AWAY from degenerate paths
without deleting the paths.** The model falls in less often; the holes are still in the floor.
⇒ **More data reduces how often you hit an attractor; it does not remove it.** Removing
`list(list(` and the giant-number mode needs a *targeted* fix (unlikelihood training) or far
more scale — not another 6k steps. Do not remember this leg as a bigger win than it was.

**The medical list-marker soup was SAMPLING NOISE, not an attractor.** The alarming bacterial
α=0.0 output (`A. C. B. / F / 1. C. F. C. / 2. / I / [L / G / T`) has **zero** occurrences under
greedy — deterministic decode never reproduces it. One unlucky draw of five. *Method note: this
is exactly what greedy is for — it settles "real attractor vs sampling artifact" definitively,
and it's why both instruments are kept.*

**Medical degradation — mechanism (owner's read).** Medical has **no dataset at all**; whatever
ability existed rode in on **fineweb-edu spillover**, and this leg cut general 40%→34% while
raising code (zero medical content) 20%→33%. The only tributary shrank. Under *greedy* medical is
comparably bad at both checkpoints (phrase-looping), so this shows in the sampled probe only —
softening but not refuting the crowding-out read. ⇒ Medical capability **cannot be maintained
incidentally**; it needs real medical seed data (**curriculum Rung 1**), which is also direct
mission progress.

**Parity gate (α=0.0 vs α=0.7) — NOT closing.** Across 36,658 / 46,000 / 52,000 / 58,000 the gap
sits ~+0.03–0.04 with the distinct1 gap persistently negative (teacher-steered stays more
varied). Per the curriculum asymmetry that is the **capacity-limited** flavour, not exhaustion →
the teacher still has transferable signal, and growth (Rung 2) is indicated before graduation
(Rung 3). Raw: `reports/onpolicy_rollout_probe_58000_xpu_uncached_n5.txt` ·
`reports/collapse_metrics_58000_xpu_aligned.txt`.

## 2026-07-26 — 🔬 @46,000 DOMAIN-ALIGNED SWEEP (collapse_metrics, greedy vs T=0.8): artifact-vs-real split

Broadened the student-only greedy probe to domains that match training + mission: **general / math /
code / medical / science**, 8 each (40 core) + chat/qa OOD (`--probe-set all`, expanded 2026-07-26).
Ran it **greedy** and **T=0.8 top_k=40** on 46,000, so greedy-vs-sampled separates decode-artifact
from real degeneracy. (Bug found + fixed en route: sampled `--generate` on XPU segfaulted — `topk`/
`multinomial` are on the XPU-segfault list; now samples on CPU. commit 94906d5.)

**Greedy baseline (harsh — argmax loops):** 33/40 core not-degenerate; 7 hard collapses in exactly TWO
attractors — **`**` markdown** (transformer, chest-pain, Newton) and **giant-number spam `100000…`**
(sum, probability, speed-of-light, DNA). Code strongest (7/8 valid Python). Dominant non-collapse mode
is grammatical phrase-looping ("the concept of the concept…").

**Sampled (T=0.8) splits artifact from real:**
- **`**` + phrase-loops → BREAK** under sampling (chest-pain, Newton, binary_search escape to varied
  text). ⇒ **decode artifacts; the distribution underneath is healthy.**
- **Giant-number attractor → PERSISTS** (sum, probability, transformer still fall into digit-spam under
  sampling). ⇒ **a REAL degeneracy pocket on numeric-answer prompts** — a genuine weakness (candidate for
  the unlikelihood-training lever / better math data), not something sampling or tokens alone clears.

**New diagnostic — CJK leakage = a diffuseness/undertraining marker (owner's catch).** Sampled runs emit
occasional Chinese characters (e.g. inside a code docstring: `""" 臰天…`). Mechanism: the Ouro tokenizer
is **multilingual**, but the student trained **English-only** (FineWeb-Edu / open-web-math / codeparrot /
English teacher text), so it never learned to *suppress* the CJK tokens — they keep residual probability.
Greedy never shows them (argmax is always the confident English token); **top-k sampling reaches the tail
wherever the distribution is flat/uncertain**, so CJK surfaces exactly at the model's least-trained points.
⇒ Chinese-in-output is a **marker of local distribution diffuseness / undertraining**; it should recede as
training sharpens the distribution. Weaker domains → flatter → more leakage (the domain-level version of
"lack of tokens on that subject").

**Medical + science (mission/generalisation, NEW sets):** 7/8 and 6/8 not-collapsed; they **speak the
domain vocab** (infection, disease, blood pressure / chemical, atoms, elements) but produce **no facts** —
grammatical themed filler. Correct for domains **not distilled on yet**: vocabulary generalises, facts
don't. This is the **pre-medical-SFT baseline**; not-collapsing is the win, facts come with the SFT phase.

**Two instruments, and why keep both** (see the READ-FIRST banner): greedy collapse_metrics = harsh
diagnostic that EXPOSES attractors + the untrained tail; sampled ≈ real use. Greedy-vs-sampled is now the
standard way to tell an artifact from a real weakness. Raw: `reports/collapse_metrics_46000_xpu_aligned.txt`
(greedy) + `…_aligned_t08.txt` (sampled).


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
