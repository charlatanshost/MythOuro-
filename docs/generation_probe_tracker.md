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

## Longitudinal eval trajectory (n=320, bare framing, T=0.4 pen=1.15 seed=1234)

Every n=320 code eval on record, in step order. **Slot each new checkpoint in
here rather than comparing it to one predecessor** — the bands below are what
"normal" looks like, and most single-checkpoint deltas fall inside them.

| step | L3+ | L4 | L0 | note |
|---|---|---|---|---|
| 140,000 | 59.1% | 2/320 | 10% | |
| 141,500 | 77.8% | 6/320 | 6% | |
| 143,500 | 76.6% | 5/320 | 6% | α-anneal carry-forward |
| 146,500 | 53.4% | 4/320 | 8% | codemix |
| 149,500 | 54.1% | **30/320** | 10% | codemix — top L4, bought with L3+ in the low 50s |
| 151,238 | **78.8%** | 17/320 | 4% | rollout256 — top L3+ on record |
| 157,238 | 54.7% | 26/320 | 31% | |
| 163,238 | 77.5% | 19/320 | 7% | mathcode END — prose REGRESSED here |
| — exit_pdf lineage — | | | | |
| step 0 (= 278M base) | 68.8% | 9/320 | 15% | bit-exact control |
| @4,200 | 77.2% | 4/320 | **2%** | **lowest L0 on record** |
| @7,200 | 65.0% | 14/320 | **4%** | |

**Bands across the whole lineage:** L3+ **53–79%**, L4 **1–30**, L0 **4–31%**.

⇒ **L3+ has never separated an intervention** — its 26 pp band matches the
13.6 pp checkpoint sd measured 2026-08-31. **L4's two highest values (30, 26)
both came with L3+ in the low 50s** — the trade. **L0 is the discriminating
metric**, and exit_pdf holds the two lowest values ever recorded without paying
for them in L3+.

## Qualitative trajectory — same seed, α=0.0 vs α=0.25, chronological

The practice: read the SAME seed at the SAME α down the lineage. Metrics say
where to look; this says what changed. Seed: *"The treatment for a bacterial
infection usually involves"*, sample #0.

**α=0.0 (pure student) — scaffold → fragments → loop → sentences**

```
157,000 base     'immuno ¡ … A: The following statements … A: The following is
                  a sample … C: The standard name of the test group'
161,500 mathcode 'immunoconduct … • Bilateral • C. of the same, • E. of the'
163,238 mathcode '… - CABI (cascipronic) - CABI (cascipronic) x4'
grown48 masked   'the treatment of the skin problem and the presence or more
                  often a specific type of infection.'
exit_pdf @7,200  'immunoconductduct … Other research is needed to study the role
                  of antibiotics in infectious-type viruses.'
```

The base was not producing prose on this seed at all — `A:` / `A:` / `C:` list
scaffolding with no content. The new lineage produces sentences.

**α=0.25 — a stock-phrase tic disappears**

`"antibiotics and antibiotics"` opens 157,000, 161,500 AND 163,238 — stable
across 6,000 steps of the old lineage. It appears in **neither** new checkpoint.

**⇒ The trend that matters: the α=0.0 → α=0.25 GAP HAS CLOSED.** At 157,000 the
difference between unaided and teacher-assisted was fragment-soup vs coherent
prose. At exit_pdf @7,200 both are coherent. The student now produces
near-assisted quality unaided — which is what on-policy distillation is for, and
the first time it is visible in the text rather than inferred from a metric.

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


## 2026-08-27 — ⚠️ LENGTH AND CORRECTNESS ARE ANTI-CORRELATED; leg 3 was net harmful; keep step_0157238

A third leg on the SAME 92M-token corpus (157,238 → 163,238, identical config)
reversed both gains, and measuring the middle of it overturned the framing of the
previous three days.

### Solution length rose, peaked, then collapsed — and L4 moved OPPOSITE

| step | | ≥2 stmt | mean | committed | **L4/320** |
|---|---|---|---|---|---|
| 143,500 | base | 3% | 1.03 | 25% | 5 |
| 157,238 | leg 2 end | 28% | 1.35 | 42% | **26** |
| 159,500 | +2.3k | **50%** | **1.60** | 33% | **2** |
| 161,000 | +3.8k | 38% | 1.50 | 24% | 4 |
| 163,238 | leg 3 end | 4% | 1.05 | **64%** | 19 |

**At 159,500 the model wrote multi-statement solutions half the time — the best
that figure has ever been — and correctness COLLAPSED to 2/320.** At 163,238 it
returned to one statement and correctness recovered to 19/320 with `committed` at
64%, also the best ever.

⇒ **When this model writes longer code, it writes WORSE code.** The
multi-statement rise reported 2026-08-26 (3% → 15% → 28%) tracked L4 by
coincidence of which corpus was added, not because longer solutions were better
ones. "Get `body_stmts` off 1" was the wrong target: 1.6 statements of wrong code
is not closer to correct than 1.0 statements of right code. The 2026-08-26 entry's
framing is corrected by this.

### Prose degraded MONOTONICALLY — a separate failure, harmful from step one

| step | α=0.0 top_share | distinct1 | salad hits |
|---|---|---|---|
| 157,238 | 0.150 | 0.484 | **0** |
| 160,000 | 0.197 | 0.456 | 0 |
| 163,238 | **0.242** | 0.427 | **1** |

No turning point — straight decline across the leg. The **acronym salad returned**
on the bacterial seed, the exact degeneracy mode the α-anneal eliminated:

> 157,238: *"- A medical condition like the diagnosis of the infection. - The
> primary cause of the disease."*
> 163,238: *"antibiotic treatment, such as C.P. A treatment method includes the
> following components: - CASE - CABI (cascar - CCAO - CACA - CACI - CCA - CACF
> - CSC - CAST - CDA - CCI - CCA - CCA - CCA"*

Weather drifts off-topic into programming; ibuprofen picks up number fragments
(*"1. 9682 -0.01 m2"*). **The owner asked for the outputs to be read rather than
the metrics; the metrics were pointing the same way this time, but the salad is
only visible in the text.**

### ⇒ KEEP step_0157238. More steps on an unchanged corpus UNDO its gains.

157,238 is best on L4 (26/320), best on prose (0.150, zero salad) and holds 28%
multi-statement. Nothing after it improves that combination.

⚠ **AND IT WAS ALREADY GONE.** `--keep-last 5` rotated `step_0157238.pt` out
during leg 3 — it is not a multiple of the 500-step milestone interval, so it was
never protected. **Third time this class of bug has cost a checkpoint** (step
8,668; steps 40,002/36,658; step 0108471 earlier this month). Its MEASUREMENTS
survive as JSON, so nothing in this entry is unverifiable — but the weights are
not recoverable.

`step_0157000.pt` (238 steps earlier) survives and is copied to
`checkpoints_base/` as the stand-in. **It has NOT been evaluated** — treat it as
presumed-equivalent until measured, not as the checkpoint the table above
describes.

⇒ Non-milestone checkpoints are never safe. Copy anything worth keeping to
`checkpoints_base/` the moment it is identified, not after the next leg.

**A 92M-token corpus supports roughly 6,000 steps before it starts hurting** —
leg 2 (151,238 → 157,238) gained, leg 3 (157,238 → 163,238) lost. That ratio is
the number that matters for sizing the teacher corpus of the 1B run, where the
same mistake costs months instead of a night.

⇒ The next move is NEW SOURCES (PubMedQA, the 13.75M unused OpenMathInstruct
pairs), not more steps on these 92M tokens. And the gate should be **L4 and
prose**, not `body_stmts` — which this entry demotes from headline metric to
diagnostic.


## 2026-08-26 — 🎯 THE FROZEN METRIC MOVES: multi-statement solutions 3% → 28%, L4 5 → 26/320, monotone across two corpus additions

The teacher stream was the constraint, not the objective. Expanding it moved
CONTENT on two independent metrics, monotonically, in the same direction.

### The result

| checkpoint | teacher corpus | 1-stmt | **≥2 stmts** | body mean | **L4/320** | L3+ |
|---|---|---|---|---|---|---|
| anneal base @143,500 | continuation 10.9M | 97.1% | **3%** | 1.03 | **5** | 76.6 |
| codemix @149,500 | + code 35M | 85.0% | **15%** | 1.20 | **14** | 76.9 |
| **mathcode @157,238** | + math 58M (92M total) | **72.0%** | **28%** | **1.35** | **26** | 54.7 |

All n=320. **One-statement solutions fell 97% → 72%; multi-statement rose 3% →
28%, nearly 10x.** A four-statement solution appeared for the first time in
project history. `body_stmts` had been median 1 at EVERY α and EVERY checkpoint
ever measured — the α ladder, the loop-weighting arm, every SFT attempt.

Tasks solved keep changing, and getting harder:
* base — `add_two`, `double_it`
* codemix — `is_even`, `reverse_string`, `max_of_two`
* **mathcode — `add_two` 5, `sum_list` 4, `count_items` 2, `reverse_string` 15**

`sum_list` requires accumulation, not a one-liner. That is a different class of
task than anything solved before 2026-08-24.

### What it cost, and why it is worth it

L3+ fell 78.8 → 54.7 (raw). Same trade the two codemix legs showed: correctness
and completeness up, parseable fragments down. Given L3+ counts guard clauses
that never answer — `if not nums: return 0` graded L3 for `sum_list` — this is
the right side of the trade.

### Math did NOT regress

4.37% L3+ at n=320 against 11.25% at n=80. Intervals [2.2, 6.6] vs [4.3, 18.2] —
they overlap. Math has been ~4-6% all along; **the 11.25% was n=80 noise**, the
same inflation that made every code headline 5-6pp too high last week. The
oscillating series (1.2 / 3.8 / 11.2 / 5.0 / 11.25) is consistent with a flat ~5%
read at ±7pp. Adding 58M math tokens has not yet moved math itself — but it moved
CODE solution length, which is what multi-step math CoT demonstrates.

### Cross-domain probe @157,238 — medical IMPROVED, and top_share missed it

Six-seed rollout probe, α=0.0, against @140,000 (pre code+math):

| seed | @140k | @150k | @157k | distinct1 | halt |
|---|---|---|---|---|---|
| weather | 0.100 | 0.152 | **0.094** | 0.571 | 2.00 |
| bacterial | 0.179 | 0.140 | **0.085** | 0.594 | 2.00 |
| diabetes | 0.194 | 0.081 | 0.154 | 0.475 | 2.00 |
| ibuprofen | 0.081 | 0.081 | 0.144 | 0.573 | 2.00 |
| fibonacci | 0.246 | 0.100 | 0.260 | 0.448 | 2.00 |
| quadratic | 0.125 | 0.100 | 0.160 | 0.246 | 2.00 |

prose-only top_share 0.139 → 0.119. **Acronym-salad hits: 0** — no re-collapse.
`halt` is still exactly 2.00/4 on every seed, as it has been on every checkpoint
ever measured.

**The metrics are mixed and the TEXT is not.** Both medical seeds were DEGENERATE
at 140,000 and are coherent prose at 157,238:

* bacterial @140k: *"including: \* Infection, Infection, and other medical
  conditions … 100000000000000000000000000000000000"* → @157k: *"- A medical
  condition like the diagnosis of the infection. - The primary cause of the
  disease. - The patient's infection can be managed…"*
* diabetes @140k: *"- Lateral-drying - High-fat - High-fat - Low-fat - High-fat
  - High-fat - Low-fat - Low-fat…"* (a pure repetition loop) → @157k:
  *"- High-temperature: This is the most common cause of the disease and most
  often associated with cardiovascular conditions like asthma…"*

diabetes read 0.194 → 0.154 on top_share — a mild move — while the text went from
a repetition loop to structured prose. **⇒ Adding 92M tokens of code and math did
not damage the mission domain; it improved its FORM.** Content is still wrong
(high temperature is not a diabetes symptom), which is the project-wide pattern.

### ⇒ The diagnosis that was right: TOKEN STARVATION IN THE TEACHER STREAM

The 80% base stream is unlimited HF text. The 20% teacher stream — the part
carrying the distillation signal — was `data_teacher_v2` + `data_teacher_med` =
**10.9M tokens, one epoch every 3,339 steps**, re-read dozens of times by step
140,000. Every lever tried for a fortnight (α, λ, loop-weighting, dose, rollout
window, corpus format, eval budget) reshapes the sampling DISTRIBUTION; none of
them add signal. Two corpus additions did what none of those could.

Current: **92M teacher tokens, one epoch every 28,229 steps** — a 6,000-step leg
is 0.21 epochs, nothing re-read.

**Headroom is large and cheap.** OpenMathInstruct-2 has ~14M pairs and 250k are
used. OpenCodeInstruct has 200k configured, 100k used. PubMedQA (100k,
provenance-clean, the mission domain) is untouched. All convert with
`tools/make_code_corpus.py --source <name>`, no GPU, routed through the
`sft_data.py` adapters so the Tulu-3 subset filter is never bypassed.

⚠ Tulu-3 is held: median 0 statements / 81% at one or fewer (chat prose, wrong
shape), AND its WildChat filter returned 0 drops on 2,000 rows where ~10% was
expected — unresolved, and it is the one source with card-verified OpenAI
contamination.


## 2026-08-24 — ✅ TASK-COMPLETION DATA MOVES CONTENT (raw), ❌ chat framing broken by `--rollout-len 64`

Two legs on OpenCodeInstruct converted to teacher-corpus shards
(`tools/make_code_corpus.py`), 143,500 → 149,500, 6,000 steps, α=0.45, ratio 0.2
(~0.56 epochs). All numbers n=320.

### The result that stands: RAW framing

| checkpoint | L3+ | committed | L4 | tasks solved |
|---|---|---|---|---|
| base 143,500 | 76.6 ±4.6 | 25% | 5/320 | add_two, double_it |
| attempt 1 (no think block) | 54.1 ±5.5 | 43% | **30/320** | add_two, is_even, reverse_string, max_of_two |
| **attempt 2 (with think block)** | **76.9 ±4.6** | **42%** | **14/320** | is_even, reverse_string, max_of_two |

**Content moved for the first time all year.** Against the base, attempt 2 holds
L3+, nearly doubles `committed` (25% → 42%) and roughly triples L4 (5 → 14/320,
non-overlapping intervals). `reverse_string` — never solved once in anything
previously measured — is solved 7-13 times depending on the leg. Four nights of
α tuning never touched L4; one night of task-completion data tripled it.

Attempt 1 vs 2 is a real trade, not noise: attempt 1 has higher L4 (30 vs 14,
barely-separated intervals) across four tasks; attempt 2 has far better L3+
(76.9 vs 54.1). The closed `<think></think>` block bought parseability and cost
some correctness. Both are kept — `checkpoints_codemix_nothink/` and
`checkpoints_codemix/`.

Median `body_stmts` is still **1**, and every newly-solved task is a one-liner
(`s[::-1]`, `max(a,b)`, `n % 2 == 0`). Multi-statement tasks (`fibonacci`,
`is_prime`) remain unsolved. The corpus moved correctness on tasks that fit the
single statement the model can produce; it has not yet taught it to produce more.

### Chat framing: 0.0%, and THREE hypotheses died explaining it

Both legs, both frames: chat L3+ **0.0%**, `<think>` in **320/320**.

1. **❌ The corpus lacked a closed think block.** The harvested `data_teacher_chat`
   prefills one (`gen_teacher_corpus --no-think`); the converter omitted it.
   Fixed and re-run → attempt 2 is *still* 320/320 think-locked. Not the cause.
2. **❌ `--max-new 96` was too small.** A think block measured ~94 tokens median,
   so the budget looked exhausted before any answer. Re-run at **512**: still
   0.0%, all 320 at rung 0. The think block simply **expands to fill the budget**
   — 94 tokens at a 96 cap, **495 tokens at a 512 cap**. The cap was never the
   cause; the model reasons until it is stopped.
3. **❌ Closing the block is sufficient.** At 512, **48/320 DID close** — and only
   5/320 produced a code fence, with all 320 still scoring L0. Exiting the
   reasoning does not produce an answer.

### ⇒ The live lead: `--rollout-len 64` has never covered the answer

Every training run in project history uses `--rollout-len 64`. That is the
on-policy correction window: the student generates 64 tokens, the teacher
corrects them. Under chat framing the model's first 64 tokens are entirely
INSIDE a think block that runs ~495 tokens.

**So the on-policy objective has never once trained the transition out of
reasoning, or the answer that follows it.** It only ever trains "how to reason
for 64 more tokens." That accounts for both observations: reasoning expands
without limit because reasoning is the only thing on-policy training shapes, and
closing-then-answering was never in the correction window.

⚠ **The 2026-08-17 λ refutation does NOT cover this.** That probe compared the
RATE of opening `<think>` (64/80 at λ=0.7 vs 68/80 at λ=0.2) — not its length,
and not whether it closes. Different quantities; the relevant one was never tested.

**Test:** `--rollout-len 256` or higher, enough to span think + answer. Costs
~4x the rollout compute, so a night buys proportionally fewer steps — pick the
size deliberately rather than by guess. This is a TRAINING-side change; the
corpus and the eval have both now been ruled out.

### Instrument fix

`max_new` was absent from every report ever written, so no historical chat number
recorded which generation cap produced it. Now recorded (`d4e1590`). A generation
cap is part of the result.


## 2026-08-22 — ✅ n=320 SETTLES IT: the α-anneal is REAL (+18pp), and every n=80 number this week was inflated

Four metrics had picked three different "best" checkpoints, which is the signature
of measuring noise. Re-run at `--samples 32` (320 per checkpoint, same seed so the
first 8 per task reproduce the existing runs exactly):

| step | α | L3+ (n=320) | committed | restart | L4 |
|---|---|---|---|---|---|
| 140,000 | 0.50 | **59.1 ±5.4** | 31% | 20% | 2/320 |
| 141,500 | 0.45 | **77.8 ±4.6** | 48% | 28% | 6/320 |
| 143,500 | 0.45 | **76.6 ±4.6** | 25% | 16% | 5/320 |

**The anneal effect is real and large.** 140,000 vs either anneal checkpoint gives
cleanly separated intervals, +18pp. This is the first difference measured this week
that survives proper power, and it validates the 0.5 → 0.45 anneal as a genuine
capability move rather than a metric artifact.

**141,500 and 143,500 tie on L3+ and differ elsewhere, in opposite directions.**
Both gaps are non-overlapping at n=320:
* `committed` **48% vs 25%** — 141,500 finishes an answer twice as often
* `restart` **28% vs 16%** — 143,500 abandons and redefines half as often

Neither dominates. L4 is 6/320 vs 5/320 (indistinguishable), and even against
140,000's 2/320 the direction is ~3x but not significant at this n.

### ⚠ EVERY n=80 NUMBER THIS WEEK WAS INFLATED BY 5-6pp

| checkpoint | n=80 said | n=320 says |
|---|---|---|
| 143,500 | 82.5% | **76.6%** |
| 140,000 | 65.0% | **59.1%** |

The ORDERING held; the magnitudes did not. "82.5%, a new project record" was an
n=80 artifact — the true figure is 76.6%, and 141,500 matches it. **±10pp intervals
cannot resolve 8-20pp differences.** Comparisons that decide anything should run at
`--samples 32`; `--samples 8` is for a quick look, not for ranking checkpoints.

### Why the anneal helped FORM and not CONTENT

Everything that moved across the project is form: char-degeneracy 67/80 → 0/80,
restart 79% → ~20%, `committed` 12% → ~48%, acronym salad gone. Nothing has moved
content: strict L4 has been 0-2/80 since step 64,000 and median `body_stmts` is **1
at every α and every checkpoint ever measured.**

That is not a coincidence. α, λ, loop-weighting and dose all reshape the
DISTRIBUTION the student samples from; none of them add task competence. The model
trained almost entirely on CONTINUATION data — codeparrot files, teacher
continuations of web text — so it learned what code looks like and produces one
plausible statement, because nothing in its training signal ever rewarded finishing
a task. `is_even` is the only task it reliably solves because the answer fits in
the single statement it knows how to produce.

⇒ **The anneal has done its job and the ladder is closed.** The base is now
0/80 degenerate, salad-free, and restarts a quarter as often as at 64,000 — the
conditions the chat-mix legs lacked when they failed on an unstable base. The
26,130-row instruction corpus is the only asset that demonstrates task COMPLETION,
and it is the next move. Carry **step_0143500.pt**: it is weaker on `committed`,
which is exactly what instruction data attacks, and stronger on `restart`, which
data cannot easily fix.


## 2026-08-21 — 📈 `committed`: the metric that shows the anneal working, 12% → 50% across the project

Owner's read, which turned out to be right and which neither L3+ nor L4 could
show: the outputs are *relevant, coherent and on-subject — they just never reach
the end*. Measuring it directly confirms both the failure mode and the trend.

### The failure is UNFINISHED, not wrong

At step 143,500, of the 66 samples graded L3+:
* **67% returned `None`** — fell off the end of the function
* 33% returned a wrong value
* 0% raised

Median `body_stmts` is **1 at every checkpoint and every α measured.** The model
writes one statement and stops, and that statement is frequently the CORRECT
opening: `if not nums: return 0` is the right first line of `sum_list`. It is a
correct prefix, not a wrong answer.

Neither existing rung can see this. Falling off the end raises `AssertionError`
exactly like a wrong answer, so both land on **L3**; and both are simply "not
correct", so both land off **L4**.

### `committed` — returned a value on valid inputs

| step | α | L3+ | **committed** |
|---|---|---|---|
| 64,000 | 0.50 | 49% | **12%** |
| 70,500 | 0.50 | 68% | **44%** |
| 108,471 | 0.50 | 75% | **46%** |
| 120,000 | 0.50 | 60% | **31%** |
| 125,181 | 0.50 | 55% | **32%** |
| 140,000 | 0.50 | 65% | **30%** |
| **141,500** | **0.45** | 76% | **50%** ← project high |
| 142,500 | 0.45 | 80% | **48%** |
| 143,500 | 0.45 | 82% | **28%** |
| 149,500 | 0.40 | 59% | **24%** |

**12% → 50% across the project.** Two things fall out of the shape:

1. **The 120k–140k plateau at ~30% coincides exactly with the 116k–120k
   disruption.** `committed` dropped from 46% to ~30% and sat there for 20,000
   steps. The α=0.45 anneal did not merely recover it — 50% is above the
   pre-disruption 46%.
2. **It anti-correlates with L3+ at the end of the 0.45 leg.** 141,500 is 76%
   L3+/50% committed; 143,500 is 82% L3+/28% committed. The checkpoint we called
   "best" on L3+ is the WEAKEST of the three at finishing an answer. **141,500 is
   the better checkpoint**, and L3+ pointed at the wrong one.

### Caveats, stated up front

* **Body length never moved.** One statement at every α. What improved is whether
  that statement RETURNS rather than guards. Tasks needing several statements
  (`fibonacci`, `is_prime`) remain out of reach, which is why `is_even` — solvable
  in one line — is the only task that ever grades correct.
* `committed` does not mean right. Strict L4 is still 0–2/80 everywhere.
* n=80 per point.

### Now reported natively

`_PROBE` in `tools/code_eval.py` calls each function once on valid inputs inside
the EXISTING runner subprocess (before the assertions, so a failing check does not
lose the result). Reports carry `committed` and `committed_frac`, and the run
prints it next to the ladder. `tools/regrade_strict.py` recovers it from saved
completions for any historical report.


## 2026-08-21 — ⚠️ CORRECTION: the α-anneal moved PARSEABILITY, not correctness. L3+ is not a capability metric.

The 2026-08-20 entry below calls 82.5% raw code L3+ "a new project record". That
framing is withdrawn. Re-grading the SAVED completions (`tools/regrade_strict.py`,
no regeneration, `score_sample` reused unchanged so only the TESTS differ):

| step | α | L3+ (runs) | L4 shipped | **L4 STRICT** |
|---|---|---|---|---|
| 120,000 | 0.50 | 60.0% | 1/80 | **0/80** |
| 125,181 | 0.50 | 55.0% | 2/80 | **2/80** |
| 140,000 | 0.50 | 65.0% | 0/80 | **0/80** |
| 141,500 | 0.45 | 76.2% | 0/80 | **0/80** |
| 142,500 | 0.45 | 80.0% | 2/80 | **2/80** |
| **143,500** | **0.45** | **82.5%** | 0/80 | **0/80** |
| 145,500 | 0.40 | 58.8% | 3/80 | **2/80** |
| 147,500 | 0.40 | 60.0% | 1/80 | **1/80** |
| 149,500 | 0.40 | 58.8% | 3/80 | **2/80** |

**L3+ and correctness are uncorrelated.** The highest-L3+ checkpoint (82.5%) solves
NOTHING; the lowest (58.8%) solves two. Across 30,000 steps and three α values,
strict L4 stays between 0 and 2 of 80 — counting noise. The α ladder moved L3+ by
27 points and moved correctness not at all.

**What the 82.5% actually is.** Reading the completions behind it: `add_two` ->
`if a == b: return 0`; `sum_list` -> `if not nums: return 0`; `get_first` ->
`return sorted(items)`; `double_it` -> `n = int(n)` followed by a comment about
binary search. Guard clauses and topic drift — parseable fragments that do not
answer the question. 66/80 graded L3 on exactly this.

**The shipped tests also over-count.** They are 1-3 assertions per task. `is_even`
was two assertions on two values, so `return (n % 2 == 0) and (n % 3 == 1)` passed
and was graded L4 at 149,500. Three shipped L4s were false positives (double_it
@120,000, is_even @145,500 and @149,500). `_STRICT` in the re-grader adds edge
cases — zero, empty, negative, identity — because guard-clause fragments are
exactly what thin tests wave through.

**What still stands from the 08-20 entry:** the α=0.0 prose result. Prose-only
top_share went 0.139 (α=0.5) → 0.092 (α=0.45) → 0.108 (α=0.40), the letter-salad
mode is gone from the text, and that was the first movement in α=0.0 since step
90,351. Owner's read of the α=0.0 outputs — clearly better than pre-120,000 — is
consistent with both the numbers and the text. **The anneal did something real; it
just was not code correctness.**

⇒ **Do not run another α leg to chase L3+.** No α setting moves L4 off the floor,
which makes this a capability wall, not a tuning problem. Report L4 STRICT
alongside L3+ from here, and treat L3+ as a degeneracy/parseability proxy only.

## 2026-08-20 — ✅ α-ANNEAL 0.5 → 0.45: RAW CODE 65.0% → 82.5%, a new project record, in 3,500 steps

The second α-anneal in the project's history, and the most productive single leg
so far. 140,000 → 143,500 (~4.9h), one variable changed.

### The result

| step | RAW code L3+ | looped | char-degen |
|---|---|---|---|
| 140,000 (α=0.5 endpoint) | 65.0% | 1/80 | 0/80 |
| 141,500 | 76.2% | 1/80 | 0/80 |
| 142,500 | 80.0% | 1/80 | 0/80 |
| **143,500** | **82.5%** | 1/80 | 0/80 |

Monotone, and it passed the previous all-time record (**75.0% @108,471**) after only
1,500 steps. Chat-framed code was flat — 33.8% → 31.2%, inside the ±10pp interval —
so this is a raw-frame gain, not a frame trade. Degeneracy never moved.

Whole-project arc for raw code: **75.0 → 60.0 → 55.0 → 65.0 → 82.5.** The
116k–120k disruption is not merely recovered but beaten by 7.5 points.

### The trigger, and why it fired

The documented 2026-06-30 rule: capability present at high α but NOT internalized
into α=0.0, with α=0.0 flat. It had been flat for **50,000 steps** —
0.161 (90,351) / 0.201 (100,000) / 0.170 (108,471) / 0.155 (125,181) / 0.154 (140,000)
— where the 0.6→0.5 anneal had moved that same metric 0.18 → 0.12 in 216 steps.
`onpolicy_plan.md` already named 0.45 and shelved it under "HOLD 0.5, pour TOKENS";
the pour reached its 140,000 target, so the condition the hold depended on was met.

### ⚠ THREE INSTRUMENTS, THREE DIFFERENT VERDICTS — the mean was the worst one

α=0.0 per seed, @140,000 → @143,500 (top_share):

| seed | before | after | |
|---|---|---|---|
| weather | 0.100 | 0.080 | ✓ |
| bacterial | 0.179 | 0.100 | ✓✓ |
| diabetes | 0.194 | 0.100 | ✓✓ |
| ibuprofen | 0.081 | 0.090 | ~ |
| fibonacci | 0.246 | **0.370** | ✗ |
| quadratic | 0.125 | 0.180 | ✗ |
| **MEAN** | **0.154** | **0.153** | **flat** |

1. **The six-seed MEAN said the anneal did nothing** (0.154 → 0.153). The
   readout script's own verdict logic would have printed "FLAT — α is not the
   lever, go to rung 6." It was wrong.
2. **Per-seed said prose up, code down.** Prose-only across four seeds:
   top_share 0.139 → 0.092 (−34%), distinct1 0.543 → 0.603 — the FIRST movement
   in α=0.0 since step 90,351. The text agrees: the `C-C-C and C-C-C` letter-salad
   mode is gone and continuations hold their topic. Meanwhile `fibonacci` went
   from a repeated-`sum(n-1)` loop to a giant float literal.
3. **Deployment-settings eval said capability UP, a lot** (65.0 → 82.5).

The reconciliation: the rollout probe samples at T=1.0 / top_k 50 / **no repetition
penalty**, far harsher than `code_eval` at pen 1.15. The α=0.0 code degeneracy is
real at worst-case sampling and does NOT reach deployment behaviour. Had we stopped
at either the mean or the rollout probe we would have concluded the anneal failed or
cost code capability. **Neither aggregate nor worst-case sampling substitutes for
measuring capability at the settings that will be used.**

### Loss did NOT behave as the precedent predicted, and it did not matter

Per-500 means: 0.6567 → 0.6717 → **0.6886** (peak) → 0.6372 → 0.6093 → 0.5937 →
0.6133 → 0.6080. The 2026-06-30 note said to expect a RISE and treat it as good.
It rose for ~1,000 steps and then fell well below the α=0.5 baseline (~0.69).
Neither reading was informative — lowering α changes what the loss is computed
over, so it is not comparable across the change. **Judge an anneal on α=0.0 and on
capability, never on loss.**

### Next

α=0.40. The trigger that justified 0.45 no longer applies in the same form (α=0.0
prose has finally moved), but the justification is now stronger — results, not a
plateau. Watch the same abort signal: a fragile seed spiking past 0.40 at α=0.0.
`checkpoints_newmix` remains the untouched α=0.5 control and `step_0140000.pt` the
fallback; `checkpoints_base/` holds 108,000 / 116,000 / 125,181.


## 2026-08-18 — 🔀 PHASE TRANSITION at ~116k–120k: the pour is converting a CONTINUATION model into a CHAT model

Four instruments, run in one evening on the newmix pour (108,471 → 125,181). Two
of them appeared to contradict each other for about an hour. They don't — the
disagreement WAS the finding.

### The measurements

**Raw-framed code (pen 1.15), the headline metric all year:**

| step | L3+ | looped | char-degen |
|---|---|---|---|
| 108,471 | **75.0%** | 1/80 | 0/80 |
| 120,000 | 60.0% | 6/80 | 0/80 |
| 125,181 | **55.0%** | 1/80 | 0/80 |

**Chat-framed code (pen 1.15, `--extract`), same weights, same prompts:**

| step | L3+ | `<think>` |
|---|---|---|
| 108,471 | **6.2%** | 51/80 |
| 116,000 | 10.0% | 62/80 |
| 125,181 | **37.5%** | 56/80 |

**Next-token CE at the trained depth** (`best_exit_probe`, all measured in ONE
session — see the reproducibility note below):

| step | mean CE | headroom |
|---|---|---|
| 108,000 | 0.348 | 0.153 |
| 112,000 | **0.280** | 0.106 |
| 116,000 | **0.286** | 0.108 |
| 120,000 | 0.494 | 0.242 |
| 125,181 | 0.561 | 0.274 |

**Soft-KL to the teacher** (`kd_exhaustion`, 27 checkpoints, 72,000 → 124,000, on
the CACHED held-out sample from 2026-07-29 so it is comparable to that series):
mean **2.1604**, half-to-half drift **−0.0881**, step-noise rms 0.0665, sign-flips
19/25. In the 116k–124k window the largest step is +0.0375 — against a 0.0665
noise floor, and against the +0.0587-and-stays-elevated signature the real
2026-07-27 mix change produced at step 52,000. **No break.**

### The reading: it is a phase transition, not damage

Everything after ~116,000 moves together — CE on continuation corpora up, chat
capability up 3.75x, raw-frame code down — while soft-KL to the teacher keeps
falling straight through. That last fact is what resolves it. The teacher is
**Ouro-2.6B-Thinking, a chat model**. Converging on it MEANS getting worse at raw
continuation and better at chat. All four instruments agree once you stop treating
raw-frame code as "the" capability number.

Supporting evidence from the text itself: 42/80 raw-framed completions now emit
`<|im_end|>` unprompted and open `<think>`, with no chat data anywhere in this run.

**⇒ The headline metric has been measuring the frame the model is LEAVING.** A
local coding assistant is used in chat frame, not by continuing a function body.
On the metric that matches the goal the pour delivered **6.2% → 37.5% with zero
chat data** — after three SFT attempts and two chat-mix legs produced nothing.
That also settles rung 9 for good: `<think>` went 51 → 62 → 56 (non-monotone)
while chat capability went 6.2 → 10.0 → 37.5. The marker was never the mechanism.

### What did NOT move

* **Math is flat.** pen-1.0 series across five points: 1.25 / 3.75 / 11.25 / 5.0 /
  11.25 — 11.25% was already reached at step 100,000. Overlapping intervals, no
  trend. Also: math scores **5.0% at BOTH pen 1.0 and pen 1.15** at 108,471, where
  code moves 48.8% → 75.0% across the same two settings. The penalty rescues code
  and does nothing for math: two different failures, not one.
* **Correctness.** L4 is 2/80 at 125,181. L3+ means "produces runnable Python" —
  a sample graded L3 for `add_two` was `a = 2 + a`, which runs and does not add.
  The failure mode has shifted from DEGENERATE to FLUENT-BUT-WRONG (pen-1.0
  char-degenerate 67/80 → 53/80), which reads as competent output and is not.
* **Halt depth is pinned at exactly 2.00/4** on every sample of every domain —
  math, medical, general, all four rollout α values. ACT is not making a poor
  routing decision; it is making no decision at all.

### ⚠ Instrument note — `best_exit_probe` is within-session only

Its eval text STREAMS from remote HF datasets. Re-running step 125,181 reproduced
to **+0.0012** mean (per-domain |Δ| ≤ 0.0043), so it is deterministic within a
session. But step 108,000 measured 0.348 here where an archived run at 108,471 —
471 steps away — had recorded 0.2036. **Cross-session absolute CE from this tool
is not comparable; only curves measured in one sitting are.** A cross-session
comparison was used earlier in the evening to claim "the pour degraded the model
1.7–3.5x"; that claim was withdrawn. `kd_exhaustion` does not have this problem —
it caches a fixed held-out sample — and is the tool to prefer for any series.

### ⚠ step_0108471.pt was rotated away — the THIRD time this class of bug has cost us

The trainer's own `--keep-last 5` deleted it: 108,471 was a resume checkpoint, not
an even-2000 milestone. `prune_checkpoints.py` never touched it — `checkpoints_newmix`
is in its `_PROTECTED` list — so the protection everyone trusted was guarding a
door the deletion never came through. Precedents: step 8,668, then 40,002/36,658
(which is what `--ckpt-milestone-every` was added to fix). Milestones survived, so
`step_0108000.pt` (471 steps away) stands in, and every baseline MEASUREMENT
survives as JSON.

**Mitigation applied:** `checkpoints_base/` now holds 108,000, 116,000 and 125,181
outside the rotation. 116,000 (best CE, pre-transition) and 125,181 (best chat,
post-transition) are the two most important checkpoints on the project and were
both one rotation cycle from deletion.

### Attribution, because it matters for how the next call gets made

The owner's read was correct throughout and was argued against three times:
"looking good, just a little noisy" (chat capability was rising and the series
genuinely is noisy — 19/25 sign-flips), "read the outputs on all subjects, not
just code" (which is what surfaced the pinned halt depth and the fluent-but-wrong
shift), and "new material caused a spike" (there IS a real, reproducible step
change at 116k–120k).

Refuted this evening, all mine: that the CE curve showed smooth on-policy dose
drift (it is a step change); that the pour was degrading the base (frame-relative);
and that teacher exhaustion would show as rising soft-KL (it fell). The exhaustion
result was reported as "your hypothesis refuted" when it was a refutation of my own
operationalisation of it.


## 2026-08-15 (later) — ⚡ HARVEST 3.6x FASTER: 1.4B teacher + measured max_new/batch

Chat harvest went 24 → 86 accepted tok/s across two changes, both measured rather
than estimated.

**1. `--max-new 512 --batch 30` (1.9x).** The gap to the continuation harvest
decomposed exactly — computed 4.3x against an observed 4.9x:

    continuation  30 lanes x 768 kept x 0.67 /  768 steps = 20.1 tok/step
    chat @1024    18 lanes x 374 kept x 0.71 / 1024 steps =  4.6 tok/step

Three multiplicative causes: `--max-new 1024` forced batch 18 instead of 30 on
memory, instruction answers keep only 374 of 1024 decoded tokens, and it ran a
third more steps. Swept against the ACTUAL accepted-length distribution (p50 320,
p80 554, p90 709): 512 covers 77% at batch 30. Took 512 over the faster 384/37
because 384 keeps only the shortest 62% and bakes a selection bias into the
corpus.

**2. Ouro-1.4B-Thinking as the harvest teacher (a further 1.9x).** Config is
identical to the 2.6B except DEPTH: 24 layers vs 48, same hidden 2048, 16 heads,
16 KV heads, intermediate 5632, vocab 49,152, 4 UT steps. Half the layers means
half the kernel launches (decode is launch-bound) and half the KV cache.

### Quality held — measured, not eyeballed

`tools/compare_corpora.py`, 781 rows vs 2,545:

| | 2.6B | 1.4B |
|---|---|---|
| tok/s | 45 | **86** |
| grounding median | 0.462 | **0.473** |
| grounding p10 | 0.280 | 0.268 |
| copy run median / p90 | 5 / 13 | **4 / 10** |
| answer words median | 123 | 97 |
| malformed | 0 | 0 |

**Grounding** = fraction of the answer's content words that appear in the source
passage. It is the direct measure of the failure the continuation harvest had
("the PAWL study", 30-mg ibuprofen against a real 200-400mg). Comparable at
+0.011. **Copy run** guards the other side — a corpus that parrots the passage
scores perfect grounding and teaches nothing — and the 1.4B copies LESS.

⚠ **Grounding measures TOPICALITY, not correctness.** It proves the teacher is not
drifting off the passage; it cannot prove the answers are right. Spot-read still
required before training.

### A false-fail that cost a run

The first 1.4B attempt ran at **3 tok/s** because
`teacher KV-cache validation FAILED (greedy token mismatch at step 6)` silently
dropped it to full recompute — O(n²) instead of O(n). The gate tested
greedy-argmax equality BEFORE looking at KL and raised immediately, discarding the
number that would explain the rejection. Six tokens after the `</think>` prefill
is exactly where continuations are near-equally likely, and in bf16 a near-tie
flips argmax while KL stays negligible — the failure mode
`harvest_speedup_plan.md` already documented. The gate now reports KL and the
reference top-2 gap; the second run PASSED at max KL 2.68e-03. **Nothing was
loosened — it still fails closed, it just says what it saw.**

⇒ At 86 tok/s a 12-hour night yields ~3.7M tokens, against 1.0M before. The
chat-mix leg failed on DOSE (10 epochs over 0.96M tokens); this is what makes a
corpus large enough for the question to be askable.


## 2026-08-17 (later) — λ HYPOTHESIS REFUTED; L3+ has a real U-shape; L4 is UNMEASURABLE at n=80

### 1. ❌ The on-policy teacher is NOT what stops the model answering

Ran the chat leg at `--onpolicy-lambda 0.2` (chat-mix used 0.7), then evaluated
the MATCHED-STEP control that already existed as a milestone
(`checkpoints_chatmix2/step_0109500.pt`, λ=0.7 at 1,029 steps):

| arm | RAW | CHAT | `<think>` |
|---|---|---|---|
| base (0 chat steps) | 75.0% | 6.2% | 51/80 |
| λ=0.2 @1,000 | 72.5% | 2.5% | 68/80 |
| **λ=0.7 @1,029 (control)** | 41.2% | 7.5% | **64/80** |
| λ=0.7 @3,029 | 68.8% | 0.0% | 79/80 |

At matched steps λ=0.7 gives **64/80** against λ=0.2's **68/80** — four samples
apart, inside noise, and pointing the WRONG way. The `<think>` rate tracks CHAT
EXPOSURE alone: 51 at 0 steps → ~66 at 1k → 79 at 3k. **Hypothesis refuted.**

⚠ The probe was designed badly and the control is what saved it: I ran 1,000 steps
against chat-mix's 3,000 "to bound repetition risk", changing two variables at
once and destroying attribution. The matched cell existed for free as a
milestone — worth checking for one BEFORE spending a leg next time.

### 2. ✅ L3+ has a REAL U-SHAPE — endpoint evals hide it

    step     108471   109500   110000   110500   111000   111500
    L3+       75.0%    41.2%    53.8%    60.0%    66.2%    68.8%
              base     trough  ————— smooth monotone recovery —————

Chat training sharply disrupts code capability early, then recovers steadily. The
endpoint reads −6.2pp; the model actually passed through **41.2%**. Six points,
monotone after the trough — this is a training dynamic, not sampling noise.

⇒ **Every conclusion in this tracker drawn from a single endpoint eval is
under-informed.** Milestones are already saved; evaluating three of them costs
~20 minutes and no training.

### 3. 🔬 L4 IS NOT MEASURABLE AT n=80 — and I over-read it

    L4 per 80:  1 → 4 → 1 → 3 → 2 → 0

No trend. At a 2-5% rate with n=80 the 95% CI is ~±5pp, so 1/80 and 4/80 are
indistinguishable. On seeing 4/80 at the trough I wrote that it was "the most
correct code we have measured" and that L3+/L4 were anti-correlated — the
milestone sweep killed both claims. Correctness genuinely was produced
(`return a + b`, `return n * 2`, `return sum(nums)`), but the RATE cannot be
resolved by this instrument.

⇒ Any future claim about L4 needs `--samples` in the hundreds, not 8. Until then
**L4 is a existence check, not a metric** — "it can sometimes write correct code"
is supportable; "it writes more correct code than X" is not.


## 2026-08-17 — CHAT-MIX RETRY: dose theory CONFIRMED on capability, REFUTED on behaviour

3,000 steps from `step_0108471` on the 26,130-row corpus (~7.3M tokens) —
**1.35 epochs** against the first attempt's 10.3. Training healthy: `n_loops` 4
throughout, `gnorm` median 0.52, loss 0.777 → 0.715.

|  | RAW | CHAT (extracted) | `<think>` | fence | `<\|im_end\|>` |
|---|---|---|---|---|---|
| base 108,471 | **75.0%** | **6.2%** | 51/80 | 25 | 41 |
| chatmix 1M / 10 epochs | 50.0% | 2.5% | 80/80 | 3 | 1 |
| **chatmix2 7.3M / 1.35 epochs** | **68.8%** | **0.0%** | 79/80 | 3 | 2 |

### ✅ Dose was the right diagnosis for CAPABILITY LOSS

Cutting epochs 10.3 → 1.35 cut raw code loss from **−25pp to −6.2pp**, roughly
proportional. The corpus-size work (2 harvest nights, 1.4B teacher switch, the
3.6x throughput fixes) did exactly what it was meant to.

### ❌ Dose is NOT the cause of the "never answers" behaviour

`<think>` appears in **79/80** chat-framed samples at 1.35 epochs, against 80/80
at 10.3. A 7.6x dose reduction moved it by ONE SAMPLE. Fences and `<|im_end|>`
stay at 3 and 2. The model opens a reasoning block, writes fluent coherent prose
about the problem, and never emits code — `adj_degenerate 0/80`, so this is not
degeneracy, it is *never finishing*.

**The base already does this in 51/80 with no chat training at all.** So the
corpus does not create the behaviour; it amplifies an existing one from 64% to
99%.

### 🔬 HYPOTHESIS: the on-policy teacher is fighting the corpus

Every corpus row is `<think>\n\n</think>\n\n` + answer — an EMPTY reasoning
block. Training on it should teach "open, close immediately, answer". Instead the
student opens and rambles. Simultaneously the same leg runs
`--onpolicy-lambda 0.7` against **Ouro-2.6B-Thinking**, which reasons at length by
default. So the corpus teaches the MARKER while the on-policy KL teaches what to
put INSIDE it, at strength 0.7 — and the base's pre-existing 51/80 is exactly what
prior on-policy distillation against a Thinking teacher would produce.

⇒ **Testable:** a short leg at reduced `--onpolicy-lambda` (0.2, or 0) on the same
corpus. If the `<think>` rate falls toward the corpus's own shape, the on-policy
teacher is the cause and the fix is either a lower λ for chat legs or suppressing
reasoning in the rollout teacher. ⚠ On-policy is what CURED the repetition
attractor (2026-06-29), so lowering it risks that returning — watch
`adj_repeat_frac` and `lrs_frac`, not just the ladder.

### math: ambiguous

L3+ 5.0% → 6.2%, copied 30/80 → 26/80, but `rel_err` 0.4286 → 0.8095. Slightly
more answers present, materially worse ones.

### Consequence

The corpus is fine and the dose is now right; the residual −6.2pp is a real but
modest cost. What blocks instruction-following is a mechanism the corpus cannot
fix on its own. `checkpoints_newmix/step_0108471.pt` remains the line.


## 2026-08-15 (later) — 🔧 `--extract`: the 0% floor was PARTLY an artifact, and the leg is a clear negative

`code_eval --extract` normalises a completion before grading: end the turn at
`<|im_end|>`, strip `<think>` reasoning, and score a fenced full definition on its
own rather than concatenating the prompt in front of it.

**Validated offline on the four saved cells — no GPU, because the reports store
every completion:**

| model | framing | as scored | EXTRACTED |
|---|---|---|---|
| base 108,471 | raw | 75.0% | **75.0%** |
| chatmix 111,471 | raw | 50.0% | **50.0%** |
| base 108,471 | chat | 0.0% | **6.2%** |
| chatmix 111,471 | chat | 0.0% | **2.5%** |

**Exact no-op in raw framing**, so every number recorded before today stays
comparable — that was the design requirement.

### The 0% floor was real but only PARTLY an artifact

Extraction lifts the base off the floor, to 6.2% — not to 75%. So `--chat-template`
genuinely costs these models most of their code capability; the scorer was hiding
a small signal, not a large one. Nine historical chat-framed evals were measuring
a floor, but the underlying capability under chat framing was also low.

### 🔴 The chat-mix leg is a clear negative on BOTH axes

    raw framing   75.0% -> 50.0%   (-25pp: lost general code capability)
    chat framing   6.2% ->  2.5%   (WORSE at the thing it was trained for)

Training on instruction data made the model **worse at instruction framing**. That
is the outcome that settles it — not the raw-framing loss, which could have been
argued as a fair trade for instruction-following. There is no trade here.

Mechanism, from the marker counts: under chat framing chatmix opens `<think>` in
**80/80** samples and never emits code (base: 51/80). ~10 epochs over a
0.96M-token corpus taught it to enter reasoning mode reliably and never leave.

### Consequence

* The corpus at this dose is a **net negative**; `checkpoints_chatmix` is kept as
  the negative result only. Base `step_0108471` remains the line.
* If instruction data is revisited it needs to be **much larger and much less
  repeated** — the failure is 10 epochs on 1M tokens, not the idea.
* `--extract` is OFF by default. Turn it on for any chat-framed comparison; a
  0.0% without it is uninformative.


## 2026-08-15 — CHAT-MIX LEG: costs 25pp of code, and `--chat-template` is a 0% FLOOR

3,000 steps from `step_0108471`, teacher stream swapped from continuation data to
the 2,545-row instruction corpus at the same `--teacher-data-ratio 0.2`. Training
was healthy: `n_loops` 4 throughout, `gnorm` median 0.48, loss settled 0.757 →
0.722. 5.09 s/step — the chat stream cost nothing extra.

### The 2x2 that should have been run from the start

|  | raw framing | chat framing |
|---|---|---|
| base 108,471 | **75.0%** | **0.0%** |
| chatmix 111,471 | **50.0%** | **0.0%** |

**My first reading compared base-RAW against chatmix-CHAT and called it a
collapse.** Two variables at once — same error class as the off-distribution
baseline the day before. The controls resolve it into three separate findings.

### 1. 🔬 `--chat-template` IS A 0% FLOOR — nine historical evals are uninformative

The BASE scores 0.0% under chat framing too. Every `--chat-template` code_eval in
this project's history — SFT 2000/2500/3000/34000/36202, big-batch 2127, chatmix
— returns 0.0%, across checkpoints with wildly different capability. The framing,
not the checkpoint, is what those numbers measured.

**The SFT collapse verdict still stands**, because it rested on the TEXT
(`def def __init__(self, self)`, punctuation spam) and on raw-framed math. But the
0.0% figures quoted alongside it were not evidence.

### 2. Chat framing puts the chat-trained model into permanent reasoning mode

    completions containing:   fence   <|im_end|>   <think>
    base + RAW                   56       32          23
    base + CHAT                  25       41          51
    chatmix + RAW                54       39          28
    chatmix + CHAT                3        1          80   <- 80/80

Under chat framing chatmix opens `<think>` in **every sample** and never emits
code. The base does it in 51/80. That IS a real behavioural change from training —
the model learned to reason indefinitely instead of answering. Fluent and
on-register (`"Okay, let's see. The user provided a Python function..."`),
`adj_degenerate 0/80`, `lrs 0.051` — **not degenerate, just never finishing.**

### 3. 🔴 The 25pp drop is REAL capability loss, not marker pollution

75.0% → 50.0% in raw framing, where the baseline was set. I expected leaked ChatML
tokens to explain it; they do not — marker counts are near-identical between
base-RAW (56/32/23) and chatmix-RAW (54/39/28). 60/80 → 40/80 is well outside
noise. **3,000 steps of chat-only teacher data cost a quarter of the code
capability**, and ~10 epochs over a 0.96M-token corpus is the likely mechanism.

Note the base ALREADY emits `<|im_end|>` in 32/80 raw completions — that
contamination predates this leg and comes from on-policy distillation against
Ouro, which emits ChatML natively.

### Consequence

* The instruction corpus at this dose is a **net negative**: −25pp code, no
  measurable instruction gain (the instrument returns 0% for everything).
* **`code_eval` cannot currently measure instruction-following at all.** It does
  not strip markdown fences or stop at `<|im_end|>` before grading, and under chat
  framing every model bottoms out. Building that instrument is a prerequisite for
  any further chat-data work — otherwise the next leg is unmeasurable too.
* `checkpoints_newmix/step_0108471.pt` remains the line, untouched.


## 2026-08-14 — ⏸ DEPTH ROUTING IS PHASE-GATED, NOT DEAD (corrected same day)

`tools/step_geometry_probe.py` on `checkpoints_newmix/step_0108471.pt`, 3,072
positions, 8 loops scored (chance 12.5%).

| selector | agrees with oracle | CE |
|---|---|---|
| ORACLE (argmin per-loop CE) | 100.0% | **0.1541** |
| **fixed depth 3 (what inference emits)** | — | **0.2124** |
| second_diff (2509.23314) | 14.6% | 0.3142 |
| min_step | 10.4% | 0.4214 |
| UncertaintyHead | 22.2% | 0.4935 |
| converge<0.01 / 0.02 / 0.05 | ~6.3% | ~0.639 |

### 1. Halting on latent dynamics buys nothing

The best geometric rule is WORSE than doing nothing (0.3142 vs 0.2124). It does
beat the trained head (0.4935) — so the head is the worst selector of the three —
but the bar is a fixed depth, and nothing clears it. **Do not build the decode
rule.** Caveat retained: this measures POST-CODA states, not the raw recurrent
iterates 2509.23314 studies, so the idea is not fully cleared — but it is not
supported here.

### 2. EVERY selector we have built is worse than doing nothing

    fixed depth 3   0.2124   <- just use the trained depth
    second_diff     0.3142   (+48%)
    min_step        0.4214   (+98%)
    UncertaintyHead 0.4935   (+132%)

Four routing attempts — ACT thresholding, `BestOfTrajectoryGenerator`,
`train_depth_policy.py`, and now geometry — and not one beats emitting from loop
3. The oracle at 0.1541 says **27% of per-token headroom genuinely exists**;
nothing we have tried captures any of it. That is the honest state of the depth
axis: the target is real, every estimator of it is worse than a constant.

### 3. ⚠️ CORRECTION — the CE curve does NOT kill depth growth

    loop:  0      1      2      3      4      5      6      7
    CE:    1.358  0.531  0.298  0.212  0.230  0.288  0.422  0.640
                                ^min   ^-- LOOPS 4-7 WERE NEVER TRAINED --^

**This was first written up as "depth beyond 4 is actively harmful ⇒ grow_depth
is DEAD". That conclusion is wrong.** `max_loop_iters` is 4, so with `--n-loops
8` loops 4-7 reuse slot 3's per-loop adaptation
(`t_idx = min(loop_t, max_loops-1)`, mythouro/main.py). They are off-distribution
BY CONSTRUCTION. The curve measures **untrained depth EXTRAPOLATION**, which was
already known to fail (2026-07-31 sweep) and which `grow_depth.py` never
proposed: that tool EXPANDS the four per-loop tensors and then you TRAIN them.

⇒ The curve says nothing about **trained** deeper loops. `grow_depth.py` stays
**SHELVED and phase-gated**, not dead. What would actually test it is expanding
to 8 and training, which needs a base worth spending the nights on.

### 3b. The same gate applies to every selector result above

The oracle's 27% headroom, the head's 0.4935, geometry's 0.3142 — all measured on
a checkpoint at **5.0% math L3+ / 0.0% L4**. A selector cannot beat the constant
when the constant is the only trained-good option, and per-loop CE differences on
a model this weak are differences between degrees of wrong. **These are
phase-gated results, not closed questions** — the owner's framing, and the
correct one: depth routing is not a capability lever at this capability, so it
cannot be evaluated here either way.

### What this DOES establish

Use a **fixed depth of 4 for now** — not because routing is impossible, but
because no selector beats a constant *on this model*, and the machinery costs
nothing to leave in place (mythouro/main.py:1955 returns `h_K` at inference, so
ACT is telemetry plus a rarely-firing break).

**Do not re-run any of this until the base is materially stronger.** Four
selector attempts have now been evaluated against a floor, and a fifth would tell
us the same nothing. The gate is capability, and the lever for capability is
tokens and objective — not depth.

## 2026-08-14 — ✅ FIRST INSTRUCTION CORPUS: 2,545 verified teacher exchanges

`data_teacher_chat_clean/` — 2,545 rows, ~540k words of assistant text, ~1.0M
tokens. The first instruction-shaped data this model has ever had access to.
Before this, every teacher row was a CONTINUATION and the only instruction data
it ever met was SFT, which collapsed it at every dose and batch size.

| check | result |
|---|---|
| malformed (not exactly 3 `<\|im_start\|>`) | **0** / 2,545 |
| unclosed `<think>` | **0** |
| missing final `<\|im_end\|>` | **0** |
| reopened `<think>` | **0** (168 dropped from the raw 2,713) |
| answer length | median 1,344 chars, p90 ~2,975, max 5,860 — well under the cap |

**Content verified by reading, not just structure.** A PubMed abstract came back
as a structured summary (objective, methodology) with the numbers drawn from the
passage. A code passage produced an accurate component-by-component explanation
grounded in the actual imports. Grounding the instruction in a real corpus snippet
is also why this should hallucinate less than the continuation harvest, which
produced "the PAWL study" and 30-mg ibuprofen.

**Settings that got there**, each measured rather than guessed: `--chat-template
--no-think --prompt-len 256 --max-new 1024 --min-new 32 --batch 18
--prealloc-cache`. `--no-think` prefills a CLOSED empty reasoning block; verified
0/18 compliance on a 2-minute probe before committing a night, and 94% across the
full run (the 6% failures are the 168 dropped rows).

**Throughput: 24 accepted tok/s**, against 118 for the continuation harvest.
Acceptance 70.5%; `unterminated` was 22% of all attempts even at `--max-new 1024`.
The gap is structural — batched decode cannot stop until the LONGEST row in the
batch finishes, so short answers idle in their lanes.

### Continuous batching: costed, and DECLINED

Modelled properly (lognormal fitted to the measured median 430 / max 1024, B=18):
perfect continuous batching is worth **2.0x**, not the 3.9x an earlier
`mean × (1+ln B)` heuristic suggested. Cheap alternatives were simulated and do
not work: stopping at a completion quantile is **1.02x**, and shorter answers
alone are **0.83x** — cutting the mean without cutting the variance makes the
max-to-mean ratio worse.

2x costs per-lane cache cursors plus per-row attention masks inside Ouro's
`eager_attention_forward` — dynamically-loaded remote code, where a subtle masking
bug corrupts every harvested token while passing every structural check. That is
the same failure class as the unconditional `<|im_end|>` (2026-08-13). **Owner's
call — "whatever will not harm the model" — and declined on that basis.** Revisit
only if chat data is shown to help and more of it is wanted.

**⇒ The open question is now a TRAINING question, not a harvest one:** does
instruction-shaped teacher data change this model's behaviour at all? 1.0M tokens
is small — at `--teacher-data-ratio 0.2` a 10k-step leg would re-read it many
times — so the first test should be SHORT and judged on whether output shape moves
toward ChatML at all, not on eval scores.


## 2026-08-11 — ❌ RUNG 3 DECIDED: uniform loop weighting DESTROYS the model

`--loop-loss-weighting uniform`, 8,706 steps from the 108,471 base. Not a
transient. Not a redistribution. The model is gone.

| loop | base accuracy | loop-weighted | Δ | base ECE | new ECE |
|---|---|---|---|---|---|
| 0 | 0.6957 | 0.1559 | **−0.540** | 0.2875 | 0.1237 |
| 1 | 0.8959 | 0.1115 | **−0.784** | 0.0940 | 0.1712 |
| 2 | 0.9623 | 0.0867 | **−0.876** | 0.0305 | 0.1963 |
| 3 | 0.9801 | 0.0750 | **−0.905** | 0.0132 | 0.2490 |

5,120 positions per loop. Next-token accuracy at the emitted depth fell from
**98.0% to 7.5%**.

**And the trajectory INVERTED.** Base accuracy rises with depth (0.696 → 0.980);
loop-weighted it FALLS (0.156 → 0.075). Deeper loops are now worse than shallower
ones — a shape no exit policy, halting head or best-exit selector can repair.

math_eval agrees: L3+ 5.0% → 0.0% at every depth, `rel_err` 0.429 → 0.929, and the
depth sweep stayed flat at 4/6/8, so **the accuracy wall did not lift.** The
`copied_from_prompt` collapse 30 → 1 matches the SFT-collapse signature (3k: 2/80,
36.2k: 1/80) — the model stopped echoing because it stopped producing
prompt-shaped text at all, not because it learned to compute.

### The "expected transient" explanation is FALSIFIED

Before the readout I argued final-loop degradation was expected, since uniform
weighting hands loop 3 only ¼ of the gradient it used to get, and I stated the
test that would kill that explanation: **loops 0–2 must rise.** They did not.
Loop 0 fell 0.54 and loop 1 fell 0.78. Nothing was redistributed; everything was
destroyed.

### Where the judgement went wrong: uniform is the HARSHEST arm, not the control

I chose `uniform` first and called it "the honest control arm, not a strawman".
That was backwards. Of the three weightings, uniform puts the **most** weight on
the shallow loops:

* `exit_pdf` concentrates on the halt depth (~2–3)
* `progressive` (`t^α`) favours later loops
* `uniform` gives loop 0 — a waypoint that has never been an output state in
  108,471 steps of training — a full 25% of the distillation gradient

Forcing an intermediate latent to behave as a finished output is precisely the
Silent Thinking objection ([2603.21676](https://arxiv.org/abs/2603.21676)),
arriving as destruction rather than as shortcut-taking. Ouro pairs per-step
weighting with an exit-probability distribution for exactly this reason; we
applied the weights *without* that concentration.

### 🔬 THE INSTRUMENT DECLARED THIS HEALTHY

`tools/per_loop_calibration.py` printed:

> *VERDICT: per-loop ECE roughly uniform (final 0.249, worst 0.249) —
> uncertainty-based per-loop selection is defensible.*

on a model at 7.5% accuracy. Worse, **loop-0 ECE IMPROVED, 0.288 → 0.124** —
because the head had correctly learned to predict "I will be wrong" (mean
uncertainty 0.72 against an error rate of 0.84). Calibration genuinely improved
while the model was destroyed. ECE measures agreement between confidence and
error, which a uniformly broken model satisfies trivially.

**Fixed:** the verdict is now gated on accuracy and withheld entirely when best
per-loop accuracy < 0.5, or when accuracy decreases with depth. Replayed against
both checkpoints: base → "calibrated at final loop only"; loop-weighted →
"WITHHELD (accuracy floor)". Fifth instance of metrics pointing the wrong way,
and the second that is a genuine instrument defect rather than a misreading.

### Consequence

Rung 3 as specified is **dead**. The base `checkpoints_newmix/step_0108471.pt` is
untouched and remains the line; `checkpoints_loopweighted` is kept only as the
negative result. If per-loop supervision is revisited it must NOT be uniform —
the candidate is LoopFormer's shortcut-consistency
([2602.11451](https://arxiv.org/abs/2602.11451)), which aligns short trajectories
**to the long one** so the deep computation stays the target, rather than making
every depth an independent output. `progressive` with a high α, or `exit_pdf` with
`--depth-reg-coeff` lowered so the halt distribution is not dragged toward
uniform, would be gentler tests of the same idea.

Cost: one overnight. The literature contradiction that motivated it
(docs/looped_lm_landscape.md §0.1) is now resolved *for this model, at this
capability*: final-only supervision is load-bearing here.


## 2026-08-11 — rung 3 leg 1: 8,706 steps loop-weighted, NO depth collapse, ρ(A) UNMOVED

`checkpoints_loopweighted/step_0117206.pt` — `--loop-loss-weighting uniform` from
the 108,471 base. 13h23m, 8,650 logged steps, **5.57 s/step vs the pour's 5.00 —
only 1.11x**, not the 2x predicted. Four codas and four head+loss passes cost ~11%.

**The abort condition did NOT fire.** `n_loops` held at 4 for every logged step;
no collapse toward loop 0, so the λ₀→1 shortcut failure that Silent Thinking
(2603.21676) predicts, and that we hit once with the ACT-weighted sum, did not
recur under uniform per-loop weighting. `gnorm` median 0.96, 3 of 174 above 5.

### ρ(A) is unchanged, and that is a second explanation for the flat depth sweep

    step 108,500   ρ(A) ∈ [0.220824, 0.265697]
    step 117,000   ρ(A) ∈ [0.220296, 0.270053]

8,706 steps of per-loop supervision moved the contractivity of
`recurrent.injection` essentially not at all. At ρ ≈ 0.245 the LINEAR part of the
update converges geometrically — residual 0.36% of initial by loop 4, 0.001% by
loop 8 — so loops 5-8 have almost nothing left to change.

⇒ A mechanistic account of the flat 4/6/8 sweep **independent of the supervision
question**. If the loop is this contractive, no change of objective produces depth
extrapolation without also changing the contraction.

**Caveat, stated because it limits the claim:** this is `recurrent.injection`
only — the linear `A·h + B·e` term. The Transformer term in the same update is
not in that number and may do substantial per-loop work. Suggestive, not
decisive; the depth sweep is still the readout.

**Connects to DeepLoop ([2607.13491](https://arxiv.org/abs/2607.13491)) more
directly than noted when filing it:** that paper adjusts residual scaling *as a
function of visit count*, and ours does not vary with loop count at all. Worth
re-reading before rung 5, and possibly before rung 3 leg 2.

**Ended on the documented XPU teardown hang** (max1100_field_notes §168): the
checkpoint saved at 12:31 and verified loadable (step 117206, 160 tensors,
optimizer present), then the SYCL runtime deadlocked in teardown holding its
allocation. `kill -9` is the remedy and the next job OOMs without it. The run
scripts say "Ctrl-C is safe" without mentioning the cooperative delay, the
double-signal trap, or this hang — a doc gap that has now cost time twice.


## 2026-08-10 (final) — ✅ RUNG 0 DECIDED: batch size was NOT the mechanism

`checkpoints_v8_bigbatch/step_0002127.pt`, effective batch **256** (32x the prior
runs), stopped early at 2,127 of 2,500 steps.

| scored in each model's own format | base | SFT 3k @b8 | SFT 36.2k @b8 | **SFT 2127 @b256** |
|---|---|---|---|---|
| code per-sample L3+ @pen1.15 | **75.0%** | 0.0% | 0.0% | **0.0%** |
| math per-sample L3+ | 5.0% | 1.2% | 0.0% | **0.0%** |
| math copied-from-prompt | 30/80 | 2/80 | 1/80 | 12/80 |
| math median rel_err | 0.400 | — | — | 0.857 |

**This run saw the MOST data of the three** — 2,127 x 256 = 544k samples, against
24k (3k@8) and 290k (36.2k@8). 32x the batch, ~2x the samples of the longest
prior run, same 0.0% floor. arXiv 2412.13337's batch-size lever does not rescue
SFT here, and the exposure-bias reading of 2026-08-10 stands with batch **ruled
out** rather than merely unexamined.

Batch changed the FLAVOUR of the collapse, not its destination:

    SFT 3k     def def __init__(self, self):
    SFT 36.2k  - - - - : : : : | | | = = = conclude:
    SFT 2127   is is is not only only one one one one one of them:):
       @b256   ThisThisThis_is_tweakcasecasecase(1000) as as as many, but but then
    math       "12 + 7 =" -> "11 ... 211 ..." then newline spam

Word-salad with token doubling rather than punctuation spam — arguably the
roadmap's v4 stage rather than 36.2k's endpoint — but still L0 on every code task.

### 🔬 EVERY DEGENERACY METRIC SAID THIS WAS HEALTHY

    looped_frac 0.013 | char_degenerate 0 | median_lrs_frac 0.071 | max_line_repeat 1

All four look fine. The text is plainly degenerate. The pathology here is
**immediate ADJACENT-TOKEN repetition** (`is is is`, `one one one one one`,
`ThisThisThis`, `casecasecase`), and every degeneracy metric we own measures
either LINE repetition or longest-repeated-SUBSTRING — neither of which fires on
adjacent duplicates. `lrs_frac` in particular is near-zero precisely because the
repeats are short and local.

⇒ **Fourth instance of metrics pointing the wrong way** (after math answers,
medical abstracts, the alpha-ladder spike). This one is a concrete instrument
gap, not a judgement error: `tools/code_eval.py` needs an adjacent-token-repeat
rate alongside `lrs_frac`. Until it exists, the degeneracy verdict from these
tools is only valid for the failure modes it was built for.


## 2026-08-10 (later) — 🔧 THE SFT MIX IS NOT THE MIX WE CONFIGURED

Chasing ~200,000 unattributed rejections in the big-batch run's dataset
diagnostic. Three findings, only one of which is a defect in the data.

### The diagnostic was structurally unable to show the top reject reason

`MixedSFTDataset` counts adapter rejections in `stats[key]["no_messages"]` and
prints only `stats[key]["reject_reasons"]`. The two never met, so the largest
rejection category was invisible while the log line's own comment claimed it
showed "the top rejection reason for each source". It hid 103,939 general
rejections behind an attributed 3,929, and 101,975 code rejections behind 8.
**Fixed:** `adapter_rejected` is merged into the printed breakdown, plus an
`UNACCOUNTED=` term so the counts must now reconcile against
`attempted - yielded`.

### Why each source rejects — one is policy, one is quality, neither is a bug

* **clean_general 59.2% accept.** `_to_messages_tulu` drops Tulu-3's
  OpenAI-derived subsets (`_TULU_EXCLUDED_SOURCES`). That is the clean-data
  constraint working exactly as required — not a defect, and not to be
  "fixed".
* **clean_code 38.2% accept.** `_to_messages_opencode` requires EVERY unit test
  to pass. Defensible: execution verification is that dataset's whole quality
  signal.
* Earlier today I recorded code at ~52% rejection and general at 97.3%
  acceptance. Both wrong — the 97.3% came from an early window that was not
  representative of the run. Corrected here.

### 🔴 THE ACTUAL DEFECT: ratios are applied BEFORE rejection

Sources are drawn by configured weight and only *then* filtered, so acceptance
rate silently re-weights the corpus. Measured over 750,000 draws:

| source | drawn | kept | accept | configured | REALIZED | drift |
|---|---|---|---|---|---|---|
| clean_general | 254,989 | 151,050 | 59.2% | 34.0% | 27.8% | −6.2pp |
| clean_math | 239,940 | 239,688 | 99.9% | 32.0% | **44.1%** | **+12.1pp** |
| clean_code | 164,949 | 62,974 | 38.2% | 22.0% | **11.6%** | **−10.4pp** |
| clean_pubmedqa | 90,122 | 90,090 | 100.0% | 12.0% | 16.6% | +4.6pp |

**Code was configured at 22% and delivered 11.6% — very nearly halved — while
math ran 38% hotter than intended.** Every SFT run to date trained on this
distorted mix, and it compounds with the separately-measured loss-bearing-token
skew (65.9% general vs 12.7% math vs 8.2% pubmedqa), which pulls gradient share
further from the configured intent.

### The fix is NOT to loosen the filters

Today's other result is that 36,202 offline SFT steps made the model strictly
worse, so **data volume is not the lever** and trading correctness for quantity
would be the wrong direction — particularly on code, where the filter is the only
thing certifying the target actually runs. The correct fix is to make the DRAW
weights compensate for measured acceptance so the realized mix matches the
configured one, leaving the quality bar untouched. Not yet built.


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

## 2026-09-03 (evening) — ✅ EXIT_PDF @7,200: the distinct1 dip was a TRANSIENT

Second readout on the exit_pdf lineage, 3,000 steps past the first. Both
instruments, three checkpoints each, all in one session.

| | base | @4,200 | **@7,200** | mathcode (regressed) |
|---|---|---|---|---|
| halt depth | 2.49/4 | 3.21/4 | **3.22/4** | — |
| code L0 | 12.6% | 2.5% | **4.7%** | — |
| code L3+ | 71.1% | 75.6–77.2% | 65.0% | — |
| prose top_share ↓ | 0.104 | 0.118 | **0.106** | 0.159 |
| prose distinct1 ↑ | 0.571 | 0.519 | **0.527** | 0.482 |

### The question this settled

`distinct1` fell 0.571 → 0.519 at 4,200 (~2 sd) and was flagged as the metric to
watch, since a sustained slide is how the repetition attractor announced itself
all summer. **It did not slide.** 0.519 → 0.527 over 3,000 further steps, and
`top_share` recovered fully to base (0.106 vs 0.104).

⇒ The dip was a **settling transient**. What remains is a persistent but stable
~0.044 reduction in lexical variety — real, modest, non-compounding. That is the
honest cost of the objective, and the pour is clean to continue.

### What is durable, and what is noise

**Durable (replicated across independent checkpoints):**
* halt depth **2.49 → 3.22**, moved 3.21 → 3.22 over 3,000 steps — converged
* code **L0 12.6% → 2.5% / 4.7%**, p=2e-7 and p=7e-5 against 960 base samples

**Noise, and should not be quoted:**
* code L3+ at 65.0 / 71.1 / 75.6 / 77.2 spans less than one 13.6 pp checkpoint sd
* code L4 at 3 / 9 / 14 / 52 — the metric swung 30/15/1 within one lineage

### ⚠ Housekeeping failure worth recording

`--keep-last 8` rotated away `step_0004200.pt` — the checkpoint every depth-sweep
and eval number in the 09-02/09-03 entries was measured on. The reports survive
so nothing is lost analytically, but the model is gone and those numbers can
never be re-derived. **A checkpoint that carries a published result belongs
outside the rotation**, the way `checkpoints_base/` preserves the pre-growth
lineage. `step_0007200` has been copied to
`checkpoints_base/exitpdf_0007200.pt`.

## 2026-09-03 — 🔬 WHAT ACTUALLY BROKE THE REGRESSION WALL (and why we nearly missed it)

The wall was broken by the **growth leg**, not by `exit_pdf` — and not through
capacity. Recording this because nobody was looking for it, and the only reason
it surfaced was the owner asking whether undertrained experts were dragging the
model down.

### The measurement that shows it — two legs, same parent, same corpus

All α=0.0, all measured in one session:

| | top_share ↓ | distinct1 ↑ |
|---|---|---|
| base `step_0157000` | 0.137 | 0.520 |
| **mathcode leg** — 24 experts, code+math | **0.159** | **0.482** |
| **grown48 → masked → pruned24** — 48 experts, code+math | **0.104** | **0.571** |
| exit_pdf on pruned24 | 0.118 | 0.519 |

Two legs from effectively the same parent on the same corpus. **The 24-expert leg
regressed on both metrics; the 48-expert leg improved on both.** Then masking the
added experts KEEPS the improvement.

### It was not capacity, and every measurement says so

* activated params never moved off **180,726,115** at 24 or 48 experts
* the new experts never differentiated — ~90% twins, asymptotic (fit 0.893),
  with initialisation and load balancing both independently ruled out
* **masking them made the model BETTER**, not worse

So the benefit was entirely at **training time**. The growth was, functionally, a
dropout mask made of parameters.

### ⚠️ THE OWNER'S MECHANISM, TESTED AND RULED OUT

Proposed: the twins act as a **reference copy** — the originals drift onto the
new corpus while the twins retain what is being forgotten, anchoring it.

That makes a prediction the dilution account does not: twins should sit CLOSER
to the step-0 weights than the originals do. Measured on `grown48`:

```
cos(original@8696, original@0) = 0.9497     <- how far the originals moved
cos(twin@8696,     original@0) = 0.9424     <- how far the twins stayed
                      difference  -0.0073
```

**The twins did not stay anchored** — they drifted the same distance, marginally
further. No reference copy. Good hypothesis, clean refutation, and worth more
than the untested version because it made a falsifiable prediction.

### What is left, and its honest status

**Inferred, not established: dilution.** With half of every token's routing going
to weak experts, each original expert receives a smaller, noisier update —
effectively a lower learning rate on the corpus term, so less over-fitting onto
code+math while the on-policy component (70% of micro-steps) carries relatively
more of the signal.

⚠ **This cannot be separated from `--rollout-batch`.** The two legs also differed:
mathcode ran 32, grown48 ran 8 (48-expert crash avoidance). Lower rollout-batch
means less diverse on-policy sampling, and that is a live alternative.

### ⇒ THE TEST THAT SETTLES IT — one night, one variable

24 experts, code+math, **expert dropout ~50% on routing**, everything else
matched to the mathcode leg *including rollout-batch 32*.

* prose escapes the wall → **dilution confirmed**, and the effect is available
  without ever promoting a checkpoint
* prose regresses like mathcode → **rollout-batch was the variable**, which is a
  far cheaper lever than either

### Why this was unforeseen

Growth was justified on capacity. Capacity never moved. The benefit arrived
through a channel nobody was measuring, and it was invisible while the diluting
experts were still live — the 48-expert model reads WORSE than the base
(L0 12.6% → 31%). It only appeared once they were masked, and masking only
happened because the owner asked whether the undertrained experts were the drag.

**The lesson is not "growth works".** It is that a result can be real, large, and
attributable to a mechanism entirely different from the one that motivated the
experiment — and that the instrument which reveals it may be one nobody thought
to build.

## 2026-09-03 — ⏹ DEPTH SWEEP: the wall did NOT lift. Contractivity binds, not supervision.

`exit_pdf` model, no training, evaluated beyond its trained depth:

| n_loops | L3+ | L0 | L4 | task L4+ |
|---|---|---|---|---|
| 4 (trained) | 75.6% | 3% | 3/320 | 2/10 |
| 6 | 76.2% | 3% | 5/320 | 3/10 |
| 8 | 75.6% | 2% | 3/320 | 2/10 |

**4 → 8 loops: L3+ +0.0 pp.** Extra depth at inference buys nothing.

### Why this is a CLEAN result rather than a repeat of the old flat sweep

The 2026-07-31 sweep was also flat at 4/6/8, but under final-only supervision —
the exact condition the literature says produces a wall at the trained depth, so
it could not distinguish "objective walls the model" from "depth is useless".

This one is measured on a model that **demonstrably uses depth**: halt at 3.21/4
with mass on the last two loops, learned and stable. The flatness is therefore
NOT the model ignoring its loops. **Loops 5-8 have nothing left to contribute.**

ρ(A) predicts exactly that: mean 0.289, so the linear `A·h + B·e` term is at
0.7% of initial by loop 4 and 0.005% by loop 8. **The recurrence has converged
by the depth it was trained to.** The 2026-08-11 contractivity argument was right
— and it is now confirmed on a working objective rather than on the bugged
rung-3 leg where it was first measured.

### ⇒ The literature's "24x beyond trained depth" does not transfer

`recurrent-depth-ttc` reports iterative-target supervision extrapolating 24x.
We now have iterative-target supervision and get 1.0x. The difference is not the
objective — it is that our recurrence is far more contractive than theirs. That
is a property of **our architecture**, not our training recipe.

### ⇒ The depth lever is the CONTRACTION, and it is a build not a flag

DeepLoop ([2607.13491](https://arxiv.org/abs/2607.13491)) scales residuals as a
function of visit count and moves the DeepNorm exponent 1/4 → 1/2 as recurrent
depth rises. `references.md` already flags that `grow_depth.py` *"expands four
per-loop tensors but changes NO normalisation scaling, and we run sandwich norm"*
— so rung 5 was always gated on this, and the sweep confirms the gate was right.

### ⇒ THE DEPTH LEVER IS DeepLoop, AND HERE IS WHAT IT WOULD TOUCH

**DeepLoop: Depth Scaling for Looped Transformers**
([arXiv:2607.13491](https://arxiv.org/abs/2607.13491)) formalises residual
scaling when the SAME blocks are revisited — a perturbation bound with a
visit-alignment coefficient — and moves the Post-LN DeepNorm exponent from
**1/4 to 1/2 as recurrent depth increases**. Validated at GPT-2 scale.

**Our residual scale does not vary with loop count at all.** `RecurrentBlock`:

```python
x = x + self.resid_drop(self.attn(self.attn_norm(x), ...))
if self.use_sandwich: x = self.post_attn_norm(x)
x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
if self.use_sandwich: x = self.post_ffn_norm(x)
```

Loop 1 and loop 8 get identical treatment. Nothing in the block is a function of
which visit this is. Combined with ρ(A) mean **0.289**, that is precisely the
regime where revisiting adds nothing: the state has already settled and each
further pass applies the same contraction to an already-converged vector.

**This is why `tools/grow_depth.py` is gated.** `references.md` records it:
*"expands four per-loop tensors but changes NO normalisation scaling, and we run
sandwich norm."* Growing 4→8 without loop-dependent scaling would add loops into
the regime the sweep just measured as empty.

**And the same note gives independent corroboration:** DeepLoop is *"a concrete
candidate explanation for Ouro's own 'tried 8, dropped to 4 after loss spikes and
gradient oscillations'."* Our teacher hit this too, at production scale, and
retreated to exactly our depth. Two systems, same wall, same suspected cause.

⇒ **Order of operations for any future depth work:** loop-dependent residual
scaling FIRST, then `grow_depth.py`, then re-run this sweep. Doing it in the
other order tests nothing — which the flat 4/6/8 result now demonstrates rather
than merely predicts.

### Three axes now closed on evidence, one open

| axis | verdict |
|---|---|
| expert-count growth | **closed** — twins both times, activated params never moved off 180,726,115 |
| depth extrapolation | **closed** — contraction binds; needs DeepLoop-style residual scaling first |
| **the objective (exit_pdf)** | **WORKS** — L0 12.6% → 2.5% (p=2e-7), ACT decides for the first time |
| tokens (main-thread #2) | **OPEN, and never run** |

**⇒ The pour is the work.** It is main-thread #2, open since June, and #3 records
it as the go/no-go for capital. It can now be measured through an objective that
does not wall the model at its trained depth — which is exactly why running it
before this week would have produced a pessimistic curve and possibly the wrong
call on real money.

## 2026-09-02 — 🟢 EXIT_PDF: THE WALL WAS THE OBJECTIVE. ACT finally decides.

4,200 steps of `--loop-loss-weighting exit_pdf --depth-reg-coeff 0.1` on the
pruned 278M, the first time Ouro's own objective has ever run here.

### The headline — ACT made a decision, and it held

| | halt distribution | mean depth | KL to uniform |
|---|---|---|---|
| base (depth-reg 0.3) | [0.249, 0.252, 0.256, 0.244] | 2.49/4 | 0.0001 |
| exit_pdf @4,000 | [0.012, 0.189, 0.366, 0.433] | **3.22/4** | 0.2893 |
| exit_pdf @4,104 | [0.011, 0.198, 0.357, 0.434] | **3.21/4** | 0.2862 |
| exit_pdf @4,200 | [0.011, 0.206, 0.339, 0.443] | **3.21/4** | 0.2825 |

Loop 0 abandoned (0.249 → 0.011); mass on loops 3 and 4. Three consecutive
checkpoints agree to two decimals — a **stable learned distribution**, not drift.

**This is what four prior efforts could not do.** "Halt depth pinned at exactly
2.00/4 on every sample of every domain… ACT is not making a poor routing
decision; it is making no decision at all" was **the depth regulariser forbidding
a decision** — a KL toward uniform, whose mean depth is ~2 by construction. Drop
it to 0.1, supervise every loop, and ACT decides within 1,200 steps.

### Capability — a favourable trade, not a free lunch

| | base | exit_pdf | read |
|---|---|---|---|
| code **L0** (nothing usable) | 12.6% (121/960) | **2.5%** (8/320) | **z=−5.20, p=2e-7** |
| code L3+ | 71.1% | 77.2% | +6.0 pp, within the 13.6 pp sd |
| code L4 | ~9–30/320 | 4/320 | L4 is ~12x noisier; unreadable at n=1 |
| prose top_share ↓ | 0.104 | 0.118 ±0.018 | within noise |
| prose distinct1 ↑ | 0.571 | **0.519 ±0.026** | **−0.052, ~2 sd — mild real loss** |

**Five times fewer catastrophic failures**, at the cost of slightly narrower
lexical variety. Consistent with a model now spending 3.21 of 4 loops per token:
more deliberation, more reliability, marginally less sampling diversity.

⚠ **`distinct1` is the number to track.** A sustained slide is the early
signature of the repetition attractor fought all summer. One 2 sd dip at 4,200
steps is not that, but it is the metric that would show it first.

### ⇒ It does NOT reproduce the 2026-08-14 precedent

That run moved depth 2.00 → 3.60 via `train_depth_policy.py` and **the task got
worse** (math L3+ 11.2% → 5.0%). Here depth rose and the code task **improved**.
Two mechanisms, opposite outcomes. The difference: this came from the training
objective, not a bolt-on policy stage optimising the halting head directly.

Corroboration worth noting — that 2026-08-14 entry found the trained head put
"mass on loops 3 and 4, the two lowest-CE depths." exit_pdf independently
arrived at the same two loops. Different mechanisms, same answer.

### ⇒ WHAT THIS MEANS FOR THE PROJECT

`looped_lm_landscape.md` §0.1 said it on 2026-08-10: *"depth is not dead, it is
walled, the wall is the objective, and rung 5 (grow depth) treats the symptom
while rung 3 treats the cause."* Rung 3 was then closed on 2026-08-11 by a bug
that made loop-weighted training supervise `coda(prelude(x))` with the recurrent
block bypassed — and the growth programme followed, treating the symptom for six
days and two promotions.

**The lever was the objective, and it was documented, implemented, and one flag
away the entire time.**

## 2026-09-01 (late night) — 🐛 RUNG 3 WAS CLOSED ON A BUG. Loop weighting has never run.

Trying to launch `exit_pdf` produced a loud error, and chasing it invalidated a
roadmap verdict.

### The defect

`RecurrentBlock` gated trajectory capture on **`not self.training`**:

```python
collect = self.collect_trajectory and kv_cache is None and not self.training
```

That guard was written for the INFERENCE best-of-trajectory feature.
`MythOuro.forward_loop_states` was written later, for loop-loss-weighting, and
needs the trajectory **during training** — it even sets
`trajectory_requires_grad=True` and the code says why: *"needs the graph
precisely BECAUSE it builds a training loss at every loop — without it the
recurrent block and prelude receive no gradient at all."* The older guard
silently defeated it.

So in training `last_trajectory` was `None`, and `forward_loop_states` fell
through to:

```python
if traj is None:              # n_loops == 0     <- the comment is wrong
    traj = e.unsqueeze(2)     # e = PRELUDE OUTPUT
```

**Every `--loop-loss-weighting` run supervised `coda(prelude(x))` with the entire
recurrent block bypassed.** Measured before the fix: `states (2,32,1,1280)`
against `halt (2,36,4)` — K=1 where four loops ran.

### ⇒ Rung 3's verdict is RETRACTED

*"uniform loop weighting DESTROYS the model"* (2026-08-11: per-loop accuracy
0.980 → 0.075, trajectory inverted) is exactly what training a model to answer
from its prelude alone produces. **That was the bug, not a property of loop
supervision.** `uniform` and `progressive` failed SILENTLY; only `exit_pdf`
raised, because someone wrote a shape check with the note *"do not let this
silently fall back, or the A/B measures nothing."* That note is the only reason
this was ever found.

### After the fix

```
TRAIN mode: states (2,32,4,1280)   halt (2,32,4)   sums to 1   grad ✓
mean halt mass per loop: [0.249, 0.253, 0.254, 0.244]
```

Default inference path unchanged (`collect_trajectory` and
`trajectory_requires_grad` both default False). 433 + 4 tests pass;
`tests/test_loop_states.py` pins all four properties, including that the
recurrent block actually receives gradient.

### 🔍 And it reinterprets "halt is pinned at 2.00/4"

The halt distribution is **uniform** — 0.25 per loop. That is
`--depth-reg-coeff 0.3` doing its job: a KL pulling the halt distribution toward
uniform. A uniform distribution over 4 loops has mean depth ~2 **by
construction**. So the long-standing finding *"ACT is not making a poor routing
decision; it is making no decision at all"* is better read as **the regulariser
forbidding a decision**, not the halt head failing to make one.

It also means `exit_pdf` ≈ `uniform` at depth-reg 0.3 — precisely what
`run_loopweighted.sh` predicted when it skipped exit_pdf, and why
`run_exitpdf.sh` lowers depth-reg to 0.1 (the documented v3-onward default) in
the same run.

**Sixth silent failure of the week.** Five logged warnings or reported success;
this one silently trained the wrong states for an entire rung and produced a
published verdict.

## 2026-09-01 (night) — 📚 THE LITERATURE SAYS OUR WALL MAY BE THE OBJECTIVE, AND THE TREATMENT IS UNRUN

Owner's prompt: check the papers, that is how the conclusions were reached.
Doing so lands three things directly on this week's results.

### 1. Final-only supervision is a documented cause of accuracy walls

`looped_lm_landscape.md` §0.1 records the contradiction with evidence both ways.
MythOuro supervises **final-only (`h_K`)** — and not by choice, but because the
ACT-weighted sum let the optimiser pin λ₀≈1 and collapse depth.

* [recurrent-depth-ttc](https://github.com/duongtrongnguyen123/recurrent-depth-ttc):
  iterative-target supervision extrapolates **24x beyond trained depth**;
  **final-only supervision causes accuracy WALLS.** Same prelude/core/coda layout.
* **Ouro** ([2510.25741](https://arxiv.org/abs/2510.25741)) — *our teacher* —
  trains `L = Σ_t p_φ(t|x)·L^(t)`, per-step weighted by exit probability.
* **RLTT** ([2602.10520](https://arxiv.org/abs/2602.10520)): distributing credit
  across the latent trajectory beats terminal-only by **+5.8% / +10.9%**,
  measured on Ouro.

The landscape doc's own reading: *"depth is not dead, it is walled, the wall is
the objective, and rung 5 (grow depth) treats the symptom while rung 3 treats the
cause."* **We spent 2026-08-27..09-01 growing WIDTH.**

Corroborating our own instruments: halt depth is pinned at **exactly 2.00/4 on
every sample of every domain** and has been on every checkpoint measured; the
`--n-loops` sweep was flat at 4/6/8; L4 has never cleared ~9%.

### 2. `exit_pdf` — Ouro's exact weighting — EXISTS AND HAS NEVER BEEN RUN

`--loop-loss-weighting` accepts `off | uniform | progressive | exit_pdf`, where
exit_pdf is documented in-flag as *"the model's own halt distribution, i.e.
Ouro's p(t|x) exactly."* Rung 3 tested **uniform only** and concluded loop
weighting destroys the model.

Uniform is the harshest arm — the roadmap says so: *"it gives loop 0, never an
output state, 25% of the gradient."* It inverted the depth trajectory
(0.980 → 0.075, deeper loops worse than shallower), which is what forcing loop 0
to match loop 3 should do. **That result does not generalise to exit_pdf.**

`run_loopweighted.sh` skipped exit_pdf deliberately and for a good reason:
`--depth-reg-coeff 0.3` is a KL from the halt distribution toward UNIFORM, so
exit_pdf would be pulled toward uniform anyway. The roadmap already names the
remedy: **"progressive/exit_pdf with depth-reg lowered."** Never run.

### ⚠ 3. The honest tension — this is a two-sided experiment, not a fix

With halt **pinned at 2.00**, exit_pdf concentrates nearly all weight on loop 2:
that converts final-only supervision into *loop-2-only*, not distributed
supervision. Lowering `--depth-reg-coeff` is what lets the halt distribution
move — but depth-reg is exactly what prevents ACT loop-collapse, the failure that
made us supervise `h_K` in the first place. **The two knobs pull against each
other and must move together.**

The landscape doc pre-registered how to read it: *"depth collapsing toward loop 0
vindicates Silent Thinking; the flat-depth wall lifting vindicates per-loop
supervision."* Watch the halt distribution and `loop_efficiency` beside the evals.

### 4. Two more unadopted levers from the same batch

* **Muon instead of AdamW** ([2511.07384](https://arxiv.org/abs/2511.07384),
  switched specifically for recurrent stability) — *"cheapest untested lever in
  this batch on a rig where stability has been the recurring failure."*
* **A data curriculum** — general "healing" data first, task data after. *"We ramp
  loop depth but never data composition."* This bears directly on the mathcode
  regression, which poured code+math onto a general model and produced salad on
  MEDICAL seeds — a curriculum failure with a named remedy in the literature.

### 5. And the scale gap, stated plainly

Retrofitted recurrence reaches **51.2% GSM8K at 32 recurrent steps** vs 46.2% for
the non-recurrent baseline. We train at **4** and evaluate at 4, with halt pinned
at 2. *"Depth, not width, drives multi-step accuracy"* — and width is the axis we
just spent a week on.

## 2026-09-01 (evening) — 🟢 PROSE: the wall REPRODUCES, and the masked model is past it

First prose measurement ever taken on a grown model, and the first controlled
contrast between a leg that regressed and one that did not. **Everything below
was measured in ONE session**, α=0.0 (pure student, no teacher mixed in).

| | top_share ↓ | distinct1 ↑ | n |
|---|---|---|---|
| **MASKED grown48 (24 live)** | **0.104** ±0.015 | **0.571** ±0.016 | 3 |
| 278M base `step_0157000` | 0.137 | 0.520 | 1 |
| mathcode leg (the REGRESSION) | 0.159 ±0.036 | 0.482 ±0.025 | 3 |
| — of which `163,238` | 0.205 | 0.450 | |

### 1. The regression is REAL and it reproduces

`163,238` reads **0.205** in-session against **0.242** archived — same direction,
same magnitude, and the worst of everything measured. The mathcode leg sits below
its own parent (0.159 vs 0.137). **The wall is not a measurement artifact.** I
questioned that on 2026-08-31 on the strength of the checkpoint-variance finding;
that questioning was wrong and is withdrawn here.

### 2. The masked model is past the wall — perfect rank separation

masked spans **0.083–0.117**; mathcode spans **0.118–0.205**. **No overlap**,
3 vs 3, which is the minimum achievable p (0.05, exact). And it beats the base it
descends from on both metrics.

### 3. Both legs poured the SAME corpus

`mathcode` (278M, 6,000 steps from 157,238) and `grown48` (397M, 8,696 steps from
157,000) both poured code+math from effectively the same parent. One regressed
prose; the other improved it.

**HYPOTHESIS, not a finding — accidental regularisation.** The difference is that
grown48 trained with 24 undertrained experts absorbing roughly half of every
token's routing. That dilution may have prevented the original experts from
over-specialising onto code+math — which is exactly the mechanism that produced
salad on *medical* seeds in the mathcode leg. Mask them at eval and you cash in an
un-regressed model. **Untested.** The clean test is a 24-expert leg on the same
corpus with an equivalent regulariser (expert dropout, or a held-out expert
fraction) — if it also escapes the wall, the mechanism is regularisation and
growth was never needed for it.

### 4. The text agrees with the metrics — read it

```
163,238  (archived):  "such as C.P. ... - CASE - CABI (cascar - CCAO - CACA
                       - CACI - CCA - CACF - CSC - CAST - CDA - CCI - CCA"
masked grown48:       "the treatment of the skin problem and the presence or
                       more often a specific type of infection. We know that
                       the combination of the new treatment for the diseases..."
```

Acronym salad gone. Register correctly selected per domain across four seeds —
clinical bullets for diabetes, Python for fibonacci, LaTeX for the quadratic,
pharmacological prose for ibuprofen. **One honest exception:**
`"using ant-antantantantant or D or d"` — a token-level stutter on the bacterial
seed, not the sustained cascade that defined the regression.

### ⇒ 5. AND THE UN-PARK CONDITION FOR GROWTH WAS NEVER MET

`ideas.md` parked "grow the model" on 2026-06-18: *fails triage #1, token-starved,
bigger model = hungrier, worse*. Un-park condition, written at the time:

> **"Token-curve shows we've reached compute-optimal at current size."**

**That curve was never run.** Growth was un-parked in August on the regression-wall
evidence instead. And today's prose result is a data point ON that curve — 8,696
further steps improved prose over the base — which says the model is **still
gaining from tokens**, i.e. still not compute-optimal, i.e. the un-park condition
is *still* unmet.

**Main thread #2 — "push 5-10x tokens, inspect every ~50M tokens, does capability
keep growing?" — is the work.** #3 records that this curve is the go/no-go for
capital and the proof artifact for a collaborator. It remains the highest-value
thing on the board and it has been open since June.

## 2026-09-01 (final) — ✅ THE NEW EXPERTS WERE THE DAMAGE. Growth should be REVERTED.

Owner's hypothesis: the grown models look worse because with topk=4 of 48, half
of every token's experts are now the UNDERTRAINED ones. Tested by re-applying the
-100 sentinel to experts 24-47 (`tools/mask_new_experts.py`) — same weights, same
training, new experts silenced, model functionally 24-expert again.

### CONFIRMED, and large

| | L3+ | L0 |
|---|---|---|
| grown48, 48 live | 48.4% | 102/320 |
| grown48, MASKED to 24 (mean of 3 ckpts) | **71.1%** (sd 4.7) | **40** |
| 278M base | 68.8% | 49 |

**Silencing the new experts fully recovers the model** — L3+ back to base, L0
*below* base (z=-5.90, p<1e-6). The 24 undertrained experts were degrading every
token they touched. **Growth's harm is real, mechanical, and reversible.**

### ⚠️ WITHDRAWN THE SAME SESSION: "L4 tripled"

Off `step_0008696` alone, masked L4 read **30/320** against the base's 9 — z=3.47,
p=0.0005, and it broke the trade pattern. It did not replicate:

```
masked L4 by checkpoint:  8696 -> 30,  7000 -> 15,  8000 -> 1
mean 15.3, SE 6.8, interval 1.7-29.0 — INCLUDES the base's 9. Not significant.
```

One draw, written up as a finding. The protocol from 2026-08-31 said means over
>=3 checkpoints and was ignored the moment a result looked good.

### ⚠️ L4 IS ~12x NOISIER THAN L3+ AND IS UNUSABLE ON ONE CHECKPOINT

```
L4   mean 15.3  sd 11.8   relative sd 0.77
L3+  mean 71.1  sd  4.7   relative sd 0.066
```

30 -> 15 -> 1 across checkpoints ~1,000 steps apart. **Every single-checkpoint L4
number this project has quoted is suspect**, including "code L4 = 0 across 240
generations" (2026-07-30), which was part of the evidence that the wall was
capacity and that the student should be grown.

### ⇒ WHAT TO DO

1. **Revert the growth.** Prune experts 24-47 and go back to 24. The grown models
   are strictly worse live, equal when masked, and cost 34% more per checkpoint.
   Two controlled experiments say the new experts will never differentiate.
2. **No evidence the 8,696 steps helped.** Masked ~= base on both metrics. The
   pour on a grown model bought nothing; it was spent diluting itself.
3. **L4 needs >=3 checkpoints, always.** So does L3+, at a lower factor.

## 2026-09-01 — 📖 READ THE TEXT: growth made TYPICAL output WORSE, and best-of-32 hid it

The owner asked for prompt→output pairs rather than the ladder aggregate. Doing
that changed the reading twice, and the second correction is the one that counts.

### First pass (WRONG — best-of-32 is pass@32, not behaviour)

Taking each task's highest-rung sample, `grown48` looked BEST: it produced
`return sum(nums)` and `return n % 2 == 0` where the base failed, while L3+
ranked it worst. That is cherry-picking — one sample out of 32 says what the
model CAN do, not what it does.

### Second pass — the full rung distribution over all 320 samples

| model | L0 | L2 | L3 | L4 |
|---|---|---|---|---|
| **278M base** | **15%** | 15% | **65%** | 2% |
| 397M grown48 | **31%** | 19% | 44% | 3% |
| 397M v2+balancer_off | **30%** | 10% | 57% | 1% |

**Both grown models DOUBLE the L0 rate** — no usable function at all — 15% →
30-31%, consistently and independently across two separate lineages. Text and
metrics agree once you stop reading the best sample: growth did not help and
plausibly hurt.

### What the text shows that no aggregate did

The base is **L3 65% / L4 2%**. It writes code that RUNS and is WRONG on
two-thirds of samples:

```
is_even       ->  if n == 0: return False        (runs; wrong for every input)
count_items   ->  count_items = 0 ; for i in range(len(items)):   (never returns)
add_two       ->  if b == 0: return 1 ... def add_two(a,b): return a+b
```

That last one is the shape of the whole problem: **the correct answer is present,
buried in noise the model also emits.** This is not fluency (solved), not
halting, and not obviously capacity. It is CORRECTNESS, and it is where the
entire remaining gap lives.

### ⚠ Instrument note — the rambling tail is NOT scored

`code_eval.py:344` truncates at `<|im_end|>` before grading, so the paragraphs of
wrong explanation after the code do not affect the rung. Read completions
truncated the same way or you will judge text the grader discards — I did, for
one message, and it inverted the conclusion.

## 2026-09-01 (later) — 📉 CAPABILITY AFTER GROWTH: NULL. Three lineages, one number.

All n=320, bare framing, T=0.4 pen=1.15 seed=1234 — identical settings, and the
control is the BIT-EXACT promoted 278M so no cross-session drift is involved.

| checkpoint | L3+ | L4 | task L4+ |
|---|---|---|---|
| 278M base (grown48 step 0 control) | **68.8%** | 9/320 | 3/10 |
| 397M grown48 @ 8,696 (code+math) | 48.4% | 12/320 | 4/10 |
| 397M growth_v2 + balancer_off @ 6,000 | 59.1% | 5/320 | 2/10 |

**Spread 20.3 pp against a 13.6 pp checkpoint-to-checkpoint sd — all three sit
within ~1.5 sd of one another.** After ~15,000 post-growth steps across two
separate promotions, on two different corpora, with and without the load
balancer, capability is statistically indistinguishable from the unchanged base.
Both grown models read *lower* than the control, though not significantly.

**⇒ Growth is mechanically sound (G1) and delivered nothing measurable (G2).**
That is now four independent lines of evidence pointing the same way:

1. Experts converge to ~90% twins — asymptotic, at any token budget (fit 0.893)
2. Router perturbation to cos 0.704 made differentiation *slower*, not faster
3. Removing the uniformity controller changed the curve by −0.003
4. Capability is flat against the pre-growth base, twice

### The one thing growth could never have done, restated

```
activated params, 24 experts:  180,726,115
activated params, 48 experts:  180,726,115
```

Only 4 of 48 fire per token. Expert-count growth adds storage the model cannot
reach within a token, so it was never able to move the quantity the capacity
hypothesis identifies. **Net2Wider is the only growth axis that raises activated
params**, and it is the one that has never been built.

### ⇒ Recommended base for Net2Wider: `checkpoints_base/step_0157000.pt`

Not the 397M. The grown models measure no better, cost 34% more per checkpoint
(4.5 GB vs 3.35 GB), and carry 24 experts that two controlled experiments say
will not differentiate. Nothing measurable is lost by starting from the 278M.

## 2026-09-01 — ✅ BALANCER RULED OUT. G2b is established: STOP ADDING EXPERTS.

The cleanest A/B this project has run: same base (`growth_v2/step_0003000`), same
corpus, same every other flag. **One variable: `--router-bias-lr 0.0` vs 1e-3.**
`growth_v2` is the balancer-on control from that exact checkpoint.

### Result — the balancer was NOT manufacturing the twins

| step | cos(gate), balancer OFF | balancer-ON prediction | delta |
|---|---|---|---|
| 3,000 | 0.9601 | 0.9578 | +0.0023 |
| 4,000 | 0.9494 | 0.9513 | −0.0019 |
| 5,000 | 0.9439 | 0.9469 | −0.0030 |
| 6,000 | **0.9407** | 0.9438 | **−0.0031** |

−0.003 is nothing. Removing the uniformity controller entirely changed the
differentiation curve not at all.

### And utilisation stayed balanced WITHOUT the controller

```
balancer OFF   cv = 0.360  0.294  0.571  0.193  0.198   (bias L2 frozen 4.569)
balancer ON    cv = 0.419  0.547  0.236  0.234          (bias L2 climbing)
```

No collapse, no dying experts, min% never below 0.5%. **The model self-balances.**

**⇒ Two independent explanations for the twinning are now eliminated:
initialisation (growth v2, router perturbed to cos 0.704) and the uniformity
controller (this run). The convergence is INTRINSIC.** G2b stands on evidence.

### ⚠ Limit of this test

`--router-bias-lr 0` stops the *update*; it does not zero the accumulated bias,
which stayed frozen at L2 4.569. So the DYNAMIC suppression hypothesis is
refuted — nothing was actively pushing the twins together — but a static offset
effect is untested. Given cv sits near 0.2 unforced, that offset appears close
to what the model wants anyway.

### Free side-finding: the aux-loss-free balancer is doing almost no work

Utilisation is near-uniform with the controller off. The DeepSeek-V3 machinery
is not earning its place at this scale, and **`cv` is a weaker health signal
than it looked** — it reads uniform whether or not anything is enforcing it.
Do not treat a healthy `cv` as evidence that routing is doing something.

### ⇒ NEXT AXIS: Net2Wider, and for a measured reason

Expert-count growth cannot raise the quantity the capacity hypothesis says binds:

```
activated params, 24 experts:  180,726,115
activated params, 48 experts:  180,726,115
```

Only 4 of 48 fire per token. Net2Wider widens the experts that DO fire, so it is
the only axis that moves activated params. Function-preserving under SiLU/SwiGLU
(duplicate a unit, halve its outgoing weight). `grow_width.py` does not exist —
roadmap estimates ~2 sessions, pure dev time, no capital.

## 2026-08-31 (late) — ⚠️ TWO CORRECTIONS, and the balancer hypothesis

### 1. The pre-growth pours WERE already balanced across all four domains

I claimed a four-domain pour "has never been run" and argued the regression was
domain forgetting from narrow corpora. **Wrong — I reasoned from directory names
instead of contents.** `data_teacher_v2` is itself multi-domain:

```
data_teacher_v2   general 1976  math 1402  code 622   (per 4k sample)
data_teacher_med  medical 4000
```

So `newmix` (v2+med), the pre-growth pour, was **already all four domains and
genuinely balanced**:

| corpus | general | medical | math | code |
|---|---|---|---|---|
| **newmix (v2+med)** — pre-growth, 17,132 rows | **32.2%** | **29.3%** | **26.1%** | **12.4%** |
| **broadmix (all four dirs)** — what I proposed, 367,132 rows | 1.5% | 1.4% | 69.3% | 27.8% |

**The "broadened" corpus is not broader — it is 97% code+math**, far narrower in
effective balance than what was already run. And rung A closed with *"6,500
further steps at 0.45 regressed on every axis"* on the balanced mix.

**⇒ The owner's capacity hypothesis is better supported than the metric critique
suggested.** A balanced four-domain pour was run, it regressed, and the salad
returned in the text. That is not a two-checkpoint artifact.

### 2. Expert-count growth never raised the capacity that binds

```
activated params, 24 experts:  180,726,115
activated params, 48 experts:  180,726,115     <- UNCHANGED
```

Only 4 of 48 fire per token, so promotion added *storage* while leaving
*per-token capacity* untouched. If the model trades because it cannot retain
within the capacity it applies to each token, adding experts it cannot reach was
never going to help — which is exactly what two promotions measured. **This makes
the growth failure consistent with the capacity hypothesis rather than a
refutation of it.** Net2Wider is the only axis that raises activated params.

### 3. THE BALANCER HYPOTHESIS — and a 29x claim I withdrew within the hour

More tokens will NOT differentiate the current twins. Exponential fit on the
grown48 curve: **asymptote cos 0.893**, predicted 0.8939 at 20,000 steps and
0.8930 at 200,000. The room exists; nothing fills it.

So why is there an asymptote? The DeepSeek-V3 aux-loss-free balancer runs every
step with a **uniform** target, `bias[i] += lr * sign(mean_count - count[i])`,
at a hardcoded `cfg.router_bias_lr = 1e-3` that was **never exposed as a flag**.
Measured on both grown checkpoints:

| checkpoint | content logit spread | router_bias spread | ratio |
|---|---|---|---|
| step_0008696 | 2.83 | 3.03 | **1.07** |
| step_0003000 | 3.48 | 2.48 | **0.71** |

Roughly **half** of what decides which experts fire is a controller whose only
goal is that they fire equally often. On a grown model whose new experts begin as
exact clones, forcing them to stay equally USED is a plausible mechanism for
forcing them to stay equally TRAINED.

⚠️ **I first measured this ratio at 29x** and concluded routing was
content-blind. That assumed unit-norm router inputs; the router actually reads an
RMSNorm output with norm ~27 (`main.py:1116`). Corrected to ~1x before anything
was built on it. The balancer is a co-driver, not a blindfold — and the dramatic
version of this claim was wrong.

**Testable for one argparse line:** `--router-bias-lr` now exists, and
`run_balancer_test.sh` is the cleanest A/B this project has had — same base
checkpoint, same corpus, same every other flag, one variable. `growth_v2` is the
balancer-on control and its fitted curve predicts cos 0.944 at step 6,000.

### 4. The schema bug that silently halved the corpus

`data_teacher_med` and `data_teacher_v2` carried a `seed_len` column that
`code`/`math` did not. `load_dataset("json", ...)` locks its schema to the first
file and refuses to cast the rest, so **med and v2 were dropped entirely** — 410
skipped batches in 50 minutes while the corpus banner printed all four directory
names. `run_grown48_broadmix.sh` had therefore never worked. Fixed by
`tools/normalize_teacher_shards.py`.

**Third bug this week whose signature is "logs a warning and keeps running while
training the wrong thing"**, after growth_metadata dropped on save and the broken
`-c` glob in the eval tail. Startup assertions for this class are owed.

## 2026-08-31 (evening) — ❌ GROWTH v2: router symmetry was NOT the ceiling. STOP GROWING EXPERTS.

Pre-registered test, gate written before the run: re-promote 24→48 with the
router rows perturbed (`--router-perturb-scale 1.0`, cos 0.704 at step 0 against
the old promotion's 1.000), pour 3,000 steps, and require the expert
differentiation curve to sit BELOW the old one AND still be falling.

**It failed, and it failed upward** — v2 differentiated *slower*:

| step | v2 cos(gate) | old | delta |
|---|---|---|---|
| 0 | 0.9988 | 1.0000 | −0.0012 |
| 1,000 | 0.9811 | 0.9822 | −0.0011 |
| 2,000 | 0.9646 | 0.9598 | **+0.0048** |
| 3,000 | 0.9601 | 0.9433 | **+0.0168** |

### The finding, and it does NOT depend on the cross-run comparison

Within v2 alone — one run, no confound:

```
step      0    1000    2000    3000
cos(router) 0.7041  0.6186  0.5465  0.5323   <- DIVERGING
cos(gate)   0.9988  0.9811  0.9646  0.9601   <- CONVERGING, decelerating
Δcos(gate)/500      -0.0139 -0.0063 -0.0016
```

**The router pulled the twins further apart while the experts drifted back
together.** They are provably receiving different token distributions —
routing directions only 53% aligned — and they still learn the same function.

**⇒ Symmetry was never the bottleneck. Initialisation is ruled out.**

### What that leaves, and it is the v5 conclusion on the current lineage

If routing diversity does not buy expert diversity, the task does not contain 48
distinguishable sub-functions at this scale and token budget. Every expert
converges to roughly the same general-purpose FFN. That is exactly the
2026-06-06 v5 post-mortem — "at this scale the model can't find distinct work
for the experts" — reproduced on the fresh lineage at 48 experts, this time with
**initialisation eliminated as the explanation**.

The 2026-06-29 retraction of "MoE growth is tapped out" was right about its
*reason* (v5's ceiling was token dilution on a collapsed base) and is not
contradicted here. This is a different and better-controlled result: healthy
base, 142M post-growth tokens, function-preserving promotion, symmetry broken on
purpose — and the experts still converge.

### ⇒ DECISION: stop adding experts. Widen them instead.

`docs/growth_design.md` lists four axes. MoE width is now measured out on this
lineage. **Net2Wider is the indicated next axis and it is the right shape of fix
for this specific failure**: it widens the experts that already exist rather than
adding more for the model to find work for. It is function-preserving under
SiLU/SwiGLU (duplicate a unit, halve its outgoing weight), and `grow_width.py`
does not exist yet — roadmap estimates ~2 sessions, pure dev time.

### ⚠ My confound, recorded

v2 ran the broadened corpus (code+math+med+v2) where the old run was code+math
only, so the cross-run delta column above conflates router perturbation with
corpus. The within-v2 dissociation is what carries the conclusion, and that is
single-run. The corpus half of the test is unresolved, which is fine — it was
never what this run was for.

Also fixed to get here: the teacher shards carried two different schemas and
`med` + `v2` were being dropped entirely (410 skipped batches, corpus banner
still printing all four directories). See `tools/normalize_teacher_shards.py`.
**`run_grown48_broadmix.sh` had never actually worked.**

## 2026-08-31 — 🔬 G2 READOUT: the leg is FLAT, and the two-checkpoint protocol is BROKEN

First capability measurement of the grown 397M model. The headline is not about
growth — it is that **the way this project has been measuring all month cannot
support the conclusions drawn from it.**

### The five-point sweep (n=320 each, bare framing, T=0.4 pen=1.15 seed=1234)

| step | L3+ | ±95% | L4 |
|---|---|---|---|
| 0 (= bit-exact 278M base) | **68.8%** | 5.1 | 9/320 |
| 2,000 | **31.2%** | 5.1 | 15/320 |
| 4,000 | **65.6%** | 5.2 | 3/320 |
| 6,000 | 58.4% | 5.4 | 8/320 |
| 8,696 (final) | 48.4% | 5.5 | 12/320 |

**L3+ mean 54.5%, sd 13.6 pp, range 31.2–68.8. Checkpoint-to-checkpoint variance
is 5.1x the per-measurement sampling noise.** Consecutive swings: −37.6, +34.4,
−7.2, −10.0 pp, between checkpoints 2,000 steps apart in ONE run.

### ⇒ The verdict: the leg did nothing measurable

Mean-based, which is what the variance demands:

* early (0, 2,000) **50.0%** → late (6,000, 8,696) **53.4%** = **+3.4 pp**
* against a within-leg sd of **13.6 pp**. Flat.
* L4 across the leg: 9, 15, 3, 8, 12 — counts of 320, 95% CI ~±6 samples. Noise.

**G2 is UNRESOLVED, not failed.** 8,696 steps of code+math on the grown model
moved neither metric.

### ⚠️ TWO CLAIMS MADE TONIGHT AND WITHDRAWN THE SAME NIGHT

1. **"Significant regression, z=−5.22, p<0.0001."** WRONG — an endpoint
   artifact. That test compared step 0 (68.8%) against step 8,696 (48.4%)
   counting only *sampling* noise inside each checkpoint. Once
   checkpoint-to-checkpoint variance is included the gap is ~1.3 sd. Both
   endpoints happen to sit far from the mean, in opposite directions.
2. **"L3+ and L4 are anti-correlated."** WRONG — read off four points. Across
   all 15 bare-framing n=320 evals, **r = −0.178**. Not significant.

### ⇒ THE PROTOCOL CHANGE, and it invalidates more than this leg

**Every "trade" documented this month was read off TWO checkpoints:** code
corpus 76.6→54.1, more steps 54.7→77.5, 157,238 vs 163,238. Against a metric
with **13.6 pp checkpoint-to-checkpoint sd**, those are all ~1 sd moves. The
trade pattern that motivated the entire growth programme may substantially be an
artifact of single-checkpoint comparison.

**Rule from here: compare MEANS over >=3 checkpoints per condition, with the
spread quoted. Never endpoints.** A single checkpoint's L3+ is one draw from a
distribution ~37 pp wide; it is not a property of the run.

Also: `step_0157000` was NOT equivalent to the evaluated `157,238` as
`run_grown48.sh` presumed — measured within-session it reads L4 **9/320** where
the archive records **26/320**. The header flagged it as an unevaluated
stand-in; that caution was warranted.

### Why growth could not have shown up here anyway — the experts are still twins

Measured on `step_0008696.pt`, each new expert against the parent it was cloned
from:

```
mean cos(gate) 0.909   mean cos(up) 0.912   |down_new| / |down_parent| 0.404
```

They **woke up** (`down` grew from a zeroed init to 40% of parent magnitude) but
never **differentiated** — still ~91% aligned with their twins. Balanced routing
(`cv` 0.206) was never evidence of distinct work.

The cause is dose, and it is arithmetic:

| | tokens seen per expert |
|---|---|
| original 24, pre-growth (topk 4 of 24) | **430M** |
| new 24, post-growth only (topk 4 of 48) | **11.9M** |

**36x less**, and `perturb_scale` at promotion was **0.0** — so they began as
*exact* clones with only SGD noise to separate them. `grow.py`'s own docstring
says to raise it to ~1e-3 to accelerate divergence. Do that on the next
promotion.

**⇒ The 397M model is currently 24 trained experts plus 24 half-strength echoes.
That is not a test of the capacity hypothesis; the capacity does not exist yet.**

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
