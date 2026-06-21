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
