# Teacher & data curriculum — the sequenced ladder

**Status: STRATEGY, 2026-07-21.** Consolidates a plan that was scattered across
the tokenizer-graduation deep-dive (ideas.md), the "distill from larger models"
idea, and the Nemotron MOPD reference note. Answers: *when the current teacher
stops helping, what do we do — and in what order?*

Context (2026-07-21): the R=0.2 A/B verdict showed teacher-generated data moves
the plateau that fresh web data couldn't (tracker 2026-07-21). So the teacher
corpus is now the primary quality lever — and this doc plans its whole lifecycle.

## The core principle (owner, 2026-07-21)

> **Exhaust cheap variety before expensive swaps. Only switch to a new teacher
> once the student has become EQUAL to the current teacher.**

Rationale: a 2.6B teacher holds far more than a 278M student has extracted
through one narrow seed distribution. Most of the teacher's value is still
locked up. Switching teachers early (or growing the model early) spends capital
to solve a problem cheap data variety hasn't been given a chance to solve yet.

## The ladder (climb in order; each rung gated on the one below being exhausted)

### Rung 1 — New seed DOMAINS, same teacher (cheap, lots of runway)

Vary *what we ask Ouro to write about*, not who writes it. New seed corpora →
teacher generates in new territory, same pipeline (`gen_teacher_corpus`, just a
new `_DATASET_SPECS` entry / seed source):
- broaden general prose (books/Gutenberg-class → discourse coherence)
- **medical text** → the endgame domain; teacher writes clinical prose
- textbook/explanatory corpora → phi-style "teach by explaining"
- targeted code/math regeneration (the fixed random-window seeding should make
  these *good* now — the 07-21 boilerplate bug is fixed)

This rung likely has **months** of runway. Cost: seed sourcing + harvest time.
No model changes, no vocab issues, no capital.

### Rung 2 — Grow the STUDENT (gated: Rung 1 exhausted AND student < teacher)

If teacher-domain variety plateaus *and* the student is still visibly weaker
than the teacher (see "How to measure parity"), the wall is **student capacity**,
not data. Fix: Net2Wider growth toward ~1B (roadmap Path A; growth infra already
built). A bigger student can (a) absorb more of the *same* teacher, and (b) is a
prerequisite for Rung 3 — you don't graduate to a bigger teacher a small student
can't keep up with.

### Rung 3 — New (bigger) TEACHER (gated: student ≈ current teacher)

**Only when the student has reached parity with Ouro.** The blocking constraint,
already documented (roadmap "Tokenizer graduation"): **logit distillation needs
a MATCHED tokenizer.** You cannot point the student at Qwen/Llama/etc — different
vocab → the soft-KL signal has nothing to align against. Options:
- **Same-vocab bigger teacher** if one exists in Ouro's tokenizer family — a
  drop-in swap, cheapest Rung-3 path.
- **Tokenizer graduation** otherwise: re-anchor the student's vocab to the new
  teacher's family (transplant embeddings/head, heal with short training), then
  clean logit KD resumes. Real work; the gated milestone.
- **Multi-teacher (MOPD-style, Nemotron ref):** blend Ouro (recurrent-depth
  signal) + a bigger dense teacher (capacity) once vocab is shared. Elegant way
  to keep Ouro's depth signal while adding a stronger teacher.

## How to measure "student ≈ teacher" (the Rung-3 gate)

The switch criterion needs a concrete signal, not a vibe. Three, in order of
directness — we already have machinery for all of them:

1. **Distillation KL → floor.** The soft-KL loss on held-out text *is* the
   student-teacher distribution gap. When it stops falling and sits near its
   irreducible floor (the student can't get closer given capacity), the student
   has absorbed what this teacher can transfer. Already logged every step.
2. **α=0.0 converges to α=0.7 in the probe.** The rollout probe already measures
   the student alone (α=0.0) vs teacher-dominated (α=0.7). When α=0.0 generation
   quality catches up to α=0.7 — i.e. the student no longer *needs* the teacher's
   guidance to produce teacher-quality text — it has internalized the teacher.
   This is the most interpretable parity signal and needs no new tooling.
3. **Eval parity.** Student and teacher score comparably on the same held-out
   perplexity / downstream tasks. Coarsest, but the external-facing number.

**Important asymmetry:** #1/#2 can plateau *above* parity — that's the Rung-2
signal (capacity-limited, grow the student), NOT the Rung-3 signal. Parity means
the gap *closed*, not that progress *stalled*. Distinguishing "closed the gap"
from "hit my ceiling below the teacher" is the whole diagnostic:
- gap closed (KL near a low floor, α=0.0 ≈ α=0.7) → **Rung 3** (new teacher)
- progress stalled with gap still open (α=0.0 << α=0.7) → **Rung 2** (grow student)

## One-line summary of the sequence

new seed domains → (plateau + gap open) grow student → (plateau + gap closed)
graduate teacher → repeat. Cheap levers first; capital only when a measured gate
says the cheap lever is spent.

## Cross-refs

- Rung-1 mechanics: `docs/teacher_corpus_plan.md`, `tools/gen_teacher_corpus.py`
- Rung-2 growth: roadmap "Path A / Net2Wider", `mythouro/grow.py`
- Rung-3 vocab: roadmap "Tokenizer graduation", ideas.md deep-dive + Zett/FOCUS/
  WECHSEL reference-shelf entry; multi-teacher = Nemotron MOPD ref (ideas.md)
