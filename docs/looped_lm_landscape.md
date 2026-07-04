# Looped-LM landscape — adopt / fork / preserve (2026-07-04)

**Purpose.** A solo-planner's map of the (fast-growing) looped / recurrent-depth LLM literature:
what each neighbor does, which techniques are worth folding into MythOuro's **future**
architecture, the one strategic fork the space is forcing, and what stays uniquely ours. This is a
**forward architecture-planning artifact**, not a training to-do — the current bottleneck is
coherence (tokens + the landed fixes) and nothing here moves that. But MythOuro is planned by one
person, so the future architecture has to be planned *now*; this externalizes it so it isn't all
in-head.

**Vetting caveat.** Entries are **abstract-level reads** — several are 2026 papers past the
Jan-2026 model cutoff, surfaced via AI summaries that mis-stated at least one (see Hyperloop below:
an overview called it "parallel paths"; the paper is within-trajectory residual enrichment).
Confirm each mechanism against the full paper before any writeup or adoption. The *conclusions* are
trusted because two independent passes vetted them (my read + owner's firsthand corrections, e.g.
Xe Link); the *paper internals* still want a full-text check.

## 1. The landscape — who works which axis

Everyone is working the **same looped / weight-shared primitive**, but on **different axes**. None
does MythOuro's specific combination.

| Work | Axis | Mechanism (abstract-level) | Same as ours? |
|---|---|---|---|
| **Ouro** | (teacher) | recurrent-depth base we distil from | n/a — our teacher |
| **PLT / LoopCoder-v2** (2606.18023) | efficiency | parallelize/cheapen depth (CLP + shared-KV attn), "only loop once" | **No** — *opposite*: eliminates stranding |
| **Hyperloop** (2604.21254, Yoon Kim) | param-efficiency | matrix-valued residual streams via hyper-connections after each loop | **No** — within-trajectory residual enrichment, one output |
| **MELT** (per overview) | decode memory | share loop-1 KV across loops, selective update | **No** — memory trick; we already have KV machinery |
| **RRT / Continuous Depth-wise Batching** (2410.20672) | throughput | batch tokens at different loop depths through the shared block, ~2–3× | **No** — throughput; shares our *saturation premise* |
| self-consistency / best-of-N / deep ensembles | quality | sample N, aggregate | **Underpins** our Mode-B |
| MDASH (2026) | quality | structured multi-model disagreement | **Informs** our arbitration |
| DSpark (2026-06-27) | speed | speculative decoding (draft/verify) | **Our self-speculative child** |
| **MythOuro parallel-loops** | quality | **N *diverse* trajectories + per-token uncertainty arbitration** | — (ours) |

## 2. ADOPT — fold into the future model / serving

- **Hyper-connections (from Hyperloop) — the strongest fit for our core.** Enriches cross-loop
  residual flow (matrix-valued residual streams) for **minimal parameters** — exactly where our
  recurrent depth accumulates its computation, and squarely on our parameter-/token-efficiency
  thesis. Additive, orthogonal to the §3 fork. **Retrain-required → design into the next
  architecture rev.** Top adopt candidate. (Confirm the hyper-connections formulation, Xie et al.
  2026, against the full paper first.)
- **Continuous depth-wise batching (RRT) — throughput, no retrain.** Batch tokens/rollouts at
  different loop depths through the shared block to saturate the latency-bound card. The "go WIDE"
  principle made concrete for the **on-policy rollout generation at the Max phase**. Serving-time;
  adopt whenever, no architectural commitment. See
  [decode_kernel_optimization.md](decode_kernel_optimization.md).
- **MELT-style KV-sharing — memory, reference only.** Already handled (`CrossLoopKVCache`,
  `compress_kv_cache`); MELT's loop-1-share is an alternative to consult *if* we revisit decode
  memory. Not a gap.

## 3. FORK — the one strategic decision the space is forcing

The literature is bifurcating into two philosophies of the sequential-depth cost, and **our
designs sit on one side**:

- **Exploit the stranded compute** *(ours)* — keep depth sequential, monetize the idle compute:
  parallel diverse paths (quality — [parallel_loops.md](parallel_loops.md)), self-speculative
  depth (speed — [decode_kernel_optimization.md](decode_kernel_optimization.md) §6), depth-wise
  batching (throughput).
- **Eliminate the stranded compute** — PLT-style parallelized depth: make one trajectory fast,
  nothing left to monetize.

**These are in tension.** Adopt PLT-style parallel depth and the "diverse paths are ~free" premise
evaporates (the card is no longer idle). So this is a fork to pick *deliberately*, not drift into.

**Decision (2026-07-04): commit to EXPLOIT.** Rationale: the moat (§4) exists *only* on the
sequential-depth-with-exploitation side; going "eliminate" makes MythOuro a faster commodity
looped-LM and sheds every differentiator. Hyper-connections (§2) is adoptable regardless — it's
orthogonal to the fork. Revisit only if decode latency becomes a hard product blocker that
exploitation genuinely can't meet.

## 4. PRESERVE — the moat (in none of the above)

None of the surveyed work does these; they are what make MythOuro *itself*:

- **Per-token cross-path uncertainty arbitration** via a trained uncertainty head (parallel-loops).
- **The medical two-lane** design (calibrated uncertainty → the mission).
- **MoE-recurrent** combination (fine-grained routed experts inside the shared loop).
- The **stranded-compute framing** as an *original systems argument* for this architecture.

Keep these central. The space filling with looped-LM papers **validates the direction without
eroding the novelty** — treat neighbors as a **menu of base-primitive upgrades** (§2), a **fork to
hold** (§3), and a **moat to protect** (§4).

## 5. Integration into parallel-loops — which cost each kills

Parallel-loops ([parallel_loops.md](parallel_loops.md)) has two known costs (its §3): **N× compute**
and **N× memory**. Several neighbors are *precisely* the fixes — so they're the **efficiency
substrate** for the idea, not competitors. Most-solid first.

- **Depth-wise batching (RRT) → kills N× COMPUTE.** Paths are already batch rows; with ACT-budget
  diversity (shallow vs deep paths) they sit at *different loop depths*, which is exactly what RRT
  batches through the shared block. The throughput substrate for the depth-diverse variant —
  low-risk, breaks no diversity assumption.
- **KV-sharing (MELT + cross-path) → kills N× MEMORY.** MELT shares KV across loops *within* a
  path; and the N paths share the input prefix, so they can share KV *across* paths too —
  strongest in **Mode A** (paths aligned on the agreed token), weaker in Mode B (paths diverge).
  Attacks the "never free in memory" tax directly.
- **Self-speculative (DSpark) → fuses quality + speed via AGREEMENT.** Use cross-path agreement as
  the speculative-accept signal: agree → accept fast, skip deeper compute; disagree → spend compute
  + arbitrate. The §5-of-parallel_loops MDASH "agreement-as-signal" turned into a *speed* mechanism
  — arbitration-quality *and* speculative-speed from one confidence signal. **Heuristic (lossy)
  accept, not lossless rejection-sampling** — same family as the current `inference.py` early-exit.
- **Hyper-connections → orthogonal enrichment + a bolder bet.** Cleanly: enrich each path's block
  (better per-path representation, minimal params; stacks with all the above). Bolder: use the
  matrix-valued residual **lanes AS cheap micro-paths** and arbitrate across lanes instead of N
  full trajectories → sidesteps the N× cost entirely. **Rides on the parallel_loops §4 diversity
  crux** (do lanes stay decorrelated, or collapse to copies?) — big-if, big-payoff.
- **PLT → does NOT integrate.** The opposing fork (§3): parallelized depth eliminates the stranding
  the paths exploit, so folding it in is self-defeating (no idle compute → paths aren't free). Its
  KV-share *sub*-technique could be borrowed independently (overlaps MELT); its core cannot.

**Net:** RRT + MELT de-risk the two *known* weaknesses (compute, memory); self-speculative +
hyper-connections offer the quality benefit *more cheaply*. Parallel-loops stays the novel
differentiator; the neighborhood is the substrate that makes N-path ensembling **practical instead
of N×-expensive.** The two bolder syntheses (lanes-as-micro-paths, agreement-as-accept) ride on the
diversity-decorrelation question (parallel_loops §4) — validate that first. All stage-gated behind
coherence.

## See also

- [parallel_loops.md](parallel_loops.md) — the parallel-paths design + §7 prior-art (PLT, RRT, Hyperloop).
- [decode_kernel_optimization.md](decode_kernel_optimization.md) — §6 self-speculative (speed child) + depth-wise batching.
- [ideas.md](ideas.md) — experiment shelf.
