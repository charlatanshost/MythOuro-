# MythOuro Clean SFT Dataset Registry

Tracks candidate SFT datasets and their **verified provenance**, to clear the
licensing gate in the roadmap ("Licensing & data provenance"). The *legacy* SFT
mix (OpenHermes / Magicoder / MetaMathQA) contains OpenAI-generated data, which
constrains distribution — but **removing those was necessary, NOT sufficient.**

## ✅ Provenance follow-up (2026-07-28) — the 06-20 "verify via paper" items, closed

The 2026-06-20 table below is **AUTHORITATIVE over the 2026-06-11 entries further
down.** Evidence: the entries were compiled largely from dataset cards, and cards
under-disclose — **MIRIAD** looked fine at 06-11 and the 06-20 pass caught it as
OpenAI-generated; **Tulu-3** likewise (see below). Treat any 06-11 description as
a lead, not a verdict.

| dataset | 2026-07-28 finding | status |
|---|---|---|
| **Tulu-3** (`clean_general`, 30%) | **CARD-VERIFIED CONTAMINATED.** The 18 subsets include **"WildChat GPT-4" — 100,000 samples (~10.6% of 939,344)**. The 06-11 entry's "generated strictly via open-weight models" is **FALSE**. Unfiltered, this put ~3.2% OpenAI data into the SFT mix. Card also notes "some portions ... are non-commercial" (ODC-BY-1.0). | ⚠️ **FILTER IMPLEMENTED** — `_to_messages_tulu` drops `wildchat` + `evol` subsets at ingestion (`mythouro/sft_data.py`). |
| **OpenCodeInstruct** (`clean_code`, 20%) | **CLEARED — paper-verified** (arXiv 2504.04030): instructions by **Qwen2.5-32B-Instruct**; solutions by **Qwen2.5-Coder-32B-Instruct / Qwen2.5-32B-Instruct / QwQ-32B-Preview**; quality-judging + unit-test generation by Qwen2.5-32B. **No OpenAI models at any stage.** | ✅ **use** |
| **ChemData700K** (`clean_chem`) | ❌ **CONTAMINATED — REMOVED (paper-verified, arXiv 2402.06852 v1).** "Template-based" was only the SEED step: *"we utilize **GPT-4** … to generate a range of Q&A pairs that, while semantically consistent, vary in their expression, creating what we refer to as a diverse template"*, and GPT-4 also authored the multi-turn dialogue *"scripts"* ("Play as Playwrights"). Construction, not evaluation. | ❌ **DROPPED 2026-07-28** |
| **NuminaMath-CoT** (`clean_numina`) | ❌ **CONTAMINATED — REMOVED.** Its constituent sources include **`orca_math` (153k of ~860k, ~18%)**, and Orca-Math-200K's card states *"all answers … generated using **Azure GPT4-Turbo**"*. The 06-11 entry's "filtered to exclude proprietary API generation" is **FALSE**. `synthetic_math` (167k, ~19%) is *also* undisclosed, so the tainted fraction may be larger. At 12% of the mix this was ~2.2% OpenAI-derived data. **Note NuminaMath-TIR also appears INSIDE Tulu-3** as a subset. | ❌ **DROPPED 2026-07-28** — 0.12 reassigned to `clean_math` (OpenMathInstruct-2, verified) |

**Bottom line after the 07-28 pass — the clean mix is now provenance-verified end to end:**

| source | weight | provenance |
|---|---|---|
| `clean_general` (Tulu-3) | 0.34 | ⚠️ contaminated, **filtered** (`wildchat`/`evol` subsets dropped at ingestion) |
| `clean_math` (OpenMathInstruct-2) | 0.32 | ✅ Llama-3.1-405B |
| `clean_code` (OpenCodeInstruct) | 0.22 | ✅ Qwen2.5 family, paper-verified |
| `clean_pubmedqa` | 0.12 | ✅ rule-based, pre-GPT |

**⚠️ FOUR FOR FOUR — every 06-11 card-derived claim that was independently checked
turned out contaminated** (MIRIAD, Tulu-3, NuminaMath-CoT, ChemData700K). The base
rate here is 100%. **Treat dataset-card provenance statements as leads requiring
paper-level verification, never as verdicts.** Two of the four were only catchable
by reading the paper (NuminaMath's `orca_math` subset; ChemData's GPT-4 template
expansion) — the cards alone would have passed both.

**🔴 OPEN GAP — chemistry / hard-science coverage is now ZERO.** Dropping ChemData
removed the only science source. PubMedQA was deliberately NOT scaled up to absorb
it (narrow yes/no biomedical QA; this doc already records that low-diversity
structured data drives SFT mode-collapse). **A clean science/chem replacement is
needed** — candidates must be verified at the paper level, not the card. Prefer
human-authored or rule-derived sources (exam sets, structured-DB templating with
NO LLM expansion step) over anything "instruction-tuned" by an unnamed model.

## ⚠️ Provenance verification (2026-06-20)

Per-dataset **card checks** (not just the 2026-06-11 schema/id probes) found OpenAI
provenance still present. **The mix must be filtered/rebuilt before the next SFT run.**

| Dataset | Provenance | Action |
|---|---|---|
| OpenAssistant `oasst2` | ✅ human-written (Apache 2.0) | **use** |
| PubMedQA `pqa_artificial` | ✅ rule-based auto-gen, pre-GPT (MIT) | **use** |
| OpenMathInstruct-2 | ✅ Llama-3.1-405B — open, not OpenAI (Apache) | **use** (keep GSM8K contamination guard) |
| **MIRIAD** | ❌ **OpenAI-generated** — card: S2ORC "input to **OpenAI's language models**"; + an OpenAI-ToS ban on medical-diagnosis use | **DROP** |
| **Tulu-3 SFT mixture** | ⚠️ contains **WildChat-GPT-4** + likely GPT-4o persona subsets + some CC-BY-NC | **subset-filter only** — keep human/open-model/permissive subsets; drop GPT-4/4o + NC. Do **not** ingest wholesale |
| **NuminaMath-CoT** | ⚠️ **generator undisclosed** on card; broader NuminaMath used GPT-4o → suspect | **verify generator** (arXiv) before use |
| OpenCodeInstruct | ⚠️ generator undisclosed on card (no OpenAI mention; NVIDIA NeMo-Skills, CC-BY-4.0) | verify via paper (arXiv 2504.04030) |
| ChemData700K | ⚠️ construction undisclosed on card (no OpenAI mention; MIT) | verify via paper (arXiv 2402.06852) |

**Medical-data gap (the important one):** MIRIAD was the core medical source and is
out. Clean replacements — **doing both:**
1. Human-authored exam sets **MedQA (USMLE) / MedMCQA** + the clean **PubMedQA**
   (verify MedQA/MedMCQA licenses at ingestion).
2. **Regenerate MIRIAD-style QA with an open model** (Llama/Qwen grounded on
   PubMed/S2ORC passages) — same approach, clean generator, full provenance control.

> Note: the inline "✅ verified" tags in the entries below mean **HF id/schema
> verified (2026-06-11)** — *not* provenance-verified. Provenance status is the
> table above.

> Compiled by the user, 2026-06-11. **All HF ids verified live by streaming
> probe on 2026-06-11** (schemas confirmed; adapters in `mythouro/sft_data.py`
> are written against the probed schemas). Verified ids: `allenai/tulu-3-sft-mixture`, `OpenAssistant/oasst2` (ingested via Tulu's
> converted slice instead — see notes), `nvidia/OpenMathInstruct-2`,
> `AI-MO/NuminaMath-CoT`, `nvidia/OpenCodeInstruct`, `qiaojin/PubMedQA`
> (pqa_artificial), `AI4Chem/ChemData700K`, `miriad/miriad-4.4M`.
>
> **IMPLEMENTED 2026-06-11**: `MixedSFTDataset(mix="clean")` is the DEFAULT
> (`training/sft.py --data-mix clean|legacy`), with per-source caps via
> non-streaming split slices, execution-status filtering on OpenCodeInstruct,
> and the GSM8K/ARC contamination guard ON by default (OpenMathInstruct-2 is
> augmented_gsm8k — verified live: 125,477 benchmark 13-grams indexed).

---

## 1. General Instruction & Multi-Turn Chat

### Tulu 3 SFT Mixture (`allenai/tulu-3-sft-mixture`)
* **License:** Apache 2.0 (individual subsets carry their own permissive
  licenses like CC-BY 4.0; the ingestion script should filter any legacy
  non-commercial inputs if strictly required, but the core mixture is cleared
  for open development).
* **Description:** Curated by AllenAI. A clean blend of human-written data
  (OpenAssistant, ScienceQA, FLAN) and synthetic instruction data generated
  strictly via open-weight models.
* **Primary use:** General alignment, multi-turn dialogue, core
  instruction-following.

### OpenAssistant Conversations (`OpenAssistant/oasst2`)
* **License:** Apache 2.0
* **Description:** ~130k-message human-to-human assistant corpus. Zero
  synthetic generation — completely immune to proprietary-model contamination.
* **Primary use:** Human-like dialogue styling, tone alignment, safe
  interaction boundaries.

## 2. Mathematical & Analytical Reasoning

### OpenMathInstruct-2 (`nvidia/OpenMathInstruct-2`)
* **License:** Apache 2.0 (per NVIDIA release; verify card at ingestion)
* **Description:** ~14M math question–solution pairs; all synthetic CoT
  generated with the open-weight Llama-3.1-405B-Instruct.
* **Primary use:** Deep mathematical reasoning, GSM8K-style optimization,
  multi-step numerical logic.

### NuminaMath-CoT (`AI-MO/NuminaMath-CoT` ✅ verified)
* **License:** Apache 2.0 / MIT (derived from open math competitions and
  public datasets)
* **Description:** 850k+ math problems with chain-of-thought solutions,
  filtered to exclude proprietary API generation.
* **Primary use:** Advanced problem-solving, olympiad-level math reasoning.

## 3. Coding & Software Engineering

### OpenCodeInstruct (`nvidia/OpenCodeInstruct` ✅ verified — ships unit-test execution status; failing samples filtered at ingestion)
* **License:** Permissive / open source (check subset alignment at ingestion)
* **Description:** ~5M coding instruction samples using OSS-Instruct-style
  methodology via open architectures, with **compiler execution feedback
  loops** verifying functional correctness.
* **Primary use:** Multi-language syntax, debugging, algorithms.

## 4. Medical & Life Sciences

### MIRIAD (`miriad/miriad-4.4M`) — ❌ **DROP: OpenAI-generated** (provenance check 2026-06-20)
* **License:** ODC-By v1.0 — **but** the card adds an OpenAI-ToS restriction: the
  outputs **must not be used for medical diagnosis or clinical decision-making about
  real individuals** — doubly disqualifying for this project's medical use.
* **❌ Provenance:** the QA pairs were **generated by OpenAI's language models**
  (card: *"we used S2ORC documents as input to OpenAI's language models…"*). S2ORC
  was only the *grounding*; the *generator* is OpenAI → fails the OpenAI-free gate.
  The earlier "no proprietary model knowledge" note described the input, not the
  generator. **Replace** — see the medical-data-gap plan in the verification section
  above (MedQA/MedMCQA + PubMedQA, and/or regenerate with an open model).
* **Primary use:** Clinical knowledge, biomedical definitions, diagnostic
  reasoning.

### PubMedQA (`pubmed_qa` / `qiaojin/PubMedQA`)
* **License:** MIT
* **Description:** Gold-standard biomedical QA on authentic peer-reviewed
  PubMed abstracts ("yes/no/maybe" + textual evidence).
* **Primary use:** Medical literature comprehension, truthfulness validation.

## 5. Engineering, Physics, & Hard Sciences

### OpenWebMath SFT pipeline / AutoMATH components (not yet ingested — no canonical id; revisit if the hard-sciences ratio needs more depth)
* **License:** MIT / Apache 2.0 (varies by mirror; derived from open-access
  scientific text).
* **Description:** SFT data from open-access engineering/math/physics papers,
  converted into structured markdown + code-execution problems.
* **Primary use:** Hard-sciences intuition, formula evaluation.

### ChemLLM SFT components (`AI4Chem/ChemData700K` ✅ verified)
* **License:** Apache 2.0 / CC-BY 4.0
* **Description:** Structured chemical files (SMILES strings) mapped to
  natural-language descriptions of molecular behaviour and reactions.
* **Primary use:** Chemical informatics, organic chemistry logic.

---

## Ingestion instructions

1. **Source filtering:** parse only the designated permissive sub-directories
   of these datasets.
2. **Text normalization:** sanitize math syntax across the math / physics /
   engineering blocks; standardize on inline and block LaTeX delimiters.
3. **De-duplication:** run MinHash LSH dedup when mixing OpenMathInstruct-2 +
   NuminaMath-CoT so repetitive math formats don't over-saturate the SFT mix.

## Pipeline integration notes (added at registration)

- **The tooling for #3 already exists**: `data/dedup.py` (MinHash LSH,
  Llama-3-convention defaults) — point it at the merged math subsets.
- **Run `data/contamination.py` against the eval benchmarks** before training:
  OpenMathInstruct-2 / NuminaMath train *toward* GSM8K-style problems and we
  EVAL on GSM8K — the 13-gram contamination filter exists for exactly this.
- **Schema adapters:** new sources slot into `MixedSFTDataset` the same way
  OpenHermes/MetaMath did (per-source adapter + ChatML render + loss mask).
- **Scale sanity:** 14M (OpenMathInstruct-2) and 5M (OpenCodeInstruct) samples
  dwarf our step budgets — subsample per source; mix ratios are the lever, per
  the roadmap's "variety over volume" finding.
- **What this unlocks:** an SFT'd checkpoint with **no OpenAI-provenance
  constraint** — the first distributable-grade MythOuro, per the roadmap's
  licensing section.
