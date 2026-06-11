# MythOuro Clean SFT Dataset Registry

Tracks high-quality, commercially viable SFT datasets **entirely free of
OpenAI-output provenance** — this registry clears the licensing gate
documented in the roadmap ("Licensing & data provenance"): the current SFT mix
(OpenHermes / Magicoder / MetaMathQA) contains OpenAI-generated data,
constraining distribution. Everything here was generated or verified via
open-weight models (Llama 3.1, Qwen 2.5) or execution loops.

> Compiled by the user, 2026-06-11. **Verify exact HF ids + license text at
> ingestion time** — ids marked ⚠ need confirmation.

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

### NuminaMath-CoT (⚠ verify id — likely `AI-MO/NuminaMath-CoT`)
* **License:** Apache 2.0 / MIT (derived from open math competitions and
  public datasets)
* **Description:** 850k+ math problems with chain-of-thought solutions,
  filtered to exclude proprietary API generation.
* **Primary use:** Advanced problem-solving, olympiad-level math reasoning.

## 3. Coding & Software Engineering

### OpenCodeInstruct (⚠ verify id — likely `nvidia/OpenCodeInstruct`)
* **License:** Permissive / open source (check subset alignment at ingestion)
* **Description:** ~5M coding instruction samples using OSS-Instruct-style
  methodology via open architectures, with **compiler execution feedback
  loops** verifying functional correctness.
* **Primary use:** Multi-language syntax, debugging, algorithms.

## 4. Medical & Life Sciences

### MIRIAD (⚠ verify id) — Medical Instruction and RetrIeval Dataset
* **License:** ODC-By v1.0 (permits commercial modification/reuse/distribution
  with attribution).
* **Description:** Million-scale structured medical QA; synthetic generation
  used input context strictly from the Semantic Scholar Open Research Corpus
  (S2ORC) — no proprietary model knowledge.
* **Primary use:** Clinical knowledge, biomedical definitions, diagnostic
  reasoning.

### PubMedQA (`pubmed_qa` / `qiaojin/PubMedQA`)
* **License:** MIT
* **Description:** Gold-standard biomedical QA on authentic peer-reviewed
  PubMed abstracts ("yes/no/maybe" + textual evidence).
* **Primary use:** Medical literature comprehension, truthfulness validation.

## 5. Engineering, Physics, & Hard Sciences

### OpenWebMath SFT pipeline / AutoMATH components (⚠ verify specific mirrors)
* **License:** MIT / Apache 2.0 (varies by mirror; derived from open-access
  scientific text).
* **Description:** SFT data from open-access engineering/math/physics papers,
  converted into structured markdown + code-execution problems.
* **Primary use:** Hard-sciences intuition, formula evaluation.

### ChemLLM SFT components (`AI4Chem/ChemData` ⚠ verify)
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
