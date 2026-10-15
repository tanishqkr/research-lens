# Project Status: Scientific Contradiction Finder

**Current Date:** 2026-10-12  
**Status:** Implementation Frozen & Verified (Milestones 1–4, Remediation Phase 1, and Acceptance Audit Completed)

---

## 1. Project Freeze Declaration

Following the successful completion of Milestones 1 through 4, Remediation Phase 1 (resolving directional classifier symmetry, comparative arm slot attribution, template data leakage, and genuine ablations), and the final read-only acceptance audit, the implementation of the **Scientific Contradiction Finder** is formally **frozen**.

No further modifications to schemas, extraction heuristics, relation classifiers, retrieval adapters, scoring formulas, or UI components are planned. All ongoing documentation directly reflects the verified, reproducible state of the codebase.

---

## 2. Verified Technical Milestones

- [x] **Milestone 1: Pipeline Integrity & Evidence Extraction**
  - Robust retrieval diagnostics with provider error handling, SQLite persistent caching, and bounded exponential backoff.
  - Verbatim finding extraction tracking character start/end offsets with exact substring provenance against source text.
  - Distinguishing primary empirical studies from systematic reviews and meta-analyses.
- [x] **Milestone 2: Scientific Reasoning & Condition Extraction**
  - Condition slot extraction for Intervention ($I$), Comparator ($C$), Outcome ($O$), and Setting ($S$).
  - Correct multi-arm and comparative slot attribution (e.g. LCDs vs LFDs, VLCDs, active comparators).
  - Fixed directional conflict symmetry in `classifier.py` (`is_fb_active`).
  - Safe abstention on sparse evidence, ambiguous comparators, and unanchored $\ge 3$-arm trials.
  - Cautious qualification of observational associations vs. randomized controlled trials to avoid asserting ungrounded causal contradictions.
- [x] **Milestone 3: Benchmark Integrity & Complete Evaluation**
  - Curated and validated the **Independent Primary Literature Benchmark** ($N=35$) with verified DOIs/PMIDs and explicit adjudication criteria.
  - Repaired template leakage in CAS-Con150 via grouped claim-family splitting (`leakage_safe=True`).
  - Implemented genuine feature-disabled ablations (`enable_setting_gating`, `enable_incomparability_check`) replacing post-hoc string overwrites.
  - Fully documented confusion matrices, per-class metrics, error cases, and principal limitations in [`docs/EVALUATION.md`](file:///Users/tanishporwal/Desktop/scientific_contradiction/docs/EVALUATION.md).
- [x] **Milestone 4: Product Hardening, UI Wiring & Reproducibility**
  - Next.js 15 App Router research workbench with TypeScript, Tailwind CSS, and Framer Motion (`frontend/`).
  - FastAPI ASGI server (`src/scientific_contradiction_finder/api/app.py`) with complete retrieval diagnostics, triage logs, and study comparisons.
  - Streamlit verification dashboard (`src/scientific_contradiction_finder/ui/app.py`) wired to backend fields with query state isolation.
  - Offline-first execution requiring $0 external API spend.
  - Clean reproducible test harness via `uv`.

---

## 3. Verified Metrics & Test Results

### Automated Test Suite & Code Quality
- **Unit & Integration Tests:** **139 passed** in ~84s (`uv run pytest`).
- **Static Analysis & Linting:** Ruff **100% clean** (0 errors, 0 warnings).
- **Frontend Quality:** TypeScript strict (`tsc --noEmit`), ESLint, and Next.js production build **100% clean**.
- **Execution Script:** `uv run python scripts/run_full_evaluation.py` executes deterministically and updates `data/evaluation_summary_corrected.json`.

### Primary Headline Evaluation: Independent Literature Benchmark ($N=35$)
- **Overall Accuracy:** **88.57%** (31 / 35 pairs)
- **Macro F1:** **0.8821**
- **Contradiction Precision:** **100.0%** (4 / 4 predicted contradictions; zero false contradictions)
- **Contradiction Recall:** **57.1%** (4 / 7 ground-truth contradictions detected; 3 cautiously qualified due to cohort/design differences)
- **Unconditioned TF-IDF Baseline:** 37.14% Accuracy, 0.3024 Macro F1
- *Scope Caveat:* The independent benchmark contains 35 pairs (7 per class). These results serve as **preliminary empirical evidence**, not statistical proof of universal generalization across scientific literature broadly.

### Synthetic Reference Suite: CAS-Con150 ($N=150$)
- **Leakage-Safe Split ($N_{\text{test}}=46$):** Condition-Aware Model achieves **100.0% Accuracy** (Macro F1 1.0000) vs. TF-IDF Baseline at **19.57% Accuracy** (chance level).
- *Synthetic Caveat:* Built from 29 base templates; historical scores (93.3% or 100%) reflect synthetic behavior and must not be cited as headline real-world performance.

---

## 4. Principal Documented Limitations

1. **Missed Genuine Contradictions:** High contradiction precision (100%) comes at the expense of recall (57.1%), as conservative thresholds qualify opposing findings when cohorts or comparators diverge.
2. **Cautious Qualification of Observational vs. RCT Comparisons:** The pipeline hedges against false equivalence between observational associations and randomized clinical trials.
3. **Annotation Ambiguity:** In complex clinical domains, expert adjudications may diverge on whether differing baselines represent true contradictions or expected clinical effect modifications.
4. **Multi-Arm Attribution Constraints:** The extractor abstains (`arm_name: None`) on trials with $\ge 3$ arms lacking explicit prepositions.
5. **Reliance on Open-Access Abstracts:** The system cannot access paywalled full texts (Elsevier, Wiley), relying on open-access abstracts that may omit secondary endpoints or subgroup details.

---

## 5. Repository Integrity & Checkpoints

- **Pre-Remediation Checkpoint:** Commit `a0581cd` (Tag: `checkpoint-milestones-1-4-pre-remediation`, Branch: `checkpoint/milestones-1-4-pre-remediation`)
- **Remediation Commit:** `cf098c9`
- **Attribution Refinement Commit:** `b008ff3`
- **Current Branch:** `main` (clean working tree)
- **Remote Push:** None performed (strictly local repository preservation)
