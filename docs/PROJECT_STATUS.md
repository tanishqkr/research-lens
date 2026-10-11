# Project Status: Scientific Contradiction Finder

**Current Date:** 2026-10-11  
**Phase:** Phase 0 (Repository Reconnaissance & Audit) -> Transitioning to Phase 1

---

## Phase Progression Checklist

- [x] **Phase 0: Repository reconnaissance and research audit**
  - [x] Inspect environment, Python 3.11, disk, memory, and dependencies.
  - [x] Protect private blueprint and startup documents in `.exclude/` and `.gitignore`.
  - [x] Create `.env.example` and safeguard secrets.
  - [x] Audit datasets (SciFact, Evidence Inference 2.0, SciNLI, CAS-Con150).
  - [x] Document ADRs, research audit, and configuration.
- [ ] **Phase 1: Data contracts, datasets and simple baselines**
  - [ ] Implement Pydantic data schemas (`Paper`, `Finding`, `ConditionSlots`, `EvidenceSpan`, `FindingPair`, `Prediction`).
  - [ ] Implement benchmark data loaders & local fixtures.
  - [ ] Implement lexical baseline and offline evaluation CLI.
  - [ ] Unit test schemas, parsing, split integrity, and baseline metrics.
- [ ] **Phase 2: Scholarly paper discovery and retrieval**
  - [ ] OpenAlex client with inverted index reconstruction, backoff, and caching.
  - [ ] arXiv client with XML parser and rate limiting.
  - [ ] Polarity query expansion & BM25 + dense hybrid retrieval with RRF.
- [ ] **Phase 3: Finding, evidence and condition extraction**
  - [ ] Sentence segmentation & finding rationale extraction.
  - [ ] PICO/Condition slot tagger ($I, C, O, S$) with confidence and offsets.
  - [ ] Provenance substring verification.
- [ ] **Phase 4: Relationship classification and condition-aware experiment**
  - [ ] Unconditioned text-only baseline classifier.
  - [ ] Condition compatibility evaluator & slot difference encoder.
  - [ ] Condition-aware 5-class classifier with uncertainty/abstention.
- [ ] **Phase 5: Evaluation, reliability and error analysis**
  - [ ] Comprehensive benchmark evaluation on CAS-Con150 and SciFact.
  - [ ] Ablation study (unconditioned vs condition-aware vs individual slots).
  - [ ] Confusion matrix, error taxonomy, and `docs/EVALUATION.md`.
- [ ] **Phase 6: Local demo, documentation and final handoff**
  - [ ] Interactive Streamlit application with evidence provenance view.
  - [ ] Complete `docs/SYLLABUS_MAP.md` and `README.md`.
  - [ ] Final smoke tests and git hygiene check.

---

## Active Blockers
*None.* Development is proceeding offline-first without requiring paid API keys.
