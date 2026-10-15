# Research Audit: Scientific Contradiction Finder

**Date:** 2026-10-11  
**Status:** Completed Baseline Verification

---

## 1. Executive Summary & Research Goal

The goal of the **Scientific Contradiction Finder** is to verify empirical scientific claims against published literature, extract evidence sentences and experimental conditions (Intervention $I$, Comparator $C$, Outcome $O$, Setting $S$), and determine whether cross-paper findings represent:
1. `SUPPORTS_OR_CONSISTENT`
2. `CONTRADICTS`
3. `QUALIFIES_OR_PARTIAL_DISAGREEMENT`
4. `INCOMPARABLE_CONDITIONS`
5. `INSUFFICIENT_EVIDENCE`

### Research Hypothesis
> Explicitly representing experimental-condition differences ($I, C, O, S$) improves the classification of cross-paper finding relationships over a baseline that compares finding text alone.

This is treated as an empirical hypothesis to be tested and benchmarked, not an assumed outcome. Honest baselines, ablations, and fallback to provenance verification are established.

---

## 2. Benchmark Dataset Audit

### 2.1 SciFact (Wadden et al., EMNLP 2020)
- **Official Repository:** `https://github.com/allenai/scifact` / Hugging Face `allenai/scifact`
- **License:** Apache 2.0
- **Size & Format:** 1,409 scientific claims paired with research abstracts, rationale sentence annotations, and stance labels (`SUPPORT`, `CONTRADICT`, `NOT_ENOUGH_INFO`).
- **Suitability:** High for claim verification, rationale sentence extraction, and polarity baseline training.
- **Limitation for our task:** SciFact is single-claim to single-abstract; it does not naturally contain cross-paper finding pairs with condition divergence. We use it for rationale sentence extraction evaluation and claim-stance verification baselines.

### 2.2 Evidence Inference 2.0 (DeYoung et al., BCB 2020)
- **Official Repository:** `https://github.com/jayded/evidence-inference`
- **License:** MIT / CC-BY
- **Task:** Given an RCT paper (PICO context: Intervention, Comparator, Outcome), determine if the intervention significantly increases, decreases, or has no significant effect on the outcome.
- **Suitability & Caution:** Crucially, Evidence Inference 2.0 operates within single clinical trials (intra-paper PICO stance). It does *not* out-of-the-box provide cross-document contradiction labels. However, its extracted $(I, C, O)$ tuples provide gold standards for slot extraction and can be paired across papers investigating identical or diverging interventions/outcomes.

### 2.3 SciNLI (Sadat and Caragea, ACL 2022)
- **Task:** Scientific NLI formulated as Entailment, Contradiction, Neutral between sentence pairs extracted from scientific papers.
- **Suitability:** Good for training or evaluating scientific NLI sentence-level cross-encoders, but lacks condition parameterization.

### 2.4 CAS-Con150 (Condition-Aware Scientific Conflict Evaluation Set)
- To directly test the cross-paper condition hypothesis without data leakage, we construct a curated benchmark of 150 scientific finding pairs across biomedical and NLP domains with ground truth for the 5-class taxonomy:
  - 30 Consistent / Supporting
  - 30 Direct Contradiction (matching conditions, opposite findings)
  - 40 Qualified Disagreement (partially divergent settings/metrics/subpopulations)
  - 30 Incomparable Conditions (different interventions or metrics)
  - 20 Insufficient Evidence (unsubstantiated claims or missing experimental details)

> **Post-Implementation Audit Note (2026-10-12):** As documented in `docs/EVALUATION.md`, CAS-Con150 is a synthetic, template-derived benchmark expanded from 29 base claim templates. For primary empirical evaluation, the project relies on the 35-pair Independent Primary Literature Benchmark (`data/annotations/independent_eval_set.json`), while CAS-Con150 is evaluated under leakage-safe claim-family grouping to prevent memorization. Historical scores (93.3% or 100%) on CAS-Con150 reflect synthetic behavior and must not be cited as headline real-world performance.


---

## 3. Scholarly API Audit

### 3.1 OpenAlex API
- **Endpoint:** `https://api.openalex.org/works`
- **Authentication:** Free tier provides 100,000 requests/day with API key or via mailto parameter for the polite pool.
- **Rate Limit:** 10 requests/sec.
- **Abstract format:** Stored as an inverted index (`abstract_inverted_index`). Requires client-side reconstruction into plain text.
- **Design Decision:** Build robust inverted-index reconstruction, strict caching in local SQLite, and unit-tested mock adapters before any live requests.

### 3.2 arXiv API
- **Endpoint:** `http://export.arxiv.org/api/query`
- **Authentication:** No API key required.
- **Rate Limit:** Max 1 request every 3 seconds; polite batching.
- **Format:** Atom XML feed containing title, summary (abstract), authors, DOI, and arXiv ID.

### 3.3 Semantic Scholar & Crossref
- Optional fallbacks. OpenAlex and arXiv provide full coverage for biomedical and computational literature abstracts without paid tiers.

---

## 4. Model Selection & Licencing

1. **Tokenization & Syntactic Parsing:** `spaCy` (`en_core_web_sm`), MIT License.
2. **Lexical Retrieval:** `rank-bm25` (BM25Okapi), Apache 2.0.
3. **Dense Retrieval:** `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0, 80MB, fast CPU inference) as the primary offline/local bi-encoder.
4. **NLI Backbone:** `cross-encoder/nli-deberta-v3-small` (Apache 2.0, ~140MB) or lightweight feature classifier with scikit-learn for strict local CPU execution.
5. **LLM Inference Policy:** Zero paid APIs (OpenAI, Anthropic, Gemini API are strictly avoided). Purely local, reproducible inference.

---

## 5. Novelty, Assumptions & Risks

- **Novelty Risk:** If cross-encoders already implicitly model condition nuance from raw text, condition extraction might add noise rather than accuracy.
- **Mitigation:** We benchmark an unconditioned baseline against condition-aware classifiers and report precision, recall, macro-F1, and ablation results honestly.
