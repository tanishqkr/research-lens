# Syllabus Mapping: Scientific Contradiction Finder

This document tracks how techniques in this project map to the Natural Language Processing curriculum, maintaining strict honesty about what is implemented versus what is out of scope.

---

## Curriculum Coverage Matrix

| Curriculum Topic | Specific Implementation in Project | Status | Details / Location |
| :--- | :--- | :--- | :--- |
| **1. NLP Fundamentals & Pipeline Design** | End-to-end multi-stage pipeline: query expansion -> hybrid retrieval -> rationale extraction -> condition slot tagging -> condition-aware relational classification -> provenance verification. | **Fully Implemented** | `src/scientific_contradiction_finder/` |
| **2. Text Preprocessing & Representations** | Tokenization, case folding, stopword removal, inverted-index decoding, TF-IDF vectorization, sparse lexical representations. | **Fully Implemented** | `ingestion/normalizer.py`, `retrieval/bm25_retriever.py`, `extraction/` |
| **3. Lexical Semantics & Antonymy** | Scientific polarity query generation (direct antonym expansion, negative operator inversion, comparison framing). | **Fully Implemented** | `retrieval/query_parser.py` |
| **4. Information Retrieval (Lexical & Dense)** | BM25Okapi scoring combined with dense bi-encoder embeddings (`all-MiniLM-L6-v2`) via Reciprocal Rank Fusion ($k=60$). | **Fully Implemented** | `retrieval/bm25_retriever.py`, `retrieval/dense_retriever.py`, `retrieval/hybrid_reranker.py` |
| **5. Syntactic Parsing & Linguistic Structure** | Sentence boundary detection, rule-based PICO slot extraction, dependency pattern parsing for subject/verb/object relations. | **Fully Implemented** | `extraction/sentence_splitter.py`, `extraction/condition_tagger.py` |
| **6. Information Extraction & Slot Filling** | Sentence-level rationale extraction; 4-tuple condition slot filling ($I, C, O, S$); strict character offset tracking. | **Fully Implemented** | `extraction/finding_extractor.py`, `extraction/condition_tagger.py`, `schemas.py` |
| **7. Natural Language Inference & Stance** | 5-class cross-finding relationship classification; comparative condition gating; abstention for uncertain evidence. | **Fully Implemented** | `reasoning/classifier.py`, `reasoning/compatibility.py` |
| **8. Evaluation Metrics & Error Taxonomy** | Precision, Recall, Macro-F1, per-class F1, confusion matrices, ablation analysis (unconditioned vs condition-aware). | **Fully Implemented** | `evaluation/evaluator.py`, `docs/EVALUATION.md` |
| **9. Word-Level Language Modeling (N-grams)** | *Demonstrative / Standalone only.* Not forced into the contradiction classification pipeline where discriminative representations are superior. | **Documented Out of Core Scope** | Documented rationale in ADRs. |
| **10. Coreference Resolution & Discourse** | Intra-abstract coreference is approximated via sentence context windows; full neural cross-document coreference deferred due to latency/footprint. | **Documented Scope Boundary** | Handled via paragraph-level context in `finding_extractor.py`. |
