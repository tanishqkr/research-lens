# Architecture & Technical Decision Records

## ADR-001: Local-First Python Stack with Zero Paid API Dependency
- **Context:** The system must run reproducibly and reliably with $0 spend.
- **Decision:** Use standard open-source Python libraries (`pydantic`, `httpx`, `pandas`, `scikit-learn`, `rank-bm25`, `spacy`, `torch`, `transformers`, `streamlit`). Never invoke metered LLM APIs.
- **Consequence:** Eliminates rate limit surprises, API costs, and network instability.

## ADR-002: Dual-Stage Retrieval: Lexical BM25 + Dense Bi-Encoder with RRF
- **Context:** Scientific vocabulary requires both exact keyword matching (drug names, metrics, gene symbols) and semantic paraphrasing.
- **Decision:** Implement BM25Okapi as the foundational first stage. Supplement with `all-MiniLM-L6-v2` dense embeddings, merging candidate rankings via Reciprocal Rank Fusion (RRF, $k=60$). Provide full offline fallback using pre-indexed abstracts.
- **Consequence:** Resilient to synonym mismatch and typo issues without requiring heavy vector databases.

## ADR-003: Five-Class Taxonomy for Scientific Finding Relations
- **Context:** Binary or ternary NLI (entailment/contradiction/neutral) fails to capture when studies disagree because their populations, baselines, or metrics diverge.
- **Decision:** Implement a 5-class relation schema:
  1. `SUPPORTS_OR_CONSISTENT`
  2. `CONTRADICTS`
  3. `QUALIFIES_OR_PARTIAL_DISAGREEMENT`
  4. `INCOMPARABLE_CONDITIONS`
  5. `INSUFFICIENT_EVIDENCE`
- **Consequence:** Preserves nuanced scientific disagreements and prevents false-alarm contradiction alerts.

## ADR-004: Strict Evidence Provenance with Substring Verification
- **Context:** Generative models or hallucinated extractions erode trust in scientific tools.
- **Decision:** Every extracted finding rationale and evidence passage must be verified as an exact substring of the original source paper abstract, tracking start/end character offsets.
- **Consequence:** 100% verifiable citations in the UI and report outputs.

## ADR-005: OpenAlex API Quota Protection & Offline SQLite Caching
- **Context:** OpenAlex API quota must be preserved.
- **Decision:** All HTTP calls are wrapped with SQLite persistent caching, SHA-256 query hashing, exponential backoff, rate limiting, and an offline test mode. Mocked unit tests verify the adapter before any live request.
- **Consequence:** Zero accidental quota consumption during development and testing.
