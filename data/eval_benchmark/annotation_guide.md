# Scientific Contradiction Finder: 30-Query Benchmark Annotation Guide & Rubric

This document defines the formal evaluation methodology, scoring rubric, and adjudication procedure for evaluating the **Scientific Contradiction Finder** across 30 benchmark scientific questions.

---

## 1. Evaluation Dimensions

Each evaluated query report is scored across five core dimensions:

### Dimension 1: Retrieval Quality & Semantic Relevance
- **Candidate Retrieval Rate**: Did the system successfully retrieve relevant scholarly papers from OpenAlex and arXiv?
- **Relevance Precision**: What proportion of retrieved and triaged papers actually study the query's target population, intervention/exposure, comparator, and outcome (PICO)?
- **Irrelevance Rejection**: For spurious or implausible queries (e.g. cold water after chocolate, magnetic wristbands for ozone), does the triage stage correctly reject off-target literature rather than forcing false relevance?

**Scoring**:
- `1.0 (Pass)`: Retrieved literature directly matches query domain and PICO concepts, OR correctly rejects off-target papers when query is implausible.
- `0.5 (Partial)`: Retrieved literature is peripherally related (e.g. correct disease but wrong intervention, or in vitro proxy without clinical relevance clearly flagged).
- `0.0 (Fail)`: Retrieved literature is completely unrelated, or false semantic relevance was assigned to completely spurious papers.

---

### Dimension 2: Claim-Level Evidence Grounding & Entailment
- **Source Passage Entailment**: Does the exact source passage in the evidence ledger strictly entail the substantive factual claim made?
- **Direction Accuracy**: Is the finding direction (`POSITIVE`, `NEGATIVE`, `NO_DIFFERENCE`, `UNCLEAR`) faithful to the statistical results and conclusions reported in the cited text?
- **Overclaiming Guard**: Does the ledger refrain from turning observational odds ratios into causal claims or rodent lifespan extensions into human longevity facts?

**Scoring**:
- `1.0 (Pass)`: All extracted evidence claims in the ledger are strictly entailed by their quoted passages with correct direction.
- `0.5 (Partial)`: Claims are generally grounded but minor extrapolations exist (e.g. slight overstatement of certainty or missing condition qualifier).
- `0.0 (Fail)`: Hallucinated claims, fabricated statistics, inverted effect directions, or ungrounded causal assertions.

---

### Dimension 3: Scientific Relationship Classification
- **Classification Agreement**: How does the synthesized relationship (`CONTRADICTION`, `COMPATIBLE` / context-dependent, `CONSISTENT`, `UNRESOLVED` / `INSUFFICIENT_EVIDENCE`, `NO_EVALUABLE_EVIDENCE`) compare to both:
  1. The **Scientific Ground Truth** (established state of broader scientific literature).
  2. The **Retrieval-Constrained Ground Truth** (what the specifically retrieved open-access corpus actually supports).
- **Nuance Handling**: Does the system distinguish genuine empirical contradictions (e.g. RCT vs observational) from context-dependent moderators (e.g. baseline severity, dosage, climate setting)?

**Scoring**:
- `1.0 (Pass)`: System output correctly matches the expected relationship or accurately reflects retrieval constraints (e.g. marking unresolved when retrieved papers lack direct comparators).
- `0.5 (Partial)`: Relationship is partially correct (e.g. flagged as compatible but missed key contradictory nuance, or vice versa).
- `0.0 (Fail)`: Misclassifying established consensus as contradiction, claiming strong support for implausible/unsupported claims, or hallucinating false consensus.

---

### Dimension 4: Abstention & False Confidence Control
- **Appropriate Abstention**: When evidence is missing, preliminary, indirect, or when the query is pseudoscientific, does the system appropriately set subclaims to `UNRESOLVED` and overall assessment to `inconclusive` or abstention?
- **Zero Hallucination of Equivalence**: In comparative queries (e.g. intermittent fasting vs calorie restriction), does the system avoid claiming "equivalence" unless direct head-to-head evidence is actually present?

**Scoring**:
- `1.0 (Pass)`: System correctly abstains or flags lack of evidence without asserting false positive claims.
- `0.0 (Fail)`: False confidence: asserting supported status on unverified claims or declaring proof when literature is absent.

---

### Dimension 5: Executive Summary Consistency
- **Ledger-Summary Alignment**: Is the executive summary ("Bottom Line", "Strongest Supporting Evidence", "Why Studies Disagree") strictly consistent with the subclaim assessments and evidence ledger?
- **Enforcement Validation**: No claim should be asserted as definitive in the executive summary if its corresponding subclaim is marked `UNRESOLVED` or if the ledger has zero qualifying entries.

**Scoring**:
- `1.0 (Pass)`: Perfect internal consistency across summary, subclaims, and evidence ledger.
- `0.0 (Fail)`: Contradiction between executive summary statements and subclaim assessment statuses.

---

## 2. Benchmark Query Categories (30 Total)

| Category | Count | Benchmark IDs | Core Evaluation Objective |
| :--- | :--- | :--- | :--- |
| **Genuine Contradictions** | 6 | BM-01 to BM-06 | Detect opposing empirical findings across study designs or trial cohorts. |
| **Context-Dependent** | 6 | BM-07 to BM-12 | Identify moderating conditions (climate, baseline risk, fatigue state, species). |
| **Consistent Findings** | 5 | BM-13 to BM-17 | Confirm established scientific consensus without inventing artificial conflict. |
| **Unresolved / Insufficient** | 5 | BM-18 to BM-22 | Calibrate uncertainty when clinical trials are lacking, null, or indirect. |
| **Irrelevant / Implausible** | 4 | BM-23 to BM-26 | Test abstention, triage rejection, and refusal to fabricate evidence. |
| **Conditional / Comparative** | 4 | BM-27 to BM-30 | Evaluate nuanced head-to-head comparisons across specific clinical endpoints. |

---

## 3. Adjudication Workflow

1. **Input**: Run query through `ContradictionFinderPipeline.analyze_as_research_analyst(query, limit=10, max_deep_dives=5)`.
2. **Raw Recording**: Serialize full `ResearchReport` to `data/eval_benchmark/runs/<run_id>/<query_id>.json`.
3. **Metric Calculation**:
   - Compare `overall_evidence_assessment` and subclaim statuses against expected relationship.
   - Audit evidence ledger passages for direct support.
   - Check triage classification rejection rate on irrelevant queries.
4. **Summary & Error Analysis**: Aggregate scores per category and document any systemic failure modes.
