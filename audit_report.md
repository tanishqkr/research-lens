# Adversarial Audit Report: 30-Query Benchmark Evaluation & Classification Diagnosis

**Date of Audit**: 2026-10-11  
**Auditor**: Independent Evaluation Agent  
**Audited Artifacts**:
- `data/eval_benchmark/benchmark_v1.jsonl` (30 ground-truth reference records)
- `data/eval_benchmark/runs/benchmark_run_live/` (30 raw output JSON files)
- `scripts/run_benchmark.py` (Benchmark runner implementation)
- `query_error_analysis.csv` (Complete query-by-query failure taxonomy)
- `metrics_recomputed.json` (Recomputed deterministic metrics)

---

## 1. Executive Verdict: Truth vs. Stated Performance

> **Candid Verdict**:  
> The benchmark **does NOT support the claim that this system can autonomously detect scientific contradictions** in general scientific literature (where strict accuracy is only **16.7%**, 1 out of 6).  
> Instead, the benchmark **strongly supports a narrower, highly valuable claim**: the system functions effectively as a **conservative, highly grounded evidence retrieval and abstention assistant** that reliably extracts verifiable quotes (**96.6%** entailment), safely rejects pseudoscience and implausible hypotheses (**100%** abstention), and enforces summary consistency (**96.7%**), but frequently fails to classify scientific relationships due to **open-access paywall barriers, upstream triage fallbacks, and single-sided retrieval**.

### Benchmark Metric Recomputation vs. Runner Claims

| Metric | Runner Reported | Recomputed Metric | Exact Formula (Num / Denom) | Root Cause of Discrepancy |
| :--- | :---: | :---: | :---: | :--- |
| **Retrieval Pass Rate** | 80.0% | **80.0%** | 24 / 30 | Definition: retrieved >0 papers & deep-dived >0 (or 0 deep-dives for implausible). Confirmed. |
| **Claim Grounding Precision** | 100.0% | **96.6%** | 28 / 29 | Runner checked only `len(passage) > 20`. In BM-02, the LLM extracted a passage about *new-onset diabetes* for a query on *cataracts*. |
| **Relationship Classification** | 69.2% | **43.3%** (Strict)<br>**40.0%** (Retrieval-cond.) | 13 / 30 (Strict)<br>6 / 15 (Retrieval-cond.) | Runner used **partial credit heuristic** (summing 20.75 / 30) giving 0.5 for `inconclusive` on missing papers and 0.75 for partial matches. |
| **Abstention Accuracy** | 100.0% | **100.0%** | 30 / 30 | Zero false-positive assertions on implausible folklore (4/4) and insufficient evidence (5/5). |
| **Summary Consistency** | 96.7% | **96.7%** | 29 / 30 | 1 concrete defect in BM-19: summary claimed `broadly supports` while subclaim was marked `UNRESOLVED`. |
| **Mean Composite Score** | 87.8% | **N/A** (Misleading composite) | Weighted sum | Masked 43.3% classification accuracy by blending with 100% abstention and 100% grounding. |

---

## 2. Audit of Metric Definitions, Calculations, and Runner Bias

### 2.1 The "69.2%" Relationship Score Deconstructed
The runner reported `0.692` for `relationship_classification_score`.
- **Exact Numerator**: `20.75`
- **Exact Denominator**: `30`
- **How 20.75 was obtained**:
  - `1.0` was awarded to:
    - 4/4 Implausible queries (BM-23, BM-24, BM-25, BM-26) because ledger was empty and assessment was `inconclusive`.
    - 5/5 Insufficient evidence queries (BM-18, BM-19, BM-20, BM-21, BM-22) because assessment was `inconclusive` or subclaims were `UNRESOLVED`.
    - 2/5 Consistent finding queries (BM-15, BM-16).
    - 1/6 Genuine contradictions (BM-01).
    - 1/6 Context-dependent queries (BM-07).
    - *Subtotal*: `13 * 1.0 = 13.0`
  - `0.75` partial credit was awarded to:
    - 2/5 Consistent finding queries (BM-13, BM-14) because subclaims had `SUPPORTED` but summary had context qualifiers.
    - *Subtotal*: `2 * 0.75 = 1.5`
  - `0.50` partial credit was awarded to:
    - 3/6 Genuine contradictions (BM-04, BM-05, BM-06) because the system defaulted to `inconclusive` or `all_unresolved`!
    - 3/6 Context-dependent queries (BM-09, BM-11, BM-12) because the system defaulted to `inconclusive`!
    - 1/5 Consistent finding queries (BM-17) because of `inconclusive`!
    - 3/4 Conditional comparative queries (BM-28, BM-29, BM-30) because of `inconclusive`!
    - *Subtotal*: `10 * 0.50 = 5.0`
  - `0.25` partial credit was awarded to:
    - 2/6 Genuine contradictions (BM-02, BM-03) where only one side was retrieved and classified as single-sided!
    - 2/6 Context-dependent queries (BM-08, BM-10).
    - 1/4 Conditional comparative queries (BM-27).
    - *Subtotal*: `5 * 0.25 = 1.25`
  - **Sum**: `13.0 + 1.5 + 5.0 + 1.25 = 20.75`. `20.75 / 30 = 0.69167 (~69.2%)`.

> **Defect in Metric**: The runner rewarded the application **0.50 points whenever retrieval failed and the system said "inconclusive"**, even when the reference ground truth was a genuine contradiction or an established consensus! This artificially inflated the score from **43.3%** to **69.2%**.

### 2.2 The Composite Score Formula Bias
The runner defined `composite_score` as:
$$\text{Composite} = 0.20 \times \text{Retrieval} + 0.25 \times \text{Grounding} + 0.25 \times \text{Relationship} + 0.15 \times \text{Abstention} + 0.15 \times \text{Consistency}$$
- If a query completely fails retrieval, triages 0 papers, extracts 0 ledger claims, and outputs `inconclusive`:
  - `Retrieval`: 0.0 (or 1.0 if implausible)
  - `Grounding`: 1.0 (defaulted to 1.0 when ledger claims = 0!)
  - `Relationship`: 0.5 (awarded partial credit for `inconclusive`)
  - `Abstention`: 1.0
  - `Consistency`: 1.0
  - **Resulting score for a complete failure**: $0.0(0.20) + 1.0(0.25) + 0.5(0.25) + 1.0(0.15) + 1.0(0.15) = \mathbf{0.675}$ (**67.5% score for doing nothing**).
- This explains why BM-05, BM-17, BM-29, and BM-30 (which failed retrieval completely) all received scores of `0.68`.

### 2.3 The "100% Claim Grounding" Audit
The runner checked only:
```python
passage = c.get("exact_source_passage", "").strip()
if passage and len(passage) > 20:
    grounded_claims += 1
```
Out of 29 ledger claims extracted across the 30 queries:
- **28 claims** are strictly verbatim quotes from the papers and describe valid endpoints.
- **1 claim is a concrete defect**: In **BM-02** (`query`: *"Does statin therapy increase the risk of cataract development?"*), the paper `W2801534920` was an umbrella review discussing multiple statin adverse effects. The extractor extracted:
  > *"Randomized controlled trials (RCTs) and genetic studies show that statin therapy is associated with a modest increase in the risk of new-onset diabetes mellitus..."*
  The extractor assigned this to the ledger under outcome `"New-onset diabetes mellitus"`, ignoring that the query was asking about **cataracts**!
- True Claim Grounding Precision is therefore **28 / 29 = 96.6%**, not 100.0%.

---

## 3. Verification of the 30 Benchmark Reference Records

All 30 benchmark records in `data/eval_benchmark/benchmark_v1.jsonl` were audited against PubMed and primary literature databases:
1. **Category Distribution**: Exactly verified:
   - Genuine Contradictions: 6 (BM-01 to BM-06)
   - Context-Dependent: 6 (BM-07 to BM-12)
   - Consistent Findings: 5 (BM-13 to BM-17)
   - Unresolved / Insufficient: 5 (BM-18 to BM-22)
   - Irrelevant / Implausible: 4 (BM-23 to BM-26)
   - Conditional / Comparative: 4 (BM-27 to BM-30)
2. **DOI / PMID Verifications**:
   - BM-01 (RECOVERY trial): `10.1056/NEJMoa2022926` / PMID 33027584 — **Verified**.
   - BM-02 (Statins & Cataracts meta-analysis): `10.7326/M14-0417` / PMID 25364887 — **Verified**.
   - BM-03 (Mammography Cochrane review): `10.1002/14651858.CD001877.pub5` / PMID 23737396 — **Verified**.
   - BM-04 (NICE-SUGAR trial): `10.1056/NEJMoa0810625` / PMID 19318384 — **Verified**.
   - BM-05 (LEAP trial): `10.1056/NEJMoa1414850` / PMID 25705822 — **Verified**.
   - BM-06 (PURE sodium study): `10.1056/NEJMoa1311889` / PMID 25119607 — **Verified**.
   - BM-07 to BM-22, BM-24 to BM-25, BM-27 to BM-30 — **All DOIs / PMIDs refer to real, high-impact published papers**.
3. **Audit of "Genuine Contradiction" Validity**:
   - *BM-01 (HCQ)*: Genuine contradiction between early observational studies/small non-randomized trials (positive) and large multicenter RCTs (null/harm). Both evaluate hospitalized COVID-19 patients with mortality endpoints.
   - *BM-02 (Statin cataracts)*: Genuine contradiction between large observational databases (OR ~1.27) and randomized trials (RR ~0.99).
   - *BM-03 (Mammography 40-49)*: Genuine contradiction between Cochrane meta-analysis (no significant mortality benefit, high overdiagnosis) and Canadian/Swedish trials claiming 15-20% relative reduction.
   - *BM-04 (Tight glycemic control)*: Genuine contradiction between Leuven trial (mortality benefit) and NICE-SUGAR trial (increased mortality).
   - *BM-05 (Early peanut introduction)*: Genuine contradiction between historic consensus guidelines (delayed introduction prevents allergy) and LEAP RCT (early introduction prevents allergy).
   - *BM-06 (Sodium reduction in normotensives)*: Genuine contradiction between AHA/WHO linear reduction models and PURE cohort observational J-curve.

---

## 4. Query-by-Query Error Diagnosis

A full 30-row tabular analysis has been exported to [`query_error_analysis.csv`](file:///Users/tanishporwal/Desktop/scientific_contradiction/query_error_analysis.csv).

### Summary of Failure Modes across the 30 Queries

```
Total Queries: 30
├── Fully Successful (13 / 30, 43.3%)
│   ├── All 4 Irrelevant/Implausible (BM-23, BM-24, BM-25, BM-26) -> Correct Abstention
│   ├── 4 Unresolved/Insufficient (BM-18, BM-20, BM-21, BM-22) -> Correct Abstention
│   ├── 2 Consistent Findings (BM-15, BM-16) -> Correctly classified broadly supports
│   ├── 2 Context/Contradiction (BM-01, BM-07) -> Correctly captured divergence / context
│   └── BM-19 -> Correct subclaim abstention (though minor summary heuristic defect)
│
└── Failures & Deficiencies (17 / 30, 56.7%)
    ├── Stage 1: Triage Rate-Limit Fallback (6 queries: BM-05, BM-17, BM-20, BM-22, BM-29, BM-30)
    │   └── 60.7% of all triage calls hit Groq 429 rate limits, causing fail-closed rejection
    │       of ALL retrieved candidate papers, leading to 0 deep dives.
    ├── Stage 2: Open-Access Paywall / Abstract-Only Deficit (5 queries: BM-09, BM-11, BM-12, BM-21, BM-28)
    │   └── Papers were selected for deep dive, but PDFs were paywalled and abstracts lacked
    │       quantitative empirical sentences, resulting in 0 ledger claims.
    ├── Stage 3: Single-Sided Retrieval & Relationship Misclassification (5 queries: BM-02, BM-03, BM-04, BM-06, BM-08)
    │   └── OpenAlex returned papers supporting only ONE side of a known contradiction,
    │       causing the system to classify it as single-sided rather than contradictory.
    └── Stage 4: Extractor Outcome Mismatch (1 query: BM-02)
        └── Extracted diabetes passage for a cataract query.
```

---

## 5. Separation of Retrieval Failure from Reasoning Failure

Evaluating reasoning when retrieval fails creates a misleading picture:
- **End-to-End Strict Relationship Accuracy**: **13 / 30 (43.3%)**
- **Retrieval-Conditioned Relationship Accuracy** (evaluated only on the 15 queries where $\ge 1$ finding was admitted to the ledger): **6 / 15 (40.0%)**

Why is retrieval-conditioned accuracy only 40.0%?
1. **Single-Sided Corpus Retrieval**: When given a query with known contradictory literature (e.g. *BM-06: Sodium reduction and normotensive mortality*), OpenAlex retrieved only studies showing higher sodium increases mortality (`W2169859718`), and did not retrieve the PURE study showing low sodium increases mortality. Because only one side was in the ledger, the reasoning engine had no opposing data to compare, defaulting to single-sided reduction or `inconclusive`.
2. **Lack of Head-to-Head Comparative Trials**: For *BM-27 (DOACs vs Warfarin)*, the pipeline retrieved a stroke prevention network meta-analysis (`W2769275060`) and a liver cirrhosis cohort (`W4389448233`), both showing lower or equivalent stroke risk, but failed to retrieve trials specifically reporting the GI bleeding excess that creates the conditional trade-off.

---

## 6. Audit of the Summary Consistency Defect in BM-19

In query **BM-19** (*"Does dietary ingestion of microplastics cause clinical cardiovascular events in humans?"*):
- The ledger contained 1 claim from the landmark NEJM paper (`W4392520497`, Marfella et al.):
  > *"Patients in whom MNPs were detected within the atheroma were at higher risk for a primary end-point event than those in whom these substances were not detected (hazard ratio, 4.53; 95% confidence interval, 2.00 to 10.27; P<0.001)."*
- The subclaim assessment correctly marked the subclaim **`UNRESOLVED`** because observational atheroma detection does not prove dietary ingestion causation.
- However, the pipeline's heuristic summary generator fallback triggered:
  ```python
  elif neg_count > 0 and pos_count == 0:
      overall = "broadly supports an inverse or reducing relationship"
      bl = "The available empirical studies demonstrate consistent reductions or negative impacts."
  ```
- Because the finding was tagged with direction `NEGATIVE` (elevated hazard was coded as adverse), the heuristic set `overall_evidence_assessment = "broadly supports an inverse or reducing relationship"`!
- The post-processing rule in `enforce_summary_consistency()` checked `if exec_summary.overall_evidence_assessment in ("broadly supports", "broadly challenges")`, but did not catch this exact phrase string variant, allowing the summary to contradict the subclaim.

---

## 7. Prioritized List of the Three Most Consequential Improvements

### 1. Robust Triage Batching and Rate-Limit Resiliency
- **Problem**: 167 of 275 triage requests (60.7%) triggered the HTTP 429 fail-closed fallback, causing completely relevant papers (e.g., in BM-05, BM-17, BM-30) to be discarded as `IRRELEVANT` before deep dive.
- **Solution**:
  - Batch triage calls: Instead of sending 10 individual HTTP requests per query to Groq, send a single JSON array containing all 10 candidate titles/abstracts in one prompt.
  - Implement a token bucket rate limiter with exponential backoff on 429 rather than falling closed to `IRRELEVANT`.

### 2. Dual-Polarity Retrieval for Contradiction Discovery
- **Problem**: In 4 of the 6 genuine contradiction benchmarks (BM-02, BM-03, BM-04, BM-06), OpenAlex returned papers representing only one side of the debate. The reasoning engine cannot detect a contradiction if the search engine returns a monologue.
- **Solution**:
  - Implement explicit adversarial dual-querying: For any comparative or causal question, run two complementary OpenAlex searches:
    - Search A (Positive/Benefit): `[exposure] [outcome] "increased risk" OR "effective" OR "superior"`
    - Search B (Null/Equivalence/Harm): `[exposure] [outcome] "no difference" OR "equivalent" OR "ineffective" OR "trial"`
  - Fuse both candidate sets via Reciprocal Rank Fusion (RRF) before triage.

### 3. Topic-Constrained Outcome Verification in Evidence Extraction
- **Problem**: In BM-02, the LLM extracted a diabetes finding from an umbrella review abstract instead of the cataract finding. In BM-16, 3 of 4 ledger entries captured general trial heterogeneity or baseline blood pressure descriptions rather than the treatment effect.
- **Solution**:
  - Ground extraction prompt on the target PICO slots: Explicitly instruct `interpret_passage` that the `outcome` must match the specific queried endpoint (e.g. `cataracts`, not other adverse effects in an umbrella review). If the candidate sentence discusses an unqueried endpoint, it must be rejected as `is_empirical_finding=false`.
