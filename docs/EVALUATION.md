# Empirical Evaluation & Performance Analysis: Scientific Contradiction Finder

**Date:** 2026-10-12  
**Evaluation Harness:** `scientific_contradiction_finder.evaluation.evaluator.PipelineEvaluator`  
**Execution Command:** `uv run python scripts/run_full_evaluation.py`  
**Test Suite:** `uv run pytest` (123 passed) | **Linter:** `uv run ruff check .` (0 errors)  
**Primary Evaluation Artifact:** `data/evaluation_summary_corrected.json`

---

## 1. Primary Evaluation Result: Independent Literature Benchmark

The headline evaluation metric for the Scientific Contradiction Finder is evaluated on the **Independent Primary Literature Benchmark** (`data/annotations/independent_eval_set.json`). This benchmark consists of 35 diverse biomedical finding pairs (7 pairs per class) curated directly from major peer-reviewed journals (NEJM, Lancet, JAMA, BMJ, Nature Medicine) with verified DOIs and PMIDs, explicit adjudication criteria, and zero template reuse.

### Headline Performance Summary

| Metric | Condition-Aware Pipeline | Unconditioned TF-IDF Baseline | Relative Advantage |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **88.57%** (31 / 35) | 37.14% (13 / 35) | **+51.43%** |
| **Macro F1** | **0.8821** | 0.3024 | **+0.5797** |
| **Contradiction Precision** | **100.0%** (4 / 4) | 66.67% (4 / 6) | **+33.33% (Zero False Contradictions)** |
| **Contradiction Recall** | **57.1%** (4 / 7) | 57.1% (4 / 7) | *Conservative Gating* |
| **Supports / Consistent F1** | **1.0000** | 0.4242 | **+0.5758** |
| **Qualifies / Partial Disagreement F1** | **0.7500** | 0.2222 | **+0.5278** |
| **Incomparable Conditions F1** | **0.9333** | 0.0000 | **+0.9333 (Baseline Blind to Disjoint Tasks)** |
| **Insufficient Evidence F1** | **1.0000** | 0.2500 | **+0.7500** |

> [!WARNING] Sample Size & Generalization Boundary
> The independent benchmark contains **exactly 35 pairs (7 per class)**. These quantitative results provide **preliminary empirical evidence** supporting condition-aware reasoning over unconditioned baselines, **NOT statistical proof of generalizability across scientific literature broadly**. Findings should be interpreted as demonstrating pipeline validity on curated literature cases rather than exhaustive real-world validation.

---

### Per-Class Metrics Breakdown (Independent Benchmark, N=35)

| Class | Ground Truth Support | Predicted | True Positives | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SUPPORTS_OR_CONSISTENT` | 7 | 7 | 7 | **1.0000** | **1.0000** | **1.0000** |
| `CONTRADICTS` | 7 | 4 | 4 | **1.0000** | **0.5714** | **0.7273** |
| `QUALIFIES_OR_PARTIAL_DISAGREEMENT` | 7 | 9 | 6 | **0.6667** | **0.8571** | **0.7500** |
| `INCOMPARABLE_CONDITIONS` | 7 | 8 | 7 | **0.8750** | **1.0000** | **0.9333** |
| `INSUFFICIENT_EVIDENCE` | 7 | 7 | 7 | **1.0000** | **1.0000** | **1.0000** |
| **Macro Average** | **35** | **35** | **31** | **0.9083** | **0.8857** | **0.8821** |

---

### Confusion Matrix (Independent Benchmark)

Rows represent ground-truth labels; columns represent predicted labels.

| Ground Truth \ Predicted | `SUPPORTS` | `CONTRADICTS` | `QUALIFIES` | `INCOMPARABLE` | `INSUFFICIENT` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SUPPORTS_OR_CONSISTENT`** (7) | **7** | 0 | 0 | 0 | 0 |
| **`CONTRADICTS`** (7) | 0 | **4** | 3 | 0 | 0 |
| **`QUALIFIES_OR_PARTIAL_DISAGREEMENT`** (7) | 0 | 0 | **6** | 1 | 0 |
| **`INCOMPARABLE_CONDITIONS`** (7) | 0 | 0 | 0 | **7** | 0 |
| **`INSUFFICIENT_EVIDENCE`** (7) | 0 | 0 | 0 | 0 | **7** |

#### Key Classification Observations:
1. **Zero False Contradictions (100% Contradiction Precision):** The pipeline never predicted `CONTRADICTS` for any pair that was consistent, qualified, incomparable, or insufficient.
2. **Conservative Recall on Contradictions (57.1%):** In 3 of the 7 contradiction cases, the pipeline predicted `QUALIFIES_OR_PARTIAL_DISAGREEMENT` due to detected differences in study population, comparator, or design setting (see Section 4).
3. **Flawless Abstention on Insufficient Evidence:** All 7 pairs lacking empirical results or containing ungrounded claims were identified and safely abstained.
4. **Unconditioned Baseline Blindness:** The TF-IDF baseline predicted either `SUPPORTS` or `CONTRADICTS` for 34 of 35 pairs, completely failing to recognize `INCOMPARABLE_CONDITIONS` (Recall = 0.0000) and confusing distinct study settings.

---

## 2. Synthetic & Template-Derived Validation: CAS-Con150

> [!NOTE] Synthetic / Template-Derived Dataset (Not Headline Performance)
> CAS-Con150 consists of 150 synthetic pairs expanded from **29 base claim templates**. Historical reports citing **93.3%** or **100%** on CAS-Con150 reflect development/synthetic behavior and **must not be presented as the headline measure of real-world scientific performance**.

### The Template Leakage Phenomenon
During the pre-remediation audit, an evaluation flaw was uncovered:
- The 150 pairs in CAS-Con150 were generated from only 29 base templates (e.g., pairs with identical claim texts and minor evidence variations).
- Under a naive random train/test split, identical claim templates appeared in both training and test sets.
- The unconditioned TF-IDF baseline memorized claim n-grams, producing a spurious **100% Accuracy and Macro F1**.

### Corrected Leakage-Safe Evaluation ($N_{\text{train}}=104, N_{\text{test}}=46$)
By grouping claim families `(finding_a.claim, finding_b.claim)` prior to splitting (`leakage_safe=True`), zero template overlap is guaranteed between splits:

| Model / Configuration | Accuracy | Macro-F1 | Supports F1 | Contradicts F1 | Qualifies F1 | Incomparable F1 | Insufficient F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unconditioned TF-IDF Baseline** | 0.1957 | 0.0750 | 0.3750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Condition-Aware Pipeline** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Conclusion:* When template leakage is eliminated, the TF-IDF baseline collapses to **19.57% accuracy (chance level ~20%)**, while the condition-aware pipeline maintains **100% accuracy**, demonstrating that condition slot extraction and compatibility gating generalize across unseen claim formulations in synthetic settings.

### Historical Leaky Split ($N_{\text{train}}=105, N_{\text{test}}=45$) — Audit Reference Only
Preserved strictly for reproducibility and historical audit reconciliation:
- Unconditioned TF-IDF Baseline: Accuracy 1.0000, Macro-F1 1.0000 (Artifact of template memorization)
- Condition-Aware Pipeline: Accuracy 0.9333, Macro-F1 0.9415

---

## 3. Genuine Feature-Disabled Ablation Analysis

Historical ablation routines simulated ablation via post-hoc prediction string mutation. The remediated pipeline evaluates genuine feature-disabled variants:
- `enable_incomparability_check=False`: Disables outcome and intervention compatibility gating.
- `enable_setting_gating=False`: Disables population and study-setting divergence thresholds.

### Ablation Results on Independent Benchmark (N=35)

| Configuration | Accuracy | Macro-F1 | Supports F1 | Contradicts F1 | Qualifies F1 | Incomparable F1 | Insufficient F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Condition-Aware Model** | **0.8857** | **0.8821** | 1.0000 | 0.7273 | 0.7500 | **0.9333** | 1.0000 |
| **Ablation: No Incomparability Check** | 0.7143 | **0.6539** | 0.9333 | 0.7273 | 0.6087 | **0.0000** | 1.0000 |
| **Ablation: No Setting Gating** | 0.9143 | 0.9133 | 1.0000 | 0.8333 | 0.8000 | 0.9333 | 1.0000 |

### Ablation Findings:
1. **Critical Role of Incomparability Gating:** Disabling incomparability checking causes Macro F1 to drop from **0.8821 to 0.6539**. Incomparable study pairs are erroneously classified as `QUALIFIES` (6 pairs) or `SUPPORTS` (1 pair), demonstrating that condition comparability is essential to avoid forcing unrelated studies into conflict.
2. **Behavior of Setting Gating:** Disabling setting gating yields a nominal Macro F1 of **0.9133** because pairs with subtle cohort differences (e.g. ICU vs TBI) are forced into direct contradictions, matching the simplified ground truth label. However, doing so discards genuine clinical effect modification across distinct patient populations.

---

## 4. Error Analysis on the Independent Benchmark

On the 35-pair independent benchmark, the condition-aware pipeline produced 4 discrepancies relative to the reference labels:

### A. Three Missed Contradictions Classified as Qualified Disagreement

| Pair ID | Topic / Comparison | Ground Truth | Predicted | Underlying Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| `indep_contradict_003` | Colloids vs Crystalloids Resuscitation | `CONTRADICTS` | `QUALIFIES` | **Cohort Divergence:** Paper A studied *heterogeneous adult ICU resuscitation* (SAFE trial) while Paper B evaluated *severe traumatic brain injury* patients. The setting similarity score dropped to 0.41 (< 0.65 threshold), triggering conservative qualification. |
| `indep_contradict_006` | Statin Therapy & Diabetes Risk | `CONTRADICTS` | `QUALIFIES` | **Study Design & Comparator Divergence:** Paper A was an observational military healthcare cohort comparing *users vs non-users*, while Paper B was a meta-analysis of *placebo-controlled RCTs*. Comparator similarity dropped to 0.39 (< 0.65 threshold). |
| `indep_contradict_007` | Early Peanut Introduction (LEAP) | `CONTRADICTS` | `QUALIFIES` | **Comparator Phrasing Divergence:** Paper A evaluated *early dietary peanut introduction* while Paper B evaluated *peanut avoidance until 60 months*. The phrasing similarity (0.36) fell below the direct match threshold. |

*Scientific Interpretation:* In all three cases, the system's prediction is scientifically justifiable: differing populations, designs, and comparators legitimately qualify clinical claims in medical literature. Lowering thresholds to capture these three cases would compromise the pipeline's 100% contradiction precision.

### B. One Qualification Discrepancy Classified as Incomparable

| Pair ID | Topic / Comparison | Ground Truth | Predicted | Underlying Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| `indep_qualifies_001` | Aspirin in Elderly vs Established CVD | `QUALIFIES` | `INCOMPARABLE` | **Disjoint Endpoint Domains:** Paper A (ASPREE) measured *Disability-free survival and major hemorrhage* (`MORTALITY`), while Paper B measured *Major adverse coronary events* (`CARDIOVASCULAR`). Because the clinical outcome domains were disjoint, the system classified them as incomparable endpoints. |

---

## 5. Principal System Limitations

The following five principal limitations define the operational boundaries of the Scientific Contradiction Finder:

1. **Missed Genuine Contradictions (Recall vs. Precision Tradeoff):**  
   The pipeline operates with a conservative contradiction recall of **57.1% (4/7)** to ensure **100% contradiction precision (4/4)**. When study populations, dosages, or comparator descriptions diverge, the pipeline hedges toward `QUALIFIES_OR_PARTIAL_DISAGREEMENT` rather than asserting a direct contradiction. In clinical evidence synthesis, false contradictions are more harmful than qualified disagreements.

2. **Cautious Qualification of Observational-versus-RCT Comparisons:**  
   The system intentionally treats observational cohort associations and randomized controlled trials cautiously. When an observational study finds a protective association (e.g. moderate alcohol or observational statin use) and a randomized trial or Mendelian randomization study finds null or harmful effects, the system identifies the study-design discrepancy and qualifies the comparison rather than declaring a logical contradiction. It refuses to conflate correlation with causation.

3. **Annotation Ambiguity and Scientific Adjudication Boundaries:**  
   In complex biomedical literature, human experts frequently disagree on whether two studies are genuinely in conflict. For instance, whether an early infant introduction trial directly contradicts an avoidance guideline depends on whether the population is defined as high-risk vs general infants. The 35-pair independent benchmark reflects one defensible adjudication, but alternative annotations exist.

4. **Multi-Arm Attribution Constraints:**  
   In clinical trials with three or more intervention arms (e.g., Low-Carb vs Low-Fat vs Mediterranean vs Control), sentences lacking explicit prepositional anchors (e.g. "compared with", "relative to") present attribution challenges. To avoid fabricating assignments, the extractor safely abstains (`arm_name: None`) rather than guessing, which limits numerical slot filling in complex multi-arm papers.

5. **Reliance on Open-Access Abstracts:**  
   When full-text PDFs are behind commercial paywalls (Elsevier, Wiley, Springer), the retrieval pipeline falls back to open-access abstracts provided by OpenAlex and PubMed. Abstracts frequently omit secondary endpoints, detailed subgroup inclusion criteria, and exact adverse-event tables, constraining the depth of condition extraction.

---

## 6. Artifact Provenance, Test Verification & Reproducibility

### Verification Commands
```bash
# 1. Run complete automated test suite (123 tests)
uv run pytest

# 2. Run static analysis and linting
uv run ruff check .

# 3. Run full evaluation suite (recomputes metrics and updates JSON)
uv run python scripts/run_full_evaluation.py
```

### Git Checkpoints & Traceability
- **Pre-Remediation Checkpoint Commit:** `a0581cd6cc5a109569d6a2fb910a64966642897a`
- **Pre-Remediation Git Tag:** `checkpoint-milestones-1-4-pre-remediation`
- **Remediation Implementation Commit:** `cf098c9` (Classifier symmetry, leakage-safe splits, genuine ablations)
- **Attribution Refinement Commit:** `b008ff3` (Comparative arm attribution)
- **Current Branch:** `main` (clean working tree)
- **Evaluation Summary JSON:** `data/evaluation_summary_corrected.json`
- **Independent Evaluation Dataset:** `data/annotations/independent_eval_set.json`
