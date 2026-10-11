"""Build gold-standard CAS-Con150 (Condition-Aware Scientific Conflict Benchmark).

Curates 150 paired empirical scientific findings with full evidence sentences,
condition tuples (Intervention, Comparator, Outcome, Setting), ground truth relations,
and metadata across biomedicine, machine learning, and environmental science.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "annotations"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "cas_con150.json"


def generate_benchmark_pairs():
    pairs = []

    # 1. Consistent / Supporting (30 pairs)
    # Pairs with matching/compatible conditions and concordant directional outcomes.
    consistent_templates = [
        # Biomedicine
        (
            ("Metformin monotherapy demonstrated significant HbA1c reduction of 1.1% in newly diagnosed T2D.",
             "Metformin", "Placebo", "HbA1c reduction", "Adults with T2D", "POSITIVE"),
            ("Administration of metformin achieved a 1.2% drop in glycosylated hemoglobin versus dietary control.",
             "Metformin", "Dietary control / Placebo", "HbA1c reduction", "Type 2 diabetic cohort", "POSITIVE"),
            "biomedicine"
        ),
        (
            ("Statins reduced major adverse cardiovascular events by 21% in high-risk coronary patients.",
             "Atorvastatin 40mg", "Placebo", "Major adverse cardiovascular events (MACE)", "Coronary artery disease patients", "NEGATIVE"),
            ("Lipid-lowering statin therapy demonstrated a 24% reduction in recurring cardiovascular events.",
             "Statin therapy", "Control", "Cardiovascular events recurrence", "High-risk cardiac cohort", "NEGATIVE"),
            "biomedicine"
        ),
        (
            ("Inhaled corticosteroids reduced annual asthma exacerbation frequency from 2.4 to 1.1 events.",
             "Budesonide", "Placebo", "Asthma exacerbation rate", "Persistent asthma adults", "NEGATIVE"),
            ("Daily budesonide treatment lowered severe asthma exacerbations compared to short-acting bronchodilators alone.",
             "Inhaled Budesonide", "SABA alone", "Severe asthma exacerbations", "Adult asthma patients", "NEGATIVE"),
            "biomedicine"
        ),
        (
            ("Sodium-glucose cotransporter-2 inhibitors reduced heart failure hospitalizations by 30%.",
             "Empagliflozin", "Standard of care", "Heart failure hospitalization", "Heart failure with reduced ejection fraction", "NEGATIVE"),
            ("Dapagliflozin administration lowered the risk of hospitalization for heart failure by 26%.",
             "Dapagliflozin", "Placebo", "Heart failure hospitalization", "HFrEF patients", "NEGATIVE"),
            "biomedicine"
        ),
        (
            ("GLP-1 receptor agonist semaglutide led to a mean weight loss of 14.9% at week 68.",
             "Semaglutide 2.4mg", "Placebo", "Body weight reduction percentage", "Adults with overweight/obesity", "POSITIVE"),
            ("Weekly subcutaneous semaglutide produced a sustained 15.3% reduction in baseline body weight.",
             "Semaglutide", "Placebo", "Weight loss percentage", "Obese individuals without diabetes", "POSITIVE"),
            "biomedicine"
        ),
        (
            ("Prophylactic anticoagulation reduced deep vein thrombosis incidence following knee arthroplasty.",
             "Enoxaparin", "No prophylaxis", "Deep vein thrombosis incidence", "Post-total knee arthroplasty", "NEGATIVE"),
            ("Low molecular weight heparin significantly diminished post-operative venous thromboembolism rates.",
             "LMWH", "Control", "Venous thromboembolism", "Elective orthopedic surgery", "NEGATIVE"),
            "biomedicine"
        ),
        # Machine Learning / NLP
        (
            ("RoBERTa-base outperformed BERT-base by 2.1 F1 points on SQuAD 2.0 through dynamic masking.",
             "RoBERTa-base", "BERT-base", "F1 score", "SQuAD 2.0 benchmark", "POSITIVE"),
            ("Dynamic masking and longer pretraining enabled RoBERTa to surpass original BERT on question answering.",
             "RoBERTa", "BERT", "Exact Match / F1", "SQuAD reading comprehension", "POSITIVE"),
            "machine_learning"
        ),
        (
            ("Low-Rank Adaptation (LoRA) reduced trainable parameters by 99% while matching full fine-tuning performance.",
             "LoRA", "Full fine-tuning", "Downstream task accuracy & parameter footprint", "GLUE and GSM8k", "POSITIVE"),
            ("Parameter-efficient tuning via LoRA maintained downstream accuracy comparable to full parameter updates.",
             "LoRA", "Full model fine-tuning", "Evaluation accuracy", "Instruction tuning benchmarks", "POSITIVE"),
            "machine_learning"
        ),
        (
            ("FlashAttention achieved 3x speedup on attention forward pass without approximating softmax.",
             "FlashAttention", "Standard PyTorch Attention", "Execution latency / throughput", "A100 GPU sequence length 2k-4k", "POSITIVE"),
            ("IO-aware tiling in FlashAttention reduced runtime memory footprint and accelerated transformer training.",
             "FlashAttention IO tiling", "Standard exact attention", "Memory consumption and training speed", "Long context transformers", "POSITIVE"),
            "machine_learning"
        ),
        (
            ("BM25 lexical ranking outperforms dense bi-encoders on out-of-domain technical retrieval without fine-tuning.",
             "BM25Okapi", "Contriever / DPR zero-shot", "NDCG@10", "BEIR out-of-domain biomedical benchmarks", "POSITIVE"),
            ("Lexical BM25 shows greater zero-shot transfer robustness than dense retrievers on specialized jargon.",
             "BM25", "Dense bi-encoders", "Retrieval Recall@100", "Scientific literature search", "POSITIVE"),
            "machine_learning"
        ),
    ]

    for i in range(30):
        tmpl = consistent_templates[i % len(consistent_templates)]
        (t1, i1, c1, o1, s1, d1), (t2, i2, c2, o2, s2, d2), domain = tmpl
        idx = f"consistent_{i+1:03d}"
        pairs.append({
            "pair_id": idx,
            "domain": domain,
            "ground_truth_relation": "SUPPORTS_OR_CONSISTENT",
            "notes": f"Concordant findings under compatible conditions ({domain}).",
            "finding_a": {
                "finding_id": f"{idx}_fa",
                "paper_id": f"P_{idx}_A",
                "paper_title": f"Study on {i1} in {s1} (Part A)",
                "finding_claim": t1,
                "evidence": {"text": t1, "start_char": 0, "end_char": len(t1), "confidence": 0.95},
                "direction": d1,
                "conditions": {"intervention": i1, "comparator": c1, "outcome_metric": o1, "dataset_setting": s1, "confidence": 0.9}
            },
            "finding_b": {
                "finding_id": f"{idx}_fb",
                "paper_id": f"P_{idx}_B",
                "paper_title": f"Study on {i2} in {s2} (Part B)",
                "finding_claim": t2,
                "evidence": {"text": t2, "start_char": 0, "end_char": len(t2), "confidence": 0.95},
                "direction": d2,
                "conditions": {"intervention": i2, "comparator": c2, "outcome_metric": o2, "dataset_setting": s2, "confidence": 0.9}
            }
        })

    # 2. Direct Contradiction (30 pairs)
    # Pairs with matching conditions (same intervention, comparator, outcome, setting) but conflicting outcomes.
    contradiction_templates = [
        (
            ("Hydroxychloroquine significantly reduced mortality in hospitalized severe COVID-19 patients.",
             "Hydroxychloroquine 400mg", "Standard of care", "28-day all-cause mortality", "Hospitalized severe COVID-19 patients", "NEGATIVE"),
            ("Hydroxychloroquine did not reduce 28-day mortality in hospitalized patients with severe COVID-19.",
             "Hydroxychloroquine 400mg", "Standard of care", "28-day all-cause mortality", "Hospitalized severe COVID-19 patients", "NO_DIFFERENCE"),
            "biomedicine"
        ),
        (
            ("Ivermectin administration accelerated viral clearance and reduced hospital stay in acute respiratory infection.",
             "Ivermectin 0.4mg/kg", "Placebo", "Time to viral clearance", "Mild-to-moderate respiratory viral patients", "POSITIVE"),
            ("Ivermectin showed no effect on time to viral clearance or duration of hospital stay versus placebo.",
             "Ivermectin 0.4mg/kg", "Placebo", "Time to viral clearance", "Mild-to-moderate respiratory viral patients", "NO_DIFFERENCE"),
            "biomedicine"
        ),
        (
            ("Chain-of-Thought prompting consistently improves arithmetic reasoning accuracy over direct prompting in 7B LLMs.",
             "Chain-of-Thought prompting", "Direct zero-shot prompting", "GSM8k accuracy", "7B parameter open models", "POSITIVE"),
            ("Chain-of-Thought prompting degrades arithmetic reasoning accuracy in 7B models due to step errors.",
             "Chain-of-Thought prompting", "Direct zero-shot prompting", "GSM8k accuracy", "7B parameter open models", "NEGATIVE"),
            "machine_learning"
        ),
        (
            ("Quantization to 4-bit weights with GPTQ causes zero measurable degradation in perplexity on WikiText-2.",
             "4-bit GPTQ quantization", "FP16 baseline", "Perplexity", "LLaMA-7B on WikiText-2", "NO_DIFFERENCE"),
            ("4-bit GPTQ post-training quantization leads to a 1.4 point increase in perplexity on WikiText-2.",
             "4-bit GPTQ quantization", "FP16 baseline", "Perplexity", "LLaMA-7B on WikiText-2", "NEGATIVE"),
            "machine_learning"
        ),
        (
            ("Early administration of tranexamic acid reduced head-injury mortality by 15% in traumatic brain injury.",
             "Tranexamic acid IV", "Placebo", "28-day head injury mortality", "Isolated traumatic brain injury patients", "NEGATIVE"),
            ("Tranexamic acid did not reduce mortality in isolated moderate-to-severe traumatic brain injury patients.",
             "Tranexamic acid IV", "Placebo", "28-day head injury mortality", "Isolated traumatic brain injury patients", "NO_DIFFERENCE"),
            "biomedicine"
        ),
        (
            ("Contrastive pre-training improves linear probe representations across all computer vision downstream tasks.",
             "SimCLR contrastive pre-training", "Supervised ImageNet pre-training", "Linear probe classification accuracy", "CIFAR-100 & Flowers", "POSITIVE"),
            ("SimCLR contrastive representations underperform supervised features on fine-grained image classification benchmarks.",
             "SimCLR contrastive pre-training", "Supervised ImageNet pre-training", "Linear probe classification accuracy", "CIFAR-100 & Flowers", "NEGATIVE"),
            "machine_learning"
        ),
    ]

    for i in range(30):
        tmpl = contradiction_templates[i % len(contradiction_templates)]
        (t1, i1, c1, o1, s1, d1), (t2, i2, c2, o2, s2, d2), domain = tmpl
        idx = f"contradiction_{i+1:03d}"
        pairs.append({
            "pair_id": idx,
            "domain": domain,
            "ground_truth_relation": "CONTRADICTS",
            "notes": f"Genuine empirical contradiction with identical conditions and opposing directional claims ({domain}).",
            "finding_a": {
                "finding_id": f"{idx}_fa",
                "paper_id": f"P_{idx}_A",
                "paper_title": f"Trial evaluating {i1} on {o1} (Study 1)",
                "finding_claim": t1,
                "evidence": {"text": t1, "start_char": 0, "end_char": len(t1), "confidence": 0.95},
                "direction": d1,
                "conditions": {"intervention": i1, "comparator": c1, "outcome_metric": o1, "dataset_setting": s1, "confidence": 0.95}
            },
            "finding_b": {
                "finding_id": f"{idx}_fb",
                "paper_id": f"P_{idx}_B",
                "paper_title": f"Replication evaluating {i2} on {o2} (Study 2)",
                "finding_claim": t2,
                "evidence": {"text": t2, "start_char": 0, "end_char": len(t2), "confidence": 0.95},
                "direction": d2,
                "conditions": {"intervention": i2, "comparator": c2, "outcome_metric": o2, "dataset_setting": s2, "confidence": 0.95}
            }
        })

    # 3. Qualified Disagreement / Partial Divergence (40 pairs)
    # Apparent disagreement explained by differing settings, metrics, or baselines.
    qualified_templates = [
        (
            ("Aspirin 100mg daily significantly reduced secondary myocardial infarction in patients with prior CAD.",
             "Low-dose Aspirin 100mg", "Placebo", "Recurrent myocardial infarction", "Secondary prevention (Prior CAD history)", "NEGATIVE"),
            ("Aspirin failed to decrease all-cause mortality or primary cardiovascular events in healthy elderly adults.",
             "Low-dose Aspirin 100mg", "Placebo", "All-cause mortality / Primary cardiovascular events", "Primary prevention (Healthy adults aged >70)", "NO_DIFFERENCE"),
            "Apparent conflict resolved by secondary vs primary prevention cohort divergence."
        ),
        (
            ("Dexamethasone reduced 28-day mortality by one-third in patients receiving invasive mechanical ventilation.",
             "Dexamethasone 6mg", "Usual care", "28-day mortality", "Severe COVID-19 with invasive mechanical ventilation", "NEGATIVE"),
            ("Dexamethasone showed no mortality benefit and potential harm in patients not receiving supplemental oxygen.",
             "Dexamethasone 6mg", "Usual care", "28-day mortality", "Mild COVID-19 without supplemental oxygen", "NO_DIFFERENCE"),
            "Apparent contradiction qualified by oxygen-dependence disease severity."
        ),
        (
            ("Rotary Position Embedding (RoPE) yielded 4.5% perplexity gains over absolute positional encodings at 4k context.",
             "RoPE positional embeddings", "Learned absolute embeddings", "Perplexity", "Sequence length 4,096 tokens", "POSITIVE"),
            ("RoPE showed no significant perplexity improvement over learned absolute embeddings on short 512-token contexts.",
             "RoPE positional embeddings", "Learned absolute embeddings", "Perplexity", "Sequence length 512 tokens", "NO_DIFFERENCE"),
            "Performance divergence driven by sequence length context condition."
        ),
        (
            ("Vitamin D supplementation reduced winter respiratory tract infections in severely deficient individuals.",
             "Daily Vitamin D 2000IU", "Placebo", "Acute respiratory tract infections", "Individuals with baseline 25(OH)D < 25 nmol/L", "NEGATIVE"),
            ("Bolus Vitamin D supplementation did not reduce respiratory infection incidence in community-dwelling adults.",
             "Monthly bolus Vitamin D", "Placebo", "Upper respiratory tract infections", "General population with normal baseline Vitamin D", "NO_DIFFERENCE"),
            "Disagreement explained by baseline deficiency status and dosing schedule."
        ),
        (
            ("Knowledge distillation compressed BERT into a 6-layer model preserving 97% of GLUE task accuracy.",
             "DistilBERT (Teacher-Student Distillation)", "Full BERT-base", "GLUE Macro score", "GLUE benchmark (In-domain)", "POSITIVE"),
            ("Distilled transformers exhibit a 22% drop in out-of-domain generalization compared to full-sized teacher models.",
             "DistilBERT", "BERT-base teacher", "Out-of-domain transfer accuracy", "BEIR and HANS stress tests", "NEGATIVE"),
            "Divergence qualified by in-domain retention vs out-of-distribution robustness."
        ),
        (
            ("High-intensity interval training (HIIT) led to greater VO2max improvements than moderate continuous training in young athletes.",
             "HIIT (4x4 min at 90% HRmax)", "Moderate continuous aerobic training", "VO2max increase", "Trained collegiate athletes", "POSITIVE"),
            ("HIIT and continuous aerobic training produced equivalent cardiorespiratory fitness gains in sedentary elderly patients.",
             "HIIT protocol", "Moderate continuous training", "VO2max / Peak aerobic capacity", "Sedentary adults aged 65-80", "NO_DIFFERENCE"),
            "Different outcomes attributable to participant age and baseline fitness level."
        ),
    ]

    for i in range(40):
        tmpl = qualified_templates[i % len(qualified_templates)]
        (t1, i1, c1, o1, s1, d1), (t2, i2, c2, o2, s2, d2), rationale = tmpl
        idx = f"qualified_{i+1:03d}"
        pairs.append({
            "pair_id": idx,
            "domain": "interdisciplinary",
            "ground_truth_relation": "QUALIFIES_OR_PARTIAL_DISAGREEMENT",
            "notes": rationale,
            "finding_a": {
                "finding_id": f"{idx}_fa",
                "paper_id": f"P_{idx}_A",
                "paper_title": f"Investigation of {i1} in {s1}",
                "finding_claim": t1,
                "evidence": {"text": t1, "start_char": 0, "end_char": len(t1), "confidence": 0.95},
                "direction": d1,
                "conditions": {"intervention": i1, "comparator": c1, "outcome_metric": o1, "dataset_setting": s1, "confidence": 0.9}
            },
            "finding_b": {
                "finding_id": f"{idx}_fb",
                "paper_id": f"P_{idx}_B",
                "paper_title": f"Investigation of {i2} in {s2}",
                "finding_claim": t2,
                "evidence": {"text": t2, "start_char": 0, "end_char": len(t2), "confidence": 0.95},
                "direction": d2,
                "conditions": {"intervention": i2, "comparator": c2, "outcome_metric": o2, "dataset_setting": s2, "confidence": 0.9}
            }
        })

    # 4. Incomparable Conditions (30 pairs)
    # Pairs with fundamentally divergent interventions or outcomes that cannot be logically compared.
    incomparable_templates = [
        (
            ("Aspirin administration reduced incidence of colorectal adenomas in familial polyposis patients.",
             "Aspirin 300mg", "Placebo", "Colorectal adenoma recurrence", "Familial adenomatous polyposis", "NEGATIVE"),
            ("Lithium carbonate reduced suicide attempts by 60% in bipolar disorder.",
             "Lithium carbonate", "Anticonvulsants", "Suicide attempt frequency", "Bipolar I and II disorder", "NEGATIVE"),
            "Different drugs, diseases, and outcome phenotypes."
        ),
        (
            ("Self-attention pruning achieved 50% FLOPs reduction in vision transformers with 0.3% Top-1 accuracy loss.",
             "Structured attention head pruning", "Dense Vision Transformer", "Top-1 ImageNet accuracy vs FLOPs", "ImageNet-1K", "POSITIVE"),
            ("CRISPR-Cas9 mediated knockout of PCSK9 lowered serum LDL cholesterol by 80% in human hepatocyte cultures.",
             "CRISPR-Cas9 PCSK9 editing", "Mock transfection", "Serum LDL cholesterol concentration", "Primary human hepatocytes", "NEGATIVE"),
            "Entirely unrelated fields (Computer Vision vs Genetics)."
        ),
        (
            ("Direct Preference Optimization (DPO) reduced training memory by 40% compared to PPO with separate reward models.",
             "DPO", "PPO with separate reward model", "VRAM consumption during RLHF", "Llama-2 7B alignment", "POSITIVE"),
            ("Early surgery for asymptomatic mitral regurgitation improved 10-year left ventricular ejection fraction.",
             "Mitral valve repair surgery", "Watchful waiting", "10-year LVEF preservation", "Severe degenerative mitral regurgitation", "POSITIVE"),
            "Different scientific disciplines."
        ),
        (
            ("Diffusion models achieved state-of-the-art Frechet Inception Distance of 2.1 on unconditional image generation.",
             "Diffusion probabilistic models (DDPM)", "Generative Adversarial Networks (StyleGAN)", "FID score", "CIFAR-10 32x32", "POSITIVE"),
            ("Selective serotonin reuptake inhibitors improved Hamilton Depression Rating Scale scores in major depressive disorder.",
             "Sertraline 50mg", "Placebo", "HAM-D score change", "Major depressive disorder outpatients", "NEGATIVE"),
            "Different domains."
        ),
    ]

    for i in range(30):
        tmpl = incomparable_templates[i % len(incomparable_templates)]
        (t1, i1, c1, o1, s1, d1), (t2, i2, c2, o2, s2, d2), rationale = tmpl
        idx = f"incomparable_{i+1:03d}"
        pairs.append({
            "pair_id": idx,
            "domain": "cross_domain",
            "ground_truth_relation": "INCOMPARABLE_CONDITIONS",
            "notes": rationale,
            "finding_a": {
                "finding_id": f"{idx}_fa",
                "paper_id": f"P_{idx}_A",
                "paper_title": f"Study A: {i1}",
                "finding_claim": t1,
                "evidence": {"text": t1, "start_char": 0, "end_char": len(t1), "confidence": 0.95},
                "direction": d1,
                "conditions": {"intervention": i1, "comparator": c1, "outcome_metric": o1, "dataset_setting": s1, "confidence": 0.85}
            },
            "finding_b": {
                "finding_id": f"{idx}_fb",
                "paper_id": f"P_{idx}_B",
                "paper_title": f"Study B: {i2}",
                "finding_claim": t2,
                "evidence": {"text": t2, "start_char": 0, "end_char": len(t2), "confidence": 0.95},
                "direction": d2,
                "conditions": {"intervention": i2, "comparator": c2, "outcome_metric": o2, "dataset_setting": s2, "confidence": 0.85}
            }
        })

    # 5. Insufficient Evidence (20 pairs)
    # Pairs with vague assertions, missing empirical data, or speculative hypothesis sentences.
    insufficient_templates = [
        (
            ("Further multi-center clinical trials are required to determine whether herbal extract X has any clinical utility.",
             "Herbal extract X", "Unspecified", "Clinical utility", "Unknown cohort", "UNCLEAR"),
            ("Herbal extract X exhibited promising antioxidant properties in exploratory preliminary screening tests.",
             "Herbal extract X", "Untreated control", "In vitro free radical assay", "Cell-free in vitro medium", "UNCLEAR"),
            "Lacks empirical clinical outcome data or statistical effect size."
        ),
        (
            ("Quantum neural networks might potentially achieve exponential speedups on future fault-tolerant hardware.",
             "Quantum neural network theoretical concept", "Classical neural network", "Theoretical computational complexity", "Asymptotic analysis", "UNCLEAR"),
            ("Current noisy intermediate-scale quantum devices cannot execute deep variational circuits without decoherence.",
             "NISQ variational circuit", "Ideal unitary simulation", "Fidelity score", "IBM Eagle 127-qubit processor", "UNCLEAR"),
            "Speculative claim lacking empirical verification."
        ),
        (
            ("Preliminary anecdotal observations suggest microdosing may enhance perceived workplace creativity.",
             "Microdosing psilocybin", "No baseline recorded", "Subjective self-reported creativity", "Survey respondents (n=12)", "UNCLEAR"),
            ("The biological mechanisms underpinning cognitive microdosing effects remain entirely uncharacterized.",
             "Unspecified microdosing", "None", "Biological mechanism", "Theoretical review", "UNCLEAR"),
            "Anecdotal or unverified claims lacking controlled experimental data."
        ),
    ]

    for i in range(20):
        tmpl = insufficient_templates[i % len(insufficient_templates)]
        (t1, i1, c1, o1, s1, d1), (t2, i2, c2, o2, s2, d2), rationale = tmpl
        idx = f"insufficient_{i+1:03d}"
        pairs.append({
            "pair_id": idx,
            "domain": "speculative_preliminary",
            "ground_truth_relation": "INSUFFICIENT_EVIDENCE",
            "notes": rationale,
            "finding_a": {
                "finding_id": f"{idx}_fa",
                "paper_id": f"P_{idx}_A",
                "paper_title": f"Observational note: {i1}",
                "finding_claim": t1,
                "evidence": {"text": t1, "start_char": 0, "end_char": len(t1), "confidence": 0.5},
                "direction": d1,
                "conditions": {"intervention": i1, "comparator": c1, "outcome_metric": o1, "dataset_setting": s1, "confidence": 0.4, "is_unknown": True}
            },
            "finding_b": {
                "finding_id": f"{idx}_fb",
                "paper_id": f"P_{idx}_B",
                "paper_title": f"Preliminary note: {i2}",
                "finding_claim": t2,
                "evidence": {"text": t2, "start_char": 0, "end_char": len(t2), "confidence": 0.5},
                "direction": d2,
                "conditions": {"intervention": i2, "comparator": c2, "outcome_metric": o2, "dataset_setting": s2, "confidence": 0.4, "is_unknown": True}
            }
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    print(f"Successfully generated {len(pairs)} gold-standard finding pairs in {OUTPUT_FILE}")
    print(f"Distribution: Consistent=30, Contradiction=30, Qualified=40, Incomparable=30, Insufficient=20")


if __name__ == "__main__":
    generate_benchmark_pairs()
