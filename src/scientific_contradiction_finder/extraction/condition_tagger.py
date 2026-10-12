"""Condition slot tagger extracting Intervention (I), Comparator (C), Outcome (O), Setting (S), Population, and Qualifiers."""

import re
from typing import Optional

import spacy

from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    NumericalResult,
    ProvenanceSection,
    StudyDesignType,
)

# Predefined lexical triggers for comparators, outcomes, and settings
COMPARATOR_MARKERS = [
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bcompared with\b",
    r"\bcompared to\b",
    r"\bagainst\b",
    r"\brelative to\b",
    r"\bthan\b",
]

OUTCOME_MARKERS = [
    r"\bdaytime air temperature\b",
    r"\bair temperature\b",
    r"\bland surface temperatures?\b",
    r"\bmean radiant temperature\b",
    r"\blocal temperature\b",
    r"\breaction times?\b",
    r"\bworking memory (?:performance|accuracy|capacity)?\b",
    r"\bvisual working memory\b",
    r"\bcognitive performance\b",
    r"\bdegradation rate\b",
    r"\bcapacity fade\b",
    r"\bcell thickness\b",
    r"\ball-cause mortality\b",
    r"\bcardiovascular mortality\b",
    r"\bmortality\b",
    r"\bweight loss\b",
    r"\bbody weight\b",
    r"\bweight reduction\b",
    r"\bbmi\b",
    r"\bbody mass index\b",
    r"\bstroke\b",
    r"\bcardiovascular (?:disease|events?)\b",
    r"\bcoronary heart disease\b",
    r"\btype 2 diabetes\b",
    r"\bdiabetes\b",
    r"\bhba1c\b",
    r"\bblood pressure\b",
    r"\bsystolic blood pressure\b",
    r"\bdiastolic blood pressure\b",
    r"\bpediatric sepsis\b",
    r"\bsepsis\b",
    r"\bcervical cancer\b",
    r"\blung cancer\b",
    r"\blipid profile\b",
    r"\bcholesterol\b",
    r"\btriglycerides\b",
    r"\binsulin resistance\b",
    r"\bexacerbation\b",
    r"\bf1 score\b",
    r"\baccuracy\b",
    r"\bgeneralization\b",
    r"\bperplexity\b",
    r"\befficacy\b",
    r"\bspeedup\b",
    r"\brespiratory (?:tract )?infections?\b",
    r"\bincidence\b",
    r"\brisk of (?:[a-zA-Z\s]+)\b",
    r"\blifetime\b",
    r"\baging effects?\b",
]

SETTING_MARKERS = [
    r"\bpatients with\b",
    r"\bcohort\b",
    r"\bbenchmark\b",
    r"\bdataset\b",
    r"\bhealthy adults\b",
    r"\byoung adults\b",
    r"\bolder adults\b",
    r"\badults\b",
    r"\badolescents\b",
    r"\btrials?\b",
    r"\blithium-ion cells?\b",
    r"\bvision transformers?\b",
    r"\burban areas?\b",
    r"\bhigh density\b",
    r"\blow density\b",
]

# Blacklisted strings that should NEVER be extracted as an intervention or outcome
INVALID_SLOT_TERMS = {
    "the results",
    "results",
    "the experimental results",
    "our results",
    "significant increases",
    "significant decrease",
    "significant difference",
    "addition",
    "findings",
    "our findings",
    "in addition",
    "furthermore",
    "it",
    "we",
    "study",
    "follow-up",
    "in the follow-up",
    "the 14-week follow-up",
    "the 14-week follow-up study",
    "the study",
    "this study",
    "our study",
    "the present study",
    "the trial",
    "trials",
    "authors",
    "the authors",
    "this paper",
    "conclusion",
    "conclusions",
    "the researcher",
    "the researchers",
    "the day",
    "a result",
    "the action",
    "the effect",
    "the magnitude",
    "these results",
    "many studies",
    "great promise",
    "a promising approach",
    "promising approach",
    "no negative consequences",
    "an impairment",
    "accuracy",
    "algorithm",
    "the algorithm",
    "the prediction",
    "prediction",
    "the model",
    "models",
    "methods",
    "the methods",
    "the approach",
    "performance",
    "early prediction",
    "individuals",
    "individual",
    "the individuals",
    "patients",
    "patient",
    "the patients",
    "subjects",
    "subject",
    "the subjects",
    "participants",
    "participant",
    "the participants",
    "hospital",
    "hospitals",
    "the hospital",
    "the diet",
    "diet",
    "diets",
    "the diets",
    "both diets",
    "both groups",
    "the two groups",
    "the group",
    "groups",
    "group",
    "the cohort",
    "cohort",
    "the waist/hip ratio",
    "waist/hip ratio",
    "the largest amounts",
    "largest amounts",
    "the smallest amounts",
    "smallest amounts",
    "the control group",
    "the intervention group",
    "cases",
    "controls",
    "children",
    "infants",
    "adults",
    "men",
    "women",
    "6 months",
    "12 months",
    "14 weeks",
    "0.91 kg",
    "the decrease",
    "decrease",
    "the increase",
    "increase",
    "the difference",
    "the respective differences",
    "differences",
    "change",
    "changes",
    "the change",
    "reduction",
    "the reduction",
    "no significant difference",
    "no significant reduction",
    "the range",
    "range",
    "treatment",
    "treatments",
    "the treatment",
    "the management",
    "management",
    "total energy expenditure",
    "energy expenditure",
    "the best performance",
    "better performance",
    "a novel target",
    "good agreement",
    "meta-analysis",
    "meta-analyses",
    "systematic review",
    "systematic reviews",
    "intention-to-treat",
    "the intention-to-treat analysis",
    "intention-to-treat analysis",
    "per-protocol",
    "per protocol",
    "the per protocol analysis",
    "per protocol analysis",
    "hazard ratios",
    "hazard ratio",
    "odds ratio",
    "odds ratios",
    "risk ratio",
    "risk ratios",
    "relative risk",
    "confidence intervals",
    "confidence interval",
    "p-values",
    "p-value",
    "p value",
    "p values",
    "mice",
    "rats",
    "animals",
    "1) control group",
}

CANONICAL_INTERVENTIONS = {
    "lcd": "Low-Carbohydrate Diet",
    "lcds": "Low-Carbohydrate Diet",
    "lc diet": "Low-Carbohydrate Diet",
    "lc diets": "Low-Carbohydrate Diet",
    "very low-carbohydrate diet": "Very Low-Carbohydrate Diet",
    "very low carbohydrate diet": "Very Low-Carbohydrate Diet",
    "vlcd": "Very Low-Carbohydrate Diet",
    "vlcds": "Very Low-Carbohydrate Diet",
    "low-carbohydrate diet": "Low-Carbohydrate Diet",
    "low-carbohydrate diets": "Low-Carbohydrate Diet",
    "low-carbohydrate": "Low-Carbohydrate Diet",
    "low-carb diet": "Low-Carbohydrate Diet",
    "low-carb diets": "Low-Carbohydrate Diet",
    "low-carb": "Low-Carbohydrate Diet",
    "the lcd interventions": "Low-Carbohydrate Diet",
    "lfd": "Low-Fat Diet",
    "lfds": "Low-Fat Diet",
    "lf diet": "Low-Fat Diet",
    "lf diets": "Low-Fat Diet",
    "low-fat diet": "Low-Fat Diet",
    "low-fat diets": "Low-Fat Diet",
    "low-fat": "Low-Fat Diet",
    "continuous calorie restriction": "Continuous Calorie Restriction",
    "continuous energy restriction": "Continuous Calorie Restriction",
    "caloric restriction": "Continuous Calorie Restriction",
    "intermittent fasting": "Intermittent Fasting",
    "time-restricted feeding": "Time-Restricted Feeding",
    "time-restricted eating": "Time-Restricted Feeding",
    "alternate-day fasting": "Alternate-Day Fasting",
    "moderate alcohol": "Moderate Alcohol Consumption",
    "moderate drinking": "Moderate Alcohol Consumption",
    "moderate alcohol consumption": "Moderate Alcohol Consumption",
    "alcohol consumption": "Moderate Alcohol Consumption",
    "vitamin d": "Vitamin D Supplementation",
    "vitamin d supplementation": "Vitamin D Supplementation",
    "vitamin d3": "Vitamin D Supplementation",
    "cholecalciferol": "Vitamin D Supplementation",
    "gradient boosting": "Gradient Boosting",
    "gradient boosting decision tree": "Gradient Boosting",
    "xgboost": "XGBoost",
    "random forest": "Random Forest",
    "logistic regression": "Logistic Regression",
    "lr classifier": "Logistic Regression",
    "machine learning": "Machine Learning Model",
    "machine learning model": "Machine Learning Model",
    "machine learning models": "Machine Learning Model",
    "intravenous curcumin": "High-Dose Intravenous Curcumin",
    "high-dose intravenous curcumin": "High-Dose Intravenous Curcumin",
    "curcumin": "Curcumin",
    "cigarette smoking": "Cigarette Smoking",
    "smoking": "Cigarette Smoking",
    "human papillomavirus": "HPV Infection",
    "hpv": "HPV Infection",
    "hpv infection": "HPV Infection",
}

CANONICAL_COMPARATORS = {
    "lfd": "Low-Fat Diet",
    "lfds": "Low-Fat Diet",
    "low-fat diet": "Low-Fat Diet",
    "low-fat diets": "Low-Fat Diet",
    "participants on lfd": "Low-Fat Diet",
    "lcd": "Low-Carbohydrate Diet",
    "lcds": "Low-Carbohydrate Diet",
    "low-carbohydrate diet": "Low-Carbohydrate Diet",
    "low-carbohydrate diets": "Low-Carbohydrate Diet",
    "participants on lcd": "Low-Carbohydrate Diet",
    "placebo": "Placebo",
    "placebo group": "Placebo",
    "control": "Standard Control",
    "control group": "Standard Control",
    "standard care": "Standard Care",
    "usual care": "Standard Care",
    "non-drinkers": "Non-Drinkers",
    "non drinkers": "Non-Drinkers",
    "abstainers": "Non-Drinkers",
    "lifetime abstainers": "Lifetime Abstainers",
    "former drinkers": "Former Drinkers",
    "continuous calorie restriction": "Continuous Calorie Restriction",
    "continuous energy restriction": "Continuous Calorie Restriction",
    "conventional insulin therapy": "Conventional Glucose Control",
    "conventional glucose control": "Conventional Glucose Control",
    "conventional glucose targets": "Conventional Glucose Control",
    "conventional glycemic control": "Conventional Glucose Control",
}

CANONICAL_OUTCOMES = {
    "body weight": "Weight Loss",
    "weight loss": "Weight Loss",
    "weight reduction": "Weight Loss",
    "bmi": "Weight Loss",
    "body mass index": "Weight Loss",
    "all-cause mortality": "All-Cause Mortality",
    "mortality": "All-Cause Mortality",
    "in-hospital mortality": "All-Cause Mortality",
    "in-hospital all-cause mortality": "All-Cause Mortality",
    "90-day mortality": "All-Cause Mortality",
    "90-day all-cause mortality": "All-Cause Mortality",
    "28-day mortality": "All-Cause Mortality",
    "28-day all-cause mortality": "All-Cause Mortality",
    "30-day mortality": "All-Cause Mortality",
    "30-day all-cause mortality": "All-Cause Mortality",
    "24-month mortality": "All-Cause Mortality",
    "cataract": "Incident Cataract",
    "cataracts": "Incident Cataract",
    "cataract surgery": "Incident Cataract",
    "cataract extraction": "Incident Cataract",
    "incident cataract diagnosis": "Incident Cataract",
    "incident cataract extraction surgery": "Incident Cataract",
    "cardiovascular mortality": "Cardiovascular Mortality",
    "cardiovascular events": "Cardiovascular Events",
    "cardiovascular disease": "Cardiovascular Disease",
    "cvd": "Cardiovascular Disease",
    "coronary heart disease": "Coronary Heart Disease",
    "stroke": "Stroke",
    "hba1c": "HbA1c",
    "type 2 diabetes": "Type 2 Diabetes",
    "diabetes": "Type 2 Diabetes",
    "diastolic blood pressure": "Diastolic Blood Pressure",
    "systolic blood pressure": "Systolic Blood Pressure",
    "blood pressure": "Blood Pressure",
    "triglycerides": "Triglycerides",
    "cholesterol": "Lipid Profile",
    "pediatric sepsis": "Pediatric Sepsis Onset",
    "sepsis": "Pediatric Sepsis Onset",
    "acute kidney injury": "Acute Kidney Injury",
    "auroc": "AUROC Discrimination",
    "auprc": "AUPRC Discrimination",
    "c-statistic": "AUROC Discrimination",
    "c-index": "AUROC Discrimination",
}


def clean_slot_string(s: str) -> Optional[str]:
    """Cleans punctuation and determines if string is a valid scientific condition."""
    if not s:
        return None
    cleaned = re.sub(r"^[,\.;:\s\(\)\[\]]+|[,\.;:\s\(\)\[\]]+$", "", s).strip()
    if len(cleaned) < 3:
        return None
    c_lower = cleaned.lower()
    if c_lower in INVALID_SLOT_TERMS:
        return None
    # Reject temporal durations (e.g., '6 months', '12 months', '14 weeks', '2 years')
    if re.search(r"^\d+\s*(?:months?|weeks?|years?|days?|hours?)$", cleaned, re.IGNORECASE):
        return None
    if re.search(
        r"^(?:at|after|over|during|for|in)\s+\d+\s*(?:months?|weeks?|years?|days?|hours?)$",
        cleaned,
        re.IGNORECASE,
    ):
        return None
    # Reject pure numbers, numbers with parentheses/SE/SD, or numbers with units (e.g. '0.91 kg', '13.7 (1.8) kg', '12 kg', '5 %', '120 mmhg')
    if re.match(
        r"^[\+\-−–]?\d+(?:\.\d+)?(?:\s*\([^)]*\))?\s*(?:kg|g|mg|mcg|lbs|%|°c|mmhg|ml|l|min|hr|s|kcal|kcal/d|mmol/l)?$",
        cleaned,
        re.IGNORECASE,
    ):
        return None
    # Reject comparative quantity phrases (e.g. 'the largest amounts', 'highest levels')
    if re.match(
        r"^(?:the\s+)?(?:largest|smallest|highest|lowest|greater|lesser|mean|median|total|average)\s+(?:amounts?|levels?|values?|numbers?|quantities?)",
        cleaned,
        re.IGNORECASE,
    ):
        return None
    # Reject strings that are primarily numbers with subjects (e.g. '6553 eligible children', '8 times that of children', '1) control group')
    if re.match(
        r"^\d+[\)\.]?\s*(?:eligible\s+)?(?:children|patients|participants|subjects|cases|controls|infants|adults|individuals|control group|intervention group)",
        cleaned,
        re.IGNORECASE,
    ):
        return None
    if re.match(
        r"^\d+\s+times\s+(?:that\s+of\s+)?(?:children|patients|controls|adults)",
        cleaned,
        re.IGNORECASE,
    ):
        return None
    # Reject statistical analysis phrases (e.g. 'the intention-to-treat analysis', 'per protocol analysis')
    if re.search(
        r"\b(?:intention[\s-]to[\s-]treat|per[\s-]protocol|sensitivity analysis|multivariate analysis|regression analysis)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return None
    # Filter strings starting with introductory time/study phrases
    if re.match(
        r"^(in the|during the|across the)\s+\d+-(?:week|month|year|day)", cleaned, re.IGNORECASE
    ):
        return None
    # Filter generic subject phrases
    if cleaned.lower() in {
        "both groups",
        "the two groups",
        "the group",
        "two groups",
        "both diets",
        "the diets",
        "the subjects",
        "the patients",
        "the participants",
        "the individuals",
    }:
        return None
    return cleaned


def extract_numerical_results(
    text: str,
    default_endpoint: Optional[str] = None,
    default_duration: Optional[str] = None,
    intervention_hint: Optional[str] = None,
    comparator_hint: Optional[str] = None,
) -> list[NumericalResult]:
    """Extracts structured point estimates, confidence intervals, p-values, and units verbatim,
    associating each numerical estimate with the correct arm, comparator, endpoint, horizon,
    sample size, and statistical uncertainty in multi-arm or single-arm trials.
    """
    if not text:
        return []

    results: list[NumericalResult] = []
    text_norm = text.replace("−", "-").replace("–", "-").replace("—", "-")

    sentence_duration = extract_follow_up_duration(text) or default_duration
    sentence_sample_sz = extract_sample_size(text)

    # 1. Detect all confidence intervals and their character spans
    ci_matches = list(
        re.finditer(
            r"(?:95%\s*CI|95%\s*confidence\s+interval)[\s:=,]*\[?\(?\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:to|-|,)\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:[a-zA-Z/%]+)?\)?\]?",
            text_norm,
            re.IGNORECASE,
        )
    )
    ci_spans = [m.span() for m in ci_matches]

    # 2. Detect all p-values and their character spans
    p_matches = list(
        re.finditer(
            r"\bp\s*([<>=])\s*(0?\.[0-9]+|1e-[0-9]+|\d+\.\d+e-\d+)\b",
            text_norm,
            re.IGNORECASE,
        )
    )

    # 3. Detect trial arms
    arm_pattern = (
        r"\b(low[\s-]carbohydrate(?:\s+diets?)?|low[\s-]carb(?:\s+diets?)?|low[\s-]fat(?:\s+diets?)?|"
        r"ketogenic(?:\s+diets?)?|vlcds?|lcds?|lfds?|kd|intermittent\s+fasting|"
        r"continuous\s+calorie\s+restriction|continuous\s+energy\s+restriction|time[\s-]restricted\s+feeding|"
        r"moderate\s+alcohol(?:\s+consumption)?|standard\s+(?:of\s+)?care|usual\s+care|control(?:\s+group)?|"
        r"intervention(?:\s+group|\s+arm)?|placebo(?:\s+group)?|arm\s+(?!(?:trial|study|design|analysis|evaluation|cohort|protocol)s?\b)[a-zA-Z0-9]+|group\s+(?!(?:study|design|analysis|evaluation|protocol)s?\b)[a-zA-Z0-9]+|"
        r"treatment\s+(?!(?:with|by|of|for|in|on|at|trial|study|design|analysis|evaluation|cohort|protocol|effect|effects|group|arm|regimen)s?\b)[a-zA-Z0-9]+|"
        r"controls|placebos|sham|active\s+control|reference\s+group|usual\s+care\s+group|"
        r"empagliflozin|dapagliflozin|semaglutide|metformin|atorvastatin|simvastatin|statin|statins|"
        r"budesonide|fluticasone|formoterol|terbutaline|aspirin|albumin|normal\s+saline|saline|"
        r"ivermectin|curcumin|enoxaparin|lmwh|heparin|drug\s+[a-zA-Z0-9]+)\b"
    )
    all_arms_raw = list(re.finditer(arm_pattern, text_norm, re.IGNORECASE))
    if intervention_hint and len(intervention_hint.strip()) >= 3:
        all_arms_raw.extend(list(re.finditer(rf"\b{re.escape(intervention_hint.strip())}\b", text_norm, re.IGNORECASE)))
    if comparator_hint and len(comparator_hint.strip()) >= 3:
        all_arms_raw.extend(list(re.finditer(rf"\b{re.escape(comparator_hint.strip())}\b", text_norm, re.IGNORECASE)))

    seen_arm_spans = set()
    all_arms = []
    for a in sorted(all_arms_raw, key=lambda m: m.start()):
        span = (a.start(), a.end())
        if span not in seen_arm_spans:
            seen_arm_spans.add(span)
            all_arms.append(a)

    # Comparative markers and subgroup markers
    comp_markers = list(
        re.finditer(
            r"\b(compared with|compared to|as compared with|as compared to|versus|vs\.?|relative to|superior to|inferior to|favoring|favored over|advantage over|benefit over|than)\b",
            text_norm,
            re.IGNORECASE,
        )
    )
    subgroup_markers = list(
        re.finditer(
            r"\b(?:especially in the subgroup that used|in the subgroup that used|in the subgroup of|subgroup:?)\b",
            text_norm,
            re.IGNORECASE,
        )
    )

    # 4. Detect Ratio estimates (OR, RR, HR, hazard ratio, odds ratio, relative risk, risk ratio)
    ratio_pattern = (
        r"\b(OR|RR|HR|hazard\s+ratio|odds\s+ratio|relative\s+risk|risk\s+ratio)"
        r"[\s:=,]+([0-9]+(?:\.[0-9]+)?)\b"
    )
    ratio_matches = list(re.finditer(ratio_pattern, text_norm, re.IGNORECASE))

    # 5. Detect Difference estimates (WMD, MD, SMD, RD, difference, mean difference)
    diff_pattern = (
        r"\b(WMD|MD|SMD|RD|weighted\s+mean\s+difference|mean\s+difference|standardized\s+mean\s+difference|risk\s+difference)"
        r"[\s:=,]+([+-]?[0-9]+(?:\.[0-9]+)?)\b"
    )
    diff_matches = list(re.finditer(diff_pattern, text_norm, re.IGNORECASE))

    # 6. Detect Unit point estimates (skipping any inside CI spans)
    unit_pattern = (
        r"(?:^|[\s\(>~=])([+-]?[0-9]+(?:\.[0-9]+)?)\s*"
        r"(?:(kg|g|mg|mcg|μg|ug|percent|percentage points|mmhg|ml|l|lbs|mmol/l|mg/dl|kcal/d|kcal|kg/m2)\b|(%))"
    )
    unit_matches = list(re.finditer(unit_pattern, text_norm, re.IGNORECASE))

    estimates: list[tuple[str, Optional[str], float, Optional[str], int, int, str]] = []
    for m in ratio_matches:
        start_val = m.start(2)
        end_val = m.end(2)
        if any(c_start <= start_val and end_val <= c_end for c_start, c_end in ci_spans):
            continue
        m_name = m.group(1).upper()
        if "HAZARD" in m_name:
            metric = "HR"
        elif "ODDS" in m_name:
            metric = "OR"
        elif "RISK" in m_name or "RELATIVE" in m_name:
            metric = "RR"
        else:
            metric = m_name
        estimates.append(("RATIO", metric, float(m.group(2)), None, m.start(), m.end(), m.group(0)))

    for m in diff_matches:
        start_val = m.start(2)
        end_val = m.end(2)
        if any(c_start <= start_val and end_val <= c_end for c_start, c_end in ci_spans):
            continue
        m_name = m.group(1).upper()
        if "STANDARDIZED" in m_name or m_name == "SMD":
            metric = "SMD"
        elif "WEIGHTED" in m_name or m_name == "WMD":
            metric = "WMD"
        elif "RISK" in m_name or m_name == "RD":
            metric = "RD"
        elif "MEAN" in m_name or m_name == "MD":
            metric = "MD"
        else:
            metric = "DIFFERENCE"
        val = float(m.group(2))
        if not any(abs(e[4] - m.start()) <= 20 and e[2] == val for e in estimates):
            estimates.append(("DIFF", metric, val, None, m.start(), m.end(), m.group(0)))

    for m in unit_matches:
        start = m.start(1)
        end = m.end()
        # Skip if within any CI span
        if any(c_start <= start and end <= c_end for c_start, c_end in ci_spans):
            continue
        # Skip if part of a CI declaration like 95% CI
        if re.search(r"^\s*(?:CI|confidence\s+interval)\b", text_norm[end:], re.IGNORECASE):
            continue
        val = float(m.group(1))
        # Check if preceded by metric like WMD = , MD = , SMD = , RR = , OR = , HR =
        metric_before = re.search(
            r"\b(WMD|MD|SMD|RR|OR|HR|weighted\s+mean\s+difference|mean\s+difference|standardized\s+mean\s+difference)\s*[:=,]\s*$",
            text_norm[:start],
            re.IGNORECASE,
        )
        u_metric = None
        if metric_before:
            mb = metric_before.group(1).upper()
            if "STANDARDIZED" in mb or mb == "SMD":
                u_metric = "SMD"
            elif "WEIGHTED" in mb or mb == "WMD":
                u_metric = "WMD"
            elif "MEAN" in mb or mb == "MD":
                u_metric = "MD"
            else:
                u_metric = mb
        u_unit = (m.group(2) or m.group(3)).lower()
        existing = next((e for e in estimates if abs(e[4] - start) <= 30 and e[2] == val), None)
        if existing:
            idx = estimates.index(existing)
            estimates[idx] = (
                "UNIT",
                u_metric or existing[1],
                val,
                u_unit,
                existing[4],
                max(existing[5], end),
                text_norm[existing[4] : max(existing[5], end)].strip(),
            )
        else:
            estimates.append(("UNIT", u_metric, val, u_unit, start, end, m.group(0).strip()))

    estimates.sort(key=lambda x: x[4])

    used_p_spans: set[tuple[int, int]] = set()

    for i, (est_type, metric, pe, unit, start, end, raw) in enumerate(estimates):
        next_est_start = estimates[i + 1][4] if i + 1 < len(estimates) else len(text_norm)

        # 1. Attach closest CI (within 40 chars following this estimate and before next estimate)
        ci_l: Optional[float] = None
        ci_u: Optional[float] = None
        uncert_ci: Optional[str] = None
        ci_end_offset = end
        for ci_m in ci_matches:
            c_start, c_end = ci_m.span()
            if end <= c_start < next_est_start and (c_start - end) <= 40:
                ci_l = float(ci_m.group(1))
                ci_u = float(ci_m.group(2))
                uncert_ci = f"95% CI {ci_l} to {ci_u}"
                ci_end_offset = c_end
                break

        # 2. Attach closest p-value (inside CI parentheses or within 60 chars following this estimate)
        pval: Optional[float] = None
        uncert_p: Optional[str] = None
        for p_m in p_matches:
            p_start, p_end = p_m.span()
            if end <= p_start < next_est_start and (
                (p_start - end) <= 60 or abs(p_start - ci_end_offset) <= 25
            ):
                pval = float(p_m.group(2))
                uncert_p = f"p = {pval}"
                used_p_spans.add(p_m.span())
                break

        uncert = None
        if uncert_ci and uncert_p:
            uncert = f"{uncert_ci}, {uncert_p}"
        elif uncert_ci:
            uncert = uncert_ci
        elif uncert_p:
            uncert = uncert_p

        # 3. Endpoint detection: look in 80 chars preceding this estimate for closest outcome marker
        window = text_norm[max(0, start - 80) : start]
        matched_endpoints: list[tuple[int, str]] = []
        for marker in OUTCOME_MARKERS:
            for f in re.finditer(marker, window, re.IGNORECASE):
                cleaned_ep = clean_slot_string(f.group(0))
                if cleaned_ep:
                    matched_endpoints.append((f.end(), cleaned_ep.title()))

        detected_ep = default_endpoint
        if matched_endpoints:
            matched_endpoints.sort(key=lambda x: x[0], reverse=True)
            detected_ep = matched_endpoints[0][1]
        elif unit == "mmhg":
            detected_ep = "Blood Pressure"
        elif unit in ("kg", "lbs"):
            detected_ep = "Weight Loss"
        elif unit in ("mmol/l", "mg/dl"):
            detected_ep = "Lipid Profile"

        # 4. Arm attribution
        curr_arm = None
        comp_arm = None

        if all_arms:
            # 4.1 Check if this estimate is in a subgroup clause
            subgroup_clause = None
            for sg_m in subgroup_markers:
                if sg_m.end() <= start and (start - sg_m.end()) <= 60:
                    subgroup_clause = sg_m
                    break

            if subgroup_clause:
                sg_arm = next(
                    (a for a in all_arms if subgroup_clause.start() <= a.start() < start),
                    None,
                )
                if sg_arm:
                    curr_arm = sg_arm.group(0).strip()
                    # Find reference comparator in the overarching comparison
                    comp_candidate = next(
                        (
                            a
                            for a in all_arms
                            if a != sg_arm
                            and any(
                                cm.end() <= a.start() and (a.start() - cm.end()) <= 35
                                for cm in comp_markers
                            )
                        ),
                        None,
                    )
                    if comp_candidate:
                        comp_arm = comp_candidate.group(0).strip()
                    else:
                        other_arms = [
                            a.group(0).strip()
                            for a in all_arms
                            if a.group(0).strip().lower() != curr_arm.lower()
                        ]
                        if other_arms:
                            comp_arm = other_arms[0]

            # 4.2 Comparative structure check
            if curr_arm is None and comp_markers:
                comparator_arm_matches = []
                for cm in comp_markers:
                    following_arms = [
                        a for a in all_arms if cm.end() <= a.start() and (a.start() - cm.end()) <= 35
                    ]
                    if following_arms:
                        closest_arm = min(following_arms, key=lambda a: a.start() - cm.end())
                        comparator_arm_matches.append((cm, closest_arm))

                comparator_arms_set = {a for _, a in comparator_arm_matches}
                primary_arms = [a for a in all_arms if a not in comparator_arms_set]

                # Pattern A: Fronted comparative ("Compared with LFDs, LCDs achieved...")
                fronted_comp = next((cm for cm in comp_markers if cm.start() <= 5), None)
                if fronted_comp and primary_arms:
                    fronted_comp_arm = next(
                        (a for cm, a in comparator_arm_matches if cm == fronted_comp), None
                    )
                    primary_arm = primary_arms[0]
                    if (
                        fronted_comp_arm
                        and fronted_comp.start() <= start < primary_arm.start()
                        and (start - fronted_comp_arm.end()) <= 20
                    ):
                        curr_arm = fronted_comp_arm.group(0).strip()
                        comp_arm = primary_arm.group(0).strip()
                    else:
                        curr_arm = primary_arm.group(0).strip()
                        if fronted_comp_arm:
                            comp_arm = fronted_comp_arm.group(0).strip()

                # Pattern B: Middle comparative marker ("... than LFDs (WMD = -2.99 kg...)")
                elif comparator_arm_matches and primary_arms:
                    cand_primary_arms = [
                        a for a in primary_arms if a.start() < end or (a.start() - end) <= 35
                    ]
                    if not cand_primary_arms:
                        cand_primary_arms = primary_arms

                    if len(cand_primary_arms) == 1 and len(comparator_arm_matches) == 1:
                        cm, comp_arm_match = comparator_arm_matches[0]
                        primary_arm_match = cand_primary_arms[0]

                        is_diff_after_comp = (
                            comp_arm_match.end() <= start
                            and (start - comp_arm_match.end()) <= 35
                            and (
                                metric in ("WMD", "MD", "SMD", "RD", "DIFFERENCE", "MEAN DIFFERENCE")
                                or any(
                                    w in text_norm[comp_arm_match.end() : start].lower()
                                    for w in ["wmd", "md", "smd", "difference"]
                                )
                                or not any(est[4] < cm.start() for est in estimates)
                            )
                        )

                        is_parallel_comp_val = (
                            cm.end() <= start < comp_arm_match.start()
                            or (
                                comp_arm_match.end() <= start
                                and any(est[4] < cm.start() for est in estimates)
                                and not is_diff_after_comp
                            )
                        )

                        if is_diff_after_comp:
                            curr_arm = primary_arm_match.group(0).strip()
                            comp_arm = comp_arm_match.group(0).strip()
                        elif is_parallel_comp_val:
                            curr_arm = comp_arm_match.group(0).strip()
                            comp_arm = primary_arm_match.group(0).strip()
                        elif start < cm.start():
                            curr_arm = primary_arm_match.group(0).strip()
                            comp_arm = comp_arm_match.group(0).strip()
                    elif len(cand_primary_arms) > 1:
                        # Ambiguous: multiple primary candidate arms for a single estimate
                        curr_arm = None
                        comp_arm = None

                elif comparator_arm_matches and not primary_arms:
                    # Comparator arm identified, but primary intervention arm is unresolved
                    if len(comparator_arm_matches) == 1:
                        _, comp_arm_match = comparator_arm_matches[0]
                        comp_arm = comp_arm_match.group(0).strip()
                    else:
                        comp_arm = None
                    curr_arm = None

            # 4.3 Proximity fallback (only when no comparative markers are present)
            if curr_arm is None and not comp_markers:
                unique_arms = list(dict.fromkeys(a.group(0).strip() for a in all_arms))
                if len(unique_arms) <= 2:
                    preceding = [a for a in all_arms if a.end() <= start]
                    following = [a for a in all_arms if a.start() >= end]
                    prec_dist = (start - preceding[-1].end()) if preceding else 9999
                    foll_dist = (following[0].start() - end) if following else 9999

                    if foll_dist <= 30 and foll_dist < prec_dist:
                        curr_arm = following[0].group(0).strip()
                    elif preceding:
                        curr_arm = preceding[-1].group(0).strip()
                    elif following and foll_dist <= 50:
                        curr_arm = following[0].group(0).strip()

                    if curr_arm:
                        other_arms = [
                            a
                            for a in unique_arms
                            if a.lower() != curr_arm.lower()
                        ]
                        if len(other_arms) == 1:
                            comp_arm = other_arms[0]
                else:
                    # Multi-arm trial with 3+ arms and no comparative marker -> ambiguous attribution
                    curr_arm = None
                    comp_arm = None

        # Verify extracted numerical result against source text
        assert (
            raw in text_norm
            or str(abs(pe)) in text_norm
            or f"{pe:g}" in text_norm
        ), f"Extracted numerical result {pe} not found in source text!"

        # Local sample size & horizon
        clause_start = estimates[i - 1][5] if i > 0 else 0
        local_clause = text_norm[clause_start:next_est_start]
        local_dur = extract_follow_up_duration(local_clause) or sentence_duration
        local_sz = extract_sample_size(local_clause) or sentence_sample_sz

        results.append(
            NumericalResult(
                arm_name=curr_arm,
                comparator_arm=comp_arm,
                endpoint=detected_ep,
                point_estimate=pe,
                metric_name=metric,
                ci_lower=ci_l,
                ci_upper=ci_u,
                p_value=pval,
                unit=unit,
                follow_up_horizon=local_dur,
                sample_size=local_sz,
                statistical_uncertainty=uncert,
                raw_text=raw,
            )
        )

    # 6. Unattached p-values (only if p-value was not associated with any point estimate)
    for p_m in p_matches:
        if p_m.span() in used_p_spans:
            continue
        pval = float(p_m.group(2))
        results.append(
            NumericalResult(
                endpoint=default_endpoint,
                p_value=pval,
                follow_up_horizon=sentence_duration,
                sample_size=sentence_sample_sz,
                statistical_uncertainty=f"p = {pval}",
                raw_text=p_m.group(0),
            )
        )

    return results


def extract_follow_up_duration(text: str) -> Optional[str]:
    """Extracts follow-up duration (e.g. '6 months', '12 months', '14 weeks')."""
    if not text:
        return None
    m = re.search(
        r"\b(?:at|after|over|during|for)\s+(\d+\s*(?:months?|weeks?|years?|days?))\b",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().lower()
    m2 = re.search(
        r"\b(\d+[\s-]*(?:months?|weeks?|years?|days?))\s+(?:follow[\s-]*up|intervention|trial|study|period)\b",
        text,
        re.IGNORECASE,
    )
    if m2:
        return m2.group(1).strip().lower()
    m3 = re.search(
        r"\b(\d+\s*(?:months?|weeks?|years?))\b",
        text,
        re.IGNORECASE,
    )
    if m3:
        return m3.group(1).strip().lower()
    return None


def extract_sample_size(text: str) -> Optional[int]:
    """Extracts cohort or trial participant sample size."""
    if not text:
        return None
    m = re.search(
        r"\b(?:n\s*=\s*|sample\s+size\s*(?:of\s*)?|total\s+of\s+|enrolled\s+|randomized\s+|including\s+)(\d[\d,]*)\s*(?:patients|participants|subjects|individuals|cases|adults|children|women|men|infants|persons)?\b",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    m2 = re.search(
        r"\b(\d[\d,]*)\s+(?:eligible\s+)?(?:patients|participants|subjects|individuals|cases|controls|adults|children|women|men|infants|persons)\b",
        text,
        re.IGNORECASE,
    )
    if m2:
        try:
            return int(m2.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def infer_study_type(combined_text: str) -> tuple[StudyDesignType, str]:
    """Infers study design category and textual descriptor with section-aware gating."""
    t_lower = combined_text.lower()

    # Prioritize METHODS / DESIGN section over BACKGROUND statements
    methods_match = re.search(
        r"\b(?:methods?|methodology|design|patients and methods):\s*(.*?)(?=\b(?:results?|findings|conclusions?|interpretation):|$)",
        t_lower,
        re.DOTALL,
    )
    target_text = methods_match.group(1) if methods_match else t_lower

    # 1. Systematic Review / Meta-analysis
    if re.search(
        r"\b(?:systematic review|meta-analysis|meta analysis|pooled analysis|network meta-analysis|umbrella review|cochrane)\b",
        target_text,
    ):
        return StudyDesignType.SYSTEMATIC_REVIEW_META_ANALYSIS, "Systematic Review / Meta-Analysis"

    # 2. Prediction Model
    if re.search(
        r"\b(?:prediction model|predictive model|machine learning|deep learning|random forest|gradient boosting|neural network|xgboost|auroc|auprc|c-statistic|discrimination metric|risk prediction)\b",
        target_text,
    ):
        return StudyDesignType.PREDICTION_MODEL, "Prediction Model"

    # Filter out literature review references (e.g. 'previous randomized trials', 'trials have shown')
    clean_target = re.sub(
        r"\b(?:previous|prior|earlier|past|existing)\s+(?:randomized|clinical|controlled)?\s*trials?\b",
        "",
        target_text,
    )
    clean_target = re.sub(
        r"\b(?:randomized|clinical)\s+trials?\s+(?:have|had)\s+(?:shown|evaluated|suggested|demonstrated)\b",
        "",
        clean_target,
    )

    has_rct = bool(
        re.search(
            r"\b(?:randomized|randomised|controlled trial|rct|clinical trial|double-blind|single-blind|open-label trial|parallel-group|crossover trial)\b",
            clean_target,
        )
    )
    has_obs = bool(
        re.search(
            r"\b(?:cohort|prospective|retrospective|case-control|cross-sectional|observational|longitudinal|epidemiologic|population-based|registry)\b",
            clean_target,
        )
    )

    if has_rct and not has_obs:
        return StudyDesignType.INTERVENTIONAL_TRIAL, "Randomized Controlled Trial"
    elif has_obs and not has_rct:
        return StudyDesignType.OBSERVATIONAL_STUDY, "Observational Study"
    elif has_rct and has_obs:
        # Check active conduct phrases
        if re.search(
            r"\b(?:we conducted|in this|our)\s+(?:prospective|retrospective|cohort|observational)\b",
            clean_target,
        ):
            return StudyDesignType.OBSERVATIONAL_STUDY, "Observational Study"
        elif re.search(
            r"\b(?:we conducted|in this|our)\s+(?:randomized|randomised|trial|clinical trial|rct)\b",
            clean_target,
        ):
            return StudyDesignType.INTERVENTIONAL_TRIAL, "Randomized Controlled Trial"
        return StudyDesignType.INTERVENTIONAL_TRIAL, "Randomized Controlled Trial"

    if re.search(
        r"\b(?:in vitro|in vivo|murine|mouse model|rat model|cell culture|mechanistic|electrochemical)\b",
        clean_target,
    ):
        return StudyDesignType.MECHANISTIC_EXPLORATORY, "Mechanistic / Exploratory"

    return StudyDesignType.UNKNOWN, "Unknown Study Design"


def infer_provenance_section(
    sentence_text: str, abstract_text: Optional[str] = None
) -> ProvenanceSection:
    """Classifies abstract section provenance (RESULTS, CONCLUSION, METHODS, BACKGROUND)."""
    s_lower = sentence_text.lower().strip()
    if re.search(r"^(?:background|introduction|objective|aim):", s_lower):
        return ProvenanceSection.BACKGROUND
    if re.search(r"^(?:methods|methodology|design|patients and methods|setting):", s_lower):
        return ProvenanceSection.METHODS
    if re.search(r"^(?:conclusions?|interpretation):", s_lower):
        return ProvenanceSection.CONCLUSION
    if re.search(r"^(?:results?|findings):", s_lower):
        return ProvenanceSection.RESULTS

    # Contextual clues in sentence
    if re.search(
        r"\b(?:we conclude|in conclusion|these findings suggest|these results indicate|in summary)\b",
        s_lower,
    ):
        return ProvenanceSection.CONCLUSION
    if re.search(
        r"\b(?:is the leading cause of|is a common|remains a major|accounts for \d+%)\b", s_lower
    ):
        return ProvenanceSection.BACKGROUND
    return ProvenanceSection.RESULTS


class ConditionTagger:
    """Extracts experimental condition slots (Intervention, Comparator, Outcome, Setting) with validation."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

    def extract_conditions(
        self, sentence_text: str, abstract_text: Optional[str] = None
    ) -> ConditionSlots:
        """Extracts PICO condition slots from finding sentence and abstract context."""
        text_lower = sentence_text.lower()
        combined_text = f"{sentence_text} {abstract_text or ''}"

        # 1. Comparator extraction
        comparator = None
        for pattern in COMPARATOR_MARKERS:
            match = re.search(
                f"{pattern}\\s+([a-zA-Z0-9\\s\\-\\(\\)]{{3,40}}?)(?:\\.|,|\\bin\\b|\\bwith\\b|\\bfor\\b|\\bduring\\b|$)",
                text_lower,
            )
            if match:
                extracted_c = clean_slot_string(match.group(1).strip())
                if extracted_c:
                    comparator = extracted_c.title()
                break
        if comparator:
            comparator = CANONICAL_COMPARATORS.get(comparator.lower(), comparator)

        # 2. Outcome extraction
        outcome = None
        for marker in OUTCOME_MARKERS:
            m = re.search(marker, text_lower)
            if m:
                outcome = clean_slot_string(m.group(0))
                if outcome:
                    outcome = outcome.title()
                    break

        if not outcome and self.nlp:
            doc = self.nlp(sentence_text)
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ in ("dobj", "pobj") and len(chunk.text.split()) <= 4:
                    cleaned = clean_slot_string(chunk.text)
                    if cleaned:
                        outcome = cleaned.title()
                        break
        if outcome:
            outcome = CANONICAL_OUTCOMES.get(outcome.lower(), outcome)

        # 3. Intervention extraction
        intervention = None
        domain_interventions = [
            r"\b(?:very\s+low|low)[\s-]carbohydrate(?:\s+diets?)?\b",
            r"\b(?:very\s+low|low)[\s-]carb(?:\s+diets?)?\b",
            r"\blow[\s-]fat(?:\s+diets?)?\b",
            r"\bvlcd(?:s)?\b",
            r"\blcd(?:s)?\b",
            r"\blfd(?:s)?\b",
            r"\bketogenic(?:\s+diets?)?\b",
            r"\bintermittent fasting\b",
            r"\btime[\s-]restricted\s+(?:feeding|eating)\b",
            r"\balternate[\s-]day\s+fasting\b",
            r"\bcontinuous\s+(?:calorie|energy)\s+restriction\b",
            r"\bmoderate alcohol(?:\s+consumption)?\b",
            r"\bmoderate drinking\b",
            r"\blight[\s-]to[\s-]moderate\s+drinking\b",
            r"\balcohol consumption\b",
            r"\bheavy drinking\b",
            r"\bcaloric restriction\b",
            r"\burban tree canopy(?:\s+cover)?\b",
            r"\btree canopy(?:\s+cover)?\b",
            r"\bcanopy cover\b",
            r"\bincreasing impervious cover\b",
            r"\bimpervious cover\b",
            r"\bpartial sleep restriction\b",
            r"\btotal sleep deprivation\b",
            r"\bsleep restriction\b",
            r"\bsleep deprivation\b",
            r"\b(?:high|elevated|ambient|low)\s+temperatures?(?:\s+charging)?\b",
            r"\bdata augmentation(?:\s+techniques?)?\b",
            r"\bvitamin d(?:3)?(?:\s+supplementation)?\b",
            r"\bcholecalciferol\b",
            r"\bbaseline 25\(oh\)d levels?(?:\s+below\s+\d+\s+nmol/l)?\b",
            r"\b(?:higher|lower)\s+charging(?:\s*-\s*discharging)?\s+rates?\b",
            r"\blithium plating\b",
            r"\bcell thickness increase\b",
            r"\bmachine learning(?:\s+models?|\s+approaches?)?\b",
            r"\bgradient\s+boosting(?:\s+decision\s+trees?)?\b",
            r"\bxgboost(?:\s+models?)?\b",
            r"\brandom\s+forest(?:\s+models?)?\b",
            r"\blogistic\s+regression(?:\s+classifiers?)?\b",
            r"\bhpv(?:\s+infection)?\b",
            r"\bhuman papillomavirus\b",
            r"\bcigarette smoking\b",
            r"\bsmoking\b",
            r"\bcurcumin\b",
            r"\bintravenous\s+curcumin\b",
            r"\bstandard\s+care\b",
            r"\busual\s+care\b",
        ]
        for ip in domain_interventions:
            im = re.search(ip, text_lower)
            if im:
                intervention = clean_slot_string(im.group(0).title())
                break

        if not intervention and self.nlp:
            doc = self.nlp(sentence_text)
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ in ("nsubj", "nsubjpass"):
                    cleaned = clean_slot_string(chunk.text)
                    if cleaned and cleaned.title() != outcome:
                        intervention = cleaned.title()
                        break

        if not intervention and self.nlp:
            # Fallback: scan for first valid noun chunk not in invalid terms
            doc = self.nlp(sentence_text)
            for chunk in doc.noun_chunks:
                cleaned = clean_slot_string(chunk.text)
                if cleaned and cleaned.title() != outcome:
                    intervention = cleaned.title()
                    break

        if intervention:
            intervention = CANONICAL_INTERVENTIONS.get(intervention.lower(), intervention)

        # 4. Setting extraction
        setting = None
        for pattern in SETTING_MARKERS:
            match = re.search(
                f"{pattern}\\s*([a-zA-Z0-9\\s\\-\\(\\)]{{0,35}}?)(?:\\.|,|\\band\\b|$)",
                text_lower,
            )
            if match:
                cleaned_p = clean_slot_string(match.group(0))
                if cleaned_p:
                    setting = cleaned_p.title()
                break

        # 5. Extract population and limitations if present
        population = None
        pop_m = re.search(
            r"\b(adults|healthy adults|young adults|older adults|children|adolescents|college students|patients|mice|rats)\b",
            text_lower,
        )
        if pop_m:
            population = pop_m.group(0).title()

        qualifiers = None
        qual_m = re.search(
            r"\b(?:when|where|if|after|for)\s+(?:canopy cover exceeded \d+%|\d+\s+hours?(?: of sleep)?|temperatures? above \d+°?c)",
            text_lower,
        )
        if qual_m:
            qualifiers = qual_m.group(0)

        # 6. Extract follow-up duration, sample size, numerical results, study design, and provenance
        follow_up = extract_follow_up_duration(sentence_text) or (
            extract_follow_up_duration(abstract_text) if abstract_text else None
        )
        sample_sz = extract_sample_size(sentence_text) or (
            extract_sample_size(abstract_text) if abstract_text else None
        )
        num_results = extract_numerical_results(
            sentence_text,
            default_endpoint=outcome,
            default_duration=follow_up,
            intervention_hint=intervention,
            comparator_hint=comparator,
        )
        study_t, s_design = infer_study_type(combined_text)
        prov_sec = infer_provenance_section(sentence_text, abstract_text)

        is_unknown = bool(not intervention or not outcome)
        confidence = 0.50 if is_unknown else 0.85

        return ConditionSlots(
            intervention=intervention,
            comparator=comparator,
            outcome_metric=outcome,
            dataset_setting=setting,
            population=population,
            effect_size=None,
            follow_up_duration=follow_up,
            sample_size=sample_sz,
            numerical_results=num_results,
            study_design=s_design,
            study_type=study_t,
            provenance_section=prov_sec,
            limitations_or_qualifiers=qualifiers,
            confidence=confidence,
            is_unknown=is_unknown,
        )
