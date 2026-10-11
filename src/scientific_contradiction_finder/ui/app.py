"""Streamlit Web Dashboard for Scientific Contradiction Finder."""

import asyncio

import streamlit as st

from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline
from scientific_contradiction_finder.schemas import RelationLabel, ResearchReport

st.set_page_config(
    page_title="Scientific Contradiction Finder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar settings
st.sidebar.header("⚙️ Configuration & Modes")
mode = st.sidebar.radio(
    "Select Workflow Mode:",
    ["Live Empirical Search (OpenAlex / arXiv)", "Demo Benchmark Pairs (CAS-Con150)"],
)

use_groq = st.sidebar.checkbox("Enable Groq Query Structuring (llama/qwen)", value=True)
use_groq_interp = st.sidebar.checkbox("Enable Groq Candidate Sentence Auditor", value=False)


# Initialize pipeline
@st.cache_resource
def get_pipeline(use_llm: bool):
    return ContradictionFinderPipeline(use_groq=use_llm)


pipeline = get_pipeline(use_groq)

st.title("🔬 Scientific Contradiction Finder")
st.markdown("""
**Condition-Aware Discovery & Analysis of Conflicting Scientific Findings**
Differentiates between direct contradictions, qualified disagreements, and incomparable settings by extracting empirical conditions ($I, C, O, S$).
""")

if mode == "Demo Benchmark Pairs (CAS-Con150)":
    st.subheader("📚 Benchmark Evaluation Explorer")
    loader = BenchmarkLoader()
    pairs = loader.load_cas_con150()

    category_filter = st.selectbox(
        "Filter by Ground-Truth Relationship:",
        [
            "ALL",
            "SUPPORTS_OR_CONSISTENT",
            "CONTRADICTS",
            "QUALIFIES_OR_PARTIAL_DISAGREEMENT",
            "INCOMPARABLE_CONDITIONS",
            "INSUFFICIENT_EVIDENCE",
        ],
    )

    filtered_pairs = (
        pairs
        if category_filter == "ALL"
        else [p for p in pairs if p.ground_truth_relation.value == category_filter]
    )
    st.info(f"Showing **{len(filtered_pairs)}** pairs for category `{category_filter}`.")

    selected_pair = st.selectbox(
        "Choose finding pair to inspect:",
        options=filtered_pairs,
        format_func=lambda p: (
            f"[{p.ground_truth_relation.value}] {p.pair_id}: {p.finding_a.finding_claim[:60]}... vs {p.finding_b.finding_claim[:60]}..."
        ),
    )

    if selected_pair:
        pred = pipeline.classifier.classify_pair(selected_pair)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📄 Finding A")
            st.markdown(f"**Title:** {selected_pair.finding_a.paper_title}")
            st.markdown(f"**Claim / Finding:** _{selected_pair.finding_a.finding_claim}_")
            st.markdown(f"**Reported Direction:** `{selected_pair.finding_a.direction.value}`")
            with st.expander("📌 Extracted Conditions (Study A)", expanded=True):
                st.write(
                    f"- **Intervention ($I$):** {selected_pair.finding_a.conditions.intervention or '_Unstated_'}"
                )
                st.write(
                    f"- **Comparator ($C$):** {selected_pair.finding_a.conditions.comparator or '_Unstated / Standard Baseline_'}"
                )
                st.write(
                    f"- **Outcome Metric ($O$):** {selected_pair.finding_a.conditions.outcome_metric or '_Unstated_'}"
                )
                st.write(
                    f"- **Population:** {selected_pair.finding_a.conditions.population or '_Not specified_'}"
                )
                st.write(
                    f"- **Setting / Cohort ($S$):** {selected_pair.finding_a.conditions.dataset_setting or '_General scientific setting_'}"
                )
                if selected_pair.finding_a.conditions.limitations_or_qualifiers:
                    st.write(
                        f"- **Conditions / Qualifiers:** {selected_pair.finding_a.conditions.limitations_or_qualifiers}"
                    )
                if selected_pair.finding_a.conditions.effect_size:
                    st.write(
                        f"- **Reported Effect:** {selected_pair.finding_a.conditions.effect_size}"
                    )

        with col2:
            st.markdown("### 📄 Finding B")
            st.markdown(f"**Title:** {selected_pair.finding_b.paper_title}")
            st.markdown(f"**Claim / Finding:** _{selected_pair.finding_b.finding_claim}_")
            st.markdown(f"**Reported Direction:** `{selected_pair.finding_b.direction.value}`")
            with st.expander("📌 Extracted Conditions (Study B)", expanded=True):
                st.write(
                    f"- **Intervention ($I$):** {selected_pair.finding_b.conditions.intervention or '_Unstated_'}"
                )
                st.write(
                    f"- **Comparator ($C$):** {selected_pair.finding_b.conditions.comparator or '_Unstated / Standard Baseline_'}"
                )
                st.write(
                    f"- **Outcome Metric ($O$):** {selected_pair.finding_b.conditions.outcome_metric or '_Unstated_'}"
                )
                st.write(
                    f"- **Population:** {selected_pair.finding_b.conditions.population or '_Not specified_'}"
                )
                st.write(
                    f"- **Setting / Cohort ($S$):** {selected_pair.finding_b.conditions.dataset_setting or '_General scientific setting_'}"
                )
                if selected_pair.finding_b.conditions.limitations_or_qualifiers:
                    st.write(
                        f"- **Conditions / Qualifiers:** {selected_pair.finding_b.conditions.limitations_or_qualifiers}"
                    )
                if selected_pair.finding_b.conditions.effect_size:
                    st.write(
                        f"- **Reported Effect:** {selected_pair.finding_b.conditions.effect_size}"
                    )

        st.markdown("---")
        st.markdown("### 🧠 Condition-Aware Reasoning & Verdict")

        verdict_color = {
            RelationLabel.SUPPORTS_OR_CONSISTENT: "green",
            RelationLabel.CONTRADICTS: "red",
            RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT: "orange",
            RelationLabel.INCOMPARABLE_CONDITIONS: "blue",
            RelationLabel.INSUFFICIENT_EVIDENCE: "gray",
        }.get(pred.predicted_relation, "blue")

        support_score = getattr(pred, "heuristic_score", pred.confidence_score)
        st.markdown(
            f"<h4>Predicted Relationship: <span style='color:{verdict_color};'>{pred.predicted_relation.value}</span> "
            f"(Comparison Support Score: {support_score * 100:.1f}%)</h4>",
            unsafe_allow_html=True,
        )
        st.caption(
            "ℹ️ *Comparison Support Score represents heuristic evidence certainty and condition alignment (0–100%). It is not an uncalibrated posterior probability.*"
        )

        st.markdown(f"**Reasoning Explanation:** {pred.explanation}")

        # Provenance verification badge
        if pred.provenance_verified:
            st.success(
                "✅ **Provenance Verified:** Verbatim evidence substrings matched source abstracts."
            )
        else:
            st.warning("⚠️ Provenance warning: Evidence substring could not be strictly resolved.")

        with st.expander("🔍 Condition Slot Differences & Metric Breakdown", expanded=True):
            diff = pred.condition_differences
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Intervention Sim ($I$)",
                f"{diff.intervention_similarity:.2f}",
                "Match" if diff.intervention_match else "Diff",
            )
            c2.metric(
                "Comparator Sim ($C$)",
                f"{diff.comparator_similarity:.2f}",
                "Match" if diff.comparator_match else "Diff",
            )
            c3.metric(
                "Outcome Sim ($O$)",
                f"{diff.outcome_similarity:.2f}",
                "Match" if diff.outcome_match else "Diff",
            )
            c4.metric(
                "Setting Sim ($S$)",
                f"{diff.setting_similarity:.2f}",
                "Match" if diff.setting_match else "Diff",
            )

            st.write(f"**Differences:** {diff.summary_of_differences}")
            st.write(f"**Overall Compatibility:** `{diff.overall_compatibility:.2f}`")

elif mode == "Live Empirical Search (OpenAlex / arXiv)":
    st.subheader("🌐 Search Scientific Literature")

    preset = st.selectbox(
        "Choose an empirical query preset or write your own below:",
        [
            "Custom Query",
            "Do randomized controlled trials show that low-carbohydrate diets produce greater weight loss than low-fat diets, and does the difference persist over time?",
            "Does moderate alcohol consumption reduce all-cause mortality, or can the apparent protective association be explained by study design and confounding?",
            "Does intermittent fasting produce greater long-term weight loss than continuous calorie restriction when energy intake is comparable?",
            "Does vitamin D supplementation prevent cardiovascular events in generally healthy adults, and how do randomized trials compare with observational findings?",
            "Which machine-learning approaches have been evaluated for early prediction of pediatric sepsis, and how reliable is the evidence that one approach outperforms another?",
            "Does high-dose intravenous curcumin improve survival in severe acute pancreatitis?",
            "Does increasing urban tree canopy cover reduce local temperatures, and does the effect vary with city density and climate?",
            "Does sleep restriction impair working memory in adults, and do the effects differ between partial sleep restriction and total sleep deprivation?",
            "Do lithium-ion batteries degrade faster when charged at high temperatures compared with moderate temperatures?",
            "Does image augmentation improve image classification accuracy in vision transformer models compared with training without augmentation?",
            "Does vitamin D supplementation reduce the risk of acute respiratory infections, and does the effect depend on baseline vitamin D levels?",
        ],
    )

    default_val = (
        "Do randomized controlled trials show that low-carbohydrate diets produce greater weight loss than low-fat diets, and does the difference persist over time?"
        if preset == "Custom Query"
        else preset
    )
    query = st.text_input("Enter Empirical Claim or Scientific Question:", value=default_val)

    # Session state initialization for query isolation
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = None
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = None

    col_btn, _ = st.columns([1, 4])
    submit = col_btn.button("Analyze Conflict", type="primary")

    if submit and query:
        with st.spinner(
            "Searching literature across OpenAlex & arXiv, extracting findings, and evaluating conditions..."
        ):
            res = asyncio.run(
                pipeline.run(query, limit=10, use_groq_interpretation=use_groq_interp)
            )
            st.session_state["current_query"] = query
            st.session_state["search_results"] = res

    # Check for query text modification to prevent displaying stale results
    if st.session_state.get("search_results") is not None:
        if query != st.session_state.get("current_query"):
            st.info(
                "ℹ️ Query text was changed. Stale results are hidden. Click **Analyze Conflict** to run analysis for the updated query."
            )
        else:
            res = st.session_state["search_results"]
            report: ResearchReport = res.get("research_report")
            if report and report.executive_summary:
                es = report.executive_summary
                st.markdown("## 📋 Executive Research Summary")

            # Status and Intent Badges
            exec_state_color = "green" if report.execution_state.value == "COMPLETED" else "orange"
            st.markdown(
                f"<span style='background-color:#4A5568; color:white; padding:3px 10px; border-radius:6px; font-size:0.85em; font-weight:bold;'>Intent: {report.query_intent.value}</span> "
                f"<span style='background-color:{exec_state_color}; color:white; padding:3px 10px; border-radius:6px; font-size:0.85em; font-weight:bold;'>Pipeline State: {report.execution_state.value}</span> "
                f"<span style='color:#718096; font-size:0.85em; margin-left:10px;'>Run ID: <code>{report.run_id or 'N/A'}</code></span>",
                unsafe_allow_html=True,
            )
            st.write("")

            # 1. Bottom line
            st.markdown(f"### 💡 Bottom Line\n**{es.bottom_line}**")

            # Evidence assessment badge
            assessment_color = (
                "green"
                if "support" in es.overall_evidence_assessment.lower()
                else (
                    "orange"
                    if "mixed" in es.overall_evidence_assessment.lower()
                    or "context" in es.overall_evidence_assessment.lower()
                    else "blue"
                )
            )
            st.markdown(
                f"**Overall Evidence Assessment:** <span style='color:{assessment_color}; font-weight:bold;'>{es.overall_evidence_assessment.upper()}</span>",
                unsafe_allow_html=True,
            )

            # Subclaim Assessment Matrix
            subclaims_to_show = es.subclaim_assessments or report.subclaim_assessments
            if subclaims_to_show:
                st.markdown("#### 🎯 Independent Subclaim Evidence Assessments")
                for sc in subclaims_to_show:
                    sc_color = (
                        "green"
                        if sc.status.value == "SUPPORTED"
                        else ("orange" if sc.status.value == "QUALIFIED" else "red")
                    )
                    sc_badge = f"<span style='background-color:{sc_color}; color:white; padding:2px 8px; border-radius:4px; font-size:0.8em; font-weight:bold;'>{sc.status.value}</span>"
                    st.markdown(
                        f"- **{sc.subclaim_text}** — {sc_badge}<br>&nbsp;&nbsp;_{sc.rationale}_",
                        unsafe_allow_html=True,
                    )
                st.write("")

            c_sum1, c_sum2 = st.columns(2)
            with c_sum1:
                st.markdown("#### 🟢 Strongest Supporting Evidence")
                st.info(es.strongest_supporting_evidence)
            with c_sum2:
                st.markdown("#### 🔴 Strongest Conflicting / Qualifying Evidence")
                st.warning(es.strongest_conflicting_evidence)

            with st.expander(
                "🔬 Why Studies Disagree (Moderator & Context Analysis)", expanded=True
            ):
                st.markdown(es.why_studies_disagree)
                if report.disagreements:
                    st.markdown("##### 📌 Specific Disagreement Cases & Verification")
                    for d in report.disagreements:
                        v_color = (
                            "green"
                            if d.verification_status.value == "SUPPORTED"
                            else (
                                "orange"
                                if d.verification_status.value == "PLAUSIBLE_UNVERIFIED"
                                else "gray"
                            )
                        )
                        v_badge = f"<span style='color:{v_color}; font-weight:bold;'>[{d.verification_status.value}]</span>"
                        st.markdown(
                            f"- **{d.topic_or_metric}:** {', '.join(d.studies_in_disagreement)} {v_badge}",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Reason: {d.plausible_moderator_or_reason}")
                        if d.supporting_evidence_ids:
                            st.caption(
                                f"Ledger Evidence IDs: `{', '.join(d.supporting_evidence_ids)}`"
                            )
                        if d.suggested_follow_up_investigation:
                            st.caption(f"Follow-up: _{d.suggested_follow_up_investigation}_")

            c_lim1, c_lim2 = st.columns(2)
            with c_lim1:
                st.markdown("#### ⚠️ Evidence Robustness & Limitations")
                st.write(es.confidence_and_limitations)
            with c_lim2:
                st.markdown("#### 📊 Research Coverage")
                st.write(es.research_coverage)
                st.caption(f"Status: {report.stopped_reason}")

            st.divider()

        # Display Structured Evidence Ledger
        if report and report.evidence_ledger:
            st.markdown(
                f"### 📑 Structured Evidence Ledger ({len(report.evidence_ledger)} Verified Findings)"
            )
            for claim in report.evidence_ledger:
                badge = "📖 [Full-Text]" if claim.full_text_used else "📄 [Abstract]"
                design_color = {
                    "INTERVENTIONAL_TRIAL": "#2E7D32",
                    "OBSERVATIONAL_STUDY": "#1565C0",
                    "SYSTEMATIC_REVIEW_META_ANALYSIS": "#6A1B9A",
                    "PREDICTION_MODEL": "#E65100",
                    "UNKNOWN": "#616161",
                }.get(claim.study_type.value, "#616161")
                design_badge = f"<span style='background-color:{design_color}; color:white; padding:2px 8px; border-radius:4px; font-size:0.8em; font-weight:bold;'>{claim.study_type.value}</span>"

                title_link = f"[{claim.paper_title}]({claim.url or claim.doi or '#'})"
                st.markdown(
                    f"**{badge} {design_badge} [{claim.paper_id}] {title_link}**",
                    unsafe_allow_html=True,
                )
                st.markdown(f'> *"{claim.exact_source_passage}"*')
                c1, c2, c3 = st.columns(3)
                c1.caption(f"**Intervention / Model ($I$):** {claim.intervention_exposure or '_Unstated_'}")
                c2.caption(f"**Outcome / Target ($O$):** {claim.outcome or '_Unstated_'}")
                c3.caption(f"**Direction:** `{claim.direction.value}`")

                c4, c5 = st.columns(2)
                if claim.population_setting:
                    c4.caption(f"**Population / Setting ($S$):** {claim.population_setting}")
                if claim.follow_up_duration:
                    c5.caption(f"**Follow-Up Duration:** {claim.follow_up_duration}")
                if claim.comparator:
                    st.caption(f"**Comparator ($C$):** {claim.comparator}")
                if claim.conditions_and_moderators:
                    st.caption(f"**Conditions/Moderators:** {claim.conditions_and_moderators}")

                if claim.numerical_results:
                    with st.expander(
                        f"📊 Attributed Multi-Arm Numerical Results ({len(claim.numerical_results)} Estimates)",
                        expanded=False,
                    ):
                        for nr in claim.numerical_results:
                            arm_txt = f"**Arm:** `{nr.arm_name}`" if nr.arm_name else "**Estimate**"
                            comp_txt = f" vs `{nr.comparator_arm}`" if nr.comparator_arm else ""
                            val_txt = f": **{nr.point_estimate} {nr.unit or ''}**" if nr.point_estimate is not None else ""
                            uncertainty_txt = f" ({nr.statistical_uncertainty})" if nr.statistical_uncertainty else ""
                            horizon_txt = f" at {nr.follow_up_horizon}" if nr.follow_up_horizon else ""
                            n_txt = f" [n={nr.sample_size}]" if nr.sample_size else ""
                            st.write(f"- {arm_txt}{comp_txt}{val_txt}{uncertainty_txt}{horizon_txt}{n_txt}")

                if claim.prediction_model_details:
                    pm = claim.prediction_model_details
                    with st.expander("🤖 Prediction Model Architecture & Metrics", expanded=False):
                        st.write(f"- **Method:** {pm.model_method or 'N/A'}")
                        st.write(f"- **Target Condition:** {pm.target_condition or 'N/A'}")
                        st.write(f"- **Horizon:** {pm.prediction_horizon or 'N/A'}")
                        st.write(f"- **Metrics:** {pm.metrics_and_values}")
                st.divider()

        # Literature Triage Section
        if report and report.triage_records:
            with st.expander(
                f"🔍 Literature Triage & Paper Selection ({len(report.triage_records)} Candidates)",
                expanded=False,
            ):
                for tr in report.triage_records:
                    icon = "✅ Deep-Dive" if tr.selected_for_deep_dive else "➖ Screened"
                    fallback_tag = " `[Heuristic Fallback]`" if tr.is_heuristic_fallback else ""
                    st.markdown(f"- **{icon} [{tr.category.value}]{fallback_tag} {tr.title}**")
                    st.caption(f"Rationale: {tr.relevance_rationale}")

        # Show pipeline diagnostics
        diag = res.get("diagnostics", {})
        oa_diag = diag.get("openalex", {})
        ax_diag = diag.get("arxiv", {})

        with st.expander("🛠️ Retrieval Diagnostics & Search Queries", expanded=False):
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            col_d1.metric(
                "OpenAlex Papers",
                f"{oa_diag.get('results_count', 0)}",
                f"Status {oa_diag.get('status') or 'N/A'}",
            )
            col_d2.metric(
                "arXiv Papers",
                f"{ax_diag.get('results_count', 0)}",
                f"Status {ax_diag.get('status') or 'N/A'}",
            )
            col_d3.metric("Usable Abstracts", f"{diag.get('usable_abstracts', 0)}")
            col_d4.metric("Extracted Findings", f"{res['findings_extracted']}")

            if diag.get("groq_assisted") and diag.get("groq_structured"):
                st.info(
                    f"💡 **Groq-Assisted Structured Query:** Keywords used: `{diag['groq_structured']['search_keywords']}`"
                )

        # Display cross-paper finding comparisons
        if res["comparisons"]:
            st.markdown("### ⚖️ Detailed Cross-Paper Finding Comparisons")
            for pair, pred in res["comparisons"][:10]:
                v_color = {
                    RelationLabel.SUPPORTS_OR_CONSISTENT: "green",
                    RelationLabel.CONTRADICTS: "red",
                    RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT: "orange",
                    RelationLabel.INCOMPARABLE_CONDITIONS: "blue",
                    RelationLabel.INSUFFICIENT_EVIDENCE: "gray",
                }.get(pred.predicted_relation, "blue")

                with st.expander(
                    f"[{pred.predicted_relation.value}] {pair.finding_a.paper_title[:45]} vs {pair.finding_b.paper_title[:45]}",
                    expanded=False,
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(
                            f"**Study A:** [{pair.finding_a.paper_title}]({pair.finding_a.paper_url or '#'})"
                        )
                        st.markdown(f"_{pair.finding_a.finding_claim}_")
                        st.caption(
                            f"Direction: `{pair.finding_a.direction.value}` | $I$: {pair.finding_a.conditions.intervention} | $C$: {pair.finding_a.conditions.comparator} | $O$: {pair.finding_a.conditions.outcome_metric}"
                        )
                    with c2:
                        st.markdown(
                            f"**Study B:** [{pair.finding_b.paper_title}]({pair.finding_b.paper_url or '#'})"
                        )
                        st.markdown(f"_{pair.finding_b.finding_claim}_")
                        st.caption(
                            f"Direction: `{pair.finding_b.direction.value}` | $I$: {pair.finding_b.conditions.intervention} | $C$: {pair.finding_b.conditions.comparator} | $O$: {pair.finding_b.conditions.outcome_metric}"
                        )

                    is_incomparable = pred.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS
                    score_label = "Incomparability Confidence" if is_incomparable else "Comparison Support Score"
                    score_val = pred.confidence_score if is_incomparable else getattr(pred, "heuristic_score", pred.confidence_score)

                    st.markdown(
                        f"**Verdict:** <span style='color:{v_color}; font-weight:bold;'>{pred.predicted_relation.value}</span> "
                        f"({score_label}: {score_val * 100:.1f}%)",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Explanation:** {pred.explanation}")
                    st.write(
                        f"**Condition Differences:** {pred.condition_differences.summary_of_differences}"
                    )
                    st.write(
                        f"**Overall Compatibility:** `{pred.condition_differences.overall_compatibility:.2f}`"
                    )
        elif res["papers_found"] > 0:
            st.info(
                "Retrieved papers did not contain multiple distinct empirical finding statements meeting comparison criteria."
            )
        else:
            st.warning(
                "⚠️ No relevant scholarly publications were retrieved matching this query from the configured providers."
            )

