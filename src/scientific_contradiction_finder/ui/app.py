"""Streamlit Web Dashboard for Scientific Contradiction Finder."""

import asyncio

import streamlit as st

from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline
from scientific_contradiction_finder.schemas import RelationLabel

st.set_page_config(
    page_title="Scientific Contradiction Finder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return ContradictionFinderPipeline()


pipeline = get_pipeline()

st.title("🔬 Scientific Contradiction Finder")
st.markdown("""
**Condition-Aware Discovery & Analysis of Conflicting Scientific Findings**
Differentiates between direct contradictions, qualified disagreements, and incomparable settings by extracting empirical conditions ($I, C, O, S$).
""")

# Sidebar
st.sidebar.header("⚙️ Configuration & Demo Modes")
mode = st.sidebar.radio(
    "Select Workflow Mode:",
    ["Live Empirical Search (OpenAlex / arXiv)", "Demo Benchmark Pairs (CAS-Con150)"],
)

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
                    f"- **Intervention ($I$):** {selected_pair.finding_a.conditions.intervention}"
                )
                st.write(f"- **Comparator ($C$):** {selected_pair.finding_a.conditions.comparator}")
                st.write(
                    f"- **Outcome Metric ($O$):** {selected_pair.finding_a.conditions.outcome_metric}"
                )
                st.write(
                    f"- **Setting / Cohort ($S$):** {selected_pair.finding_a.conditions.dataset_setting}"
                )

        with col2:
            st.markdown("### 📄 Finding B")
            st.markdown(f"**Title:** {selected_pair.finding_b.paper_title}")
            st.markdown(f"**Claim / Finding:** _{selected_pair.finding_b.finding_claim}_")
            st.markdown(f"**Reported Direction:** `{selected_pair.finding_b.direction.value}`")
            with st.expander("📌 Extracted Conditions (Study B)", expanded=True):
                st.write(
                    f"- **Intervention ($I$):** {selected_pair.finding_b.conditions.intervention}"
                )
                st.write(f"- **Comparator ($C$):** {selected_pair.finding_b.conditions.comparator}")
                st.write(
                    f"- **Outcome Metric ($O$):** {selected_pair.finding_b.conditions.outcome_metric}"
                )
                st.write(
                    f"- **Setting / Cohort ($S$):** {selected_pair.finding_b.conditions.dataset_setting}"
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

        st.markdown(
            f"<h4>Predicted Relationship: <span style='color:{verdict_color};'>{pred.predicted_relation.value}</span> "
            f"(Confidence: {pred.confidence_score * 100:.1f}%)</h4>",
            unsafe_allow_html=True,
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
            "Do higher temperatures accelerate lithium-ion battery degradation?",
            "Does data augmentation improve the generalization of vision transformers?",
            "Does vitamin D supplementation reduce respiratory infections in adults?",
        ],
    )

    default_val = (
        "Do higher temperatures accelerate lithium-ion battery degradation?"
        if preset == "Custom Query"
        else preset
    )
    query = st.text_input("Enter Empirical Claim or Scientific Question:", value=default_val)

    col_btn, _ = st.columns([1, 4])
    submit = col_btn.button("Analyze Conflict", type="primary")

    if submit and query:
        with st.spinner(
            "Searching literature across OpenAlex & arXiv, extracting findings, and evaluating conditions..."
        ):
            res = asyncio.run(pipeline.run(query, limit=10))

        # Show pipeline diagnostics
        diag = res.get("diagnostics", {})
        oa_diag = diag.get("openalex", {})
        ax_diag = diag.get("arxiv", {})

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

        if res["papers_found"] > 0:
            st.success(
                f"Retrieved **{res['papers_found']}** deduplicated papers; extracted **{res['findings_extracted']}** empirical findings."
            )
        else:
            st.warning("No papers with abstracts could be retrieved for this query.")

        # Display extracted findings
        if res["findings"]:
            with st.expander(
                f"📑 Extracted Evidence Findings ({len(res['findings'])})", expanded=True
            ):
                for f in res["findings"]:
                    st.markdown(f"**[{f.paper_id}] {f.paper_title}**")
                    st.markdown(f"> _{f.evidence.text}_")
                    st.caption(
                        f"Direction: `{f.direction.value}` | Intervention: `{f.conditions.intervention}` | Outcome: `{f.conditions.outcome_metric}` | Setting: `{f.conditions.dataset_setting}`"
                    )
                    st.divider()

        # Display cross-paper finding comparisons
        if res["comparisons"]:
            st.markdown("### ⚖️ Cross-Paper Finding Comparisons")
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
                    expanded=True,
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(
                            f"**Study A:** [{pair.finding_a.paper_title}]({pair.finding_a.paper_url or '#'})"
                        )
                        st.markdown(f"_{pair.finding_a.finding_claim}_")
                        st.caption(
                            f"Direction: `{pair.finding_a.direction.value}` | $I$: {pair.finding_a.conditions.intervention} | $O$: {pair.finding_a.conditions.outcome_metric}"
                        )
                    with c2:
                        st.markdown(
                            f"**Study B:** [{pair.finding_b.paper_title}]({pair.finding_b.paper_url or '#'})"
                        )
                        st.markdown(f"_{pair.finding_b.finding_claim}_")
                        st.caption(
                            f"Direction: `{pair.finding_b.direction.value}` | $I$: {pair.finding_b.conditions.intervention} | $O$: {pair.finding_b.conditions.outcome_metric}"
                        )

                    st.markdown(
                        f"**Verdict:** <span style='color:{v_color}; font-weight:bold;'>{pred.predicted_relation.value}</span> "
                        f"(Confidence: {pred.confidence_score * 100:.1f}%)",
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
