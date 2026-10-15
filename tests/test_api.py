"""Unit and integration tests for Starlette backend API."""

from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from scientific_contradiction_finder.api import app as api_module
from scientific_contradiction_finder.api.app import app
from scientific_contradiction_finder.schemas import (
    ExecutiveSummary,
    PipelineExecutionState,
    QueryIntent,
    ResearchReport,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "providers" in data
    assert data["providers"]["openalex"] is True


def test_presets_endpoint(client):
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) > 5


def test_benchmark_pairs_endpoint(client):
    response = client.get("/api/benchmark/pairs")
    assert response.status_code == 200
    data = response.json()
    assert data["total_pairs"] > 0
    assert len(data["pairs"]) == data["total_pairs"]

    first_pair = data["pairs"][0]
    assert "finding_a" in first_pair
    assert "finding_b" in first_pair
    assert "ground_truth_relation" in first_pair


def test_benchmark_classify_endpoint(client):
    # Fetch a pair first
    pairs_res = client.get("/api/benchmark/pairs")
    first_pair = pairs_res.json()["pairs"][0]

    # Post to classify
    classify_res = client.post("/api/benchmark/classify", json={"pair": first_pair})
    assert classify_res.status_code == 200
    data = classify_res.json()
    assert "prediction" in data
    pred = data["prediction"]
    assert "predicted_relation" in pred
    assert "condition_differences" in pred
    assert "explanation" in pred


def test_analyze_empty_query_rejected(client):
    response = client.post("/api/analyze", json={"query": ""})
    assert response.status_code == 400
    assert "error" in response.json()


def test_analyze_invalid_json(client):
    response = client.post("/api/analyze", content="invalid json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400


def test_benchmark_classify_missing_pair(client):
    response = client.post("/api/benchmark/classify", json={})
    assert response.status_code == 400
    assert "Missing 'pair' object" in response.json().get("error", "")


def test_analyze_endpoint_mock_success(client, monkeypatch):
    mock_report = ResearchReport(
        query="Does vitamin D prevent cardiovascular disease?",
        run_id="run_test_123",
        query_intent=QueryIntent.CONTRADICTION_ANALYSIS,
        execution_state=PipelineExecutionState.COMPLETED,
        executive_summary=ExecutiveSummary(
            bottom_line="RCT evidence demonstrates no statistically significant benefit.",
            overall_evidence_assessment="broadly challenges",
            strongest_supporting_evidence="None identified.",
            strongest_conflicting_evidence="VITAL trial found no reduction in major CV events.",
            why_studies_disagree="Baseline nutritional sufficiency in trial cohorts.",
            confidence_and_limitations="High certainty from double-blind RCTs.",
            research_coverage="Screened 10 papers, triaged 5, extracted 3 findings.",
        ),
        candidate_papers_retrieved=10,
        papers_triaged=5,
        papers_deep_dived=3,
        full_texts_accessible=1,
        findings_extracted=3,
        comparisons_evaluated=2,
    )

    mock_run = AsyncMock(return_value={
        "query": "Does vitamin D prevent cardiovascular disease?",
        "run_id": "run_test_123",
        "query_intent": "CONTRADICTION_ANALYSIS",
        "execution_state": "COMPLETED",
        "papers_found": 10,
        "findings_extracted": 3,
        "comparisons": [],
        "findings": [],
        "diagnostics": {"raw_total": 10, "after_dedup": 8},
        "research_report": mock_report,
    })

    pipeline = api_module.get_pipeline()
    monkeypatch.setattr(pipeline, "run", mock_run)

    response = client.post(
        "/api/analyze",
        json={"query": "Does vitamin D prevent cardiovascular disease?", "limit": "invalid_limit_defaults_to_10"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Does vitamin D prevent cardiovascular disease?"
    assert data["papers_found"] == 10
    assert data["findings_extracted"] == 3
    assert "research_report" in data
    assert data["research_report"]["run_id"] == "run_test_123"
    assert data["research_report"]["executive_summary"]["bottom_line"] == "RCT evidence demonstrates no statistically significant benefit."


def test_analyze_endpoint_pipeline_failure(client, monkeypatch):
    mock_run = AsyncMock(side_effect=RuntimeError("Upstream scholarly provider timeout"))

    pipeline = api_module.get_pipeline()
    monkeypatch.setattr(pipeline, "run", mock_run)

    response = client.post("/api/analyze", json={"query": "Does vitamin D prevent cardiovascular disease?"})
    assert response.status_code == 500
    data = response.json()
    assert "Failed to complete research pipeline execution." in data.get("error", "")
    assert "Upstream scholarly provider timeout" in data.get("detail", "")

