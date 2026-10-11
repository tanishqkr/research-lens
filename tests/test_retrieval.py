from scientific_contradiction_finder.retrieval.bm25_retriever import BM25Retriever
from scientific_contradiction_finder.retrieval.cache import SQLiteCache
from scientific_contradiction_finder.retrieval.deduplicator import DocumentDeduplicator
from scientific_contradiction_finder.retrieval.hybrid_reranker import ReciprocalRankFusion
from scientific_contradiction_finder.retrieval.openalex_client import decode_openalex_inverted_index
from scientific_contradiction_finder.retrieval.query_parser import PolarityQueryExpander
from scientific_contradiction_finder.schemas import Paper


def test_sqlite_cache_roundtrip(tmp_path):
    db_file = tmp_path / "test_cache.sqlite"
    cache = SQLiteCache(db_path=db_file)

    key = cache.compute_key("openalex", "/works", {"search": "metformin"})
    assert cache.get(key) is None

    cache.set(key, "openalex", "/works", {"results": [{"id": "W123", "title": "Test"}]})
    cached = cache.get(key)
    assert cached is not None
    assert cached["results"][0]["title"] == "Test"


def test_polarity_expansion():
    expander = PolarityQueryExpander()
    expanded = expander.expand("metformin reduces cardiovascular mortality")
    assert "reduces" in expanded.original_query
    # Check that opposing query replaced "reduces" with "increases"
    assert any("increases" in q for q in expanded.polarity_inversions)
    assert len(expanded.expanded_queries) >= 3


def test_decode_openalex_inverted_index():
    inv_index = {
        "Metformin": [0],
        "lowers": [1],
        "blood": [2],
        "glucose": [3],
        "significantly": [4],
    }
    decoded = decode_openalex_inverted_index(inv_index)
    assert decoded == "Metformin lowers blood glucose significantly"


def test_document_deduplication():
    p1 = Paper(
        paper_id="1",
        title="Impact of Vitamin D on Immunity",
        abstract="Abstract 1",
        doi="10.1001/jama.001",
        source_provider="openalex",
    )
    # Duplicate DOI
    p2 = Paper(
        paper_id="2",
        title="Different Title but Same DOI",
        abstract="Abstract 2",
        doi="10.1001/jama.001",
        source_provider="arxiv",
    )
    # Duplicate Normalized Title
    p3 = Paper(
        paper_id="3",
        title="impact of vitamin d on immunity!",
        abstract="Abstract 3",
        doi="10.1001/jama.002",
        source_provider="openalex",
    )
    p4 = Paper(
        paper_id="4",
        title="Completely Unique Study on Ketone Bodies",
        abstract="Abstract 4",
        doi="10.1001/jama.003",
        source_provider="openalex",
    )

    deduped = DocumentDeduplicator.deduplicate([p1, p2, p3, p4])
    assert len(deduped) == 2
    assert deduped[0].paper_id == "1"
    assert deduped[1].paper_id == "4"


def test_bm25_retrieval_and_rrf():
    p1 = Paper(
        paper_id="1",
        title="Metformin reduces HbA1c in diabetes",
        abstract="Clinical trial showing glycemic control with oral metformin.",
        source_provider="openalex",
    )
    p2 = Paper(
        paper_id="2",
        title="Deep Reinforcement Learning in Robotics",
        abstract="Autonomous quadcopter navigation via policy gradients.",
        source_provider="arxiv",
    )

    bm25 = BM25Retriever([p1, p2])
    results = bm25.retrieve("metformin glycemic control", top_k=2)
    assert len(results) == 2
    assert results[0][0].paper_id == "1"
    assert results[0][1] > results[1][1]

    # Test RRF
    rrf = ReciprocalRankFusion(k=60)
    fused = rrf.fuse([results, results], top_n=2)
    assert len(fused) == 2
    assert fused[0][0].paper_id == "1"
