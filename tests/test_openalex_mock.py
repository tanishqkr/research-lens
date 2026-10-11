import pytest
import respx
import httpx
from scientific_contradiction_finder.retrieval.openalex_client import OpenAlexClient
from scientific_contradiction_finder.retrieval.cache import SQLiteCache


@pytest.mark.asyncio
@respx.mock
async def test_openalex_client_mock_search(tmp_path):
    cache = SQLiteCache(db_path=tmp_path / "cache.sqlite")
    client = OpenAlexClient(cache=cache, timeout_seconds=5)

    mock_response = {
        "results": [
            {
                "id": "https://openalex.org/W12345",
                "title": "Metformin and Cardiovascular Outcomes",
                "publication_year": 2022,
                "doi": "https://doi.org/10.1001/jama.2022.01",
                "abstract_inverted_index": {
                    "Metformin": [0],
                    "reduced": [1],
                    "all-cause": [2],
                    "mortality": [3]
                },
                "authorships": [
                    {"author": {"display_name": "Jane Doe"}}
                ]
            }
        ]
    }

    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    papers = await client.search_works("metformin", limit=10)
    assert len(papers) == 1
    assert papers[0].paper_id == "W12345"
    assert papers[0].title == "Metformin and Cardiovascular Outcomes"
    assert papers[0].abstract == "Metformin reduced all-cause mortality"
    assert papers[0].authors == ["Jane Doe"]


@pytest.mark.asyncio
@respx.mock
async def test_openalex_client_rate_limit_backoff(tmp_path):
    cache = SQLiteCache(db_path=tmp_path / "cache.sqlite")
    client = OpenAlexClient(cache=cache, timeout_seconds=5, max_retries=2)

    # First request returns 429, second returns 200
    route = respx.get("https://api.openalex.org/works")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json={"results": []})
    ]

    papers = await client.search_works("test_backoff", limit=5)
    assert papers == []
    assert route.call_count == 2
