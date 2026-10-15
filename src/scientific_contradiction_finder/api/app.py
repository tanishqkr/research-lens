"""Lightweight ASGI Backend for Scientific Contradiction Finder.

Wires the Next.js research workbench directly to the Python pipeline,
preserving typed schemas, diagnostics, and benchmark evaluation capabilities.
"""

import logging
from typing import Any, Dict, List

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline
from scientific_contradiction_finder.schemas import FindingPair, ResearchReport

logger = logging.getLogger("scientific_api")
logging.basicConfig(level=logging.INFO)

# Global pipeline instance (cached to avoid redundant model reloading)
_pipeline_instance: ContradictionFinderPipeline | None = None
_benchmark_loader: BenchmarkLoader | None = None


def get_pipeline() -> ContradictionFinderPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ContradictionFinderPipeline()
    return _pipeline_instance


def get_loader() -> BenchmarkLoader:
    global _benchmark_loader
    if _benchmark_loader is None:
        _benchmark_loader = BenchmarkLoader()
    return _benchmark_loader


CURATED_PRESETS: List[str] = [
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
]


async def health_endpoint(request: Request) -> JSONResponse:
    """Returns system status, service info, and provider availability."""
    pipeline = get_pipeline()
    return JSONResponse(
        {
            "status": "healthy",
            "service": "scientific-contradiction-finder",
            "version": "0.1.0",
            "providers": {
                "openalex": True,
                "arxiv": True,
                "groq": bool(pipeline.groq_client.is_available),
            },
            "offline_mode": pipeline.cfg.app.offline_mode,
        }
    )


async def presets_endpoint(request: Request) -> JSONResponse:
    """Returns curated empirical scientific query presets."""
    return JSONResponse({"presets": CURATED_PRESETS})


async def analyze_endpoint(request: Request) -> JSONResponse:
    """Runs the live empirical research pipeline for a given query."""
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON in request body."}, status_code=400)

    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse(
            {"error": "Query parameter is required and cannot be empty."}, status_code=400
        )

    try:
        limit = int(body.get("limit", 10))
    except (ValueError, TypeError):
        limit = 10
    limit = max(1, min(limit, 25))
    use_groq = bool(body.get("use_groq_interpretation", False) or body.get("use_groq", False))

    pipeline = get_pipeline()

    try:
        raw_res = await pipeline.run(query=query, limit=limit, use_groq_interpretation=use_groq)

        # Normalize comparisons for clean JSON serialization
        comparisons_json = []
        for pair, pred in raw_res.get("comparisons", []):
            comparisons_json.append(
                {
                    "pair": pair.model_dump(mode="json"),
                    "prediction": pred.model_dump(mode="json"),
                }
            )

        report: ResearchReport = raw_res["research_report"]
        report_json = report.model_dump(mode="json")

        findings_json = [f.model_dump(mode="json") for f in raw_res.get("findings", [])]

        response_data = {
            "query": raw_res["query"],
            "run_id": raw_res.get("run_id"),
            "query_intent": raw_res.get("query_intent"),
            "execution_state": raw_res.get("execution_state"),
            "papers_found": raw_res.get("papers_found", 0),
            "findings_extracted": raw_res.get("findings_extracted", 0),
            "comparisons": comparisons_json,
            "findings": findings_json,
            "diagnostics": raw_res.get("diagnostics", {}),
            "research_report": report_json,
        }
        return JSONResponse(response_data)
    except Exception as exc:
        logger.exception("Error executing contradiction pipeline: %s", exc)
        return JSONResponse(
            {
                "error": "Failed to complete research pipeline execution.",
                "detail": str(exc),
            },
            status_code=500,
        )


async def benchmark_pairs_endpoint(request: Request) -> JSONResponse:
    """Returns CAS-Con150 benchmark finding pairs."""
    try:
        loader = get_loader()
        pairs = loader.load_cas_con150()
        pairs_json = [p.model_dump(mode="json") for p in pairs]
        return JSONResponse({"total_pairs": len(pairs_json), "pairs": pairs_json})
    except Exception as exc:
        logger.exception("Error loading benchmark pairs: %s", exc)
        return JSONResponse(
            {"error": "Failed to load benchmark pairs.", "detail": str(exc)}, status_code=500
        )


async def benchmark_classify_endpoint(request: Request) -> JSONResponse:
    """Classifies a benchmark finding pair using the condition-aware relation classifier."""
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON in request body."}, status_code=400)

    pair_data = body.get("pair")
    if not pair_data:
        return JSONResponse({"error": "Missing 'pair' object in request body."}, status_code=400)

    try:
        pair = FindingPair(**pair_data)
        pipeline = get_pipeline()
        prediction = pipeline.classifier.classify_pair(pair)
        return JSONResponse({"prediction": prediction.model_dump(mode="json")})
    except Exception as exc:
        logger.exception("Error classifying pair: %s", exc)
        return JSONResponse(
            {"error": "Failed to classify finding pair.", "detail": str(exc)}, status_code=500
        )


routes = [
    Route("/api/health", health_endpoint, methods=["GET"]),
    Route("/api/presets", presets_endpoint, methods=["GET"]),
    Route("/api/analyze", analyze_endpoint, methods=["POST"]),
    Route("/api/benchmark/pairs", benchmark_pairs_endpoint, methods=["GET"]),
    Route("/api/benchmark/classify", benchmark_classify_endpoint, methods=["POST"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware)
