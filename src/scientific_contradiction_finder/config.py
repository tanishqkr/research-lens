"""Configuration management for Scientific Contradiction Finder."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load local environment variables from .env if present
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AppSettings(BaseModel):
    name: str = "Scientific Contradiction Finder"
    version: str = "0.1.0"
    log_level: str = os.getenv("SCF_LOG_LEVEL", "INFO")
    offline_mode: bool = os.getenv("SCF_OFFLINE_MODE", "false").lower() in ("true", "1", "yes")


class StorageSettings(BaseModel):
    data_dir: Path = Path(os.getenv("SCF_DATA_DIR", PROJECT_ROOT / "data"))
    cache_dir: Path = Path(os.getenv("SCF_CACHE_DIR", PROJECT_ROOT / "data" / "cache"))
    sqlite_path: Path = Path(os.getenv("SCF_SQLITE_PATH", PROJECT_ROOT / "data" / "scf.sqlite"))
    cache_enabled: bool = True
    cache_ttl_days: int = 7


class RetrievalSettings(BaseModel):
    openalex_api_key: str | None = os.getenv("OPENALEX_API_KEY")
    openalex_mailto: str | None = os.getenv("CONTACT_EMAIL") or os.getenv("OPENALEX_MAILTO")
    max_results_per_provider: int = int(os.getenv("SCF_MAX_RESULTS_PER_PROVIDER", 25))
    request_timeout_seconds: int = int(os.getenv("SCF_REQUEST_TIMEOUT_SECONDS", 15))
    max_retries: int = 3
    openalex_base_url: str = "https://api.openalex.org"
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    bm25_top_k: int = 25
    dense_top_k: int = 25
    rrf_top_k: int = int(os.getenv("SCF_TOP_K_EVIDENCE", 20))
    rrf_k: int = 60


class ModelSettings(BaseModel):
    dense_retrieval_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    device: str = os.getenv("DEVICE", "cpu")


class ExtractionSettings(BaseModel):
    top_evidence_sentences: int = 3
    min_evidence_length_chars: int = 20


class ReasoningSettings(BaseModel):
    similarity_threshold_compatible: float = 0.70
    similarity_threshold_incomparable: float = 0.35
    confidence_abstention_threshold: float = 0.40


class Config(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    reasoning: ReasoningSettings = Field(default_factory=ReasoningSettings)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "default.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()


# Global default configuration instance
default_config = Config.load()
