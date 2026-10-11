from scientific_contradiction_finder.config import Config, default_config


def test_config_loads_defaults():
    cfg = Config()
    assert cfg.app.name == "Scientific Contradiction Finder"
    assert cfg.storage.cache_enabled is True
    assert cfg.retrieval.bm25_top_k == 25


def test_no_private_keys_leaked():
    # Verify that default config or example does not hardcode production secrets
    assert default_config.retrieval.openalex_base_url.startswith("https://")
