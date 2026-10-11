"""SQLite caching layer for HTTP requests, scholarly queries, and abstract payloads."""

import sqlite3
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Any, Dict


class SQLiteCache:
    """Persistent local SQLite cache for scholarly API responses."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            self.db_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "scf.sqlite"
        else:
            self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def compute_key(provider: str, endpoint: str, params: Dict[str, Any]) -> str:
        serialized = json.dumps(params, sort_keys=True)
        raw = f"{provider}:{endpoint}:{serialized}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response_json FROM api_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set(self, cache_key: str, provider: str, endpoint: str, data: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO api_cache (cache_key, provider, endpoint, response_json, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (cache_key, provider, endpoint, json.dumps(data), time.time()))
            conn.commit()

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM api_cache")
            conn.commit()
