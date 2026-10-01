import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_request(database_path: Path, request: dict[str, Any]) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                prompt TEXT,
                duration_ms REAL NOT NULL,
                provider_duration_ms REAL,
                provider TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                diversity_span_id TEXT,
                kmz_path TEXT,
                error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO requests (
                created_at, mode, status, result_json, prompt, duration_ms,
                provider_duration_ms, provider, model, prompt_tokens,
                completion_tokens, total_tokens, diversity_span_id, kmz_path, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                request["mode"],
                request["status"],
                json.dumps(request.get("result"), ensure_ascii=False),
                request.get("prompt"),
                request["duration_ms"],
                request.get("provider_duration_ms"),
                request.get("provider"),
                request.get("model"),
                request.get("prompt_tokens"),
                request.get("completion_tokens"),
                request.get("total_tokens"),
                request.get("diversity_span_id"),
                request.get("kmz_path"),
                request.get("error"),
            ),
        )
        connection.commit()