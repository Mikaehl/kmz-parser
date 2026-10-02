import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_request(database_path: Path, request: dict[str, Any]) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
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
                inconsistency_count INTEGER,
                potential_solution_count INTEGER,
                error TEXT
            )
            """
        )
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(requests)")
        }
        for column in ("inconsistency_count", "potential_solution_count"):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE requests ADD COLUMN {column} INTEGER")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS check_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                anomaly_index INTEGER NOT NULL,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                inconsistency TEXT NOT NULL,
                explanation TEXT NOT NULL,
                potential_solution TEXT NOT NULL,
                ai_provider TEXT,
                ai_model TEXT,
                ai_engine TEXT,
                FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE
            )
            """
        )
        request_cursor = connection.execute(
            """
            INSERT INTO requests (
                created_at, mode, status, result_json, prompt, duration_ms,
                provider_duration_ms, provider, model, prompt_tokens,
                completion_tokens, total_tokens, diversity_span_id, kmz_path,
                inconsistency_count, potential_solution_count, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                request.get("inconsistency_count"),
                request.get("potential_solution_count"),
                request.get("error"),
            ),
        )
        if request["mode"] == "check":
            provider = request.get("provider")
            model = request.get("model")
            engine_parts = [part for part in (provider, model) if isinstance(part, str) and part]
            engine_name = "/".join(engine_parts) or None
            for anomaly_index, anomaly in enumerate(request.get("anomalies", []), start=1):
                connection.execute(
                    """
                    INSERT INTO check_anomalies (
                        request_id, anomaly_index, object_type, object_id,
                        inconsistency, explanation, potential_solution,
                        ai_provider, ai_model, ai_engine
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_cursor.lastrowid,
                        anomaly_index,
                        anomaly["type d'objet"],
                        anomaly["id d'objet"],
                        anomaly["incohérence trouvée"],
                        anomaly["explication de l'incohérence"],
                        anomaly["solution de résolution possible de l'incohérence"],
                        provider,
                        model,
                        engine_name,
                    ),
                )
        connection.commit()


def list_check_anomalies(database_path: Path) -> list[dict[str, Any]]:
    if not database_path.is_file():
        return []
    with closing(sqlite3.connect(database_path)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'check_anomalies'"
        ).fetchone()
        if table is None:
            return []
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT anomaly.request_id, anomaly.anomaly_index, anomaly.object_type,
                       anomaly.object_id, anomaly.inconsistency, anomaly.explanation,
                       anomaly.potential_solution, anomaly.ai_engine
                FROM check_anomalies AS anomaly
                JOIN requests ON requests.id = anomaly.request_id
                ORDER BY requests.id DESC, anomaly.anomaly_index
                """
            )
        ]