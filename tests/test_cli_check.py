import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from includes.cli import run
from includes.request_store import record_request


CHECK_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Folder><name>Sites</name>
    <Placemark id="SITE-A"><name>Site A</name><Point><coordinates>2.35,48.86,0</coordinates></Point></Placemark>
  </Folder>
  <Folder><name>Cables</name>
    <Placemark id="CABLE-A"><name>Cable A</name><LineString><coordinates>2.35,48.86,0 2.36,48.87,0</coordinates></LineString></Placemark>
  </Folder>
</Document></kml>"""


class CheckCliTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "provider": "ollama",
                    "application": {"output_directory": "output", "log_directory": "logs"},
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_clist_displays_saved_anomalies_as_a_terminal_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._write_config(root)
            database_path = root / "logs" / "requests.sqlite3"
            record_request(
                database_path,
                {
                    "mode": "check",
                    "status": "success",
                    "result": {"inconsistency_count": 1, "potential_solution_count": 1},
                    "prompt": "check prompt",
                    "duration_ms": 100,
                    "provider_duration_ms": 80,
                    "provider": "ollama",
                    "model": "check-model",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "inconsistency_count": 1,
                    "potential_solution_count": 1,
                    "anomalies": [
                        {
                            "type d'objet": "LineString",
                            "id d'objet": "CABLE-A",
                            "incohérence trouvée": "Extrémité non raccordée",
                            "explication de l'incohérence": "Le point final ne rejoint aucun joint.",
                            "solution de résolution possible de l'incohérence": "Corriger la géométrie.",
                        }
                    ],
                },
            )
            stdout = io.StringIO()
            with (
                patch("includes.cli.configure_logging", return_value=(Mock(), Mock())),
                patch("includes.cli.check_kml_consistency") as check_kml,
                redirect_stdout(stdout),
            ):
                exit_code = run(["--clist", "--config", str(config_path)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("ID objet", output)
        self.assertIn("CABLE-A", output)
        self.assertIn("Extrémité non", output)
        self.assertIn("raccordée", output)
        self.assertIn("Corriger la géométrie.", output)
        self.assertIn("ollama/check-", output)
        self.assertIn("model", output)
        self.assertIn("+", output)
        check_kml.assert_not_called()

    def test_clist_reports_when_no_anomalies_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._write_config(root)
            stdout = io.StringIO()
            with (
                patch("includes.cli.configure_logging", return_value=(Mock(), Mock())),
                redirect_stdout(stdout),
            ):
                exit_code = run(["--clist", "--config", str(config_path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("Aucune anomalie enregistrée", stdout.getvalue())

    def test_check_records_only_counts_and_ai_metadata_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "network.kml"
            input_path.write_text(CHECK_KML, encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "provider": "ollama",
                        "ollama": {"model": "check-model", "timeout_seconds": 13},
                        "application": {"output_directory": "output", "log_directory": "logs"},
                        "prompts_file": str(Path(__file__).resolve().parents[1] / "prompts.yaml"),
                    }
                ),
                encoding="utf-8",
            )
            logs_path = root / "logs"
            logs_path.mkdir()
            with closing(sqlite3.connect(logs_path / "requests.sqlite3")) as connection:
                connection.execute(
                    """CREATE TABLE requests (
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
                    )"""
                )
            anomaly = {
                "type d'objet": "LineString",
                "id d'objet": "CABLE-A",
                "incohérence trouvée": "Extrémité non raccordée",
                "explication de l'incohérence": "L'extrémité ne rejoint aucun objet du réseau.",
                "solution de résolution possible de l'incohérence": "Corriger les coordonnées de l'extrémité.",
            }
            logger = Mock()

            def report_anomaly(**kwargs: object) -> list[dict[str, str]]:
                objects = kwargs["objects"]
                self.assertEqual([item["object_id"] for item in objects], ["SITE-A", "CABLE-A"])
                kwargs["request_metadata"].update(
                    {
                        "prompt": '[{"role":"system","content":"check prompt"}]',
                        "provider": "ollama",
                        "model": "check-model",
                        "prompt_tokens": 120,
                        "completion_tokens": 35,
                        "total_tokens": 155,
                        "potential_solution_count": 1,
                    }
                )
                return [anomaly]

            with (
                patch("includes.cli.configure_logging", return_value=(logger, Mock())),
                patch("includes.cli.check_kml_consistency", side_effect=report_anomaly),
                patch("includes.cli.parse_kml_file") as parse_routes,
                redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = run(
                    [
                        str(input_path),
                        "--check",
                        "--config",
                        str(config_path),
                    ]
                )

            with closing(sqlite3.connect(root / "logs" / "requests.sqlite3")) as connection:
                record = connection.execute(
                    """SELECT mode, status, result_json, prompt, provider, model,
                              prompt_tokens, completion_tokens, total_tokens,
                              inconsistency_count, potential_solution_count, duration_ms
                       FROM requests"""
                ).fetchone()
                anomaly_rows = connection.execute(
                    """SELECT object_type, object_id, inconsistency, explanation,
                              potential_solution, ai_provider, ai_model, ai_engine
                       FROM check_anomalies ORDER BY anomaly_index"""
                ).fetchall()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertFalse((root / "output" / "network_check.json").exists())
        parse_routes.assert_not_called()
        self.assertEqual(record[0:2], ("check", "success"))
        self.assertEqual(
            json.loads(record[2]),
            {"inconsistency_count": 1, "potential_solution_count": 1},
        )
        self.assertEqual(record[3:6], ('[{"role":"system","content":"check prompt"}]', "ollama", "check-model"))
        self.assertEqual(record[6:9], (120, 35, 155))
        self.assertEqual(record[9:11], (1, 1))
        self.assertEqual(
            anomaly_rows,
            [
                (
                    "LineString",
                    "CABLE-A",
                    "Extrémité non raccordée",
                    "L'extrémité ne rejoint aucun objet du réseau.",
                    "Corriger les coordonnées de l'extrémité.",
                    "ollama",
                    "check-model",
                    "ollama/check-model",
                )
            ],
        )
        self.assertGreater(record[11], 0)
        self.assertTrue(
            any("Durée totale de la requête" in call.args[0] for call in logger.info.call_args_list)
        )
        self.assertTrue(
            any("1 incohérence(s) détectée(s)" in call.args[0] for call in logger.info.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()