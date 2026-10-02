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
from includes.kml_parser import RouteCandidate


class ModelListCliTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "provider": "ollama",
                    "ollama": {
                        "base_url": "http://ollama.test:11434",
                        "model": "configured-model",
                        "timeout_seconds": 17,
                    },
                    "application": {"output_directory": "output", "log_directory": "logs"},
                    "prompts_file": str(Path(__file__).resolve().parents[1] / "prompts.yaml"),
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_lists_models_without_input_file_or_ai_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "ollama": {
                            "base_url": "http://ollama.test:11434",
                            "timeout_seconds": 17,
                        },
                        "application": {"output_directory": "output", "log_directory": "logs"},
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with (
                patch("includes.cli.configure_logging", return_value=(Mock(), Mock())),
                patch("includes.cli.list_ollama_models", return_value=["llama3.1:latest", "qwen2.5:7b"]) as list_models,
                patch("includes.cli.select_routes") as select_routes,
                redirect_stdout(stdout),
            ):
                exit_code = run(["--mlist", "--config", str(config_path)])

            with closing(sqlite3.connect(root / "logs" / "requests.sqlite3")) as connection:
                request = connection.execute(
                    "SELECT mode, status, result_json, provider FROM requests"
                ).fetchone()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "1. llama3.1:latest\n2. qwen2.5:7b\n")
        list_models.assert_called_once_with("http://ollama.test:11434", 17)
        select_routes.assert_not_called()
        self.assertEqual(request[0:2], ("mlist", "success"))
        self.assertEqual(json.loads(request[2]), ["llama3.1:latest", "qwen2.5:7b"])
        self.assertEqual(request[3], "ollama")

    def test_muse_overrides_configured_model_without_changing_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._write_config(root)
            input_path = root / "routes.kml"
            input_path.touch()
            route = RouteCandidate("R001", "Cable", "A", "B", 1, [(0, 0), (1, 0)])
            with (
                patch("includes.cli.configure_logging", return_value=(Mock(), Mock())),
                patch("includes.cli.list_ollama_models", return_value=["model-one", "model-two"]),
                patch("includes.cli.parse_kml_file", return_value=[route]),
                patch("includes.cli.select_routes", side_effect=lambda **kwargs: kwargs["candidates"]) as select_routes,
                redirect_stdout(io.StringIO()),
            ):
                exit_code = run(
                    [
                        str(input_path),
                        "--mode",
                        "distance",
                        "--muse",
                        "2",
                        "--config",
                        str(config_path),
                    ]
                )

            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(select_routes.call_args.kwargs["model"], "model-two")
        self.assertEqual(saved_config["ollama"]["model"], "configured-model")

    def test_mchange_persists_selected_model_without_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._write_config(root)
            stdout = io.StringIO()
            with (
                patch("includes.cli.configure_logging", return_value=(Mock(), Mock())),
                patch("includes.cli.list_ollama_models", return_value=["model-one", "model-two"]),
                patch("includes.cli.select_routes") as select_routes,
                redirect_stdout(stdout),
            ):
                exit_code = run(["--mchange", "2", "--config", str(config_path)])

            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(saved_config["ollama"]["model"], "model-two")
        self.assertIn("model-two", stdout.getvalue())
        select_routes.assert_not_called()


if __name__ == "__main__":
    unittest.main()