import io
import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from xml.etree import ElementTree

import yaml

from includes.cli import run


COMPARE_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Site A</name><styleUrl>#homegardenbusiness</styleUrl><Point><coordinates>0,0,0</coordinates></Point></Placemark>
  <Placemark><name>MH 1</name><Point><coordinates>0.01,0,0</coordinates></Point></Placemark>
  <Placemark><name>Site B</name><styleUrl>#homegardenbusiness</styleUrl><Point><coordinates>0.02,0,0</coordinates></Point></Placemark>
  <Placemark><name>Alt junction</name><Point><coordinates>0.01,0.01,0</coordinates></Point></Placemark>
  <Placemark><name>Cable A</name><LineString><coordinates>0.001,0,0 0.009,0,0</coordinates></LineString></Placemark>
  <Placemark><name>Cable B</name><LineString><coordinates>0.011,0,0 0.019,0,0</coordinates></LineString></Placemark>
  <Placemark><name>Cable C</name><LineString><coordinates>0.001,0,0 0.01,0.01,0</coordinates></LineString></Placemark>
  <Placemark><name>Cable D</name><LineString><coordinates>0.01,0.01,0 0.019,0,0</coordinates></LineString></Placemark>
</Document></kml>"""


class CompareCliTests(unittest.TestCase):
    def _run_compare(self, ai_route_names: list[str]) -> tuple[int, Path, Mock]:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        input_path = root / "network.kml"
        input_path.write_text(COMPARE_KML, encoding="utf-8")
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "provider": "ollama",
                    "application": {"output_directory": "output", "log_directory": "logs"},
                    "prompts_file": str(Path(__file__).resolve().parents[1] / "prompts.yaml"),
                }
            ),
            encoding="utf-8",
        )
        logger = Mock()

        def select_ai_spans(**kwargs: object) -> list[object]:
            kwargs["request_metadata"].update(
                {
                    "prompt": '[{"role":"system","content":"compare prompt"}]',
                    "prompt_tokens": 14,
                    "completion_tokens": 6,
                    "total_tokens": 20,
                }
            )
            candidates = kwargs["candidates"]
            by_name = {candidate.route_name: candidate for candidate in candidates}
            return [by_name[name] for name in ai_route_names]

        with (
            patch("includes.cli.configure_logging", return_value=(logger, Mock())),
            patch("includes.cli.select_routes", side_effect=select_ai_spans),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = run(
                [
                    str(input_path),
                    "--mode",
                    "compare",
                    "--a-end",
                    "Site A",
                    "--z-end",
                    "Site B",
                    "--config",
                    str(config_path),
                ]
            )
        return exit_code, root / "output", logger

    def test_matching_routes_create_one_kmz_describing_both_methods(self) -> None:
        exit_code, output_directory, _ = self._run_compare(["Cable A", "Cable B"])

        kmz_files = list(output_directory.glob("*.kmz"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(kmz_files), 1)
        with zipfile.ZipFile(kmz_files[0]) as archive:
            root = ElementTree.fromstring(archive.read("doc.kml"))
        description = root.findtext("{http://www.opengis.net/kml/2.2}Document/{http://www.opengis.net/kml/2.2}description")
        self.assertIn("IA et Dijkstra", description)
        self.assertIn("llama3.1", description)
        database_path = output_directory.parent / "logs" / "requests.sqlite3"
        with closing(sqlite3.connect(database_path)) as connection:
            row = connection.execute(
                """SELECT mode, status, result_json, prompt, duration_ms,
                          provider_duration_ms, provider, model, prompt_tokens,
                          completion_tokens, total_tokens, diversity_span_id, kmz_path
                   FROM requests"""
            ).fetchone()
        self.assertEqual(row[0:2], ("compare", "success"))
        self.assertIn("Site A", row[2])
        self.assertEqual(row[3], '[{"role":"system","content":"compare prompt"}]')
        self.assertGreater(row[4], 0)
        self.assertGreater(row[5], 0)
        self.assertEqual(row[6:12], ("ollama", "llama3.1", 14, 6, 20, None))
        self.assertEqual(Path(row[12]), kmz_files[0].resolve())

    def test_different_routes_return_error_without_creating_outputs(self) -> None:
        exit_code, output_directory, logger = self._run_compare(["Cable C", "Cable D"])

        self.assertEqual(exit_code, 2)
        self.assertFalse(list(output_directory.glob("*.kmz")))
        self.assertFalse(list(output_directory.glob("*.json")))
        self.assertTrue(
            any("Comparaison échouée" in call.args[0] for call in logger.error.call_args_list)
        )
        with closing(sqlite3.connect(output_directory.parent / "logs" / "requests.sqlite3")) as connection:
            row = connection.execute("SELECT status, error FROM requests").fetchone()
        self.assertEqual(row[0], "error")
        self.assertIn("Comparaison échouée", row[1])


if __name__ == "__main__":
    unittest.main()