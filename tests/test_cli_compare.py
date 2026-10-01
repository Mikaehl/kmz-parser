import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
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

    def test_different_routes_return_error_without_creating_outputs(self) -> None:
        exit_code, output_directory, logger = self._run_compare(["Cable C", "Cable D"])

        self.assertEqual(exit_code, 2)
        self.assertFalse(list(output_directory.glob("*.kmz")))
        self.assertFalse(list(output_directory.glob("*.json")))
        self.assertTrue(
            any("Comparaison échouée" in call.args[0] for call in logger.error.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()