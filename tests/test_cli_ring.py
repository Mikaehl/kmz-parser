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


RING_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Site A</name><styleUrl>#homegardenbusiness</styleUrl><Point><coordinates>0,0,0</coordinates></Point></Placemark>
  <Placemark><name>MH 1</name><Point><coordinates>0.01,0,0</coordinates></Point></Placemark>
  <Placemark><name>Site B</name><styleUrl>#homegardenbusiness</styleUrl><Point><coordinates>0.02,0,0</coordinates></Point></Placemark>
  <Placemark><name>Alt junction</name><Point><coordinates>0.01,0.01,0</coordinates></Point></Placemark>
  <Placemark><name>Cable A</name><LineString><coordinates>0.001,0,0 0.009,0,0</coordinates></LineString></Placemark>
  <Placemark><name>Cable B</name><LineString><coordinates>0.011,0,0 0.019,0,0</coordinates></LineString></Placemark>
  <Placemark><name>Cable C</name><LineString><coordinates>0.001,0,0 0.01,0.01,0</coordinates></LineString></Placemark>
    <Placemark><name>Cable D</name><LineString><coordinates>0.01,0.01,0 0.019,0,0</coordinates></LineString></Placemark>
    <Placemark><name>Cable E</name><LineString><coordinates>0.001,0.001,0 0.006,-0.001,0 0.014,0.001,0 0.019,-0.001,0</coordinates></LineString></Placemark>
</Document></kml>"""


class RingCliTests(unittest.TestCase):
    def _prepare_project(self, root: Path, kml: str = RING_KML) -> tuple[Path, Path]:
        input_path = root / "network.kml"
        input_path.write_text(kml, encoding="utf-8")
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
        return input_path, config_path

    def test_ring_ai_selects_two_non_crossing_routes_in_one_kmz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path, config_path = self._prepare_project(root)
            logger = Mock()

            def select_ring(**kwargs: object) -> list[list[object]]:
                candidates = kwargs["candidates"]
                self.assertEqual(
                    [candidate.route_name for candidate in candidates],
                    ["Cable A", "Cable B", "Cable C", "Cable D", "Cable E"],
                )
                by_name = {candidate.route_name: candidate for candidate in candidates}
                return [
                    [by_name["Cable A"], by_name["Cable B"]],
                    [by_name["Cable C"], by_name["Cable D"]],
                ]

            with (
                patch("includes.cli.configure_logging", return_value=(logger, Mock())),
                patch("includes.cli.select_ring_routes", side_effect=select_ring),
                patch("includes.cli.find_shortest_route") as find_shortest_route,
                redirect_stdout(io.StringIO()),
            ):
                exit_code = run(
                    [
                        str(input_path),
                        "--mode",
                        "ring",
                        "--a-end",
                        "Site A",
                        "--z-end",
                        "Site B",
                        "--build",
                        "--config",
                        str(config_path),
                    ]
                )

            output_directory = root / "output"
            standard_result = json.loads((output_directory / "network_ring.json").read_text(encoding="utf-8"))
            built_result_path = next(output_directory.glob("*_Site_A_Site_B_*.json"))
            built_results = json.loads(built_result_path.read_text(encoding="utf-8"))
            kmz_path = next(output_directory.glob("*.kmz"))
            with zipfile.ZipFile(kmz_path) as archive:
                kml_root = ElementTree.fromstring(archive.read("doc.kml"))

        self.assertEqual(exit_code, 0)
        find_shortest_route.assert_not_called()
        self.assertEqual(len(standard_result), 2)
        self.assertEqual(len(built_results), 2)
        self.assertTrue(standard_result[0]["Route"].startswith("Route 1 (IA):"))
        self.assertTrue(standard_result[1]["Route"].startswith("Route 2 (IA):"))
        first_spans = {span["id"] for span in built_results[0]["Spans"]}
        second_spans = {span["id"] for span in built_results[1]["Spans"]}
        self.assertFalse(first_spans.intersection(second_spans))
        self.assertEqual(len(kml_root.findall(".//{http://www.opengis.net/kml/2.2}Folder/{http://www.opengis.net/kml/2.2}Placemark")), 4)
        styles = {
            style.get("id"): style.findtext("{http://www.opengis.net/kml/2.2}LineStyle/{http://www.opengis.net/kml/2.2}color")
            for style in kml_root.findall(".//{http://www.opengis.net/kml/2.2}Style")
            if style.get("id", "").startswith("routeStyle")
        }
        self.assertEqual(len(styles), 2)
        self.assertNotEqual(styles["routeStyle"], styles["routeStyle2"])

    def test_ring_fails_when_ai_returns_shared_spans(self) -> None:
        direct_route_kml = """<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
          <Placemark><name>Site A</name><Point><coordinates>0,0,0</coordinates></Point></Placemark>
          <Placemark><name>Site B</name><Point><coordinates>1,0,0</coordinates></Point></Placemark>
          <Placemark><name>Only cable</name><LineString><coordinates>0.001,0,0 0.999,0,0</coordinates></LineString></Placemark>
        </Document></kml>"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path, config_path = self._prepare_project(root, direct_route_kml)
            logger = Mock()
            with (
                patch("includes.cli.configure_logging", return_value=(logger, Mock())),
                patch(
                    "includes.cli.select_ring_routes",
                    side_effect=lambda **kwargs: [[kwargs["candidates"][0]], [kwargs["candidates"][0]]],
                ) as select_ring,
                patch("includes.cli.find_shortest_route") as find_shortest_route,
                redirect_stdout(io.StringIO()),
            ):
                exit_code = run(
                    [
                        str(input_path),
                        "--mode",
                        "ring",
                        "--a-end",
                        "Site A",
                        "--z-end",
                        "Site B",
                        "--config",
                        str(config_path),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertFalse((root / "output" / "network_ring.json").exists())
        select_ring.assert_called_once()
        find_shortest_route.assert_not_called()


if __name__ == "__main__":
    unittest.main()