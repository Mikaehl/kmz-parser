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


CURRENT_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>New Site A</name><Point><coordinates>0,0,0</coordinates></Point></Placemark>
  <Placemark><name>New Junction</name><Point><coordinates>1,0,0</coordinates></Point></Placemark>
  <Placemark><name>New Site B</name><Point><coordinates>2,0,0</coordinates></Point></Placemark>
  <Placemark><name>New Cable A</name><LineString><coordinates>0.001,0,0 1.001,0,0</coordinates></LineString></Placemark>
  <Placemark><name>New Cable B</name><LineString><coordinates>0.999,0,0 2.001,0,0</coordinates></LineString></Placemark>
</Document></kml>"""


class DiverseSpanCliTests(unittest.TestCase):
    def test_span_reference_falls_back_to_non_common_routes_without_matching_sites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "different_network.kml"
            input_path.write_text(CURRENT_KML, encoding="utf-8")
            output_directory = root / "output"
            output_directory.mkdir()
            reference_path = output_directory / "0001_Old_Site_A_Old_Site_B_2000m.json"
            reference_path.write_text(
                json.dumps(
                    [
                        {
                            "Route": "Old Cable A -> Old Cable B",
                            "A-END": "Old Site A",
                            "Z-END": "Old Site B",
                            "distance": 2,
                            "Spans": [
                                {"id": "old-cable-a-id", "name": "Old Cable A"},
                                {"id": "old-cable-b-id", "name": "Old Cable B"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
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
            output_json = root / "result.json"
            logger = Mock()
            data_logger = Mock()

            def select_non_common_routes(**kwargs: object) -> list[object]:
                candidates = kwargs["candidates"]
                self.assertEqual([route.route_name for route in candidates], ["New Cable A", "New Cable B"])
                return candidates

            with (
                patch("includes.cli.configure_logging", return_value=(logger, data_logger)),
                patch("includes.cli.select_routes", side_effect=select_non_common_routes),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = run(
                    [
                        str(input_path),
                        "--mode",
                        "diverse",
                        "--span",
                        "0001",
                        "--build",
                        "--config",
                        str(config_path),
                        "--output",
                        str(output_json),
                    ]
                )

            results = json.loads(output_json.read_text(encoding="utf-8"))
            kmz_path = next(output_directory.glob("*.kmz"))
            with zipfile.ZipFile(kmz_path) as archive:
                root = ElementTree.fromstring(archive.read("doc.kml"))
            description = root.findtext("{http://www.opengis.net/kml/2.2}Document/{http://www.opengis.net/kml/2.2}description")

        self.assertEqual(exit_code, 0)
        self.assertEqual([result["Route"] for result in results], ["New Cable A", "New Cable B"])
        self.assertIn("Mode utilisé : IA", description)
        self.assertIn("Modèle IA : llama3.1", description)
        self.assertIn("référence span 0001", description)
        self.assertIn("spans exclus : old-cable-a-id, old-cable-b-id", description)
        self.assertTrue(
            any(
                call.args[0] == "Route analysis"
                and call.kwargs["extra"]["data"]["model"] == "llama3.1"
                and "old-cable-a-id" in call.kwargs["extra"]["data"]["diversity"]
                for call in data_logger.info.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
