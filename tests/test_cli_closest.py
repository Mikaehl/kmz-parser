import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from includes.cli import run


KML_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Site A</name><styleUrl>#homegardenbusiness</styleUrl><Point><coordinates>0,0,0</coordinates></Point></Placemark>
  <Placemark><name>MH 1</name><Point><coordinates>0.01,0,0</coordinates></Point></Placemark>
  <Placemark><name>Site B</name><styleUrl>#homegardenbusiness</styleUrl><Point><coordinates>0.02,0,0</coordinates></Point></Placemark>
  <Placemark><name>Cable A</name><LineString><coordinates>0.001,0,0 0.009,0,0</coordinates></LineString></Placemark>
  <Placemark><name>Cable B</name><LineString><coordinates>0.011,0,0 0.019,0,0</coordinates></LineString></Placemark>
</Document></kml>"""


class ClosestCliTests(unittest.TestCase):
    def _create_inputs(self, directory: Path, max_distance_km: float) -> tuple[Path, Path, Path]:
        kml_path = directory / "network.kml"
        kml_path.write_text(KML_CONTENT, encoding="utf-8")
        config_path = directory / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "provider": "ollama",
                    "application": {"output_directory": "output", "log_directory": "logs"},
                    "closest": {"max_distance_km": max_distance_km},
                    "prompts_file": str(Path(__file__).resolve().parents[1] / "prompts.yaml"),
                }
            ),
            encoding="utf-8",
        )
        return kml_path, config_path, directory / "result.json"

    @staticmethod
    def _select_nearest(**kwargs: object) -> list[object]:
        candidates = kwargs["candidates"]
        return [min(candidates, key=lambda candidate: candidate.distance_km)]

    def test_closest_selects_manhole_and_writes_result_within_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path, config_path, output_path = self._create_inputs(Path(temporary_directory), 10)
            with (
                patch("includes.cli.configure_logging", return_value=(Mock(), Mock())),
                patch("includes.cli.geocode_address", return_value=(0.0101, 0.0)),
                patch("includes.cli.select_routes", side_effect=self._select_nearest),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = run(
                    [str(kml_path), "--closest", "10 Main Street", "--config", str(config_path), "--output", str(output_path)]
                )

            result = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(result[0]["Route"], "MH 1")
        self.assertEqual(result[0]["A-END"], "MH 1")

    def test_closest_rejects_manhole_beyond_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path, config_path, output_path = self._create_inputs(Path(temporary_directory), 0.5)
            logger = Mock()
            with (
                patch("includes.cli.configure_logging", return_value=(logger, Mock())),
                patch("includes.cli.geocode_address", return_value=(0.02, 0.0)),
                patch("includes.cli.select_routes", side_effect=self._select_nearest),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = run(
                    [str(kml_path), "--closest", "10 Main Street", "--config", str(config_path), "--output", str(output_path)]
                )

        self.assertEqual(exit_code, 1)
        self.assertFalse(output_path.exists())
        logger.error.assert_called_with("Trop éloigné de l'un des points de raccordement possible.")

    def test_djk_calculates_shortest_route_without_calling_ai(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path, config_path, output_path = self._create_inputs(Path(temporary_directory), 10)
            stdout = io.StringIO()
            logger = Mock()
            data_logger = Mock()
            with (
                patch("includes.cli.configure_logging", return_value=(logger, data_logger)),
                patch("includes.cli.select_routes") as select_routes,
                redirect_stdout(stdout),
            ):
                exit_code = run(
                    [
                        str(kml_path),
                        "--mode",
                        "djk",
                        "--a-end",
                        "Site A",
                        "--z-end",
                        "Site B",
                        "--config",
                        str(config_path),
                        "--output",
                        str(output_path),
                    ]
                )

            result = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(result[0]["Route"], "Cable A -> Cable B")
        self.assertEqual(result[0]["A-END"], "Site A")
        self.assertEqual(result[0]["Z-END"], "Site B")
        select_routes.assert_not_called()
        logger.info.assert_any_call(
            "Mode utilisé : Dijkstra ; modèle IA : non utilisé ; diversité : aucune."
        )
        self.assertTrue(
            any(
                call.args[0] == "Route analysis"
                and call.kwargs["extra"]["data"]
                == {"strategy": "Dijkstra", "model": None, "diversity": None}
                for call in data_logger.info.call_args_list
            )
        )
        self.assertEqual(json.loads(stdout.getvalue())[0]["Route"], "Cable A -> Cable B")


if __name__ == "__main__":
    unittest.main()
