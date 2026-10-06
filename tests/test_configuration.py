import tempfile
import unittest
from pathlib import Path

import yaml

from includes.configuration import load_settings


class ConfigurationTests(unittest.TestCase):
    def test_application_log_info_defaults_to_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text("{}", encoding="utf-8")

            settings, _ = load_settings(config_path)

        self.assertEqual(settings["application"]["log_info"], 0)

    def test_application_log_info_accepts_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump({"application": {"log_info": 2}}),
                encoding="utf-8",
            )

            settings, _ = load_settings(config_path)

        self.assertEqual(settings["application"]["log_info"], 2)

    def test_application_log_info_rejects_invalid_value(self) -> None:
        for value in (-1, True, 2.5, "2"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary_directory:
                config_path = Path(temporary_directory) / "config.yaml"
                config_path.write_text(
                    yaml.safe_dump({"application": {"log_info": value}}),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "application.log_info must be a non-negative integer",
                ):
                    load_settings(config_path)

    def test_route_planning_uses_default_dijkstra_tool_call_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text("{}", encoding="utf-8")

            settings, _ = load_settings(config_path)

        self.assertEqual(settings["route_planning"]["max_dijkstra_tool_calls"], 8)

    def test_route_planning_accepts_custom_dijkstra_tool_call_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {"route_planning": {"max_dijkstra_tool_calls": 3}}
                ),
                encoding="utf-8",
            )

            settings, _ = load_settings(config_path)

        self.assertEqual(settings["route_planning"]["max_dijkstra_tool_calls"], 3)

    def test_route_planning_rejects_invalid_dijkstra_tool_call_limit(self) -> None:
        invalid_limits = (0, -1, True, 2.5, "3")
        for limit in invalid_limits:
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as temporary_directory:
                config_path = Path(temporary_directory) / "config.yaml"
                config_path.write_text(
                    yaml.safe_dump(
                        {"route_planning": {"max_dijkstra_tool_calls": limit}}
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "route_planning.max_dijkstra_tool_calls must be a positive integer",
                ):
                    load_settings(config_path)


if __name__ == "__main__":
    unittest.main()
