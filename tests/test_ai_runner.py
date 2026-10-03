import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from includes.kml_parser import parse_kml_file, parse_kml_objects
from includes.route_planner import find_shortest_route
from tools.ai_test_runner import _case_arguments, _provider_details


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AiTestRunnerTests(unittest.TestCase):
    def test_suite_lists_fifty_cases_without_calling_the_ai(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "ai_test_runner.py"),
                "--list",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("50 tests.", process.stdout)
        for test_id in ("S001", "D001", "R001", "A001"):
            self.assertIn(test_id, process.stdout)

    def test_all_suite_inputs_are_parseable_kml_files(self) -> None:
        suite_path = PROJECT_ROOT / "tests" / "ai_tests.yaml"
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        input_paths = {
            (suite_path.parent / case["input"]).resolve()
            for case in suite["tests"]
        }

        self.assertEqual(len(suite["tests"]), 50)
        for input_path in input_paths:
            with self.subTest(input=input_path.name):
                self.assertTrue(parse_kml_objects(input_path))

    def test_route_cases_have_valid_endpoints_and_cable_exclusions(self) -> None:
        suite_path = PROJECT_ROOT / "tests" / "ai_tests.yaml"
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        route_cases = [
            case
            for case in suite["tests"]
            if case["mode"] in {"shorter", "compare", "ring"}
            or case.get("parameters", {}).get("cable")
        ]
        candidates = parse_kml_file(suite_path.parent / "data" / "ai_suite.kml")

        for case in route_cases:
            parameters = case.get("parameters", {})
            with self.subTest(test=case["id"]):
                find_shortest_route(
                    candidates,
                    parameters.get("a_end", "Site A"),
                    parameters.get("z_end", "Site B"),
                    excluded_span_id=parameters.get("cable"),
                )

    def test_openrouter_option_is_forwarded_and_reported_as_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            config_path = temporary_root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "provider": "ollama",
                        "prompts_file": str(PROJECT_ROOT / "prompts.yaml"),
                    }
                ),
                encoding="utf-8",
            )
            case = {
                "id": "S001",
                "mode": "shorter",
                "input": "data/ai_suite.kml",
                "parameters": {"a_end": "Site A", "z_end": "Site B"},
            }
            isolated_root = temporary_root / "isolated"
            isolated_root.mkdir()

            command = _case_arguments(
                case,
                PROJECT_ROOT / "tests" / "ai_tests.yaml",
                config_path,
                isolated_root,
                use_openrouter=True,
            )
            provider, _ = _provider_details(config_path, use_openrouter=True)

        self.assertIn("--or", command)
        self.assertEqual(provider, "openrouter")


if __name__ == "__main__":
    unittest.main()
