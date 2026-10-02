import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from includes.cli import _build_parser, _display_prompt_and_results, _load_api_key
from includes.configuration import load_locale
from includes.kml_parser import RouteCandidate
from includes.ollama_client import (
    check_kml_consistency,
    list_ollama_models,
    select_ring_routes,
    select_routes,
)


class OllamaClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            RouteCandidate("R001", "Long route", "A", "B", 8.0, [(-73.0, 45.0), (-72.9, 45.0)]),
            RouteCandidate("R002", "Short route", "A", "B", 2.0, [(-73.0, 45.0), (-72.98, 45.0)]),
        ]
        self.prompts = {
            "common": {"fr": "COMMON ROUTE DEFINITION"},
            "shorter": {"fr": "Choisir la plus courte"},
            "closest": {"fr": "Choisir le manhole le plus proche"},
            "check": {"fr": "Vérifie la cohérence des objets."},
            "ring": {"fr": "Trouver deux itinéraires."},
        }

    def _mock_response(self, route_id: str) -> io.BytesIO:
        response_body = json.dumps(
            {"message": {"content": json.dumps({"routes": [{"route_id": route_id}]})}}
        )
        return io.BytesIO(response_body.encode("utf-8"))

    def _mock_openrouter_response(self, route_id: str) -> io.BytesIO:
        response_body = json.dumps(
            {"choices": [{"message": {"content": json.dumps({"routes": [{"route_id": route_id}]})}}]}
        )
        return io.BytesIO(response_body.encode("utf-8"))

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_shorter_mode_returns_locally_calculated_minimum(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = self._mock_response("R002")

        selected = select_routes(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.1",
            timeout_seconds=10,
            mode="shorter",
            locale="fr",
            prompt_config=self.prompts,
            candidates=self.candidates,
        )

        self.assertEqual([route.route_id for route in selected], ["R002"])

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_ollama_request_metadata_captures_prompt_and_available_tokens(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = io.BytesIO(
            json.dumps(
                {
                    "model": "llama3.1",
                    "message": {"content": json.dumps({"routes": [{"route_id": "R002"}]})},
                    "prompt_eval_count": 14,
                    "eval_count": 6,
                }
            ).encode("utf-8")
        )
        request_metadata: dict[str, object] = {}

        select_routes(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.1",
            timeout_seconds=10,
            mode="shorter",
            locale="fr",
            prompt_config=self.prompts,
            candidates=self.candidates,
            request_metadata=request_metadata,
        )

        messages = json.loads(request_metadata["prompt"])
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(json.loads(messages[1]["content"])["mode"], "shorter")
        self.assertEqual(request_metadata["model"], "llama3.1")
        self.assertEqual(request_metadata["prompt_tokens"], 14)
        self.assertEqual(request_metadata["completion_tokens"], 6)
        self.assertEqual(request_metadata["total_tokens"], 20)

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_consistency_check_returns_only_validated_object_anomalies(self, mock_urlopen: object) -> None:
        anomaly = {
            "type d'objet": "LineString",
            "id d'objet": "Cable-1",
            "incohérence trouvée": "Extrémité sans point correspondant",
            "explication de l'incohérence": "Aucun point du réseau ne correspond à l'extrémité.",
            "solution de résolution possible de l'incohérence": "Raccorder l'extrémité au point correct.",
        }
        mock_urlopen.return_value = io.BytesIO(
            json.dumps(
                {"message": {"content": json.dumps({"inconsistencies": [anomaly]})}}
            ).encode("utf-8")
        )
        objects = [{"object_id": "Cable-1", "type": "LineString", "name": "Cable"}]
        request_metadata: dict[str, object] = {}

        inconsistencies = check_kml_consistency(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.1",
            timeout_seconds=10,
            locale="fr",
            prompt_config=self.prompts,
            objects=objects,
            request_metadata=request_metadata,
        )

        request_body = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertEqual(inconsistencies, [anomaly])
        self.assertEqual(
            json.loads(request_body["messages"][1]["content"])["objects"],
            [{"i": "Cable-1", "t": "L", "n": "Cable"}],
        )
        self.assertIn('"inconsistencies"', request_body["messages"][0]["content"])
        self.assertEqual(request_metadata["model"], "llama3.1")

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_consistency_check_rejects_unknown_object_ids(self, mock_urlopen: object) -> None:
        anomaly = {
            "type d'objet": "Point",
            "id d'objet": "invented-id",
            "incohérence trouvée": "Unknown object",
            "explication de l'incohérence": "No such object exists.",
            "solution de résolution possible de l'incohérence": "Remove it.",
        }
        mock_urlopen.return_value = io.BytesIO(
            json.dumps(
                {"message": {"content": json.dumps({"inconsistencies": [anomaly]})}}
            ).encode("utf-8")
        )

        with self.assertRaisesRegex(ValueError, "unknown object id"):
            check_kml_consistency(
                provider="ollama",
                base_url="http://localhost:11434",
                model="llama3.1",
                timeout_seconds=10,
                locale="fr",
                prompt_config=self.prompts,
                objects=[{"object_id": "known-id", "type": "Point"}],
            )

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_consistency_check_returns_one_row_per_issue_for_same_object(self, mock_urlopen: object) -> None:
        first = {
            "type d'objet": "LineString",
            "id d'objet": "Cable-1",
            "incohérence trouvée": "Issue one",
            "explication de l'incohérence": "Explanation one",
            "solution de résolution possible de l'incohérence": "Fix one",
        }
        second = {
            **first,
            "incohérence trouvée": "Issue two",
            "explication de l'incohérence": "Explanation two",
            "solution de résolution possible de l'incohérence": "",
        }
        mock_urlopen.return_value = io.BytesIO(
            json.dumps(
                {"message": {"content": json.dumps({"inconsistencies": [first, second]})}}
            ).encode("utf-8")
        )

        request_metadata: dict[str, object] = {}
        results = check_kml_consistency(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.1",
            timeout_seconds=10,
            locale="fr",
            prompt_config=self.prompts,
            objects=[{"object_id": "Cable-1", "type": "LineString"}],
            request_metadata=request_metadata,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["incohérence trouvée"], "Issue one")
        self.assertEqual(results[1]["incohérence trouvée"], "Issue two")
        self.assertEqual(results[0]["id d'objet"], results[1]["id d'objet"])
        self.assertEqual(results[0]["solution de résolution possible de l'incohérence"], "Fix one")
        self.assertEqual(request_metadata["potential_solution_count"], 1)

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_shorter_mode_rejects_non_shortest_model_choice(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = self._mock_response("R001")

        with self.assertRaisesRegex(ValueError, "locally calculated shortest route"):
            select_routes(
                provider="ollama",
                base_url="http://localhost:11434",
                model="llama3.1",
                timeout_seconds=10,
                mode="shorter",
                locale="fr",
                prompt_config=self.prompts,
                candidates=self.candidates,
            )

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_closest_mode_sends_address_context_and_verifies_nearest_manhole(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = self._mock_response("R002")
        context = {"address": "10 Main Street", "geocoded_coordinates": {"longitude": -72.99, "latitude": 45}}

        selected = select_routes(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.1",
            timeout_seconds=10,
            mode="closest",
            locale="fr",
            prompt_config=self.prompts,
            candidates=self.candidates,
            context=context,
        )

        request_body = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertEqual(json.loads(request_body["messages"][1]["content"])["context"], context)
        self.assertEqual([route.route_id for route in selected], ["R002"])

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_closest_mode_rejects_non_nearest_ai_choice(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = self._mock_response("R001")

        with self.assertRaisesRegex(ValueError, "locally calculated nearest manhole"):
            select_routes(
                provider="ollama",
                base_url="http://localhost:11434",
                model="llama3.1",
                timeout_seconds=10,
                mode="closest",
                locale="fr",
                prompt_config=self.prompts,
                candidates=self.candidates,
            )

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_openrouter_uses_authenticated_openai_compatible_request(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = self._mock_openrouter_response("R002")
        request_metadata: dict[str, object] = {}

        selected = select_routes(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            timeout_seconds=10,
            mode="shorter",
            locale="fr",
            prompt_config=self.prompts,
            candidates=self.candidates,
            api_key="test-key",
            site_url="https://example.test",
            app_name="KMZ Route Parser",
            request_metadata=request_metadata,
        )

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("Http-referer"), "https://example.test")
        self.assertEqual(request.get_header("X-title"), "KMZ Route Parser")
        request_body = json.loads(request.data)
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertIn("COMMON ROUTE DEFINITION", request_body["messages"][0]["content"])
        self.assertIn("Choisir la plus courte", request_body["messages"][0]["content"])
        self.assertEqual([route.route_id for route in selected], ["R002"])

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_openrouter_request_metadata_captures_available_tokens(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = io.BytesIO(
            json.dumps(
                {
                    "model": "openai/gpt-4o-mini",
                    "usage": {"prompt_tokens": 101, "completion_tokens": 17, "total_tokens": 118},
                    "choices": [
                        {"message": {"content": json.dumps({"routes": [{"route_id": "R002"}]})}}
                    ],
                }
            ).encode("utf-8")
        )
        request_metadata: dict[str, object] = {}

        select_routes(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            timeout_seconds=10,
            mode="shorter",
            locale="fr",
            prompt_config=self.prompts,
            candidates=self.candidates,
            api_key="test-key",
            request_metadata=request_metadata,
        )

        self.assertEqual(request_metadata["prompt_tokens"], 101)
        self.assertEqual(request_metadata["completion_tokens"], 17)
        self.assertEqual(request_metadata["total_tokens"], 118)

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_http_429_includes_provider_message_and_retry_after(self, mock_urlopen: object) -> None:
        response_body = json.dumps({"error": {"message": "free model rate limit reached"}}).encode()
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429,
            "Too Many Requests",
            {"Retry-After": "12"},
            io.BytesIO(response_body),
        )

        with self.assertRaisesRegex(RuntimeError, "HTTP 429.*free model rate limit reached.*Retry-After: 12"):
            select_routes(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                model="qwen/qwen3.8-27b:free",
                timeout_seconds=10,
                mode="shorter",
                locale="fr",
                prompt_config=self.prompts,
                candidates=self.candidates,
                api_key="test-key",
                max_retries=0,
            )

    @patch("includes.ollama_client.time.sleep")
    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_openrouter_retries_429_configured_number_of_times(
        self, mock_urlopen: object, mock_sleep: object
    ) -> None:
        rate_limit_error = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429,
            "Too Many Requests",
            {"Retry-After": "3"},
            io.BytesIO(b"{}"),
        )
        mock_urlopen.side_effect = [rate_limit_error, rate_limit_error, self._mock_openrouter_response("R002")]

        selected = select_routes(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            timeout_seconds=10,
            mode="shorter",
            locale="fr",
            prompt_config=self.prompts,
            candidates=self.candidates,
            api_key="test-key",
            max_retries=2,
            retry_delay_seconds=0.5,
        )

        self.assertEqual(mock_urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in mock_sleep.call_args_list], [3.0, 3.0])
        self.assertEqual([route.route_id for route in selected], ["R002"])

    def test_openrouter_cli_aliases_select_provider(self) -> None:
        parser = _build_parser(load_locale("en"))

        closest = parser.parse_args(["routes.kml", "--closest", "10 Main Street"])
        self.assertEqual(closest.closest, "10 Main Street")
        self.assertIsNone(closest.mode)

        for alias in ("--or", "--openrouter"):
            parsed = parser.parse_args(
                ["routes.kml", "--mode", "shorter", "--a-end", "Point A", "--z-end", "Point B", "--build", alias]
            )
            self.assertEqual(parsed.provider_alias, "openrouter")
            self.assertEqual(parsed.a_end, "Point A")
            self.assertEqual(parsed.z_end, "Point B")
            self.assertTrue(parsed.build)

        diverse = parser.parse_args(
            [
                "routes.kml",
                "--mode",
                "diverse",
                "--span",
                "0001",
                "--build",
            ]
        )
        self.assertEqual(diverse.span, "0001")
        self.assertTrue(diverse.build)

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_lists_models_from_ollama_tags_endpoint(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = io.BytesIO(
            json.dumps(
                {
                    "models": [
                        {"name": "llama3.1:latest"},
                        {"model": "qwen2.5:7b"},
                    ]
                }
            ).encode("utf-8")
        )

        models = list_ollama_models("http://localhost:11434/", 10)

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/tags")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(models, ["llama3.1:latest", "qwen2.5:7b"])

    @patch("includes.ollama_client.urllib.request.urlopen")
    def test_ring_ai_returns_two_ordered_routes(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = io.BytesIO(
            json.dumps(
                {
                    "message": {
                        "content": json.dumps(
                            {"routes": [{"route_ids": ["R001", "R002"]}, {"route_ids": ["R003"]}]}
                        )
                    }
                }
            ).encode("utf-8")
        )
        candidates = [
            RouteCandidate("R001", "Cable 1", "A", "J", 1, [(0, 0), (1, 0)]),
            RouteCandidate("R002", "Cable 2", "J", "B", 1, [(1, 0), (2, 0)]),
            RouteCandidate("R003", "Cable 3", "A", "B", 3, [(0, 0), (2, 0)]),
        ]

        routes = select_ring_routes(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.1",
            timeout_seconds=10,
            locale="fr",
            prompt_config=self.prompts,
            candidates=candidates,
            a_end="A",
            z_end="B",
        )

        self.assertEqual(
            [[route.route_id for route in path] for path in routes],
            [["R001", "R002"], ["R003"]],
        )

    def test_reads_openrouter_api_key_from_config_directory_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            key_file = Path(temporary_directory) / "API_KEY_OPENROUTER"
            key_file.write_text("test-key\n", encoding="utf-8")

            api_key = _load_api_key("API_KEY_OPENROUTER", Path(temporary_directory))

        self.assertEqual(api_key, "test-key")

    def test_rejects_empty_openrouter_api_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            key_file = Path(temporary_directory) / "API_KEY_OPENROUTER"
            key_file.write_text("  \n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "API_KEY_OPENROUTER"):
                _load_api_key("API_KEY_OPENROUTER", Path(temporary_directory))

    def test_displays_prompt_immediately_above_compact_result(self) -> None:
        result_data = [{"Route": "Circuit", "A-END": "A", "Z-END": "Z", "distance": 1.25}]
        system_prompt = "Prompt line one\nPrompt line two"
        output = io.StringIO()

        with redirect_stdout(output):
            _display_prompt_and_results(system_prompt, result_data)

        self.assertEqual(
            output.getvalue(),
            system_prompt + "\n" + json.dumps(result_data, ensure_ascii=False, separators=(",", ":")) + "\n",
        )


if __name__ == "__main__":
    unittest.main()
