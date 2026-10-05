import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from includes.configuration import load_settings


MODES = {"shorter", "ai-route", "compare", "diverse", "ring", "analysis"}
PARAMETER_FLAGS = {
    "a_end": "--a-end",
    "z_end": "--z-end",
    "cable": "--cable",
    "span": "--span",
    "locale": "--locale",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    if not isinstance(data, dict):
        raise ValueError(f"Le fichier {path} doit contenir un objet YAML.")
    return data


def _validate_suite(suite: dict[str, Any]) -> list[dict[str, Any]]:
    cases = suite.get("tests")
    if not isinstance(cases, list) or not cases:
        raise ValueError("La suite YAML doit contenir une liste 'tests' non vide.")
    identifiers: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Le test numéro {index} doit être un objet YAML.")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Le test numéro {index} doit avoir un identifiant.")
        if identifier in identifiers:
            raise ValueError(f"Identifiant de test dupliqué : {identifier}")
        identifiers.add(identifier)
        if case.get("mode") not in MODES:
            raise ValueError(f"Mode invalide pour le test {identifier}: {case.get('mode')}")
        if not isinstance(case.get("input"), str) or not case["input"].strip():
            raise ValueError(f"Le test {identifier} doit définir un fichier 'input'.")
        parameters = case.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"Les paramètres du test {identifier} doivent être un objet YAML.")
        unknown_parameters = set(parameters) - (set(PARAMETER_FLAGS) | {"build"})
        if unknown_parameters:
            raise ValueError(
                f"Paramètre(s) inconnu(s) pour le test {identifier}: "
                f"{', '.join(sorted(unknown_parameters))}"
            )
        if "build" in parameters and not isinstance(parameters["build"], bool):
            raise ValueError(f"Le paramètre 'build' du test {identifier} doit être booléen.")
    return cases


def _provider_details(config_path: Path, use_openrouter: bool = False) -> tuple[str, str]:
    settings, _ = load_settings(config_path)
    provider = "openrouter" if use_openrouter else settings["provider"]
    if provider == "or":
        provider = "openrouter"
    if provider not in {"ollama", "openrouter"}:
        raise ValueError("Le fournisseur configuré doit être ollama ou openrouter.")
    return provider, str(settings[provider]["model"])


def _case_arguments(
    case: dict[str, Any],
    suite_path: Path,
    config_path: Path,
    temporary_root: Path,
    use_openrouter: bool = False,
) -> list[str]:
    input_path = (suite_path.parent / case["input"]).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Fichier d'entrée introuvable pour {case['id']}: {input_path}")

    parameters = case.get("parameters", {})
    isolated_config = _load_yaml(config_path)
    application = isolated_config.setdefault("application", {})
    output_directory = temporary_root / "output"
    log_directory = temporary_root / "logs"
    application["output_directory"] = str(output_directory)
    application["log_directory"] = str(log_directory)

    config_directory = config_path.resolve().parent
    prompts_path = (config_directory / isolated_config["prompts_file"]).resolve()
    isolated_config["prompts_file"] = str(prompts_path)
    openrouter = isolated_config.get("openrouter", {})
    if isinstance(openrouter, dict) and openrouter.get("api_key_file"):
        openrouter["api_key_file"] = str(
            (config_directory / openrouter["api_key_file"]).resolve()
        )
    isolated_config_path = temporary_root / "config.yaml"
    isolated_config_path.write_text(
        yaml.safe_dump(isolated_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    command = [sys.executable, str(PROJECT_ROOT / "kmz_parser.py"), str(input_path)]
    if case["mode"] == "analysis":
        command.append("--check")
    else:
        command.extend(["--mode", case["mode"]])
    for parameter, flag in PARAMETER_FLAGS.items():
        if parameter in parameters:
            command.extend([flag, str(parameters[parameter])])
    if parameters.get("build"):
        command.append("--build")
    command.extend(["--config", str(isolated_config_path)])
    if use_openrouter:
        command.append("--or")
    return command


def _run_case(
    case: dict[str, Any],
    suite_path: Path,
    config_path: Path,
    timeout_seconds: int,
    use_openrouter: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kmz-parser-ai-test-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        command = _case_arguments(
            case,
            suite_path,
            config_path,
            temporary_root,
            use_openrouter,
        )
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=timeout_seconds,
                check=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            database_path = temporary_root / "logs" / "requests.sqlite3"
            if not database_path.is_file():
                raise RuntimeError(
                    f"kmz-parser n'a pas créé sa base de requêtes "
                    f"(code de sortie {process.returncode})."
                )
            with closing(sqlite3.connect(database_path)) as connection:
                row = connection.execute(
                    """
                    SELECT status, result_json, duration_ms, provider, model,
                           prompt_tokens, completion_tokens, total_tokens, error
                    FROM requests ORDER BY id DESC LIMIT 1
                    """
                ).fetchone()
            if row is None:
                raise RuntimeError("kmz-parser n'a enregistré aucune requête.")
            result = json.loads(row[1]) if row[1] is not None else None
            error = row[8]
            if process.returncode != 0 and not error:
                error = process.stderr.strip() or f"code de sortie {process.returncode}"
            return {
                "exit_code": process.returncode,
                "status": row[0],
                "result": result,
                "duration_ms": round(elapsed_ms, 3),
                "program_duration_ms": row[2],
                "provider": row[3],
                "model": row[4],
                "prompt_tokens": row[5],
                "completion_tokens": row[6],
                "total_tokens": row[7],
                "error": error,
            }
        except subprocess.TimeoutExpired as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            raise RuntimeError(
                f"Délai maximal de {timeout_seconds}s dépassé après {elapsed_ms / 1000:.1f}s."
            ) from error


def _write_expected(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        yaml.safe_dump(
            {"results": results},
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _same_result(expected: Any, actual: Any) -> bool:
    return expected == actual


def _score(
    passed: int,
    total: int,
    elapsed_ms: float,
    token_total: int | None,
    expected: dict[str, Any],
    cases: list[dict[str, Any]],
    initializing: bool,
) -> tuple[float, float | None]:
    result_score = 100 * passed / total if total else 0.0
    all_expected = all(case["id"] in expected for case in cases)
    baseline_durations = [
        expected[case["id"]].get("duration_ms")
        for case in cases
        if case["id"] in expected
    ]
    baseline_tokens = [
        expected[case["id"]].get("total_tokens")
        for case in cases
        if case["id"] in expected
    ]
    if initializing:
        baseline_durations = []
        baseline_tokens = []
        all_expected = False

    if initializing:
        efficiency_score = (
            50 * (passed / total if total else 0) + 50
            if token_total is not None
            else None
        )
    elif (
        not all_expected
        or len(baseline_durations) != total
        or len(baseline_tokens) != total
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in baseline_durations)
        or any(not isinstance(value, int) or value < 0 for value in baseline_tokens)
        or token_total is None
    ):
        efficiency_score = None
    else:
        baseline_time = sum(baseline_durations)
        baseline_token_total = sum(baseline_tokens)
        time_efficiency = (
            1.0 if elapsed_ms == 0 else min(1.0, baseline_time / elapsed_ms)
        )
        token_efficiency = (
            1.0
            if token_total == 0
            else min(1.0, baseline_token_total / token_total)
        )
        efficiency_score = (
            50 * (passed / total if total else 0)
            + 25 * time_efficiency
            + 25 * token_efficiency
        )
    return result_score, efficiency_score


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Exécute et compare la suite YAML de tests IA de kmz-parser."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=PROJECT_ROOT / "tests" / "ai_tests.yaml",
        help="Fichier YAML contenant les cas de test.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=PROJECT_ROOT / "tests" / "ai_expected.yaml",
        help="Fichier YAML des résultats de référence.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Configuration kmz-parser (fournisseur et modèle IA).",
    )
    parser.add_argument(
        "--or",
        dest="use_openrouter",
        action="store_true",
        help="Force OpenRouter et transmet --or à chaque appel de kmz-parser.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=660,
        help="Délai maximal par test (défaut : 660 secondes).",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Exécute les tests et initialise leurs résultats de référence.",
    )
    parser.add_argument(
        "--test",
        metavar="ID",
        help="Exécute uniquement le test ayant cet identifiant.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Affiche les tests sans lancer kmz-parser.",
    )
    arguments = parser.parse_args()
    try:
        suite_path = arguments.suite.resolve()
        config_path = arguments.config.resolve()
        suite = _load_yaml(suite_path)
        cases = _validate_suite(suite)
        if arguments.test is not None:
            cases = [case for case in cases if case["id"] == arguments.test]
            if not cases:
                raise ValueError(f"Identifiant de test introuvable : {arguments.test}")
        if arguments.timeout_seconds < 1:
            raise ValueError("--timeout-seconds doit être supérieur à zéro.")
        if arguments.list:
            for case in cases:
                print(f"{case['id']}: {case['mode']} — {case.get('description', '')}")
            print(f"{len(cases)} tests.")
            return 0
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration introuvable : {config_path}")
        provider, model = _provider_details(config_path, arguments.use_openrouter)
        expected_path = arguments.expected.resolve()
        if arguments.init:
            if arguments.test is not None and expected_path.is_file():
                expected_data = _load_yaml(expected_path)
                expected_results = expected_data.get("results", {})
                if not isinstance(expected_results, dict):
                    raise ValueError("Le fichier de références doit contenir un objet 'results'.")
            else:
                expected_results = {}
                if arguments.test is None:
                    _write_expected(expected_path, expected_results)
        else:
            expected_data = _load_yaml(expected_path)
            expected_results = expected_data.get("results", {})
            if not isinstance(expected_results, dict):
                raise ValueError("Le fichier de références doit contenir un objet 'results'.")
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        parser.error(str(error))

    modes = ", ".join(sorted({case["mode"] for case in cases}))
    print(f"IA : {provider} | Modèle : {model} | Modes : {modes}")
    if arguments.init:
        print(f"Initialisation des résultats dans {expected_path}")

    passed = 0
    executed = 0
    elapsed_total_ms = 0.0
    tokens_collected = 0
    token_test_count = 0
    failed = False
    for case in cases:
        started = time.perf_counter()
        try:
            actual = _run_case(
                case,
                suite_path,
                config_path,
                arguments.timeout_seconds,
                arguments.use_openrouter,
            )
        except (FileNotFoundError, OSError, RuntimeError, sqlite3.Error, ValueError) as error:
            actual = {
                "exit_code": 1,
                "status": "error",
                "result": None,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "total_tokens": None,
                "error": str(error),
            }
        executed += 1
        elapsed_total_ms += actual["duration_ms"]
        if isinstance(actual.get("total_tokens"), int):
            tokens_collected += actual["total_tokens"]
            token_test_count += 1

        successful = actual["exit_code"] == 0 and actual["status"] == "success"
        if arguments.init:
            conforms = successful
            if conforms:
                expected_results[case["id"]] = {
                    "exit_code": actual["exit_code"],
                    "status": actual["status"],
                    "result": actual["result"],
                    "duration_ms": actual["duration_ms"],
                    "total_tokens": actual["total_tokens"],
                }
                _write_expected(expected_path, expected_results)
        else:
            reference = expected_results.get(case["id"])
            conforms = (
                successful
                and isinstance(reference, dict)
                and reference.get("exit_code") == actual["exit_code"]
                and reference.get("status") == actual["status"]
                and _same_result(reference.get("result"), actual["result"])
            )
        if conforms:
            passed += 1
        else:
            failed = True

        duration = actual["duration_ms"] / 1000
        status_label = "CONFORME" if conforms else "NON CONFORME"
        details = ""
        if not conforms:
            details = actual.get("error") or (
                "référence absente; lancez avec --init"
                if not arguments.init and case["id"] not in expected_results
                else "le résultat diffère de la référence"
            )
            details = f" — {str(details).replace(chr(10), ' ')[:180]}"
        print(f"{case['id']:<8} {duration:>8.2f}s  {status_label}{details}", flush=True)
        if arguments.init and conforms and actual["provider"] and actual["model"]:
            provider, model = actual["provider"], actual["model"]

    token_total = tokens_collected if token_test_count == executed else None
    result_score, efficiency_score = _score(
        passed,
        len(cases),
        elapsed_total_ms,
        token_total,
        expected_results,
        cases,
        arguments.init,
    )
    print(
        f"\nTests conformes : {passed}/{len(cases)} "
        f"({result_score:.1f}/100) | Temps total : {elapsed_total_ms / 1000:.2f}s"
    )
    token_label = (
        f"{tokens_collected} jetons"
        if token_total is not None
        else f"indisponible ({token_test_count}/{executed} tests avec métrique)"
    )
    print(f"Total de jetons : {token_label}")
    if efficiency_score is None:
        print("Note performance : non disponible (références temps/jetons complètes requises)")
    else:
        print(
            "Note performance : "
            f"{efficiency_score:.1f}/100 "
            "(50 % conformité, 25 % temps vs référence, 25 % jetons vs référence)"
        )
    if arguments.init:
            initialized = sum(case["id"] in expected_results for case in cases)
            print(f"Références initialisées : {initialized}/{len(cases)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
