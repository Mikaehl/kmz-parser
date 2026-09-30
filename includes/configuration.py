from pathlib import Path
import math
from typing import Any

import yaml


DEFAULT_CONFIG = {
    "provider": "ollama",
    "network": {
        "endpoint_match_km": 0.5,
    },
    "closest": {
        "geocoder_url": "https://nominatim.openstreetmap.org/search",
        "user_agent": "KMZRouteParser/1.0",
        "timeout_seconds": 15,
        "max_distance_km": 10,
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3.1",
        "timeout_seconds": 120,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "api_key_file": "API_KEY_OPENROUTER",
        "site_url": "",
        "app_name": "KMZ Route Parser",
        "timeout_seconds": 120,
        "max_retries": 5,
        "retry_delay_seconds": 2,
    },
    "application": {
        "locale": "fr",
        "output_directory": "output",
        "log_directory": "logs",
        "log_level": "INFO",
    },
    "prompts_file": "prompts.yaml",
}


def _merge_config(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    merged = defaults.copy()
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_settings(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    settings = _merge_config(DEFAULT_CONFIG, load_yaml(config_path))
    application = settings["application"]
    if application["locale"] not in {"fr", "en"}:
        raise ValueError("application.locale must be 'fr' or 'en'")
    endpoint_match_km = float(settings["network"]["endpoint_match_km"])
    if not math.isfinite(endpoint_match_km) or endpoint_match_km < 0:
        raise ValueError("network.endpoint_match_km must be a finite non-negative number")
    settings["network"]["endpoint_match_km"] = endpoint_match_km
    closest_max_distance_km = float(settings["closest"]["max_distance_km"])
    if not math.isfinite(closest_max_distance_km) or closest_max_distance_km < 0:
        raise ValueError("closest.max_distance_km must be a finite non-negative number")
    settings["closest"]["max_distance_km"] = closest_max_distance_km
    settings["closest"]["timeout_seconds"] = int(settings["closest"]["timeout_seconds"])
    settings["ollama"]["timeout_seconds"] = int(settings["ollama"]["timeout_seconds"])
    return settings, config_path.parent


def load_locale(locale: str) -> dict[str, str]:
    locale_path = Path(__file__).resolve().parent.parent / "locales" / f"{locale}.yaml"
    return load_yaml(locale_path)


def load_prompts(settings: dict[str, Any], base_directory: Path) -> dict[str, Any]:
    prompts_path = (base_directory / settings["prompts_file"]).resolve()
    return load_yaml(prompts_path)
