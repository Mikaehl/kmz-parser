import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from includes.kml_parser import RouteCandidate


class ProviderConfigurationError(ValueError):
    pass


def _describe_http_error(error: urllib.error.HTTPError) -> str:
    description = f"HTTP {error.code} {error.reason}"
    try:
        response_body = error.read().decode("utf-8", errors="replace").strip()
    except OSError:
        response_body = ""
    if response_body:
        try:
            error_data = json.loads(response_body).get("error", response_body)
            if isinstance(error_data, dict):
                response_body = str(error_data.get("message") or json.dumps(error_data, ensure_ascii=False))
            else:
                response_body = str(error_data)
        except (AttributeError, json.JSONDecodeError):
            pass
        description += f": {response_body.replace(chr(10), ' ').replace(chr(13), ' ')[:1000]}"
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        description += f" (Retry-After: {retry_after})"
    return description


def _retry_delay(error: urllib.error.HTTPError, retry_index: int, base_delay: float) -> float:
    delay = min(base_delay * (2**retry_index), 60.0)
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                delay = max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return min(max(delay, 0.0), 60.0)


def _token_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def list_ollama_models(base_url: str, timeout_seconds: int) -> list[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(str(error)) from error

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("Ollama response must contain a models array")
    return [
        name
        for model_info in models
        if isinstance(model_info, dict)
        and isinstance(name := (model_info.get("name") or model_info.get("model")), str)
    ]


def build_system_prompt(prompt_config: dict[str, Any], mode: str, locale: str) -> str:
    try:
        common_prompt = prompt_config["common"][locale]
        mode_prompt = prompt_config[mode][locale]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing common or mode prompt for '{mode}' and locale '{locale}'") from error

    return (
        f"{common_prompt}\n\n{mode_prompt}\n"
        "You are selecting route records from a supplied candidate list. Use only route_id values "
        "from that list. Return valid JSON only, with this exact shape: "
        '{"routes":[{"route_id":"R001"}]}. Do not include markdown or additional keys. '
        "The distance and endpoints will be filled from the source data by the calling program."
    )


def build_consistency_prompt(prompt_config: dict[str, Any], locale: str) -> str:
    try:
        instructions = prompt_config["check"][locale]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing consistency-check prompt for locale '{locale}'") from error
    return (
        f"{instructions}\n"
        "Return valid JSON only, with this exact shape: "
        '{"inconsistencies":[{"type d\'objet":"...","id d\'objet":"...",'
        '"incohérence trouvée":"...","explication de l\'incohérence":"...",'
        '"solution de résolution possible de l\'incohérence":"..."}]}. '
        "Return an empty inconsistencies array when no issue is found. "
        "The inventory uses compact keys: i=exact object ID, t=type (P=Point, L=LineString), "
        "n=name when different from i, f=folder, s=style, d=extended data, and g=geometry. "
        "A Point geometry is [longitude,latitude]. A LineString geometry is "
        "[vertex_count,[start_lon,start_lat],[end_lon,end_lat]]. "
        "Use only i values present in the supplied inventory for id d'objet. "
        "If no plausible fix is identifiable, return an empty string for the solution field."
    )


def check_kml_consistency(
    provider: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
    locale: str,
    prompt_config: dict[str, Any],
    objects: list[dict[str, object]],
    api_key: str | None = None,
    site_url: str = "",
    app_name: str = "",
    max_retries: int = 5,
    retry_delay_seconds: float = 2,
    system_prompt: str | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    instructions = system_prompt or build_consistency_prompt(prompt_config, locale)
    compact_objects: list[dict[str, object]] = []
    for item in objects:
        object_id = str(item.get("object_id", ""))
        object_type = str(item.get("type", "Placemark"))
        compact: dict[str, object] = {
            "i": object_id,
            "t": {"Point": "P", "LineString": "L"}.get(object_type, object_type),
        }
        name = item.get("name")
        if name and str(name) != object_id.split("#", 1)[0]:
            compact["n"] = name
        folders = item.get("folders")
        if isinstance(folders, list) and folders:
            compact["f"] = folders[-1]
        if item.get("style_url"):
            compact["s"] = str(item["style_url"]).rsplit("_", 1)[-1]
        if item.get("extended_data"):
            compact["d"] = item["extended_data"]
        geometries = item.get("geometries")
        if isinstance(geometries, list) and geometries:
            compact_geometries: list[object] = []
            for geometry in geometries:
                if not isinstance(geometry, dict):
                    continue
                if "coordinates" in geometry:
                    compact_geometries.append(
                        [round(float(coordinate), 6) for coordinate in geometry["coordinates"]]
                    )
                elif "start" in geometry and "end" in geometry:
                    compact_geometries.append(
                        [
                            geometry.get("coordinate_count"),
                            [round(float(coordinate), 6) for coordinate in geometry["start"]],
                            [round(float(coordinate), 6) for coordinate in geometry["end"]],
                        ]
                    )
                elif "sample" in geometry:
                    compact_geometries.append(
                        [
                            [round(float(coordinate), 6) for coordinate in point]
                            for point in geometry["sample"]
                        ]
                    )
            if compact_geometries:
                compact["g"] = compact_geometries
        compact_objects.append(compact)

    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": json.dumps(
                {"objects": compact_objects},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    if request_metadata is not None:
        request_metadata.update(
            {
                "prompt": json.dumps(messages, ensure_ascii=False),
                "provider": provider,
                "model": model,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
        )

    headers = {"Content-Type": "application/json"}
    if provider == "ollama":
        endpoint = f"{base_url.rstrip('/')}/api/chat"
        request_body = {"model": model, "stream": False, "format": "json", "messages": messages}
    elif provider == "openrouter":
        if not api_key:
            raise ProviderConfigurationError("OpenRouter API key is required")
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        request_body = {
            "model": model,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    retry_index = 0
    try:
        while True:
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    api_response = json.loads(response.read().decode("utf-8"))
                if request_metadata is not None:
                    request_metadata["model"] = api_response.get("model") or model
                    if provider == "ollama":
                        prompt_tokens = _token_count(api_response.get("prompt_eval_count"))
                        completion_tokens = _token_count(api_response.get("eval_count"))
                        total_tokens = (
                            prompt_tokens + completion_tokens
                            if prompt_tokens is not None and completion_tokens is not None
                            else None
                        )
                    else:
                        usage = api_response.get("usage")
                        usage = usage if isinstance(usage, dict) else {}
                        prompt_tokens = _token_count(usage.get("prompt_tokens"))
                        completion_tokens = _token_count(usage.get("completion_tokens"))
                        total_tokens = _token_count(usage.get("total_tokens"))
                    request_metadata.update(
                        {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                    )
                break
            except urllib.error.HTTPError as error:
                if provider != "openrouter" or error.code != 429 or retry_index >= max_retries:
                    raise RuntimeError(_describe_http_error(error)) from error
                delay = _retry_delay(error, retry_index, retry_delay_seconds)
                error.close()
                time.sleep(delay)
                retry_index += 1
        if provider == "ollama":
            content = api_response["message"]["content"]
        else:
            content = api_response["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(str(error)) from error

    inconsistencies = result.get("inconsistencies") if isinstance(result, dict) else None
    if not isinstance(inconsistencies, list):
        raise ValueError("AI response must contain an inconsistencies array")
    valid_object_ids = {str(item["object_id"]) for item in objects if "object_id" in item}
    required_fields = (
        "type d'objet",
        "id d'objet",
        "incohérence trouvée",
        "explication de l'incohérence",
        "solution de résolution possible de l'incohérence",
    )
    normalized: list[dict[str, str]] = []
    potential_solution_count = 0
    for item in inconsistencies:
        if not isinstance(item, dict) or any(not isinstance(item.get(field), str) for field in required_fields):
            raise ValueError("Each inconsistency must contain all required text fields")
        normalized_item = {field: item[field].strip() for field in required_fields}
        if any(not normalized_item[field] for field in required_fields[:-1]):
            raise ValueError("Inconsistency description fields cannot be empty")
        if normalized_item[required_fields[-1]]:
            potential_solution_count += 1
        object_id = normalized_item["id d'objet"]
        if object_id not in valid_object_ids:
            raise ValueError(f"AI response refers to unknown object id '{object_id}'")
        normalized.append(normalized_item)
    if request_metadata is not None:
        request_metadata["potential_solution_count"] = potential_solution_count
    return normalized


def select_routes(
    provider: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
    mode: str,
    locale: str,
    prompt_config: dict[str, Any],
    candidates: list[RouteCandidate],
    api_key: str | None = None,
    site_url: str = "",
    app_name: str = "",
    max_retries: int = 5,
    retry_delay_seconds: float = 2,
    system_prompt: str | None = None,
    context: dict[str, Any] | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> list[RouteCandidate]:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    instructions = system_prompt or build_system_prompt(prompt_config, mode, locale)
    user_payload: dict[str, Any] = {
        "mode": mode,
        "candidates": [candidate.to_model_dict() for candidate in candidates],
    }
    if context is not None:
        user_payload["context"] = context
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]
    if request_metadata is not None:
        request_metadata.update(
            {
                "prompt": json.dumps(messages, ensure_ascii=False),
                "provider": provider,
                "model": model,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
        )
    headers = {"Content-Type": "application/json"}
    if provider == "ollama":
        endpoint = f"{base_url.rstrip('/')}/api/chat"
        request_body = {"model": model, "stream": False, "format": "json", "messages": messages}
    elif provider == "openrouter":
        if not api_key:
            raise ProviderConfigurationError("OpenRouter API key is required")
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        request_body = {
            "model": model,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    retry_index = 0
    try:
        while True:
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    api_response = json.loads(response.read().decode("utf-8"))
                if request_metadata is not None:
                    request_metadata["model"] = api_response.get("model") or model
                    if provider == "ollama":
                        prompt_tokens = _token_count(api_response.get("prompt_eval_count"))
                        completion_tokens = _token_count(api_response.get("eval_count"))
                        total_tokens = (
                            prompt_tokens + completion_tokens
                            if prompt_tokens is not None and completion_tokens is not None
                            else None
                        )
                    else:
                        usage = api_response.get("usage")
                        usage = usage if isinstance(usage, dict) else {}
                        prompt_tokens = _token_count(usage.get("prompt_tokens"))
                        completion_tokens = _token_count(usage.get("completion_tokens"))
                        total_tokens = _token_count(usage.get("total_tokens"))
                    request_metadata.update(
                        {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                    )
                break
            except urllib.error.HTTPError as error:
                if provider != "openrouter" or error.code != 429 or retry_index >= max_retries:
                    raise RuntimeError(_describe_http_error(error)) from error
                delay = _retry_delay(error, retry_index, retry_delay_seconds)
                error.close()
                time.sleep(delay)
                retry_index += 1
        if provider == "ollama":
            content = api_response["message"]["content"]
        else:
            content = api_response["choices"][0]["message"]["content"]
        selection = json.loads(content)
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(str(error)) from error

    route_ids = selection.get("routes") if isinstance(selection, dict) else None
    if not isinstance(route_ids, list) or any(not isinstance(item, dict) for item in route_ids):
        raise ValueError("API response must contain a routes array")
    identifiers = [item.get("route_id") for item in route_ids]
    candidate_by_id = {candidate.route_id: candidate for candidate in candidates}
    if not identifiers or any(identifier not in candidate_by_id for identifier in identifiers):
        raise ValueError("API selected unknown or no route identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Ollama selected a route more than once")

    if mode == "distance":
        expected_ids = {candidate.route_id for candidate in candidates}
        if set(identifiers) != expected_ids:
            raise ValueError("Distance mode must return every candidate route")
    elif mode == "shorter":
        shortest = min(candidates, key=lambda candidate: candidate.distance_km)
        if len(identifiers) != 1 or identifiers[0] != shortest.route_id:
            raise ValueError("Shorter mode must return the locally calculated shortest route")
    elif mode == "closest":
        closest = min(candidates, key=lambda candidate: candidate.distance_km)
        if len(identifiers) != 1 or identifiers[0] != closest.route_id:
            raise ValueError("Closest mode must return the locally calculated nearest manhole")
    elif mode == "diverse" and len(identifiers) != min(2, len(candidates)):
        raise ValueError("Diverse mode must return two routes, or all routes if fewer exist")

    return [candidate_by_id[identifier] for identifier in identifiers]
