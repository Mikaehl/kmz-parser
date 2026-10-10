import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from includes.kml_parser import RouteCandidate
from includes.route_planner import (
    RoutePlanningError,
    exclude_crossing_spans,
    exclude_reference_spans,
    find_shortest_route,
)


class ProviderConfigurationError(ValueError):
    pass


class AiSelectionError(ValueError):
    """The provider answered, but the answer cannot be used as a route selection."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}

    def explanation(self) -> str:
        """Human readable diagnostics: what the AI returned and why it was rejected."""
        lines: list[str] = []
        reason = self.detail.get("reason")
        if reason:
            lines.append(f"Reason: {reason}")
        response = self.detail.get("response")
        if response is not None:
            lines.append(f"AI returned: {_response_preview(response)}")
        tool_arguments = self.detail.get("tool_arguments")
        if tool_arguments is not None:
            lines.append(f"Tool call arguments: {_compact_text(tool_arguments)}")
        references = self.detail.get("valid_route_references") or []
        lines.append(f"Valid route references: {_reference_preview(sorted(references))}")
        tool_errors = self.detail.get("tool_errors") or []
        if tool_errors:
            lines.append(
                "Dijkstra tool failures: "
                + "; ".join(_compact_text(error) for error in tool_errors)
            )
        return "\n".join(lines)


def _compact_text(value: Any) -> str:
    return " ".join(str(value).split())


def _response_preview(content: Any, limit: int = 400) -> str:
    if content is None:
        return "<empty>"
    compact = _compact_text(content)
    if not compact:
        return "<empty>"
    if len(compact) > limit:
        return f"{compact[:limit]}... [{len(compact)} characters total]"
    return compact


def _reference_preview(references: list[str], limit: int = 10) -> str:
    if not references:
        return "none (the AI never received a successful Dijkstra route)"
    shown = references[:limit]
    suffix = f"... [{len(references)} references total]" if len(references) > limit else ""
    return ", ".join(shown) + suffix


def _record_rejected_response(
    request_metadata: dict[str, Any] | None,
    logger: logging.Logger | None,
    data_logger: logging.Logger | None,
    reason: str,
    content: Any,
    discovered_routes: dict[str, Any] | None = None,
) -> None:
    """Persist the offending provider payload so it can be inspected in logs and history."""
    if request_metadata is not None:
        request_metadata["ai_response"] = content if isinstance(content, str) else str(content)
        request_metadata["ai_response_reason"] = reason
    if logger is not None:
        logger.error(
            "Rejected AI response (%s): %s",
            reason,
            _response_preview(content),
        )
        if discovered_routes:
            logger.error(
                "Valid route references for this request: %s",
                _reference_preview(sorted(discovered_routes)),
            )
    if data_logger is not None:
        data_logger.info(
            "Rejected AI response",
            extra={
                "data": {
                    "reason": reason,
                    "response": content,
                    "valid_route_references": sorted(discovered_routes or {}),
                }
            },
        )


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


def build_ring_system_prompt(prompt_config: dict[str, Any], locale: str) -> str:
    try:
        instructions = prompt_config["ring"][locale]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing ring prompt for locale '{locale}'") from error
    return (
        f"{instructions}\n"
        "Return valid JSON only, with this exact shape: "
        '{"routes":[{"route_ids":["R001","R002"]},{"route_ids":["R003"]}]}. '
        "Return exactly two routes. Each route is an ordered sequence of route_id values "
        "from the supplied candidate list. Do not invent IDs."
    )


def build_ai_ring_system_prompt(prompt_config: dict[str, Any], locale: str) -> str:
    try:
        common_prompt = prompt_config["common"][locale]
        instructions = prompt_config["ring"][locale]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing common or ring prompt for locale '{locale}'") from error
    return (
        f"{common_prompt}\n\n{instructions}\n"
        "The program requests the two ring legs in separate steps. For each step, use "
        "the find_dijkstra_route tool to calculate a continuous route, and you may call "
        "it again with different excluded segments and/or points. In the second step, "
        "the first route's segments and any segments crossing its geometry are already "
        "removed. Select exactly one route returned by the tool as the route for the "
        "current step. Return valid JSON only, with this exact shape: "
        '{"route_reference":"<route_id returned by the tool>"}. '
        "Do not invent a route reference."
    )


def build_ai_route_system_prompt(prompt_config: dict[str, Any], locale: str) -> str:
    try:
        common_prompt = prompt_config["common"][locale]
        instructions = prompt_config["ai-route"][locale]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing common or AI route prompt for locale '{locale}'") from error
    return (
        f"{common_prompt}\n\n{instructions}\n"
        "Use the find_dijkstra_route tool to calculate candidate routes. You may call it "
        "again with different excluded segments and/or points. Select exactly one route "
        "returned by the tool as your reference. Return valid JSON only, with this exact "
        'shape: {"route_reference":"<route_id returned by the tool>"}. '
        "Do not invent a route reference."
    )


def select_ai_route(
    provider: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
    locale: str,
    prompt_config: dict[str, Any],
    candidates: list[RouteCandidate],
    a_end: str,
    z_end: str,
    api_key: str | None = None,
    site_url: str = "",
    app_name: str = "",
    max_retries: int = 5,
    retry_delay_seconds: float = 2,
    max_dijkstra_tool_calls: int = 8,
    dijkstra_logger: logging.Logger | None = None,
    logger: logging.Logger | None = None,
    data_logger: logging.Logger | None = None,
    system_prompt: str | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> RouteCandidate:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if max_dijkstra_tool_calls < 1:
        raise ValueError("max_dijkstra_tool_calls must be a positive integer")
    instructions = system_prompt or build_ai_route_system_prompt(prompt_config, locale)
    available_segments = [
        {
            "span_id": span.source_id or span.route_id,
            "route_id": span.route_id,
            "name": span.route_name,
            "a_end": span.a_end,
            "z_end": span.z_end,
        }
        for span in candidates
    ]
    available_points = sorted(
        {
            endpoint.strip()
            for span in candidates
            for endpoint in (span.a_end, span.z_end)
        },
        key=str.casefold,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "a_end": a_end,
                    "z_end": z_end,
                    "segments": available_segments,
                    "points": available_points,
                },
                ensure_ascii=False,
            ),
        },
    ]
    tool_description = {
        "name": "find_dijkstra_route",
        "description": (
            "Calculate the shortest cable route between the requested endpoints. "
            "Exclude any listed cable segments (by span ID, route ID, or exact name) "
            "and/or intermediate points (by exact point name)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "exclude_segments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Cable span IDs, route IDs, or exact names to avoid.",
                },
                "exclude_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Intermediate point names that the route must not pass through.",
                },
            },
            "additionalProperties": False,
        },
    }
    tools = [{"type": "function", "function": tool_description}]
    headers = {"Content-Type": "application/json"}
    if provider == "ollama":
        endpoint = f"{base_url.rstrip('/')}/api/chat"
        headers["Accept"] = "application/json"
    elif provider == "openrouter":
        if not api_key:
            raise ProviderConfigurationError("OpenRouter API key is required")
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    discovered_routes: dict[str, RouteCandidate] = {}
    tool_errors: list[str] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    prompt_tokens_known = True
    completion_tokens_known = True
    tool_call_count = 0
    while True:
        if request_metadata is not None:
            request_metadata.update(
                {
                    "prompt": json.dumps(messages, ensure_ascii=False),
                    "provider": provider,
                    "model": model,
                }
            )
        request_body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": messages,
            "tools": tools,
        }
        if provider == "openrouter":
            request_body["tool_choice"] = "auto"
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
                    break
                except urllib.error.HTTPError as error:
                    if provider != "openrouter" or error.code != 429 or retry_index >= max_retries:
                        raise RuntimeError(_describe_http_error(error)) from error
                    delay = _retry_delay(error, retry_index, retry_delay_seconds)
                    error.close()
                    time.sleep(delay)
                    retry_index += 1
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(str(error)) from error

        if not isinstance(api_response, dict):
            raise ValueError("AI provider returned an invalid tool response")
        if provider == "ollama":
            assistant_message = api_response.get("message")
            prompt_count = _token_count(api_response.get("prompt_eval_count"))
            completion_count = _token_count(api_response.get("eval_count"))
        else:
            choices = api_response.get("choices")
            assistant_message = (
                choices[0].get("message")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict)
                else None
            )
            usage = api_response.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            prompt_count = _token_count(usage.get("prompt_tokens"))
            completion_count = _token_count(usage.get("completion_tokens"))
        if prompt_count is None:
            prompt_tokens_known = False
        else:
            prompt_tokens_total += prompt_count
        if completion_count is None:
            completion_tokens_known = False
        else:
            completion_tokens_total += completion_count
        if request_metadata is not None:
            request_metadata["model"] = api_response.get("model") or model
            request_metadata["prompt_tokens"] = (
                prompt_tokens_total if prompt_tokens_known else None
            )
            request_metadata["completion_tokens"] = (
                completion_tokens_total if completion_tokens_known else None
            )
            request_metadata["total_tokens"] = (
                prompt_tokens_total + completion_tokens_total
                if prompt_tokens_known and completion_tokens_known
                else None
            )
        if not isinstance(assistant_message, dict):
            raise ValueError("AI provider response did not contain an assistant message")
        tool_calls = assistant_message.get("tool_calls")
        if tool_calls:
            if not isinstance(tool_calls, list):
                raise ValueError("AI provider returned invalid tool calls")
            messages.append({"role": "assistant", **assistant_message})
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    raise ValueError("AI provider returned an invalid tool call")
                function = tool_call.get("function")
                if not isinstance(function, dict) or function.get("name") != "find_dijkstra_route":
                    raise ValueError("AI provider requested an unsupported tool")
                if tool_call_count >= max_dijkstra_tool_calls:
                    detail = {
                        "reason": "tool_call_limit",
                        "tool_calls_attempted": tool_call_count + 1,
                        "max_dijkstra_tool_calls": max_dijkstra_tool_calls,
                        "valid_route_references": sorted(discovered_routes),
                        "tool_arguments": function.get("arguments", {}),
                        "tool_errors": tool_errors,
                    }
                    if request_metadata is not None:
                        request_metadata["ai_response_reason"] = "tool_call_limit"
                    if logger is not None:
                        logger.error(
                            "Rejected AI response (tool_call_limit): the AI requested a "
                            "Dijkstra tool call beyond the configured limit of %s "
                            "(call arguments: %s). Valid route references: %s",
                            max_dijkstra_tool_calls,
                            _compact_text(function.get("arguments", {})),
                            _reference_preview(sorted(discovered_routes)),
                        )
                    if data_logger is not None:
                        data_logger.info("Rejected AI response", extra={"data": detail})
                    raise AiSelectionError(
                        "AI exceeded the maximum of "
                        f"{max_dijkstra_tool_calls} Dijkstra tool calls",
                        detail,
                    )
                tool_call_count += 1
                if request_metadata is not None:
                    request_metadata["dijkstra_tool_calls"] = tool_call_count
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as error:
                        raise ValueError("AI returned invalid Dijkstra tool arguments") from error
                if not isinstance(arguments, dict):
                    raise ValueError("AI returned invalid Dijkstra tool arguments")
                unknown_arguments = set(arguments) - {"exclude_segments", "exclude_points"}
                if unknown_arguments:
                    raise ValueError(
                        "AI returned unsupported Dijkstra tool argument(s): "
                        + ", ".join(sorted(str(key) for key in unknown_arguments))
                    )

                def normalize_exclusions(argument_name: str) -> list[str]:
                    values = arguments.get(argument_name)
                    if values is None:
                        return []
                    if isinstance(values, str):
                        return [values]
                    if isinstance(values, list) and all(isinstance(item, str) for item in values):
                        return values
                    raise ValueError(
                        f"AI returned invalid '{argument_name}' Dijkstra tool argument; "
                        "expected a string, an array of strings, or null"
                    )

                excluded_segments = normalize_exclusions("exclude_segments")
                excluded_points = normalize_exclusions("exclude_points")
                try:
                    route = find_shortest_route(
                        candidates,
                        a_end,
                        z_end,
                        excluded_span_ids=excluded_segments,
                        excluded_points=excluded_points,
                        strict_exclusions=True,
                        logger=dijkstra_logger,
                    )
                    discovered_routes[route.route_id] = route
                    tool_result = {
                        "route_reference": route.route_id,
                        "route_id": route.route_id,
                        "route": route.route_name,
                        "a_end": route.a_end,
                        "z_end": route.z_end,
                        "distance_km": round(route.distance_km, 3),
                        "span_ids": route.span_ids,
                        "span_names": route.span_names,
                    }
                except RoutePlanningError as error:
                    tool_result = {"error": str(error)}
                    tool_errors.append(str(error))
                if provider == "ollama":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": "find_dijkstra_route",
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                else:
                    call_id = tool_call.get("id")
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError("OpenRouter tool call did not contain an ID")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
            continue

        content = assistant_message.get("content")
        if not isinstance(content, str):
            _record_rejected_response(
                request_metadata, logger, data_logger, "missing_content", content, discovered_routes
            )
            raise AiSelectionError(
                "AI response did not select a Dijkstra route reference",
                {
                    "reason": "missing_content",
                    "response": content,
                    "valid_route_references": sorted(discovered_routes),
                    "tool_errors": tool_errors,
                },
            )
        try:
            selection = json.loads(content)
        except json.JSONDecodeError as error:
            _record_rejected_response(
                request_metadata, logger, data_logger, "not_json", content, discovered_routes
            )
            raise AiSelectionError(
                f"AI returned an invalid route reference: the response is not valid JSON "
                f"({error.msg} at line {error.lineno} column {error.colno}). "
                f"AI response: {_response_preview(content)}",
                {
                    "reason": "not_json",
                    "response": content,
                    "json_error": f"{error.msg} at line {error.lineno} column {error.colno}",
                    "valid_route_references": sorted(discovered_routes),
                    "tool_errors": tool_errors,
                },
            ) from error
        route_reference = (
            selection.get("route_reference") if isinstance(selection, dict) else None
        )
        if not isinstance(route_reference, str) or route_reference not in discovered_routes:
            if isinstance(selection, dict):
                if "route_reference" not in selection:
                    reason = "missing_route_reference"
                    detail = (
                        "the JSON object has no 'route_reference' key "
                        f"(keys found: {', '.join(sorted(map(str, selection))) or 'none'})"
                    )
                elif not isinstance(route_reference, str):
                    reason = "invalid_route_reference_type"
                    detail = (
                        f"'route_reference' is not a string but {type(route_reference).__name__}: "
                        f"{_response_preview(route_reference)}"
                    )
                else:
                    reason = "unknown_route_reference"
                    detail = (
                        f"'{route_reference}' was not returned by any successful Dijkstra call"
                    )
            else:
                reason = "unexpected_json_type"
                detail = f"the AI returned {type(selection).__name__} instead of a JSON object"
            _record_rejected_response(
                request_metadata, logger, data_logger, reason, content, discovered_routes
            )
            raise AiSelectionError(
                f"AI selected an unknown or uncalculated route reference: {detail}. "
                f"Valid route references: {_reference_preview(sorted(discovered_routes))}",
                {
                    "reason": reason,
                    "response": content,
                    "route_reference": route_reference,
                    "valid_route_references": sorted(discovered_routes),
                    "tool_errors": tool_errors,
                },
            )
        return discovered_routes[route_reference]


def select_ring_routes(
    provider: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
    locale: str,
    prompt_config: dict[str, Any],
    candidates: list[RouteCandidate],
    a_end: str,
    z_end: str,
    api_key: str | None = None,
    site_url: str = "",
    app_name: str = "",
    max_retries: int = 5,
    retry_delay_seconds: float = 2,
    system_prompt: str | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> list[list[RouteCandidate]]:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    instructions = system_prompt or build_ring_system_prompt(prompt_config, locale)
    user_payload = {
        "mode": "ring",
        "a_end": a_end,
        "z_end": z_end,
        "candidates": [candidate.to_model_dict() for candidate in candidates],
    }
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
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

    routes = result.get("routes") if isinstance(result, dict) else None
    if not isinstance(routes, list) or len(routes) != 2:
        raise ValueError("Ring response must contain exactly two routes")
    candidate_by_id = {candidate.route_id: candidate for candidate in candidates}
    selected_routes: list[list[RouteCandidate]] = []
    for route in routes:
        route_ids = route.get("route_ids") if isinstance(route, dict) else None
        if not isinstance(route_ids, list) or not route_ids or any(not isinstance(item, str) for item in route_ids):
            raise ValueError("Each ring route must contain a non-empty route_ids array")
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("A ring route cannot use a span more than once")
        if any(route_id not in candidate_by_id for route_id in route_ids):
            raise ValueError("AI selected unknown cable span identifiers")
        selected_routes.append([candidate_by_id[route_id] for route_id in route_ids])
    return selected_routes


def select_ai_ring_routes(
    provider: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
    locale: str,
    prompt_config: dict[str, Any],
    candidates: list[RouteCandidate],
    a_end: str,
    z_end: str,
    api_key: str | None = None,
    site_url: str = "",
    app_name: str = "",
    max_retries: int = 5,
    retry_delay_seconds: float = 2,
    max_dijkstra_tool_calls: int = 8,
    dijkstra_logger: logging.Logger | None = None,
    logger: logging.Logger | None = None,
    data_logger: logging.Logger | None = None,
    system_prompt: str | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> list[list[RouteCandidate]]:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if max_dijkstra_tool_calls < 1:
        raise ValueError("max_dijkstra_tool_calls must be a positive integer")
    instructions = system_prompt or build_ai_ring_system_prompt(prompt_config, locale)
    selected_routes: list[RouteCandidate] = []
    route_candidates = candidates
    prompt_tokens_total = 0
    completion_tokens_total = 0
    prompt_tokens_known = True
    completion_tokens_known = True
    dijkstra_tool_calls = 0

    def merge_leg_metadata(leg: dict[str, Any], tool_calls: int) -> None:
        """Keep the leg diagnostics even when the leg failed before returning."""
        if request_metadata is None:
            return
        request_metadata.update(leg)
        request_metadata["dijkstra_tool_calls"] = tool_calls
        request_metadata["prompt_tokens"] = prompt_tokens_total if prompt_tokens_known else None
        request_metadata["completion_tokens"] = (
            completion_tokens_total if completion_tokens_known else None
        )
        request_metadata["total_tokens"] = (
            prompt_tokens_total + completion_tokens_total
            if prompt_tokens_known and completion_tokens_known
            else None
        )

    for route_index in range(2):
        remaining_tool_calls = max_dijkstra_tool_calls - dijkstra_tool_calls
        if remaining_tool_calls < 1:
            detail = {
                "reason": "tool_call_limit",
                "max_dijkstra_tool_calls": max_dijkstra_tool_calls,
                "tool_calls_attempted": dijkstra_tool_calls,
            }
            merge_leg_metadata({}, dijkstra_tool_calls)
            if logger is not None:
                logger.error(
                    "Rejected AI response (tool_call_limit): the ring route selection "
                    "already used all %s Dijkstra tool calls.",
                    max_dijkstra_tool_calls,
                )
            if data_logger is not None:
                data_logger.info("Rejected AI response", extra={"data": detail})
            raise AiSelectionError(
                "AI exceeded the maximum of "
                f"{max_dijkstra_tool_calls} Dijkstra tool calls",
                detail,
            )
        leg_metadata: dict[str, Any] = {}
        try:
            selected_route = select_ai_route(
                provider=provider,
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                locale=locale,
                prompt_config=prompt_config,
                candidates=route_candidates,
                a_end=a_end,
                z_end=z_end,
                api_key=api_key,
                site_url=site_url,
                app_name=app_name,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                max_dijkstra_tool_calls=remaining_tool_calls,
                dijkstra_logger=dijkstra_logger,
                logger=logger,
                data_logger=data_logger,
                system_prompt=instructions,
                request_metadata=leg_metadata,
            )
        except Exception:
            merge_leg_metadata(leg_metadata, dijkstra_tool_calls + int(leg_metadata.get("dijkstra_tool_calls", 0)))
            raise
        selected_routes.append(selected_route)
        dijkstra_tool_calls += int(leg_metadata.get("dijkstra_tool_calls", 0))

        prompt_tokens = leg_metadata.get("prompt_tokens")
        completion_tokens = leg_metadata.get("completion_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens_known = False
        else:
            prompt_tokens_total += prompt_tokens
        if not isinstance(completion_tokens, int):
            completion_tokens_known = False
        else:
            completion_tokens_total += completion_tokens

        if route_index == 0:
            route_candidates = exclude_reference_spans(
                route_candidates,
                selected_route.span_ids,
            )
            route_candidates = exclude_crossing_spans(route_candidates, selected_route)
            if not route_candidates:
                raise RoutePlanningError(
                    "No cable spans remain for a non-crossing second ring route"
                )

        merge_leg_metadata(leg_metadata, dijkstra_tool_calls)

    candidate_by_id = {
        (candidate.source_id or candidate.route_id).strip().casefold(): candidate
        for candidate in candidates
    }
    selected_span_routes: list[list[RouteCandidate]] = []
    for route in selected_routes:
        try:
            selected_span_routes.append(
                [candidate_by_id[span_id.strip().casefold()] for span_id in route.span_ids]
            )
        except KeyError as error:
            raise ValueError(
                f"Dijkstra ring route refers to unknown span '{error.args[0]}'"
            ) from error
    return selected_span_routes


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
