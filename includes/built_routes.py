import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from includes.route_planner import RoutePlanningError


@dataclass
class BuiltRouteReference:
    identifier: str
    json_path: Path
    a_end: str
    z_end: str
    span_ids: list[str]


def load_built_route_reference(identifier: str, output_directory: Path) -> BuiltRouteReference:
    if not identifier.isdigit():
        raise RoutePlanningError(f"Built route id '{identifier}' must be numeric")

    filename_pattern = re.compile(r"^(\d+)_.*\.json$", re.IGNORECASE)
    matches = [
        path
        for path in output_directory.glob("*.json")
        if (match := filename_pattern.match(path.name)) and int(match.group(1)) == int(identifier)
    ]
    if not matches:
        raise RoutePlanningError(f"Built route JSON with id '{identifier}' was not found")
    if len(matches) > 1:
        raise RoutePlanningError(f"Built route id '{identifier}' matches multiple JSON files")

    route_path = matches[0]
    try:
        with route_path.open("r", encoding="utf-8") as route_file:
            route_data: Any = json.load(route_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RoutePlanningError(f"Cannot read built route JSON '{route_path}': {error}") from error

    if isinstance(route_data, list) and len(route_data) == 1:
        route_data = route_data[0]
    if not isinstance(route_data, dict):
        raise RoutePlanningError(f"Built route JSON '{route_path}' must contain one route object")

    a_end = route_data.get("A-END")
    z_end = route_data.get("Z-END")
    if not isinstance(a_end, str) or not isinstance(z_end, str):
        raise RoutePlanningError(f"Built route JSON '{route_path}' has no valid A-END/Z-END values")

    span_ids: list[str] = []
    stored_spans = route_data.get("Spans")
    if isinstance(stored_spans, list):
        for span in stored_spans:
            if isinstance(span, str):
                span_ids.append(span)
            elif isinstance(span, dict):
                span_id = span.get("id")
                span_name = span.get("name")
                if isinstance(span_id, str) and span_id:
                    span_ids.append(span_id)
                elif isinstance(span_name, str) and span_name:
                    span_ids.append(span_name)
    if not span_ids:
        route_name = route_data.get("Route")
        if isinstance(route_name, str):
            span_ids = [part.strip() for part in route_name.split(" -> ") if part.strip()]
    if not span_ids:
        raise RoutePlanningError(f"Built route JSON '{route_path}' has no span list")

    return BuiltRouteReference(identifier, route_path, a_end, z_end, list(dict.fromkeys(span_ids)))