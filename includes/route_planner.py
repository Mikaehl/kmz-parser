import heapq
import logging

from includes.kml_parser import RouteCandidate


class RoutePlanningError(ValueError):
    pass


def _cross(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _segment_intersection(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> tuple[bool, tuple[float, float] | None]:
    epsilon = 1e-12
    first_vector = (first_end[0] - first_start[0], first_end[1] - first_start[1])
    second_vector = (second_end[0] - second_start[0], second_end[1] - second_start[1])
    offset = (second_start[0] - first_start[0], second_start[1] - first_start[1])
    denominator = _cross(first_vector, second_vector)

    if abs(denominator) > epsilon:
        first_ratio = _cross(offset, second_vector) / denominator
        second_ratio = _cross(offset, first_vector) / denominator
        if -epsilon <= first_ratio <= 1 + epsilon and -epsilon <= second_ratio <= 1 + epsilon:
            return True, (
                first_start[0] + first_ratio * first_vector[0],
                first_start[1] + first_ratio * first_vector[1],
            )
        return False, None

    if abs(_cross(offset, first_vector)) > epsilon:
        return False, None

    axis = 0 if abs(first_vector[0]) >= abs(first_vector[1]) else 1
    overlap_start = max(
        min(first_start[axis], first_end[axis]),
        min(second_start[axis], second_end[axis]),
    )
    overlap_end = min(
        max(first_start[axis], first_end[axis]),
        max(second_start[axis], second_end[axis]),
    )
    if overlap_end < overlap_start - epsilon:
        return False, None
    if overlap_end - overlap_start > epsilon:
        return True, None

    for point in (first_start, first_end, second_start, second_end):
        if (
            min(first_start[axis], first_end[axis]) - epsilon <= point[axis]
            <= max(first_start[axis], first_end[axis]) + epsilon
            and min(second_start[axis], second_end[axis]) - epsilon <= point[axis]
            <= max(second_start[axis], second_end[axis]) + epsilon
        ):
            return True, point
    return False, None


def route_geometries_cross(first: RouteCandidate, second: RouteCandidate) -> bool:
    first_segments = first.segments or [first.points]
    second_segments = second.segments or [second.points]
    first_endpoints = [point for point in (first.points[:1] + first.points[-1:])]
    second_endpoints = [point for point in (second.points[:1] + second.points[-1:])]
    shared_terminals = [
        (first_point, second_point)
        for first_point in first_endpoints
        for second_point in second_endpoints
        if abs(first_point[0] - second_point[0]) <= 1e-9
        and abs(first_point[1] - second_point[1]) <= 1e-9
    ]

    for first_segment in first_segments:
        for second_segment in second_segments:
            for first_start, first_end in zip(first_segment, first_segment[1:]):
                for second_start, second_end in zip(second_segment, second_segment[1:]):
                    intersects, intersection = _segment_intersection(
                        first_start,
                        first_end,
                        second_start,
                        second_end,
                    )
                    if not intersects:
                        continue
                    if intersection is not None and any(
                        abs(intersection[0] - first_terminal[0]) <= 1e-9
                        and abs(intersection[1] - first_terminal[1]) <= 1e-9
                        and abs(intersection[0] - second_terminal[0]) <= 1e-9
                        and abs(intersection[1] - second_terminal[1]) <= 1e-9
                        for first_terminal, second_terminal in shared_terminals
                    ):
                        continue
                    return True
    return False


def exclude_crossing_spans(
    spans: list[RouteCandidate],
    reference_route: RouteCandidate,
) -> list[RouteCandidate]:
    return [span for span in spans if not route_geometries_cross(span, reference_route)]


def exclude_reference_spans(
    spans: list[RouteCandidate],
    excluded_span_ids: list[str],
) -> list[RouteCandidate]:
    excluded_keys = {identifier.strip().casefold() for identifier in excluded_span_ids}
    return [
        span
        for span in spans
        if not excluded_keys.intersection(
            {
                span.route_id.strip().casefold(),
                (span.source_id or "").strip().casefold(),
                span.route_name.strip().casefold(),
            }
        )
    ]


def find_shortest_route(
    spans: list[RouteCandidate],
    a_end: str,
    z_end: str,
    excluded_span_id: str | None = None,
    excluded_span_ids: list[str] | None = None,
    excluded_points: list[str] | None = None,
    strict_exclusions: bool = False,
    logger: logging.Logger | None = None,
) -> RouteCandidate:
    if logger is not None:
        logger.info(
            "Dijkstra call: A-END=%r, Z-END=%r, excluded_segments=%s, excluded_points=%s",
            a_end,
            z_end,
            excluded_span_ids or ([excluded_span_id] if excluded_span_id else []),
            excluded_points or [],
        )
    all_spans = spans
    exclusion_keys = list(excluded_span_ids or [])
    if excluded_span_id is not None:
        exclusion_keys.append(excluded_span_id)
    excluded_spans: set[int] = set()
    for identifier in exclusion_keys:
        cable_key = identifier.strip().casefold()
        exact_matches = [
            span
            for span in spans
            if cable_key in {
                span.route_id.strip().casefold(),
                (span.source_id or "").strip().casefold(),
            }
        ]
        matching_spans = exact_matches or [
            span for span in spans if cable_key == span.route_name.strip().casefold()
        ]
        if not matching_spans and (strict_exclusions or identifier == excluded_span_id):
            raise RoutePlanningError(f"Cable span '{identifier}' was not found in the KML")
        if not matching_spans:
            continue
        if len(matching_spans) > 1:
            raise RoutePlanningError(f"Cable span id '{identifier}' is ambiguous")
        excluded_spans.add(id(matching_spans[0]))
    if excluded_spans:
        spans = [span for span in spans if id(span) not in excluded_spans]

    point_names: dict[str, str] = {}
    for span in all_spans:
        for endpoint in (span.a_end, span.z_end):
            point_names.setdefault(endpoint.strip().casefold(), endpoint.strip())

    start_key = a_end.strip().casefold()
    end_key = z_end.strip().casefold()
    excluded_point_keys = {point.strip().casefold() for point in (excluded_points or [])}
    for point in excluded_points or []:
        point_key = point.strip().casefold()
        if not point_key:
            raise RoutePlanningError("Excluded point names cannot be empty")
        if point_key not in point_names:
            raise RoutePlanningError(f"Point '{point}' was not found in the cable spans")
    if start_key in excluded_point_keys or end_key in excluded_point_keys:
        raise RoutePlanningError("A-END and Z-END cannot be excluded points")
    if start_key not in point_names:
        raise RoutePlanningError(f"A-END point '{a_end}' was not found in the cable spans")
    if end_key not in point_names:
        raise RoutePlanningError(f"Z-END point '{z_end}' was not found in the cable spans")
    if start_key == end_key:
        raise RoutePlanningError("A-END and Z-END must be different points")

    graph: dict[str, list[tuple[str, RouteCandidate]]] = {}
    for span in spans:
        first = span.a_end.strip().casefold()
        second = span.z_end.strip().casefold()
        if first == second or first in excluded_point_keys or second in excluded_point_keys:
            continue
        graph.setdefault(first, []).append((second, span))
        graph.setdefault(second, []).append((first, span))

    distances = {start_key: 0.0}
    previous: dict[str, tuple[str, RouteCandidate]] = {}
    queue = [(0.0, start_key)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        if current == end_key:
            break
        for neighbor, span in graph.get(current, []):
            candidate_distance = distance + span.distance_km
            if candidate_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate_distance
                previous[neighbor] = (current, span)
                heapq.heappush(queue, (candidate_distance, neighbor))

    if end_key not in distances:
        raise RoutePlanningError(f"No continuous cable route connects '{a_end}' to '{z_end}'")

    path: list[tuple[str, str, RouteCandidate]] = []
    current = end_key
    while current != start_key:
        parent, span = previous[current]
        path.append((parent, current, span))
        current = parent
    path.reverse()

    route_points: list[tuple[float, float]] = []
    route_spans: list[RouteCandidate] = []
    route_segments: list[list[tuple[float, float]]] = []
    for source, destination, span in path:
        points = span.points if span.a_end.strip().casefold() == source else list(reversed(span.points))
        route_points.extend(points)
        route_segments.append(points)
        route_spans.append(span)

    route = RouteCandidate(
        route_id="+".join(span.route_id for span in route_spans),
        route_name=" -> ".join(span.route_name for span in route_spans),
        a_end=point_names[start_key],
        z_end=point_names[end_key],
        distance_km=distances[end_key],
        points=route_points,
        segments=route_segments,
        source_id="+".join(span.source_id or span.route_id for span in route_spans),
        span_ids=[span.source_id or span.route_id for span in route_spans],
        span_names=[span.route_name for span in route_spans],
    )
    if logger is not None:
        logger.info(
            "Dijkstra result: route_id=%s, segments=%s, distance_km=%.3f",
            route.route_id,
            route.span_ids,
            route.distance_km,
        )
    return route


def build_route_from_spans(
    spans: list[RouteCandidate],
    a_end: str,
    z_end: str,
) -> RouteCandidate:
    point_names: dict[str, str] = {}
    for span in spans:
        for endpoint in (span.a_end, span.z_end):
            point_names.setdefault(endpoint.strip().casefold(), endpoint.strip())

    start_key = a_end.strip().casefold()
    end_key = z_end.strip().casefold()
    if start_key not in point_names:
        raise RoutePlanningError(f"A-END point '{a_end}' was not found in the cable spans")
    if end_key not in point_names:
        raise RoutePlanningError(f"Z-END point '{z_end}' was not found in the cable spans")
    if start_key == end_key:
        raise RoutePlanningError("A-END and Z-END must be different points")
    if not spans:
        raise RoutePlanningError("The selected route contains no cable spans")

    path: list[tuple[str, str, RouteCandidate, list[tuple[float, float]]]] = []
    current = start_key
    for span in spans:
        first = span.a_end.strip().casefold()
        second = span.z_end.strip().casefold()
        if first == current:
            destination = second
            points = span.points
        elif second == current:
            destination = first
            points = list(reversed(span.points))
        else:
            raise RoutePlanningError(
                f"Selected span '{span.route_name}' does not continue the route from '{point_names[current]}'"
            )
        if destination == current:
            raise RoutePlanningError(f"Selected span '{span.route_name}' has identical endpoints")
        path.append((current, destination, span, points))
        current = destination

    if current != end_key:
        raise RoutePlanningError(f"Selected spans do not reach Z-END point '{z_end}'")

    route_points: list[tuple[float, float]] = []
    route_segments: list[list[tuple[float, float]]] = []
    for _, _, _, points in path:
        route_points.extend(points)
        route_segments.append(points)
    route_spans = [span for _, _, span, _ in path]
    return RouteCandidate(
        route_id="+".join(span.route_id for span in route_spans),
        route_name=" -> ".join(span.route_name for span in route_spans),
        a_end=point_names[start_key],
        z_end=point_names[end_key],
        distance_km=sum(span.distance_km for span in route_spans),
        points=route_points,
        segments=route_segments,
        source_id="+".join(span.source_id or span.route_id for span in route_spans),
        span_ids=[span.source_id or span.route_id for span in route_spans],
        span_names=[span.route_name for span in route_spans],
    )