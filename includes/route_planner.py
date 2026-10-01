import heapq

from includes.kml_parser import RouteCandidate


class RoutePlanningError(ValueError):
    pass


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
) -> RouteCandidate:
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
        if not matching_spans and identifier == excluded_span_id:
            raise RoutePlanningError(f"Cable span '{identifier}' was not found in the KML")
        if not matching_spans:
            continue
        if len(matching_spans) > 1:
            raise RoutePlanningError(f"Cable span id '{identifier}' is ambiguous")
        excluded_spans.add(id(matching_spans[0]))
    if excluded_spans:
        spans = [span for span in spans if id(span) not in excluded_spans]

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

    graph: dict[str, list[tuple[str, RouteCandidate]]] = {}
    for span in spans:
        first = span.a_end.strip().casefold()
        second = span.z_end.strip().casefold()
        if first == second:
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

    return RouteCandidate(
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