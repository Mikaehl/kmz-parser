import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree


EARTH_RADIUS_KM = 6371.0088
ENDPOINT_MATCH_KM = 0.5


class KmlInputError(ValueError):
    pass


@dataclass
class RouteCandidate:
    route_id: str
    route_name: str
    a_end: str
    z_end: str
    distance_km: float
    points: list[tuple[float, float]]
    segments: list[list[tuple[float, float]]] = field(default_factory=list)
    source_id: str | None = None
    span_ids: list[str] = field(default_factory=list)
    span_names: list[str] = field(default_factory=list)

    def to_model_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "span_id": self.source_id or self.route_id,
            "route": self.route_name,
            "a_end": self.a_end,
            "z_end": self.z_end,
            "distance_km": round(self.distance_km, 3),
            "geometry_sample": _sample_points(self.points),
        }

    def to_result_dict(self) -> dict[str, object]:
        return {
            "Route": self.route_name,
            "A-END": self.a_end,
            "Z-END": self.z_end,
            "distance": round(self.distance_km, 3),
        }


@dataclass(frozen=True)
class NamedPoint:
    point_id: str
    name: str
    coordinates: tuple[float, float]
    style_url: str = ""


def _local_name(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child_text(element: ElementTree.Element, name: str) -> str | None:
    for child in element.iter():
        if child is not element and _local_name(child) == name and child.text:
            return child.text.strip()
    return None


def _parse_coordinates(text: str | None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not text:
        return points
    for coordinate in text.split():
        values = coordinate.split(",")
        if len(values) < 2:
            continue
        try:
            longitude, latitude = float(values[0]), float(values[1])
        except ValueError:
            continue
        if -180 <= longitude <= 180 and -90 <= latitude <= 90:
            points.append((longitude, latitude))
    return points


def _distance_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    longitude_1, latitude_1 = map(math.radians, first)
    longitude_2, latitude_2 = map(math.radians, second)
    latitude_delta = latitude_2 - latitude_1
    longitude_delta = longitude_2 - longitude_1
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, haversine)))


def distance_between_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    return _distance_between(first, second)


def _route_length(points: list[tuple[float, float]]) -> float:
    return sum(_distance_between(start, end) for start, end in zip(points, points[1:]))


def _sample_points(points: list[tuple[float, float]], maximum: int = 12) -> list[list[float]]:
    if len(points) <= maximum:
        sampled = points
    else:
        indexes = [round(index * (len(points) - 1) / (maximum - 1)) for index in range(maximum)]
        sampled = [points[index] for index in indexes]
    return [[round(longitude, 6), round(latitude, 6)] for longitude, latitude in sampled]


def _point_names(root: ElementTree.Element) -> list[tuple[str, tuple[float, float]]]:
    named_points: list[tuple[str, tuple[float, float]]] = []
    for placemark in (element for element in root.iter() if _local_name(element) == "Placemark"):
        name = _child_text(placemark, "name")
        for point_element in (element for element in placemark.iter() if _local_name(element) == "Point"):
            coordinates = _parse_coordinates(_child_text(point_element, "coordinates"))
            if name and coordinates:
                named_points.append((name, coordinates[0]))
    return named_points


def _named_points(root: ElementTree.Element) -> list[NamedPoint]:
    named_points: list[NamedPoint] = []
    for placemark in (element for element in root.iter() if _local_name(element) == "Placemark"):
        name = _child_text(placemark, "name")
        if not name:
            continue
        point_elements = [element for element in placemark.iter() if _local_name(element) == "Point"]
        for point_index, point_element in enumerate(point_elements, start=1):
            coordinates = _parse_coordinates(_child_text(point_element, "coordinates"))
            if coordinates:
                point_id = placemark.attrib.get("id") or f"{name}-{point_index}"
                named_points.append(
                    NamedPoint(
                        point_id,
                        name,
                        coordinates[0],
                        _child_text(placemark, "styleUrl") or "",
                    )
                )
    return named_points


def parse_named_points(input_path: Path) -> list[NamedPoint]:
    return _named_points(_load_kml_root(input_path))


def parse_kml_objects(input_path: Path) -> list[dict[str, object]]:
    root = _load_kml_root(input_path)
    objects: list[dict[str, object]] = []
    used_ids: set[str] = set()
    id_counts: dict[str, int] = {}
    geometry_names = {"Point", "LineString", "Polygon", "Track", "Model"}

    def visit(element: ElementTree.Element, folders: tuple[str, ...]) -> None:
        local_name = _local_name(element)
        current_folders = folders
        if local_name == "Folder":
            folder_name = _child_text(element, "name")
            if folder_name:
                current_folders = (*folders, folder_name)
        elif local_name == "Placemark":
            name = _child_text(element, "name") or ""
            geometries: list[dict[str, object]] = []
            for geometry in element.iter():
                geometry_type = _local_name(geometry)
                if geometry_type not in geometry_names:
                    continue
                coordinates: list[tuple[float, float]] = []
                for coordinates_element in geometry.iter():
                    if _local_name(coordinates_element) == "coordinates":
                        coordinates.extend(_parse_coordinates(coordinates_element.text))
                geometry_summary: dict[str, object] = {"type": geometry_type}
                if coordinates:
                    if geometry_type == "Point":
                        geometry_summary["coordinates"] = list(coordinates[0])
                    elif geometry_type == "LineString":
                        geometry_summary["coordinate_count"] = len(coordinates)
                        geometry_summary["start"] = list(coordinates[0])
                        geometry_summary["end"] = list(coordinates[-1])
                    else:
                        geometry_summary["sample"] = _sample_points(coordinates)
                geometries.append(geometry_summary)

            object_type = geometries[0]["type"] if geometries else "Placemark"
            kml_id = element.attrib.get("id")
            base_id = str(kml_id or name or f"{object_type}-{len(objects) + 1}")
            id_counts[base_id] = id_counts.get(base_id, 0) + 1
            object_id = base_id if id_counts[base_id] == 1 else f"{base_id}#{id_counts[base_id]}"
            while object_id in used_ids:
                id_counts[base_id] += 1
                object_id = f"{base_id}#{id_counts[base_id]}"
            used_ids.add(object_id)

            extended_data: dict[str, str] = {}
            for data_element in element.iter():
                data_type = _local_name(data_element)
                if data_type in {"Data", "SimpleData"}:
                    data_name = data_element.attrib.get("name")
                    data_value = _child_text(data_element, "value") if data_type == "Data" else data_element.text
                    if data_name and data_value:
                        extended_data[data_name] = data_value.strip()

            object_data: dict[str, object] = {"object_id": object_id, "type": object_type}
            if kml_id:
                object_data["kml_id"] = kml_id
            if name:
                object_data["name"] = name
            if current_folders:
                object_data["folders"] = list(current_folders)
            style_url = _child_text(element, "styleUrl")
            if style_url:
                object_data["style_url"] = style_url
            if extended_data:
                object_data["extended_data"] = extended_data
            if geometries:
                object_data["geometries"] = geometries
            objects.append(object_data)
        for child in element:
            visit(child, current_folders)

    visit(root, ())
    return objects


def find_manhole_points(
    named_points: list[NamedPoint],
    routes: list[RouteCandidate],
) -> list[NamedPoint]:
    endpoint_degrees: dict[str, int] = {}
    for route in routes:
        for endpoint in (route.a_end, route.z_end):
            endpoint_key = endpoint.casefold()
            endpoint_degrees[endpoint_key] = endpoint_degrees.get(endpoint_key, 0) + 1

    manholes = [
        point
        for point in named_points
        if "homegardenbusiness" not in point.style_url.casefold()
        and (
            endpoint_degrees.get(point.name.casefold(), 0) > 1
            or "mh" in point.name.casefold()
            or "intermediate" in point.name.casefold()
        )
    ]
    unique_points: dict[tuple[str, tuple[float, float]], NamedPoint] = {}
    for point in manholes:
        unique_points.setdefault((point.name.casefold(), point.coordinates), point)
    return list(unique_points.values())


def _endpoint_name(
    coordinate: tuple[float, float],
    named_points: list[tuple[str, tuple[float, float]]],
    match_tolerance_km: float,
) -> str:
    nearest: tuple[str, float] | None = None
    for name, point in named_points:
        distance = _distance_between(coordinate, point)
        if nearest is None or distance < nearest[1]:
            nearest = (name, distance)
    if nearest is not None and nearest[1] <= match_tolerance_km:
        return nearest[0]
    longitude, latitude = coordinate
    return f"{longitude:.6f}, {latitude:.6f}"


def _read_kml(input_path: Path) -> bytes:
    extension = input_path.suffix.lower()
    if extension == ".kml":
        return input_path.read_bytes()
    if extension != ".kmz":
        raise KmlInputError("Expected a .kml or .kmz file")
    if not zipfile.is_zipfile(input_path):
        raise KmlInputError("KMZ file is not a valid ZIP archive")

    try:
        with zipfile.ZipFile(input_path) as archive:
            kml_names = [
                name for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith(".kml")
            ]
            if not kml_names:
                raise KmlInputError("KMZ archive contains no KML file")
            doc_kml = next((name for name in kml_names if Path(name).name.lower() == "doc.kml"), kml_names[0])
            return archive.read(doc_kml)
    except (zipfile.BadZipFile, KeyError, OSError) as error:
        raise KmlInputError(f"Cannot read KML content from KMZ archive: {error}") from error


def _load_kml_root(input_path: Path) -> ElementTree.Element:
    try:
        root = ElementTree.fromstring(_read_kml(input_path))
    except ElementTree.ParseError as error:
        raise KmlInputError(f"Invalid KML XML: {error}") from error
    if _local_name(root) != "kml":
        raise KmlInputError("XML root element must be <kml>")
    return root


def parse_kml_file(
    input_path: Path,
    endpoint_match_km: float = ENDPOINT_MATCH_KM,
) -> list[RouteCandidate]:
    if not math.isfinite(endpoint_match_km) or endpoint_match_km < 0:
        raise ValueError("endpoint_match_km must be a finite non-negative number")
    root = _load_kml_root(input_path)

    named_points = _point_names(root)
    candidates: list[RouteCandidate] = []

    for placemark in (element for element in root.iter() if _local_name(element) == "Placemark"):
        route_name = _child_text(placemark, "name")
        source_id = placemark.attrib.get("id") or route_name
        for line_element in (element for element in placemark.iter() if _local_name(element) == "LineString"):
            points = _parse_coordinates(_child_text(line_element, "coordinates"))
            if len(points) < 2:
                continue
            route_id = f"R{len(candidates) + 1:03d}"
            candidates.append(
                RouteCandidate(
                    route_id=route_id,
                    route_name=route_name or f"Route {len(candidates) + 1}",
                    a_end=_endpoint_name(points[0], named_points, endpoint_match_km),
                    z_end=_endpoint_name(points[-1], named_points, endpoint_match_km),
                    distance_km=_route_length(points),
                    points=points,
                    source_id=source_id or route_id,
                )
            )
    return candidates
