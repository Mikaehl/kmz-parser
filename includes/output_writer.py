import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from includes.kml_parser import RouteCandidate


KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
SITE_ICON_HREF = "http://maps.google.com/mapfiles/kml/shapes/homegardenbusiness.png"
ElementTree.register_namespace("", KML_NAMESPACE)
ROUTE_COLORS = (
    ("cyan", "ffffff00"),
    ("violet", "ffee82ee"),
    ("vert", "ff008000"),
    ("rouge", "ff0000ff"),
    ("bleu", "ffff0000"),
    ("jaune", "ff00ffff"),
    ("orange", "ff00a5ff"),
)


def write_results(routes: list[RouteCandidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [route.to_result_dict() for route in routes]
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def write_built_route_json(
    route: RouteCandidate,
    output_path: Path,
    additional_routes: list[RouteCandidate] | None = None,
) -> None:
    payload = []
    for built_route in [route, *(additional_routes or [])]:
        route_data = built_route.to_result_dict()
        if built_route.span_ids:
            route_data["Spans"] = [
                {"id": span_id, "name": span_name}
                for span_id, span_name in zip(built_route.span_ids, built_route.span_names)
            ]
        payload.append(route_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def _safe_filename_part(value: str) -> str:
    safe_value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value)
    safe_value = re.sub(r"\s+", "_", safe_value).strip(" ._")
    return safe_value or "point"


def build_route_filename(
    route: RouteCandidate,
    identifier: int | None = None,
    extension: str = "kmz",
) -> str:
    distance_meters = round(route.distance_km * 1000)
    stem = (
        f"{_safe_filename_part(route.a_end)}_"
        f"{_safe_filename_part(route.z_end)}_"
        f"{distance_meters}m"
    )
    prefix = f"{identifier:04d}_" if identifier is not None else ""
    return f"{prefix}{stem}.{extension.lstrip('.')}"


def next_build_identifier(output_directory: Path) -> int:
    identifier_pattern = re.compile(r"^(\d+)_.*\.(?:json|kmz)$", re.IGNORECASE)
    existing_identifiers = [
        int(match.group(1))
        for path in output_directory.iterdir()
        if (match := identifier_pattern.match(path.name))
    ] if output_directory.exists() else []
    return max(existing_identifiers, default=0) + 1


def build_route_output_paths(
    route: RouteCandidate,
    output_directory: Path,
) -> tuple[int, Path, Path]:
    identifier = next_build_identifier(output_directory)
    json_path = output_directory / build_route_filename(route, identifier, "json")
    kmz_path = output_directory / build_route_filename(route, identifier, "kmz")
    return identifier, json_path, kmz_path


def write_route_kmz(
    route: RouteCandidate,
    output_path: Path,
    build_identifier: int = 1,
    description: str | None = None,
    additional_routes: list[RouteCandidate] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = ElementTree.Element(f"{{{KML_NAMESPACE}}}kml")
    document = ElementTree.SubElement(root, f"{{{KML_NAMESPACE}}}Document")
    ElementTree.SubElement(document, f"{{{KML_NAMESPACE}}}name").text = output_path.name
    if description:
        ElementTree.SubElement(document, f"{{{KML_NAMESPACE}}}description").text = description

    site_style = ElementTree.SubElement(document, f"{{{KML_NAMESPACE}}}Style", id="siteStyle")
    icon_style = ElementTree.SubElement(site_style, f"{{{KML_NAMESPACE}}}IconStyle")
    icon = ElementTree.SubElement(icon_style, f"{{{KML_NAMESPACE}}}Icon")
    ElementTree.SubElement(icon, f"{{{KML_NAMESPACE}}}href").text = SITE_ICON_HREF

    routes_to_write = [route, *(additional_routes or [])]
    route_style_ids: list[str] = []
    for route_index in range(len(routes_to_write)):
        style_id = "routeStyle" if route_index == 0 else f"routeStyle{route_index + 1}"
        route_style_ids.append(style_id)
        _, route_color = ROUTE_COLORS[(build_identifier - 1 + route_index) % len(ROUTE_COLORS)]
        style = ElementTree.SubElement(document, f"{{{KML_NAMESPACE}}}Style", id=style_id)
        line_style = ElementTree.SubElement(style, f"{{{KML_NAMESPACE}}}LineStyle")
        ElementTree.SubElement(line_style, f"{{{KML_NAMESPACE}}}color").text = route_color
        ElementTree.SubElement(line_style, f"{{{KML_NAMESPACE}}}width").text = "5"

    sites_folder = ElementTree.SubElement(document, f"{{{KML_NAMESPACE}}}Folder")
    ElementTree.SubElement(sites_folder, f"{{{KML_NAMESPACE}}}name").text = "sites"
    if route.points:
        for site_name, coordinate in (
            (route.a_end, route.points[0]),
            (route.z_end, route.points[-1]),
        ):
            site_placemark = ElementTree.SubElement(sites_folder, f"{{{KML_NAMESPACE}}}Placemark")
            ElementTree.SubElement(site_placemark, f"{{{KML_NAMESPACE}}}name").text = site_name
            ElementTree.SubElement(site_placemark, f"{{{KML_NAMESPACE}}}styleUrl").text = "#siteStyle"
            point = ElementTree.SubElement(site_placemark, f"{{{KML_NAMESPACE}}}Point")
            ElementTree.SubElement(point, f"{{{KML_NAMESPACE}}}coordinates").text = (
                f"{coordinate[0]:.8f},{coordinate[1]:.8f},0"
            )

    route_folder = ElementTree.SubElement(document, f"{{{KML_NAMESPACE}}}Folder")
    ElementTree.SubElement(route_folder, f"{{{KML_NAMESPACE}}}name").text = "route"
    for route_to_write, style_id in zip(routes_to_write, route_style_ids):
        placemark = ElementTree.SubElement(route_folder, f"{{{KML_NAMESPACE}}}Placemark")
        ElementTree.SubElement(placemark, f"{{{KML_NAMESPACE}}}name").text = route_to_write.route_name
        ElementTree.SubElement(placemark, f"{{{KML_NAMESPACE}}}styleUrl").text = f"#{style_id}"
        geometry = ElementTree.SubElement(placemark, f"{{{KML_NAMESPACE}}}MultiGeometry")
        segments = route_to_write.segments or [route_to_write.points]
        for segment in segments:
            if len(segment) < 2:
                continue
            line_string = ElementTree.SubElement(geometry, f"{{{KML_NAMESPACE}}}LineString")
            ElementTree.SubElement(line_string, f"{{{KML_NAMESPACE}}}tessellate").text = "1"
            coordinates = " ".join(f"{longitude:.8f},{latitude:.8f},0" for longitude, latitude in segment)
            ElementTree.SubElement(line_string, f"{{{KML_NAMESPACE}}}coordinates").text = coordinates

    kml_content = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", kml_content)
