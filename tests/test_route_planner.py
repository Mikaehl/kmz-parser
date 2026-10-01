import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from includes.built_routes import load_built_route_reference
from includes.output_writer import (
    build_route_filename,
    build_route_output_paths,
    write_built_route_json,
    write_route_kmz,
)
from includes.kml_parser import RouteCandidate, parse_kml_file
from includes.route_planner import RoutePlanningError, find_shortest_route


SAMPLE_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Point A</name><Point><coordinates>0,0,0</coordinates></Point></Placemark>
  <Placemark><name>Intermediate A</name><Point><coordinates>1,0,0</coordinates></Point></Placemark>
  <Placemark><name>Intermediate B</name><Point><coordinates>2,0,0</coordinates></Point></Placemark>
  <Placemark><name>Point B</name><Point><coordinates>3,0,0</coordinates></Point></Placemark>
    <Placemark id="cable-a-id"><name>Cable A</name><LineString><coordinates>0.003,0,0 1.003,0,0</coordinates></LineString></Placemark>
  <Placemark><name>Cable B</name><LineString><coordinates>2.003,0,0 0.997,0,0</coordinates></LineString></Placemark>
  <Placemark><name>Cable C</name><LineString><coordinates>1.997,0,0 3.003,0,0</coordinates></LineString></Placemark>
  <Placemark><name>Detour cable</name><LineString><coordinates>0.003,0,0 1.5,2,0 3.003,0,0</coordinates></LineString></Placemark>
</Document></kml>"""


class RoutePlannerTests(unittest.TestCase):
    def test_finds_shortest_composite_route_through_intermediate_manholes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "network.kml"
            kml_path.write_text(SAMPLE_KML, encoding="utf-8")
            spans = parse_kml_file(kml_path)

        route = find_shortest_route(spans, "Point A", "Point B")

        self.assertEqual(route.route_id, "R001+R002+R003")
        self.assertEqual(route.route_name, "Cable A -> Cable B -> Cable C")
        self.assertEqual(route.a_end, "Point A")
        self.assertEqual(route.z_end, "Point B")
        self.assertGreater(route.distance_km, 300)
        self.assertLess(route.distance_km, 400)
        self.assertEqual(route.points[0], (0.003, 0.0))
        self.assertEqual(route.points[-1], (3.003, 0.0))

    def test_rejects_unknown_endpoint(self) -> None:
        with self.assertRaisesRegex(RoutePlanningError, "A-END point 'Unknown'"):
            find_shortest_route([], "Unknown", "Point B")

    def test_rejects_unconnected_endpoints(self) -> None:
        disconnected_spans = [
            RouteCandidate("R001", "Cable A", "Point A", "Intermediate A", 10, [(0, 0), (1, 0)]),
            RouteCandidate("R002", "Cable B", "Intermediate B", "Point B", 10, [(2, 0), (3, 0)]),
        ]
        with self.assertRaisesRegex(RoutePlanningError, "No continuous cable route"):
            find_shortest_route(disconnected_spans, "Point A", "Point B")

    def test_diverse_route_excludes_kml_cable_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "network.kml"
            kml_path.write_text(SAMPLE_KML, encoding="utf-8")
            spans = parse_kml_file(kml_path)

        cable_a = next(span for span in spans if span.route_name == "Cable A")
        self.assertEqual(cable_a.source_id, "cable-a-id")
        alternative = find_shortest_route(
            spans,
            "Point A",
            "Point B",
            excluded_span_id="cable-a-id",
        )

        self.assertEqual(alternative.route_name, "Detour cable")
        self.assertNotIn(cable_a.route_name, alternative.route_name)

    def test_span_reference_excludes_all_segments_from_built_route_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "network.kml"
            output_directory = Path(temporary_directory) / "output"
            kml_path.write_text(SAMPLE_KML, encoding="utf-8")
            spans = parse_kml_file(kml_path)
            original_route = find_shortest_route(spans, "Point A", "Point B")
            _, reference_json, _ = build_route_output_paths(original_route, output_directory)
            write_built_route_json(original_route, reference_json)
            reference = load_built_route_reference("0001", output_directory)
            alternative = find_shortest_route(
                spans,
                reference.a_end,
                reference.z_end,
                excluded_span_ids=reference.span_ids,
            )

        self.assertEqual(reference.span_ids, ["cable-a-id", "Cable B", "Cable C"])
        self.assertEqual(alternative.route_name, "Detour cable")

    def test_span_reference_reads_legacy_route_name_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            legacy_route_path = output_directory / "0001_Point_A_Point_B_1m.json"
            legacy_route_path.write_text(
                json.dumps(
                    [{"Route": "Cable A -> Cable B", "A-END": "Point A", "Z-END": "Point B", "distance": 1}]
                ),
                encoding="utf-8",
            )

            reference = load_built_route_reference("1", output_directory)

        self.assertEqual(reference.span_ids, ["Cable A", "Cable B"])
        self.assertEqual(reference.a_end, "Point A")
        self.assertEqual(reference.z_end, "Point B")

    def test_rejects_unknown_cable_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "network.kml"
            kml_path.write_text(SAMPLE_KML, encoding="utf-8")
            spans = parse_kml_file(kml_path)

        with self.assertRaisesRegex(RoutePlanningError, "Cable span 'missing' was not found"):
            find_shortest_route(spans, "Point A", "Point B", excluded_span_id="missing")

    def test_builds_kmz_with_safe_endpoint_filename_and_separate_spans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "network.kml"
            kml_path.write_text(SAMPLE_KML, encoding="utf-8")
            route = find_shortest_route(parse_kml_file(kml_path), "Point A", "Point B")
            output_path = Path(temporary_directory) / build_route_filename(route)

            write_route_kmz(route, output_path, description="Mode utilisé : Dijkstra")

            self.assertRegex(output_path.name, r"^Point_A_Point_B_\d+m\.kmz$")
            with zipfile.ZipFile(output_path) as archive:
                self.assertEqual(archive.namelist(), ["doc.kml"])
                root = ElementTree.fromstring(archive.read("doc.kml"))

            self.assertEqual(root.findtext("{http://www.opengis.net/kml/2.2}Document/{http://www.opengis.net/kml/2.2}name"), output_path.name)
            self.assertEqual(
                root.findtext("{http://www.opengis.net/kml/2.2}Document/{http://www.opengis.net/kml/2.2}description"),
                "Mode utilisé : Dijkstra",
            )
        lines = root.findall(".//{http://www.opengis.net/kml/2.2}LineString")
        self.assertEqual(len(lines), 3)

        folders = root.findall(".//{http://www.opengis.net/kml/2.2}Folder")
        self.assertEqual(
            [folder.findtext("{http://www.opengis.net/kml/2.2}name") for folder in folders],
            ["sites", "route"],
        )
        sites_folder, route_folder = folders
        sites = sites_folder.findall("{http://www.opengis.net/kml/2.2}Placemark")
        self.assertEqual(
            [site.findtext("{http://www.opengis.net/kml/2.2}name") for site in sites],
            ["Point A", "Point B"],
        )
        self.assertTrue(
            all(
                site.findtext("{http://www.opengis.net/kml/2.2}styleUrl") == "#siteStyle"
                for site in sites
            )
        )
        icon_href = root.findtext(
            ".//{http://www.opengis.net/kml/2.2}Style[@id='siteStyle']"
            "/{http://www.opengis.net/kml/2.2}IconStyle/{http://www.opengis.net/kml/2.2}Icon"
            "/{http://www.opengis.net/kml/2.2}href"
        )
        self.assertEqual(icon_href, "http://maps.google.com/mapfiles/kml/shapes/homegardenbusiness.png")
        route_placemark = route_folder.find("{http://www.opengis.net/kml/2.2}Placemark")
        self.assertEqual(route_placemark.findtext("{http://www.opengis.net/kml/2.2}styleUrl"), "#routeStyle")
        line_style = root.find(".//{http://www.opengis.net/kml/2.2}LineStyle")
        self.assertEqual(line_style.findtext("{http://www.opengis.net/kml/2.2}width"), "5")
        self.assertEqual(line_style.findtext("{http://www.opengis.net/kml/2.2}color"), "ffffff00")

    def test_route_color_rotates_by_build_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "network.kml"
            kml_path.write_text(SAMPLE_KML, encoding="utf-8")
            route = find_shortest_route(parse_kml_file(kml_path), "Point A", "Point B")
            output_path = Path(temporary_directory) / "route.kmz"
            write_route_kmz(route, output_path, build_identifier=2)

            with zipfile.ZipFile(output_path) as archive:
                root = ElementTree.fromstring(archive.read("doc.kml"))

        color = root.findtext(
            ".//{http://www.opengis.net/kml/2.2}LineStyle/{http://www.opengis.net/kml/2.2}color"
        )
        self.assertEqual(color, "ffee82ee")

    def test_build_outputs_share_an_incrementing_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "network.kml"
            output_directory = Path(temporary_directory) / "output"
            kml_path.write_text(SAMPLE_KML, encoding="utf-8")
            route = find_shortest_route(parse_kml_file(kml_path), "Point A", "Point B")

            first_identifier, first_json, first_kmz = build_route_output_paths(route, output_directory)
            output_directory.mkdir()
            first_json.touch()
            first_kmz.touch()
            second_identifier, second_json, second_kmz = build_route_output_paths(route, output_directory)
        route_name = f"Point_A_Point_B_{round(route.distance_km * 1000)}m"
        self.assertEqual(first_json.name, f"0001_{route_name}.json")
        self.assertEqual(first_kmz.name, f"0001_{route_name}.kmz")
        self.assertEqual(first_identifier, 1)
        self.assertEqual(second_json.name, f"0002_{route_name}.json")
        self.assertEqual(second_kmz.name, f"0002_{route_name}.kmz")
        self.assertEqual(second_identifier, 2)


if __name__ == "__main__":
    unittest.main()
