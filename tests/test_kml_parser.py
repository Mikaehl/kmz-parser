import tempfile
import unittest
import zipfile
from pathlib import Path

from includes.kml_parser import (
    KmlInputError,
    find_manhole_points,
    parse_kml_file,
    parse_named_points,
)


SAMPLE_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Start</name><Point><coordinates>-73.000000,45.000000,0</coordinates></Point></Placemark>
  <Placemark><name>Finish</name><Point><coordinates>-72.990000,45.000000,0</coordinates></Point></Placemark>
  <Placemark><name>River path</name><LineString><coordinates>
    -73.000000,45.000000,0 -72.995000,45.002000,0 -72.990000,45.000000,0
  </coordinates></LineString></Placemark>
</Document></kml>"""


class KmlParserTests(unittest.TestCase):
    def test_parses_kmz_route_and_calculates_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kmz_path = Path(temporary_directory) / "sample.kmz"
            with zipfile.ZipFile(kmz_path, "w") as archive:
                archive.writestr("doc.kml", SAMPLE_KML)

            routes = parse_kml_file(kmz_path)

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].route_name, "River path")
        self.assertEqual(routes[0].a_end, "Start")
        self.assertEqual(routes[0].z_end, "Finish")
        self.assertGreater(routes[0].distance_km, 0.8)
        self.assertEqual(set(routes[0].to_result_dict()), {"Route", "A-END", "Z-END", "distance"})

    def test_parses_kml_without_archive_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "sample.kml"
            kml_path.write_text(SAMPLE_KML, encoding="utf-8")

            routes = parse_kml_file(kml_path)

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].route_name, "River path")

    def test_rejects_xml_that_is_not_kml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "not-kml.kml"
            input_path.write_text("<root><Placemark /></root>", encoding="utf-8")

            with self.assertRaisesRegex(KmlInputError, "root element must be <kml>"):
                parse_kml_file(input_path)

    def test_rejects_unexpected_file_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "routes.xml"
            input_path.write_text(SAMPLE_KML, encoding="utf-8")

            with self.assertRaisesRegex(KmlInputError, "Expected a .kml or .kmz"):
                parse_kml_file(input_path)

    def test_endpoint_match_tolerance_is_configurable(self) -> None:
        kml_content = """<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
          <Placemark><name>Origin</name><Point><coordinates>0,0,0</coordinates></Point></Placemark>
          <Placemark><name>Destination</name><Point><coordinates>1,0,0</coordinates></Point></Placemark>
          <Placemark><name>Cable</name><LineString><coordinates>0.003,0,0 1,0,0</coordinates></LineString></Placemark>
        </Document></kml>"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "tolerance.kml"
            kml_path.write_text(kml_content, encoding="utf-8")

            matched_routes = parse_kml_file(kml_path, endpoint_match_km=0.5)
            unmatched_routes = parse_kml_file(kml_path, endpoint_match_km=0.2)

        self.assertEqual(matched_routes[0].a_end, "Origin")
        self.assertNotEqual(unmatched_routes[0].a_end, "Origin")

    def test_manhole_detection_excludes_home_icon_sites(self) -> None:
        kml_content = """<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
          <Placemark><name>Site A</name><styleUrl>#homegardenbusiness</styleUrl><Point><coordinates>0,0,0</coordinates></Point></Placemark>
          <Placemark><name>MH 1</name><Point><coordinates>1,0,0</coordinates></Point></Placemark>
          <Placemark><name>Site B</name><styleUrl>#homegardenbusiness</styleUrl><Point><coordinates>2,0,0</coordinates></Point></Placemark>
          <Placemark><name>Cable 1</name><LineString><coordinates>0.001,0,0 1.001,0,0</coordinates></LineString></Placemark>
          <Placemark><name>Cable 2</name><LineString><coordinates>0.999,0,0 1.999,0,0</coordinates></LineString></Placemark>
        </Document></kml>"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            kml_path = Path(temporary_directory) / "manholes.kml"
            kml_path.write_text(kml_content, encoding="utf-8")
            named_points = parse_named_points(kml_path)
            routes = parse_kml_file(kml_path)

        manholes = find_manhole_points(named_points, routes)
        self.assertEqual([point.name for point in manholes], ["MH 1"])


if __name__ == "__main__":
    unittest.main()
