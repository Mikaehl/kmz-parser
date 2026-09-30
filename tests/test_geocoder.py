import io
import json
import unittest
import urllib.parse
from unittest.mock import patch

from includes.geocoder import GeocodingError, geocode_address


class GeocoderTests(unittest.TestCase):
    @patch("includes.geocoder.urllib.request.urlopen")
    def test_geocodes_address_with_configured_user_agent(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = io.BytesIO(json.dumps([{"lat": "45.5", "lon": "-73.6"}]).encode())

        coordinates = geocode_address(
            "10 Main Street, Montréal",
            "https://nominatim.example/search",
            "RouteParserTest/1.0",
            12,
        )

        request = mock_urlopen.call_args.args[0]
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        self.assertEqual(query["q"], ["10 Main Street, Montréal"])
        self.assertEqual(request.get_header("User-agent"), "RouteParserTest/1.0")
        self.assertEqual(coordinates, (-73.6, 45.5))

    @patch("includes.geocoder.urllib.request.urlopen")
    def test_geocoding_reports_address_not_found(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = io.BytesIO(b"[]")

        with self.assertRaisesRegex(GeocodingError, "Address was not found"):
            geocode_address("Unknown address", "https://nominatim.example/search", "Test/1", 12)


if __name__ == "__main__":
    unittest.main()