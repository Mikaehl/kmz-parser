import json
import urllib.error
import urllib.parse
import urllib.request


class GeocodingError(ValueError):
    pass


def geocode_address(
    address: str,
    geocoder_url: str,
    user_agent: str,
    timeout_seconds: int,
) -> tuple[float, float]:
    query = urllib.parse.urlencode({"q": address, "format": "jsonv2", "limit": 1})
    request = urllib.request.Request(
        f"{geocoder_url}?{query}",
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            results = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise GeocodingError(f"Address lookup failed: {error}") from error
    if not isinstance(results, list) or not results:
        raise GeocodingError("Address was not found")
    try:
        latitude = float(results[0]["lat"])
        longitude = float(results[0]["lon"])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise GeocodingError("Address lookup returned invalid coordinates") from error
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise GeocodingError("Address lookup returned out-of-range coordinates")
    return longitude, latitude