# KMZ Route Parser

A Python command-line tool that reads routes from a KMZ or KML file, calculates route lengths from their coordinates, and asks the selected Ollama or OpenRouter provider to select or report routes. KMZ files are read from the archive; KML files are parsed directly without extraction. Distances are calculated locally in kilometers using the haversine formula; the model cannot change the reported values. The input must be well-formed XML with a `<kml>` root element.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The default provider is Ollama. Run it locally and download the configured model (default: `llama3.1`). To use OpenRouter, put your API key on a single line in the `API_KEY_OPENROUTER` file in the same directory as `config.yaml`, then select OpenRouter. The key file is ignored by Git. The model and API settings can be changed in the `openrouter` section of `config.yaml` (default model: `qwen/qwen3.8-27b:free`). `max_retries` sets the number of retries after the initial request for HTTP 429 responses (default: 5); `retry_delay_seconds` sets the base wait between retries (default: 2 seconds). A provider-supplied `Retry-After` value takes precedence.

## Usage

```powershell
python kmz_parser.py .\routes.kmz --mode distance
python kmz_parser.py .\network.kml --closest "10 Main Street, Montreal"
python kmz_parser.py .\routes.kmz --mode shorter --locale en
python kmz_parser.py .\network.kml --mode djk --a-end "Point A" --z-end "Point B" --build
python kmz_parser.py .\network.kml --mode compare --a-end "Point A" --z-end "Point B"
python kmz_parser.py .\network.kml --mode shorter --a-end "Point A" --z-end "Point B" --build
python kmz_parser.py .\network.kml --mode diverse --a-end "Point A" --z-end "Point B" --cable "Span A" --build
python kmz_parser.py .\network.kml --mode diverse --span 0001 --build
python kmz_parser.py .\routes.kmz --mode diverse --output .\result.json
python kmz_parser.py .\routes.kmz --mode shorter --or
python kmz_parser.py .\routes.kmz --mode shorter --openrouter
python kmz_parser.py .\routes.kmz --mode shorter --provider openrouter
```

The provider can also be selected by setting `provider: openrouter` (or `provider: or`) in `config.yaml`; `--provider` overrides that setting.

The system prompt used for the request is printed immediately above the compact, one-line JSON result. `--closest ADDRESS` geocodes an address with Nominatim and asks the selected AI provider to choose the nearest eligible manhole. The configurable `closest.max_distance_km` defaults to 10; a farther match is rejected with an error. Geocoding requires an internet connection.

`distance` returns every parsed route, `shorter` requires `--a-end` and `--z-end` and asks the AI provider to select the locally calculated shortest path. `djk` accepts the same endpoints but calculates and returns the shortest path directly with Dijkstra, without loading prompts or contacting an AI provider. `compare` asks the AI to calculate a continuous span sequence and compares it with Dijkstra; a mismatch exits with an error and creates no KMZ. Matching routes create one KMZ describing both methods. Add `--build` to write a numbered route JSON alongside KMZ output, using `<A-END>_<Z-END>_<distance-in-meters>` in both filenames (for example, `0001_Point_A_Point_B_333585m.json` and `.kmz`). The build JSON includes the source span IDs and names. The KMZ contains `sites` and `route` folders, endpoint placemarks, and individual cable spans. Site placemarks use the `homegardenbusiness.png` house icon, and the KML document title matches the generated KMZ filename. The route line is 5 pixels wide; its color cycles by build ID through cyan, violet, green, red, blue, yellow, and orange. Cable spans remain separate geometries in the KMZ to avoid drawing artificial links across endpoint matching gaps. `diverse` can still select generally diverse routes; to find an alternate path based on a previous build, provide its numeric prefix with `--span` (for example, `--span 0001`). The route endpoints are read from that build's JSON. Spans from that route are excluded from the current input; if its endpoint sites exist in the current network, the shortest remaining path between them is used, otherwise the AI selects from the remaining non-common spans. Explicit `--a-end` and `--z-end` values can specify endpoints in the current network. `--span` can also be combined with `--build` to save the next numbered JSON/KMZ pair. The older `--a-end`, `--z-end`, and `--cable` combination remains supported. Cable endpoints are associated with named KML Point placemarks within the configured `network.endpoint_match_km` tolerance (default: 0.5 km). The standard JSON output is an array of objects with `Route`, `A-END`, `Z-END`, and `distance`; distance is a numeric value in kilometers.

Prompts are configured in `prompts.yaml`. Translated CLI and log messages are in `locales/fr.yaml` and `locales/en.yaml`. The application writes `info.log`, structured `data.log`, and `error.log` under the configured log directory.

## Tests

```powershell
python -m unittest discover -s tests -v
```
