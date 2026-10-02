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
python kmz_parser.py --mlist
python kmz_parser.py .\network.kmz --check
python kmz_parser.py --clist
python kmz_parser.py --mchange 2
python kmz_parser.py .\routes.kmz --mode shorter --muse 2
python kmz_parser.py .\routes.kmz --mode distance
python kmz_parser.py .\network.kml --closest "10 Main Street, Montreal"
python kmz_parser.py .\routes.kmz --mode shorter --locale en
python kmz_parser.py .\network.kml --mode djk --a-end "Point A" --z-end "Point B" --build
python kmz_parser.py .\network.kml --mode compare --a-end "Point A" --z-end "Point B"
python kmz_parser.py .\network.kml --mode ring --a-end "Point A" --z-end "Point B" --build
python kmz_parser.py .\network.kml --mode shorter --a-end "Point A" --z-end "Point B" --build
python kmz_parser.py .\network.kml --mode diverse --a-end "Point A" --z-end "Point B" --cable "Span A" --build
python kmz_parser.py .\network.kml --mode diverse --span 0001 --build
python kmz_parser.py .\routes.kmz --mode diverse --output .\result.json
python kmz_parser.py .\routes.kmz --mode shorter --or
python kmz_parser.py .\routes.kmz --mode shorter --openrouter
python kmz_parser.py .\routes.kmz --mode shorter --provider openrouter
```

The provider can also be selected by setting `provider: openrouter` (or `provider: or`) in `config.yaml`; `--provider` overrides that setting. `--mlist` displays the Ollama models with 1-based indexes. `--muse N` uses the indexed model for one run without changing the configuration; `--mchange N` updates `ollama.model` in the selected config file.

The system prompt used for the request is printed immediately above the compact, one-line JSON result. `--closest ADDRESS` geocodes an address with Nominatim and asks the selected AI provider to choose the nearest eligible manhole. The configurable `closest.max_distance_km` defaults to 10; a farther match is rejected with an error. Geocoding requires an internet connection.

`ring` requires `--a-end` and `--z-end`. The AI calculates both continuous routes directly from the available spans; Dijkstra is not used by this mode. The program verifies continuity, disjoint spans, and no geometric crossings except at A-END and Z-END. Without `--build`, both AI routes are written to the normal JSON output; with `--build`, one KMZ and one numbered build JSON contain both routes, with a different line color for each. If the AI does not return two valid routes, the command fails instead of writing a partial ring. `--build` writes numbered route output using `<A-END>_<Z-END>_<distance-in-meters>` in filenames. `compare` asks the AI to calculate a continuous span sequence and compares it with Dijkstra; a mismatch exits with an error and creates no KMZ. `djk` calculates the shortest route locally without AI. The KMZ contains `sites` and `route` folders and keeps cable spans as separate geometries. `diverse` selects diverse routes; it can exclude a previous build with `--span` or a cable with `--cable`. Cable endpoints are associated with named KML points using the configured `network.endpoint_match_km` tolerance (default: 0.5 km). The standard JSON output contains `Route`, `A-END`, `Z-END`, and `distance` for each route.
`--check` runs an AI consistency review of every Placemark in a KML/KMZ. It stores one row per anomaly in the `check_anomalies` table of `logs/requests.sqlite3`, including object type/ID, inconsistency, explanation, possible solution, provider, model, and AI engine name. The `requests` table stores the total inconsistency and potential-solution counts. No detailed JSON report is created and findings are not printed. Run `--clist` to display all stored anomalies in a terminal table. Existing databases are upgraded automatically.

Prompts are configured in `prompts.yaml`. Translated CLI and log messages are in `locales/fr.yaml` and `locales/en.yaml`. The application writes `info.log`, structured `data.log`, and `error.log` under the configured log directory. It also records each parsed CLI request in `logs/requests.sqlite3` (inside the configured log directory), including its result or error, exact AI messages when applicable, elapsed time, provider/model, available token counts, diversity span or cable ID, and generated KMZ path.

## Tests

```powershell
python -m unittest discover -s tests -v
```
