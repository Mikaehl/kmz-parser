import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from includes.built_routes import load_built_route_reference
from includes.configuration import load_locale, load_prompts, load_settings
from includes.geocoder import GeocodingError, geocode_address
from includes.kml_parser import (
    KmlInputError,
    RouteCandidate,
    distance_between_km,
    find_manhole_points,
    parse_kml_file,
    parse_named_points,
)
from includes.logging_setup import configure_logging
from includes.ollama_client import ProviderConfigurationError, build_system_prompt, select_routes
from includes.output_writer import (
    build_route_output_paths,
    write_built_route_json,
    write_results,
    write_route_kmz,
)
from includes.route_planner import (
    RoutePlanningError,
    build_route_from_spans,
    exclude_reference_spans,
    find_shortest_route,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
VALID_MODES = ("distance", "shorter", "djk", "compare", "diverse")


def _format_message(locale: dict[str, str], key: str, **values: Any) -> str:
    return locale.get(key, key).format(**values)


def _load_api_key(key_file: str, base_directory: Path) -> str:
    key_path = (base_directory / key_file).resolve()
    try:
        api_key = key_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ProviderConfigurationError(str(key_path)) from error
    if not api_key:
        raise ProviderConfigurationError(str(key_path))
    return api_key


def _display_results(result_data: list[dict[str, object]]) -> None:
    print(json.dumps(result_data, ensure_ascii=False, separators=(",", ":")))


def _display_prompt_and_results(system_prompt: str, result_data: list[dict[str, object]]) -> None:
    print(system_prompt)
    _display_results(result_data)


def _build_parser(locale: dict[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=locale["app_description"])
    parser.add_argument("input_file", type=Path, help=locale["argument_input"])
    parser.add_argument("--mode", "-m", choices=VALID_MODES, help=locale["argument_mode"])
    parser.add_argument("--closest", metavar="ADDRESS", help=locale["argument_closest"])
    parser.add_argument("--a-end", help=locale["argument_a_end"])
    parser.add_argument("--z-end", help=locale["argument_z_end"])
    parser.add_argument("--cable", help=locale["argument_cable"])
    parser.add_argument("--span", help=locale["argument_span"])
    parser.add_argument("--build", action="store_true", help=locale["argument_build"])
    parser.add_argument("--output", type=Path, help=locale["argument_output"])
    parser.add_argument("--config", type=Path, default=PROJECT_DIRECTORY / "config.yaml", help=locale["argument_config"])
    parser.add_argument("--locale", choices=("fr", "en"), help=locale["argument_locale"])
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument(
        "--provider",
        choices=("ollama", "openrouter", "or"),
        help=locale["argument_provider"],
    )
    provider_group.add_argument(
        "--openrouter",
        dest="provider_alias",
        action="store_const",
        const="openrouter",
        help=locale["argument_openrouter"],
    )
    provider_group.add_argument(
        "--or",
        dest="provider_alias",
        action="store_const",
        const="openrouter",
        help=locale["argument_or"],
    )
    parser.add_argument("--verbose", action="store_true", help=locale["argument_verbose"])
    return parser


def _preliminary_locale(arguments: list[str]) -> str:
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--config", type=Path, default=PROJECT_DIRECTORY / "config.yaml")
    preliminary_parser.add_argument("--locale", choices=("fr", "en"))
    preliminary, _ = preliminary_parser.parse_known_args(arguments)
    try:
        settings, _ = load_settings(preliminary.config)
        return preliminary.locale or settings["application"]["locale"]
    except (OSError, ValueError, KeyError):
        return preliminary.locale or "fr"


def run(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    locale_code = _preliminary_locale(arguments)
    locale = load_locale(locale_code)
    parser = _build_parser(locale)
    parsed = parser.parse_args(arguments)
    if parsed.closest:
        if parsed.mode or parsed.a_end or parsed.z_end or parsed.cable or parsed.span or parsed.build:
            parser.error(locale["error_closest_parameters"])
        mode = "closest"
    else:
        if not parsed.mode:
            parser.error(locale["error_mode_required"])
        mode = parsed.mode

    try:
        settings, base_directory = load_settings(parsed.config)
        locale_code = parsed.locale or settings["application"]["locale"]
        locale = load_locale(locale_code)
        log_level = "DEBUG" if parsed.verbose else settings["application"]["log_level"]
        application = settings["application"]
        log_directory = (base_directory / application["log_directory"]).resolve()
        logger, data_logger = configure_logging(log_directory, log_level)

        if not parsed.input_file.is_file():
            raise FileNotFoundError(_format_message(locale, "error_input_missing", path=parsed.input_file))

        logger.info(_format_message(locale, "log_started", path=parsed.input_file))
        candidates = parse_kml_file(
            parsed.input_file,
            endpoint_match_km=settings["network"]["endpoint_match_km"],
        )
        if not candidates and mode != "closest":
            raise ValueError(locale["error_no_routes"])
        logger.info(_format_message(locale, "log_routes_found", count=len(candidates)))
        data_logger.info(
            "Route candidates",
            extra={"data": [candidate.to_model_dict() for candidate in candidates]},
        )

        closest_context: dict[str, Any] | None = None
        diversity_details: str | None = None
        if parsed.build and mode not in {"shorter", "djk", "compare", "diverse"}:
            logger.error(locale["error_build_route_mode"])
            return 2

        if mode == "closest":
            named_points = parse_named_points(parsed.input_file)
            manholes = find_manhole_points(named_points, candidates)
            if not manholes:
                logger.error(locale["error_no_manholes"])
                return 2
            closest_settings = settings["closest"]
            address_coordinates = geocode_address(
                parsed.closest,
                closest_settings["geocoder_url"],
                closest_settings["user_agent"],
                closest_settings["timeout_seconds"],
            )
            selection_candidates = [
                RouteCandidate(
                    route_id=f"M{index:04d}",
                    route_name=point.name,
                    a_end=point.name,
                    z_end=point.name,
                    distance_km=distance_between_km(address_coordinates, point.coordinates),
                    points=[point.coordinates],
                    source_id=point.point_id,
                )
                for index, point in enumerate(manholes, start=1)
            ]
            closest_context = {
                "address": parsed.closest,
                "geocoded_coordinates": {
                    "longitude": address_coordinates[0],
                    "latitude": address_coordinates[1],
                },
                "distance_unit": "km",
            }
        elif mode in {"shorter", "djk", "compare"}:
            if not parsed.a_end or not parsed.z_end:
                logger.error(locale["error_shorter_endpoints"])
                return 2
            if parsed.cable or parsed.span:
                logger.error(locale["error_cable_diverse_only"])
                return 2
            if mode == "compare":
                selection_candidates = candidates
            else:
                selection_candidates = [find_shortest_route(candidates, parsed.a_end, parsed.z_end)]
        elif parsed.mode == "diverse":
            if parsed.span:
                if parsed.cable or bool(parsed.a_end) != bool(parsed.z_end):
                    logger.error(locale["error_diverse_parameters"])
                    return 2
                output_directory = (base_directory / application["output_directory"]).resolve()
                previous_route = load_built_route_reference(parsed.span, output_directory)
                diversity_details = _format_message(
                    locale,
                    "diversity_span_details",
                    reference=parsed.span,
                    spans=", ".join(previous_route.span_ids),
                )
                selection_candidates = exclude_reference_spans(candidates, previous_route.span_ids)
                if not selection_candidates:
                    raise RoutePlanningError("No route remains after excluding the reference route spans")
                start_point = parsed.a_end or previous_route.a_end
                end_point = parsed.z_end or previous_route.z_end
                available_points = {
                    endpoint.strip().casefold()
                    for span in selection_candidates
                    for endpoint in (span.a_end, span.z_end)
                }
                if start_point.strip().casefold() in available_points and end_point.strip().casefold() in available_points:
                    try:
                        selection_candidates = [
                            find_shortest_route(selection_candidates, start_point, end_point)
                        ]
                    except RoutePlanningError:
                        pass
            elif parsed.cable:
                if not parsed.a_end or not parsed.z_end:
                    logger.error(locale["error_diverse_parameters"])
                    return 2
                diversity_details = _format_message(
                    locale,
                    "diversity_cable_details",
                    cable=parsed.cable,
                )
                selection_candidates = [
                    find_shortest_route(
                        candidates,
                        parsed.a_end,
                        parsed.z_end,
                        excluded_span_id=parsed.cable,
                    )
                ]
            elif parsed.a_end or parsed.z_end:
                logger.error(locale["error_diverse_parameters"])
                return 2
            else:
                selection_candidates = candidates
            if parsed.build and not (parsed.span or (parsed.a_end and parsed.z_end and parsed.cable)):
                logger.error(locale["error_diverse_parameters"])
                return 2
        else:
            if parsed.a_end or parsed.z_end or parsed.cable or parsed.span:
                logger.error(locale["error_route_parameters"])
                return 2
            selection_candidates = candidates

        dijkstra_route = (
            find_shortest_route(candidates, parsed.a_end, parsed.z_end)
            if mode == "compare"
            else None
        )
        if mode == "djk":
            selected = selection_candidates
            system_prompt = ""
            strategy = locale["strategy_dijkstra"]
            model = None
        else:
            prompts = load_prompts(settings, base_directory)
            system_prompt = build_system_prompt(prompts, mode, locale_code)
            provider = parsed.provider_alias or parsed.provider or settings["provider"]
            if provider == "or":
                provider = "openrouter"
            if provider not in {"ollama", "openrouter"}:
                raise ValueError("provider must be 'ollama', 'openrouter', or 'or'")
            provider_settings = settings[provider]
            api_key = None
            if provider == "openrouter":
                api_key = _load_api_key(provider_settings["api_key_file"], base_directory)
            strategy = locale["strategy_ai"]
            model = provider_settings["model"]
            selected = select_routes(
                provider=provider,
                base_url=provider_settings["base_url"],
                model=provider_settings["model"],
                timeout_seconds=int(provider_settings["timeout_seconds"]),
                mode=mode,
                locale=locale_code,
                prompt_config=prompts,
                candidates=selection_candidates,
                api_key=api_key,
                site_url=provider_settings.get("site_url", ""),
                app_name=provider_settings.get("app_name", ""),
                max_retries=int(provider_settings.get("max_retries", 5)) if provider == "openrouter" else 0,
                retry_delay_seconds=float(provider_settings.get("retry_delay_seconds", 2)),
                system_prompt=system_prompt,
                context=(
                    {"a_end": parsed.a_end, "z_end": parsed.z_end}
                    if mode == "compare"
                    else closest_context
                ),
            )
        if mode == "compare":
            ai_route = build_route_from_spans(selected, parsed.a_end, parsed.z_end)
            ai_span_keys = [span_id.strip().casefold() for span_id in ai_route.span_ids]
            dijkstra_span_keys = [span_id.strip().casefold() for span_id in dijkstra_route.span_ids]
            if ai_span_keys != dijkstra_span_keys:
                logger.error(
                    _format_message(
                        locale,
                        "error_compare_mismatch",
                        ai_route=ai_route.route_name,
                        dijkstra_route=dijkstra_route.route_name,
                    )
                )
                return 2
            selected = [dijkstra_route]
            strategy = locale["strategy_compare"]
        analysis_data = {
            "strategy": strategy,
            "model": model,
            "diversity": diversity_details,
        }
        logger.info(
            _format_message(
                locale,
                "log_analysis",
                strategy=strategy,
                model=model or locale["model_not_used"],
                diversity=diversity_details or locale["diversity_none"],
            )
        )
        data_logger.info("Route analysis", extra={"data": analysis_data})
        if mode == "closest" and selected[0].distance_km > settings["closest"]["max_distance_km"]:
            logger.error(locale["error_too_far"])
            return 1
        output_path = parsed.output or (
            base_directory
            / application["output_directory"]
            / f"{parsed.input_file.stem}_{mode}.json"
        )
        output_path = output_path.resolve()
        write_results(selected, output_path)
        if parsed.build or mode == "compare":
            description_lines = [
                _format_message(locale, "kmz_description_strategy", strategy=strategy)
            ]
            if model:
                description_lines.append(_format_message(locale, "kmz_description_model", model=model))
            if diversity_details:
                description_lines.append(
                    _format_message(locale, "kmz_description_diversity", diversity=diversity_details)
                )
            build_identifier, build_json_path, kmz_path = build_route_output_paths(
                selected[0],
                (base_directory / application["output_directory"]).resolve(),
            )
            if parsed.build:
                write_built_route_json(selected[0], build_json_path)
                logger.info(_format_message(locale, "log_output_written", path=build_json_path))
            write_route_kmz(
                selected[0],
                kmz_path,
                build_identifier,
                description="\n".join(description_lines),
            )
            logger.info(_format_message(locale, "log_kmz_written", path=kmz_path))
        result_data = [route.to_result_dict() for route in selected]
        data_logger.info("Selected routes", extra={"data": result_data})
        logger.info(_format_message(locale, "log_output_written", path=output_path))
        if mode == "djk":
            _display_results(result_data)
        else:
            _display_prompt_and_results(system_prompt, result_data)
        return 0
    except ProviderConfigurationError as error:
        message = _format_message(locale, "error_api_key_missing", path=error)
        if "logger" in locals():
            logger.error(message)
        else:
            logging.error(message)
        return 1
    except KmlInputError as error:
        message = _format_message(locale, "error_file_invalid", error=error)
        if "logger" in locals():
            logger.error(message)
        else:
            logging.error(message)
        return 2
    except GeocodingError as error:
        message = _format_message(locale, "error_geocoding", error=error)
        if "logger" in locals():
            logger.error(message)
        else:
            logging.error(message)
        return 1
    except RoutePlanningError as error:
        message = _format_message(locale, "error_route_planning", error=error)
        if "logger" in locals():
            logger.error(message)
        else:
            logging.error(message)
        return 2
    except FileNotFoundError as error:
        if "logger" in locals():
            logger.error(str(error))
        else:
            logging.error(str(error))
        return 2
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        message = _format_message(locale, "error_api", error=error) if isinstance(error, RuntimeError) else _format_message(locale, "error_config", error=error)
        if "logger" in locals():
            logger.error(message, exc_info=parsed.verbose)
        else:
            logging.error(message)
        return 1


def main() -> None:
    raise SystemExit(run())
