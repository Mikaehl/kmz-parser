import json
import logging
from pathlib import Path
from typing import Any


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "data"):
            payload["data"] = record.data
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_directory: Path, level: str) -> tuple[logging.Logger, logging.Logger]:
    log_directory.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    app_logger = logging.getLogger("kmz_parser")
    data_logger = logging.getLogger("kmz_parser.data")
    app_logger.handlers.clear()
    data_logger.handlers.clear()
    app_logger.setLevel(numeric_level)
    data_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    data_logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    info_handler = logging.FileHandler(log_directory / "info.log", encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    error_handler = logging.FileHandler(log_directory / "error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    app_logger.addHandler(info_handler)
    app_logger.addHandler(error_handler)
    app_logger.addHandler(console_handler)

    data_handler = logging.FileHandler(log_directory / "data.log", encoding="utf-8")
    data_handler.setFormatter(JsonLineFormatter())
    data_logger.addHandler(data_handler)
    return app_logger, data_logger
