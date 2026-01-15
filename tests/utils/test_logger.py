from __future__ import annotations

import logging
import os
import warnings

__all__ = ["get_logger"]

loggers: dict[str, logging.Logger] = {}
log_level = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str, level: str = log_level) -> logging.Logger:
    """Get a logger for each module.

    param name: module name
    param level: log level, default to INFO
    """
    if found_logger := loggers.get(name):
        return found_logger
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        warnings.warn(f"invalid log level: {level}, use INFO instaed.", stacklevel=1)
        level = "INFO"

    logger = logging.getLogger(name)

    logger.setLevel(level)
    if not logger.hasHandlers():
        log_format = logging.Formatter(
            "[%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d] %(message)s"
        )
        log_handler = logging.StreamHandler()
        log_handler.setFormatter(log_format)
        logger.addHandler(log_handler)

    loggers[name] = logger
    return logger
