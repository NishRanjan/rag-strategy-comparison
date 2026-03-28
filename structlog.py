"""Minimal local fallback for the subset of structlog used in this repo."""

from __future__ import annotations

import logging
from typing import Any


class BoundLogger:
    """Small adapter that mimics the logging calls used in this project."""

    def __init__(self, logger: logging.Logger) -> None:
        """Store the wrapped standard-library logger."""
        self._logger = logger

    def info(self, event: str, **kwargs: Any) -> None:
        """Log an info event with key-value context."""
        self._logger.info("%s %s", event, kwargs if kwargs else "")

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log a warning event with key-value context."""
        self._logger.warning("%s %s", event, kwargs if kwargs else "")

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log a debug event with key-value context."""
        self._logger.debug("%s %s", event, kwargs if kwargs else "")


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a lightweight logger compatible with structlog.get_logger."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    return BoundLogger(logging.getLogger(name or "ai-sandbox"))

