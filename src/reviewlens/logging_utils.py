"""Logging setup shared by the CLI, the experiments and the app.

Modules never print; they log with ``logging.getLogger(__name__)``. Because every module lives
under the ``reviewlens`` package, configuring the ``reviewlens`` logger once controls them all,
while third-party libraries keep their own (quieter) defaults.

Messages use a ``key=value`` style so they stay easy to grep, e.g.
``logger.info("Loaded dataset rows=%d path=%s", n_rows, path)``.
"""

from __future__ import annotations

import logging

PACKAGE_LOGGER = "reviewlens"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_NAME = "reviewlens-console"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure the package logger. Safe to call repeatedly (no duplicate handlers)."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level.upper())

    if not any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler()  # writes to stderr, keeping stdout for command output
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)

    return logger
