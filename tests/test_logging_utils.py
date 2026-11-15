from __future__ import annotations

import logging

from reviewlens.logging_utils import PACKAGE_LOGGER, setup_logging


def test_setup_logging_sets_level() -> None:
    logger = setup_logging("debug")
    assert logger.name == PACKAGE_LOGGER
    assert logger.level == logging.DEBUG


def test_setup_logging_is_idempotent() -> None:
    setup_logging("INFO")
    setup_logging("INFO")
    handlers = logging.getLogger(PACKAGE_LOGGER).handlers
    assert len(handlers) == 1


def test_child_loggers_inherit_level() -> None:
    setup_logging("WARNING")
    child = logging.getLogger(f"{PACKAGE_LOGGER}.data.loader")
    assert child.getEffectiveLevel() == logging.WARNING
