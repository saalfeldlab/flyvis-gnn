"""Centralized logging setup for flyvis-gnn.

Usage::

    from flyvis_gnn.log import get_logger
    logger = get_logger(__name__)
    logger.info("training started")
    logger.debug("batch loss: %f", loss)
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with a stdout StreamHandler (added once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
