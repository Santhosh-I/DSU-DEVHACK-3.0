"""
Plastic-Ledger — Logging Utilities
====================================
Rich-based logger setup for consistent pipeline logging.

Usage:
    from pipeline.utils.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("Processing scene %s", scene_id)
"""

import os
import logging

from rich.logging import RichHandler
from rich.console import Console

console = Console()

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def get_logger(name: str, level: str = None) -> logging.Logger:
    """Create a rich-formatted logger.

    Args:
        name: Logger name (typically ``__name__``).
        level: Override log level.  Falls back to the ``LOG_LEVEL``
            environment variable, then ``INFO``.

    Returns:
        Configured :class:`logging.Logger` instance.

    Raises:
        ValueError: If *level* is not a valid Python log level string.
    """
    effective_level = level or _LOG_LEVEL

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    handler = RichHandler(
        console=console,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, effective_level, logging.INFO))
    logger.propagate = False

    return logger
