"""Logging utilities powered by Loguru."""

from __future__ import annotations

import sys

from loguru import logger as _logger

DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[module]: <20}</cyan> | "
    "<level>{message}</level>"
)


def configure_logging(level: str = "INFO") -> None:
    """Configure the global Loguru logger.

    Parameters
    ----------
    level: str, optional
        Minimum log level to emit. Defaults to ``"INFO"``.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    - Existing handlers are removed before the new configuration is applied.
    """

    _logger.remove()
    _logger.add(sys.stdout, level=level, format=DEFAULT_FORMAT)


def get_logger(module: str):
    """Return a logger bound with module metadata.

    Parameters
    ----------
    module: str
        Module name to attach as structured metadata.

    Returns
    -------
    Logger
        Loguru logger with ``module`` bound for contextual output.

    Raises
    ------
    None

    Notes
    -----
    - Call :func:`configure_logging` once before producing log output.
    """

    return _logger.bind(module=module)


configure_logging()
