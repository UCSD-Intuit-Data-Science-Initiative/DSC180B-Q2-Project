"""Time-related helpers."""

from __future__ import annotations

from datetime import datetime

from main_module.logging import get_logger

logger = get_logger("main_module.utils.time")


def timestamp() -> str:
    """Return an ISO-formatted timestamp for logging and filenames.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Coordinated Universal Time (UTC) timestamp in ISO format.

    Raises
    ------
    None

    Notes
    -----
    - Uses ``datetime.utcnow`` with second precision and a trailing ``Z``.
    """

    value = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    logger.debug("Generated timestamp", value=value)
    return value
