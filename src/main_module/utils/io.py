"""I/O helpers for the data science template."""

from __future__ import annotations

from pathlib import Path

from main_module.logging import get_logger

logger = get_logger("main_module.utils.io")


def ensure_directory(path: Path | str) -> Path:
    """Ensure a directory exists and return its path.

    Parameters
    ----------
    path: Path | str
        Directory location to create if missing.

    Returns
    -------
    Path
        Path object pointing to the ensured directory.

    Raises
    ------
    None

    Notes
    -----
    - Uses ``Path.mkdir`` with ``parents=True`` and ``exist_ok=True``.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists", path=str(directory))
    return directory
