"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from main_module.logging import get_logger

logger = get_logger("main_module.data.load_data")


def load_csv(
    path: Path | str,
    *,
    parse_dates: bool | list[str] = False,
) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path: Path | str
        Location of the CSV file on disk.
    parse_dates: bool | list[str], optional
        Column names to parse as dates, or ``True`` to infer automatically.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed CSV data.

    Raises
    ------
    FileNotFoundError
        Raised when the provided ``path`` does not exist.

    Notes
    -----
    - This helper uses ``pandas.read_csv`` under the hood.
    """

    csv_path = Path(path)
    logger.debug("Attempting to load CSV", path=str(csv_path))
    if not csv_path.exists():
        msg = f"CSV file not found: {csv_path}"
        raise FileNotFoundError(msg)

    return pd.read_csv(csv_path, parse_dates=parse_dates)
