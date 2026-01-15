"""Data processing helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from main_module.logging import get_logger

logger = get_logger("main_module.data.preprocess")


def clean_dataframe(
    df: pd.DataFrame,
    *,
    dropna: bool = True,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Clean a DataFrame by dropping NA rows and selecting optional columns.

    Parameters
    ----------
    df: pd.DataFrame
        Input data to clean.
    dropna: bool, optional
        Whether to remove rows containing NA values. Defaults to ``True``.
    columns: Sequence[str] | None, optional
        Subset of column names to retain. Defaults to ``None`` (keep all
        columns).

    Returns
    -------
    pd.DataFrame
        A cleaned copy of the source DataFrame.

    Raises
    ------
    None

    Notes
    -----
    - The input ``df`` is not modified in place.
    """

    cleaned = df.copy()
    logger.debug("Created DataFrame copy", rows=len(cleaned))
    if dropna:
        cleaned = cleaned.dropna()
        logger.debug("Dropped NA rows", rows=len(cleaned))
    if columns is not None:
        cleaned = cleaned.loc[:, list(columns)]
        logger.debug("Selected columns", columns=list(columns))
    return cleaned


def select_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a DataFrame restricted to selected columns.

    Parameters
    ----------
    df: pd.DataFrame
        Source DataFrame.
    columns: Sequence[str]
        Column names to retain.

    Returns
    -------
    pd.DataFrame
        A copy containing only the requested columns.

    Raises
    ------
    None

    Notes
    -----
    - Raises ``KeyError`` if any requested column is missing; behaviour is
      consistent with ``DataFrame.loc``.
    """

    logger.debug("Selecting columns", columns=list(columns))
    return df.loc[:, list(columns)].copy()
