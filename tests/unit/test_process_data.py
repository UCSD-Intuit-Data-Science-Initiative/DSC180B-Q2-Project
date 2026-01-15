"""Tests for preprocess module."""

from __future__ import annotations

import pandas as pd

from main_module.data import clean_dataframe, select_columns


def test_clean_dataframe_dropna() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})

    cleaned = clean_dataframe(df, dropna=True)

    assert len(cleaned) == 2
    assert cleaned["a"].isna().sum() == 0


def test_select_columns_returns_subset() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    subset = select_columns(df, ["b"])

    assert list(subset.columns) == ["b"]
