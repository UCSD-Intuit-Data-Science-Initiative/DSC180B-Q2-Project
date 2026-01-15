"""Tests for load_data module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from main_module.data import load_csv


def test_load_csv_reads_dataframe(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.to_csv(data_path, index=False)

    loaded = load_csv(data_path)

    assert loaded.equals(df)


def test_load_csv_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_csv("non-existent.csv")
