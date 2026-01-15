"""Tests for modeling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from main_module.modeling import LinearModelResult, train_linear_model


def test_train_linear_model_returns_result() -> None:
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    target = pd.Series([0.0, 1.0, 2.0])

    result = train_linear_model(features, target)

    assert isinstance(result, LinearModelResult)
    np.testing.assert_allclose(result.coefficients, [1.0], rtol=1e-3)
    assert result.intercept == pytest.approx(0.0, abs=1e-3)


def test_train_linear_model_raises_on_empty_data() -> None:
    with pytest.raises(ValueError):
        train_linear_model(pd.DataFrame(), pd.Series(dtype=float))
