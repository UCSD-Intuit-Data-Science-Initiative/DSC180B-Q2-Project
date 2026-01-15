"""Prediction helpers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from main_module.logging import get_logger

logger = get_logger("main_module.modeling.predict")


class SupportsPredict(Protocol):
    """Protocol for models exposing a predict method.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    - Implementations must expose ``predict(features: np.ndarray) ->
      np.ndarray``.
    """

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:  # pragma: no cover - protocol
        """Predict outputs for provided features."""


def run_predictions(
    model: SupportsPredict,
    features: pd.DataFrame,
) -> np.ndarray:
    """Run predictions given a trained model and feature matrix.

    Parameters
    ----------
    model: SupportsPredict
        Trained model implementing a ``predict`` method.
    features: pd.DataFrame
        Feature matrix compatible with the model.

    Returns
    -------
    np.ndarray
        Model predictions derived from ``features``.

    Raises
    ------
    None

    Notes
    -----
    - Converts the DataFrame to a NumPy array before calling ``predict``.
    """

    logger.debug("Running predictions", rows=len(features))
    return model.predict(features.to_numpy())
