"""Model evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from main_module.logging import get_logger

logger = get_logger("main_module.modeling.evaluate")


def mean_squared_error(predictions: np.ndarray, target: pd.Series) -> float:
    """Compute the mean squared error between predictions and target.

    Parameters
    ----------
    predictions: np.ndarray
        Model predictions.
    target: pd.Series
        Ground-truth target values.

    Returns
    -------
    float
        Mean squared error between predictions and target.

    Raises
    ------
    None

    Notes
    -----
    - Uses NumPy for the vectorised computation.
    """

    residuals = predictions.astype(float) - target.to_numpy(dtype=float)
    logger.debug("Computed residuals", count=residuals.size)
    return float(np.mean(residuals**2))
