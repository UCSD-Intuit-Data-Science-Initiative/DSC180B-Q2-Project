"""Baseline modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from main_module.logging import get_logger

logger = get_logger("main_module.modeling.train")


@dataclass(slots=True)
class LinearModelResult:
    """Output of a simple linear regression fit.

    Parameters
    ----------
    coefficients: np.ndarray
        Array of fitted coefficients.
    intercept: float
        Learned bias term.
    feature_names: list[str]
        Names corresponding to each coefficient.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    - Instances are lightweight containers for reporting model parameters.
    """

    coefficients: np.ndarray
    intercept: float
    feature_names: list[str]


def train_linear_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> LinearModelResult:
    """Fit a linear regression using the normal equation.

    Parameters
    ----------
    features: pd.DataFrame
        Design matrix of explanatory variables.
    target: pd.Series
        Response variable to model.

    Returns
    -------
    LinearModelResult
        Container with fitted coefficients, intercept, and feature names.

    Raises
    ------
    ValueError
        If ``features`` or ``target`` is empty.

    Notes
    -----
    - The normal equation is ``theta = (X^T X)^{-1} X^T y``.
    - Suitable for small to medium-sized datasets; large matrices may be
      slow or unstable.
    """

    if features.empty:
        logger.error("Features DataFrame is empty")
        raise ValueError("features DataFrame must not be empty")
    if target.empty:
        logger.error("Target Series is empty")
        raise ValueError("target Series must not be empty")

    logger.info(
        "Training linear model",
        rows=len(features),
        columns=list(features.columns),
    )

    x = np.column_stack([np.ones(len(features)), features.to_numpy()])
    y = target.to_numpy()
    theta = np.linalg.pinv(x.T @ x) @ x.T @ y

    intercept = float(theta[0])
    coefficients = theta[1:]
    logger.debug(
        "Model trained",
        intercept=intercept,
        coefficients=coefficients.tolist(),
    )
    return LinearModelResult(
        coefficients=coefficients,
        intercept=intercept,
        feature_names=list(features.columns),
    )
