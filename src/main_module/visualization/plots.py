"""Visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(
    series: pd.Series,
    *,
    bins: int = 20,
) -> plt.Axes:
    """Plot a histogram of a pandas Series and return the axis.

    Parameters
    ----------
    series: pd.Series
        Series to visualise.
    bins: int, optional
        Number of histogram bins. Defaults to ``20``.

    Returns
    -------
    plt.Axes
        Matplotlib axis containing the histogram.

    Raises
    ------
    None

    Notes
    -----
    - Generates a new figure each time the function is called.
    """

    fig, ax = plt.subplots()
    series.plot(kind="hist", bins=bins, ax=ax)
    title = str(series.name) if series.name is not None else "Distribution"
    ax.set_title(title)
    ax.set_xlabel(title if series.name is not None else "Value")
    ax.set_ylabel("Frequency")
    return ax
