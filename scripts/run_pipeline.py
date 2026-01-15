"""Example training pipeline entrypoint.

This scaffold demonstrates how to wire together the utility and modeling
layers provided by ``main_module``. Customize the data loading and model
training steps to match your project requirements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_module import (
    LinearModelResult,
    clean_dataframe,
    load_csv,
    train_linear_model,
)
from main_module.logging import get_logger

logger = get_logger("scripts.run_pipeline")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Populated namespace containing command-line settings.

    Raises
    ------
    None

    Notes
    -----
    - Extend this parser with additional options as pipelines evolve.
    """

    parser = argparse.ArgumentParser(description="Train linear model pipeline")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Name of the target column",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="""
        Generate synthetic data and run the pipeline without an input file.
        """,
    )
    return parser.parse_args()


def main() -> None:
    """Execute the training workflow.

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
    - Replace preprocessing/modeling logic with project-specific
      modules as needed.
    """

    args = parse_args()

    if args.demo:
        target_column = args.target
        logger.info("Generating synthetic data for demo run")
        df = pd.DataFrame(
            {
                "feature_1": [0.0, 1.0, 2.0, 3.0],
                "feature_2": [1.0, 0.0, 1.0, 0.0],
                target_column: [0.0, 1.5, 3.2, 4.8],
            }
        )
    else:
        if args.data is None or args.target is None:
            msg = "Either provide --demo or both --data and --target."
            raise ValueError(msg)
        logger.info("Loading dataset", path=str(args.data))
        df = load_csv(args.data)
        target_column = args.target

    features = clean_dataframe(df.drop(columns=[target_column]))
    target = clean_dataframe(df[[target_column]])[target_column]
    result: LinearModelResult = train_linear_model(features, target)

    logger.info("Trained linear model", intercept=result.intercept)
    for name, coef in zip(
        result.feature_names,
        result.coefficients,
        strict=True,
    ):
        logger.info("Coefficient", feature=name, value=coef)


if __name__ == "__main__":
    main()
