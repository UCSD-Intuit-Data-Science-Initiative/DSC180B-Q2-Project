"""
train.py
--------
Standalone training script. Run this ONCE to train the HybridForecaster
and save it to forecaster.joblib.

After this file produces forecaster.joblib, the API server (app.py / main.py)
loads the saved model at startup instead of retraining — startup goes from
several minutes down to a few seconds.

Usage:
    # With Docker (recommended):
    docker-compose run --rm backend python src/main_module/api/train.py

    # Locally with Poetry:
    PYTHONPATH=src poetry run python src/main_module/api/train.py
"""

import sys
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is 4 levels up from this file (api/ → main_module/ → src/ → root)
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DATA_PATH = _ROOT / "data" / "interim" / "mock_intuit_2year_data.csv"
_MODEL_OUT = Path(__file__).resolve().parent / "forecaster.joblib"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    if not _DATA_PATH.exists():
        print(f"ERROR: Data file not found at {_DATA_PATH}")
        print("Make sure data/interim/mock_intuit_2year_data.csv exists.")
        sys.exit(1)

    from main_module.workforce import HybridForecaster

    print("=" * 60)
    print("Training HybridForecaster...")
    print(f"  Data : {_DATA_PATH}")
    print(f"  Output: {_MODEL_OUT}")
    print("=" * 60)

    forecaster = HybridForecaster()
    forecaster.train(
        str(_DATA_PATH),
        test_year=2025,
        tune_hyperparameters=True,
        n_trials=10,
    )

    print("=" * 60)
    print("Saving model with joblib...")
    joblib.dump(forecaster, _MODEL_OUT)
    size_mb = _MODEL_OUT.stat().st_size / (1024 * 1024)
    print(f"Saved to {_MODEL_OUT}  ({size_mb:.1f} MB)")
    print("=" * 60)
    print("Done. You can now start the server:")
    print("  PYTHONPATH=src uvicorn src.main_module.api.app:app --port 8001")
    print("  PYTHONPATH=src uvicorn src.main_module.api.main:app --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
