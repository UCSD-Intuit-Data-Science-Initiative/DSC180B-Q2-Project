import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from demand_forecasting_model import CallDemandForecaster
from hybrid_forecaster import HybridForecaster


def analyze_data_coverage(filepath):
    print("=" * 70)
    print("DATA COVERAGE ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    date_min = df["Date"].min()
    date_max = df["Date"].max()
    total_days = (date_max - date_min).days
    unique_days = df["Date"].dt.date.nunique()

    print(
        f"\nDate Range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}"
    )
    print(f"Total Days Spanned: {total_days}")
    print(f"Days with Data: {unique_days}")
    print(f"Coverage: {unique_days/total_days*100:.1f}%")

    print("\n--- MONTHLY BREAKDOWN ---")
    df["YearMonth"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("YearMonth").size()

    for period, count in monthly.items():
        print(f"  {period}: {count:,} calls")

    print("\n--- SEASONAL DISTRIBUTION ---")
    df["Month"] = df["Date"].dt.month

    tax_season = df[df["Month"].isin([1, 2, 3, 4])]
    off_season = df[~df["Month"].isin([1, 2, 3, 4])]

    print(
        f"  Tax Season (Jan-Apr): {len(tax_season):,} calls ({len(tax_season)/len(df)*100:.1f}%)"
    )
    print(
        f"  Off Season (May-Dec): {len(off_season):,} calls ({len(off_season)/len(df)*100:.1f}%)"
    )

    return {
        "date_min": date_min,
        "date_max": date_max,
        "total_days": total_days,
        "unique_days": unique_days,
        "monthly_counts": monthly.to_dict(),
    }


def recommend_split(data_info):
    print("\n" + "=" * 70)
    print("RECOMMENDED SPLIT STRATEGIES FOR 2-YEAR DATA")
    print("=" * 70)

    date_min = data_info["date_min"]
    date_max = data_info["date_max"]

    train_end_75 = date_min + pd.Timedelta(
        days=int(data_info["total_days"] * 0.75)
    )
    train_end_80 = date_min + pd.Timedelta(
        days=int(data_info["total_days"] * 0.80)
    )

    print("\n1. STANDARD 80/20 SPLIT (Recommended for initial testing)")
    print(
        f"   Training: {date_min.strftime('%Y-%m-%d')} to {train_end_80.strftime('%Y-%m-%d')}"
    )
    print(
        f"   Testing:  {train_end_80.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}"
    )
    print("   Pros: Good test set size, standard practice")
    print("   Cons: May not cover full seasonality in test")

    print("\n2. SEASONAL SPLIT (Recommended for Intuit)")
    datetime(date_min.year, 7, 1)
    end_year_1 = datetime(date_min.year, 12, 31)
    print(
        f"   Training: {date_min.strftime('%Y-%m-%d')} to ~{end_year_1.strftime('%Y-%m-%d')} (Year 1)"
    )
    print("   Testing:  Year 2 (full year including tax season)")
    print("   Pros: Tests on unseen tax season, realistic deployment scenario")
    print("   Cons: Only 1 tax season in training")

    print("\n3. WALK-FORWARD VALIDATION (Best for robust estimates)")
    print("   Uses 5 expanding windows through Year 2")
    print("   Reports: Mean ± Std of MAE across folds")
    print("   Pros: Uses all data, confidence intervals on metrics")
    print("   Cons: More complex, slower training")

    print("\n4. FULL DATA TRAINING (For production deployment)")
    print(
        f"   Training: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')} (all data)"
    )
    print("   Testing:  Future predictions only")
    print("   Pros: Maximum data for best predictions")
    print("   Cons: No holdout test set to evaluate")

    print("\n" + "-" * 70)
    print("RECOMMENDATION FOR INTUIT CALL CENTER:")
    print("-" * 70)
    print("""
For your 2-year dataset with strong tax seasonality:

1. DEVELOPMENT PHASE: Use Walk-Forward Validation (Option 3)
   - Gives you confidence intervals on model performance
   - Tests across different time periods
   - Run: python train_model.py --strategy walk_forward

2. FINAL VALIDATION: Use 80/20 Split (Option 1)
   - Simple holdout for final sanity check
   - Run: python train_model.py --strategy holdout --test_size 0.20

3. PRODUCTION DEPLOYMENT: Full Data Training (Option 4)
   - Use all 2 years for best predictions
   - Run: python train_model.py --strategy full
   - Monitor performance with model_monitor.py
""")


def train_with_strategy(
    filepath, strategy="walk_forward", test_size=0.20, model_type="hybrid"
):
    print("\n" + "=" * 70)
    print(f"TRAINING WITH STRATEGY: {strategy.upper()}")
    print(f"MODEL TYPE: {model_type.upper()}")
    print("=" * 70)

    if model_type == "hybrid":
        forecaster = HybridForecaster()
        model_path = "hybrid_forecast_model.pkl"
    else:
        forecaster = CallDemandForecaster(model_type="ensemble")
        model_path = "demand_forecast_model.pkl"

    if strategy == "full":
        print("\nTraining on FULL dataset (no holdout test set)")
        print(
            "WARNING: No test metrics available - use model monitoring in production"
        )

        if model_type == "hybrid":
            forecaster.train(filepath)
        else:
            df = forecaster.load_and_preprocess_data(filepath)
            interval_df = forecaster.aggregate_to_intervals(df)
            feature_df = forecaster.create_features(interval_df)
            X, y = forecaster.prepare_training_data(feature_df)

            X_scaled = forecaster.scaler.fit_transform(X)
            models = forecaster._build_models()

            for name, model in models.items():
                print(f"  Training {name}...")
                model.fit(X_scaled, y)
                forecaster.models[name] = model

            from sklearn.ensemble import VotingRegressor
            from sklearn.metrics import mean_absolute_error

            best_models = list(models.keys())[:3]
            ensemble_estimators = [
                (name, models[name]) for name in best_models
            ]
            forecaster.model = VotingRegressor(estimators=ensemble_estimators)
            forecaster.model.fit(X_scaled, y)

            train_pred = np.maximum(forecaster.model.predict(X_scaled), 0)
            print(
                f"\n  Training MAE (on full data): {mean_absolute_error(y, train_pred):.4f}"
            )

        forecaster.is_fitted = True

    elif strategy == "holdout":
        print(
            f"\nUsing {int((1-test_size)*100)}/{int(test_size*100)} time-based split"
        )

        if model_type == "hybrid":
            forecaster.train(filepath)
        else:
            forecaster.train(filepath, test_size=test_size)

    elif strategy == "walk_forward":
        print("\nUsing Walk-Forward Validation (5 folds)")

        if model_type == "hybrid":
            forecaster.train(filepath)
        else:
            forecaster.train(filepath, test_size=test_size)

    forecaster.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    return forecaster


def main():
    parser = argparse.ArgumentParser(
        description="Train demand forecasting model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mock_intuit_2year_data.csv",
        help="Path to data file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="walk_forward",
        choices=["walk_forward", "holdout", "full", "analyze"],
        help="Training strategy",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Test set size for holdout strategy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["hybrid", "single"],
        help="Model type",
    )

    args = parser.parse_args()

    data_info = analyze_data_coverage(args.data)

    if args.strategy == "analyze":
        recommend_split(data_info)
        return

    recommend_split(data_info)

    train_with_strategy(
        args.data,
        strategy=args.strategy,
        test_size=args.test_size,
        model_type=args.model,
    )


if __name__ == "__main__":
    main()
