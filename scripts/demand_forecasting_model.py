import pickle
import warnings
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


US_HOLIDAYS = {
    (1, 1): "New Year's Day",
    (1, 15): "MLK Day (approx)",
    (2, 19): "Presidents Day (approx)",
    (5, 27): "Memorial Day (approx)",
    (7, 4): "Independence Day",
    (9, 2): "Labor Day (approx)",
    (10, 14): "Columbus Day (approx)",
    (11, 11): "Veterans Day",
    (11, 28): "Thanksgiving (approx)",
    (12, 25): "Christmas Day",
    (12, 31): "New Year's Eve",
}

TAX_IMPORTANT_DATES = {
    (1, 15): "Q4 Estimated Tax Due",
    (1, 31): "W-2/1099 Deadline",
    (4, 15): "Tax Filing Deadline",
    (4, 18): "Tax Filing Deadline (extended)",
    (6, 15): "Q2 Estimated Tax Due",
    (9, 15): "Q3 Estimated Tax Due",
    (10, 15): "Extension Deadline",
}


class CallDemandForecaster:
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.model = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        self.model_metrics = {}

    def load_and_preprocess_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df["Arrival time"] = pd.to_datetime(
            df["Arrival time"], format="%m/%d/%Y %H:%M:%S"
        )
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        return df

    def aggregate_to_intervals(
        self, df: pd.DataFrame, interval_minutes: int = 30
    ) -> pd.DataFrame:
        df = df.copy()
        df["interval_start"] = df["Arrival time"].dt.floor(
            f"{interval_minutes}min"
        )

        interval_counts = (
            df.groupby("interval_start")
            .agg(
                call_count=("Session ID / Contact ID", "count"),
                turbotax_count=(
                    "Product group",
                    lambda x: (x == "TurboTax").sum(),
                ),
                quickbooks_count=(
                    "Product group",
                    lambda x: (x == "QuickBooks").sum(),
                ),
                avg_hold_time=("Hold time during call", "mean"),
                answered_count=("Answered?", lambda x: (x == "Yes").sum()),
            )
            .reset_index()
        )

        date_range = pd.date_range(
            start=df["interval_start"].min(),
            end=df["interval_start"].max(),
            freq=f"{interval_minutes}min",
        )
        full_intervals = pd.DataFrame({"interval_start": date_range})

        interval_counts = full_intervals.merge(
            interval_counts, on="interval_start", how="left"
        )
        interval_counts["call_count"] = (
            interval_counts["call_count"].fillna(0).astype(int)
        )
        interval_counts["turbotax_count"] = (
            interval_counts["turbotax_count"].fillna(0).astype(int)
        )
        interval_counts["quickbooks_count"] = (
            interval_counts["quickbooks_count"].fillna(0).astype(int)
        )
        interval_counts["avg_hold_time"] = interval_counts[
            "avg_hold_time"
        ].fillna(0)
        interval_counts["answered_count"] = (
            interval_counts["answered_count"].fillna(0).astype(int)
        )

        return interval_counts

    def _is_holiday(self, dt: datetime) -> Tuple[bool, str]:
        key = (dt.month, dt.day)
        if key in US_HOLIDAYS:
            return True, US_HOLIDAYS[key]
        return False, ""

    def _is_tax_important_date(self, dt: datetime) -> Tuple[bool, str]:
        key = (dt.month, dt.day)
        if key in TAX_IMPORTANT_DATES:
            return True, TAX_IMPORTANT_DATES[key]
        return False, ""

    def _days_to_tax_deadline(self, dt: datetime) -> int:
        year = dt.year
        tax_deadline = datetime(year, 4, 15)
        if dt > tax_deadline:
            tax_deadline = datetime(year + 1, 4, 15)
        return (tax_deadline - dt).days

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["hour"] = df["interval_start"].dt.hour
        df["minute"] = df["interval_start"].dt.minute
        df["day_of_week"] = df["interval_start"].dt.dayofweek
        df["day_of_month"] = df["interval_start"].dt.day
        df["month"] = df["interval_start"].dt.month
        df["year"] = df["interval_start"].dt.year
        df["week_of_year"] = (
            df["interval_start"].dt.isocalendar().week.astype(int)
        )
        df["quarter"] = df["interval_start"].dt.quarter
        df["day_of_year"] = df["interval_start"].dt.dayofyear

        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_business_hours"] = (
            (df["hour"] >= 5) & (df["hour"] < 17) & (df["day_of_week"] < 5)
        ).astype(int)
        df["is_morning_peak"] = (
            (df["hour"] >= 9) & (df["hour"] <= 11) & (df["day_of_week"] < 5)
        ).astype(int)
        df["is_afternoon_peak"] = (
            (df["hour"] >= 14) & (df["hour"] < 17) & (df["day_of_week"] < 5)
        ).astype(int)
        df["is_lunch_hour"] = (
            (df["hour"] >= 12) & (df["hour"] <= 13) & (df["day_of_week"] < 5)
        ).astype(int)
        df["is_open"] = (
            (df["hour"] >= 5) & (df["hour"] < 17) & (df["day_of_week"] < 5)
        ).astype(int)
        df["is_early_morning"] = (
            (df["hour"] >= 5) & (df["hour"] < 8) & (df["day_of_week"] < 5)
        ).astype(int)

        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        df["is_tax_season"] = ((df["month"] >= 1) & (df["month"] <= 4)).astype(
            int
        )
        df["is_tax_deadline"] = (
            (df["month"] == 4) & (df["day_of_month"] <= 15)
        ).astype(int)
        df["is_tax_deadline_week"] = (
            (df["month"] == 4)
            & (df["day_of_month"] >= 8)
            & (df["day_of_month"] <= 15)
        ).astype(int)
        df["is_extension_deadline"] = (
            (df["month"] == 10)
            & (df["day_of_month"] >= 1)
            & (df["day_of_month"] <= 15)
        ).astype(int)
        df["is_year_end"] = ((df["month"] == 12) | (df["month"] == 1)).astype(
            int
        )

        df["days_to_tax_deadline"] = df["interval_start"].apply(
            self._days_to_tax_deadline
        )
        df["tax_urgency"] = np.maximum(0, 30 - df["days_to_tax_deadline"]) / 30

        df["is_holiday"] = df["interval_start"].apply(
            lambda x: int(self._is_holiday(x)[0])
        )
        df["is_tax_important"] = df["interval_start"].apply(
            lambda x: int(self._is_tax_important_date(x)[0])
        )
        df["is_day_before_holiday"] = df["is_holiday"].shift(-48, fill_value=0)
        df["is_day_after_holiday"] = df["is_holiday"].shift(48, fill_value=0)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        df["time_slot"] = df["hour"] * 2 + (df["minute"] // 30)

        for lag in [1, 2, 3, 4, 48, 96, 336, 672]:
            df[f"lag_{lag}"] = df["call_count"].shift(lag)

        df["lag_same_time_yesterday"] = df["call_count"].shift(48)
        df["lag_same_time_last_week"] = df["call_count"].shift(336)

        for window in [4, 8, 12, 48, 96, 336]:
            df[f"rolling_mean_{window}"] = (
                df["call_count"].rolling(window=window, min_periods=1).mean()
            )
            df[f"rolling_std_{window}"] = (
                df["call_count"].rolling(window=window, min_periods=1).std()
            )
            df[f"rolling_max_{window}"] = (
                df["call_count"].rolling(window=window, min_periods=1).max()
            )
            df[f"rolling_min_{window}"] = (
                df["call_count"].rolling(window=window, min_periods=1).min()
            )

        df["rolling_median_48"] = (
            df["call_count"].rolling(window=48, min_periods=1).median()
        )
        df["rolling_median_336"] = (
            df["call_count"].rolling(window=336, min_periods=1).median()
        )

        df["ewm_mean_12"] = df["call_count"].ewm(span=12, adjust=False).mean()
        df["ewm_mean_48"] = df["call_count"].ewm(span=48, adjust=False).mean()

        df["hourly_trend"] = df["rolling_mean_4"] - df["rolling_mean_48"]
        df["daily_trend"] = df["rolling_mean_48"] - df["rolling_mean_336"]

        df = df.fillna(0)

        return df

    def prepare_training_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        feature_cols = [
            "hour",
            "minute",
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "quarter",
            "day_of_year",
            "is_weekend",
            "is_business_hours",
            "is_morning_peak",
            "is_afternoon_peak",
            "is_lunch_hour",
            "is_open",
            "is_early_morning",
            "is_monday",
            "is_friday",
            "is_tax_season",
            "is_tax_deadline",
            "is_tax_deadline_week",
            "is_extension_deadline",
            "is_year_end",
            "days_to_tax_deadline",
            "tax_urgency",
            "is_holiday",
            "is_tax_important",
            "is_day_before_holiday",
            "is_day_after_holiday",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "doy_sin",
            "doy_cos",
            "time_slot",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_48",
            "lag_96",
            "lag_336",
            "lag_672",
            "lag_same_time_yesterday",
            "lag_same_time_last_week",
            "rolling_mean_4",
            "rolling_std_4",
            "rolling_max_4",
            "rolling_min_4",
            "rolling_mean_8",
            "rolling_std_8",
            "rolling_max_8",
            "rolling_min_8",
            "rolling_mean_12",
            "rolling_std_12",
            "rolling_max_12",
            "rolling_min_12",
            "rolling_mean_48",
            "rolling_std_48",
            "rolling_max_48",
            "rolling_min_48",
            "rolling_mean_96",
            "rolling_std_96",
            "rolling_max_96",
            "rolling_min_96",
            "rolling_mean_336",
            "rolling_std_336",
            "rolling_max_336",
            "rolling_min_336",
            "rolling_median_48",
            "rolling_median_336",
            "ewm_mean_12",
            "ewm_mean_48",
            "hourly_trend",
            "daily_trend",
        ]

        self.feature_columns = feature_cols
        X = df[feature_cols]
        y = df["call_count"]

        return X, y

    def _build_models(self) -> Dict:
        models = {}

        models["gradient_boosting"] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )

        models["random_forest"] = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        models["ridge"] = Ridge(alpha=1.0)

        if HAS_XGBOOST:
            models["xgboost"] = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )

        if HAS_LIGHTGBM:
            models["lightgbm"] = LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
            )

        return models

    def _walk_forward_validation(self, X, y, models, n_splits=5):
        n_samples = len(X)
        min_train_size = int(n_samples * 0.5)
        fold_size = (n_samples - min_train_size) // n_splits

        cv_results = {
            name: {"mae": [], "rmse": [], "r2": []} for name in models.keys()
        }

        print(f"\n  Performing {n_splits}-fold walk-forward validation...")

        for fold in range(n_splits):
            train_end = min_train_size + fold * fold_size
            test_end = min(train_end + fold_size, n_samples)

            X_train_fold = X.iloc[:train_end]
            y_train_fold = y.iloc[:train_end]
            X_test_fold = X.iloc[train_end:test_end]
            y_test_fold = y.iloc[train_end:test_end]

            if len(X_test_fold) == 0:
                continue

            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train_fold)
            X_test_scaled = scaler_fold.transform(X_test_fold)

            for name, model in models.items():
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_scaled, y_train_fold)
                y_pred = np.maximum(model_copy.predict(X_test_scaled), 0)

                cv_results[name]["mae"].append(
                    mean_absolute_error(y_test_fold, y_pred)
                )
                cv_results[name]["rmse"].append(
                    np.sqrt(mean_squared_error(y_test_fold, y_pred))
                )
                cv_results[name]["r2"].append(r2_score(y_test_fold, y_pred))

        return cv_results

    def _verify_distribution(self, train_df, test_df, train_y, test_y):
        print("\n" + "=" * 70)
        print("DISTRIBUTION VERIFICATION: Training vs Testing")
        print("=" * 70)

        issues = []

        train_months = train_df["month"].value_counts(normalize=True).sort_index()
        test_months = test_df["month"].value_counts(normalize=True).sort_index()

        print("\n  Monthly Distribution:")
        print(f"  {'Month':<8} {'Train %':<12} {'Test %':<12} {'Diff':<10} {'Status'}")
        print("  " + "-" * 55)

        for month in range(1, 13):
            train_pct = train_months.get(month, 0) * 100
            test_pct = test_months.get(month, 0) * 100
            diff = abs(train_pct - test_pct)
            status = "✅" if diff < 3 else "⚠️"
            if diff >= 3:
                issues.append(f"Month {month}: {diff:.1f}% difference")
            print(f"  {month:<8} {train_pct:<12.1f} {test_pct:<12.1f} {diff:<10.1f} {status}")

        train_dow = train_df["day_of_week"].value_counts(normalize=True).sort_index()
        test_dow = test_df["day_of_week"].value_counts(normalize=True).sort_index()

        print("\n  Day of Week Distribution:")
        print(f"  {'Day':<8} {'Train %':<12} {'Test %':<12} {'Diff':<10} {'Status'}")
        print("  " + "-" * 55)

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for dow in range(5):
            train_pct = train_dow.get(dow, 0) * 100
            test_pct = test_dow.get(dow, 0) * 100
            diff = abs(train_pct - test_pct)
            status = "✅" if diff < 2 else "⚠️"
            if diff >= 2:
                issues.append(f"{dow_names[dow]}: {diff:.1f}% difference")
            print(f"  {dow_names[dow]:<8} {train_pct:<12.1f} {test_pct:<12.1f} {diff:<10.1f} {status}")

        train_peak = (train_df["is_tax_season"] == 1).mean() * 100
        test_peak = (test_df["is_tax_season"] == 1).mean() * 100
        peak_diff = abs(train_peak - test_peak)

        print("\n  Seasonality Distribution:")
        print(f"  {'Feature':<20} {'Train %':<12} {'Test %':<12} {'Diff':<10} {'Status'}")
        print("  " + "-" * 65)

        status = "✅" if peak_diff < 5 else "⚠️"
        if peak_diff >= 5:
            issues.append(f"Tax season: {peak_diff:.1f}% difference")
        print(f"  {'Tax Season':<20} {train_peak:<12.1f} {test_peak:<12.1f} {peak_diff:<10.1f} {status}")

        if "is_holiday" in train_df.columns:
            train_holiday = (train_df["is_holiday"] == 1).mean() * 100
            test_holiday = (test_df["is_holiday"] == 1).mean() * 100
            holiday_diff = abs(train_holiday - test_holiday)
            status = "✅" if holiday_diff < 2 else "⚠️"
            print(f"  {'Holiday':<20} {train_holiday:<12.1f} {test_holiday:<12.1f} {holiday_diff:<10.1f} {status}")

        train_mean = train_y.mean()
        test_mean = test_y.mean()
        train_std = train_y.std()
        test_std = test_y.std()
        mean_diff_pct = abs(train_mean - test_mean) / train_mean * 100

        print("\n  Call Volume Statistics:")
        print(f"  {'Metric':<20} {'Train':<15} {'Test':<15} {'Diff %':<10} {'Status'}")
        print("  " + "-" * 65)

        status = "✅" if mean_diff_pct < 10 else "⚠️"
        if mean_diff_pct >= 10:
            issues.append(f"Mean calls: {mean_diff_pct:.1f}% difference")
        print(f"  {'Mean calls/interval':<20} {train_mean:<15.2f} {test_mean:<15.2f} {mean_diff_pct:<10.1f} {status}")

        std_diff_pct = abs(train_std - test_std) / train_std * 100
        status = "✅" if std_diff_pct < 15 else "⚠️"
        print(f"  {'Std calls/interval':<20} {train_std:<15.2f} {test_std:<15.2f} {std_diff_pct:<10.1f} {status}")

        print("\n" + "-" * 70)
        if len(issues) == 0:
            print("  ✅ DISTRIBUTION VERIFICATION PASSED")
            print("  Training and testing sets have similar distributions.")
        else:
            print(f"  ⚠️  DISTRIBUTION VERIFICATION: {len(issues)} potential issue(s)")
            print("  Differences found (may affect model generalization):")
            for issue in issues:
                print(f"    - {issue}")
        print("-" * 70)

        return len(issues) == 0

    def train(
        self, filepath: str, test_year: int = 2024
    ) -> Tuple[Dict, pd.DataFrame]:
        print("Loading and preprocessing data...")
        df = self.load_and_preprocess_data(filepath)

        print("Aggregating calls into 30-minute intervals...")
        interval_df = self.aggregate_to_intervals(df)

        print("Creating features...")
        feature_df = self.create_features(interval_df)

        print("Preparing training data...")
        X, y = self.prepare_training_data(feature_df)

        feature_df["year"] = feature_df["interval_start"].dt.year

        train_mask = feature_df["year"] < test_year
        test_mask = feature_df["year"] >= test_year

        X_train = X[train_mask.values]
        X_test = X[test_mask.values]
        y_train = y[train_mask.values]
        y_test = y[test_mask.values]

        train_feature_df = feature_df[train_mask]
        test_feature_df = feature_df[test_mask]

        train_dates = train_feature_df["interval_start"]
        test_dates = test_feature_df["interval_start"]

        print("\n" + "=" * 70)
        print(f"DATA SPLIT: Year-Based (Train on <{test_year}, Test on {test_year})")
        print("=" * 70)
        print(
            f"  Training: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')} ({len(X_train):,} samples)"
        )
        print(
            f"  Testing:  {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')} ({len(X_test):,} samples)"
        )

        train_years = train_feature_df["year"].value_counts().sort_index()
        test_years = test_feature_df["year"].value_counts().sort_index()
        print("\n  Samples by year:")
        for year, count in train_years.items():
            print(f"    {year} (train): {count:,}")
        for year, count in test_years.items():
            print(f"    {year} (test):  {count:,}")

        self._verify_distribution(train_feature_df, test_feature_df, y_train, y_test)

        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = self._build_models()

        cv_results = self._walk_forward_validation(
            X_train, y_train, models, n_splits=5
        )

        print(f"\nTraining {len(models)} models on full training set...")
        model_results = []

        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train_scaled, y_train)

            train_pred = np.maximum(model.predict(X_train_scaled), 0)
            test_pred = np.maximum(model.predict(X_test_scaled), 0)

            cv_mae = cv_results[name]["mae"]
            cv_rmse = cv_results[name]["rmse"]
            cv_r2 = cv_results[name]["r2"]

            metrics = {
                "model": name,
                "train_mae": mean_absolute_error(y_train, train_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "train_r2": r2_score(y_train, train_pred),
                "test_mae": mean_absolute_error(y_test, test_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
                "test_r2": r2_score(y_test, test_pred),
                "cv_mae_mean": np.mean(cv_mae),
                "cv_mae_std": np.std(cv_mae),
                "cv_rmse_mean": np.mean(cv_rmse),
                "cv_r2_mean": np.mean(cv_r2),
            }
            model_results.append(metrics)
            self.models[name] = model

        results_df = pd.DataFrame(model_results).sort_values("cv_mae_mean")

        if self.model_type == "ensemble":
            print("  Building ensemble model...")
            best_models = results_df.head(3)["model"].tolist()
            ensemble_estimators = [
                (name, self.models[name]) for name in best_models
            ]

            self.model = VotingRegressor(estimators=ensemble_estimators)
            self.model.fit(X_train_scaled, y_train)

            train_pred = np.maximum(self.model.predict(X_train_scaled), 0)
            test_pred = np.maximum(self.model.predict(X_test_scaled), 0)

            ensemble_metrics = {
                "model": "ensemble",
                "train_mae": mean_absolute_error(y_train, train_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "train_r2": r2_score(y_train, train_pred),
                "test_mae": mean_absolute_error(y_test, test_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
                "test_r2": r2_score(y_test, test_pred),
                "cv_mae_mean": np.mean(
                    [
                        results_df[results_df["model"] == m][
                            "cv_mae_mean"
                        ].values[0]
                        for m in best_models
                    ]
                ),
                "cv_mae_std": np.mean(
                    [
                        results_df[results_df["model"] == m][
                            "cv_mae_std"
                        ].values[0]
                        for m in best_models
                    ]
                ),
                "cv_rmse_mean": np.mean(
                    [
                        results_df[results_df["model"] == m][
                            "cv_rmse_mean"
                        ].values[0]
                        for m in best_models
                    ]
                ),
                "cv_r2_mean": np.mean(
                    [
                        results_df[results_df["model"] == m][
                            "cv_r2_mean"
                        ].values[0]
                        for m in best_models
                    ]
                ),
            }
            results_df = pd.concat(
                [results_df, pd.DataFrame([ensemble_metrics])],
                ignore_index=True,
            )
        else:
            best_model_name = results_df.iloc[0]["model"]
            self.model = self.models[best_model_name]
            print(f"  Selected best model: {best_model_name}")

        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(
            f"\n{'Model':<20} {'Test MAE':<12} {'Test RMSE':<12} {'Test R2':<10}"
        )
        print("-" * 60)
        for _, row in results_df.iterrows():
            print(
                f"{row['model']:<20} {row['test_mae']:<12.3f} {row['test_rmse']:<12.3f} {row['test_r2']:<10.4f}"
            )

        print("\n" + "-" * 60)
        print("WALK-FORWARD CROSS-VALIDATION RESULTS (5-fold)")
        print("-" * 60)
        print(f"{'Model':<20} {'CV MAE':<15} {'CV RMSE':<12} {'CV R2':<10}")
        print("-" * 60)
        for _, row in results_df.iterrows():
            cv_mae_str = f"{row['cv_mae_mean']:.3f} ± {row['cv_mae_std']:.3f}"
            print(
                f"{row['model']:<20} {cv_mae_str:<15} {row['cv_rmse_mean']:<12.3f} {row['cv_r2_mean']:<10.4f}"
            )

        self.is_fitted = True
        self.interval_df = interval_df[train_mask].copy()
        self.feature_df = train_feature_df.copy()
        self.model_metrics = results_df.to_dict("records")

        final_model = (
            "ensemble"
            if self.model_type == "ensemble"
            else results_df.iloc[0]["model"]
        )
        final_metrics = (
            results_df[results_df["model"] == final_model].iloc[0].to_dict()
        )

        feature_importance = self._get_feature_importance()

        return final_metrics, feature_importance

    def _get_feature_importance(self) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "estimators_") and isinstance(
            self.model.estimators_, list
        ):
            importances = np.zeros(len(self.feature_columns))
            count = 0
            for item in self.model.estimators_:
                if isinstance(item, tuple):
                    name, est = item
                else:
                    est = item
                if hasattr(est, "feature_importances_"):
                    importances += est.feature_importances_
                    count += 1
            if count > 0:
                importances /= count
        else:
            importances = np.zeros(len(self.feature_columns))

        return pd.DataFrame(
            {"feature": self.feature_columns, "importance": importances}
        ).sort_values("importance", ascending=False)

    def predict_with_confidence(
        self, X_scaled: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = []

        if hasattr(self.model, "estimators_") and isinstance(
            self.model.estimators_, list
        ):
            for item in self.model.estimators_:
                if isinstance(item, tuple):
                    name, est = item
                else:
                    est = item
                pred = est.predict(X_scaled)
                predictions.append(pred)
        else:
            predictions.append(self.model.predict(X_scaled))

        predictions = np.array(predictions)
        mean_pred = np.maximum(np.mean(predictions, axis=0), 0)

        if len(predictions) > 1:
            std_pred = np.std(predictions, axis=0)
        else:
            std_pred = mean_pred * 0.15

        lower_bound = np.maximum(mean_pred - 1.96 * std_pred, 0)
        upper_bound = mean_pred + 1.96 * std_pred

        return mean_pred, lower_bound, upper_bound

    def _get_analogous_historical_data(self, target_dt, historical_data):
        target_month = target_dt.month
        target_dt.day
        target_dow = target_dt.dayofweek
        target_hour = target_dt.hour
        target_minute = target_dt.minute

        historical_data = historical_data.copy()
        historical_data["month"] = historical_data["interval_start"].dt.month
        historical_data["day"] = historical_data["interval_start"].dt.day
        historical_data["dow"] = historical_data["interval_start"].dt.dayofweek
        historical_data["hour"] = historical_data["interval_start"].dt.hour
        historical_data["minute"] = historical_data["interval_start"].dt.minute

        same_time_same_dow = historical_data[
            (historical_data["month"] == target_month)
            & (historical_data["dow"] == target_dow)
            & (historical_data["hour"] == target_hour)
            & (historical_data["minute"] == target_minute)
        ]

        if len(same_time_same_dow) > 0:
            return same_time_same_dow

        same_month_dow_hour = historical_data[
            (historical_data["month"] == target_month)
            & (historical_data["dow"] == target_dow)
            & (historical_data["hour"] == target_hour)
        ]

        if len(same_month_dow_hour) > 0:
            return same_month_dow_hour

        same_month_dow = historical_data[
            (historical_data["month"] == target_month)
            & (historical_data["dow"] == target_dow)
        ]

        return (
            same_month_dow
            if len(same_month_dow) > 0
            else historical_data.tail(672)
        )

    def predict_single_interval(
        self, target_datetime, historical_data=None
    ) -> int:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if historical_data is None:
            historical_data = self.feature_df

        target_dt = pd.to_datetime(target_datetime)
        target_dt = target_dt.floor("30min")

        is_holiday, _ = self._is_holiday(target_dt)
        is_tax_important, _ = self._is_tax_important_date(target_dt)

        features = {
            "hour": target_dt.hour,
            "minute": target_dt.minute,
            "day_of_week": target_dt.dayofweek,
            "day_of_month": target_dt.day,
            "month": target_dt.month,
            "week_of_year": target_dt.isocalendar().week,
            "quarter": target_dt.quarter,
            "day_of_year": target_dt.timetuple().tm_yday,
            "is_weekend": int(target_dt.dayofweek >= 5),
            "is_business_hours": int(
                5 <= target_dt.hour < 17 and target_dt.dayofweek < 5
            ),
            "is_morning_peak": int(
                9 <= target_dt.hour <= 11 and target_dt.dayofweek < 5
            ),
            "is_afternoon_peak": int(
                14 <= target_dt.hour < 17 and target_dt.dayofweek < 5
            ),
            "is_lunch_hour": int(
                12 <= target_dt.hour <= 13 and target_dt.dayofweek < 5
            ),
            "is_open": int(
                5 <= target_dt.hour < 17 and target_dt.dayofweek < 5
            ),
            "is_early_morning": int(
                5 <= target_dt.hour < 8 and target_dt.dayofweek < 5
            ),
            "is_monday": int(target_dt.dayofweek == 0),
            "is_friday": int(target_dt.dayofweek == 4),
            "is_tax_season": int(1 <= target_dt.month <= 4),
            "is_tax_deadline": int(
                target_dt.month == 4 and target_dt.day <= 15
            ),
            "is_tax_deadline_week": int(
                target_dt.month == 4 and 8 <= target_dt.day <= 15
            ),
            "is_extension_deadline": int(
                target_dt.month == 10 and 1 <= target_dt.day <= 15
            ),
            "is_year_end": int(target_dt.month in [12, 1]),
            "days_to_tax_deadline": self._days_to_tax_deadline(target_dt),
            "tax_urgency": max(0, 30 - self._days_to_tax_deadline(target_dt))
            / 30,
            "is_holiday": int(is_holiday),
            "is_tax_important": int(is_tax_important),
            "is_day_before_holiday": 0,
            "is_day_after_holiday": 0,
            "hour_sin": np.sin(2 * np.pi * target_dt.hour / 24),
            "hour_cos": np.cos(2 * np.pi * target_dt.hour / 24),
            "dow_sin": np.sin(2 * np.pi * target_dt.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * target_dt.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * target_dt.month / 12),
            "month_cos": np.cos(2 * np.pi * target_dt.month / 12),
            "doy_sin": np.sin(2 * np.pi * target_dt.timetuple().tm_yday / 365),
            "doy_cos": np.cos(2 * np.pi * target_dt.timetuple().tm_yday / 365),
            "time_slot": target_dt.hour * 2 + (target_dt.minute // 30),
        }

        recent_data = historical_data[
            historical_data["interval_start"] < target_dt
        ].tail(672)

        max_historical_date = historical_data["interval_start"].max()
        is_future_prediction = target_dt > max_historical_date

        if is_future_prediction:
            analogous_data = self._get_analogous_historical_data(
                target_dt, historical_data
            )

            if len(analogous_data) > 0:
                avg_calls = analogous_data["call_count"].mean()
                std_calls = (
                    analogous_data["call_count"].std()
                    if len(analogous_data) > 1
                    else avg_calls * 0.2
                )
                max_calls = analogous_data["call_count"].max()
                min_calls = analogous_data["call_count"].min()
                median_calls = analogous_data["call_count"].median()
            else:
                avg_calls = historical_data["call_count"].mean()
                std_calls = historical_data["call_count"].std()
                max_calls = historical_data["call_count"].max()
                min_calls = historical_data["call_count"].min()
                median_calls = historical_data["call_count"].median()

            for lag in [1, 2, 3, 4, 48, 96, 336, 672]:
                features[f"lag_{lag}"] = avg_calls

            features["lag_same_time_yesterday"] = avg_calls
            features["lag_same_time_last_week"] = avg_calls

            for window in [4, 8, 12, 48, 96, 336]:
                features[f"rolling_mean_{window}"] = avg_calls
                features[f"rolling_std_{window}"] = std_calls
                features[f"rolling_max_{window}"] = max_calls
                features[f"rolling_min_{window}"] = min_calls

            features["rolling_median_48"] = median_calls
            features["rolling_median_336"] = median_calls

            features["ewm_mean_12"] = avg_calls
            features["ewm_mean_48"] = avg_calls

            features["hourly_trend"] = 0
            features["daily_trend"] = 0

        elif len(recent_data) > 0:
            for lag in [1, 2, 3, 4, 48, 96, 336, 672]:
                features[f"lag_{lag}"] = (
                    recent_data["call_count"].iloc[-lag]
                    if len(recent_data) >= lag
                    else 0
                )

            features["lag_same_time_yesterday"] = (
                recent_data["call_count"].iloc[-48]
                if len(recent_data) >= 48
                else 0
            )
            features["lag_same_time_last_week"] = (
                recent_data["call_count"].iloc[-336]
                if len(recent_data) >= 336
                else 0
            )

            for window in [4, 8, 12, 48, 96, 336]:
                window_data = recent_data["call_count"].tail(window)
                features[f"rolling_mean_{window}"] = window_data.mean()
                features[f"rolling_std_{window}"] = (
                    window_data.std() if len(window_data) > 1 else 0
                )
                features[f"rolling_max_{window}"] = window_data.max()
                features[f"rolling_min_{window}"] = window_data.min()

            features["rolling_median_48"] = (
                recent_data["call_count"].tail(48).median()
            )
            features["rolling_median_336"] = (
                recent_data["call_count"].tail(336).median()
            )

            features["ewm_mean_12"] = (
                recent_data["call_count"]
                .ewm(span=12, adjust=False)
                .mean()
                .iloc[-1]
            )
            features["ewm_mean_48"] = (
                recent_data["call_count"]
                .ewm(span=48, adjust=False)
                .mean()
                .iloc[-1]
            )

            features["hourly_trend"] = (
                features["rolling_mean_4"] - features["rolling_mean_48"]
            )
            features["daily_trend"] = (
                features["rolling_mean_48"] - features["rolling_mean_336"]
            )
        else:
            lag_features = [
                "lag_1",
                "lag_2",
                "lag_3",
                "lag_4",
                "lag_48",
                "lag_96",
                "lag_336",
                "lag_672",
                "lag_same_time_yesterday",
                "lag_same_time_last_week",
            ]
            for f in lag_features:
                features[f] = 0

            for window in [4, 8, 12, 48, 96, 336]:
                features[f"rolling_mean_{window}"] = 0
                features[f"rolling_std_{window}"] = 0
                features[f"rolling_max_{window}"] = 0
                features[f"rolling_min_{window}"] = 0

            features["rolling_median_48"] = 0
            features["rolling_median_336"] = 0
            features["ewm_mean_12"] = 0
            features["ewm_mean_48"] = 0
            features["hourly_trend"] = 0
            features["daily_trend"] = 0

        for key in features:
            if pd.isna(features[key]):
                features[key] = 0

        X = pd.DataFrame([features])[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        prediction = max(0, round(prediction))

        return prediction

    def predict_day(self, target_date: str) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        target_date = pd.to_datetime(target_date).normalize()
        day_of_week = target_date.dayofweek
        intervals = pd.date_range(start=target_date, periods=48, freq="30min")

        is_weekend = day_of_week >= 5

        predictions = []
        for interval in intervals:
            hour = interval.hour
            is_open = (not is_weekend) and (5 <= hour < 17)

            if is_open:
                pred = self.predict_single_interval(interval)
            else:
                pred = 0

            predictions.append(
                {
                    "interval_start": interval,
                    "time": interval.strftime("%H:%M"),
                    "predicted_calls": pred,
                    "is_open": is_open,
                }
            )

        result_df = pd.DataFrame(predictions)

        is_holiday, holiday_name = self._is_holiday(target_date)
        is_tax_date, tax_date_name = self._is_tax_important_date(target_date)

        print(f"\n{'='*60}")
        print(f"DEMAND FORECAST FOR {target_date.strftime('%A, %B %d, %Y')}")
        print("Operating Hours: Monday-Friday, 5:00 AM - 5:00 PM PT")
        if is_weekend:
            print("  ** CLOSED - WEEKEND **")
        if is_holiday:
            print(f"  ** HOLIDAY: {holiday_name} **")
        if is_tax_date:
            print(f"  ** TAX DATE: {tax_date_name} **")
        print(f"{'='*60}")
        print(
            f"{'Time':<10} {'Predicted Calls':<15} {'Staffing Level':<15} {'Status':<10}"
        )
        print("-" * 60)

        for _, row in result_df.iterrows():
            calls = row["predicted_calls"]
            is_open = row["is_open"]
            status = "OPEN" if is_open else "CLOSED"

            if not is_open:
                staffing = "N/A"
            elif calls == 0:
                staffing = "Minimal"
            elif calls <= 5:
                staffing = "Low"
            elif calls <= 15:
                staffing = "Medium"
            elif calls <= 30:
                staffing = "High"
            else:
                staffing = "Maximum"
            print(f"{row['time']:<10} {calls:<15} {staffing:<15} {status:<10}")

        open_intervals = result_df[result_df["is_open"]]
        total_calls = result_df["predicted_calls"].sum()

        if (
            len(open_intervals) > 0
            and open_intervals["predicted_calls"].sum() > 0
        ):
            peak_idx = open_intervals["predicted_calls"].idxmax()
            peak_time = result_df.loc[peak_idx, "time"]
            peak_calls = open_intervals["predicted_calls"].max()
        else:
            peak_time = "N/A"
            peak_calls = 0

        print("-" * 60)
        print(f"Total predicted calls for the day: {total_calls}")
        if not is_weekend:
            print(f"Peak time: {peak_time} with {peak_calls} predicted calls")

        return result_df

    def save_model(self, filepath: str = "demand_forecast_model.pkl"):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        model_data = {
            "model": self.model,
            "models": self.models,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "feature_df": self.feature_df,
            "model_metrics": self.model_metrics,
            "model_type": self.model_type,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str = "demand_forecast_model.pkl"):
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.models = model_data.get("models", {})
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.feature_df = model_data["feature_df"]
        self.model_metrics = model_data.get("model_metrics", [])
        self.model_type = model_data.get("model_type", "ensemble")
        self.is_fitted = True

        print(f"Model loaded from {filepath}")


def main():
    forecaster = CallDemandForecaster(model_type="ensemble")

    metrics, feature_importance = forecaster.train(
        "mock_intuit_2year_data.csv"
    )

    forecaster.save_model("demand_forecast_model.pkl")

    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    test_dates = [
        "2025-04-15",
        "2025-07-15",
        "2025-01-15",
    ]

    for date in test_dates:
        forecaster.predict_day(date)
        print()

    print("\nSingle interval prediction example:")
    single_pred = forecaster.predict_single_interval("2025-04-15 10:30:00")
    print(f"Predicted calls for April 15, 2025 at 10:30 AM: {single_pred}")


if __name__ == "__main__":
    main()
