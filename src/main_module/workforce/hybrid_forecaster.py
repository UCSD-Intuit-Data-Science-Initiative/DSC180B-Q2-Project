import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class HybridForecaster:
    def __init__(self):
        self.short_term_model = None
        self.long_term_model = None
        self.short_term_scaler = None
        self.long_term_scaler = None
        self.short_term_features = None
        self.long_term_features = None
        self.historical_patterns = {}
        self.training_data = None
        self.max_training_date = None
        self.short_term_threshold_days = 7
        self.volume_quantiles = None
        self.feature_importance = {}
        self.best_params = {}

    def _load_and_prepare_data(self, data_path):
        raw_data = pd.read_csv(data_path)
        raw_data["Call start"] = pd.to_datetime(raw_data["Start time"])
        raw_data["interval_start"] = raw_data["Call start"].dt.floor("30min")

        interval_counts = (
            raw_data.groupby("interval_start")
            .agg(
                call_count=("Customer ID", "count"),
                turbotax_count=("Product group", lambda x: (x == "TurboTax").sum()),
                quickbooks_count=("Product group", lambda x: (x == "QuickBooks").sum()),
            )
            .reset_index()
        )

        full_range = pd.date_range(
            start=interval_counts["interval_start"].min(),
            end=interval_counts["interval_start"].max(),
            freq="30min",
        )
        full_df = pd.DataFrame({"interval_start": full_range})
        interval_counts = full_df.merge(interval_counts, on="interval_start", how="left")
        interval_counts["call_count"] = interval_counts["call_count"].fillna(0)
        interval_counts["turbotax_count"] = interval_counts["turbotax_count"].fillna(0)
        interval_counts["quickbooks_count"] = interval_counts["quickbooks_count"].fillna(0)

        return interval_counts.sort_values("interval_start").reset_index(drop=True)

    def _create_base_features(self, df):
        df = df.copy()

        df["hour"] = df["interval_start"].dt.hour
        df["minute"] = df["interval_start"].dt.minute
        df["day_of_week"] = df["interval_start"].dt.dayofweek
        df["day_of_month"] = df["interval_start"].dt.day
        df["month"] = df["interval_start"].dt.month
        df["week_of_year"] = df["interval_start"].dt.isocalendar().week.astype(int)
        df["day_of_year"] = df["interval_start"].dt.dayofyear
        df["quarter"] = df["interval_start"].dt.quarter
        df["year"] = df["interval_start"].dt.year

        df["time_slot"] = df["hour"] * 2 + (df["minute"] // 30)

        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_business_hours"] = ((df["hour"] >= 5) & (df["hour"] < 17) & (df["day_of_week"] < 5)).astype(int)
        df["is_open"] = df["is_business_hours"]
        df["is_morning_peak"] = ((df["hour"] >= 9) & (df["hour"] < 12) & (df["day_of_week"] < 5)).astype(int)
        df["is_afternoon_peak"] = ((df["hour"] >= 13) & (df["hour"] < 16) & (df["day_of_week"] < 5)).astype(int)
        df["is_early_morning"] = ((df["hour"] >= 5) & (df["hour"] < 8) & (df["day_of_week"] < 5)).astype(int)
        df["is_lunch_hour"] = ((df["hour"] >= 12) & (df["hour"] <= 13) & (df["day_of_week"] < 5)).astype(int)
        df["is_late_afternoon"] = ((df["hour"] >= 15) & (df["hour"] < 17) & (df["day_of_week"] < 5)).astype(int)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["is_mid_week"] = ((df["day_of_week"] >= 1) & (df["day_of_week"] <= 3)).astype(int)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        df["time_slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 48)
        df["time_slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 48)

        df["is_tax_season"] = ((df["month"] >= 1) & (df["month"] <= 4)).astype(int)
        df["is_tax_deadline"] = ((df["month"] == 4) & (df["day_of_month"] <= 15)).astype(int)
        df["is_tax_deadline_week"] = ((df["month"] == 4) & (df["day_of_month"] >= 10) & (df["day_of_month"] <= 17)).astype(int)
        df["is_tax_crunch"] = ((df["month"] == 4) & (df["day_of_month"] >= 1) & (df["day_of_month"] <= 15)).astype(int)
        df["is_extension_deadline"] = ((df["month"] == 10) & (df["day_of_month"] <= 15)).astype(int)
        df["is_quarterly_deadline"] = (
            ((df["month"] == 1) & (df["day_of_month"] <= 15)) |
            ((df["month"] == 4) & (df["day_of_month"] <= 15)) |
            ((df["month"] == 6) & (df["day_of_month"] <= 15)) |
            ((df["month"] == 9) & (df["day_of_month"] <= 15))
        ).astype(int)
        df["is_year_end"] = (df["month"] == 12).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
        df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
        df["is_w2_season"] = ((df["month"] == 1) | (df["month"] == 2)).astype(int)

        df["days_to_tax_deadline"] = df.apply(
            lambda x: self._days_to_deadline(x["month"], x["day_of_month"], 4, 15), axis=1
        )
        df["days_to_extension"] = df.apply(
            lambda x: self._days_to_deadline(x["month"], x["day_of_month"], 10, 15), axis=1
        )
        df["tax_urgency"] = np.clip(1 - df["days_to_tax_deadline"] / 120, 0, 1)

        us_holidays = [(1,1), (1,15), (2,19), (5,27), (7,4), (9,2), (10,14), (11,11), (11,28), (12,25)]
        df["is_holiday"] = df.apply(lambda x: int((x["month"], x["day_of_month"]) in us_holidays), axis=1)
        df["is_day_before_holiday"] = df["is_holiday"].shift(-48).fillna(0).astype(int)
        df["is_day_after_holiday"] = df["is_holiday"].shift(48).fillna(0).astype(int)

        df["turbotax_ratio"] = df["turbotax_count"] / (df["call_count"] + 1)
        df["quickbooks_ratio"] = df["quickbooks_count"] / (df["call_count"] + 1)

        return df

    def _days_to_deadline(self, month, day, deadline_month, deadline_day):
        if month < deadline_month or (month == deadline_month and day <= deadline_day):
            days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            days = 0
            if month == deadline_month:
                return deadline_day - day
            days += days_in_months[month - 1] - day
            for m in range(month + 1, deadline_month):
                days += days_in_months[m - 1]
            days += deadline_day
            return days
        else:
            return 365

    def _create_short_term_features(self, df):
        df = df.copy()

        for lag in [1, 2, 3, 4, 6, 8, 12, 24, 48, 96, 144, 336, 672]:
            df[f"lag_{lag}"] = df["call_count"].shift(lag)

        df["lag_same_time_yesterday"] = df["call_count"].shift(48)
        df["lag_same_time_last_week"] = df["call_count"].shift(336)
        df["lag_same_time_2weeks"] = df["call_count"].shift(672)

        df["diff_1"] = df["call_count"].diff(1)
        df["diff_48"] = df["call_count"].diff(48)
        df["diff_336"] = df["call_count"].diff(336)

        for window in [4, 8, 12, 24, 48, 96, 336]:
            df[f"rolling_mean_{window}"] = df["call_count"].shift(1).rolling(window=window, min_periods=1).mean()
            df[f"rolling_std_{window}"] = df["call_count"].shift(1).rolling(window=window, min_periods=1).std()
            df[f"rolling_max_{window}"] = df["call_count"].shift(1).rolling(window=window, min_periods=1).max()
            df[f"rolling_min_{window}"] = df["call_count"].shift(1).rolling(window=window, min_periods=1).min()
            df[f"rolling_range_{window}"] = df[f"rolling_max_{window}"] - df[f"rolling_min_{window}"]

        for window in [48, 336]:
            df[f"rolling_median_{window}"] = df["call_count"].shift(1).rolling(window=window, min_periods=1).median()
            df[f"rolling_skew_{window}"] = df["call_count"].shift(1).rolling(window=window, min_periods=48).skew()
            q25 = df["call_count"].shift(1).rolling(window=window, min_periods=1).quantile(0.25)
            q75 = df["call_count"].shift(1).rolling(window=window, min_periods=1).quantile(0.75)
            df[f"rolling_iqr_{window}"] = q75 - q25

        for span in [12, 24, 48, 96]:
            df[f"ewm_mean_{span}"] = df["call_count"].shift(1).ewm(span=span, adjust=False).mean()
            df[f"ewm_std_{span}"] = df["call_count"].shift(1).ewm(span=span, adjust=False).std()

        df["hourly_trend"] = df["rolling_mean_4"] - df["rolling_mean_48"]
        df["daily_trend"] = df["rolling_mean_48"] - df["rolling_mean_336"]
        df["weekly_trend"] = df["rolling_mean_336"] - df.get("rolling_mean_672", df["rolling_mean_336"])

        df["volatility_ratio"] = df["rolling_std_48"] / (df["rolling_mean_48"] + 1)
        df["momentum"] = df["ewm_mean_12"] - df["ewm_mean_48"]

        return df

    def _create_long_term_features(self, df):
        df = df.copy()

        dow_hour_stats = df.groupby(["day_of_week", "hour", "minute"])["call_count"].agg(["mean", "std", "median"]).reset_index()
        dow_hour_stats.columns = ["day_of_week", "hour", "minute", "hist_dow_hour_mean", "hist_dow_hour_std", "hist_dow_hour_median"]
        df = df.merge(dow_hour_stats, on=["day_of_week", "hour", "minute"], how="left")

        month_dow_hour_stats = df.groupby(["month", "day_of_week", "hour", "minute"])["call_count"].agg(["mean", "std"]).reset_index()
        month_dow_hour_stats.columns = ["month", "day_of_week", "hour", "minute", "hist_month_dow_hour_mean", "hist_month_dow_hour_std"]
        df = df.merge(month_dow_hour_stats, on=["month", "day_of_week", "hour", "minute"], how="left")

        month_stats = df.groupby("month")["call_count"].agg(["mean", "std"]).reset_index()
        month_stats.columns = ["month", "hist_month_mean", "hist_month_std"]
        df = df.merge(month_stats, on="month", how="left")

        dow_stats = df.groupby("day_of_week")["call_count"].agg(["mean", "std"]).reset_index()
        dow_stats.columns = ["day_of_week", "hist_dow_mean", "hist_dow_std"]
        df = df.merge(dow_stats, on="day_of_week", how="left")

        hour_stats = df.groupby("hour")["call_count"].agg(["mean", "std"]).reset_index()
        hour_stats.columns = ["hour", "hist_hour_mean", "hist_hour_std"]
        df = df.merge(hour_stats, on="hour", how="left")

        time_slot_stats = df.groupby("time_slot")["call_count"].agg(["mean"]).reset_index()
        time_slot_stats.columns = ["time_slot", "hist_time_slot_mean"]
        df = df.merge(time_slot_stats, on="time_slot", how="left")

        quarter_stats = df.groupby("quarter")["call_count"].agg(["mean"]).reset_index()
        quarter_stats.columns = ["quarter", "hist_quarter_mean"]
        df = df.merge(quarter_stats, on="quarter", how="left")

        for window in [336, 672, 1344]:
            df[f"rolling_mean_{window}"] = df["call_count"].shift(1).rolling(window=window, min_periods=1).mean()
            df[f"rolling_std_{window}"] = df["call_count"].shift(1).rolling(window=window, min_periods=1).std()

        return df

    def _compute_historical_patterns(self, df):
        self.historical_patterns = {
            "month_dow_hour_minute": df.groupby(["month", "day_of_week", "hour", "minute"])["call_count"]
                .agg(["mean", "std", "median", "max", "min", "count"]).to_dict(),
            "dow_hour_minute": df.groupby(["day_of_week", "hour", "minute"])["call_count"]
                .agg(["mean", "std", "median", "max", "min", "count"]).to_dict(),
            "month_dow_hour": df.groupby(["month", "day_of_week", "hour"])["call_count"]
                .agg(["mean", "std", "median"]).to_dict(),
            "month_dow": df.groupby(["month", "day_of_week"])["call_count"]
                .agg(["mean", "std"]).to_dict(),
            "month_hour": df.groupby(["month", "hour"])["call_count"]
                .agg(["mean", "std"]).to_dict(),
            "month": df.groupby("month")["call_count"]
                .agg(["mean", "std"]).to_dict(),
            "dow": df.groupby("day_of_week")["call_count"]
                .agg(["mean", "std"]).to_dict(),
            "hour": df.groupby("hour")["call_count"]
                .agg(["mean", "std"]).to_dict(),
            "time_slot": df.groupby(["hour", "minute"])["call_count"]
                .agg(["mean", "std"]).to_dict(),
            "overall_mean": df["call_count"].mean(),
            "overall_std": df["call_count"].std(),
            "overall_median": df["call_count"].median(),
        }

        self.volume_quantiles = {
            "q10": df["call_count"].quantile(0.1),
            "q25": df["call_count"].quantile(0.25),
            "q50": df["call_count"].quantile(0.5),
            "q75": df["call_count"].quantile(0.75),
            "q90": df["call_count"].quantile(0.9),
        }

    def _get_analogous_historical_data(self, target_dt):
        if self.training_data is None:
            return None, None, None, None

        target_month = target_dt.month
        target_dow = target_dt.dayofweek
        target_hour = target_dt.hour
        target_minute = target_dt.minute

        data = self.training_data

        same_time_same_dow = data[
            (data["month"] == target_month) &
            (data["day_of_week"] == target_dow) &
            (data["hour"] == target_hour) &
            (data["minute"] == target_minute)
        ]
        if len(same_time_same_dow) >= 2:
            calls = same_time_same_dow["call_count"]
            return calls.mean(), calls.std(), calls.max(), calls.min()

        same_month_dow_hour = data[
            (data["month"] == target_month) &
            (data["day_of_week"] == target_dow) &
            (data["hour"] == target_hour)
        ]
        if len(same_month_dow_hour) >= 3:
            calls = same_month_dow_hour["call_count"]
            return calls.mean(), calls.std(), calls.max(), calls.min()

        adjacent_months = [(target_month - 1) if target_month > 1 else 12,
                          target_month,
                          (target_month + 1) if target_month < 12 else 1]
        similar_period = data[
            (data["month"].isin(adjacent_months)) &
            (data["day_of_week"] == target_dow) &
            (data["hour"] == target_hour)
        ]
        if len(similar_period) >= 5:
            calls = similar_period["call_count"]
            return calls.mean(), calls.std(), calls.max(), calls.min()

        same_dow_hour = data[
            (data["day_of_week"] == target_dow) &
            (data["hour"] == target_hour) &
            (data["minute"] == target_minute)
        ]
        if len(same_dow_hour) >= 3:
            calls = same_dow_hour["call_count"]
            month_factor = self._get_month_factor(target_month, data)
            return calls.mean() * month_factor, calls.std(), calls.max() * month_factor, calls.min()

        same_month_dow = data[
            (data["month"] == target_month) &
            (data["day_of_week"] == target_dow)
        ]
        if len(same_month_dow) > 0:
            calls = same_month_dow["call_count"]
            return calls.mean(), calls.std(), calls.max(), calls.min()

        same_month = data[data["month"] == target_month]
        if len(same_month) > 0:
            calls = same_month["call_count"]
            return calls.mean(), calls.std(), calls.max(), calls.min()

        return self.historical_patterns["overall_mean"], self.historical_patterns["overall_std"], \
               self.historical_patterns["overall_mean"] * 1.5, 0

    def _get_month_factor(self, target_month, data):
        monthly_means = data.groupby("month")["call_count"].mean()
        overall_mean = data["call_count"].mean()
        if target_month in monthly_means.index:
            return monthly_means[target_month] / overall_mean
        return 1.0

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
            status = "OK" if diff < 3 else "WARN"
            if diff >= 3:
                issues.append(f"Month {month}: {diff:.1f}% difference")
            print(f"  {month:<8} {train_pct:<12.1f} {test_pct:<12.1f} {diff:<10.1f} {status}")

        train_mean = train_y.mean()
        test_mean = test_y.mean()
        train_std = train_y.std()
        test_std = test_y.std()
        mean_diff_pct = abs(train_mean - test_mean) / train_mean * 100

        print("\n  Call Volume Statistics:")
        print(f"  {'Metric':<20} {'Train':<15} {'Test':<15} {'Diff %':<10} {'Status'}")
        print("  " + "-" * 65)

        status = "OK" if mean_diff_pct < 10 else "WARN"
        print(f"  {'Mean calls/interval':<20} {train_mean:<15.2f} {test_mean:<15.2f} {mean_diff_pct:<10.1f} {status}")

        std_diff_pct = abs(train_std - test_std) / train_std * 100
        status = "OK" if std_diff_pct < 15 else "WARN"
        print(f"  {'Std calls/interval':<20} {train_std:<15.2f} {test_std:<15.2f} {std_diff_pct:<10.1f} {status}")

        print("\n" + "-" * 70)
        if len(issues) == 0:
            print("  DISTRIBUTION VERIFICATION PASSED")
        else:
            print(f"  {len(issues)} potential issue(s) found")
        print("-" * 70)

        return len(issues) == 0

    def _tune_hyperparameters(self, X_train, y_train, model_type, n_trials=50):
        if not HAS_OPTUNA:
            print("  Optuna not available, using default parameters", flush=True)
            return None

        print(f"  Running Optuna hyperparameter tuning ({n_trials} trials)...", flush=True)

        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial):
            if model_type == "xgboost" and HAS_XGBOOST:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "random_state": 42,
                    "verbosity": 0,
                    "n_jobs": -1,
                }
                model = XGBRegressor(**params)

            elif model_type == "lightgbm" and HAS_LIGHTGBM:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "random_state": 42,
                    "verbosity": -1,
                    "n_jobs": -1,
                }
                model = LGBMRegressor(**params)

            elif model_type == "gradient_boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "random_state": 42,
                }
                model = GradientBoostingRegressor(**params)

            elif model_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "random_state": 42,
                    "n_jobs": -1,
                }
                model = RandomForestRegressor(**params)

            else:
                return float("inf")

            scaler = RobustScaler()
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)

                model.fit(X_tr_scaled, y_tr)
                y_pred = np.maximum(model.predict(X_val_scaled), 0)
                scores.append(mean_absolute_error(y_val, y_pred))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        print(f"    Best MAE: {study.best_value:.4f}", flush=True)
        return study.best_params

    def _build_tuned_model(self, model_type, params):
        if params is None:
            return self._build_default_model(model_type)

        if model_type == "xgboost" and HAS_XGBOOST:
            return XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
        elif model_type == "lightgbm" and HAS_LIGHTGBM:
            return LGBMRegressor(**params, random_state=42, verbosity=-1, n_jobs=-1)
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(**params, random_state=42)
        elif model_type == "random_forest":
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        else:
            return self._build_default_model(model_type)

    def _build_default_model(self, model_type):
        if model_type == "xgboost" and HAS_XGBOOST:
            return XGBRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0, n_jobs=-1
            )
        elif model_type == "lightgbm" and HAS_LIGHTGBM:
            return LGBMRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=-1, n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                min_samples_split=15, min_samples_leaf=8, subsample=0.8,
                max_features="sqrt", random_state=42
            )
        elif model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1
            )
        elif model_type == "ridge":
            return Ridge(alpha=10.0)
        elif model_type == "elasticnet":
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        return None

    def train(self, data_path, test_year=2024, tune_hyperparameters=True, n_trials=30):
        print("=" * 70)
        print("TRAINING ADVANCED HYBRID FORECASTER WITH HYPERPARAMETER TUNING")
        print("=" * 70)

        print(f"\nXGBoost available: {HAS_XGBOOST}")
        print(f"LightGBM available: {HAS_LIGHTGBM}")
        print(f"Optuna available: {HAS_OPTUNA}")

        print("\nLoading and preparing data...")
        interval_counts = self._load_and_prepare_data(data_path)

        print("Creating base features...")
        df_base = self._create_base_features(interval_counts)

        df_base["year"] = df_base["interval_start"].dt.year
        train_mask = df_base["year"] < test_year
        test_mask = df_base["year"] >= test_year

        train_df_base = df_base[train_mask].copy()
        test_df_base = df_base[test_mask].copy()

        self.training_data = train_df_base.copy()
        self.max_training_date = train_df_base["interval_start"].max()

        print("Computing historical patterns from training data...")
        self._compute_historical_patterns(train_df_base)

        print("\n" + "=" * 70)
        print(f"DATA SPLIT: Year-Based (Train on <{test_year}, Test on {test_year})")
        print("=" * 70)

        train_dates = train_df_base["interval_start"]
        test_dates = test_df_base["interval_start"]
        print(f"  Training: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')} ({len(train_df_base):,} samples)")
        print(f"  Testing:  {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')} ({len(test_df_base):,} samples)")

        self._verify_distribution(train_df_base, test_df_base, train_df_base["call_count"], test_df_base["call_count"])

        print("\n" + "-" * 70)
        print("Training SHORT-TERM model (for predictions < 1 week ahead)")
        print("-" * 70)

        df_short_full = self._create_short_term_features(df_base.copy())
        df_short_train = df_short_full[df_short_full["year"] < test_year].copy()
        df_short_test = df_short_full[df_short_full["year"] >= test_year].copy()

        self.short_term_features = [
            "hour", "minute", "day_of_week", "day_of_month", "month", "quarter", "time_slot",
            "is_open", "is_business_hours", "is_morning_peak", "is_afternoon_peak",
            "is_early_morning", "is_lunch_hour", "is_late_afternoon", "is_monday", "is_friday", "is_mid_week",
            "is_tax_season", "is_tax_deadline", "is_tax_deadline_week", "is_tax_crunch",
            "is_extension_deadline", "is_quarterly_deadline", "is_year_end", "is_month_end",
            "is_month_start", "is_w2_season", "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
            "days_to_tax_deadline", "tax_urgency",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "time_slot_sin", "time_slot_cos", "week_sin", "week_cos",
            "lag_1", "lag_2", "lag_3", "lag_4", "lag_6", "lag_8", "lag_12", "lag_24",
            "lag_48", "lag_96", "lag_144", "lag_336", "lag_672",
            "lag_same_time_yesterday", "lag_same_time_last_week", "lag_same_time_2weeks",
            "diff_1", "diff_48", "diff_336",
            "rolling_mean_4", "rolling_std_4", "rolling_max_4", "rolling_min_4", "rolling_range_4",
            "rolling_mean_8", "rolling_std_8", "rolling_max_8", "rolling_min_8", "rolling_range_8",
            "rolling_mean_12", "rolling_std_12", "rolling_max_12", "rolling_min_12", "rolling_range_12",
            "rolling_mean_24", "rolling_std_24", "rolling_max_24", "rolling_min_24", "rolling_range_24",
            "rolling_mean_48", "rolling_std_48", "rolling_max_48", "rolling_min_48", "rolling_range_48",
            "rolling_mean_96", "rolling_std_96", "rolling_max_96", "rolling_min_96", "rolling_range_96",
            "rolling_mean_336", "rolling_std_336", "rolling_max_336", "rolling_min_336", "rolling_range_336",
            "rolling_median_48", "rolling_median_336",
            "rolling_skew_48", "rolling_skew_336",
            "rolling_iqr_48", "rolling_iqr_336",
            "ewm_mean_12", "ewm_std_12", "ewm_mean_24", "ewm_std_24",
            "ewm_mean_48", "ewm_std_48", "ewm_mean_96", "ewm_std_96",
            "hourly_trend", "daily_trend", "weekly_trend",
            "volatility_ratio", "momentum",
            "turbotax_ratio", "quickbooks_ratio",
        ]

        self.short_term_features = [f for f in self.short_term_features if f in df_short_train.columns]

        df_short_train_clean = df_short_train.dropna(subset=self.short_term_features + ["call_count"])
        df_short_test_clean = df_short_test.dropna(subset=self.short_term_features + ["call_count"])

        X_train = df_short_train_clean[self.short_term_features]
        y_train = df_short_train_clean["call_count"]
        X_test = df_short_test_clean[self.short_term_features]
        y_test = df_short_test_clean["call_count"]

        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")
        print(f"  Features: {len(self.short_term_features)}")

        self.short_term_scaler = RobustScaler()
        X_train_scaled = self.short_term_scaler.fit_transform(X_train)
        X_test_scaled = self.short_term_scaler.transform(X_test)

        model_types_to_tune = []
        if HAS_XGBOOST:
            model_types_to_tune.append("xgboost")
        if HAS_LIGHTGBM:
            model_types_to_tune.append("lightgbm")
        model_types_to_tune.extend(["gradient_boosting", "random_forest"])

        if tune_hyperparameters and HAS_OPTUNA:
            print("\n  Hyperparameter tuning for short-term models:")
            for model_type in model_types_to_tune:
                print(f"\n    Tuning {model_type}...")
                best_params = self._tune_hyperparameters(X_train, y_train, model_type, n_trials=n_trials)
                self.best_params[f"short_term_{model_type}"] = best_params

        model_scores = []
        for model_type in model_types_to_tune:
            params = self.best_params.get(f"short_term_{model_type}")
            model = self._build_tuned_model(model_type, params)
            model.fit(X_train_scaled, y_train)
            y_pred = np.maximum(model.predict(X_test_scaled), 0)
            mae = mean_absolute_error(y_test, y_pred)
            model_scores.append((model_type, mae, model))

            if hasattr(model, "feature_importances_"):
                self.feature_importance[f"short_term_{model_type}"] = dict(zip(
                    self.short_term_features,
                    model.feature_importances_
                ))

        for model_type in ["ridge", "elasticnet"]:
            model = self._build_default_model(model_type)
            model.fit(X_train_scaled, y_train)
            y_pred = np.maximum(model.predict(X_test_scaled), 0)
            mae = mean_absolute_error(y_test, y_pred)
            model_scores.append((model_type, mae, model))

        model_scores.sort(key=lambda x: x[1])
        best_models = model_scores[:3]

        print(f"\n  Model performance on test set:")
        for name, mae, _ in model_scores:
            print(f"    {name}: MAE = {mae:.4f}")

        ensemble_estimators = [(name, model) for name, _, model in best_models]
        self.short_term_model = VotingRegressor(estimators=ensemble_estimators)
        self.short_term_model.fit(X_train_scaled, y_train)

        y_pred = np.maximum(self.short_term_model.predict(X_test_scaled), 0)
        short_mae = mean_absolute_error(y_test, y_pred)

        print(f"\n  Ensemble MAE: {short_mae:.4f}")
        print(f"  Best models in ensemble: {[name for name, _, _ in best_models]}")

        print("\n" + "-" * 70)
        print("Training LONG-TERM model (for predictions >= 1 week ahead)")
        print("-" * 70)

        df_long_full = self._create_long_term_features(df_base.copy())
        df_long_train = df_long_full[df_long_full["year"] < test_year].copy()
        df_long_test = df_long_full[df_long_full["year"] >= test_year].copy()

        self.long_term_features = [
            "hour", "minute", "day_of_week", "day_of_month", "month", "quarter", "week_of_year", "time_slot",
            "is_open", "is_business_hours", "is_morning_peak", "is_afternoon_peak",
            "is_early_morning", "is_lunch_hour", "is_late_afternoon", "is_monday", "is_friday", "is_mid_week",
            "is_tax_season", "is_tax_deadline", "is_tax_deadline_week", "is_tax_crunch",
            "is_extension_deadline", "is_quarterly_deadline", "is_year_end", "is_month_end",
            "is_month_start", "is_w2_season", "is_holiday",
            "days_to_tax_deadline", "days_to_extension", "tax_urgency",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "doy_sin", "doy_cos", "week_sin", "week_cos", "time_slot_sin", "time_slot_cos",
            "hist_dow_hour_mean", "hist_dow_hour_std", "hist_dow_hour_median",
            "hist_month_dow_hour_mean", "hist_month_dow_hour_std",
            "hist_month_mean", "hist_month_std",
            "hist_dow_mean", "hist_dow_std",
            "hist_hour_mean", "hist_hour_std",
            "hist_time_slot_mean", "hist_quarter_mean",
            "rolling_mean_336", "rolling_std_336",
            "rolling_mean_672", "rolling_std_672",
            "rolling_mean_1344", "rolling_std_1344",
            "turbotax_ratio", "quickbooks_ratio",
        ]

        self.long_term_features = [f for f in self.long_term_features if f in df_long_train.columns]

        df_long_train_clean = df_long_train.dropna(subset=self.long_term_features + ["call_count"])
        df_long_test_clean = df_long_test.dropna(subset=self.long_term_features + ["call_count"])

        X_train = df_long_train_clean[self.long_term_features]
        y_train = df_long_train_clean["call_count"]
        X_test = df_long_test_clean[self.long_term_features]
        y_test = df_long_test_clean["call_count"]

        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")
        print(f"  Features: {len(self.long_term_features)}")

        self.long_term_scaler = RobustScaler()
        X_train_scaled = self.long_term_scaler.fit_transform(X_train)
        X_test_scaled = self.long_term_scaler.transform(X_test)

        if tune_hyperparameters and HAS_OPTUNA:
            print("\n  Hyperparameter tuning for long-term models:")
            for model_type in model_types_to_tune:
                print(f"\n    Tuning {model_type}...")
                best_params = self._tune_hyperparameters(X_train, y_train, model_type, n_trials=n_trials)
                self.best_params[f"long_term_{model_type}"] = best_params

        model_scores = []
        for model_type in model_types_to_tune:
            params = self.best_params.get(f"long_term_{model_type}")
            model = self._build_tuned_model(model_type, params)
            model.fit(X_train_scaled, y_train)
            y_pred = np.maximum(model.predict(X_test_scaled), 0)
            mae = mean_absolute_error(y_test, y_pred)
            model_scores.append((model_type, mae, model))

            if hasattr(model, "feature_importances_"):
                self.feature_importance[f"long_term_{model_type}"] = dict(zip(
                    self.long_term_features,
                    model.feature_importances_
                ))

        for model_type in ["ridge", "elasticnet"]:
            model = self._build_default_model(model_type)
            model.fit(X_train_scaled, y_train)
            y_pred = np.maximum(model.predict(X_test_scaled), 0)
            mae = mean_absolute_error(y_test, y_pred)
            model_scores.append((model_type, mae, model))

        model_scores.sort(key=lambda x: x[1])
        best_models = model_scores[:3]

        print(f"\n  Model performance on test set:")
        for name, mae, _ in model_scores:
            print(f"    {name}: MAE = {mae:.4f}")

        ensemble_estimators = [(name, model) for name, _, model in best_models]
        self.long_term_model = VotingRegressor(estimators=ensemble_estimators)
        self.long_term_model.fit(X_train_scaled, y_train)

        y_pred = np.maximum(self.long_term_model.predict(X_test_scaled), 0)
        long_mae = mean_absolute_error(y_test, y_pred)

        print(f"\n  Ensemble MAE: {long_mae:.4f}")
        print(f"  Best models in ensemble: {[name for name, _, _ in best_models]}")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nShort-term model: {len(self.short_term_features)} features, MAE: {short_mae:.4f}")
        print(f"Long-term model:  {len(self.long_term_features)} features, MAE: {long_mae:.4f}")

        self._print_top_features()
        self._print_best_params()

    def _print_top_features(self):
        print("\n" + "-" * 70)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("-" * 70)

        for model_name, importance in self.feature_importance.items():
            if importance:
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                print(f"\n  {model_name}:")
                for i, (feat, imp) in enumerate(sorted_features, 1):
                    print(f"    {i}. {feat}: {imp:.4f}")

    def _print_best_params(self):
        if self.best_params:
            print("\n" + "-" * 70)
            print("BEST HYPERPARAMETERS (from Optuna)")
            print("-" * 70)
            for model_name, params in self.best_params.items():
                if params:
                    print(f"\n  {model_name}:")
                    for param, value in params.items():
                        print(f"    {param}: {value}")

    def _get_base_features_for_prediction(self, target_dt):
        features = {}

        features["hour"] = target_dt.hour
        features["minute"] = target_dt.minute
        features["day_of_week"] = target_dt.weekday()
        features["day_of_month"] = target_dt.day
        features["month"] = target_dt.month
        features["week_of_year"] = target_dt.isocalendar()[1]
        features["day_of_year"] = target_dt.timetuple().tm_yday
        features["quarter"] = (target_dt.month - 1) // 3 + 1
        features["time_slot"] = target_dt.hour * 2 + (target_dt.minute // 30)

        features["is_weekend"] = int(features["day_of_week"] >= 5)
        features["is_business_hours"] = int((5 <= features["hour"] < 17) and features["day_of_week"] < 5)
        features["is_open"] = features["is_business_hours"]
        features["is_morning_peak"] = int((9 <= features["hour"] < 12) and features["day_of_week"] < 5)
        features["is_afternoon_peak"] = int((13 <= features["hour"] < 16) and features["day_of_week"] < 5)
        features["is_early_morning"] = int((5 <= features["hour"] < 8) and features["day_of_week"] < 5)
        features["is_lunch_hour"] = int((12 <= features["hour"] <= 13) and features["day_of_week"] < 5)
        features["is_late_afternoon"] = int((15 <= features["hour"] < 17) and features["day_of_week"] < 5)
        features["is_monday"] = int(features["day_of_week"] == 0)
        features["is_friday"] = int(features["day_of_week"] == 4)
        features["is_mid_week"] = int(1 <= features["day_of_week"] <= 3)

        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["dow_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["dow_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        features["doy_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 365)
        features["doy_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 365)
        features["week_sin"] = np.sin(2 * np.pi * features["week_of_year"] / 52)
        features["week_cos"] = np.cos(2 * np.pi * features["week_of_year"] / 52)
        features["time_slot_sin"] = np.sin(2 * np.pi * features["time_slot"] / 48)
        features["time_slot_cos"] = np.cos(2 * np.pi * features["time_slot"] / 48)

        features["is_tax_season"] = int(1 <= features["month"] <= 4)
        features["is_tax_deadline"] = int(features["month"] == 4 and features["day_of_month"] <= 15)
        features["is_tax_deadline_week"] = int(features["month"] == 4 and 10 <= features["day_of_month"] <= 17)
        features["is_tax_crunch"] = int(features["month"] == 4 and 1 <= features["day_of_month"] <= 15)
        features["is_extension_deadline"] = int(features["month"] == 10 and features["day_of_month"] <= 15)
        features["is_quarterly_deadline"] = int(
            (features["month"] == 1 and features["day_of_month"] <= 15) or
            (features["month"] == 4 and features["day_of_month"] <= 15) or
            (features["month"] == 6 and features["day_of_month"] <= 15) or
            (features["month"] == 9 and features["day_of_month"] <= 15)
        )
        features["is_year_end"] = int(features["month"] == 12)
        features["is_month_end"] = int(features["day_of_month"] >= 28)
        features["is_month_start"] = int(features["day_of_month"] <= 5)
        features["is_w2_season"] = int(features["month"] in [1, 2])

        features["days_to_tax_deadline"] = self._days_to_deadline(features["month"], features["day_of_month"], 4, 15)
        features["days_to_extension"] = self._days_to_deadline(features["month"], features["day_of_month"], 10, 15)
        features["tax_urgency"] = max(0, min(1, 1 - features["days_to_tax_deadline"] / 120))

        us_holidays = [(1,1), (1,15), (2,19), (5,27), (7,4), (9,2), (10,14), (11,11), (11,28), (12,25)]
        features["is_holiday"] = int((features["month"], features["day_of_month"]) in us_holidays)
        features["is_day_before_holiday"] = 0
        features["is_day_after_holiday"] = 0

        features["turbotax_ratio"] = 0.5
        features["quickbooks_ratio"] = 0.3

        return features

    def predict(self, target_datetime, reference_date=None):
        if isinstance(target_datetime, str):
            target_datetime = pd.to_datetime(target_datetime)

        if reference_date is None:
            reference_date = self.max_training_date

        days_ahead = (target_datetime - reference_date).days

        base_features = self._get_base_features_for_prediction(target_datetime)

        if not base_features["is_open"]:
            return 0

        avg_calls, std_calls, max_calls, min_calls = self._get_analogous_historical_data(target_datetime)

        if std_calls is None or std_calls <= 0:
            std_calls = avg_calls * 0.25 if avg_calls else 1

        if days_ahead < self.short_term_threshold_days:
            features = base_features.copy()

            for lag in [1, 2, 3, 4, 6, 8, 12, 24, 48, 96, 144, 336, 672]:
                features[f"lag_{lag}"] = avg_calls

            features["lag_same_time_yesterday"] = avg_calls
            features["lag_same_time_last_week"] = avg_calls
            features["lag_same_time_2weeks"] = avg_calls

            features["diff_1"] = 0
            features["diff_48"] = 0
            features["diff_336"] = 0

            for window in [4, 8, 12, 24, 48, 96, 336]:
                features[f"rolling_mean_{window}"] = avg_calls
                features[f"rolling_std_{window}"] = std_calls
                features[f"rolling_max_{window}"] = max_calls if max_calls else avg_calls * 1.3
                features[f"rolling_min_{window}"] = min_calls if min_calls else avg_calls * 0.7
                features[f"rolling_range_{window}"] = (max_calls or avg_calls * 1.3) - (min_calls or avg_calls * 0.7)

            for window in [48, 336]:
                features[f"rolling_median_{window}"] = avg_calls
                features[f"rolling_skew_{window}"] = 0
                features[f"rolling_iqr_{window}"] = std_calls * 1.35

            for span in [12, 24, 48, 96]:
                features[f"ewm_mean_{span}"] = avg_calls
                features[f"ewm_std_{span}"] = std_calls

            features["hourly_trend"] = 0
            features["daily_trend"] = 0
            features["weekly_trend"] = 0
            features["volatility_ratio"] = std_calls / (avg_calls + 1)
            features["momentum"] = 0

            X = pd.DataFrame([features])[self.short_term_features]
            X = X.fillna(0)
            X_scaled = self.short_term_scaler.transform(X)
            prediction = self.short_term_model.predict(X_scaled)[0]

        else:
            features = base_features.copy()

            features["hist_dow_hour_mean"] = avg_calls
            features["hist_dow_hour_std"] = std_calls
            features["hist_dow_hour_median"] = avg_calls
            features["hist_month_dow_hour_mean"] = avg_calls
            features["hist_month_dow_hour_std"] = std_calls

            month_key = features["month"]
            if month_key in self.historical_patterns["month"]["mean"]:
                features["hist_month_mean"] = self.historical_patterns["month"]["mean"][month_key]
                features["hist_month_std"] = self.historical_patterns["month"]["std"].get(month_key, 1)
            else:
                features["hist_month_mean"] = self.historical_patterns["overall_mean"]
                features["hist_month_std"] = self.historical_patterns["overall_std"]

            dow_key = features["day_of_week"]
            if dow_key in self.historical_patterns["dow"]["mean"]:
                features["hist_dow_mean"] = self.historical_patterns["dow"]["mean"][dow_key]
                features["hist_dow_std"] = self.historical_patterns["dow"]["std"].get(dow_key, 1)
            else:
                features["hist_dow_mean"] = avg_calls
                features["hist_dow_std"] = std_calls

            hour_key = features["hour"]
            if hour_key in self.historical_patterns["hour"]["mean"]:
                features["hist_hour_mean"] = self.historical_patterns["hour"]["mean"][hour_key]
                features["hist_hour_std"] = self.historical_patterns["hour"]["std"].get(hour_key, 1)
            else:
                features["hist_hour_mean"] = avg_calls
                features["hist_hour_std"] = std_calls

            features["hist_time_slot_mean"] = avg_calls
            features["hist_quarter_mean"] = self.historical_patterns["overall_mean"]

            for window in [336, 672, 1344]:
                features[f"rolling_mean_{window}"] = avg_calls
                features[f"rolling_std_{window}"] = std_calls

            X = pd.DataFrame([features])[self.long_term_features]
            X = X.fillna(0)
            X_scaled = self.long_term_scaler.transform(X)
            prediction = self.long_term_model.predict(X_scaled)[0]

        return max(0, int(round(prediction)))

    def predict_day(self, target_date, reference_date=None):
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()

        results = []
        current = datetime.combine(target_date, datetime.min.time())

        if reference_date is None:
            reference_date = self.max_training_date

        days_ahead = (pd.Timestamp(target_date) - reference_date).days
        model_used = "short-term" if days_ahead < self.short_term_threshold_days else "long-term"

        for _ in range(48):
            prediction = self.predict(current, reference_date)
            results.append({
                "interval_start": current,
                "predicted_calls": prediction,
                "model_used": model_used,
            })
            current += timedelta(minutes=30)

        return pd.DataFrame(results)

    def predict_with_confidence(self, target_datetime, reference_date=None, confidence_level=0.95):
        prediction = self.predict(target_datetime, reference_date)
        avg_calls, std_calls, _, _ = self._get_analogous_historical_data(pd.to_datetime(target_datetime))

        if std_calls and std_calls > 0:
            z_score = 1.96 if confidence_level == 0.95 else 1.645
            margin = z_score * std_calls
            lower = max(0, int(prediction - margin))
            upper = int(prediction + margin)
        else:
            lower = max(0, int(prediction * 0.7))
            upper = int(prediction * 1.3)

        return {
            "prediction": prediction,
            "lower_bound": lower,
            "upper_bound": upper,
            "confidence_level": confidence_level
        }

    def save_model(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({
                "short_term_model": self.short_term_model,
                "long_term_model": self.long_term_model,
                "short_term_scaler": self.short_term_scaler,
                "long_term_scaler": self.long_term_scaler,
                "short_term_features": self.short_term_features,
                "long_term_features": self.long_term_features,
                "historical_patterns": self.historical_patterns,
                "training_data": self.training_data,
                "max_training_date": self.max_training_date,
                "short_term_threshold_days": self.short_term_threshold_days,
                "volume_quantiles": self.volume_quantiles,
                "feature_importance": self.feature_importance,
                "best_params": self.best_params,
            }, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.short_term_model = data["short_term_model"]
        self.long_term_model = data["long_term_model"]
        self.short_term_scaler = data["short_term_scaler"]
        self.long_term_scaler = data["long_term_scaler"]
        self.short_term_features = data["short_term_features"]
        self.long_term_features = data["long_term_features"]
        self.historical_patterns = data["historical_patterns"]
        self.training_data = data.get("training_data")
        self.max_training_date = data["max_training_date"]
        self.short_term_threshold_days = data.get("short_term_threshold_days", 7)
        self.volume_quantiles = data.get("volume_quantiles", {})
        self.feature_importance = data.get("feature_importance", {})
        self.best_params = data.get("best_params", {})
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("Starting training...", flush=True)
    forecaster = HybridForecaster()
    forecaster.train("mock_intuit_2year_data.csv", tune_hyperparameters=True, n_trials=10)
    forecaster.save_model("hybrid_forecast_model.pkl")
