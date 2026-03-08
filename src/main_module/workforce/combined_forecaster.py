import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler


_US_HOLIDAYS = {
    (1, 1), (1, 15), (2, 19), (5, 27), (7, 4),
    (9, 2), (10, 14), (11, 11), (11, 28), (12, 25),
}

_MAJOR_HOLIDAYS = {(1, 1), (11, 28), (12, 25)}

_DAYS_IN_MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_KNOWN_ANOMALIES = [
    ("2025-08-29 00:00:00", "2025-08-29 23:59:59"),
]


def _days_to_deadline(month, day, dl_month, dl_day):
    if month < dl_month or (month == dl_month and day <= dl_day):
        if month == dl_month:
            return dl_day - day
        days = _DAYS_IN_MONTHS[month - 1] - day
        for m in range(month + 1, dl_month):
            days += _DAYS_IN_MONTHS[m - 1]
        return days + dl_day
    return 365


class CombinedForecaster:

    SHORT_TERM_FEATURES = [
        "hour", "minute", "day_of_week", "day_of_month", "month", "time_slot",
        "week_of_year", "day_of_year",
        "is_holiday", "is_major_holiday", "is_january",
        "days_to_tax_deadline", "tax_urgency", "is_post_tax_drop",
        "hour_sin", "hour_cos", "dow_sin", "month_sin", "month_cos",
        "lag_1", "lag_2", "lag_4", "lag_48", "lag_336", "lag_672",
        "lag_same_time_yesterday", "lag_same_time_last_week",
        "diff_1", "diff_48", "diff_336",
        "rolling_mean_4", "rolling_mean_12", "rolling_mean_48", "rolling_mean_336",
        "rolling_std_4", "rolling_std_48", "rolling_max_4",
        "ewm_mean_12", "ewm_mean_48",
        "hourly_trend", "daily_trend",
        "volatility_ratio", "momentum",
        "inbound_ratio", "chat_ratio", "callback_ratio",
        "lag_transfer_rate", "lag_fcr_rate", "lag_mean_hold",
        "lag_active_experts", "lag_mean_occupancy", "lag_total_avail",
        "rolling_experts_48", "rolling_occupancy_48",
        "yoy_same_dow_hour_mean",
    ]

    LONG_TERM_FEATURES = [
        "hour", "day_of_week", "day_of_month", "month", "time_slot",
        "week_of_year", "day_of_year",
        "is_holiday", "is_major_holiday", "is_january",
        "days_to_tax_deadline", "tax_urgency", "is_post_tax_drop",
        "hour_cos", "dow_sin", "month_sin", "month_cos",
        "hist_dow_hour_mean", "hist_dow_hour_std", "hist_dow_hour_median",
        "hist_month_dow_hour_mean", "hist_month_dow_hour_std",
        "hist_month_mean",
        "hist_time_slot_mean",
        "hist_week_of_year_mean", "hist_quarter_dow_mean",
        "rolling_mean_336", "rolling_mean_672",
        "ewm_mean_336", "ewm_mean_672",
        "inbound_ratio", "chat_ratio", "callback_ratio",
        "hist_transfer_rate", "hist_fcr_rate", "hist_mean_hold",
        "hist_mean_experts", "hist_mean_occupancy",
        "yoy_same_dow_hour_mean", "yoy_same_week_mean",
        "recent_quarter_mean", "recent_month_mean",
        "hist_recent_dow_hour_mean",
        "hist_dow_time_slot_mean", "hist_month_time_slot_mean",
    ]

    _LT_ENSEMBLE_CONFIGS = [
        {"n_estimators": 1500, "max_depth": 9, "learning_rate": 0.015,
         "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 1.0,
         "reg_lambda": 2.0, "min_child_samples": 15, "num_leaves": 200,
         "random_state": 42},
        {"n_estimators": 1500, "max_depth": 9, "learning_rate": 0.015,
         "subsample": 0.7, "colsample_bytree": 0.5, "reg_alpha": 1.0,
         "reg_lambda": 2.0, "min_child_samples": 15, "num_leaves": 200,
         "random_state": 7},
        {"n_estimators": 1500, "max_depth": 8, "learning_rate": 0.02,
         "subsample": 0.85, "colsample_bytree": 0.7, "reg_alpha": 1.0,
         "reg_lambda": 2.0, "min_child_samples": 20, "num_leaves": 127,
         "random_state": 123},
    ]

    def __init__(self):
        self.short_term_model = None
        self.long_term_model = None
        self.short_term_scaler = None
        self.long_term_scaler = None
        self.short_term_features = list(self.SHORT_TERM_FEATURES)
        self.long_term_features = list(self.LONG_TERM_FEATURES)
        self.historical_patterns = {}
        self._hist_index = {}
        self._holiday_profiles = {}
        self.max_training_date = None
        self.short_term_threshold_days = 7
        self.feature_importance = {}

    def _load_and_prepare_data(self, data_path):
        data_path = str(data_path)
        base_dir = Path(data_path).parent

        raw_data = pd.read_parquet(
            data_path,
            columns=["arrival_time_utc", "cc_id", "communication_channel_type"],
        )
        raw_data["arrival_time_utc"] = pd.to_datetime(raw_data["arrival_time_utc"])
        raw_data["interval_start"] = raw_data["arrival_time_utc"].dt.floor("30min")

        channel = raw_data["communication_channel_type"].fillna("")
        raw_data["_is_inbound"] = channel.str.startswith("INBOUND").astype(np.int8)
        raw_data["_is_chat"] = channel.str.contains(
            "CHAT|LiveChat|MessagingChat", case=False, regex=True,
        ).astype(np.int8)
        raw_data["_is_callback"] = channel.str.contains(
            "CALLBACK", case=False, regex=True,
        ).astype(np.int8)

        interval_counts = (
            raw_data.groupby("interval_start")
            .agg(
                call_count=("cc_id", "count"),
                inbound_count=("_is_inbound", "sum"),
                chat_count=("_is_chat", "sum"),
                callback_count=("_is_callback", "sum"),
            )
            .reset_index()
        )
        del raw_data

        full_range = pd.date_range(
            start=interval_counts["interval_start"].min(),
            end=interval_counts["interval_start"].max(),
            freq="30min",
        )
        full_df = pd.DataFrame({"interval_start": full_range})
        interval_counts = full_df.merge(interval_counts, on="interval_start", how="left")
        for col in ["call_count", "inbound_count", "chat_count", "callback_count"]:
            interval_counts[col] = interval_counts[col].fillna(0).astype(np.int32)

        interval_counts = self._smooth_anomalies(interval_counts)
        interval_counts = self._merge_dataset3(interval_counts, base_dir)
        interval_counts = self._merge_dataset4(interval_counts, base_dir)

        return interval_counts.sort_values("interval_start").reset_index(drop=True)

    def _smooth_anomalies(self, df):
        for start, end in _KNOWN_ANOMALIES:
            mask = (df["interval_start"] >= start) & (df["interval_start"] <= end)
            if mask.any():
                df.loc[mask, "call_count"] = np.nan
                df["call_count"] = df["call_count"].interpolate(method="linear").fillna(0).astype(np.int32)
                print(f"  Smoothed anomaly: {start} to {end}")
        return df

    def _merge_dataset3(self, df, base_dir):
        d3_path = base_dir / "dataset_3_historical_outcomes.parquet"
        if not d3_path.exists():
            for col in ["transfer_rate", "fcr_rate", "mean_hold"]:
                df[col] = 0.0
            return df

        print("  Merging dataset_3 (historical outcomes)...")
        d3 = pd.read_parquet(
            d3_path,
            columns=["session_start_time_utc", "transfer_count", "first_call_resolution", "hold_time_seconds"],
        )
        d3["interval_start"] = pd.to_datetime(d3["session_start_time_utc"]).dt.floor("30min")
        d3["_transferred"] = (d3["transfer_count"] > 0).astype(np.int8)
        d3["_fcr"] = (d3["first_call_resolution"] == "Y").astype(np.int8)
        d3["hold_time_seconds"] = d3["hold_time_seconds"].fillna(0)

        d3_agg = (
            d3.groupby("interval_start")
            .agg(
                transfer_rate=("_transferred", "mean"),
                fcr_rate=("_fcr", "mean"),
                mean_hold=("hold_time_seconds", "mean"),
            )
            .reset_index()
        )
        del d3

        df = df.merge(d3_agg, on="interval_start", how="left")
        for col in ["transfer_rate", "fcr_rate", "mean_hold"]:
            df[col] = df[col].fillna(0).astype(np.float32)
        return df

    def _merge_dataset4(self, df, base_dir):
        d4_path = base_dir / "dataset_4_expert_state_interval.parquet"
        if not d4_path.exists():
            for col in ["active_experts", "mean_occupancy", "total_avail"]:
                df[col] = 0.0
            return df

        print("  Merging dataset_4 (expert state)...")
        d4 = pd.read_parquet(
            d4_path,
            columns=["interval_start_utc", "expert_id", "occupancy_pct", "total_available_time_seconds"],
        )
        d4["interval_start"] = pd.to_datetime(d4["interval_start_utc"]).dt.floor("30min")

        d4_agg = (
            d4.groupby("interval_start")
            .agg(
                active_experts=("expert_id", "nunique"),
                mean_occupancy=("occupancy_pct", "mean"),
                total_avail=("total_available_time_seconds", "sum"),
            )
            .reset_index()
        )
        del d4

        df = df.merge(d4_agg, on="interval_start", how="left")
        df["active_experts"] = df["active_experts"].fillna(0).astype(np.int32)
        df["mean_occupancy"] = df["mean_occupancy"].fillna(0).astype(np.float32)
        df["total_avail"] = df["total_avail"].fillna(0).astype(np.float32)
        return df

    def _create_base_features(self, df):
        df = df.copy()
        ts = df["interval_start"]

        df["hour"] = ts.dt.hour
        df["minute"] = ts.dt.minute
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_month"] = ts.dt.day
        df["month"] = ts.dt.month
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["day_of_year"] = ts.dt.dayofyear
        df["quarter"] = ts.dt.quarter
        df["year"] = ts.dt.year
        df["time_slot"] = df["hour"] * 2 + (df["minute"] // 30)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        dtd = np.vectorize(_days_to_deadline)(df["month"].values, df["day_of_month"].values, 4, 15)
        df["days_to_tax_deadline"] = dtd
        df["tax_urgency"] = np.clip(1 - dtd / 120, 0, 1)

        df["is_holiday"] = [
            int((m, d) in _US_HOLIDAYS)
            for m, d in zip(df["month"].values, df["day_of_month"].values)
        ]
        df["is_major_holiday"] = [
            int((m, d) in _MAJOR_HOLIDAYS)
            for m, d in zip(df["month"].values, df["day_of_month"].values)
        ]
        df["is_january"] = (df["month"] == 1).astype(np.int8)
        df["is_post_tax_drop"] = (
            (dtd < 0) & (dtd > -365 + 335)
        ).astype(np.int8)
        mask_post = (df["month"] == 4) & (df["day_of_month"] > 15) | (
            (df["month"] == 5) & (df["day_of_month"] <= 15)
        )
        df["is_post_tax_drop"] = mask_post.astype(np.int8)

        cc = df["call_count"].values.astype(np.float64) + 1
        df["inbound_ratio"] = df["inbound_count"].values / cc
        df["chat_ratio"] = df["chat_count"].values / cc
        df["callback_ratio"] = df["callback_count"].values / cc

        return df

    def _add_lag_features(self, df):
        df = df.copy()
        cc = df["call_count"]

        for lag in [1, 2, 4, 48, 336, 672]:
            df[f"lag_{lag}"] = cc.shift(lag)

        df["lag_same_time_yesterday"] = cc.shift(48)
        df["lag_same_time_last_week"] = cc.shift(336)

        df["diff_1"] = cc.diff(1)
        df["diff_48"] = cc.diff(48)
        df["diff_336"] = cc.diff(336)

        shifted = cc.shift(1)
        for w in [4, 12, 48, 336]:
            df[f"rolling_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        for w in [4, 48]:
            df[f"rolling_std_{w}"] = shifted.rolling(w, min_periods=1).std()
        df["rolling_max_4"] = shifted.rolling(4, min_periods=1).max()

        df["ewm_mean_12"] = shifted.ewm(span=12, adjust=False).mean()
        df["ewm_mean_48"] = shifted.ewm(span=48, adjust=False).mean()

        df["hourly_trend"] = df["rolling_mean_4"] - df["rolling_mean_48"]
        df["daily_trend"] = df["rolling_mean_48"] - df["rolling_mean_336"]
        df["volatility_ratio"] = df["rolling_std_48"] / (df["rolling_mean_48"] + 1)
        df["momentum"] = df["ewm_mean_12"] - df["ewm_mean_48"]

        for col in ["transfer_rate", "fcr_rate", "mean_hold",
                     "active_experts", "mean_occupancy", "total_avail"]:
            df[f"lag_{col}"] = df[col].shift(1) if col in df.columns else 0.0

        if "active_experts" in df.columns:
            df["rolling_experts_48"] = df["active_experts"].shift(1).rolling(48, min_periods=1).mean()
        else:
            df["rolling_experts_48"] = 0.0

        if "mean_occupancy" in df.columns:
            df["rolling_occupancy_48"] = df["mean_occupancy"].shift(1).rolling(48, min_periods=1).mean()
        else:
            df["rolling_occupancy_48"] = 0.0

        return df

    def _add_yoy_features(self, df, train_only_df):
        fallback = train_only_df["call_count"].mean()

        yoy_dow_hour = (
            train_only_df.groupby(["month", "day_of_week", "hour", "minute"])["call_count"]
            .mean().rename("yoy_same_dow_hour_mean")
        )
        df = df.merge(yoy_dow_hour, left_on=["month", "day_of_week", "hour", "minute"], right_index=True, how="left")
        df["yoy_same_dow_hour_mean"] = df["yoy_same_dow_hour_mean"].fillna(fallback)

        yoy_week = train_only_df.groupby("week_of_year")["call_count"].mean().rename("yoy_same_week_mean")
        df = df.merge(yoy_week, left_on="week_of_year", right_index=True, how="left")
        df["yoy_same_week_mean"] = df["yoy_same_week_mean"].fillna(fallback)

        return df

    def _add_long_term_hist_features(self, df, train_only_df):
        df = df.copy()
        fallback = train_only_df["call_count"].mean()

        for keys, prefix, aggs in [
            (["day_of_week", "hour", "minute"], "hist_dow_hour", ["mean", "std", "median"]),
            (["month", "day_of_week", "hour", "minute"], "hist_month_dow_hour", ["mean", "std"]),
            (["month"], "hist_month", ["mean"]),
            (["time_slot"], "hist_time_slot", ["mean"]),
            (["week_of_year"], "hist_week_of_year", ["mean"]),
            (["quarter", "day_of_week"], "hist_quarter_dow", ["mean"]),
        ]:
            stats = train_only_df.groupby(keys)["call_count"].agg(aggs).reset_index()
            stats = stats.rename(columns={a: f"{prefix}_{a}" for a in aggs})
            df = df.merge(stats, on=keys, how="left")

        for metric, col_name in [
            ("transfer_rate", "hist_transfer_rate"),
            ("fcr_rate", "hist_fcr_rate"),
            ("mean_hold", "hist_mean_hold"),
            ("active_experts", "hist_mean_experts"),
            ("mean_occupancy", "hist_mean_occupancy"),
        ]:
            if metric in train_only_df.columns:
                stats = train_only_df.groupby(["day_of_week", "hour"])[metric].mean().rename(col_name).reset_index()
                df = df.merge(stats, on=["day_of_week", "hour"], how="left")
            if col_name not in df.columns:
                df[col_name] = 0.0

        shifted = df["call_count"].shift(1)
        for w in [336, 672]:
            df[f"rolling_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df["ewm_mean_336"] = shifted.ewm(span=336, adjust=False).mean()
        df["ewm_mean_672"] = shifted.ewm(span=672, adjust=False).mean()

        source_ts = train_only_df.set_index("interval_start")["call_count"].sort_index()
        last_quarter = source_ts.iloc[-336 * 13:] if len(source_ts) > 336 * 13 else source_ts
        last_month = source_ts.iloc[-336 * 4:] if len(source_ts) > 336 * 4 else source_ts

        for subset, col_name in [(last_quarter, "recent_quarter_mean"), (last_month, "recent_month_mean")]:
            stats = subset.groupby([subset.index.dayofweek, subset.index.hour, subset.index.minute]).mean()
            stats.index.names = ["day_of_week", "hour", "minute"]
            stats = stats.rename(col_name).reset_index()
            df = df.merge(stats, on=["day_of_week", "hour", "minute"], how="left")
            df[col_name] = df[col_name].fillna(fallback)

        recent_half = train_only_df.iloc[len(train_only_df) // 2:]
        recent_dow_hour = (
            recent_half.groupby(["day_of_week", "hour", "minute"])["call_count"]
            .mean().rename("hist_recent_dow_hour_mean").reset_index()
        )
        df = df.merge(recent_dow_hour, on=["day_of_week", "hour", "minute"], how="left")
        df["hist_recent_dow_hour_mean"] = df["hist_recent_dow_hour_mean"].fillna(fallback)

        for keys, col_name in [
            (["day_of_week", "time_slot"], "hist_dow_time_slot_mean"),
            (["month", "time_slot"], "hist_month_time_slot_mean"),
        ]:
            stats = train_only_df.groupby(keys)["call_count"].mean().rename(col_name).reset_index()
            df = df.merge(stats, on=keys, how="left")
            df[col_name] = df[col_name].fillna(fallback)

        return df

    def _compute_historical_patterns(self, df):
        self.historical_patterns = {
            "overall_mean": float(df["call_count"].mean()),
            "overall_std": float(df["call_count"].std()),
        }
        for grp_name, keys in [("month", ["month"]), ("dow", ["day_of_week"]), ("hour", ["hour"])]:
            stats = df.groupby(keys)["call_count"].agg(["mean", "std"])
            self.historical_patterns[grp_name] = {
                "mean": stats["mean"].to_dict(),
                "std": stats["std"].to_dict(),
            }
        self._hist_index = df.groupby(
            ["month", "day_of_week", "hour", "minute"]
        )["call_count"].agg(["mean", "std", "max", "min"])

        holiday_df = df[df["is_major_holiday"] == 1]
        if len(holiday_df) > 0:
            ts = holiday_df["interval_start"]
            self._holiday_profiles = (
                holiday_df.groupby([ts.dt.month, ts.dt.day, ts.dt.hour, ts.dt.minute])["call_count"]
                .mean().to_dict()
            )

    def _lookup_historical(self, month, dow, hour, minute):
        try:
            row = self._hist_index.loc[(month, dow, hour, minute)]
            return float(row["mean"]), float(row["std"]), float(row["max"])
        except KeyError:
            om = self.historical_patterns["overall_mean"]
            os_ = self.historical_patterns["overall_std"]
            return om, os_, om * 1.5

    def train(self, data_path, train_year=2024, test_year=2025):
        print("=" * 70)
        print("TRAINING COMBINED FORECASTER")
        print("=" * 70)

        print("\n  Loading and aggregating data...")
        interval_df = self._load_and_prepare_data(data_path)
        print(f"  Intervals: {len(interval_df):,}  "
              f"({interval_df['interval_start'].min().date()} to {interval_df['interval_start'].max().date()})")

        print("  Creating base features...")
        df = self._create_base_features(interval_df)

        test_months_available = sorted(df.loc[df["year"] == test_year, "month"].unique())
        last_complete_test_month = test_months_available[-1]
        last_day = df.loc[
            (df["year"] == test_year) & (df["month"] == last_complete_test_month), "day_of_month"
        ].max()
        if last_day < 28:
            test_months_available = test_months_available[:-1]
        shared_months = set(test_months_available)

        train_mask = (df["year"] == train_year) & (df["month"].isin(shared_months))
        test_mask = (df["year"] == test_year) & (df["month"].isin(shared_months))

        self.max_training_date = df[train_mask]["interval_start"].max()

        print("  Computing historical patterns & holiday profiles...")
        self._compute_historical_patterns(df[train_mask])

        print(f"\n  Train: {train_year} ({train_mask.sum():,} intervals)  Test: {test_year} ({test_mask.sum():,} intervals)")
        print(f"  Holiday profiles stored: {len(self._holiday_profiles)}")

        print("\n" + "-" * 70)
        print("  SHORT-TERM MODEL (< 7 days ahead)")
        print("-" * 70)

        df_short = self._add_lag_features(df)
        df_short = self._add_yoy_features(df_short, train_only_df=df_short[train_mask])

        self.short_term_features = [f for f in self.SHORT_TERM_FEATURES if f in df_short.columns]
        st_train = df_short[train_mask].dropna(subset=self.short_term_features + ["call_count"])
        st_test = df_short[test_mask].dropna(subset=self.short_term_features + ["call_count"])

        X_tr, y_tr = st_train[self.short_term_features], st_train["call_count"]
        X_te, y_te = st_test[self.short_term_features], st_test["call_count"]

        self.short_term_scaler = RobustScaler()
        X_tr_s = self.short_term_scaler.fit_transform(X_tr)
        X_te_s = self.short_term_scaler.transform(X_te)

        st_idx = np.arange(len(X_tr), dtype=np.float64)
        st_weights = 0.2 + 0.8 * (st_idx / st_idx.max())

        self.short_term_model = LGBMRegressor(
            n_estimators=800, max_depth=9, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05,
            reg_lambda=0.5, min_child_samples=15, num_leaves=127,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
        self.short_term_model.fit(
            X_tr_s, y_tr, sample_weight=st_weights,
            eval_set=[(X_te_s, y_te)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )

        y_pred = np.maximum(self.short_term_model.predict(X_te_s), 0)
        st_mae = mean_absolute_error(y_te, y_pred)
        st_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        st_r2 = r2_score(y_te, y_pred)
        st_wmape = np.sum(np.abs(y_te.values - y_pred)) / np.sum(y_te.values) * 100
        print(f"  MAE: {st_mae:.2f}  RMSE: {st_rmse:.2f}  R²: {st_r2:.4f}  WMAPE: {st_wmape:.2f}%")

        if hasattr(self.short_term_model, "feature_importances_"):
            imp = sorted(zip(self.short_term_features, self.short_term_model.feature_importances_), key=lambda x: x[1], reverse=True)
            self.feature_importance["short_term"] = dict(imp)
            print("  Top features:", ", ".join(n for n, _ in imp[:10]))

        print("\n" + "-" * 70)
        print("  LONG-TERM MODEL (>= 7 days ahead)")
        print("-" * 70)

        df_long = self._add_long_term_hist_features(df, train_only_df=df[train_mask])
        df_long = self._add_yoy_features(df_long, train_only_df=df_long[train_mask])

        self.long_term_features = [f for f in self.LONG_TERM_FEATURES if f in df_long.columns]
        lt_train = df_long[train_mask].dropna(subset=self.long_term_features + ["call_count"])
        lt_test = df_long[test_mask].dropna(subset=self.long_term_features + ["call_count"])

        X_tr, y_tr = lt_train[self.long_term_features], lt_train["call_count"]
        X_te, y_te = lt_test[self.long_term_features], lt_test["call_count"]

        self.long_term_scaler = RobustScaler()
        X_tr_s = self.long_term_scaler.fit_transform(X_tr)
        X_te_s = self.long_term_scaler.transform(X_te)

        lt_idx = np.arange(len(X_tr), dtype=np.float64)
        lt_weights = 0.1 + 0.9 * (lt_idx / lt_idx.max()) ** 2

        lt_models = [LGBMRegressor(**cfg, verbosity=-1, n_jobs=-1) for cfg in self._LT_ENSEMBLE_CONFIGS]
        print(f"  Training {len(lt_models)}-model ensemble...")

        for model in lt_models:
            model.fit(
                X_tr_s, y_tr, sample_weight=lt_weights,
                eval_set=[(X_te_s, y_te)],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
        self.long_term_model = lt_models

        y_pred = np.column_stack([np.maximum(m.predict(X_te_s), 0) for m in lt_models]).mean(axis=1)
        lt_mae = mean_absolute_error(y_te, y_pred)
        lt_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        lt_r2 = r2_score(y_te, y_pred)
        lt_wmape = np.sum(np.abs(y_te.values - y_pred)) / np.sum(y_te.values) * 100
        print(f"  MAE: {lt_mae:.2f}  RMSE: {lt_rmse:.2f}  R²: {lt_r2:.4f}  WMAPE: {lt_wmape:.2f}%")

        total_imp = sum(m.feature_importances_ for m in lt_models)
        imp = sorted(zip(self.long_term_features, total_imp), key=lambda x: x[1], reverse=True)
        self.feature_importance["long_term"] = dict(imp)
        print("  Top features:", ", ".join(n for n, _ in imp[:10]))

        print(f"\n{'='*70}")
        print(f"  Short-term:  MAE={st_mae:.2f}  R²={st_r2:.4f}  WMAPE={st_wmape:.2f}%")
        print(f"  Long-term:   MAE={lt_mae:.2f}  R²={lt_r2:.4f}  WMAPE={lt_wmape:.2f}%")
        print("=" * 70)

    def _base_features_for_dt(self, dt):
        h, m = dt.hour, dt.minute
        dow = dt.weekday()
        dom = dt.day
        mon = dt.month
        dtd = _days_to_deadline(mon, dom, 4, 15)
        is_post = (mon == 4 and dom > 15) or (mon == 5 and dom <= 15)

        return {
            "hour": h, "minute": m, "day_of_week": dow, "day_of_month": dom,
            "month": mon, "time_slot": h * 2 + (m // 30),
            "week_of_year": dt.isocalendar()[1],
            "day_of_year": dt.timetuple().tm_yday,
            "is_holiday": int((mon, dom) in _US_HOLIDAYS),
            "is_major_holiday": int((mon, dom) in _MAJOR_HOLIDAYS),
            "is_january": int(mon == 1),
            "days_to_tax_deadline": dtd,
            "tax_urgency": max(0.0, min(1.0, 1 - dtd / 120)),
            "is_post_tax_drop": int(is_post),
            "hour_sin": np.sin(2 * np.pi * h / 24),
            "hour_cos": np.cos(2 * np.pi * h / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * mon / 12),
            "month_cos": np.cos(2 * np.pi * mon / 12),
            "inbound_ratio": 0.3, "chat_ratio": 0.2, "callback_ratio": 0.25,
        }

    def predict(self, target_datetime, reference_date=None):
        target_datetime = pd.Timestamp(target_datetime)
        if reference_date is None:
            reference_date = self.max_training_date
        days_ahead = (target_datetime - reference_date).days

        h = target_datetime.hour
        dow = target_datetime.weekday()
        mon = target_datetime.month
        dom = target_datetime.day

        is_open = (h >= 13 or h < 1) and dow < 5
        if not is_open:
            return 0

        if (mon, dom) in _MAJOR_HOLIDAYS:
            profile_key = (mon, dom, h, target_datetime.minute)
            if profile_key in self._holiday_profiles:
                return max(0, int(round(self._holiday_profiles[profile_key])))

        feat = self._base_features_for_dt(target_datetime)

        avg, std, mx = self._lookup_historical(mon, dow, h, target_datetime.minute)

        if days_ahead < self.short_term_threshold_days:
            for lag in [1, 2, 4, 48, 336, 672]:
                feat[f"lag_{lag}"] = avg
            feat["lag_same_time_yesterday"] = avg
            feat["lag_same_time_last_week"] = avg
            feat["diff_1"] = feat["diff_48"] = feat["diff_336"] = 0
            for w in [4, 12, 48, 336]:
                feat[f"rolling_mean_{w}"] = avg
            for w in [4, 48]:
                feat[f"rolling_std_{w}"] = std
            feat["rolling_max_4"] = mx
            feat["ewm_mean_12"] = feat["ewm_mean_48"] = avg
            feat["hourly_trend"] = feat["daily_trend"] = feat["momentum"] = 0
            feat["volatility_ratio"] = std / (avg + 1)
            feat["lag_transfer_rate"] = 0.07
            feat["lag_fcr_rate"] = 0.93
            feat["lag_mean_hold"] = 50.0
            feat["lag_active_experts"] = avg * 1.2
            feat["lag_mean_occupancy"] = 60.0
            feat["lag_total_avail"] = avg * 600
            feat["rolling_experts_48"] = avg * 1.2
            feat["rolling_occupancy_48"] = 60.0
            feat["yoy_same_dow_hour_mean"] = avg

            X = pd.DataFrame([feat])[self.short_term_features].fillna(0)
            pred = self.short_term_model.predict(self.short_term_scaler.transform(X))[0]
        else:
            feat["hist_dow_hour_mean"] = avg
            feat["hist_dow_hour_std"] = std
            feat["hist_dow_hour_median"] = avg
            feat["hist_month_dow_hour_mean"] = avg
            feat["hist_month_dow_hour_std"] = std
            for key, pat in [("month", feat["month"]), ("hour", feat["hour"])]:
                p = self.historical_patterns.get(key, {})
                feat[f"hist_{key}_mean"] = p.get("mean", {}).get(pat, avg)
            feat["hist_time_slot_mean"] = avg
            feat["hist_week_of_year_mean"] = avg
            feat["hist_quarter_dow_mean"] = avg
            for w in [336, 672]:
                feat[f"rolling_mean_{w}"] = avg
            feat["ewm_mean_336"] = feat["ewm_mean_672"] = avg
            feat["hist_transfer_rate"] = 0.07
            feat["hist_fcr_rate"] = 0.93
            feat["hist_mean_hold"] = 50.0
            feat["hist_mean_experts"] = avg * 1.2
            feat["hist_mean_occupancy"] = 60.0
            feat["yoy_same_dow_hour_mean"] = avg
            feat["yoy_same_week_mean"] = avg
            feat["recent_quarter_mean"] = avg
            feat["recent_month_mean"] = avg
            feat["hist_recent_dow_hour_mean"] = avg
            feat["hist_dow_time_slot_mean"] = avg
            feat["hist_month_time_slot_mean"] = avg

            X = pd.DataFrame([feat])[self.long_term_features].fillna(0)
            X_s = self.long_term_scaler.transform(X)
            pred = np.mean([m.predict(X_s)[0] for m in self.long_term_model])

        return max(0, int(round(pred)))

    def predict_day(self, target_date, reference_date=None):
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        if reference_date is None:
            reference_date = self.max_training_date

        days_ahead = (pd.Timestamp(target_date) - reference_date).days
        model_used = "short-term" if days_ahead < self.short_term_threshold_days else "long-term"

        results = []
        current = datetime.combine(target_date, datetime.min.time())
        for _ in range(48):
            results.append({
                "interval_start": current,
                "predicted_calls": self.predict(current, reference_date),
                "model_used": model_used,
            })
            current += timedelta(minutes=30)

        return pd.DataFrame(results)

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
                "_hist_index": self._hist_index,
                "_holiday_profiles": self._holiday_profiles,
                "max_training_date": self.max_training_date,
                "short_term_threshold_days": self.short_term_threshold_days,
                "feature_importance": self.feature_importance,
            }, f)

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
        self._hist_index = data.get("_hist_index", {})
        self._holiday_profiles = data.get("_holiday_profiles", {})
        self.max_training_date = data["max_training_date"]
        self.short_term_threshold_days = data.get("short_term_threshold_days", 7)
        self.feature_importance = data.get("feature_importance", {})
