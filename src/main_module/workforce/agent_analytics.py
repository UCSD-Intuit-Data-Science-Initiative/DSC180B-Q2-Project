from pathlib import Path

import numpy as np
import pandas as pd


class AgentAnalytics:
    _PERFORMANCE_WEIGHTS = {
        "resolution_rate": 0.30,
        "fcr_rate": 0.20,
        "efficiency_score": 0.20,
        "occupancy_score": 0.15,
        "volume_score": 0.15,
    }

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self._agents = None
        self._session_stats = None
        self._interval_stats = None

    def load(self, tax_year=None):
        self._load_metadata(tax_year)
        self._load_session_outcomes(tax_year)
        self._load_interval_stats(tax_year)
        self._build_composite_scores()
        return self

    def _load_metadata(self, tax_year):
        d2 = pd.read_parquet(
            self.data_dir / "dataset_2_expert_metadata.parquet"
        )
        if tax_year is not None:
            d2 = d2[d2["tax_year"] == tax_year]
        d2 = d2[d2["active_flg"] == "Y"].copy()
        d2 = d2[d2["contacts"] >= 50]
        d2["answer_rate"] = (
            d2["answered_contacts"] / d2["contacts"].clip(lower=1) * 100
        )
        self._agents = d2

    def _load_session_outcomes(self, tax_year):
        d3_path = self.data_dir / "dataset_3_historical_outcomes.parquet"
        if not d3_path.exists():
            self._session_stats = pd.DataFrame()
            return

        cols = [
            "tax_year",
            "expert_assigned_id",
            "first_call_resolution",
            "transfer_count",
            "hold_time_seconds",
            "duration_of_call_minutes",
        ]
        d3 = pd.read_parquet(d3_path, columns=cols)
        if tax_year is not None:
            d3 = d3[d3["tax_year"] == tax_year]

        d3["_fcr"] = (d3["first_call_resolution"] == "Y").astype(np.int8)
        d3["_transferred"] = (d3["transfer_count"] > 0).astype(np.int8)
        d3["hold_time_seconds"] = d3["hold_time_seconds"].fillna(0)
        d3["duration_of_call_minutes"] = d3["duration_of_call_minutes"].fillna(
            0
        )

        self._session_stats = (
            d3.groupby("expert_assigned_id")
            .agg(
                session_count=("_fcr", "count"),
                fcr_rate=("_fcr", "mean"),
                transfer_rate_d3=("_transferred", "mean"),
                median_hold=("hold_time_seconds", "median"),
                mean_hold=("hold_time_seconds", "mean"),
                p90_hold=("hold_time_seconds", lambda x: x.quantile(0.9)),
                median_duration=("duration_of_call_minutes", "median"),
                mean_duration=("duration_of_call_minutes", "mean"),
            )
            .reset_index()
            .rename(columns={"expert_assigned_id": "expert_id"})
        )
        self._session_stats["fcr_rate"] = self._session_stats["fcr_rate"] * 100

    def _load_interval_stats(self, tax_year):
        d4_path = self.data_dir / "dataset_4_expert_state_interval.parquet"
        if not d4_path.exists():
            self._interval_stats = pd.DataFrame()
            return

        cols = [
            "tax_year",
            "expert_id",
            "total_handle_time_seconds",
            "total_available_time_seconds",
            "occupancy_pct",
            "activity_break_meal_seconds",
        ]
        d4 = pd.read_parquet(d4_path, columns=cols)
        if tax_year is not None:
            d4 = d4[d4["tax_year"] == tax_year]

        self._interval_stats = (
            d4.groupby("expert_id")
            .agg(
                total_intervals=("occupancy_pct", "count"),
                mean_occupancy=("occupancy_pct", "mean"),
                total_handle_time=("total_handle_time_seconds", "sum"),
                total_available_time=("total_available_time_seconds", "sum"),
                total_break_time=("activity_break_meal_seconds", "sum"),
            )
            .reset_index()
        )
        self._interval_stats["utilization"] = (
            self._interval_stats["total_handle_time"]
            / (
                self._interval_stats["total_handle_time"]
                + self._interval_stats["total_available_time"]
            ).clip(lower=1)
            * 100
        )

    def _build_composite_scores(self):
        df = self._agents.copy()

        if len(self._session_stats) > 0:
            df = df.merge(self._session_stats, on="expert_id", how="left")
        else:
            df["fcr_rate"] = df["resolution_rate"]
            df["median_hold"] = df["average_hold_time_seconds"]

        if len(self._interval_stats) > 0:
            df = df.merge(self._interval_stats, on="expert_id", how="left")
        else:
            df["mean_occupancy"] = 50.0
            df["utilization"] = 50.0

        df["fcr_rate"] = df["fcr_rate"].fillna(df["resolution_rate"])
        df["mean_occupancy"] = df["mean_occupancy"].fillna(50)
        df["resolution_rate"] = df["fcr_rate"].fillna(df["resolution_rate"])

        handle_median = df["average_handle_time_seconds"].median()
        df["efficiency_score"] = np.clip(
            handle_median
            / df["average_handle_time_seconds"].clip(lower=1)
            * 100,
            0,
            100,
        )

        df["occupancy_score"] = df["mean_occupancy"].clip(upper=100)

        vol_max = df["contacts"].quantile(0.95)
        df["volume_score"] = np.clip(df["contacts"] / vol_max * 100, 0, 100)

        composite = np.zeros(len(df))
        for metric, weight in self._PERFORMANCE_WEIGHTS.items():
            if metric in df.columns:
                composite += weight * df[metric].fillna(0)
        df["composite_score"] = np.round(composite, 2)

        self._agents = df

    def top_performers(
        self, n=20, segment=None, sort_by="composite_score", ascending=False
    ):
        df = self._agents.copy()
        if segment is not None:
            df = df[df["expert_segment"] == segment]

        display_cols = [
            "expert_id",
            "expert_segment",
            "business_segment",
            "contacts",
            "answered_contacts",
            "resolution_rate",
            "transfer_rate",
            "average_handle_time_seconds",
            "average_hold_time_seconds",
            "composite_score",
        ]
        extra = ["fcr_rate", "median_hold", "mean_occupancy", "utilization"]
        display_cols += [c for c in extra if c in df.columns]

        col_map = {
            "name": "expert_id",
            "aht": "average_handle_time_seconds",
            "segment": "expert_segment",
        }
        sort_col = col_map.get(sort_by, sort_by)
        if sort_col not in df.columns:
            sort_col = "expert_id"

        return (
            df[display_cols]
            .sort_values(sort_col, ascending=ascending)
            .head(n)
            .reset_index(drop=True)
        )

    def segment_summary(self):
        return (
            self._agents.groupby("expert_segment")
            .agg(
                agent_count=("expert_id", "count"),
                mean_resolution=("resolution_rate", "mean"),
                mean_transfer=("transfer_rate", "mean"),
                mean_handle_time=("average_handle_time_seconds", "mean"),
                mean_contacts=("contacts", "mean"),
                mean_composite=("composite_score", "mean"),
            )
            .round(2)
            .sort_values("mean_composite", ascending=False)
        )

    def agent_profile(self, expert_id):
        row = self._agents[self._agents["expert_id"] == expert_id]
        if len(row) == 0:
            return None
        return row.iloc[0].to_dict()

    @property
    def agents_df(self):
        return self._agents.copy()
