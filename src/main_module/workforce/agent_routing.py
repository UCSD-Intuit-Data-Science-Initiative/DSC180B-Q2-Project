from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from main_module.workforce.agent_analytics import AgentAnalytics


class AgentRouter:
    def __init__(self, data_dir: str = "data/parquet"):
        self.data_dir = Path(data_dir)
        self._product_affinity = None
        self._channel_affinity = None
        self._agents = None
        self._dow_hour_available = None

    def load(self, tax_year=None):
        self._load_product_affinity(tax_year)
        self._load_channel_affinity(tax_year)
        self._load_agents(tax_year)
        self._load_availability_patterns(tax_year)
        return self

    def _data_path(self, name: str) -> Optional[Path]:
        for base in [self.data_dir, self.data_dir.parent / "raw"]:
            p = base / name
            if p.exists():
                return p
        return None

    def _load_product_affinity(self, tax_year):
        d1_path = self._data_path("dataset_1_call_related.parquet")
        if d1_path is None:
            self._product_affinity = pd.DataFrame(
                columns=["expert_id", "product_group_sku", "volume"]
            )
            return

        cols = ["expert_id", "product_group_sku", "answered_flag"]
        if tax_year is not None:
            cols = cols + ["tax_year"]
        d1 = pd.read_parquet(d1_path, columns=cols)
        d1 = d1[d1["answered_flag"]].copy()
        if tax_year is not None and "tax_year" in d1.columns:
            d1 = d1[d1["tax_year"] == tax_year]

        d1["product_group_sku"] = d1["product_group_sku"].fillna("UNKNOWN")
        aff = (
            d1.groupby(["expert_id", "product_group_sku"])
            .size()
            .reset_index(name="volume")
        )
        self._product_affinity = aff

    def _load_channel_affinity(self, tax_year):
        d1_path = self._data_path("dataset_1_call_related.parquet")
        if d1_path is None:
            self._channel_affinity = pd.DataFrame(
                columns=["expert_id", "channel", "volume"]
            )
            return

        cols = ["expert_id", "communication_channel_type", "answered_flag"]
        if tax_year is not None:
            cols = cols + ["tax_year"]
        d1 = pd.read_parquet(d1_path, columns=cols)
        d1 = d1[d1["answered_flag"]].copy()
        if tax_year is not None and "tax_year" in d1.columns:
            d1 = d1[d1["tax_year"] == tax_year]

        ch = d1["communication_channel_type"].fillna("UNKNOWN").astype(str)
        pat = r"(INBOUND|OUTBOUND|CHAT|PHONE|VOICE)"
        d1["channel"] = ch.str.extract(pat, expand=False)
        d1["channel"] = d1["channel"].fillna("OTHER")
        aff = (
            d1.groupby(["expert_id", "channel"])
            .size()
            .reset_index(name="volume")
        )
        self._channel_affinity = aff

    def _load_agents(self, tax_year):
        d2_path = self.data_dir / "dataset_2_expert_metadata.parquet"
        if not d2_path.exists():
            raw_path = self.data_dir.parent / "raw"
            d2_path = raw_path / "dataset_2_expert_metadata.parquet"
        if not d2_path.exists():
            self._agents = pd.DataFrame()
            return

        d2 = pd.read_parquet(d2_path)
        if tax_year is not None:
            d2 = d2[d2["tax_year"] == tax_year]
        d2 = d2[d2["active_flg"] == "Y"]
        d2 = d2[d2["contacts"] >= 50]
        self._agents = d2

    def _load_availability_patterns(self, tax_year):
        d4_path = self.data_dir / "dataset_4_expert_state_interval.parquet"
        if not d4_path.exists():
            raw_path = self.data_dir.parent / "raw"
            d4_path = raw_path / "dataset_4_expert_state_interval.parquet"
        if not d4_path.exists():
            self._dow_hour_available = None
            return

        d4 = pd.read_parquet(
            d4_path,
            columns=[
                "tax_year",
                "expert_id",
                "interval_start_utc",
                "primary_activity_category_30m",
            ],
        )
        if tax_year is not None:
            d4 = d4[d4["tax_year"] == tax_year]

        d4["interval_start_utc"] = pd.to_datetime(d4["interval_start_utc"])
        d4["dow"] = d4["interval_start_utc"].dt.dayofweek.astype(int)
        d4["hour_utc"] = d4["interval_start_utc"].dt.hour.astype(int)
        d4["is_working"] = (
            d4["primary_activity_category_30m"]
            .isin(["handle_work", "available"])
            .astype(np.int8)
        )
        agg = (
            d4.groupby(["expert_id", "dow", "hour_utc"])
            .agg(work_freq=("is_working", "mean"))
            .reset_index()
        )
        self._dow_hour_available = agg[agg["work_freq"] >= 0.3]

    def recommend(
        self,
        product_group_sku: Optional[str] = None,
        channel: Optional[str] = None,
        date_utc: Optional[str] = None,
        top_n: int = 5,
        agent_analytics: Optional["AgentAnalytics"] = None,
    ):
        if self._agents is None or len(self._agents) == 0:
            return []

        agents = self._agents.copy()
        agents["score"] = 0.0
        agents["affinity_volume"] = 0

        if agent_analytics is not None and agent_analytics._agents is not None:
            aa = agent_analytics._agents
            if "resolution_rate" in aa.columns:
                rr = aa["resolution_rate"]
            elif "fcr_rate" in aa.columns:
                rr = aa["fcr_rate"]
            else:
                rr = None
            tr = aa["transfer_rate"] if "transfer_rate" in aa.columns else None
            if tr is None and "transfer_rate_d3" in aa.columns:
                tr = aa["transfer_rate_d3"] * 100
            if rr is not None:
                merge_df = aa[["expert_id"]].copy()
                merge_df["resolution_rate"] = rr
                merge_df = merge_df.drop_duplicates("expert_id", keep="first")
                agents = agents.drop(
                    columns=["resolution_rate"], errors="ignore"
                )
                agents = agents.merge(merge_df, on="expert_id", how="left")
            if tr is not None:
                merge_df = aa[["expert_id"]].copy()
                merge_df["transfer_rate"] = tr
                merge_df = merge_df.drop_duplicates("expert_id", keep="first")
                agents = agents.drop(
                    columns=["transfer_rate"], errors="ignore"
                )
                agents = agents.merge(merge_df, on="expert_id", how="left")

        if "resolution_rate" not in agents.columns:
            agents["resolution_rate"] = 80.0
        agents["resolution_rate"] = agents["resolution_rate"].fillna(80)
        if "transfer_rate" not in agents.columns:
            agents["transfer_rate"] = 10.0
        agents["transfer_rate"] = agents["transfer_rate"].fillna(10)

        if date_utc is not None and self._dow_hour_available is not None:
            ts = pd.Timestamp(date_utc)
            dow = ts.dayofweek
            hour_utc = ts.hour
            avail = self._dow_hour_available[
                (self._dow_hour_available["dow"] == dow)
                & (self._dow_hour_available["hour_utc"] == hour_utc)
            ]
            available_ids = set(avail["expert_id"].tolist())
            if available_ids:
                agents = agents[agents["expert_id"].isin(available_ids)]

        if len(agents) == 0:
            return []

        product_matched = False
        if product_group_sku and self._product_affinity is not None:
            prod_aff = self._product_affinity[
                self._product_affinity["product_group_sku"]
                == product_group_sku
            ]
            if len(prod_aff) > 0:
                product_matched = True
                vol_max = prod_aff["volume"].max()
                for _, row in prod_aff.iterrows():
                    eid = row["expert_id"]
                    vol = row["volume"]
                    mask = agents["expert_id"] == eid
                    if mask.any():
                        exp_score = vol / vol_max if vol_max > 0 else 0
                        agents.loc[mask, "score"] += 0.5 * exp_score
                        agents.loc[mask, "affinity_volume"] = vol
            if not product_matched:
                prod_aff = self._product_affinity[
                    self._product_affinity["product_group_sku"]
                    .astype(str)
                    .str.upper()
                    .str.contains(product_group_sku.upper(), na=False)
                ]
                if len(prod_aff) > 0:
                    product_matched = True
                    merged = prod_aff.groupby("expert_id")["volume"].sum()
                    vol_max = merged.max()
                    for eid, vol in merged.items():
                        mask = agents["expert_id"] == eid
                        if mask.any():
                            exp_score = vol / vol_max if vol_max > 0 else 0
                            agents.loc[mask, "score"] += 0.5 * exp_score
                            agents.loc[mask, "affinity_volume"] = int(vol)
        use_segment = (
            not product_matched
            and product_group_sku
            and "expert_segment" in agents.columns
        )
        if use_segment:
            seg = agents["expert_segment"].fillna("").astype(str)
            q = product_group_sku.upper()
            mask = seg.str.upper().str.contains(q, na=False)
            if mask.any():
                agents.loc[mask, "score"] += 0.25

        if channel and self._channel_affinity is not None:
            ch_upper = channel.upper()
            ch_match = self._channel_affinity[
                self._channel_affinity["channel"]
                .str.upper()
                .str.contains(ch_upper)
            ]
            if len(ch_match) == 0:
                ch_match = self._channel_affinity[
                    self._channel_affinity["channel"] == channel
                ]
            if len(ch_match) > 0:
                ch_vol = (
                    ch_match.groupby("expert_id")["volume"].sum().reset_index()
                )
                vol_max = ch_vol["volume"].max()
                for _, row in ch_vol.iterrows():
                    eid = row["expert_id"]
                    vol = row["volume"]
                    mask = agents["expert_id"] == eid
                    if mask.any():
                        exp_score = vol / vol_max if vol_max > 0 else 0
                        agents.loc[mask, "score"] += 0.3 * exp_score

        agents["score"] += 0.2 * (agents["resolution_rate"] / 100)
        agents["score"] -= 0.05 * (agents["transfer_rate"] / 100)
        agents = agents.drop_duplicates("expert_id", keep="first")
        sort_cols = ["score"]
        if "affinity_volume" in agents.columns:
            sort_cols.append("affinity_volume")
        if "contacts" in agents.columns:
            sort_cols.append("contacts")
        sort_cols.append("expert_id")
        asc = [False] * len(sort_cols)
        agents = agents.sort_values(sort_cols, ascending=asc)
        agents = agents.head(top_n)

        result = []
        for _, row in agents.iterrows():
            result.append(
                {
                    "expert_id": str(row["expert_id"]),
                    "name": f"Agent {str(row['expert_id'])[-4:]}",
                    "segment": str(
                        row.get("expert_segment", "Unknown") or "Unknown"
                    ),
                    "resolution_rate": round(float(row["resolution_rate"]), 1),
                    "transfer_rate": round(float(row["transfer_rate"]), 1),
                    "affinity_volume": int(row["affinity_volume"]),
                    "score": round(float(row["score"]), 3),
                }
            )
        return result
