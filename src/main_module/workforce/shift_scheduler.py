from pathlib import Path

import numpy as np
import pandas as pd


class ShiftScheduler:
    BUSINESS_HOURS_UTC = list(range(13, 24)) + [0]
    SLOT_DURATION_MINUTES = 30
    MAX_SHIFT_SLOTS = 20
    MIN_SHIFT_SLOTS = 12
    BREAK_EVERY_N_SLOTS = 6
    BREAK_DURATION_SLOTS = 1

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self._agent_availability = None
        self._agent_meta = None

    def load_agent_patterns(self, tax_year=None, recent_days=90):
        d4 = pd.read_parquet(
            self.data_dir / "dataset_4_expert_state_interval.parquet",
            columns=[
                "tax_year",
                "expert_id",
                "date",
                "interval_start_utc",
                "total_handle_time_seconds",
                "total_available_time_seconds",
                "occupancy_pct",
                "primary_activity_category_30m",
            ],
        )
        if tax_year is not None:
            d4 = d4[d4["tax_year"] == tax_year]

        d4["date"] = pd.to_datetime(d4["date"])
        cutoff = d4["date"].max() - pd.Timedelta(days=recent_days)
        d4 = d4[d4["date"] >= cutoff]

        d4["interval_start_utc"] = pd.to_datetime(d4["interval_start_utc"])
        d4["hour_utc"] = d4["interval_start_utc"].dt.hour.astype(int)
        d4["dow"] = d4["interval_start_utc"].dt.dayofweek.astype(int)
        d4["is_working"] = (
            d4["primary_activity_category_30m"]
            .isin(["handle_work", "available"])
            .astype(np.int8)
        )

        agent_patterns = (
            d4.groupby(["expert_id", "dow", "hour_utc"])
            .agg(
                work_frequency=("is_working", "mean"),
                mean_occupancy=("occupancy_pct", "mean"),
                total_intervals=("is_working", "count"),
            )
            .reset_index()
        )
        self._agent_availability = agent_patterns

        d2_path = self.data_dir / "dataset_2_expert_metadata.parquet"
        if d2_path.exists():
            d2 = pd.read_parquet(d2_path)
            if tax_year is not None:
                d2 = d2[d2["tax_year"] == tax_year]
            d2 = d2[d2["active_flg"] == "Y"]
            self._agent_meta = d2[
                [
                    "expert_id",
                    "expert_segment",
                    "business_segment",
                    "resolution_rate",
                    "average_handle_time_seconds",
                    "contacts",
                    "skill_certifications",
                ]
            ].copy()
        return self

    def schedule_day(
        self,
        target_date,
        demand_by_slot,
        min_agents_per_slot=None,
        prefer_high_performers=True,
        max_agents=None,
    ):
        target_date = pd.to_datetime(target_date)
        dow = target_date.dayofweek

        if dow >= 5:
            return pd.DataFrame(
                columns=[
                    "expert_id",
                    "slot_start_utc",
                    "slot_end_utc",
                    "assignment",
                    "shift_block",
                ]
            )

        if self._agent_availability is None:
            raise ValueError("Call load_agent_patterns() first")

        dow_avail = self._agent_availability[
            self._agent_availability["dow"] == dow
        ].copy()

        agent_scores = (
            dow_avail.groupby("expert_id")
            .agg(
                available_hours=("work_frequency", "sum"),
                mean_work_freq=("work_frequency", "mean"),
                mean_occ=("mean_occupancy", "mean"),
            )
            .reset_index()
        )

        if self._agent_meta is not None and prefer_high_performers:
            agent_scores = agent_scores.merge(
                self._agent_meta[
                    [
                        "expert_id",
                        "resolution_rate",
                        "average_handle_time_seconds",
                    ]
                ],
                on="expert_id",
                how="left",
            )
            handle_median = agent_scores[
                "average_handle_time_seconds"
            ].median()
            agent_scores["efficiency"] = np.clip(
                handle_median
                / agent_scores["average_handle_time_seconds"].clip(lower=1),
                0,
                2,
            )
            agent_scores["priority"] = (
                0.4 * agent_scores["mean_work_freq"].fillna(0)
                + 0.3 * (agent_scores["resolution_rate"].fillna(80) / 100)
                + 0.3 * agent_scores["efficiency"].fillna(1)
            )
        else:
            agent_scores["priority"] = agent_scores["mean_work_freq"]

        agent_scores = agent_scores.sort_values("priority", ascending=False)

        if max_agents is not None:
            agent_scores = agent_scores.head(max_agents)

        active_slots = sorted(set(self.BUSINESS_HOURS_UTC))
        slot_times = []
        for h in active_slots:
            for m in [0, 30]:
                slot_start = target_date.replace(hour=h, minute=m)
                slot_times.append(slot_start)

        slot_demand = {}
        for i, st in enumerate(slot_times):
            key = (st.hour, st.minute)
            if isinstance(demand_by_slot, dict):
                slot_demand[key] = demand_by_slot.get(
                    key, demand_by_slot.get(i, 0)
                )
            elif isinstance(demand_by_slot, (list, np.ndarray)):
                slot_demand[key] = (
                    demand_by_slot[i] if i < len(demand_by_slot) else 0
                )
            else:
                slot_demand[key] = 0

        avg_handle_time = 1200
        if self._agent_meta is not None:
            avg_handle_time = self._agent_meta[
                "average_handle_time_seconds"
            ].median()

        calls_per_agent_per_slot = max(1, 1800 / avg_handle_time)

        agents_needed_per_slot = {}
        for key, demand in slot_demand.items():
            needed = int(np.ceil(demand / calls_per_agent_per_slot))
            if min_agents_per_slot is not None:
                needed = max(needed, min_agents_per_slot)
            agents_needed_per_slot[key] = needed

        agent_hour_prefs = {}
        for _, row in dow_avail.iterrows():
            eid = row["expert_id"]
            h = row["hour_utc"]
            if eid not in agent_hour_prefs:
                agent_hour_prefs[eid] = {}
            agent_hour_prefs[eid][h] = row["work_frequency"]

        assignments = []
        slot_fill = {key: 0 for key in agents_needed_per_slot}

        for _, agent in agent_scores.iterrows():
            eid = agent["expert_id"]
            prefs = agent_hour_prefs.get(eid, {})

            preferred_hours = sorted(
                prefs.keys(), key=lambda h: prefs.get(h, 0), reverse=True
            )
            preferred_hours = [h for h in preferred_hours if h in active_slots]

            if not preferred_hours:
                continue

            shift_start_hour = int(preferred_hours[0])
            start_idx = (
                active_slots.index(shift_start_hour)
                if shift_start_hour in active_slots
                else 0
            )

            shift_hours = active_slots[
                start_idx : start_idx + self.MAX_SHIFT_SLOTS // 2
            ]
            if len(shift_hours) < self.MIN_SHIFT_SLOTS // 2:
                continue

            slot_count = 0
            for h in shift_hours:
                for m_offset in [0, 30]:
                    key = (h, m_offset)
                    if key not in agents_needed_per_slot:
                        continue
                    if slot_fill[key] >= agents_needed_per_slot[key] * 1.1:
                        continue

                    slot_start = target_date.replace(hour=h, minute=m_offset)
                    slot_end = slot_start + pd.Timedelta(minutes=30)

                    if (
                        slot_count > 0
                        and slot_count % (self.BREAK_EVERY_N_SLOTS * 2) == 0
                    ):
                        assignments.append(
                            {
                                "expert_id": eid,
                                "slot_start_utc": slot_start,
                                "slot_end_utc": slot_end,
                                "assignment": "break",
                                "shift_block": f"shift_{shift_start_hour:02d}",
                            }
                        )
                    else:
                        assignments.append(
                            {
                                "expert_id": eid,
                                "slot_start_utc": slot_start,
                                "slot_end_utc": slot_end,
                                "assignment": "work",
                                "shift_block": f"shift_{shift_start_hour:02d}",
                            }
                        )
                        slot_fill[key] = slot_fill.get(key, 0) + 1

                    slot_count += 1

        result = pd.DataFrame(assignments)
        if len(result) > 0:
            result = result.sort_values(
                ["slot_start_utc", "expert_id"]
            ).reset_index(drop=True)
        return result

    def coverage_report(self, schedule_df, demand_by_slot):
        if len(schedule_df) == 0:
            return pd.DataFrame()

        working = schedule_df[schedule_df["assignment"] == "work"]
        supply = (
            working.groupby("slot_start_utc").size().rename("agents_assigned")
        )

        rows = []
        for slot_start, count in supply.items():
            h, m = slot_start.hour, slot_start.minute
            key = (h, m)
            if isinstance(demand_by_slot, dict):
                demand = demand_by_slot.get(key, 0)
            else:
                demand = 0
            rows.append(
                {
                    "slot_start_utc": slot_start,
                    "agents_assigned": count,
                    "predicted_demand": demand,
                    "coverage_ratio": count / max(demand / 1.5, 1),
                }
            )

        report = pd.DataFrame(rows)
        return report.sort_values("slot_start_utc").reset_index(drop=True)

    def agent_shift_summary(self, schedule_df):
        if len(schedule_df) == 0:
            return pd.DataFrame()

        summary = (
            schedule_df.groupby("expert_id")
            .agg(
                shift_block=("shift_block", "first"),
                shift_start=("slot_start_utc", "min"),
                shift_end=("slot_end_utc", "max"),
                total_slots=("assignment", "count"),
                work_slots=("assignment", lambda x: (x == "work").sum()),
                break_slots=("assignment", lambda x: (x == "break").sum()),
            )
            .reset_index()
        )
        summary["shift_hours"] = summary["total_slots"] * 0.5
        summary["work_hours"] = summary["work_slots"] * 0.5
        return summary.sort_values("shift_start").reset_index(drop=True)
