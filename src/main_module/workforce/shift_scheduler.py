from pathlib import Path

import numpy as np
import pandas as pd


class ShiftScheduler:
    BUSINESS_HOURS_UTC = list(range(13, 24)) + [0]
    SLOT_DURATION_MINUTES = 30
    MAX_SHIFT_SLOTS = 24  # 12 hours max shift
    MIN_SHIFT_SLOTS = 12  # 6 hours min shift
    BREAK_EVERY_N_SLOTS = 10  # Break every 5 hours
    BREAK_DURATION_SLOTS = 1

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self._agent_availability = None
        self._agent_meta = None

    def load_agent_patterns(self, tax_year=None, recent_days=90):
        # Only read the columns we actually use (skip handle_time and available_time)
        needed_cols = [
            "expert_id",
            "date",
            "interval_start_utc",
            "occupancy_pct",
            "primary_activity_category_30m",
        ]
        if tax_year is not None:
            needed_cols.append("tax_year")

        d4 = pd.read_parquet(
            self.data_dir / "dataset_4_expert_state_interval.parquet",
            columns=needed_cols,
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

    @property
    def total_active_agents(self):
        if self._agent_meta is not None:
            return len(self._agent_meta)
        if self._agent_availability is not None:
            return self._agent_availability["expert_id"].nunique()
        return 0

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

        empty = pd.DataFrame(
            columns=[
                "expert_id",
                "slot_start_utc",
                "slot_end_utc",
                "assignment",
                "shift_block",
            ]
        )

        if dow >= 5:
            return empty

        if self._agent_availability is None:
            raise ValueError("Call load_agent_patterns() first")

        dow_avail = self._agent_availability[
            self._agent_availability["dow"] == dow
        ]

        # ── Agent scoring ──
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
                / agent_scores["average_handle_time_seconds"].clip(
                    lower=1
                ),
                0,
                2,
            )
            agent_scores["priority"] = (
                0.4 * agent_scores["mean_work_freq"].fillna(0)
                + 0.3
                * (agent_scores["resolution_rate"].fillna(80) / 100)
                + 0.3 * agent_scores["efficiency"].fillna(1)
            )
        else:
            agent_scores["priority"] = agent_scores["mean_work_freq"]

        agent_scores = agent_scores.sort_values(
            "priority", ascending=False
        )

        if max_agents is not None:
            agent_scores = agent_scores.head(max_agents)

        # Preserve chronological hour order
        active_hours = list(dict.fromkeys(self.BUSINESS_HOURS_UTC))

        all_slot_keys = [
            (h, m) for h in active_hours for m in (0, 30)
        ]
        n_slots = len(all_slot_keys)

        # Build correct timestamps (hour 0 = next calendar day)
        next_day = target_date + pd.Timedelta(days=1)
        first_hour = active_hours[0]
        slot_ts = {}
        for h, m in all_slot_keys:
            base = next_day if h < first_hour else target_date
            slot_ts[(h, m)] = base.replace(hour=h, minute=m)

        # Map demand to slot keys
        slot_demand = {}
        for i, key in enumerate(all_slot_keys):
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

        agents_needed = np.zeros(n_slots, dtype=np.int32)
        for i, key in enumerate(all_slot_keys):
            demand = slot_demand.get(key, 0)
            needed = int(np.ceil(demand / calls_per_agent_per_slot))
            if min_agents_per_slot is not None:
                needed = max(needed, min_agents_per_slot)
            agents_needed[i] = needed

        # Vectorised preference lookup (replaces slow iterrows)
        pref_grouped = (
            dow_avail.groupby("expert_id")[["hour_utc", "work_frequency"]]
            .apply(
                lambda g: dict(
                    zip(g["hour_utc"].values, g["work_frequency"].values)
                ),
                include_groups=False,
            )
        )
        agent_hour_prefs = pref_grouped.to_dict()

        # Pre-estimate how many agents we'll need so we can stop scoring
        # agents we'll never use.  A rough upper bound is the peak slot
        # demand (an agent covers many slots, so this over-counts, but
        # keeps the candidate pool small).
        peak_demand = int(agents_needed.max()) if len(agents_needed) else 0
        candidate_pool_size = min(
            len(agent_scores), max(peak_demand * 2, 200)
        )
        agent_pool = agent_scores.head(candidate_pool_size)

        # Convert pool to numpy for fast iteration
        pool_eids = agent_pool["expert_id"].values

        # ── Scheduling engine ──
        assignments = []
        slot_fill = np.zeros(n_slots, dtype=np.int32)
        remaining_gap = int(np.maximum(agents_needed - slot_fill, 0).sum())
        scheduled_agents = set()

        # Score every (start, length) window; penalise overfill
        def _find_best_window(prefs, weight_prefs=True):
            best_start, best_len, best_score = -1, 0, -1.0
            for si in range(n_slots):
                max_wlen = min(self.MAX_SHIFT_SLOTS, n_slots - si)
                if max_wlen < self.MIN_SHIFT_SLOTS:
                    continue
                running_score = 0.0
                has_unfilled = False
                for offset in range(max_wlen):
                    idx = si + offset
                    gap = int(agents_needed[idx]) - int(slot_fill[idx])
                    if gap > 0:
                        has_unfilled = True
                        running_score += gap
                    else:
                        running_score -= 0.5 * abs(gap)
                    if weight_prefs:
                        running_score += prefs.get(
                            all_slot_keys[idx][0], 0.1
                        ) * 0.3
                    wlen = offset + 1
                    if (
                        wlen >= self.MIN_SHIFT_SLOTS
                        and has_unfilled
                        and running_score > best_score
                    ):
                        best_start = si
                        best_len = wlen
                        best_score = running_score
            if best_start < 0:
                return None
            return best_start, best_len

        # Assign full contiguous blocks, no mid-shift skipping
        def _assign_shift(eid, start_i, length):
            nonlocal remaining_gap
            shift_hour = all_slot_keys[start_i][0]
            block_label = f"shift_{shift_hour:02d}"
            work_count = 0
            for j in range(start_i, start_i + length):
                key = all_slot_keys[j]
                ts_start = slot_ts[key]
                ts_end = ts_start + pd.Timedelta(minutes=30)
                is_break = (
                    work_count > 0
                    and work_count % self.BREAK_EVERY_N_SLOTS == 0
                )
                assignments.append(
                    {
                        "expert_id": eid,
                        "slot_start_utc": ts_start,
                        "slot_end_utc": ts_end,
                        "assignment": "break" if is_break else "work",
                        "shift_block": block_label,
                    }
                )
                if not is_break:
                    if slot_fill[j] < agents_needed[j]:
                        remaining_gap -= 1
                    slot_fill[j] += 1
                work_count += 1
            scheduled_agents.add(eid)

        # Pass 1: demand-driven, weighted by agent preference
        for eid in pool_eids:
            if remaining_gap <= 0:
                break
            prefs = agent_hour_prefs.get(eid, {})
            win = _find_best_window(prefs, weight_prefs=True)
            if win:
                _assign_shift(eid, win[0], win[1])

        # Pass 2: gap-fill with remaining agents
        if remaining_gap > 0:
            for eid in pool_eids:
                if remaining_gap <= 0:
                    break
                if eid in scheduled_agents:
                    continue
                prefs = agent_hour_prefs.get(eid, {})
                win = _find_best_window(prefs, weight_prefs=False)
                if win:
                    _assign_shift(eid, win[0], win[1])

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
