"""
Run the full workforce optimization pipeline:
  1. Train the demand forecasting model (HybridForecaster)
  2. Forecast demand for a target date
  3. Feed predictions into PerformanceEmulator + SupplyOptimizer
  4. Print staffing recommendations

Usage:
    cd scripts/
    python run_pipeline.py
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to the Python path so main_module is importable
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from main_module.workforce import (
    HybridForecaster,
    CallCenterEmulator,
    EmulatorConfig,
    SupplyOptimizer,
    OptimizationConstraints,
)

DATA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "interim" / "mock_intuit_2year_data.csv")


def main():
    # ---------------------------------------------------------------
    # Step 1: Train the demand forecasting model
    # ---------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: TRAINING DEMAND FORECASTING MODEL")
    print("=" * 70)

    forecaster = HybridForecaster()
    forecaster.train(DATA_PATH, test_year=2025, tune_hyperparameters=True, n_trials=10)

    # ---------------------------------------------------------------
    # Step 2: Forecast demand for a target date
    # ---------------------------------------------------------------
    target_date = "2025-04-15"  # Tax deadline day

    print("\n" + "=" * 70)
    print(f"STEP 2: FORECASTING DEMAND FOR {target_date}")
    print("=" * 70)

    demand_df = forecaster.predict_day(target_date)

    # HybridForecaster doesn't include time/is_open columns — derive them
    demand_df["time"] = demand_df["interval_start"].dt.strftime("%H:%M")
    demand_df["is_open"] = (
        (demand_df["interval_start"].dt.hour >= 5) &
        (demand_df["interval_start"].dt.hour < 17) &
        (demand_df["interval_start"].dt.dayofweek < 5)
    )

    # Filter to open intervals with predicted calls > 0
    open_intervals = demand_df[demand_df["is_open"] & (demand_df["predicted_calls"] > 0)]

    if open_intervals.empty:
        print("No open intervals with calls predicted. Exiting.")
        return

    calls_per_interval = open_intervals["predicted_calls"].tolist()
    times = open_intervals["time"].tolist()

    print(f"\n  Open intervals with calls: {len(open_intervals)}")
    print(f"  Total predicted calls: {sum(calls_per_interval)}")

    # ---------------------------------------------------------------
    # Step 3: Run emulator + optimizer for each interval
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: OPTIMIZING STAFFING WITH EMULATOR + OPTIMIZER")
    print("=" * 70)

    emulator = CallCenterEmulator(
        config=EmulatorConfig(
            avg_handle_time=300,          # 5 min average call
            sla_threshold_seconds=60,     # 60s SLA window
            interval_duration_seconds=1800,  # 30 min interval
        )
    )
    optimizer = SupplyOptimizer(emulator, max_supply=500)

    constraints = OptimizationConstraints(
        min_sla=0.80,           # 80% of calls answered within 60s
        max_wait_time=60.0,     # avg wait under 60s
        max_occupancy=0.85,     # agents no more than 85% utilized
    )

    print(f"\n  Constraints:")
    print(f"    Min SLA:       {constraints.min_sla:.0%}")
    print(f"    Max wait time: {constraints.max_wait_time}s")
    print(f"    Max occupancy: {constraints.max_occupancy:.0%}")

    # Optimize each interval
    results = []
    for time_str, demand in zip(times, calls_per_interval):
        result = optimizer.optimize(int(demand), constraints)
        results.append((time_str, demand, result))

    # ---------------------------------------------------------------
    # Step 4: Print results
    # ---------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  STAFFING SCHEDULE FOR {target_date}")
    print(f"{'='*80}")
    print(f"\n{'Time':<8} {'Demand':<10} {'Agents':<10} {'Wait(s)':<10} {'SLA%':<10} {'Occupancy':<12} {'Feasible':<10}")
    print("-" * 70)

    total_agents = 0
    for time_str, demand, result in results:
        m = result.predicted_metrics
        total_agents += result.headcount
        print(
            f"{time_str:<8} {demand:<10} {result.headcount:<10} "
            f"{m.avg_wait_time:<10.1f} {m.sla_compliance:<9.1f}% "
            f"{m.utilization_rate:<11.1f}% {result.is_feasible}"
        )

    # Summary
    peak_agents = max(r.headcount for _, _, r in results)
    peak_demand = max(d for _, d, _ in results)
    feasible_count = sum(1 for _, _, r in results if r.is_feasible)

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Total predicted calls : {sum(calls_per_interval)}")
    print(f"  Peak demand           : {peak_demand} calls")
    print(f"  Peak agents needed    : {peak_agents}")
    print(f"  Total agent-intervals : {total_agents}")
    print(f"  Feasible intervals    : {feasible_count}/{len(results)}")


if __name__ == "__main__":
    main()