from dataclasses import dataclass
from typing import List

import pandas as pd
from call_center_emulator import CallCenterEmulator, EmulatorConfig
from demand_forecasting_model import CallDemandForecaster
from hybrid_forecaster import HybridForecaster
from staffing_optimizer import (
    OptimizationThresholds,
    StaffingOptimizer,
)


@dataclass
class PipelineConfig:
    avg_handle_time: float = 600
    sla_threshold_seconds: float = 60
    max_avg_wait_time: float = 60
    min_sla_compliance: float = 80
    max_utilization: float = 85
    max_abandonment_rate: float = 5
    avg_patience_time: float = 180
    interval_minutes: int = 30
    emulator_model: str = "erlang_a"
    forecaster_type: str = "hybrid"


class WorkforceOptimizationPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        if self.config.forecaster_type == "hybrid":
            self.forecaster = HybridForecaster()
        else:
            self.forecaster = CallDemandForecaster()

        emulator_config = EmulatorConfig(
            avg_handle_time=self.config.avg_handle_time,
            sla_threshold_seconds=self.config.sla_threshold_seconds,
            avg_patience_time=self.config.avg_patience_time,
            interval_duration_seconds=self.config.interval_minutes * 60,
        )
        self.emulator = CallCenterEmulator(
            emulator_config, model=self.config.emulator_model
        )

        thresholds = OptimizationThresholds(
            max_avg_wait_time=self.config.max_avg_wait_time,
            min_sla_compliance=self.config.min_sla_compliance,
            max_utilization=self.config.max_utilization,
            max_abandonment_rate=self.config.max_abandonment_rate,
        )
        self.optimizer = StaffingOptimizer(self.emulator, thresholds)

        self.is_loaded = False

    def load_model(self, model_path: str = None):
        if model_path is None:
            model_path = (
                "hybrid_forecast_model.pkl"
                if self.config.forecaster_type == "hybrid"
                else "demand_forecast_model.pkl"
            )
        self.forecaster.load_model(model_path)
        self.is_loaded = True
        print(
            f"Pipeline initialized with {self.config.forecaster_type} model from {model_path}"
        )

    def train_model(self, data_path: str):
        print("Training demand forecasting model...")
        metrics, _ = self.forecaster.train(data_path)
        self.forecaster.save_model("demand_forecast_model.pkl")
        self.is_loaded = True
        return metrics

    def forecast_demand(self, target_date: str) -> pd.DataFrame:
        if not self.is_loaded:
            raise ValueError(
                "Model not loaded. Call load_model() or train_model() first."
            )

        return self.forecaster.predict_day(target_date)

    def optimize_staffing(
        self, target_date: str, custom_aht: float = None
    ) -> dict:
        if not self.is_loaded:
            raise ValueError(
                "Model not loaded. Call load_model() or train_model() first."
            )

        aht = custom_aht or self.config.avg_handle_time

        target_dt = pd.to_datetime(target_date)
        day_of_week = target_dt.dayofweek
        is_weekend = day_of_week >= 5

        print(f"\n{'='*80}")
        print(f"WORKFORCE OPTIMIZATION FOR {target_date}")
        print("Operating Hours: Monday-Friday, 5:00 AM - 5:00 PM PT")
        if is_weekend:
            print("** NOTE: This is a weekend - customer service is CLOSED **")
        print(f"{'='*80}")

        print("\n[STEP 1] DEMAND FORECASTING")
        print("-" * 40)
        demand_forecast = self.forecaster.predict_day(target_date)
        calls_per_interval = demand_forecast["predicted_calls"].tolist()

        if "time" in demand_forecast.columns:
            time_col = demand_forecast["time"].tolist()
        else:
            time_col = [
                dt.strftime("%H:%M")
                for dt in demand_forecast["interval_start"]
            ]

        print("\n[STEP 2] STAFFING OPTIMIZATION")
        print("-" * 40)
        print("Constraints:")
        print(f"  - Max avg wait time: {self.config.max_avg_wait_time}s")
        print(f"  - Min SLA compliance: {self.config.min_sla_compliance}%")
        print(f"  - Max utilization: {self.config.max_utilization}%")
        print(f"  - Max abandonment: {self.config.max_abandonment_rate}%")
        print(f"  - Avg handle time: {aht}s")

        optimization_result = self.optimizer.optimize_day(
            calls_per_interval, aht
        )

        print("\n[STEP 3] RESULTS")
        print("-" * 40)

        results_df = pd.DataFrame(
            {
                "time": time_col,
                "predicted_calls": calls_per_interval,
                "optimal_experts": optimization_result[
                    "optimal_experts_per_interval"
                ],
                "avg_wait_time": [
                    r.metrics.avg_wait_time
                    for r in optimization_result["interval_results"]
                ],
                "sla_compliance": [
                    r.metrics.sla_compliance
                    for r in optimization_result["interval_results"]
                ],
                "utilization": [
                    r.metrics.utilization_rate
                    for r in optimization_result["interval_results"]
                ],
            }
        )

        active_intervals = results_df[results_df["predicted_calls"] > 0]

        if len(active_intervals) > 0:
            print(
                f"\n{'Time':<8} {'Calls':<8} {'Experts':<10} {'Wait(s)':<10} {'SLA%':<8} {'Util%':<8}"
            )
            print("-" * 60)
            for _, row in active_intervals.iterrows():
                print(
                    f"{row['time']:<8} {int(row['predicted_calls']):<8} "
                    f"{int(row['optimal_experts']):<10} "
                    f"{row['avg_wait_time']:<10.1f} "
                    f"{row['sla_compliance']:<8.1f} "
                    f"{row['utilization']:<8.1f}"
                )

        summary = optimization_result["day_simulation"]["summary"]

        print(f"\n{'='*60}")
        print("SUPPLY RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"  Total predicted calls: {sum(calls_per_interval)}")
        print(f"  Peak interval demand: {max(calls_per_interval)} calls")
        print(f"  Peak experts needed: {optimization_result['peak_experts']}")
        print(
            f"  Total expert-intervals: {optimization_result['total_experts_needed']}"
        )
        print("\n  Expected Performance:")
        print(f"    - Avg wait time: {summary['avg_wait_time']:.1f}s")
        print(f"    - SLA compliance: {summary['avg_sla_compliance']:.1f}%")
        print(f"    - Utilization: {summary['avg_utilization']:.1f}%")
        print(f"    - Abandonment: {summary['abandonment_rate']:.1f}%")

        return {
            "date": target_date,
            "demand_forecast": demand_forecast,
            "optimization_result": optimization_result,
            "schedule": results_df,
            "summary": {
                "total_calls": sum(calls_per_interval),
                "peak_calls": max(calls_per_interval),
                "peak_experts": optimization_result["peak_experts"],
                "total_expert_intervals": optimization_result[
                    "total_experts_needed"
                ],
                "avg_wait_time": summary["avg_wait_time"],
                "sla_compliance": summary["avg_sla_compliance"],
                "utilization": summary["avg_utilization"],
                "abandonment_rate": summary["abandonment_rate"],
            },
        }

    def compare_scenarios(
        self, target_date: str, scenarios: List[dict]
    ) -> pd.DataFrame:
        if not self.is_loaded:
            raise ValueError(
                "Model not loaded. Call load_model() or train_model() first."
            )

        print(f"\n{'='*80}")
        print(f"SCENARIO COMPARISON FOR {target_date}")
        print(f"{'='*80}")

        demand_forecast = self.forecaster.predict_day(target_date)
        calls_per_interval = demand_forecast["predicted_calls"].tolist()

        results = []

        for scenario in scenarios:
            name = scenario.get("name", "Unnamed")

            thresholds = OptimizationThresholds(
                max_avg_wait_time=scenario.get(
                    "max_wait", self.config.max_avg_wait_time
                ),
                min_sla_compliance=scenario.get(
                    "min_sla", self.config.min_sla_compliance
                ),
                max_utilization=scenario.get(
                    "max_util", self.config.max_utilization
                ),
                max_abandonment_rate=scenario.get(
                    "max_abandon", self.config.max_abandonment_rate
                ),
            )

            temp_optimizer = StaffingOptimizer(self.emulator, thresholds)
            aht = scenario.get("aht", self.config.avg_handle_time)

            opt_result = temp_optimizer.optimize_day(calls_per_interval, aht)
            summary = opt_result["day_simulation"]["summary"]

            results.append(
                {
                    "Scenario": name,
                    "Max Wait": scenario.get(
                        "max_wait", self.config.max_avg_wait_time
                    ),
                    "Min SLA": scenario.get(
                        "min_sla", self.config.min_sla_compliance
                    ),
                    "Peak Experts": opt_result["peak_experts"],
                    "Total Expert-Intervals": opt_result[
                        "total_experts_needed"
                    ],
                    "Actual Avg Wait": f"{summary['avg_wait_time']:.1f}s",
                    "Actual SLA": f"{summary['avg_sla_compliance']:.1f}%",
                    "Abandonment": f"{summary['abandonment_rate']:.1f}%",
                }
            )

        results_df = pd.DataFrame(results)
        print(f"\n{results_df.to_string(index=False)}")

        return results_df

    def what_if_analysis(self, target_date: str, fixed_experts: int) -> dict:
        if not self.is_loaded:
            raise ValueError(
                "Model not loaded. Call load_model() or train_model() first."
            )

        demand_forecast = self.forecaster.predict_day(target_date)
        calls_per_interval = demand_forecast["predicted_calls"].tolist()

        experts_per_interval = [fixed_experts] * len(calls_per_interval)

        simulation = self.emulator.simulate_day(
            experts_per_interval,
            calls_per_interval,
            self.config.avg_handle_time,
        )

        print(f"\n{'='*60}")
        print(f"WHAT-IF ANALYSIS: {fixed_experts} EXPERTS PER INTERVAL")
        print(f"{'='*60}")
        print(f"Date: {target_date}")
        print(f"Total calls: {sum(calls_per_interval)}")
        print("\nProjected Metrics:")
        print(
            f"  Avg wait time: {simulation['summary']['avg_wait_time']:.1f}s"
        )
        print(
            f"  SLA compliance: {simulation['summary']['avg_sla_compliance']:.1f}%"
        )
        print(
            f"  Utilization: {simulation['summary']['avg_utilization']:.1f}%"
        )
        print(
            f"  Abandonment: {simulation['summary']['abandonment_rate']:.1f}%"
        )
        print(f"  Calls handled: {simulation['summary']['total_handled']}")
        print(f"  Calls abandoned: {simulation['summary']['total_abandoned']}")

        return simulation


def main():
    print("=" * 80)
    print("WORKFORCE OPTIMIZATION PIPELINE")
    print("=" * 80)

    config = PipelineConfig(
        avg_handle_time=600,
        sla_threshold_seconds=60,
        max_avg_wait_time=60,
        min_sla_compliance=80,
        max_utilization=85,
        max_abandonment_rate=5,
    )

    pipeline = WorkforceOptimizationPipeline(config)

    pipeline.load_model("demand_forecast_model.pkl")

    pipeline.optimize_staffing("2025-04-15")

    print("\n\n")
    scenarios = [
        {"name": "Relaxed", "max_wait": 120, "min_sla": 70},
        {"name": "Standard", "max_wait": 60, "min_sla": 80},
        {"name": "Strict", "max_wait": 30, "min_sla": 90},
        {"name": "Premium", "max_wait": 15, "min_sla": 95},
    ]
    pipeline.compare_scenarios("2025-04-15", scenarios)

    print("\n\n")
    pipeline.what_if_analysis("2025-04-15", fixed_experts=5)

    print("\n\n")
    print("=" * 80)
    print("OFF-SEASON COMPARISON (July)")
    print("=" * 80)
    pipeline.optimize_staffing("2025-07-15")


if __name__ == "__main__":
    main()
