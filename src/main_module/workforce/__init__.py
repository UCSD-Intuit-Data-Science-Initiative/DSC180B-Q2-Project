"""
Workforce Management System
===========================

Core components for contact center staffing optimization:

1. DemandForecaster - Predicts incoming call volumes
2. PerformanceEmulator - Digital Twin predicting wait time/SLA
3. SupplyOptimizer - Finds minimum staffing meeting constraints
4. Pipeline - Orchestrates the full workflow

Data Flow:
    historical_data -> DemandForecaster.fit()
                                |
    start_time -> DemandForecaster.forecast() -> demand
                                                    |
                                                    v
                    PerformanceEmulator.predict_metrics(demand, supply)
                                                    |
                                                    v
                    SupplyOptimizer.optimize(demand, constraints) -> headcount
"""

from .demand_forecaster import DemandForecaster, ForecastResult
from .call_center_emulator import CallCenterEmulator, EmulatorConfig, EmulatorMetrics
from .supply_optimizer import SupplyOptimizer, OptimizationConstraints, OptimalSupply
from .pipeline import Pipeline, PipelineResult
from .hybrid_forecaster import HybridForecaster

__all__ = [
    "DemandForecaster",
    "ForecastResult",
    "CallCenterEmulator",
    "EmulatorConfig",
    "EmulatorMetrics",
    "SupplyOptimizer",
    "OptimizationConstraints",
    "OptimalSupply",
    "Pipeline",
    "PipelineResult",
    "HybridForecaster",
]
