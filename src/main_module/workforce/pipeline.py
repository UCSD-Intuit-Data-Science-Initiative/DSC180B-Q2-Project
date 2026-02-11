"""
Pipeline Module (The Manager)
=============================

Orchestrates the full workflow: Forecast -> Optimize -> Results.
Connects the DemandForecaster and SupplyOptimizer into an end-to-end system.
"""

from dataclasses import dataclass
import pandas as pd

from .demand_forecaster import DemandForecaster, ForecastResult
from .supply_optimizer import SupplyOptimizer, OptimizationConstraints, OptimalSupply


@dataclass
class PipelineResult:
    """Complete result from running the pipeline.
    
    Attributes:
        forecast: The demand forecast used.
        staffing_schedule: List of optimal staffing for each interval.
        total_headcount: Sum of headcount across all intervals.
    """
    forecast: ForecastResult
    staffing_schedule: list[OptimalSupply]
    total_headcount: int


class Pipeline:
    """Orchestrates forecasting and optimization into one workflow.
    
    Uses Dependency Injection: takes pre-configured Forecaster and Optimizer
    so the pipeline logic is decoupled from specific implementations.
    
    Workflow (run method):
        1. Call forecaster.forecast(start_time) -> ForecastResult
        2. For each demand value in the forecast:
           a. Call optimizer.optimize(demand, constraints) -> OptimalSupply
        3. Return aggregated PipelineResult
    """
    
    def __init__(
        self, 
        forecaster: DemandForecaster, 
        optimizer: SupplyOptimizer
    ) -> None:
        """Initialize the pipeline with its components.
        
        Args:
            forecaster: Fitted DemandForecaster instance for generating
                demand predictions.
            optimizer: SupplyOptimizer instance (with its emulator) for
                finding optimal staffing levels.
        """
        self.forecaster = forecaster
        self.optimizer = optimizer
    
    def run(
        self,
        start_time: pd.Timestamp,
        constraints: OptimizationConstraints,
        horizon: int = 16
    ) -> PipelineResult:
        """Execute the full forecast-to-optimization workflow.
        
        Args:
            start_time: Beginning of the planning window.
            constraints: Business constraints (min_sla, max_wait, etc.)
            horizon: Number of time intervals to plan for.
        
        Returns:
            PipelineResult with forecast, per-interval staffing, and totals.
        """
        # Step 1: Generate demand forecast
        forecast = self.forecaster.forecast(start_time, horizon)

        # Step 2: Optimize staffing for each time interval
        staffing_schedule: list[OptimalSupply] = []
        for demand_value in forecast.demand:
            result = self.optimizer.optimize(int(demand_value), constraints)
            staffing_schedule.append(result)

        # Step 3: Aggregate results
        total_headcount = sum(r.headcount for r in staffing_schedule)

        return PipelineResult(
            forecast=forecast,
            staffing_schedule=staffing_schedule,
            total_headcount=total_headcount,
        )
