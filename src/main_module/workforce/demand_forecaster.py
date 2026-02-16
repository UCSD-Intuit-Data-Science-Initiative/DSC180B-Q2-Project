"""
Demand Forecaster Module
========================

Forecasts incoming call demand based on historical arrival data.
Phase 1 uses historical averages (by day-of-week + time-of-day).
Phase 2 can upgrade to ARIMA, Prophet, or ML models.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ForecastResult:
    """Container for demand forecast output.
    
    Attributes:
        timestamps: Time intervals for the forecast (e.g., 30-min buckets).
        demand: Predicted number of arrivals for each time interval.
    """
    timestamps: pd.DatetimeIndex
    demand: np.ndarray


class DemandForecaster:
    """Forecasts incoming call demand from historical data.
    
    Phase 1 approach: Computes average arrivals per (day_of_week, time_bucket)
    from historical data. Forecasts use these averages as expected demand.
    
    Data Flow:
        historical_data (DataFrame) -> fit() -> internal averages table
        start_time + horizon -> forecast() -> ForecastResult
    """
    
    def __init__(self, resolution_minutes: int = 30) -> None:
        """Initialize the forecaster.
        
        Args:
            resolution_minutes: Time bucket size in minutes for aggregation.
        """
        self.resolution_minutes = resolution_minutes
        self._averages: pd.DataFrame | None = None  # Populated by fit()
        self._is_fitted: bool = False
    
    def fit(self, historical_data: pd.DataFrame) -> "DemandForecaster":
        """Train the forecaster on historical arrival data.
        
        Aggregates arrivals into time buckets, then computes average arrivals
        per (day_of_week, time_bucket) combination.
        
        Args:
            historical_data: DataFrame with an 'Arrival time' column
                containing datetime strings/objects of each customer arrival.
        
        Returns:
            self: The fitted forecaster instance (for method chaining).
        """
        # Parse arrival times
        arrivals = pd.to_datetime(historical_data["Arrival time"])

        # Floor to time buckets (e.g., 30-min intervals)
        freq = f"{self.resolution_minutes}min"
        bucketed = arrivals.dt.floor(freq)

        # Count arrivals per bucket
        counts = bucketed.value_counts().reset_index()
        counts.columns = ["bucket", "arrivals"]
        counts["bucket"] = pd.to_datetime(counts["bucket"])

        # Build a complete time index (fill missing buckets with 0)
        full_range = pd.date_range(
            start=counts["bucket"].min(),
            end=counts["bucket"].max(),
            freq=freq,
        )
        counts = counts.set_index("bucket").reindex(full_range, fill_value=0)
        counts = counts.rename_axis("bucket").reset_index()

        # Extract features: day_of_week (0=Mon) and time string (e.g., "09:30")
        counts["day_of_week"] = counts["bucket"].dt.dayofweek
        counts["time"] = counts["bucket"].dt.strftime("%H:%M")

        # Compute average arrivals per (day_of_week, time)
        self._averages = (
            counts.groupby(["day_of_week", "time"])["arrivals"]
            .mean()
            .reset_index()
            .rename(columns={"arrivals": "avg_demand"})
        )

        self._is_fitted = True
        return self
    
    def forecast(
        self, 
        start_time: pd.Timestamp, 
        horizon: int = 16
    ) -> ForecastResult:
        """Generate demand forecast for future time intervals.
        
        Looks up historical averages for each (day_of_week, time) slot
        in the forecast window.
        
        Args:
            start_time: The beginning of the forecast window.
            horizon: Number of time intervals to forecast ahead.
                With 30-min resolution, horizon=16 covers 8 hours.
        
        Returns:
            ForecastResult with timestamps and predicted demand per interval.
        
        Raises:
            ValueError: If forecaster has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("Forecaster must be fitted before forecasting. Call .fit() first.")

        freq = f"{self.resolution_minutes}min"
        timestamps = pd.date_range(start=start_time, periods=horizon, freq=freq)

        demand = np.zeros(horizon, dtype=float)
        for i, ts in enumerate(timestamps):
            dow = ts.dayofweek
            time_str = ts.strftime("%H:%M")
            
            match = self._averages[
                (self._averages["day_of_week"] == dow)
                & (self._averages["time"] == time_str)
            ]
            
            if not match.empty:
                demand[i] = match["avg_demand"].values[0]
            else:
                # Fallback: overall average if no match for this slot
                demand[i] = self._averages["avg_demand"].mean()

        # Round to integers (can't have fractional arrivals)
        demand = np.round(demand).astype(int)

        return ForecastResult(timestamps=timestamps, demand=demand)
