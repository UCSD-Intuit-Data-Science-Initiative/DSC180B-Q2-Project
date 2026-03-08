"""Supply Optimizer - Finds minimum staffing meeting business constraints."""

from dataclasses import dataclass
from typing import Optional

from .call_center_emulator import CallCenterEmulator, EmulatorMetrics


@dataclass
class OptimizationConstraints:
    """Business constraints for staffing optimization.

    Attributes:
        min_sla: Minimum Service Level Agreement (0-1), default 0.80
        max_wait_time: Maximum acceptable wait time in seconds, default 60
        max_occupancy: Maximum agent occupancy (0-1), default 0.85
    """

    min_sla: float = 0.80
    max_wait_time: float = 60.0
    max_occupancy: float = 0.85


@dataclass
class OptimalSupply:
    """Result of optimization: minimum staffing meeting constraints.

    Attributes:
        headcount: Minimum number of agents needed (integer)
        predicted_metrics: EmulatorMetrics at optimal headcount
        is_feasible: Whether optimal solution satisfies all constraints
    """

    headcount: int
    predicted_metrics: EmulatorMetrics
    is_feasible: bool


class SupplyOptimizer:
    """Finds minimum staffing level satisfying business constraints.

    Uses a digital twin (CallCenterEmulator) to evaluate staffing levels
    and searches for the minimum headcount that satisfies all constraints.

    Since queueing metrics improve monotonically with staffing, a linear
    search from 1 upward finds the optimal (minimum feasible) solution.

    Example:
        >>> emulator = CallCenterEmulator()
        >>> optimizer = SupplyOptimizer(emulator)
        >>> cons = OptimizationConstraints(min_sla=0.80, max_wait_time=60)
        >>> result = optimizer.optimize(demand=150, constraints=cons)
        >>> print(f"Need {result.headcount} agents")
    """

    def __init__(
        self, emulator: CallCenterEmulator, max_supply: int = 500
    ) -> None:
        """Initialize optimizer with a call center emulator.

        Args:
            emulator: CallCenterEmulator instance for predicting metrics.
            max_supply: Upper bound on staffing search (safety limit).
        """
        self.emulator = emulator
        self.max_supply = max_supply

    def _meets_constraints(
        self, metrics: EmulatorMetrics, constraints: OptimizationConstraints
    ) -> bool:
        """Check whether predicted metrics satisfy all business constraints.

        Args:
            metrics: Predicted performance from the emulator.
            constraints: Business targets (SLA, wait time, occupancy).

        Returns:
            True if all constraints are met.
        """
        # Metrics use percentages (0-100); constraints use fractions (0-1)
        return (
            metrics.sla_compliance >= constraints.min_sla * 100
            and metrics.avg_wait_time <= constraints.max_wait_time
            and metrics.utilization_rate <= constraints.max_occupancy * 100
        )

    def optimize(
        self,
        demand: int,
        constraints: OptimizationConstraints,
        min_agents: int = 1,
        avg_handle_time: Optional[float] = None,
    ) -> OptimalSupply:
        """Find minimum staffing meeting all constraints.

        Searches staffing levels from min_agents to max_supply, calling the
        emulator for each level. Returns the first (minimum) headcount that
        satisfies all constraints. Monotonicity guarantees this is optimal.

        Args:
            demand: Expected number of arriving calls (integer).
            constraints: SLA/wait/occupancy targets.
            min_agents: Start the search here instead of 1.
            avg_handle_time: Override the emulator's default AHT (seconds).

        Returns:
            OptimalSupply with minimum headcount and predicted metrics.
            is_feasible=False if no solution found within max_supply.
        """
        if demand <= 0:
            metrics = self.emulator.simulate_interval(0, 0, avg_handle_time)
            return OptimalSupply(
                headcount=0, predicted_metrics=metrics, is_feasible=True
            )

        # Binary search: metrics improve monotonically with agent count,
        # so feasible region is [first_feasible, max_supply]. O(log n).
        lo = max(1, min_agents)
        hi = self.max_supply
        result_supply: int = self.max_supply
        result_metrics: EmulatorMetrics | None = None

        while lo <= hi:
            mid = (lo + hi) // 2
            metrics = self.emulator.simulate_interval(
                mid, demand, avg_handle_time
            )
            if self._meets_constraints(metrics, constraints):
                result_supply = mid
                result_metrics = metrics
                hi = mid - 1  # try fewer agents
            else:
                lo = mid + 1  # need more agents

        if result_metrics is not None:
            return OptimalSupply(
                headcount=result_supply,
                predicted_metrics=result_metrics,
                is_feasible=True,
            )

        # No feasible solution within max_supply
        last_metrics = self.emulator.simulate_interval(
            self.max_supply, demand, avg_handle_time
        )
        return OptimalSupply(
            headcount=self.max_supply,
            predicted_metrics=last_metrics,
            is_feasible=False,
        )
