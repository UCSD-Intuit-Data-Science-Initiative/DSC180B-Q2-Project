"""
Performance Emulator Module (Digital Twin)
==========================================

Predicts system performance metrics given demand and supply inputs.
Uses Erlang-C queueing theory as a rule-based emulator (Phase 1).

Mathematical Context:
    Erlang-C: P(wait) = f(arrival_rate, service_rate, num_agents)
    Given P(wait), we derive:
        - Average wait time
        - Occupancy (traffic intensity / num_agents)
        - SLA compliance (probability of answering within threshold)
"""

import math
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for predicted system performance.
    
    Attributes:
        wait_time: Average wait time in seconds before customer connects.
        occupancy: Agent utilization rate (0.0 to 1.0).
        sla_compliance: Probability of answering within SLA threshold (0.0 to 1.0).
    """
    wait_time: float
    occupancy: float
    sla_compliance: float


class PerformanceEmulator:
    """Digital Twin that predicts system metrics from demand and supply.
    
    Phase 1 implementation uses Erlang-C queueing equations:
        (demand, supply) -> (wait_time, occupancy, sla_compliance)
    
    The SupplyOptimizer calls this repeatedly to evaluate candidate
    staffing levels and find the minimum that satisfies constraints.
    
    Data Flow:
        demand (int) + supply (int) -> predict_metrics() -> PerformanceMetrics
    """
    
    def __init__(
        self, 
        avg_handle_time: float = 300.0,
        sla_threshold: float = 60.0,
        interval_minutes: float = 30.0
    ) -> None:
        """Initialize the emulator.
        
        Args:
            avg_handle_time: Average call duration in seconds.
            sla_threshold: Target wait time threshold in seconds for SLA calc.
            interval_minutes: Length of each time interval in minutes.
        """
        self.avg_handle_time = avg_handle_time
        self.sla_threshold = sla_threshold
        self.interval_minutes = interval_minutes

    def _erlang_c(self, traffic_intensity: float, num_agents: int) -> float:
        """Compute Erlang-C probability: P(wait > 0).
        
        This is the probability an arriving customer must wait because
        all agents are busy.
        
        Args:
            traffic_intensity: A = arrival_rate * avg_handle_time (in Erlangs).
            num_agents: Number of agents (N). Must satisfy N > A.
            
        Returns:
            Probability of waiting (0 to 1).
        """
        A = traffic_intensity
        N = num_agents

        if N <= A:
            return 1.0  # System is overloaded, everyone waits

        # Compute (A^N / N!) * (N / (N - A))
        # Use log-space to avoid overflow for large N
        log_numerator = N * math.log(A) - math.lgamma(N + 1)
        factor = N / (N - A)

        # Sum of (A^k / k!) for k = 0..N-1
        terms = []
        for k in range(N):
            terms.append(k * math.log(A) - math.lgamma(k + 1))
        max_term = max(terms) if terms else 0.0
        log_sum = max_term + math.log(sum(math.exp(t - max_term) for t in terms))

        # P(wait) = numerator * factor / (numerator * factor + sum)
        log_nf = log_numerator + math.log(factor)
        
        # Use log-sum-exp for numerical stability
        max_val = max(log_nf, log_sum)
        denominator = math.exp(log_nf - max_val) + math.exp(log_sum - max_val)
        
        return math.exp(log_nf - max_val) / denominator

    def predict_metrics(
        self, 
        demand: int, 
        supply: int
    ) -> PerformanceMetrics:
        """Predict performance metrics for a given demand and supply.
        
        Uses Erlang-C queueing model to compute:
            - wait_time = P(wait) * avg_handle_time / (N - A)
            - occupancy = A / N
            - sla_compliance = 1 - P(wait) * exp(-(N - A) * threshold / AHT)
        
        Args:
            demand: Number of expected arrivals in the time interval.
            supply: Number of agents staffed during the interval.
        
        Returns:
            PerformanceMetrics with predicted wait_time, occupancy, sla_compliance.
        """
        # Edge cases
        if demand <= 0:
            return PerformanceMetrics(wait_time=0.0, occupancy=0.0, sla_compliance=1.0)
        if supply <= 0:
            return PerformanceMetrics(wait_time=float("inf"), occupancy=1.0, sla_compliance=0.0)

        # Convert demand (arrivals per interval) to arrival rate (per second)
        interval_seconds = self.interval_minutes * 60.0
        arrival_rate = demand / interval_seconds
        service_rate = 1.0 / self.avg_handle_time  # calls per second per agent

        # Traffic intensity in Erlangs: A = arrival_rate / service_rate
        A = arrival_rate / service_rate  # = demand * avg_handle_time / interval_seconds
        N = supply

        # Occupancy (utilization) = A / N
        occupancy = min(A / N, 1.0)

        # If system is at or over capacity, return worst-case metrics
        if N <= A:
            return PerformanceMetrics(
                wait_time=float("inf"),
                occupancy=1.0,
                sla_compliance=0.0,
            )

        # Erlang-C: probability of waiting
        p_wait = self._erlang_c(A, N)

        # Average wait time: E[W] = P(wait) * (1 / (N * service_rate - arrival_rate))
        avg_wait = p_wait * self.avg_handle_time / (N - A)

        # SLA compliance: P(wait <= threshold)
        # = 1 - P(wait) * exp(-(N - A) * threshold / avg_handle_time)
        sla = 1.0 - p_wait * math.exp(
            -(N - A) * self.sla_threshold / self.avg_handle_time
        )

        return PerformanceMetrics(
            wait_time=max(avg_wait, 0.0),
            occupancy=occupancy,
            sla_compliance=max(min(sla, 1.0), 0.0),
        )
