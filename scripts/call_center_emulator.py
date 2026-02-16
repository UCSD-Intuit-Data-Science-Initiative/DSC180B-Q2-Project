from dataclasses import dataclass

import numpy as np


@dataclass
class EmulatorConfig:
    avg_handle_time: float = 600
    sla_threshold_seconds: float = 60
    sla_target_percent: float = 80
    max_wait_time_seconds: float = 300
    avg_patience_time: float = 180
    interval_duration_seconds: float = 1800


@dataclass
class EmulatorMetrics:
    avg_wait_time: float
    max_wait_time: float
    utilization_rate: float
    sla_compliance: float
    abandonment_rate: float
    service_level: float
    calls_handled: int
    calls_abandoned: int

    def meets_thresholds(
        self,
        max_avg_wait: float = 60,
        min_sla: float = 80,
        max_utilization: float = 85,
        max_abandonment: float = 5,
    ) -> bool:
        return (
            self.avg_wait_time <= max_avg_wait
            and self.sla_compliance >= min_sla
            and self.utilization_rate <= max_utilization
            and self.abandonment_rate <= max_abandonment
        )

    def __str__(self):
        return (
            f"Avg Wait: {self.avg_wait_time:.1f}s | "
            f"SLA: {self.sla_compliance:.1f}% | "
            f"Utilization: {self.utilization_rate:.1f}% | "
            f"Abandonment: {self.abandonment_rate:.1f}%"
        )


class CallCenterEmulator:
    def __init__(self, config: EmulatorConfig = None, model: str = "erlang_a"):
        self.config = config or EmulatorConfig()
        self.model = model

    def _log_factorial(self, n: int) -> float:
        if n <= 1:
            return 0.0
        return sum(np.log(i) for i in range(1, n + 1))

    def _erlang_b(self, num_agents: int, traffic_intensity: float) -> float:
        if num_agents <= 0:
            return 1.0
        if traffic_intensity <= 0:
            return 0.0

        log_numerator = num_agents * np.log(
            traffic_intensity
        ) - self._log_factorial(num_agents)

        log_terms = []
        for k in range(num_agents + 1):
            log_term = k * np.log(traffic_intensity) - self._log_factorial(k)
            log_terms.append(log_term)

        max_log = max(log_terms)
        log_denominator = max_log + np.log(
            sum(np.exp(lt - max_log) for lt in log_terms)
        )

        result = np.exp(log_numerator - log_denominator)
        return max(0, min(1, result))

    def _erlang_c(self, num_agents: int, traffic_intensity: float) -> float:
        if num_agents <= 0 or traffic_intensity <= 0:
            return 1.0

        if traffic_intensity >= num_agents:
            return 1.0

        rho = traffic_intensity / num_agents

        if rho >= 0.99:
            return 0.99

        try:
            erlang_b = self._erlang_b(num_agents, traffic_intensity)
            prob_wait = erlang_b / (1 - rho * (1 - erlang_b))
            return max(0, min(1, prob_wait))

        except (OverflowError, ValueError, ZeroDivisionError):
            return min(0.99, rho)

    def _erlang_a_metrics(
        self,
        num_agents: int,
        arrival_rate: float,
        service_rate: float,
        abandonment_rate: float,
    ) -> dict:
        if num_agents <= 0 or arrival_rate <= 0:
            return {
                "prob_wait": 0,
                "prob_abandon": 0,
                "avg_wait_served": 0,
                "avg_wait_abandoned": 0,
                "effective_arrival_rate": arrival_rate,
            }

        traffic_intensity = arrival_rate / service_rate
        theta = abandonment_rate / service_rate

        rho = traffic_intensity / num_agents

        if rho < 0.01:
            return {
                "prob_wait": 0,
                "prob_abandon": 0,
                "avg_wait_served": 0,
                "avg_wait_abandoned": 0,
                "effective_arrival_rate": arrival_rate,
            }

        if rho >= 1.0:
            prob_wait = self._erlang_c(num_agents, traffic_intensity * 0.99)

            excess_rate = arrival_rate - num_agents * service_rate
            prob_abandon = min(0.9, excess_rate / arrival_rate + 0.1)

            avg_wait_served = 1 / (
                num_agents * service_rate * 0.01 + abandonment_rate
            )
            avg_wait_abandoned = 1 / abandonment_rate

            return {
                "prob_wait": prob_wait,
                "prob_abandon": prob_abandon,
                "avg_wait_served": avg_wait_served,
                "avg_wait_abandoned": avg_wait_abandoned,
                "effective_arrival_rate": arrival_rate * (1 - prob_abandon),
            }

        prob_wait_c = self._erlang_c(num_agents, traffic_intensity)

        alpha = num_agents * (1 - rho) + theta * traffic_intensity

        if alpha > 0:
            prob_abandon = prob_wait_c * (theta * traffic_intensity) / alpha
        else:
            prob_abandon = prob_wait_c * 0.5

        prob_abandon = max(0, min(1, prob_abandon))

        denominator = num_agents * service_rate * (1 - rho) + abandonment_rate
        if denominator > 0:
            avg_wait_served = prob_wait_c / denominator
        else:
            avg_wait_served = prob_wait_c / (abandonment_rate + 0.001)

        avg_wait_abandoned = (
            1 / abandonment_rate if abandonment_rate > 0 else float("inf")
        )

        effective_arrival = arrival_rate * (1 - prob_abandon)

        return {
            "prob_wait": prob_wait_c,
            "prob_abandon": prob_abandon,
            "avg_wait_served": avg_wait_served,
            "avg_wait_abandoned": avg_wait_abandoned,
            "effective_arrival_rate": effective_arrival,
        }

    def _calculate_avg_wait_time(
        self, num_agents: int, arrival_rate: float, service_rate: float
    ) -> float:
        if num_agents <= 0 or arrival_rate <= 0:
            return 0.0

        traffic_intensity = arrival_rate / service_rate

        if traffic_intensity >= num_agents:
            return float("inf")

        prob_wait = self._erlang_c(num_agents, traffic_intensity)
        rho = traffic_intensity / num_agents

        avg_wait = prob_wait / (num_agents * service_rate * (1 - rho))

        return max(0, avg_wait)

    def _calculate_service_level(
        self,
        num_agents: int,
        arrival_rate: float,
        service_rate: float,
        target_time: float,
    ) -> float:
        if num_agents <= 0:
            return 0.0
        if arrival_rate <= 0:
            return 100.0

        traffic_intensity = arrival_rate / service_rate

        if traffic_intensity >= num_agents:
            return 0.0

        prob_wait = self._erlang_c(num_agents, traffic_intensity)
        traffic_intensity / num_agents

        exponent = (
            -(num_agents - traffic_intensity) * service_rate * target_time
        )
        service_level = 1 - prob_wait * np.exp(exponent)

        return max(0, min(100, service_level * 100))

    def _calculate_service_level_erlang_a(
        self,
        num_agents: int,
        arrival_rate: float,
        service_rate: float,
        abandonment_rate: float,
        target_time: float,
    ) -> float:
        if num_agents <= 0:
            return 0.0
        if arrival_rate <= 0:
            return 100.0

        ea_metrics = self._erlang_a_metrics(
            num_agents, arrival_rate, service_rate, abandonment_rate
        )

        prob_wait = ea_metrics["prob_wait"]
        prob_abandon = ea_metrics["prob_abandon"]

        traffic_intensity = arrival_rate / service_rate
        rho = min(traffic_intensity / num_agents, 0.99)

        decay_rate = num_agents * service_rate * (1 - rho) + abandonment_rate
        prob_wait_exceeds_target = prob_wait * np.exp(
            -decay_rate * target_time
        )

        service_level = 1 - prob_wait_exceeds_target - prob_abandon

        return max(0, min(100, service_level * 100))

    def simulate_interval(
        self,
        num_experts: int,
        incoming_calls: int,
        avg_handle_time: float = None,
    ) -> EmulatorMetrics:
        if avg_handle_time is None:
            avg_handle_time = self.config.avg_handle_time

        if incoming_calls == 0:
            return EmulatorMetrics(
                avg_wait_time=0,
                max_wait_time=0,
                utilization_rate=0,
                sla_compliance=100,
                abandonment_rate=0,
                service_level=100,
                calls_handled=0,
                calls_abandoned=0,
            )

        if num_experts <= 0:
            return EmulatorMetrics(
                avg_wait_time=float("inf"),
                max_wait_time=float("inf"),
                utilization_rate=100,
                sla_compliance=0,
                abandonment_rate=100,
                service_level=0,
                calls_handled=0,
                calls_abandoned=incoming_calls,
            )

        interval_seconds = self.config.interval_duration_seconds
        arrival_rate = incoming_calls / interval_seconds
        service_rate = 1 / avg_handle_time
        traffic_intensity = arrival_rate / service_rate
        abandonment_rate_param = 1 / self.config.avg_patience_time

        if self.model == "erlang_a":
            ea_metrics = self._erlang_a_metrics(
                num_experts, arrival_rate, service_rate, abandonment_rate_param
            )

            avg_wait_time = ea_metrics["avg_wait_served"]
            abandonment_rate = ea_metrics["prob_abandon"] * 100

            sla_compliance = self._calculate_service_level_erlang_a(
                num_experts,
                arrival_rate,
                service_rate,
                abandonment_rate_param,
                self.config.sla_threshold_seconds,
            )
        else:
            avg_wait_time = self._calculate_avg_wait_time(
                num_experts, arrival_rate, service_rate
            )

            sla_compliance = self._calculate_service_level(
                num_experts,
                arrival_rate,
                service_rate,
                self.config.sla_threshold_seconds,
            )

            if avg_wait_time == float("inf"):
                abandonment_rate = 50.0
            else:
                abandonment_prob = 1 - np.exp(
                    -avg_wait_time / self.config.max_wait_time_seconds
                )
                abandonment_rate = min(100, abandonment_prob * 100)

        utilization_rate = (
            min(100, (traffic_intensity / num_experts) * 100)
            if num_experts > 0
            else 100
        )

        if avg_wait_time == float("inf"):
            max_wait_time = self.config.max_wait_time_seconds * 2
        else:
            max_wait_time = avg_wait_time * 3

        calls_abandoned = int(incoming_calls * (abandonment_rate / 100))
        calls_handled = incoming_calls - calls_abandoned

        service_level = sla_compliance

        return EmulatorMetrics(
            avg_wait_time=avg_wait_time,
            max_wait_time=max_wait_time,
            utilization_rate=utilization_rate,
            sla_compliance=sla_compliance,
            abandonment_rate=abandonment_rate,
            service_level=service_level,
            calls_handled=calls_handled,
            calls_abandoned=calls_abandoned,
        )

    def simulate_day(
        self,
        num_experts_per_interval: list,
        calls_per_interval: list,
        avg_handle_time: float = None,
    ) -> dict:
        if len(num_experts_per_interval) != len(calls_per_interval):
            raise ValueError(
                "Expert count and call count lists must have same length"
            )

        interval_metrics = []
        total_calls = 0
        total_handled = 0
        total_abandoned = 0
        weighted_wait = 0
        weighted_sla = 0
        weighted_util = 0

        for experts, calls in zip(
            num_experts_per_interval, calls_per_interval
        ):
            metrics = self.simulate_interval(experts, calls, avg_handle_time)
            interval_metrics.append(metrics)

            total_calls += calls
            total_handled += metrics.calls_handled
            total_abandoned += metrics.calls_abandoned

            if calls > 0:
                weighted_wait += metrics.avg_wait_time * calls
                weighted_sla += metrics.sla_compliance * calls
                weighted_util += metrics.utilization_rate * calls

        if total_calls > 0:
            avg_wait = weighted_wait / total_calls
            avg_sla = weighted_sla / total_calls
            avg_util = weighted_util / total_calls
            abandonment = (total_abandoned / total_calls) * 100
        else:
            avg_wait = 0
            avg_sla = 100
            avg_util = 0
            abandonment = 0

        return {
            "interval_metrics": interval_metrics,
            "summary": {
                "total_calls": total_calls,
                "total_handled": total_handled,
                "total_abandoned": total_abandoned,
                "avg_wait_time": avg_wait,
                "avg_sla_compliance": avg_sla,
                "avg_utilization": avg_util,
                "abandonment_rate": abandonment,
                "total_experts_scheduled": sum(num_experts_per_interval),
            },
        }


def main():
    print("=" * 70)
    print("CALL CENTER EMULATOR COMPARISON: ERLANG-C vs ERLANG-A")
    print("=" * 70)

    config = EmulatorConfig(
        avg_handle_time=600, sla_threshold_seconds=60, avg_patience_time=180
    )

    emulator_c = CallCenterEmulator(config, model="erlang_c")
    emulator_a = CallCenterEmulator(config, model="erlang_a")

    incoming_calls = 15
    print("\nTest Parameters:")
    print(f"  Incoming calls: {incoming_calls}")
    print(f"  Avg handle time: {config.avg_handle_time}s")
    print(f"  SLA threshold: {config.sla_threshold_seconds}s")
    print(f"  Avg patience (Erlang-A): {config.avg_patience_time}s")

    print("\n" + "-" * 70)
    print("ERLANG-C (assumes infinite patience - no abandonment modeling)")
    print("-" * 70)
    print(
        f"{'Experts':<10} {'Avg Wait':<12} {'SLA%':<10} {'Util%':<10} {'Abandon%':<10}"
    )

    for num_experts in [3, 4, 5, 6, 7, 8]:
        metrics = emulator_c.simulate_interval(num_experts, incoming_calls)
        print(
            f"{num_experts:<10} {metrics.avg_wait_time:<12.1f} {metrics.sla_compliance:<10.1f} "
            f"{metrics.utilization_rate:<10.1f} {metrics.abandonment_rate:<10.1f}"
        )

    print("\n" + "-" * 70)
    print("ERLANG-A (models customer abandonment realistically)")
    print("-" * 70)
    print(
        f"{'Experts':<10} {'Avg Wait':<12} {'SLA%':<10} {'Util%':<10} {'Abandon%':<10}"
    )

    for num_experts in [3, 4, 5, 6, 7, 8]:
        metrics = emulator_a.simulate_interval(num_experts, incoming_calls)
        print(
            f"{num_experts:<10} {metrics.avg_wait_time:<12.1f} {metrics.sla_compliance:<10.1f} "
            f"{metrics.utilization_rate:<10.1f} {metrics.abandonment_rate:<10.1f}"
        )

    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    print("""
Erlang-C:
  - Assumes callers wait forever (unrealistic)
  - Abandonment is estimated post-hoc
  - Tends to OVERESTIMATE required staffing

Erlang-A:
  - Models customer patience explicitly
  - Abandonment affects queue dynamics
  - More realistic for production call centers
  - Industry standard at companies like Intuit
""")

    print("=" * 70)
    print("FULL DAY SIMULATION (Erlang-A)")
    print("=" * 70)

    calls_per_interval = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        5,
        8,
        10,
        12,
        15,
        18,
        20,
        18,
        15,
        12,
        10,
        8,
        6,
        5,
        4,
        3,
        2,
        2,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    experts_per_interval = [max(1, c // 3 + 1) for c in calls_per_interval]

    results = emulator_a.simulate_day(experts_per_interval, calls_per_interval)

    print("\nDay Summary:")
    print(f"  Total calls: {results['summary']['total_calls']}")
    print(f"  Calls handled: {results['summary']['total_handled']}")
    print(f"  Calls abandoned: {results['summary']['total_abandoned']}")
    print(f"  Avg wait time: {results['summary']['avg_wait_time']:.1f}s")
    print(
        f"  Avg SLA compliance: {results['summary']['avg_sla_compliance']:.1f}%"
    )
    print(f"  Avg utilization: {results['summary']['avg_utilization']:.1f}%")
    print(f"  Abandonment rate: {results['summary']['abandonment_rate']:.1f}%")
    print(
        f"  Total expert-intervals: {results['summary']['total_experts_scheduled']}"
    )


if __name__ == "__main__":
    main()
