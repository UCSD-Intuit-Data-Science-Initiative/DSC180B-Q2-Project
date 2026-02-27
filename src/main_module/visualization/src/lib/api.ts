const API_BASE = 'http://localhost:8000';

export interface DayMetrics {
  date: string;
  total_calls: number;
  peak_agents: number;
  avg_sla_compliance: number;
  avg_wait_time: number;
  feasible_intervals: number;
  total_intervals: number;
  model_ready: boolean;
}

export function formatDate(date: Date): string {
  return date.toISOString().split('T')[0]; // "YYYY-MM-DD"
}

export async function fetchMetrics(date: Date): Promise<DayMetrics> {
  const res = await fetch(`${API_BASE}/api/metrics?date=${formatDate(date)}`);
  if (!res.ok) throw new Error(`metrics fetch failed: ${res.status}`);
  return res.json();
}
