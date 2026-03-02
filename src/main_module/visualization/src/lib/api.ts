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

export interface ForecastSlot {
  time: string;
  predicted_calls: number;
  model_used: string;
}

export async function fetchForecast(date: Date): Promise<ForecastSlot[]> {
  const res = await fetch(`${API_BASE}/api/forecast?date=${formatDate(date)}`);
  if (!res.ok) throw new Error(`forecast fetch failed: ${res.status}`);
  return res.json();
}

export interface StaffingSlot {
  time: string;
  predicted_calls: number;
  agents: number;
  avg_wait_time: number;
  sla_compliance: number;
  utilization_rate: number;
  abandonment_rate: number;
  is_feasible: boolean;
}

export interface DailyTotal {
  date: Date;
  totalCalls: number;
}

export async function fetchWeeklyForecast(weekStart: Date): Promise<DailyTotal[]> {
  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(weekStart);
    d.setDate(weekStart.getDate() + i);
    return d;
  });

  return Promise.all(
    days.map(async (day) => {
      try {
        const slots = await fetchForecast(day);
        const total = slots.reduce((sum, s) => sum + s.predicted_calls, 0);
        return { date: day, totalCalls: total };
      } catch {
        return { date: day, totalCalls: 0 };
      }
    })
  );
}

export async function fetchStaffing(
  date: Date,
  minSla: number,
  maxWait: number,
  maxOccupancy: number
): Promise<StaffingSlot[]> {
  const params = new URLSearchParams({
    date: formatDate(date),
    min_sla: (minSla / 100).toString(),
    max_wait: maxWait.toString(),
    max_occupancy: (maxOccupancy / 100).toString(),
  });
  const res = await fetch(`${API_BASE}/api/staffing?${params}`);
  if (!res.ok) throw new Error(`staffing fetch failed: ${res.status}`);
  return res.json();
}
