const API_BASE = import.meta.env.VITE_API_URL || '';

export interface DayMetrics {
  date: string;
  total_calls: number;
  peak_agents: number;
  avg_sla_compliance: number;
  avg_wait_time: number;
  avg_occupancy: number;
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

export interface WeeklyForecastDay {
  date: string;        // "YYYY-MM-DD"
  day_label: string;   // "Mon 3/3"
  total_calls: number;
  range: number;       // ± historical std dev, used as error bar
}

export async function fetchWeeklyForecast(weekStart: Date): Promise<WeeklyForecastDay[]> {
  const res = await fetch(`${API_BASE}/api/weekly-forecast?week_start=${formatDate(weekStart)}`);
  if (!res.ok) throw new Error(`weekly-forecast fetch failed: ${res.status}`);
  return res.json();
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
