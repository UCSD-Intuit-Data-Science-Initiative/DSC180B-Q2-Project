const API_BASE = import.meta.env.VITE_API_URL || '';
const FETCH_TIMEOUT_MS = 30000;
const FETCH_RETRIES = 2;

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout = FETCH_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return res;
  } finally {
    clearTimeout(id);
  }
}

async function fetchWithRetry(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  let lastErr: Error | null = null;
  for (let i = 0; i <= FETCH_RETRIES; i++) {
    try {
      const res = await fetchWithTimeout(url, options);
      return res;
    } catch (err) {
      lastErr = err instanceof Error ? err : new Error(String(err));
      if (i < FETCH_RETRIES) await new Promise((r) => setTimeout(r, 1000 * (i + 1)));
    }
  }
  throw lastErr ?? new Error('Fetch failed');
}

export async function fetchApi(path: string, timeoutMs?: number): Promise<Response> {
  const url = path.startsWith('http') ? path : `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;
  const timeout = timeoutMs ?? FETCH_TIMEOUT_MS;
  let lastErr: Error | null = null;
  for (let i = 0; i <= FETCH_RETRIES; i++) {
    try {
      return await fetchWithTimeout(url, {}, timeout);
    } catch (err) {
      lastErr = err instanceof Error ? err : new Error(String(err));
      if (i < FETCH_RETRIES) await new Promise((r) => setTimeout(r, 1000 * (i + 1)));
    }
  }
  throw lastErr ?? new Error('Fetch failed');
}

const SCHEDULE_TIMEOUT_MS = 90000;

export async function fetchScheduleApi(path: string): Promise<Response> {
  const url = path.startsWith('http') ? path : `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;
  return fetchWithTimeout(url, {}, SCHEDULE_TIMEOUT_MS);
}

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
  const res = await fetchWithRetry(`${API_BASE}/api/metrics?date=${formatDate(date)}`);
  if (!res.ok) throw new Error(`metrics fetch failed: ${res.status}`);
  return res.json();
}

export interface ForecastSlot {
  time: string;
  predicted_calls: number;
  model_used: string;
}

export async function fetchForecast(date: Date): Promise<ForecastSlot[]> {
  const res = await fetchWithRetry(`${API_BASE}/api/forecast?date=${formatDate(date)}`);
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
  const res = await fetchWithRetry(`${API_BASE}/api/weekly-forecast?week_start=${formatDate(weekStart)}`);
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
  const res = await fetchWithRetry(`${API_BASE}/api/staffing?${params}`);
  if (!res.ok) throw new Error(`staffing fetch failed: ${res.status}`);
  return res.json();
}
