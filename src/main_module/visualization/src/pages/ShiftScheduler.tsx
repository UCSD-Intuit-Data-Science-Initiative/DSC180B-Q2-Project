import React, { useMemo, useState, useEffect, useCallback } from 'react';
import { ArrowLeft, Calendar, ChevronLeft, ChevronRight, Loader2, Plus, Users } from 'lucide-react';
import { Link } from 'react-router';
import { useTheme } from '../context/ThemeContext';
import { ThemeToggle } from '../components/ThemeToggle';
import { ComposedChart, BarChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';
import { fetchScheduleApi, fetchStaffing, formatDate } from '../lib/api';

interface ScheduleAssignment {
  expert_id: string;
  slot_start: string;
  slot_end: string;
  assignment: string;
  shift_block: string;
}

interface CoverageData {
  slot_start: string;
  agents_assigned: number;
  predicted_demand: number;
  coverage_ratio: number;
}

interface ShiftSummary {
  expert_id: string;
  shift_block: string;
  shift_start: string;
  shift_end: string;
  total_slots: number;
  work_slots: number;
  break_slots: number;
  shift_hours: number;
  work_hours: number;
}

interface ScheduleData {
  date: string;
  assignments: ScheduleAssignment[];
  coverage: CoverageData[];
  summary: ShiftSummary[];
}

interface AgentAvailability {
  id: string;
  name: string;
  available: boolean;
  startHour: number;
  endHour: number;
}

function seededRandom(seed: number) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function parseSlotAsUtc(iso: string) {
  if (iso.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(iso)) return new Date(iso);
  return new Date(iso + 'Z');
}

function generateRandomAvailability(dateStr: string, agentIds: string[]): AgentAvailability[] {
  const seed = dateStr.split('-').reduce((a, b) => a + parseInt(b, 10), 0);
  return agentIds.map((id, i) => {
    const r = seededRandom(seed + i * 7);
    const available = r > 0.4;
    const displayId = String(id).replace(/\.0$/, '');
    const displayName = displayId.length > 4 ? `Agent ${displayId.slice(-4)}` : `Agent ${displayId}`;
    if (!available) {
      return { id, name: displayName, available: false, startHour: 0, endHour: 0 };
    }
    const startHour = 5 + Math.floor(seededRandom(seed + i * 11) * 9);
    const endHour = Math.min(startHour + 8, 17);
    return { id, name: displayName, available: true, startHour, endHour };
  });
}

export default function ShiftScheduler() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [selectedWeekOffset, setSelectedWeekOffset] = useState(() => {
    const today = new Date();
    const day = today.getDay();
    return (day === 0 || day === 6) ? 1 : 0;
  });
  const [selectedDay, setSelectedDay] = useState(0);

  const [editMode, setEditMode] = useState(false);
  const [shiftModalAgent, setShiftModalAgent] = useState<AgentAvailability | null>(null);
  const [scheduleData, setScheduleData] = useState<ScheduleData | null>(null);
  const [staffingData, setStaffingData] = useState<Array<{ time: string; calls: number; agents: number }>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loadingStale, setLoadingStale] = useState(false);
  const [manualAssignments, setManualAssignments] = useState<Array<{ agentId: string; agentName: string; startHour: number; startMin: number; endHour: number; endMin: number }>>([]);

  useEffect(() => {
    if (!loading) return;
    const t = setTimeout(() => setLoadingStale(true), 15000);
    return () => clearTimeout(t);
  }, [loading]);

  const weekDates = useMemo(() => {
    const today = new Date();
    const monday = new Date(today);
    const day = monday.getDay();
    const diff = monday.getDate() - day + (day === 0 ? -6 : 1);
    monday.setDate(diff + selectedWeekOffset * 7);
    monday.setHours(0, 0, 0, 0);
    const dates = [];
    for (let i = 0; i < 7; i++) {
      const date = new Date(monday);
      date.setDate(monday.getDate() + i);
      dates.push(date);
    }
    return dates;
  }, [selectedWeekOffset]);

  const effectiveDay = selectedDay >= 5 ? 0 : selectedDay;
  const selectedDateStr = useMemo(() => weekDates[effectiveDay].toISOString().split('T')[0], [weekDates, effectiveDay]);
  const selectedDate = useMemo(() => weekDates[effectiveDay], [weekDates, effectiveDay]);
  const isWeekend = effectiveDay >= 5;

  const fetchScheduleData = useCallback(async () => {
    if (isWeekend) {
      setScheduleData(null);
      setStaffingData([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    setLoadingStale(false);
    let scheduleError: string | null = null;
    let staffingSlots: Array<{ time: string; predicted_calls: number; agents: number }> = [];
    try {
      staffingSlots = await fetchStaffing(selectedDate, 80, 60, 85);
      setStaffingData(staffingSlots.map(s => ({ time: s.time, calls: s.predicted_calls, agents: s.agents })));
    } catch (err) {
      setStaffingData([]);
      setError(err instanceof Error ? err.message : 'Failed to load staffing');
    }
    try {
      const scheduleRes = await fetchScheduleApi(`/api/schedule?date=${selectedDateStr}`);
      if (!scheduleRes.ok) {
        const errBody = await scheduleRes.json().catch(() => ({}));
        scheduleError = errBody.detail || `Schedule API error: ${scheduleRes.status}`;
        if (!staffingSlots.length) setError(scheduleError);
      } else {
        const sched = await scheduleRes.json();
        setScheduleData(sched);
      }
    } catch (err) {
      setScheduleData(null);
      if (!staffingSlots.length) setError(err instanceof Error ? err.message : 'Failed to load schedule');
    } finally {
      setLoading(false);
    }
  }, [selectedDateStr, selectedDate, isWeekend]);

  useEffect(() => {
    fetchScheduleData();
  }, [fetchScheduleData]);

  useEffect(() => {
    setManualAssignments([]);
  }, [selectedDateStr]);

  const requirementsChartData = useMemo(() => {
    const slots = ['05:00', '05:30', '06:00', '06:30', '07:00', '07:30', '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30'];
    const byTime = new Map<string, { recommended: number; scheduled: number }>();
    slots.forEach(t => byTime.set(t, { recommended: 0, scheduled: 0 }));
    if (staffingData.length > 0) {
      staffingData.forEach(s => {
        const e = byTime.get(s.time);
        if (e) e.recommended = Math.max(0, s.agents);
      });
    }
    scheduleData?.coverage?.forEach(c => {
      const slotTime = parseSlotAsUtc(c.slot_start);
      const h = slotTime.getUTCHours();
      const m = slotTime.getUTCMinutes();
      const localH = (h - 8 + 24) % 24;
      const timeKey = `${String(localH).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
      const entry = byTime.get(timeKey);
      if (entry) {
        entry.scheduled = c.agents_assigned;
        if (staffingData.length === 0) entry.recommended = Math.max(1, Math.ceil((c.predicted_demand || 1) / 5));
      }
    });
    manualAssignments.forEach(ma => {
      let currentMin = ma.startHour * 60 + (ma.startMin ?? 0);
      const endMin = ma.endHour * 60 + (ma.endMin ?? 0);
      while (currentMin < endMin) {
        const h = Math.floor(currentMin / 60);
        const m = currentMin % 60;
        const timeKey = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
        const entry = byTime.get(timeKey);
        if (entry) entry.scheduled += 1;
        currentMin += 30;
      }
    });
    const hasData = staffingData.length > 0 || (scheduleData?.coverage?.length ?? 0) > 0;
    if (!hasData) return [];
    return slots.map(time => {
      const entry = byTime.get(time) ?? { recommended: 0, scheduled: 0 };
      const rec = staffingData.length > 0 ? entry.recommended : (entry.recommended || 1);
      const under = entry.scheduled < rec;
      return {
        time,
        recommended: rec,
        scheduled: entry.scheduled,
        fill: under ? '#ef4444' : '#3b82f6',
      };
    });
  }, [scheduleData, staffingData, manualAssignments]);

  const barChartRaceData = useMemo(() => {
    const apiRows = (scheduleData?.summary ?? []).map(agent => {
      const start = parseSlotAsUtc(agent.shift_start);
      const end = parseSlotAsUtc(agent.shift_end);
      const startMin = start.getUTCHours() * 60 + start.getUTCMinutes();
      const endMin = end.getUTCHours() * 60 + end.getUTCMinutes();
      let duration = (endMin - startMin) / 60;
      if (duration < 0) duration += 24;
      return {
        name: `Agent ${String(agent.expert_id).replace(/\.0$/, '').slice(-4)}`,
        expert_id: String(agent.expert_id),
        startTime: start.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
        endTime: end.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
        startMin,
        duration,
        sortKey: new Date(agent.shift_start).getTime(),
      };
    });
    const manualRows = manualAssignments.map(ma => {
      const startMin = ma.startHour * 60 + (ma.startMin ?? 0);
      const endMin = ma.endHour * 60 + (ma.endMin ?? 0);
      const duration = (endMin - startMin) / 60;
      return {
        name: ma.agentName,
        expert_id: ma.agentId,
        startTime: `${String(ma.startHour).padStart(2, '0')}:${String(ma.startMin ?? 0).padStart(2, '0')}`,
        endTime: `${String(ma.endHour).padStart(2, '0')}:${String(ma.endMin ?? 0).padStart(2, '0')}`,
        startMin,
        duration,
        sortKey: startMin,
      };
    });
    const combined = [...apiRows, ...manualRows].sort((a, b) => a.sortKey - b.sortKey);
    const seen = new Set<string>();
    return combined.map((row, idx) => {
      let name = row.name;
      if (seen.has(name)) name = `${row.name} (${idx})`;
      seen.add(name);
      return { ...row, name, index: idx };
    });
  }, [scheduleData, manualAssignments]);

  const barChartDisplayLimit = 25;
  const barChartDisplayData = useMemo(() => barChartRaceData.slice(0, barChartDisplayLimit), [barChartRaceData]);
  const barChartTotalCount = barChartRaceData.length;

  const agentAvailability = useMemo(() => {
    const ids = scheduleData?.summary?.length
      ? [...new Set(scheduleData.summary.map(s => String(s.expert_id)))]
      : Array.from({ length: 21 }, (_, i) => `agent-${i + 1}`);
    return generateRandomAvailability(selectedDateStr, ids);
  }, [scheduleData, selectedDateStr]);

  const summaryMetrics = useMemo(() => {
    const available = agentAvailability.filter(a => a.available).length;
    const unavailable = agentAvailability.length - available;
    const totalSlots = staffingData.length;
    let adequate = 0;
    let understaffed = 0;
    requirementsChartData.forEach(d => {
      if (d.scheduled >= d.recommended) adequate++;
      else understaffed++;
    });
    const coveragePct = totalSlots ? Math.round((adequate / totalSlots) * 100) : 0;
    const scheduledCount = (scheduleData?.summary?.length ?? 0) + manualAssignments.length;
    const totalAgents = agentAvailability.length;
    const utilizationPct = totalAgents ? Math.round((scheduledCount / totalAgents) * 100) : 0;
    const occupancyPct = 0;
    const needsAttention = understaffed > 0 || scheduledCount === 0;
    return { available, unavailable, adequate, understaffed, coveragePct, scheduledCount, totalAgents, utilizationPct, occupancyPct, needsAttention };
  }, [agentAvailability, staffingData.length, requirementsChartData, scheduleData, manualAssignments]);

  const shiftOptions = useMemo(() => {
    if (!shiftModalAgent?.available) return [];
    const opts: Array<{ startHour: number; startMin: number; endHour: number; endMin: number; label: string; hours: number }> = [];
    const { startHour, endHour } = shiftModalAgent;
    if (startHour + 8 <= endHour) {
      opts.push({ startHour, startMin: 0, endHour: startHour + 8, endMin: 0, label: `${String(startHour).padStart(2, '0')}:00 - ${String(startHour + 8).padStart(2, '0')}:00 8h shift`, hours: 8 });
    }
    for (let h = startHour; h < endHour; h++) {
      for (const m of [0, 30]) {
        const endH = h + 6;
        const endM = m;
        const fits = endH < endHour || (endH === endHour && endM === 0);
        if (fits) {
          opts.push({
            startHour: h,
            startMin: m,
            endHour: endH,
            endMin: endM,
            label: `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')} - ${String(endH).padStart(2, '0')}:${String(endM).padStart(2, '0')} 6h shift`,
            hours: 6,
          });
        }
      }
    }
    return opts;
  }, [shiftModalAgent]);

  const handlePrevWeek = () => setSelectedWeekOffset(prev => prev - 1);
  const handleNextWeek = () => setSelectedWeekOffset(prev => prev + 1);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex flex-col items-center justify-center gap-6">
        <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-400">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Loading schedule data...</span>
        </div>
        {loadingStale && (
          <div className="max-w-md text-center space-y-3">
            <p className="text-sm text-amber-700 dark:text-amber-300">
              Taking longer than expected. The backend may be starting up or the request may have timed out.
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Ensure the backend is running: <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">make backend-up</code>
            </p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 p-6 md:p-8 transition-colors duration-300">
      <style>
        {`
          ::-webkit-scrollbar { width: 8px; height: 8px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: rgba(156, 163, 175, 0.5); border-radius: 4px; }
          * { scrollbar-width: thin; scrollbar-color: rgba(156, 163, 175, 0.5) transparent; }
        `}
      </style>

      <div className="w-full max-w-[1800px] mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/" className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Shift Scheduler</h1>
              <p className="text-slate-500 dark:text-slate-400 text-sm">Schedule agents to meet recommended supply from the emulator</p>
            </div>
          </div>
          <ThemeToggle />
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <Calendar className="w-5 h-5 text-slate-600 dark:text-slate-400" />
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">
                Week of {weekDates[0].toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
              </h2>
            </div>
            <div className="flex items-center space-x-2">
              <button onClick={handlePrevWeek} className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 transition-colors">
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  const today = new Date();
                  const day = today.getDay();
                  setSelectedWeekOffset((day === 0 || day === 6) ? 1 : 0);
                  setSelectedDay(0);
                }}
                className="px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors text-sm font-medium"
              >
                This Week
              </button>
              <button onClick={handleNextWeek} className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 transition-colors">
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-7 gap-2">
            {weekDates.map((date, idx) => {
              const isToday = date.toDateString() === new Date().toDateString();
              const isSelected = selectedDay === idx;
              const weekend = idx >= 5;
              const todayStart = new Date();
              todayStart.setHours(0, 0, 0, 0);
              const isPast = date < todayStart && !isToday;
              return (
                <button
                  key={idx}
                  onClick={() => setSelectedDay(weekend ? 0 : idx)}
                  className={`p-4 rounded-lg transition-all h-24 flex flex-col items-center justify-center ${
                    isSelected ? 'bg-blue-600 text-white shadow-lg' :
                    isToday ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300' :
                    weekend ? 'bg-slate-200 dark:bg-slate-700 text-slate-400' :
                    'bg-slate-50 dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300'
                  }`}
                >
                  <div className="text-xs font-medium mb-1">{date.toLocaleDateString('en-US', { weekday: 'short' })}</div>
                  <div className="text-lg font-bold">{date.getDate()}</div>
                  {isToday && !isSelected && <div className="text-xs mt-1">Today</div>}
                  {weekend && !isSelected && <div className="text-xs mt-1">Closed</div>}
                  {isPast && !isToday && <div className="text-xs mt-1">Past</div>}
                </button>
              );
            })}
          </div>
        </div>

        {!isWeekend && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-4">
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Available Agents</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{summaryMetrics.available}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{summaryMetrics.unavailable} unavailable</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-4">
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Time Slots Coverage</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{summaryMetrics.coveragePct}%</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{summaryMetrics.adequate} adequate, {summaryMetrics.understaffed} understaffed</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-4">
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Agent Occupancy</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{summaryMetrics.occupancyPct}%</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Target: 80-90%</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-4">
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Staff Utilization</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{summaryMetrics.utilizationPct}%</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{summaryMetrics.scheduledCount} of {summaryMetrics.totalAgents} scheduled</div>
            </div>
            <div className={`rounded-xl shadow-sm border p-4 ${summaryMetrics.needsAttention ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800' : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'}`}>
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Schedule Status</div>
              <div className={`text-xl font-bold ${summaryMetrics.needsAttention ? 'text-amber-700 dark:text-amber-300' : 'text-green-700 dark:text-green-300'}`}>
                {summaryMetrics.needsAttention ? 'Needs Attention' : 'OK'}
              </div>
            </div>
          </div>
        )}

        {!isWeekend && (
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-white">Schedule for {selectedDate.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                  {editMode ? 'Edit mode active - Click Complete when finished' : 'Click Edit Schedule to make changes'}
                </p>
              </div>
              {editMode ? (
                <div className="flex items-center gap-2">
                  <button onClick={() => { setEditMode(false); setManualAssignments([]); }} className="px-4 py-2 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg text-sm font-medium">
                    Cancel
                  </button>
                  <button onClick={() => setEditMode(false)} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium">
                    Save
                  </button>
                  <button onClick={() => setEditMode(false)} className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm font-medium">
                    Complete
                  </button>
                </div>
              ) : (
                <button onClick={() => setEditMode(true)} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium">
                  Edit Schedule
                </button>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-center justify-between gap-4">
            <p className="text-red-700 dark:text-red-300">{error}</p>
            <button onClick={fetchScheduleData} className="px-4 py-2 bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 rounded-lg hover:bg-red-200 dark:hover:bg-red-900/60 text-sm font-medium">
              Retry
            </button>
          </div>
        )}

        {isWeekend ? (
          <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-8 text-center">
            <Calendar className="w-12 h-12 mx-auto mb-4 text-slate-400" />
            <h3 className="text-lg font-bold text-slate-700 dark:text-slate-300">Weekend - Office Closed</h3>
            <p className="text-slate-500 dark:text-slate-400 mt-2">No scheduling data available for weekends.</p>
          </div>
        ) : (
          <div className="flex flex-col lg:flex-row gap-6 w-full">
            <div className="flex-1 min-w-0 space-y-6">
              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">
                  Staffing Requirements - {selectedDate.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
                </h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                  Recommended supply from emulator. Red = understaffed, Blue = overstaffed. Adjust schedule to match.
                </p>
                {requirementsChartData.length > 0 ? (
                  <div style={{ width: '100%', height: 280 }}>
                    <ResponsiveContainer width="100%" height={280} minWidth={0}>
                      <ComposedChart data={requirementsChartData} margin={{ left: 20, right: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#374151' : '#e2e8f0'} />
                        <XAxis dataKey="time" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 10 }} />
                        <YAxis domain={[0, 'auto']} allowDataOverflow stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: isDark ? '#1e293b' : '#fff', border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`, borderRadius: '8px' }}
                          formatter={(value: number, name: string, props: any) => {
                            if (name === 'Recommended') return [value, 'Recommended (emulator)'];
                            return [`${value} agents`, props.payload.fill === '#ef4444' ? 'Understaffed' : 'Overstaffed'];
                          }}
                          labelFormatter={(label) => `Slot ${label}`}
                        />
                        <Legend />
                        <Bar dataKey="scheduled" name="Scheduled" radius={[4, 4, 0, 0]}>
                          {requirementsChartData.map((_, i) => (
                            <Cell key={i} fill={requirementsChartData[i].fill} />
                          ))}
                        </Bar>
                        <Line type="monotone" dataKey="recommended" name="Recommended" stroke="#64748b" strokeWidth={2} dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-[280px] flex items-center justify-center text-slate-500 dark:text-slate-400 text-sm">
                    No data available
                  </div>
                )}
                <div className="flex items-center gap-6 mt-3 text-xs">
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-3 rounded bg-red-500" />
                    <span className="text-slate-600 dark:text-slate-400">Understaffed</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-3 rounded bg-blue-500" />
                    <span className="text-slate-600 dark:text-slate-400">Overstaffed</span>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Staffing Visualization - {selectedDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                  Visual representation of agent shifts and staffing requirements. Earliest shift at top, latest at bottom.
                </p>
                {barChartDisplayData.length > 0 ? (
                  <div className="space-y-2">
                    {barChartTotalCount > barChartDisplayLimit && (
                      <p className="text-xs text-slate-500 dark:text-slate-400">Showing first {barChartDisplayLimit} of {barChartTotalCount} agents</p>
                    )}
                    <div style={{ width: '100%', height: 320 }}>
                      <ResponsiveContainer width="100%" height={320} minWidth={0}>
                        <BarChart data={barChartDisplayData} layout="vertical" margin={{ left: 60, right: 20 }} barSize={10} barCategoryGap={2}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#374151' : '#e2e8f0'} />
                          <XAxis type="number" domain={[0, 'auto']} allowDataOverflow stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 10 }} />
                          <YAxis type="category" dataKey="name" width={55} stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 10 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: isDark ? '#1e293b' : '#fff', border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`, borderRadius: '8px' }}
                          formatter={(value: number) => [`${value}h`, 'Duration']}
                          labelFormatter={(_, payload) => payload[0]?.payload ? `${payload[0].payload.startTime} - ${payload[0].payload.endTime}` : ''}
                        />
                        <Bar dataKey="duration" fill="#10b981" radius={[0, 4, 4, 0]} name="Shift" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ) : (
                  <div className="h-[320px] flex flex-col items-center justify-center text-slate-500 dark:text-slate-400 text-sm gap-2">
                    <Users className="w-12 h-12 text-slate-400" />
                    <span>No agents scheduled yet</span>
                    <span className="text-xs">Click + button in Agent Availability to add agents to schedule.</span>
                  </div>
                )}
              </div>
            </div>

            <div className="w-full lg:w-80 xl:w-96 shrink-0 bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
              <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Agent Availability</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">
                {agentAvailability.filter(a => a.available).length} agents available on {selectedDate.toLocaleDateString('en-US', { weekday: 'long' })}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">Available ({agentAvailability.filter(a => a.available).length})</p>
              <div className="space-y-3 max-h-[500px] overflow-auto">
                {agentAvailability.map((a) => {
                  const isManual = manualAssignments.some(m => m.agentId === a.id);
                  const isApiScheduled = scheduleData?.summary?.some(s => String(s.expert_id) === a.id);
                  return (
                    <div
                      key={a.id}
                      className={`p-3 rounded-lg text-sm flex items-center justify-between gap-2 ${
                        a.available
                          ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                          : 'bg-slate-100 dark:bg-slate-700/50 border border-slate-200 dark:border-slate-600'
                      }`}
                    >
                      <div>
                        <div className="font-medium text-slate-900 dark:text-white">{a.name}</div>
                        {a.available ? (
                          <div className="text-green-700 dark:text-green-300 mt-0.5">
                            {String(a.startHour).padStart(2, '0')}:00 - {String(a.endHour).padStart(2, '0')}:00 (8h) {isApiScheduled && !isManual ? 'Auto-scheduled' : ''}
                          </div>
                        ) : (
                          <div className="text-slate-500 dark:text-slate-400 mt-0.5">Unavailable</div>
                        )}
                      </div>
                      {a.available && editMode && (
                        isManual ? (
                          <button
                            onClick={() => setManualAssignments(prev => prev.filter(m => m.agentId !== a.id))}
                            className="shrink-0 px-2 py-1 text-xs font-medium bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 rounded hover:bg-red-200 dark:hover:bg-red-900/60"
                          >
                            Remove
                          </button>
                        ) : (
                          <button
                            onClick={() => setShiftModalAgent(a)}
                            className="shrink-0 w-8 h-8 flex items-center justify-center bg-green-500 hover:bg-green-600 text-white rounded-full"
                          >
                            <Plus className="w-4 h-4" />
                          </button>
                        )
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {shiftModalAgent && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setShiftModalAgent(null)}>
                <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full p-6" onClick={e => e.stopPropagation()}>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Select shift duration and start time</h3>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                    Agent Availability: {String(shiftModalAgent.startHour).padStart(2, '0')}:00 - {String(shiftModalAgent.endHour).padStart(2, '0')}:00 (8h)
                  </p>
                  <div className="space-y-2 max-h-64 overflow-auto mb-6">
                    {shiftOptions.map((opt, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setManualAssignments(prev => [...prev, {
                            agentId: shiftModalAgent.id,
                            agentName: shiftModalAgent.name,
                            startHour: opt.startHour,
                            startMin: opt.startMin,
                            endHour: opt.endHour,
                            endMin: opt.endMin,
                          }]);
                          setShiftModalAgent(null);
                        }}
                        className="w-full text-left px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-900 dark:text-white text-sm"
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                  <div className="flex justify-end gap-2">
                    <button onClick={() => setShiftModalAgent(null)} className="px-4 py-2 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg text-sm">
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
