import React, { useMemo, useState, useEffect, useCallback, useRef } from 'react';
import { ArrowLeft, Calendar, ChevronLeft, ChevronRight, Loader2, Plus, Users, GripVertical } from 'lucide-react';
import { Link } from 'react-router';
import { useTheme } from '../context/ThemeContext';
import { ThemeToggle } from '../components/ThemeToggle';
import { ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';
import { fetchScheduleWithStaffing, formatDate } from '../lib/api';

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
  total_active_agents?: number;
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

interface DraggableShift {
  id: string;
  name: string;
  expert_id: string;
  startMin: number;
  endMin: number;
  isManual: boolean;
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

/* ─── Draggable shift bar ─── */

interface DraggableShiftBarProps {
  shift: DraggableShift;
  minHour: number;
  maxHour: number;
  onUpdate: (id: string, startMin: number, endMin: number) => void;
  editMode: boolean;
}

function DraggableShiftBar({ shift, minHour, maxHour, onUpdate, editMode }: DraggableShiftBarProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<'start' | 'end' | 'move' | null>(null);
  const [dragStartX, setDragStartX] = useState(0);
  const [initialStart, setInitialStart] = useState(0);
  const [initialEnd, setInitialEnd] = useState(0);

  const totalMinutes = (maxHour - minHour) * 60;
  const startOffset = shift.startMin - minHour * 60;
  const duration = shift.endMin - shift.startMin;
  const leftPercent = (startOffset / totalMinutes) * 100;
  const widthPercent = (duration / totalMinutes) * 100;

  const formatTime = (minutes: number) => {
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
  };

  const handleMouseDown = (e: React.MouseEvent, type: 'start' | 'end' | 'move') => {
    if (!editMode) return;
    e.preventDefault();
    e.stopPropagation();
    setDragging(type);
    setDragStartX(e.clientX);
    setInitialStart(shift.startMin);
    setInitialEnd(shift.endMin);
  };

  useEffect(() => {
    if (!dragging) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!barRef.current?.parentElement) return;
      const containerWidth = barRef.current.parentElement.offsetWidth;
      const deltaX = e.clientX - dragStartX;
      const deltaMinutes = Math.round((deltaX / containerWidth) * totalMinutes / 30) * 30;
      let newStart = initialStart;
      let newEnd = initialEnd;
      if (dragging === 'move') {
        newStart = initialStart + deltaMinutes;
        newEnd = initialEnd + deltaMinutes;
        const minMin = minHour * 60;
        const maxMin = maxHour * 60;
        if (newStart < minMin) { newEnd += minMin - newStart; newStart = minMin; }
        if (newEnd > maxMin) { newStart -= newEnd - maxMin; newEnd = maxMin; }
      } else if (dragging === 'start') {
        newStart = Math.max(minHour * 60, Math.min(initialStart + deltaMinutes, initialEnd - 60));
      } else if (dragging === 'end') {
        newEnd = Math.min(maxHour * 60, Math.max(initialEnd + deltaMinutes, initialStart + 60));
      }
      onUpdate(shift.id, newStart, newEnd);
    };
    const handleMouseUp = () => setDragging(null);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => { document.removeEventListener('mousemove', handleMouseMove); document.removeEventListener('mouseup', handleMouseUp); };
  }, [dragging, dragStartX, initialStart, initialEnd, minHour, maxHour, totalMinutes, onUpdate, shift.id]);

  const durationHours = ((shift.endMin - shift.startMin) / 60).toFixed(1);

  return (
    <div
      ref={barRef}
      className={`absolute h-6 rounded flex items-center group bg-green-500 hover:bg-green-600 ${editMode ? 'cursor-move' : ''} ${dragging ? 'opacity-80' : ''}`}
      style={{ left: `${leftPercent}%`, width: `${Math.max(widthPercent, 2)}%`, top: '50%', transform: 'translateY(-50%)' }}
      onMouseDown={(e) => handleMouseDown(e, 'move')}
      title={`${shift.name}: ${formatTime(shift.startMin)} - ${formatTime(shift.endMin)} (${durationHours}h)`}
    >
      {editMode && (
        <>
          <div className="absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-white/30 rounded-l flex items-center justify-center" onMouseDown={(e) => handleMouseDown(e, 'start')}>
            <GripVertical className="w-3 h-3 text-white/70 opacity-0 group-hover:opacity-100" />
          </div>
          <div className="absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-white/30 rounded-r flex items-center justify-center" onMouseDown={(e) => handleMouseDown(e, 'end')}>
            <GripVertical className="w-3 h-3 text-white/70 opacity-0 group-hover:opacity-100" />
          </div>
        </>
      )}
      <span className="text-[10px] text-white font-medium px-1 truncate w-full text-center pointer-events-none">
        {widthPercent > 8 ? `${formatTime(shift.startMin)}-${formatTime(shift.endMin)}` : ''}
      </span>
    </div>
  );
}

/* ─── Interactive shift chart ─── */

interface InteractiveShiftChartProps {
  shifts: DraggableShift[];
  onUpdateShift: (id: string, startMin: number, endMin: number) => void;
  editMode: boolean;
  isDark: boolean;
  displayLimit?: number;
}

function InteractiveShiftChart({ shifts, onUpdateShift, editMode, isDark, displayLimit = 10 }: InteractiveShiftChartProps) {
  const minHour = 5;
  const maxHour = 17;
  const hours = Array.from({ length: maxHour - minHour + 1 }, (_, i) => minHour + i);
  const displayShifts = shifts.slice(0, displayLimit);

  return (
    <div className="w-full">
      <div className="flex">
        <div className="w-20 shrink-0" />
        <div className="flex-1 flex border-b border-slate-200 dark:border-slate-700">
          {hours.map((h) => (
            <div key={h} className="flex-1 text-center text-[10px] text-slate-500 dark:text-slate-400 py-1 border-l border-slate-200 dark:border-slate-700 first:border-l-0">
              {String(h).padStart(2, '0')}:00
            </div>
          ))}
        </div>
      </div>
      <div className="max-h-[320px] overflow-y-auto">
        {displayShifts.map((shift) => (
          <div key={shift.id} className="flex items-center h-8 border-b border-slate-100 dark:border-slate-800">
            <div className="w-20 shrink-0 text-[11px] text-slate-600 dark:text-slate-400 truncate px-2">{shift.name}</div>
            <div className="flex-1 relative h-full bg-slate-50 dark:bg-slate-900/50">
              {hours.map((h) => (
                <div key={h} className="absolute top-0 bottom-0 border-l border-slate-200 dark:border-slate-700" style={{ left: `${((h - minHour) / (maxHour - minHour)) * 100}%` }} />
              ))}
              <DraggableShiftBar shift={shift} minHour={minHour} maxHour={maxHour} onUpdate={onUpdateShift} editMode={editMode} />
            </div>
          </div>
        ))}
      </div>
      {shifts.length > displayLimit && (
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">Showing {displayLimit} of {shifts.length} agents</p>
      )}
      <div className="flex items-center gap-4 mt-3 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-4 h-3 rounded bg-green-500" />
          <span className="text-slate-600 dark:text-slate-400">Scheduled shift</span>
        </div>
        {editMode && <span className="text-slate-500 dark:text-slate-400 ml-auto">Drag bars to adjust shift times</span>}
      </div>
    </div>
  );
}

/* ─── Main component ─── */

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
  const [shiftOverrides, setShiftOverrides] = useState<Map<string, { startMin: number; endMin: number }>>(new Map());

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
    if (isWeekend) { setScheduleData(null); setStaffingData([]); setLoading(false); return; }
    setLoading(true); setError(null); setLoadingStale(false);
    try {
      const { staffing, schedule } = await fetchScheduleWithStaffing(selectedDate, 80, 60, 85);
      setStaffingData(staffing.map(s => ({ time: s.time, calls: s.predicted_calls, agents: s.agents })));
      setScheduleData(schedule as ScheduleData);
    } catch (err) {
      setScheduleData(null); setStaffingData([]);
      setError(err instanceof Error ? err.message : 'Failed to load schedule');
    } finally { setLoading(false); }
  }, [selectedDateStr, selectedDate, isWeekend]);

  useEffect(() => { fetchScheduleData(); }, [fetchScheduleData]);
  useEffect(() => { setManualAssignments([]); setShiftOverrides(new Map()); }, [selectedDateStr]);

  /* ─── Draggable shifts data (manual first, stable sort by original start) ─── */

  const draggableShifts = useMemo((): DraggableShift[] => {
    const minMin = 5 * 60;
    const maxMin = 17 * 60;

    const apiRows = (scheduleData?.summary ?? []).map(agent => {
      const start = parseSlotAsUtc(agent.shift_start);
      const end = parseSlotAsUtc(agent.shift_end);
      const id = `api-${agent.expert_id}`;
      const override = shiftOverrides.get(id);
      let startMinBase = start.getUTCHours() * 60 + start.getUTCMinutes();
      let endMinBase = end.getUTCHours() * 60 + end.getUTCMinutes();
      // UTC → PST
      startMinBase = ((startMinBase - 8 * 60) + 24 * 60) % (24 * 60);
      endMinBase = ((endMinBase - 8 * 60) + 24 * 60) % (24 * 60);
      if (endMinBase <= startMinBase) endMinBase = Math.min(startMinBase + 12 * 60, maxMin);
      return {
        id,
        name: `Agent ${String(agent.expert_id).replace(/\.0$/, '').slice(-4)}`,
        expert_id: String(agent.expert_id),
        startMin: override?.startMin ?? startMinBase,
        endMin: override?.endMin ?? endMinBase,
        isManual: false,
        sortKey: startMinBase, // stable sort by original time
      };
    }).filter(row => row.endMin > minMin && row.startMin < maxMin);

    const manualRows = manualAssignments.map((ma, idx) => {
      const id = `manual-${ma.agentId}-${idx}`;
      const override = shiftOverrides.get(id);
      const startMinBase = ma.startHour * 60 + (ma.startMin ?? 0);
      const endMinBase = ma.endHour * 60 + (ma.endMin ?? 0);
      return {
        id,
        name: ma.agentName,
        expert_id: ma.agentId,
        startMin: override?.startMin ?? startMinBase,
        endMin: override?.endMin ?? endMinBase,
        isManual: true,
        sortKey: startMinBase,
      };
    });

    const combined = [
      ...manualRows.sort((a, b) => a.sortKey - b.sortKey),
      ...apiRows.sort((a, b) => a.sortKey - b.sortKey),
    ];
    const seen = new Set<string>();
    return combined.map((row, idx) => {
      let name = row.name;
      if (seen.has(name)) name = `${row.name} (${idx})`;
      seen.add(name);
      return { ...row, name };
    });
  }, [scheduleData, manualAssignments, shiftOverrides]);

  const handleUpdateShift = useCallback((id: string, startMin: number, endMin: number) => {
    setShiftOverrides(prev => { const next = new Map(prev); next.set(id, { startMin, endMin }); return next; });
  }, []);

  /* ─── Requirements chart (uses draggable shifts for scheduled count) ─── */

  const requirementsChartData = useMemo(() => {
    const slots = ['05:00','05:30','06:00','06:30','07:00','07:30','08:00','08:30','09:00','09:30','10:00','10:30','11:00','11:30','12:00','12:30','13:00','13:30','14:00','14:30','15:00','15:30','16:00','16:30'];
    const byTime = new Map<string, { recommended: number; scheduled: number }>();
    slots.forEach(t => byTime.set(t, { recommended: 0, scheduled: 0 }));
    if (staffingData.length > 0) {
      staffingData.forEach(s => { const e = byTime.get(s.time); if (e) e.recommended = Math.max(0, s.agents); });
    }
    draggableShifts.forEach(shift => {
      let cur = shift.startMin;
      while (cur < shift.endMin) {
        const h = Math.floor(cur / 60), m = cur % 60;
        const key = `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`;
        const entry = byTime.get(key);
        if (entry) entry.scheduled += 1;
        cur += 30;
      }
    });
    if (staffingData.length === 0) {
      scheduleData?.coverage?.forEach(c => {
        const t = parseSlotAsUtc(c.slot_start);
        const localH = (t.getUTCHours() - 8 + 24) % 24;
        const key = `${String(localH).padStart(2,'0')}:${String(t.getUTCMinutes()).padStart(2,'0')}`;
        const entry = byTime.get(key);
        if (entry) entry.recommended = Math.max(1, Math.ceil((c.predicted_demand || 1) / 5));
      });
    }
    const hasData = staffingData.length > 0 || (scheduleData?.coverage?.length ?? 0) > 0 || draggableShifts.length > 0;
    if (!hasData) return [];
    return slots.map(time => {
      const entry = byTime.get(time) ?? { recommended: 0, scheduled: 0 };
      const rec = staffingData.length > 0 ? entry.recommended : (entry.recommended || 1);
      return { time, recommended: rec, scheduled: entry.scheduled, fill: entry.scheduled < rec ? '#ef4444' : '#3b82f6' };
    });
  }, [scheduleData, staffingData, draggableShifts]);

  /* ─── Agent availability & metrics ─── */

  const totalActiveAgents = scheduleData?.total_active_agents ?? 0;

  const agentAvailability = useMemo((): AgentAvailability[] => {
    if (!scheduleData?.summary?.length) {
      return generateRandomAvailability(selectedDateStr, Array.from({ length: 21 }, (_, i) => `agent-${i + 1}`));
    }
    const summaries = scheduleData.summary.slice(0, 20);
    return summaries.map(agent => {
      const id = String(agent.expert_id);
      const displayId = id.replace(/\.0$/, '');
      const name = displayId.length > 4 ? `Agent ${displayId.slice(-4)}` : `Agent ${displayId}`;
      const start = parseSlotAsUtc(agent.shift_start);
      const end = parseSlotAsUtc(agent.shift_end);
      let startH = ((start.getUTCHours() - 8) + 24) % 24;
      let endH = ((end.getUTCHours() - 8) + 24) % 24;
      if (endH <= startH) endH = Math.min(startH + 8, 17);
      return { id, name, available: true, startHour: startH, endHour: endH };
    });
  }, [scheduleData, selectedDateStr]);

  const summaryMetrics = useMemo(() => {
    const totalAgents = totalActiveAgents || agentAvailability.length;
    const scheduledCount = (scheduleData?.summary?.length ?? 0) + manualAssignments.length;
    const available = scheduledCount;
    const unavailable = Math.max(0, totalAgents - scheduledCount);
    const totalSlots = staffingData.length;
    let adequate = 0, understaffed = 0;
    requirementsChartData.forEach(d => { if (d.scheduled >= d.recommended) adequate++; else understaffed++; });
    const coveragePct = totalSlots ? Math.round((adequate / totalSlots) * 100) : 0;
    const utilizationPct = totalAgents ? Math.round((scheduledCount / totalAgents) * 100) : 0;
    const needsAttention = understaffed > 0 || scheduledCount === 0;
    return { available, unavailable, adequate, understaffed, coveragePct, scheduledCount, totalAgents, utilizationPct, occupancyPct: 0, needsAttention };
  }, [totalActiveAgents, agentAvailability.length, staffingData.length, requirementsChartData, scheduleData, manualAssignments]);

  const shiftOptions = useMemo(() => {
    if (!shiftModalAgent?.available) return [];
    const opts: Array<{ startHour: number; startMin: number; endHour: number; endMin: number; label: string; hours: number }> = [];
    const { startHour, endHour } = shiftModalAgent;
    if (startHour + 8 <= endHour) {
      opts.push({ startHour, startMin: 0, endHour: startHour + 8, endMin: 0, label: `${String(startHour).padStart(2,'0')}:00 - ${String(startHour+8).padStart(2,'0')}:00 8h shift`, hours: 8 });
    }
    for (let h = startHour; h < endHour; h++) {
      for (const m of [0, 30]) {
        const endH = h + 6, endM = m;
        if (endH < endHour || (endH === endHour && endM === 0)) {
          opts.push({ startHour: h, startMin: m, endHour: endH, endMin: endM, label: `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')} - ${String(endH).padStart(2,'0')}:${String(endM).padStart(2,'0')} 6h shift`, hours: 6 });
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
            <p className="text-sm text-amber-700 dark:text-amber-300">Taking longer than expected. The backend may be starting up or the request may have timed out.</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Ensure the backend is running: <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">make backend-up</code></p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 p-6 md:p-8 transition-colors duration-300">
      <style>{`::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(156,163,175,.5);border-radius:4px}*{scrollbar-width:thin;scrollbar-color:rgba(156,163,175,.5) transparent}`}</style>

      <div className="w-full max-w-[1800px] mx-auto space-y-6">
        {/* Header */}
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

        {/* Week picker */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <Calendar className="w-5 h-5 text-slate-600 dark:text-slate-400" />
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">Week of {weekDates[0].toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</h2>
            </div>
            <div className="flex items-center space-x-2">
              <button onClick={handlePrevWeek} className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 transition-colors"><ChevronLeft className="w-4 h-4" /></button>
              <button onClick={() => { const d = new Date().getDay(); setSelectedWeekOffset((d===0||d===6)?1:0); setSelectedDay(0); }} className="px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors text-sm font-medium">This Week</button>
              <button onClick={handleNextWeek} className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 transition-colors"><ChevronRight className="w-4 h-4" /></button>
            </div>
          </div>
          <div className="grid grid-cols-7 gap-2">
            {weekDates.map((date, idx) => {
              const isToday = date.toDateString() === new Date().toDateString();
              const isSelected = selectedDay === idx;
              const weekend = idx >= 5;
              const todayStart = new Date(); todayStart.setHours(0,0,0,0);
              const isPast = date < todayStart && !isToday;
              return (
                <button key={idx} onClick={() => setSelectedDay(weekend ? 0 : idx)}
                  className={`p-4 rounded-lg transition-all h-24 flex flex-col items-center justify-center ${
                    isSelected ? 'bg-blue-600 text-white shadow-lg' :
                    isToday ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300' :
                    weekend ? 'bg-slate-200 dark:bg-slate-700 text-slate-400' :
                    'bg-slate-50 dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300'}`}>
                  <div className="text-xs font-medium mb-1">{date.toLocaleDateString('en-US',{weekday:'short'})}</div>
                  <div className="text-lg font-bold">{date.getDate()}</div>
                  {isToday && !isSelected && <div className="text-xs mt-1">Today</div>}
                  {weekend && !isSelected && <div className="text-xs mt-1">Closed</div>}
                  {isPast && !isToday && <div className="text-xs mt-1">Past</div>}
                </button>
              );
            })}
          </div>
        </div>

        {/* Metric cards */}
        {!isWeekend && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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

        {/* Edit bar */}
        {!isWeekend && (
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-white">Schedule for {selectedDate.toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'})}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{editMode ? 'Edit mode active - Click Complete when finished' : 'Click Edit Schedule to make changes'}</p>
              </div>
              {editMode ? (
                <div className="flex items-center gap-2">
                  <button onClick={() => { setEditMode(false); setManualAssignments([]); setShiftOverrides(new Map()); }} className="px-4 py-2 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg text-sm font-medium">Cancel</button>
                  <button onClick={() => setEditMode(false)} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium">Save</button>
                  <button onClick={() => setEditMode(false)} className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm font-medium">Complete</button>
                </div>
              ) : (
                <button onClick={() => setEditMode(true)} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium">Edit Schedule</button>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-center justify-between gap-4">
            <p className="text-red-700 dark:text-red-300">{error}</p>
            <button onClick={fetchScheduleData} className="px-4 py-2 bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 rounded-lg hover:bg-red-200 dark:hover:bg-red-900/60 text-sm font-medium">Retry</button>
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

              {/* Staffing requirements chart */}
              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Staffing Requirements - {selectedDate.toLocaleDateString('en-US',{weekday:'long',month:'short',day:'numeric'})}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">Recommended supply from emulator. Red = understaffed, Blue = adequately staffed.</p>
                {requirementsChartData.length > 0 ? (
                  <div style={{ width: '100%', height: 280 }}>
                    <ResponsiveContainer width="100%" height={280} minWidth={0}>
                      <ComposedChart data={requirementsChartData} margin={{ left: 20, right: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#374151' : '#e2e8f0'} />
                        <XAxis dataKey="time" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 10 }} />
                        <YAxis domain={[0, 'auto']} allowDataOverflow stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: isDark ? '#1e293b' : '#fff', border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`, borderRadius: '8px' }}
                          formatter={(value: number, name: string, props: any) => {
                            if (name === 'Recommended') return [value, 'Recommended (emulator)'];
                            return [`${value} agents`, props.payload.fill === '#ef4444' ? 'Understaffed' : 'Adequately staffed'];
                          }}
                          labelFormatter={(label) => `Slot ${label}`} />
                        <Legend />
                        <Bar dataKey="scheduled" name="Scheduled" radius={[4,4,0,0]}>
                          {requirementsChartData.map((_,i) => <Cell key={i} fill={requirementsChartData[i].fill} />)}
                        </Bar>
                        <Line type="monotone" dataKey="recommended" name="Recommended" stroke="#64748b" strokeWidth={2} dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-[280px] flex items-center justify-center text-slate-500 dark:text-slate-400 text-sm">No data available</div>
                )}
                <div className="flex items-center gap-6 mt-3 text-xs">
                  <div className="flex items-center gap-1"><div className="w-4 h-3 rounded bg-red-500" /><span className="text-slate-600 dark:text-slate-400">Understaffed</span></div>
                  <div className="flex items-center gap-1"><div className="w-4 h-3 rounded bg-blue-500" /><span className="text-slate-600 dark:text-slate-400">Adequately staffed</span></div>
                </div>
              </div>

              {/* Interactive shift visualization */}
              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Staffing Visualization - {selectedDate.toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric'})}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                  {editMode ? 'Drag shift bars to adjust start/end times. Drag edges to resize, drag middle to move.' : 'Visual representation of agent shifts. Click "Edit Schedule" to adjust times.'}
                </p>
                {draggableShifts.length > 0 ? (
                  <InteractiveShiftChart shifts={draggableShifts} onUpdateShift={handleUpdateShift} editMode={editMode} isDark={isDark} displayLimit={10} />
                ) : (
                  <div className="h-[320px] flex flex-col items-center justify-center text-slate-500 dark:text-slate-400 text-sm gap-2">
                    <Users className="w-12 h-12 text-slate-400" />
                    <span>No agents scheduled yet</span>
                    <span className="text-xs">Click + button in Agent Availability to add agents to schedule.</span>
                  </div>
                )}
              </div>
            </div>

            {/* Agent availability sidebar */}
            <div className="w-full lg:w-80 xl:w-96 shrink-0 bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
              <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Agent Availability</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">{summaryMetrics.scheduledCount} out of {summaryMetrics.totalAgents} agents scheduled on {selectedDate.toLocaleDateString('en-US',{weekday:'long'})}</p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">Showing {agentAvailability.length} agents</p>
              <div className="space-y-3 max-h-[500px] overflow-auto">
                {agentAvailability.map((a) => {
                  const isManual = manualAssignments.some(m => m.agentId === a.id);
                  const isApiScheduled = scheduleData?.summary?.some(s => String(s.expert_id) === a.id);
                  return (
                    <div key={a.id} className={`p-3 rounded-lg text-sm flex items-center justify-between gap-2 ${a.available ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800' : 'bg-slate-100 dark:bg-slate-700/50 border border-slate-200 dark:border-slate-600'}`}>
                      <div>
                        <div className="font-medium text-slate-900 dark:text-white">{a.name}</div>
                        {a.available ? (
                          <div className="text-green-700 dark:text-green-300 mt-0.5">{String(a.startHour).padStart(2,'0')}:00 - {String(a.endHour).padStart(2,'0')}:00 (8h) {isApiScheduled && !isManual ? 'Auto-scheduled' : ''}</div>
                        ) : (
                          <div className="text-slate-500 dark:text-slate-400 mt-0.5">Unavailable</div>
                        )}
                      </div>
                      {a.available && editMode && (
                        isManual ? (
                          <button onClick={() => setManualAssignments(prev => prev.filter(m => m.agentId !== a.id))} className="shrink-0 px-2 py-1 text-xs font-medium bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 rounded hover:bg-red-200 dark:hover:bg-red-900/60">Remove</button>
                        ) : (
                          <button onClick={() => setShiftModalAgent(a)} className="shrink-0 w-8 h-8 flex items-center justify-center bg-green-500 hover:bg-green-600 text-white rounded-full"><Plus className="w-4 h-4" /></button>
                        )
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Shift assignment modal */}
            {shiftModalAgent && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setShiftModalAgent(null)}>
                <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full p-6" onClick={e => e.stopPropagation()}>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Select shift duration and start time</h3>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">Agent Availability: {String(shiftModalAgent.startHour).padStart(2,'0')}:00 - {String(shiftModalAgent.endHour).padStart(2,'0')}:00 (8h)</p>
                  <div className="space-y-2 max-h-64 overflow-auto mb-6">
                    {shiftOptions.map((opt, i) => (
                      <button key={i} onClick={() => { setManualAssignments(prev => [...prev, { agentId: shiftModalAgent.id, agentName: shiftModalAgent.name, startHour: opt.startHour, startMin: opt.startMin, endHour: opt.endHour, endMin: opt.endMin }]); setShiftModalAgent(null); }}
                        className="w-full text-left px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-900 dark:text-white text-sm">{opt.label}</button>
                    ))}
                  </div>
                  <div className="flex justify-end gap-2">
                    <button onClick={() => setShiftModalAgent(null)} className="px-4 py-2 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg text-sm">Cancel</button>
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
