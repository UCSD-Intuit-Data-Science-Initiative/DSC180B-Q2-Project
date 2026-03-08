import React, { useMemo, useState, useEffect } from 'react';
import { ArrowLeft, Calendar, Users, Clock, AlertCircle, CheckCircle, ChevronLeft, ChevronRight, Target, Activity, Loader2 } from 'lucide-react';
import { Link } from 'react-router';
import { useTheme } from '../context/ThemeContext';
import { ThemeToggle } from '../components/ThemeToggle';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface AvailableAgent {
  expert_id: string;
  name: string;
  segment: string;
  business_segment: string;
  available_hours: number;
  mean_work_freq: number;
  mean_occupancy: number;
  resolution_rate: number;
}

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

interface AvailabilityData {
  date: string;
  day_of_week: string;
  available_agents: AvailableAgent[];
  total_available: number;
}

const API_BASE = 'http://localhost:8000';

export default function ShiftScheduler() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [isMounted, setIsMounted] = useState(false);

  const [selectedWeekOffset, setSelectedWeekOffset] = useState(0);
  const [selectedDay, setSelectedDay] = useState(() => {
    const today = new Date();
    const day = today.getDay();
    return day === 0 ? 6 : day - 1;
  });

  const [scheduleData, setScheduleData] = useState<ScheduleData | null>(null);
  const [availabilityData, setAvailabilityData] = useState<AvailabilityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setIsMounted(true);
  }, []);

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

  const selectedDateStr = useMemo(() => {
    return weekDates[selectedDay].toISOString().split('T')[0];
  }, [weekDates, selectedDay]);

  const selectedDateIsPast = useMemo(() => {
    const selectedDate = weekDates[selectedDay];
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return selectedDate < today;
  }, [weekDates, selectedDay]);

  const isWeekend = selectedDay >= 5;

  useEffect(() => {
    const fetchData = async () => {
      if (isWeekend) {
        setScheduleData(null);
        setAvailabilityData(null);
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        const [scheduleRes, availRes] = await Promise.all([
          fetch(`${API_BASE}/api/schedule?date=${selectedDateStr}`),
          fetch(`${API_BASE}/api/schedule/availability?date=${selectedDateStr}`),
        ]);

        if (!scheduleRes.ok) throw new Error(`Schedule API error: ${scheduleRes.status}`);
        if (!availRes.ok) throw new Error(`Availability API error: ${availRes.status}`);

        const schedData = await scheduleRes.json();
        const availData = await availRes.json();

        setScheduleData(schedData);
        setAvailabilityData(availData);
      } catch (err) {
        console.error('Failed to fetch schedule data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedDateStr, isWeekend]);

  const coverageStatus = useMemo(() => {
    if (!scheduleData?.coverage || scheduleData.coverage.length === 0) {
      return { understaffed: 0, adequate: 0, coveragePercent: 0, showWarning: false };
    }

    const understaffed = scheduleData.coverage.filter(c => c.coverage_ratio < 1).length;
    const adequate = scheduleData.coverage.filter(c => c.coverage_ratio >= 1).length;
    const totalSlots = scheduleData.coverage.length;
    const coveragePercent = totalSlots > 0 ? Math.round((adequate / totalSlots) * 100) : 0;

    return { understaffed, adequate, coveragePercent, showWarning: !selectedDateIsPast && understaffed > 0 };
  }, [scheduleData, selectedDateIsPast]);

  const staffingMetrics = useMemo(() => {
    const assignedCount = scheduleData?.summary?.length || 0;
    const availableCount = availabilityData?.total_available || 0;

    let avgOccupancy = 0;
    if (scheduleData?.coverage && scheduleData.coverage.length > 0) {
      const totalOcc = scheduleData.coverage.reduce((sum, c) => {
        const occ = c.agents_assigned > 0 ? Math.min(100, (c.predicted_demand / c.agents_assigned) * 10) : 0;
        return sum + occ;
      }, 0);
      avgOccupancy = totalOcc / scheduleData.coverage.length;
    }

    const utilization = availableCount > 0 ? (assignedCount / availableCount) * 100 : 0;

    return {
      assignedCount,
      availableCount,
      avgOccupancy: Math.round(avgOccupancy),
      utilization: Math.round(utilization),
      occupancyTarget: { min: 80, max: 90 },
      utilizationTarget: { min: 75, max: 95 },
    };
  }, [scheduleData, availabilityData]);

  const chartData = useMemo(() => {
    if (!scheduleData?.coverage) return [];

    return scheduleData.coverage.map(c => {
      const slotTime = new Date(c.slot_start);
      const timeStr = slotTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
      const gap = c.agents_assigned < Math.ceil(c.predicted_demand / 10)
        ? Math.ceil(c.predicted_demand / 10) - c.agents_assigned : 0;
      const excess = c.agents_assigned > Math.ceil(c.predicted_demand / 10)
        ? c.agents_assigned - Math.ceil(c.predicted_demand / 10) : 0;
      const optimal = Math.min(c.agents_assigned, Math.ceil(c.predicted_demand / 10));

      return { time: timeStr, gap, excess, optimal, demand: c.predicted_demand, agents: c.agents_assigned };
    });
  }, [scheduleData]);

  const handlePrevWeek = () => setSelectedWeekOffset(prev => prev - 1);
  const handleNextWeek = () => setSelectedWeekOffset(prev => prev + 1);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-400">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Loading schedule data...</span>
        </div>
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
          ::-webkit-scrollbar-thumb:hover { background: rgba(156, 163, 175, 0.8); }
          * { scrollbar-width: thin; scrollbar-color: rgba(156, 163, 175, 0.5) transparent; }
        `}
      </style>

      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/" className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Shift Scheduler</h1>
              <p className="text-slate-500 dark:text-slate-400 text-sm">View agent scheduling based on predicted call demand</p>
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
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button onClick={() => setSelectedWeekOffset(0)} className="px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors text-sm font-medium">
                This Week
              </button>
              <button onClick={handleNextWeek} className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 transition-colors">
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-7 gap-2">
            {weekDates.map((date, idx) => {
              const isToday = date.toDateString() === new Date().toDateString();
              const isSelected = selectedDay === idx;
              const isPast = (() => {
                const today = new Date();
                today.setHours(0, 0, 0, 0);
                return date < today;
              })();
              const weekend = idx >= 5;

              return (
                <button
                  key={idx}
                  onClick={() => setSelectedDay(idx)}
                  className={`p-4 rounded-lg transition-all h-24 flex flex-col items-center justify-center ${
                    isSelected
                      ? 'bg-blue-600 text-white shadow-lg'
                      : isToday
                      ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300'
                      : weekend
                      ? 'bg-slate-200 dark:bg-slate-700 text-slate-400 dark:text-slate-500'
                      : isPast
                      ? 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-700'
                      : 'bg-slate-50 dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300'
                  }`}
                >
                  <div className="text-xs font-medium mb-1">
                    {date.toLocaleDateString('en-US', { weekday: 'short' })}
                  </div>
                  <div className="text-lg font-bold">{date.getDate()}</div>
                  {isToday && !isSelected && <div className="text-xs mt-1">Today</div>}
                  {weekend && !isSelected && <div className="text-xs mt-1">Closed</div>}
                </button>
              );
            })}
          </div>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <p className="text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}

        {isWeekend ? (
          <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-8 text-center">
            <Calendar className="w-12 h-12 mx-auto mb-4 text-slate-400" />
            <h3 className="text-lg font-bold text-slate-700 dark:text-slate-300">Weekend - Office Closed</h3>
            <p className="text-slate-500 dark:text-slate-400 mt-2">No scheduling data available for weekends.</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5">
                <div className="flex items-center space-x-2 mb-2">
                  <Users className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Available Agents</span>
                </div>
                <p className="text-3xl font-bold text-slate-900 dark:text-white">{staffingMetrics.availableCount}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{staffingMetrics.assignedCount} scheduled</p>
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5">
                <div className="flex items-center space-x-2 mb-2">
                  <Clock className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Time Slots Coverage</span>
                </div>
                <p className="text-3xl font-bold text-slate-900 dark:text-white">{coverageStatus.coveragePercent}%</p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  {coverageStatus.adequate} adequate, {coverageStatus.understaffed} understaffed
                </p>
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5">
                <div className="flex items-center space-x-2 mb-2">
                  <Target className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                  <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Agent Occupancy</span>
                </div>
                <p className={`text-3xl font-bold ${
                  staffingMetrics.avgOccupancy < 60 ? 'text-yellow-600 dark:text-yellow-400' :
                  staffingMetrics.avgOccupancy > 90 ? 'text-red-600 dark:text-red-400' :
                  'text-green-600 dark:text-green-400'
                }`}>
                  {staffingMetrics.avgOccupancy}%
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Target: 80-90%</p>
              </div>

              <div className={`rounded-xl shadow-sm border p-5 ${
                coverageStatus.showWarning
                  ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-700'
                  : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700'
              }`}>
                <div className="flex items-center space-x-2 mb-2">
                  {coverageStatus.showWarning ? (
                    <AlertCircle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                  ) : (
                    <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                  )}
                  <span className={`text-sm font-medium ${
                    coverageStatus.showWarning
                      ? 'text-yellow-700 dark:text-yellow-300'
                      : 'text-green-700 dark:text-green-300'
                  }`}>
                    Schedule Status
                  </span>
                </div>
                <p className={`text-3xl font-bold ${
                  coverageStatus.showWarning
                    ? 'text-yellow-900 dark:text-yellow-200'
                    : 'text-green-900 dark:text-green-200'
                }`}>
                  {coverageStatus.showWarning ? 'Needs Attention' : 'Covered'}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                  Staffing Requirements - {weekDates[selectedDay].toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
                </h3>

                {chartData.length > 0 ? (
                  <>
                    <div className="flex items-center space-x-4 text-xs mb-4">
                      <div className="flex items-center space-x-1">
                        <div className="w-3 h-3 bg-red-500 rounded"></div>
                        <span className="text-slate-600 dark:text-slate-400">Understaffed</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <div className="w-3 h-3 bg-green-500 rounded"></div>
                        <span className="text-slate-600 dark:text-slate-400">Optimal</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <div className="w-3 h-3 bg-blue-500 rounded"></div>
                        <span className="text-slate-600 dark:text-slate-400">Overstaffed</span>
                      </div>
                    </div>

                    <div style={{ height: '300px' }}>
                      {isMounted && (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#374151' : '#e2e8f0'} />
                            <XAxis dataKey="time" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
                            <YAxis stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11 }} />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: isDark ? '#1e293b' : '#ffffff',
                                border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`,
                                borderRadius: '8px',
                                color: isDark ? '#f1f5f9' : '#0f172a'
                              }}
                              formatter={(value: any, name: string) => {
                                if (name === 'gap') return [value, 'Need to Add'];
                                if (name === 'excess') return [value, 'Can Reduce'];
                                if (name === 'optimal') return [value, 'Optimal'];
                                return [value, name];
                              }}
                            />
                            <Legend formatter={(value: string) => {
                              if (value === 'gap') return 'Need to Add';
                              if (value === 'excess') return 'Can Reduce';
                              if (value === 'optimal') return 'Optimal';
                              return value;
                            }} />
                            <Bar dataKey="optimal" stackId="a" fill="#10b981" name="optimal" />
                            <Bar dataKey="gap" stackId="a" fill="#ef4444" name="gap" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="excess" stackId="a" fill="#3b82f6" name="excess" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12 text-slate-500 dark:text-slate-400">
                    <Calendar className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm font-medium">No schedule data available</p>
                  </div>
                )}
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">Scheduled Agents</h3>

                {scheduleData?.summary && scheduleData.summary.length > 0 ? (
                  <div className="space-y-3 max-h-[400px] overflow-auto">
                    {scheduleData.summary.map((agent, idx) => {
                      const startTime = new Date(agent.shift_start).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
                      const endTime = new Date(agent.shift_end).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

                      return (
                        <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-xs font-bold text-blue-600 dark:text-blue-400">
                              {agent.expert_id.slice(-2).toUpperCase()}
                            </div>
                            <div>
                              <Link
                                to={`/agent/${encodeURIComponent(agent.expert_id)}`}
                                className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline"
                              >
                                {agent.expert_id}
                              </Link>
                              <p className="text-xs text-slate-500 dark:text-slate-400">{agent.work_hours}h work</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                              {startTime} - {endTime}
                            </p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">{agent.shift_block}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-500 dark:text-slate-400">
                    <Users className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No agents scheduled</p>
                  </div>
                )}
              </div>
            </div>

            {availabilityData && availabilityData.available_agents.length > 0 && (
              <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                  Available Agents for {availabilityData.day_of_week}
                </h3>
                <div className="overflow-auto max-h-[300px]">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 dark:bg-slate-900 text-slate-500 dark:text-slate-400 sticky top-0">
                      <tr>
                        <th className="px-4 py-2 text-left">Agent ID</th>
                        <th className="px-4 py-2 text-left">Segment</th>
                        <th className="px-4 py-2 text-right">Available Hours</th>
                        <th className="px-4 py-2 text-right">Work Frequency</th>
                        <th className="px-4 py-2 text-right">Resolution Rate</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                      {availabilityData.available_agents.slice(0, 20).map((agent, idx) => (
                        <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-700/30">
                          <td className="px-4 py-2">
                            <Link
                              to={`/agent/${encodeURIComponent(agent.expert_id)}`}
                              className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
                            >
                              {agent.expert_id}
                            </Link>
                          </td>
                          <td className="px-4 py-2 text-slate-600 dark:text-slate-400">{agent.segment}</td>
                          <td className="px-4 py-2 text-right text-slate-600 dark:text-slate-400">{agent.available_hours}h</td>
                          <td className="px-4 py-2 text-right text-slate-600 dark:text-slate-400">{agent.mean_work_freq.toFixed(0)}%</td>
                          <td className="px-4 py-2 text-right text-slate-600 dark:text-slate-400">{agent.resolution_rate.toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {availabilityData.available_agents.length > 20 && (
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-3 text-center">
                    Showing 20 of {availabilityData.total_available} available agents
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
