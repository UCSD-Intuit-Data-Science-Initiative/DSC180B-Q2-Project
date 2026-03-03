import React, { useMemo, useState } from 'react';
import { ArrowLeft, Calendar, Clock, Target, Phone, TrendingUp, ArrowUp, ArrowDown, ChevronDown, ChevronRight } from 'lucide-react';
import { Link } from 'react-router';

type SortKey = 'date' | 'sla' | 'waitTime' | 'occupancy' | 'calls';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  key: SortKey;
  direction: SortDirection;
}

// Simple seeded random function (must match Dashboard logic)
function seededRandom(seed: number) {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
}

// Generate daily breakdown data (48 half-hour slots) - must match Dashboard
function getDailyBreakdownData(date: Date) {
    const points = [];
    const seedBase = date.getFullYear() * 10000 + (date.getMonth() + 1) * 100 + date.getDate();

    for (let i = 0; i < 48; i++) {
      const hour = Math.floor(i / 2);
      const minute = (i % 2) * 30;
      const timeLabel = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

      // Create a nice curve matching DailyBreakdownChart logic
      const x = i;
      const base = 20;
      const peak1 = 100 * Math.exp(-Math.pow(x - 20, 2) / 50);
      const peak2 = 80 * Math.exp(-Math.pow(x - 30, 2) / 50);

      const randomVal = seededRandom(seedBase + i * 13);
      const noise = randomVal * 15;

      const volume = Math.floor(base + peak1 + peak2 + noise);

      points.push({
        time: timeLabel,
        calls: Math.max(0, volume)
      });
    }
    return points;
}

interface DailyMetric {
  date: Date;
  sla: number;
  waitTime: number;
  occupancy: number;
  calls: number;
  weekStart: Date;
}

interface WeekGroup {
  weekStart: Date;
  weekEnd: Date;
  days: DailyMetric[];
  isExpanded: boolean;
}

export default function MetricsHistory() {
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'date', direction: 'desc' });
  const [expandedWeeks, setExpandedWeeks] = useState<Set<number>>(new Set());

  // Generate historical data only up to today
  const historyData = useMemo(() => {
    const data: DailyMetric[] = [];
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Start from a reasonable date (e.g., 12 weeks ago)
    const startDate = new Date(today);
    startDate.setDate(startDate.getDate() - (12 * 7)); // 12 weeks of data

    for (let d = new Date(startDate); d <= today; d.setDate(d.getDate() + 1)) {
      const currentDate = new Date(d);

      // Calculate week start (Monday)
      const day = currentDate.getDay();
      const diff = currentDate.getDate() - day + (day === 0 ? -6 : 1);
      const weekStart = new Date(currentDate);
      weekStart.setDate(diff);
      weekStart.setHours(0, 0, 0, 0);

      // Get daily breakdown data
      const dailyData = getDailyBreakdownData(currentDate);

      let totalCalls = 0;
      let totalServicedCalls = 0;
      let totalWaitTime = 0;
      let totalOccupancy = 0;
      let intervals = 0;

      // Calculate metrics from all 48 intervals
      for (let i = 0; i < dailyData.length; i++) {
        const calls = dailyData[i].calls;
        totalCalls += calls;

        // Simulate SLA achievement (88-98% range with variation)
        const slaSeed = currentDate.getTime() + i * 100;
        const slaRate = 0.88 + seededRandom(slaSeed) * 0.10;
        totalServicedCalls += calls * slaRate;

        // Simulate wait times (25-40s range)
        const waitSeed = currentDate.getTime() + i * 200;
        const waitTime = 25 + seededRandom(waitSeed) * 15;
        totalWaitTime += waitTime;

        // Simulate occupancy (80-90% range)
        const occupancySeed = currentDate.getTime() + i * 300;
        const occupancy = 80 + seededRandom(occupancySeed) * 10;
        totalOccupancy += occupancy;

        intervals++;
      }

      const avgSLA = (totalServicedCalls / totalCalls) * 100;
      const avgWaitTime = totalWaitTime / intervals;
      const avgOccupancy = totalOccupancy / intervals;

      data.push({
        date: currentDate,
        sla: Number(avgSLA.toFixed(1)),
        waitTime: Math.round(avgWaitTime),
        occupancy: Number(avgOccupancy.toFixed(1)),
        calls: totalCalls,
        weekStart: weekStart
      });
    }

    // Sort
    return data.sort((a, b) => {
      let aValue: any = a[sortConfig.key];
      let bValue: any = b[sortConfig.key];

      if (sortConfig.key === 'date') {
        aValue = a.date.getTime();
        bValue = b.date.getTime();
      }

      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [sortConfig]);

  const handleSort = (key: SortKey) => {
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const SortIcon = ({ columnKey }: { columnKey: SortKey }) => {
    if (sortConfig.key !== columnKey) return <div className="w-4 h-4" />; // spacer
    return sortConfig.direction === 'asc' ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />;
  };

  const toggleWeek = (weekStart: Date) => {
    const weekStartTimestamp = weekStart.getTime();
    setExpandedWeeks(current => {
      if (current.has(weekStartTimestamp)) {
        current.delete(weekStartTimestamp);
      } else {
        current.add(weekStartTimestamp);
      }
      return new Set(current);
    });
  };

  const weekGroups: WeekGroup[] = useMemo(() => {
    const groups: WeekGroup[] = [];
    const weekMap = new Map<number, WeekGroup>();

    for (const metric of historyData) {
      const weekStartTime = metric.weekStart.getTime();

      if (!weekMap.has(weekStartTime)) {
        const weekEnd = new Date(metric.weekStart);
        weekEnd.setDate(weekEnd.getDate() + 6);

        const group: WeekGroup = {
          weekStart: metric.weekStart,
          weekEnd: weekEnd,
          days: [],
          isExpanded: expandedWeeks.has(weekStartTime)
        };

        weekMap.set(weekStartTime, group);
        groups.push(group);
      }

      weekMap.get(weekStartTime)!.days.push(metric);
    }

    return groups;
  }, [historyData, expandedWeeks]);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 p-6 md:p-8 transition-colors duration-300">
      <style>
        {`
          /* Custom Scrollbar Styling */
          ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
          }
          ::-webkit-scrollbar-track {
            background: transparent;
          }
          ::-webkit-scrollbar-thumb {
            background: rgba(156, 163, 175, 0.5);
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.3);
          }
          ::-webkit-scrollbar-thumb:hover {
            background: rgba(156, 163, 175, 0.8);
          }
          /* For Firefox */
          * {
            scrollbar-width: thin;
            scrollbar-color: rgba(156, 163, 175, 0.5) transparent;
          }
        `}
      </style>
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex items-center space-x-4">
           <Link to="/" className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
           </Link>
           <div>
             <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Historical Metrics</h1>
             <p className="text-slate-500 dark:text-slate-400 text-sm">Daily performance data grouped by week. Click any week to view daily breakdown. Data updates daily as time progresses.</p>
           </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 flex flex-col h-[80vh]">
            <div className="overflow-auto flex-1 rounded-xl custom-scrollbar">
                <table className="w-full text-left text-sm relative">
                    <thead className="bg-slate-50 dark:bg-slate-900 text-slate-500 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-700 sticky top-0 z-10 shadow-sm">
                        <tr>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('date')}
                            >
                                <div className="flex items-center space-x-2">
                                    <Calendar className="w-4 h-4" />
                                    <span>Date</span>
                                    <SortIcon columnKey="date" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('sla')}
                            >
                                <div className="flex items-center space-x-2">
                                    <Target className="w-4 h-4" />
                                    <span>Service Level (SLA)</span>
                                    <SortIcon columnKey="sla" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('waitTime')}
                            >
                                <div className="flex items-center space-x-2">
                                    <Clock className="w-4 h-4" />
                                    <span>Avg. Wait Time</span>
                                    <SortIcon columnKey="waitTime" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('occupancy')}
                            >
                                <div className="flex items-center space-x-2">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Avg. Occupancy</span>
                                    <SortIcon columnKey="occupancy" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('calls')}
                            >
                                <div className="flex items-center space-x-2">
                                    <Phone className="w-4 h-4" />
                                    <span>Total Calls</span>
                                    <SortIcon columnKey="calls" />
                                </div>
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                        {weekGroups.map(group => {
                            const weekAvgSLA = (group.days.reduce((sum, day) => sum + day.sla, 0) / group.days.length).toFixed(1);
                            const weekAvgWait = Math.round(group.days.reduce((sum, day) => sum + day.waitTime, 0) / group.days.length);
                            const weekAvgOccupancy = (group.days.reduce((sum, day) => sum + day.occupancy, 0) / group.days.length).toFixed(1);
                            const weekTotalCalls = group.days.reduce((sum, day) => sum + day.calls, 0);

                            return (
                            <>
                                <tr
                                    key={`week-${group.weekStart.getTime()}`}
                                    className="hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors cursor-pointer bg-slate-50/50 dark:bg-slate-800/50"
                                    onClick={() => toggleWeek(group.weekStart)}
                                >
                                    <td className="px-6 py-4 font-bold text-slate-900 dark:text-slate-100">
                                        <div className="flex items-center space-x-2">
                                            {group.isExpanded ? (
                                                <ChevronDown className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                                            ) : (
                                                <ChevronRight className="w-4 h-4 text-slate-400" />
                                            )}
                                            <span>Week of {group.weekStart.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                            Number(weekAvgSLA) >= 90 ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300' :
                                            Number(weekAvgSLA) >= 80 ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300' :
                                            'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                                        }`}>
                                            {weekAvgSLA}%
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600 dark:text-slate-400">
                                        {weekAvgWait}s
                                    </td>
                                    <td className="px-6 py-4 text-slate-600 dark:text-slate-400">
                                        {weekAvgOccupancy}%
                                    </td>
                                    <td className="px-6 py-4 text-slate-900 dark:text-slate-100 font-bold">
                                        {weekTotalCalls.toLocaleString()}
                                    </td>
                                </tr>
                                {group.isExpanded && (
                                    <tr>
                                        <td colSpan={5} className="px-0 py-0 bg-slate-50/30 dark:bg-slate-900/30">
                                            <table className="w-full text-left text-sm">
                                                <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                                                    {group.days.map(day => (
                                                        <tr key={day.date.getTime()} className="hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors">
                                                            <td className="py-3 font-medium text-slate-700 dark:text-slate-300" style={{ width: '20%', paddingLeft: '3.5rem' }}>
                                                                {day.date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                                                            </td>
                                                            <td className="px-6 py-3" style={{ width: '20%' }}>
                                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                                    Number(day.sla) >= 90 ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300' :
                                                                    Number(day.sla) >= 80 ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300' :
                                                                    'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                                                                }`}>
                                                                    {day.sla}%
                                                                </span>
                                                            </td>
                                                            <td className="px-6 py-3 text-slate-600 dark:text-slate-400" style={{ width: '20%' }}>
                                                                {day.waitTime}s
                                                            </td>
                                                            <td className="px-6 py-3 text-slate-600 dark:text-slate-400" style={{ width: '20%' }}>
                                                                {day.occupancy}%
                                                            </td>
                                                            <td className="px-6 py-3 text-slate-900 dark:text-slate-100 font-bold" style={{ width: '20%' }}>
                                                                {day.calls.toLocaleString()}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </td>
                                    </tr>
                                )}
                            </>
                        )})}
                        {historyData.length > 500 && (
                            <tr>
                                <td colSpan={5} className="px-6 py-4 text-center text-slate-500 dark:text-slate-400 italic">
                                    Showing first 500 of {historyData.length} records. Filter to see more.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
      </div>
    </div>
  );
}
