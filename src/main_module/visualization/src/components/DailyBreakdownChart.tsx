import React, { useMemo, useState, useEffect } from 'react';
import { ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Customized, Rectangle } from 'recharts';
import { Activity, ChevronLeft, ChevronRight, Phone, Users } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
// import { fetchForecast } from '../lib/api'; // Unused — chart uses staffingData prop instead

interface DailyBreakdownChartProps {
  selectedDate: Date;
  onPrevDay: () => void;
  onNextDay: () => void;
  onJumpToToday: () => void;
  staffingData?: Array<{ time: string; calls: number; agents: number }>;
  currentCallVolume?: number;
  currentRequiredAgents?: number;
  peakCallVolume?: number;
  peakRequiredAgents?: number;
  currentTimeSlot?: string;
  isToday?: boolean;
}

export function DailyBreakdownChart({ selectedDate, onPrevDay, onNextDay, onJumpToToday, staffingData, currentCallVolume, currentRequiredAgents, peakCallVolume, peakRequiredAgents, currentTimeSlot, isToday }: DailyBreakdownChartProps) {
  const { theme } = useTheme();

  // Check if selected date is today
  const isTodayMemo = useMemo(() => {
    const today = new Date();
    return selectedDate.getDate() === today.getDate() &&
           selectedDate.getMonth() === today.getMonth() &&
           selectedDate.getFullYear() === today.getFullYear();
  }, [selectedDate]);

  // Clicked slot — overrides the stat cards when user clicks a point on the chart
  const [clickedSlot, setClickedSlot] = useState<{ time: string; calls: number; agents: number } | null>(null);

  // Reset clicked slot when date changes
  useEffect(() => { setClickedSlot(null); }, [staffingData]);

  const handleChartClick = (data: any) => {
    if (data && data.activePayload && data.activePayload.length > 0) {
      const point = data.activePayload[0].payload;
      setClickedSlot({ time: point.time, calls: point.calls, agents: point.agents });
    }
  };

  // Unused — chart renders from staffingData prop, not local fetch
  // const [data, setData] = useState<{ time: string; calls: number }[]>([]);

  // Unused — chart renders from staffingData prop, not local fetch
  // useEffect(() => {
  //   fetchForecast(selectedDate)
  //     .then(slots => setData(slots.map(s => ({ time: s.time, calls: s.predicted_calls }))))
  //     .catch(console.error);
  // }, [selectedDate]);

  const formatDate = (date: Date) => {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const dateString = date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    const isToday = date.toDateString() === today.toDateString();

    return { dateString, isToday };
  };

  const isDark = theme === 'dark';

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-slate-800 p-3 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700">
          <p className="text-sm font-medium text-slate-900 dark:text-white mb-2">{payload[0].payload.time}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: <span className="font-bold">{entry.value}</span>
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="flex flex-col min-w-0">
      <style>{`
        .recharts-wrapper {
          outline: none !important;
        }
      `}</style>

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white">Daily Demand & Staffing Forecast</h3>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-sm text-slate-500 dark:text-slate-400">
                {formatDate(selectedDate).dateString}
              </span>
              {formatDate(selectedDate).isToday && (
                <span className="text-[10px] leading-none bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 px-1.5 py-0.5 rounded font-medium">
                  Today
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {!isTodayMemo && (
            <button
              onClick={onJumpToToday}
              className="p-1.5 hover:bg-slate-100/50 dark:hover:bg-slate-700/50 rounded-md transition-all text-blue-600 dark:text-blue-400 flex items-center space-x-1 mr-1 bg-slate-50/50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              title="Jump to Today"
            >
              <span className="text-xs font-semibold px-1">Today</span>
            </button>
          )}
          <button
            onClick={onPrevDay}
            className="p-1.5 hover:bg-slate-100/50 dark:hover:bg-slate-700/50 rounded-md transition-all text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 border border-transparent hover:border-slate-200 dark:hover:border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            aria-label="Previous day"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={onNextDay}
            className="p-1.5 hover:bg-slate-100/50 dark:hover:bg-slate-700/50 rounded-md transition-all text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 border border-transparent hover:border-slate-200 dark:hover:border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            aria-label="Next day"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Current Stats Display — clicking the chart overrides with that slot's data */}
      {currentCallVolume !== undefined && currentRequiredAgents !== undefined && (
        <div className="flex justify-center mb-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4" style={{ width: 'calc(100% - 130px)' }}>
            {/* Current Call Volume */}
            <div className="bg-blue-50/50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200/50 dark:border-blue-700/50">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Phone className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    {clickedSlot ? 'Call Volume' : isToday ? 'Current Call Volume' : 'Peak Call Volume'}
                  </span>
                </div>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {clickedSlot ? clickedSlot.time : currentTimeSlot}
                </span>
              </div>
              <div className="flex items-baseline space-x-1">
                <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                  {clickedSlot ? clickedSlot.calls : currentCallVolume}
                </p>
                <span className="text-sm text-slate-500 dark:text-slate-400">calls/30min</span>
              </div>
              {peakCallVolume !== undefined && (
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Daily peak: {peakCallVolume}</p>
              )}
            </div>

            {/* Required Agents */}
            <div className="bg-purple-50/50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200/50 dark:border-purple-700/50">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Users className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    {clickedSlot ? 'Required Agents' : isToday ? 'Required Agents' : 'Peak Required Agents'}
                  </span>
                </div>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {clickedSlot ? clickedSlot.time : currentTimeSlot}
                </span>
              </div>
              <div className="flex items-baseline space-x-1">
                <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                  {clickedSlot ? clickedSlot.agents : currentRequiredAgents}
                </p>
                <span className="text-sm text-slate-500 dark:text-slate-400">agents</span>
              </div>
              {peakRequiredAgents !== undefined && (
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Daily peak: {peakRequiredAgents}</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Chart — guard prevents Recharts NaN errors when data hasn't loaded yet */}
      {(!staffingData || staffingData.length === 0) ? (
        <div className="w-full flex items-center justify-center text-slate-400 dark:text-slate-500 text-sm" style={{ height: '350px' }}>
          Loading forecast data...
        </div>
      ) : (
      <div className="w-full" style={{ height: '350px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={staffingData}
            margin={{ top: 20, right: 30, left: 10, bottom: 0 }}
            onClick={handleChartClick}
            style={{ cursor: 'pointer' }}
          >
            <defs>
              <linearGradient id="colorCalls" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={isDark ? '#60a5fa' : '#3b82f6'} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={isDark ? '#60a5fa' : '#3b82f6'} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={isDark ? '#334155' : '#e2e8f0'}
              vertical={false}
            />
            <XAxis
              dataKey="time"
              stroke={isDark ? '#64748b' : '#94a3b8'}
              tick={{ fill: isDark ? '#94a3b8' : '#64748b', fontSize: 12 }}
              tickFormatter={(value, index) => {
                // Show fewer labels to avoid crowding
                if (index % 4 === 0) {
                  return value;
                }
                return '';
              }}
            />
            {/* Left Y-axis for calls */}
            <YAxis
              yAxisId="left"
              stroke={isDark ? '#60a5fa' : '#3b82f6'}
              tick={{ fill: isDark ? '#60a5fa' : '#3b82f6', fontSize: 12 }}
              label={{
                value: 'Calls',
                angle: -90,
                position: 'insideLeft',
                style: { fill: isDark ? '#60a5fa' : '#3b82f6', fontSize: 12 }
              }}
            />
            {/* Right Y-axis for agents */}
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke={isDark ? '#a78bfa' : '#8b5cf6'}
              tick={{ fill: isDark ? '#a78bfa' : '#8b5cf6', fontSize: 12 }}
              label={{
                value: 'Agents',
                angle: 90,
                position: 'insideRight',
                style: { fill: isDark ? '#a78bfa' : '#8b5cf6', fontSize: 12 }
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '10px' }}
              iconType="line"
            />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="calls"
              name="Call Volume"
              stroke={isDark ? '#60a5fa' : '#3b82f6'}
              strokeWidth={2}
              fill="url(#colorCalls)"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="agents"
              name="Required Agents"
              stroke={isDark ? '#a78bfa' : '#8b5cf6'}
              strokeWidth={2.5}
              dot={false}
            />
            {/* Vertical line showing current time (only when viewing today) */}
            {isToday && currentTimeSlot && (
              <Customized
                component={(props: any) => {
                  const { xAxisMap, yAxisMap, offset } = props;
                  if (!xAxisMap || !xAxisMap[0]) return null;
                  const xAxis = xAxisMap[0];
                  const yAxis = yAxisMap?.left;
                  const dataIndex = staffingData?.findIndex(d => d.time === currentTimeSlot);
                  if (dataIndex === undefined || dataIndex < 0) return null;

                  const bandWidth = (xAxis.width) / (staffingData?.length || 1);
                  const x = xAxis.x + dataIndex * bandWidth + bandWidth / 2;
                  const yStart = offset?.top || 20;
                  const yEnd = yAxis ? (yAxis.y + yAxis.height) : (yStart + 300);

                  return (
                    <g>
                      <line
                        x1={x}
                        y1={yStart}
                        x2={x}
                        y2={yEnd}
                        stroke={isDark ? '#f97316' : '#ea580c'}
                        strokeWidth={2}
                        strokeDasharray="4 4"
                      />
                      <text
                        x={x + 4}
                        y={yStart + 12}
                        fill={isDark ? '#f97316' : '#ea580c'}
                        fontSize={11}
                        fontWeight={600}
                      >
                        Now
                      </text>
                    </g>
                  );
                }}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      )}
    </div>
  );
}
