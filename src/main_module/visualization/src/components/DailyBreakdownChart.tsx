import React, { useMemo, useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Activity, ChevronLeft, ChevronRight } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

interface DailyBreakdownChartProps {
  selectedDate: Date;
  onPrevDay: () => void;
  onNextDay: () => void;
  onJumpToToday: () => void;
}

// Simple seeded random function
function seededRandom(seed: number) {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
}

export function DailyBreakdownChart({ selectedDate, onPrevDay, onNextDay, onJumpToToday }: DailyBreakdownChartProps) {
  const { theme } = useTheme();

  // Check if selected date is today
  const isToday = useMemo(() => {
    const today = new Date();
    return selectedDate.getDate() === today.getDate() &&
           selectedDate.getMonth() === today.getMonth() &&
           selectedDate.getFullYear() === today.getFullYear();
  }, [selectedDate]);

  // Current time state for the reference line
  const [currentTimeLabel, setCurrentTimeLabel] = useState<string | null>(null);

  useEffect(() => {
    if (isToday) {
        const updateTime = () => {
            const now = new Date();
            const hour = now.getHours();
            const minute = now.getMinutes();
            // Round to nearest 30 min interval
            const roundedMinute = minute < 15 ? 0 : minute < 45 ? 30 : 60;
            let finalHour = hour;
            let finalMinute = roundedMinute;

            if (finalMinute === 60) {
                finalHour += 1;
                finalMinute = 0;
            }
            if (finalHour > 23) {
                setCurrentTimeLabel(null); // End of day
                return;
            }

            const label = `${finalHour.toString().padStart(2, '0')}:${finalMinute.toString().padStart(2, '0')}`;
            setCurrentTimeLabel(label);
        };

        updateTime();
        const interval = setInterval(updateTime, 60000); // Update every minute
        return () => clearInterval(interval);
    } else {
        setCurrentTimeLabel(null);
    }
  }, [isToday]);


  const data = useMemo(() => {
    const points = [];

    // Create a seed based on the full date (YYYYMMDD) to ensure consistency per day
    const seedBase = selectedDate.getFullYear() * 10000 + (selectedDate.getMonth() + 1) * 100 + selectedDate.getDate();

    // Generate 48 points (every 30 mins)
    for (let i = 0; i < 48; i++) {
      const hour = Math.floor(i / 2);
      const minute = (i % 2) * 30;
      const timeLabel = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

      // Create a nice curve
      const x = i;
      const base = 20;
      const peak1 = 100 * Math.exp(-Math.pow(x - 20, 2) / 50);
      const peak2 = 80 * Math.exp(-Math.pow(x - 30, 2) / 50);

      // Seeded random noise - making sure it varies per day by using seedBase
      // also vary per time slot (i)
      const randomVal = seededRandom(seedBase + i * 13);
      const noise = randomVal * 15; // Increased noise for more visible difference

      const volume = Math.floor(base + peak1 + peak2 + noise);

      points.push({
        time: timeLabel,
        calls: Math.max(0, volume)
      });
    }
    return points;
  }, [selectedDate]);

  return (
    <div
        style={{
            background: 'rgba(255, 255, 255, 0.10)',
            boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.10), 0 1px 2px -1px rgba(0, 0, 0, 0.10)'
        }}
        className="backdrop-blur-xl p-6 rounded-xl flex flex-col min-w-0 border border-slate-100 dark:border-slate-700 dark:bg-slate-800/50"
    >
      <style>{`
        .recharts-wrapper {
          outline: none !important;
        }
      `}</style>
      <div className="mb-6 shrink-0 flex items-center justify-between">
        <div className="flex items-center space-x-2">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
                <div className="flex items-center space-x-2">
                    <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center">
                        Daily Demand Forecast
                    </h2>
                    {isToday && (
                        <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 px-2 py-0.5 rounded-full font-medium">
                            Today
                        </span>
                    )}
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                    {selectedDate.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
                </p>
            </div>
        </div>

        <div className="flex items-center space-x-1">
             <button
                onClick={onJumpToToday}
                className="p-1.5 hover:bg-slate-100/50 dark:hover:bg-slate-700/50 rounded-md transition-all text-blue-600 dark:text-blue-400 flex items-center space-x-1 mr-1 bg-slate-50/50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                title="Jump to Today"
            >
                <span className="text-xs font-semibold px-1">Today</span>
            </button>
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

      <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer width="100%" height={250} minWidth={0}>
            <AreaChart
              data={data}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
              className="outline-none"
            >
              <defs>
                <linearGradient id="colorCallsDaily" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0EA5E9" stopOpacity={0.2}/>
                  <stop offset="95%" stopColor="#0EA5E9" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" strokeOpacity={0.2} />
              <XAxis
                dataKey="time"
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#64748B', fontSize: 10 }}
                interval={5}
                dy={10}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#64748B', fontSize: 12 }}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                    return (
                        <div className={`p-3 rounded-lg shadow-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-white'}`}>
                            <p className={`text-xs font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>{label}</p>
                            <p className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                                {payload[0].value} <span className="text-xs font-normal text-slate-500 dark:text-slate-400">calls</span>
                            </p>
                        </div>
                    );
                    }
                    return null;
                }}
              />
              <Area
                type="monotone"
                dataKey="calls"
                stroke="#0EA5E9"
                fillOpacity={1}
                fill="url(#colorCallsDaily)"
                strokeWidth={2}
                animationDuration={1500}
                isAnimationActive={true}
              />
              {currentTimeLabel && (
                  <ReferenceLine x={currentTimeLabel} stroke="#EF4444" strokeDasharray="3 3" />
              )}
            </AreaChart>
          </ResponsiveContainer>
      </div>
    </div>
  );
}
