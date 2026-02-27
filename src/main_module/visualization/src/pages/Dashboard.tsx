import React, { useState, useEffect } from 'react';
import { MetricCard } from '../components/MetricCard';
import { ForecastingSection } from '../components/ForecastingSection';
import { SimulationPanel } from '../components/SimulationPanel';
import { Clock, Target, TrendingUp, Trophy, ArrowRight } from 'lucide-react';
import { Link } from 'react-router';
import { ThemeToggle } from '../components/ThemeToggle';
import { fetchMetrics, DayMetrics } from '../lib/api';

// Helper function to replicate the volume logic from DemandChart
function seededRandom(seed: number) {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
}

function getVolumeForDate(date: Date) {
    const isWeekend = date.getDay() === 0 || date.getDay() === 6;
    const baseVolume = isWeekend ? 6000 : 12500;
    const seed = date.getFullYear() * 10000 + (date.getMonth() + 1) * 100 + date.getDate();
    const randomVal = seededRandom(seed);
    const noise = randomVal * 2000 - 1000;
    const dailyTotal = Math.max(4000, Math.floor(baseVolume + noise));
    const estimatedHourly = Math.round(dailyTotal * 0.12);
    return estimatedHourly;
}

// Generate daily breakdown data (48 half-hour slots)
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

export default function Dashboard() {
  const [selectedDayVolume, setSelectedDayVolume] = useState<number | undefined>(undefined);
  const [metrics, setMetrics] = useState<DayMetrics | null>(null);

  // State lifted from ForecastingSection
  const [currentWeekStart, setCurrentWeekStart] = useState(() => {
    const now = new Date();
    // Start of current week (Monday)
    const day = now.getDay();
    const diff = now.getDate() - day + (day === 0 ? -6 : 1);
    const monday = new Date(now.setDate(diff));
    monday.setHours(0, 0, 0, 0);
    return monday;
  });

  const [selectedDay, setSelectedDay] = useState(new Date());

  // Initialize selectedDayVolume with today's volume on mount
  useEffect(() => {
    const today = new Date();
    const vol = getVolumeForDate(today);
    setSelectedDayVolume(vol);
  }, []);

  // Fetch real metrics from backend whenever selected day changes
  useEffect(() => {
    fetchMetrics(selectedDay)
      .then(setMetrics)
      .catch(console.error);
  }, [selectedDay]);

  const handlePrevWeek = () => {
    const prev = new Date(currentWeekStart);
    prev.setDate(prev.getDate() - 7);
    setCurrentWeekStart(prev);
  };

  const handleNextWeek = () => {
    const next = new Date(currentWeekStart);
    next.setDate(next.getDate() + 7);
    setCurrentWeekStart(next);
  };

  const handleJumpToToday = () => {
    const now = new Date();
    const day = now.getDay();
    const diff = now.getDate() - day + (day === 0 ? -6 : 1);
    const monday = new Date(now.setDate(diff));
    monday.setHours(0, 0, 0, 0);

    setCurrentWeekStart(monday);
    setSelectedDay(new Date());

    // Update volume for today
    const vol = getVolumeForDate(new Date());
    setSelectedDayVolume(vol);
  };

  const handleSelectDate = (date: Date, volume: number) => {
    setSelectedDay(date);
    setSelectedDayVolume(volume);
  };

  const handlePrevDay = () => {
    const prev = new Date(selectedDay);
    prev.setDate(prev.getDate() - 1);
    setSelectedDay(prev);

    // Check if we need to switch week view
    const currentWeekEnd = new Date(currentWeekStart);
    currentWeekEnd.setDate(currentWeekStart.getDate() + 6);

    if (prev < currentWeekStart) {
        handlePrevWeek();
    }

    // Update volume
    const vol = getVolumeForDate(prev);
    setSelectedDayVolume(vol);
  };

  const handleNextDay = () => {
    const next = new Date(selectedDay);
    next.setDate(next.getDate() + 1);
    setSelectedDay(next);

    // Check if we need to switch week view
    const currentWeekEnd = new Date(currentWeekStart);
    currentWeekEnd.setDate(currentWeekStart.getDate() + 6);
    currentWeekEnd.setHours(23, 59, 59, 999);

    if (next > currentWeekEnd) {
        handleNextWeek();
    }

    // Update volume
    const vol = getVolumeForDate(next);
    setSelectedDayVolume(vol);
  };

  const handleReset = () => {
      handleJumpToToday();
  };

  // Generate daily breakdown data for selected day
  const dailyBreakdownData = getDailyBreakdownData(selectedDay);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 relative transition-colors duration-300">
      <div className="transition-all duration-300 relative z-10">
        <main className="p-6 md:p-8 max-w-7xl mx-auto space-y-8">
          <div className="flex flex-col sm:flex-row justify-between items-start space-y-4 sm:space-y-0">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Intuit Call Center Dashboard</h1>
              <p className="text-slate-500 dark:text-slate-400 mt-1">Real-time insights and workforce performance metrics.</p>
            </div>
            <div className="flex items-center -mt-1">
                <ThemeToggle />
            </div>
          </div>

          <div className="relative">
             <div className="absolute top-0 right-0 -mt-10 sm:-mt-12">
                <Link to="/metrics-history" className="text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 flex items-center space-x-1">
                   <span>View All History</span>
                   <ArrowRight className="w-4 h-4" />
                </Link>
             </div>

             <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                <MetricCard
                  title="Service Level (SLA)"
                  value={metrics ? `${metrics.avg_sla_compliance.toFixed(1)}%` : '—'}
                  change=""
                  isPositive={true}
                  icon={Target}
                />
                <MetricCard
                  title="Avg. Waiting Time"
                  value={metrics ? `${Math.round(metrics.avg_wait_time)}s` : '—'}
                  change=""
                  isPositive={true}
                  icon={Clock}
                />
                <MetricCard
                  title="Total Calls Processed"
                  value={metrics ? metrics.total_calls.toLocaleString() : '—'}
                  change=""
                  isPositive={true}
                  icon={TrendingUp}
                />
             </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 min-w-0 h-full">
              <ForecastingSection
                weekStart={currentWeekStart}
                onPrevWeek={handlePrevWeek}
                onNextWeek={handleNextWeek}
                selectedDate={selectedDay}
                onSelectDate={handleSelectDate}
                onJumpToToday={handleJumpToToday}
                onPrevDay={handlePrevDay}
                onNextDay={handleNextDay}
              />
            </div>

            <div className="h-full min-w-0">
              <SimulationPanel
                initialCallVolume={selectedDayVolume}
                onReset={handleReset}
                selectedDate={selectedDay}
                dailyBreakdownData={dailyBreakdownData}
              />
            </div>
          </div>

          <div
             style={{
                background: 'rgba(255, 255, 255, 0.10)',
                boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.10), 0 1px 2px -1px rgba(0, 0, 0, 0.10)'
             }}
             className="backdrop-blur-xl rounded-xl overflow-hidden border border-slate-100 dark:border-slate-700 dark:bg-slate-800/50"
          >
             <div className="p-6 border-b border-slate-100/30 dark:border-slate-800 flex justify-between items-center">
                <div className="flex items-center space-x-2">
                  <div className="p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
                    <Trophy className="w-5 h-5 text-yellow-600 dark:text-yellow-500" />
                  </div>
                  <h2 className="text-lg font-bold text-slate-900 dark:text-white">Top Performing Agents</h2>
                </div>
             </div>
             <div className="overflow-x-auto">
               <table className="w-full text-left text-sm">
                 <thead className="bg-slate-50/50 dark:bg-slate-800/50 text-slate-500 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-700">
                   <tr>
                     <th className="px-6 py-4">Agent Name</th>
                     <th className="px-6 py-4">Status</th>
                     <th className="px-6 py-4">Calls Taken</th>
                     <th className="px-6 py-4">Average Handle Time</th>
                     <th className="px-6 py-4">Service Level (SLA)</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-slate-100/50 dark:divide-slate-700/50">
                   {[
                     { name: 'Jackie Wang', status: 'Online', calls: 45, aht: '3m 12s', sla: '98%' },
                     { name: 'Sarah He', status: 'In Call', calls: 38, aht: '2m 55s', sla: '96%' },
                     { name: 'Hao Zhang', status: 'Break', calls: 41, aht: '3m 05s', sla: '94%' },
                     { name: 'Sophia Fang', status: 'Online', calls: 32, aht: '3m 40s', sla: '97%' },
                   ].map((agent, idx) => (
                     <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-700/30 transition-colors">
                       <td className="px-6 py-4 font-medium text-slate-900 dark:text-white">{agent.name}</td>
                       <td className="px-6 py-4">
                         <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                           agent.status === 'Online' ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400' :
                           agent.status === 'In Call' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400' :
                           'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-400'
                         }`}>
                           {agent.status}
                         </span>
                       </td>
                       <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.calls}</td>
                       <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.aht}</td>
                       <td className="px-6 py-4 text-slate-900 dark:text-white font-bold">{agent.sla}</td>
                     </tr>
                   ))}
                 </tbody>
               </table>
             </div>
          </div>
        </main>
      </div>
    </div>
  );
}
