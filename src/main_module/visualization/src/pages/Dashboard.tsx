import { useState, useEffect, useCallback } from 'react';
import { MetricCard } from '../components/MetricCard';
import { WorkforceManagementPanel } from '../components/WorkforceManagementPanel';
import { DemandForecastChart } from '../components/DemandChart';
import { Clock, Target, TrendingUp, Trophy, ArrowRight, Phone, Calendar, CalendarClock, Loader2, Route, Search } from 'lucide-react';
import { Link } from 'react-router';
import { ThemeToggle } from '../components/ThemeToggle';
import { fetchMetrics, fetchApi, fetchRoutingRecommendations, DayMetrics, RoutingRecommendation } from '../lib/api';

interface TopAgent {
  expert_id: string;
  name: string;
  segment: string;
  contacts: number;
  aht: string;
  utilization: number;
  resolution_rate: number;
  composite_score: number;
}

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
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [topAgents, setTopAgents] = useState<TopAgent[]>([]);
  const [agentsLoading, setAgentsLoading] = useState(true);
  const [routingProduct, setRoutingProduct] = useState('');
  const [routingChannel, setRoutingChannel] = useState('');
  const [routingLoading, setRoutingLoading] = useState(false);
  const [routingRecommendations, setRoutingRecommendations] = useState<RoutingRecommendation[]>([]);
  const [routingError, setRoutingError] = useState<string | null>(null);

  const handleFetchRouting = async () => {
    setRoutingLoading(true);
    setRoutingError(null);
    setRoutingRecommendations([]);
    try {
      const now = new Date();
      const datetimeUtc = now.toISOString().slice(0, 16);
      const { recommendations } = await fetchRoutingRecommendations({
        product_group_sku: routingProduct || undefined,
        channel: routingChannel || undefined,
        datetime_utc: datetimeUtc,
        top_n: 5,
      });
      setRoutingRecommendations(recommendations);
    } catch (err) {
      setRoutingError(err instanceof Error ? err.message : 'Failed to load recommendations');
    } finally {
      setRoutingLoading(false);
    }
  };

  // State for simulator targets
  const [simulatorTargets, setSimulatorTargets] = useState({
    sla: 90,
    waitTime: 30,
    occupancy: 80
  });

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


  useEffect(() => {
    let cancelled = false;
    setMetricsLoading(true);
    setMetricsError(null);
    fetchMetrics(selectedDay)
      .then((m) => { if (!cancelled) setMetrics(m); })
      .catch((err) => {
        if (!cancelled) {
          setMetricsError(err instanceof Error ? err.message : 'Failed to load metrics');
        }
      })
      .finally(() => { if (!cancelled) setMetricsLoading(false); });
    return () => { cancelled = true; };
  }, [selectedDay]);

  useEffect(() => {
    const fetchTopAgents = async () => {
      try {
        setAgentsLoading(true);
        const response = await fetchApi('/api/agents?n=5&sort_by=composite_score');
        if (response.ok) {
          const data = await response.json();
          setTopAgents(data);
        }
      } catch (err) {
        console.error('Failed to fetch top agents:', err);
      } finally {
        setAgentsLoading(false);
      }
    };
    fetchTopAgents();
  }, []);

  // Initialize selectedDayVolume with today's volume on mount
  useEffect(() => {
    const today = new Date();
    const vol = getVolumeForDate(today);
    setSelectedDayVolume(vol);
  }, []);

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

  // Get current 30-minute time slot
  const getCurrentTimeSlot = () => {
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
      return { hour: 23, minute: 30, index: 47 };
    }

    const index = finalHour * 2 + (finalMinute === 30 ? 1 : 0);
    return { hour: finalHour, minute: finalMinute, index };
  };

  // State for current time slot (updates every 30 minutes)
  const [currentTimeSlot, setCurrentTimeSlot] = useState(getCurrentTimeSlot());

  useEffect(() => {
    const updateTimeSlot = () => {
      setCurrentTimeSlot(getCurrentTimeSlot());
    };

    // Update immediately
    updateTimeSlot();

    // Update every minute (will recalculate if we've crossed a 30-min boundary)
    const interval = setInterval(updateTimeSlot, 60000);

    return () => clearInterval(interval);
  }, []);

  // Generate daily breakdown data for selected day
  const dailyBreakdownData = getDailyBreakdownData(selectedDay);

  // Calculate weekly metrics based on currentWeekStart
  const calculateWeeklyMetrics = () => {
    const now = new Date();
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Determine if currentWeekStart is the current week
    const currentWeekMonday = new Date();
    const day = currentWeekMonday.getDay();
    const diff = currentWeekMonday.getDate() - day + (day === 0 ? -6 : 1);
    const thisWeekMonday = new Date(currentWeekMonday.setDate(diff));
    thisWeekMonday.setHours(0, 0, 0, 0);

    const isCurrentWeek = currentWeekStart.getTime() === thisWeekMonday.getTime();
    const isFutureWeek = currentWeekStart > thisWeekMonday;

    // Target metrics (used for future weeks and as comparison)
    const TARGET_SLA = simulatorTargets.sla;
    const TARGET_WAIT_TIME = simulatorTargets.waitTime;
    const TARGET_OCCUPANCY = simulatorTargets.occupancy;

    if (isFutureWeek) {
      // For future weeks, show target metrics and forecasted calls
      const weekEnd = new Date(currentWeekStart);
      weekEnd.setDate(weekEnd.getDate() + 6);

      let forecastedTotalCalls = 0;
      for (let i = 0; i < 7; i++) {
        const day = new Date(currentWeekStart);
        day.setDate(day.getDate() + i);
        const isWeekend = day.getDay() === 0 || day.getDay() === 6;
        const baseVolume = isWeekend ? 6000 : 12500;
        const seed = day.getFullYear() * 10000 + (day.getMonth() + 1) * 100 + day.getDate();
        const randomVal = seededRandom(seed);
        const noise = randomVal * 2000 - 1000;
        forecastedTotalCalls += Math.max(4000, Math.floor(baseVolume + noise));
      }

      return {
        sla: `${TARGET_SLA}%`,
        slaChange: 'Target',
        slaPositive: true,
        isFuture: true,
        isCurrentWeek: false,
        targetSLA: `Target: ${TARGET_SLA}%`,
        prevWeekSLA: '',

        waitTime: `${TARGET_WAIT_TIME}s`,
        waitChange: 'Target',
        waitPositive: true,
        targetWaitTime: `Target: ${TARGET_WAIT_TIME}s`,
        prevWeekWait: '',

        occupancy: `${TARGET_OCCUPANCY}%`,
        occupancyChange: 'Target',
        occupancyPositive: true,
        targetOccupancy: `Target: ${TARGET_OCCUPANCY}%`,
        prevWeekOccupancy: '',

        totalCalls: forecastedTotalCalls.toLocaleString(),
        callsChange: 'Forecast',
        callsPositive: true,
        targetCalls: '',
        prevWeekCalls: ''
      };
    }

    if (isCurrentWeek) {
      // Week-to-date calculation
      const currentDayOfWeek = now.getDay();
      const daysElapsed = currentDayOfWeek === 0 ? 6 : currentDayOfWeek - 1; // Monday = 0, Sunday = 6

      let wtdTotalCalls = 0;
      let wtdServicedCalls = 0;
      let wtdWaitTimeSum = 0;
      let wtdOccupancySum = 0;
      let wtdIntervals = 0;

      // Calculate up to current time slot on current day
      for (let dayIndex = 0; dayIndex <= daysElapsed; dayIndex++) {
        const day = new Date(currentWeekStart);
        day.setDate(day.getDate() + dayIndex);
        const dailyData = getDailyBreakdownData(day);

        const isToday = dayIndex === daysElapsed;
        const maxInterval = isToday ? currentTimeSlot.index + 1 : 48;

        for (let i = 0; i < maxInterval; i++) {
          const calls = dailyData[i].calls;
          wtdTotalCalls += calls;

          // Simulate SLA achievement (90-95% range with variation)
          const slaSeed = day.getTime() + i * 100;
          const slaRate = 0.88 + seededRandom(slaSeed) * 0.10; // 88-98%
          wtdServicedCalls += calls * slaRate;

          // Simulate wait times (25-40s range)
          const waitSeed = day.getTime() + i * 200;
          const waitTime = 25 + seededRandom(waitSeed) * 15;
          wtdWaitTimeSum += waitTime;

          // Simulate occupancy (80-90% range)
          const occupancySeed = day.getTime() + i * 300;
          const occupancy = 80 + seededRandom(occupancySeed) * 10;
          wtdOccupancySum += occupancy;

          wtdIntervals++;
        }
      }

      const wtdSLA = (wtdServicedCalls / wtdTotalCalls) * 100;
      const wtdAvgWait = wtdWaitTimeSum / wtdIntervals;
      const wtdAvgOccupancy = wtdOccupancySum / wtdIntervals;

      // Calculate previous week same period for comparison
      const prevWeekStart = new Date(currentWeekStart);
      prevWeekStart.setDate(prevWeekStart.getDate() - 7);

      let prevWtdTotalCalls = 0;
      let prevWtdServicedCalls = 0;
      let prevWtdWaitTimeSum = 0;
      let prevWtdOccupancySum = 0;
      let prevWtdIntervals = 0;

      for (let dayIndex = 0; dayIndex <= daysElapsed; dayIndex++) {
        const day = new Date(prevWeekStart);
        day.setDate(day.getDate() + dayIndex);
        const dailyData = getDailyBreakdownData(day);

        const isLastDay = dayIndex === daysElapsed;
        const maxInterval = isLastDay ? currentTimeSlot.index + 1 : 48;

        for (let i = 0; i < maxInterval; i++) {
          const calls = dailyData[i].calls;
          prevWtdTotalCalls += calls;

          const slaSeed = day.getTime() + i * 100;
          const slaRate = 0.88 + seededRandom(slaSeed) * 0.10;
          prevWtdServicedCalls += calls * slaRate;

          const waitSeed = day.getTime() + i * 200;
          const waitTime = 25 + seededRandom(waitSeed) * 15;
          prevWtdWaitTimeSum += waitTime;

          const occupancySeed = day.getTime() + i * 300;
          const occupancy = 80 + seededRandom(occupancySeed) * 10;
          prevWtdOccupancySum += occupancy;

          prevWtdIntervals++;
        }
      }

      const prevWtdSLA = (prevWtdServicedCalls / prevWtdTotalCalls) * 100;
      const prevWtdAvgWait = prevWtdWaitTimeSum / prevWtdIntervals;
      const prevWtdAvgOccupancy = prevWtdOccupancySum / prevWtdIntervals;

      const slaChange = wtdSLA - prevWtdSLA;
      const waitChange = wtdAvgWait - prevWtdAvgWait;
      const occupancyChange = wtdAvgOccupancy - prevWtdAvgOccupancy;
      const callsChange = wtdTotalCalls - prevWtdTotalCalls;

      return {
        sla: `${wtdSLA.toFixed(1)}%`,
        slaChange: `${slaChange >= 0 ? '+' : ''}${slaChange.toFixed(1)}%`,
        slaPositive: slaChange >= 0,
        isFuture: false,
        isCurrentWeek: true,
        targetSLA: `Target: ${TARGET_SLA}%`,
        // prevWeekSLA: `${prevWtdSLA.toFixed(1)}%`,  // TODO: wire to real backend data
        prevWeekSLA: '0',

        waitTime: `${Math.round(wtdAvgWait)}s`,
        waitChange: `${waitChange >= 0 ? '+' : ''}${Math.round(waitChange)}s`,
        waitPositive: waitChange <= 0,
        targetWaitTime: `Target: ${TARGET_WAIT_TIME}s`,
        // prevWeekWait: `${Math.round(prevWtdAvgWait)}s`,  // TODO: wire to real backend data
        prevWeekWait: '0',

        occupancy: `${wtdAvgOccupancy.toFixed(1)}%`,
        occupancyChange: `${occupancyChange >= 0 ? '+' : ''}${occupancyChange.toFixed(1)}%`,
        occupancyPositive: occupancyChange >= 0,
        targetOccupancy: `Target: ${TARGET_OCCUPANCY}%`,
        // prevWeekOccupancy: `${prevWtdAvgOccupancy.toFixed(1)}%`,  // TODO: wire to real backend data
        prevWeekOccupancy: '0',

        totalCalls: wtdTotalCalls.toLocaleString(),
        callsChange: `${callsChange >= 0 ? '+' : ''}${callsChange.toLocaleString()}`,
        callsPositive: callsChange >= 0,
        targetCalls: '',
        // prevWeekCalls: prevWtdTotalCalls.toLocaleString(),  // TODO: wire to real backend data
        prevWeekCalls: '0'
      };
    }

    // For past weeks, show full week metrics
    const weekEnd = new Date(currentWeekStart);
    weekEnd.setDate(weekEnd.getDate() + 6);

    let totalCalls = 0;
    for (let i = 0; i < 7; i++) {
      const day = new Date(currentWeekStart);
      day.setDate(day.getDate() + i);
      const isWeekend = day.getDay() === 0 || day.getDay() === 6;
      const baseVolume = isWeekend ? 6000 : 12500;
      const seed = day.getFullYear() * 10000 + (day.getMonth() + 1) * 100 + day.getDate();
      const randomVal = seededRandom(seed);
      const noise = randomVal * 2000 - 1000;
      totalCalls += Math.max(4000, Math.floor(baseVolume + noise));
    }

    // Calculate metrics with some variation based on week
    const weekSeed = currentWeekStart.getTime();
    const slaBase = 92 + (seededRandom(weekSeed) * 6); // 92-98%
    const sla = Math.min(98, Math.max(88, slaBase)).toFixed(1);

    const waitBase = 35 + (seededRandom(weekSeed + 1) * 25); // 35-60s
    const waitTime = Math.round(waitBase);

    const occupancyBase = 82 + (seededRandom(weekSeed + 2) * 10); // 82-92%
    const occupancy = Math.min(95, Math.max(80, occupancyBase)).toFixed(1);

    // Calculate changes from previous week
    const prevSla = 92 + (seededRandom(weekSeed - 604800000) * 6);
    const slaChange = (slaBase - prevSla).toFixed(1);

    const prevWait = 35 + (seededRandom(weekSeed + 1 - 604800000) * 25);
    const waitChange = Math.round(waitBase - prevWait);

    const prevOccupancy = 82 + (seededRandom(weekSeed + 2 - 604800000) * 10);
    const occupancyChange = (occupancyBase - prevOccupancy).toFixed(1);

    const prevTotalCalls = totalCalls - Math.floor(seededRandom(weekSeed + 3) * 4000 - 2000);
    const callsChange = totalCalls - prevTotalCalls;

    return {
      sla: `${sla}%`,
      slaChange: `${Number(slaChange) >= 0 ? '+' : ''}${slaChange}%`,
      slaPositive: Number(slaChange) >= 0,

      waitTime: `${waitTime}s`,
      waitChange: `${waitChange >= 0 ? '+' : ''}${waitChange}s`,
      waitPositive: waitChange <= 0, // Negative wait time change is positive

      occupancy: `${occupancy}%`,
      occupancyChange: `${Number(occupancyChange) >= 0 ? '+' : ''}${occupancyChange}%`,
      occupancyPositive: Number(occupancyChange) >= 0,

      totalCalls: totalCalls.toLocaleString(),
      callsChange: `${callsChange >= 0 ? '+' : ''}${callsChange.toLocaleString()}`,
      callsPositive: callsChange >= 0
    };
  };

  const weeklyMetrics = calculateWeeklyMetrics();

  // Format week date range for description
  const getWeekDateRange = () => {
    const weekEnd = new Date(currentWeekStart);
    weekEnd.setDate(weekEnd.getDate() + 6);

    const formatDate = (date: Date) => {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    return `${formatDate(currentWeekStart)} - ${formatDate(weekEnd)}`;
  };

  // Format WTD date range with time for description
  const getWTDDateTimeRange = () => {
    const startDate = new Date(currentWeekStart);
    const endDate = new Date();

    const formatDateTime = (date: Date) => {
      const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      const hour = date.getHours();
      const minute = date.getMinutes();
      const roundedMinute = minute < 15 ? 0 : minute < 45 ? 30 : 60;
      let finalHour = hour;
      let finalMinute = roundedMinute;

      if (finalMinute === 60) {
        finalHour += 1;
        finalMinute = 0;
      }

      const timeStr = `${finalHour}:${finalMinute.toString().padStart(2, '0')}`;
      return `${dateStr} ${timeStr}`;
    };

    return `${formatDateTime(startDate)} - ${formatDateTime(endDate)}`;
  };

  // Handler for simulator target changes
  const handleSimulatorTargetsChange = useCallback((targets: { sla: number; waitTime: number; occupancy: number }) => {
    setSimulatorTargets(targets);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 relative transition-colors duration-300">
      <div className="transition-all duration-300 relative z-10">
        <main className="p-6 md:p-8 max-w-7xl mx-auto space-y-8">
          <div className="flex flex-col sm:flex-row justify-between items-start space-y-4 sm:space-y-0">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Intuit Call Center Dashboard</h1>
              <p className="text-slate-500 dark:text-slate-400 mt-1">Real-time insights and workforce performance metrics.</p>
            </div>

            <div className="flex items-center space-x-3 -mt-1">
                <Link
                  to="/shift-scheduler"
                  className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors shadow-sm"
                >
                  <CalendarClock className="w-4 h-4" />
                  <span className="text-sm font-medium">Shift Scheduler</span>
                </Link>
                <ThemeToggle />
            </div>
          </div>

          <div className="relative">
             {/* Weekly Overview Section */}
             <div
               className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shadow-sm overflow-hidden"
             >
               {/* Header */}
               <div className="p-6 border-b border-slate-200/50 dark:border-slate-700/50">
                 <div className="flex items-center justify-between">
                   <div className="flex items-center space-x-3">
                     <div className="p-2.5 bg-gradient-to-br from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-700 rounded-lg shadow-sm">
                       <Calendar className="w-6 h-6 text-white" />
                     </div>
                     <div>
                       <h2 className="text-xl font-bold text-slate-900 dark:text-white">Weekly Overview</h2>
                       <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">Key performance metrics and demand forecast</p>
                     </div>
                   </div>
                 </div>
               </div>

               {/* Content */}
               <div className="p-6 space-y-6">
                 {/* Weekly Performance Metrics Label */}
                 <div>
                   <div className="flex items-center justify-between">
                     <div className="flex items-center space-x-2">
                       <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                         <Target className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                       </div>
                       <div>
                         <h3 className="text-base font-bold text-slate-900 dark:text-white">Weekly Performance Metrics</h3>
                         <p className="text-xs text-slate-500 dark:text-slate-400">
                           {weeklyMetrics.isFuture
                             ? `Target metrics and forecasted calls for week of ${getWeekDateRange()}`
                             : weeklyMetrics.isCurrentWeek
                             ? `Week-to-date data for ${getWTDDateTimeRange()}, updated every 30 minutes. Comparisons use the same period from the previous week.`
                             : `Full week data for ${getWeekDateRange()}. Comparisons use the previous week's full data.`}
                         </p>
                       </div>
                     </div>
                     <Link to="/metrics-history" className="text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 flex items-center space-x-1 whitespace-nowrap">
                       <span>View All History</span>
                       <ArrowRight className="w-4 h-4" />
                     </Link>
                   </div>
                 </div>

                 {metricsError && (
                   <div className="mb-4 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
                     <p className="text-amber-800 dark:text-amber-200 text-sm">Showing estimated data (metrics API unavailable)</p>
                   </div>
                 )}

                 {/* Metrics Grid */}
                 <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                    <MetricCard
                      title="Service Level (SLA)"
                      value={weeklyMetrics.isCurrentWeek ? (metricsLoading ? '...' : metrics ? `${metrics.avg_sla_compliance.toFixed(1)}%` : metricsError ? '—' : weeklyMetrics.sla) : weeklyMetrics.sla}
                      change={weeklyMetrics.slaChange}
                      isPositive={weeklyMetrics.slaPositive}
                      icon={Target}
                      targetText={weeklyMetrics.targetSLA}
                      prevWeekValue={weeklyMetrics.prevWeekSLA}
                    />
                    <MetricCard
                      title="Avg. Waiting Time"
                      value={weeklyMetrics.isCurrentWeek ? (metricsLoading ? '...' : metrics ? `${Math.round(metrics.avg_wait_time)}s` : metricsError ? '—' : weeklyMetrics.waitTime) : weeklyMetrics.waitTime}
                      change={weeklyMetrics.waitChange}
                      isPositive={weeklyMetrics.waitPositive}
                      icon={Clock}
                      targetText={weeklyMetrics.targetWaitTime}
                      prevWeekValue={weeklyMetrics.prevWeekWait}
                    />
                    <MetricCard
                      title="Avg. Agent Occupancy"
                      value={weeklyMetrics.isCurrentWeek ? (metricsLoading ? '...' : metrics ? `${metrics.avg_occupancy.toFixed(1)}%` : metricsError ? '—' : weeklyMetrics.occupancy) : weeklyMetrics.occupancy}
                      change={weeklyMetrics.occupancyChange}
                      isPositive={weeklyMetrics.occupancyPositive}
                      icon={TrendingUp}
                      targetText={weeklyMetrics.targetOccupancy}
                      prevWeekValue={weeklyMetrics.prevWeekOccupancy}
                    />
                    <MetricCard
                      title="Total Calls Processed"
                      value={weeklyMetrics.isCurrentWeek ? (metricsLoading ? '...' : metrics ? metrics.total_calls.toLocaleString() : metricsError ? '—' : weeklyMetrics.totalCalls) : weeklyMetrics.totalCalls}
                      change={weeklyMetrics.callsChange}
                      isPositive={weeklyMetrics.callsPositive}
                      icon={Phone}
                      targetText={weeklyMetrics.targetCalls}
                      prevWeekValue={weeklyMetrics.prevWeekCalls}
                    />
                  </div>

                 {/* Weekly Demand Forecast */}
                 <div className="h-[400px] min-h-[400px]">
                   <DemandForecastChart
                     weekStart={currentWeekStart}
                     onPrevWeek={handlePrevWeek}
                     onNextWeek={handleNextWeek}
                     selectedDate={selectedDay}
                     onSelectDate={handleSelectDate}
                     onJumpToToday={handleJumpToToday}
                   />
                 </div>
               </div>
             </div>
          </div>

          <WorkforceManagementPanel
            selectedDate={selectedDay}
            onPrevDay={handlePrevDay}
            onNextDay={handleNextDay}
            initialCallVolume={selectedDayVolume}
            onReset={handleReset}
            dailyBreakdownData={dailyBreakdownData}
            onJumpToToday={handleJumpToToday}
          />

          <div
            className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shadow-sm"
          >
            <div className="p-6 border-b border-slate-200 dark:border-slate-700">
              <div className="flex items-center space-x-2 mb-4">
                <div className="p-2 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg">
                  <Route className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-slate-900 dark:text-white">Agent Routing</h2>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Get agent recommendations for incoming calls</p>
                </div>
              </div>
              <div className="mb-4 p-4 rounded-lg bg-slate-50 dark:bg-slate-700/50 border border-slate-200 dark:border-slate-600 text-sm text-slate-700 dark:text-slate-300 space-y-3">
                <p className="font-medium text-slate-900 dark:text-white">How agents are recommended</p>
                <p>When a call comes in, the system finds the best agents by asking:</p>
                <ol className="list-decimal list-inside space-y-2 ml-2">
                  <li><strong>Have they handled this type of call before?</strong> (Product / Segment) — Enter a product code (e.g. SBSEG_US_OL_PRPM_QBO_A_PHN) or a segment name (e.g. Product Support). Product codes use call history; segment names match agents by their domain. The &quot;Affinity&quot; column shows calls handled for that product.</li>
                  <li><strong>Have they handled this channel?</strong> (Phone, Chat, etc.) — Agents with more experience on the selected channel get a boost.</li>
                  <li><strong>Do they usually resolve issues without transferring?</strong> — Agents with higher resolution rates and lower transfer rates score better.</li>
                </ol>
                <p><strong>What the score means:</strong> The score (0–1) is a single number combining all of the above. Higher = better fit for this call. When scores tie, agents with more contacts or higher affinity rank first.</p>
                <p><strong>Availability:</strong> We only show agents who are typically working at this time of day.</p>
              </div>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleFetchRouting();
                }}
                className="flex flex-wrap gap-3 items-end"
              >
                <div className="flex-1 min-w-[180px]">
                  <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Product / Segment</label>
                  <input
                    type="text"
                    value={routingProduct}
                    onChange={(e) => setRoutingProduct(e.target.value)}
                    placeholder="e.g. Product Support or SBSEG_US_OL_PRPM_QBO_A_PHN"
                    className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-700 text-slate-900 dark:text-slate-100 text-sm focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  />
                </div>
                <div className="w-36">
                  <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Channel</label>
                  <select
                    value={routingChannel}
                    onChange={(e) => setRoutingChannel(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-700 text-slate-900 dark:text-slate-100 text-sm focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  >
                    <option value="">Any</option>
                    <option value="PHONE">Phone</option>
                    <option value="CHAT">Chat</option>
                    <option value="INBOUND">Inbound</option>
                    <option value="OUTBOUND">Outbound</option>
                  </select>
                </div>
                <button
                  type="submit"
                  disabled={routingLoading}
                  className="flex items-center space-x-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-400 text-white rounded-lg transition-colors text-sm font-medium"
                >
                  {routingLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Loading...</span>
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4" />
                      <span>Get Recommendations</span>
                    </>
                  )}
                </button>
              </form>
              {routingError && (
                <p className="mt-3 text-sm text-amber-600 dark:text-amber-400">{routingError}</p>
              )}
              {routingRecommendations.length > 0 && (
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 dark:bg-slate-700/50 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-600">
                      <tr>
                        <th className="py-2 text-left font-medium">Agent</th>
                        <th className="py-2 text-left font-medium">Segment</th>
                        <th className="py-2 text-right font-medium">Resolution</th>
                        <th className="py-2 text-right font-medium">Transfer</th>
                        <th className="py-2 text-right font-medium">Affinity</th>
                        <th className="py-2 text-right font-medium">Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-600">
                      {routingRecommendations.map((rec) => (
                        <tr key={rec.expert_id} className="hover:bg-slate-50 dark:hover:bg-slate-700/50">
                          <td className="py-2">
                            <Link
                              to={`/agent/${encodeURIComponent(rec.expert_id)}`}
                              className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
                            >
                              {rec.name}
                            </Link>
                          </td>
                          <td className="py-2 text-slate-600 dark:text-slate-400">{rec.segment}</td>
                          <td className="py-2 text-right">{rec.resolution_rate}%</td>
                          <td className="py-2 text-right">{rec.transfer_rate}%</td>
                          <td className="py-2 text-right">{rec.affinity_volume}</td>
                          <td className="py-2 text-right font-medium">{rec.score.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>

          <div
             className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shadow-sm"
          >
             <div className="p-6 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
                <div className="flex items-center space-x-2">
                  <div className="p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
                    <Trophy className="w-5 h-5 text-yellow-600 dark:text-yellow-500" />
                  </div>
                  <h2 className="text-lg font-bold text-slate-900 dark:text-white">Top Performing Agents</h2>
                </div>
                <Link to="/all-agents" className="text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 flex items-center space-x-1 whitespace-nowrap">
                  <span>View All</span>
                  <ArrowRight className="w-4 h-4" />
                </Link>
             </div>
             <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-50 dark:bg-slate-700/50 text-slate-500 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-600">
                    <tr>
                      <th className="px-6 py-4">Agent ID</th>
                      <th className="px-6 py-4">Segment</th>
                      <th className="px-6 py-4">Contacts</th>
                      <th className="px-6 py-4">Average Handle Time</th>
                      <th className="px-6 py-4">Utilization Rate</th>
                      <th className="px-6 py-4">Resolution Rate</th>
                      <th
                        className="px-6 py-4"
                        title="30% resolution rate, 20% first-call resolution, 20% efficiency, 15% occupancy, 15% volume"
                      >
                        Composite Score
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-600">
                    {agentsLoading ? (
                      <tr>
                        <td colSpan={7} className="px-6 py-8 text-center">
                          <div className="flex items-center justify-center space-x-2 text-slate-500 dark:text-slate-400">
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Loading agents...</span>
                          </div>
                        </td>
                      </tr>
                    ) : topAgents.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="px-6 py-8 text-center text-slate-500 dark:text-slate-400">
                          No agent data available
                        </td>
                      </tr>
                    ) : (
                      topAgents.map((agent) => (
                        <tr key={agent.expert_id} className="hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer">
                          <td className="px-6 py-4">
                            <Link
                              to={`/agent/${encodeURIComponent(agent.expert_id)}`}
                              className="font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:underline"
                            >
                              {agent.expert_id}
                            </Link>
                          </td>
                          <td className="px-6 py-4">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400">
                              {agent.segment}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.contacts.toLocaleString()}</td>
                          <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.aht}</td>
                          <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.utilization.toFixed(1)}%</td>
                          <td className="px-6 py-4">
                            <span className={`font-bold ${
                              agent.resolution_rate >= 90 ? 'text-green-600 dark:text-green-400' :
                              agent.resolution_rate >= 80 ? 'text-blue-600 dark:text-blue-400' :
                              'text-slate-900 dark:text-white'
                            }`}>
                              {agent.resolution_rate.toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-6 py-4">
                            <span className={`font-bold ${
                              agent.composite_score >= 80 ? 'text-green-600 dark:text-green-400' :
                              agent.composite_score >= 60 ? 'text-blue-600 dark:text-blue-400' :
                              'text-slate-900 dark:text-white'
                            }`}>
                              {agent.composite_score.toFixed(1)}
                            </span>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
          </div>
        </main>
      </div>
    </div>
  );
}
