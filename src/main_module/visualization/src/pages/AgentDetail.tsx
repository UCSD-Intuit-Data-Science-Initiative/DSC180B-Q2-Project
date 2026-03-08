import React, { useMemo, useState } from 'react';
import { ArrowLeft, Phone, Clock, TrendingUp, TrendingDown, Calendar, Target, Award, Activity } from 'lucide-react';
import { Link, useParams, useLocation, useNavigate } from 'react-router';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { useTheme } from '../context/ThemeContext';
import { ThemeToggle } from '../components/ThemeToggle';

interface CallRecord {
  time: string;
  duration: string;
  durationSeconds: number;
  type: 'Inbound' | 'Outbound';
  outcome: 'Resolved' | 'Transferred' | 'Escalated';
}

export default function AgentDetail() {
  const { agentName } = useParams<{ agentName: string }>();
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const location = useLocation();
  const navigate = useNavigate();

  // Decode agent name from URL
  const decodedAgentName = agentName ? decodeURIComponent(agentName) : '';

  // Function to go back to previous page
  const handleBackClick = () => {
    navigate(-1);
  };

  // Agent data lookup
  const agentData = useMemo(() => {
    const agents = [
      { name: 'Jackie Wang', status: 'Online', calls: 45, aht: '3m 12s', ahtSeconds: 192, utilization: '87%', utilizationPercent: 87, email: 'jackie.wang@company.com', team: 'Tax Expert', shift: '9:00 AM - 5:00 PM' },
      { name: 'Sarah He', status: 'In Call', calls: 38, aht: '2m 55s', ahtSeconds: 175, utilization: '84%', utilizationPercent: 84, email: 'sarah.he@company.com', team: 'Payroll Specialist', shift: '10:00 AM - 6:00 PM' },
      { name: 'Hao Zhang', status: 'Break', calls: 41, aht: '3m 05s', ahtSeconds: 185, utilization: '86%', utilizationPercent: 86, email: 'hao.zhang@company.com', team: 'Tax Expert', shift: '8:00 AM - 4:00 PM' },
      { name: 'Sophia Fang', status: 'Online', calls: 32, aht: '3m 40s', ahtSeconds: 220, utilization: '82%', utilizationPercent: 82, email: 'sophia.fang@company.com', team: 'Bookkeeping Advisor', shift: '9:00 AM - 5:00 PM' },
      { name: 'Michael Chen', status: 'In Call', calls: 43, aht: '2m 48s', ahtSeconds: 168, utilization: '89%', utilizationPercent: 89, email: 'michael.chen@company.com', team: 'Tax Expert', shift: '9:00 AM - 5:00 PM' },
      { name: 'Lisa Park', status: 'Online', calls: 39, aht: '3m 20s', ahtSeconds: 200, utilization: '85%', utilizationPercent: 85, email: 'lisa.park@company.com', team: 'Payroll Specialist', shift: '11:00 AM - 7:00 PM' },
      { name: 'David Liu', status: 'Break', calls: 36, aht: '3m 15s', ahtSeconds: 195, utilization: '83%', utilizationPercent: 83, email: 'david.liu@company.com', team: 'Bookkeeping Advisor', shift: '8:00 AM - 4:00 PM' },
      { name: 'Emily Zhang', status: 'Online', calls: 44, aht: '2m 52s', ahtSeconds: 172, utilization: '88%', utilizationPercent: 88, email: 'emily.zhang@company.com', team: 'Tax Expert', shift: '9:00 AM - 5:00 PM' },
      { name: 'Ryan Kim', status: 'In Call', calls: 40, aht: '3m 08s', ahtSeconds: 188, utilization: '86%', utilizationPercent: 86, email: 'ryan.kim@company.com', team: 'Payroll Specialist', shift: '10:00 AM - 6:00 PM' },
      { name: 'Jessica Wu', status: 'Online', calls: 37, aht: '3m 25s', ahtSeconds: 205, utilization: '84%', utilizationPercent: 84, email: 'jessica.wu@company.com', team: 'Bookkeeping Advisor', shift: '9:00 AM - 5:00 PM' },
      { name: 'Kevin Zhao', status: 'Break', calls: 35, aht: '3m 18s', ahtSeconds: 198, utilization: '81%', utilizationPercent: 81, email: 'kevin.zhao@company.com', team: 'Tax Expert', shift: '8:00 AM - 4:00 PM' },
      { name: 'Amy Lin', status: 'Online', calls: 42, aht: '2m 58s', ahtSeconds: 178, utilization: '87%', utilizationPercent: 87, email: 'amy.lin@company.com', team: 'Payroll Specialist', shift: '9:00 AM - 5:00 PM' },
      { name: 'Tom Martinez', status: 'In Call', calls: 38, aht: '3m 10s', ahtSeconds: 190, utilization: '85%', utilizationPercent: 85, email: 'tom.martinez@company.com', team: 'Bookkeeping Advisor', shift: '11:00 AM - 7:00 PM' },
      { name: 'Rachel Chang', status: 'Online', calls: 41, aht: '3m 02s', ahtSeconds: 182, utilization: '86%', utilizationPercent: 86, email: 'rachel.chang@company.com', team: 'Tax Expert', shift: '9:00 AM - 5:00 PM' },
      { name: 'James Lee', status: 'Break', calls: 34, aht: '3m 28s', ahtSeconds: 208, utilization: '80%', utilizationPercent: 80, email: 'james.lee@company.com', team: 'Payroll Specialist', shift: '8:00 AM - 4:00 PM' },
      { name: 'Anna Chen', status: 'Offline', calls: 30, aht: '3m 35s', ahtSeconds: 215, utilization: '78%', utilizationPercent: 78, email: 'anna.chen@company.com', team: 'Bookkeeping Advisor', shift: '9:00 AM - 5:00 PM' },
      { name: 'Mark Johnson', status: 'Online', calls: 46, aht: '2m 45s', ahtSeconds: 165, utilization: '90%', utilizationPercent: 90, email: 'mark.johnson@company.com', team: 'Tax Expert', shift: '9:00 AM - 5:00 PM' },
      { name: 'Lily Wang', status: 'In Call', calls: 39, aht: '3m 05s', ahtSeconds: 185, utilization: '85%', utilizationPercent: 85, email: 'lily.wang@company.com', team: 'Payroll Specialist', shift: '10:00 AM - 6:00 PM' },
      { name: 'Chris Lee', status: 'Break', calls: 33, aht: '3m 22s', ahtSeconds: 202, utilization: '79%', utilizationPercent: 79, email: 'chris.lee@company.com', team: 'Bookkeeping Advisor', shift: '8:00 AM - 4:00 PM' },
      { name: 'Michelle Kim', status: 'Online', calls: 40, aht: '3m 12s', ahtSeconds: 192, utilization: '86%', utilizationPercent: 86, email: 'michelle.kim@company.com', team: 'Tax Expert', shift: '9:00 AM - 5:00 PM' },
      { name: 'Daniel Park', status: 'Offline', calls: 28, aht: '3m 40s', ahtSeconds: 220, utilization: '75%', utilizationPercent: 75, email: 'daniel.park@company.com', team: 'Payroll Specialist', shift: '9:00 AM - 5:00 PM' },
      { name: 'Sophie Zhang', status: 'In Call', calls: 41, aht: '2m 58s', ahtSeconds: 178, utilization: '87%', utilizationPercent: 87, email: 'sophie.zhang@company.com', team: 'Bookkeeping Advisor', shift: '11:00 AM - 7:00 PM' },
      { name: 'Alex Wu', status: 'Online', calls: 44, aht: '3m 00s', ahtSeconds: 180, utilization: '88%', utilizationPercent: 88, email: 'alex.wu@company.com', team: 'Tax Expert', shift: '9:00 AM - 5:00 PM' },
      { name: 'Grace Liu', status: 'Break', calls: 36, aht: '3m 18s', ahtSeconds: 198, utilization: '83%', utilizationPercent: 83, email: 'grace.liu@company.com', team: 'Payroll Specialist', shift: '8:00 AM - 4:00 PM' },
      { name: 'Eric Chen', status: 'Online', calls: 42, aht: '2m 52s', ahtSeconds: 172, utilization: '87%', utilizationPercent: 87, email: 'eric.chen@company.com', team: 'Tax Expert', shift: '9:00 AM - 5:00 PM' },
    ];

    return agents.find(a => a.name === decodedAgentName) || agents[0];
  }, [decodedAgentName]);

  // Generate weekly schedule
  const weeklySchedule = useMemo(() => {
    const today = new Date();
    const monday = new Date(today);
    const day = monday.getDay();
    const diff = monday.getDate() - day + (day === 0 ? -6 : 1);
    monday.setDate(diff);
    monday.setHours(0, 0, 0, 0);

    const schedule = [];
    const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

    for (let i = 0; i < 7; i++) {
      const date = new Date(monday);
      date.setDate(monday.getDate() + i);

      // Determine if agent works this day based on their shift pattern
      const seed = agentData.calls + i;
      const isWorking = Math.sin(seed) > -0.3; // ~85% chance of working

      schedule.push({
        day: dayNames[i],
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        shift: isWorking ? agentData.shift : 'Off',
        isWorking
      });
    }

    return schedule;
  }, [agentData]);

  // Generate 7-day performance trend
  const performanceTrend = useMemo(() => {
    const data = [];
    const today = new Date();
    for (let i = 6; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      const seed = date.getTime() + agentData.calls;
      const variation = Math.sin(seed) * 5;

      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        calls: Math.max(25, Math.floor(agentData.calls + variation)),
        utilization: Math.min(95, Math.max(70, agentData.utilizationPercent + Math.floor(variation))),
        aht: Math.max(120, Math.floor(agentData.ahtSeconds + variation * 5))
      });
    }
    return data;
  }, [agentData]);

  // Generate today's call history
  const callHistory = useMemo((): CallRecord[] => {
    const history: CallRecord[] = [];
    const callCount = agentData.calls;

    for (let i = 0; i < Math.min(callCount, 20); i++) {
      const hour = 9 + Math.floor((i / callCount) * 8);
      const minute = Math.floor(Math.random() * 60);
      const durationSeconds = Math.floor(agentData.ahtSeconds + (Math.random() - 0.5) * 60);
      const durationMin = Math.floor(durationSeconds / 60);
      const durationSec = durationSeconds % 60;

      history.push({
        time: `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`,
        duration: `${durationMin}m ${durationSec}s`,
        durationSeconds,
        type: Math.random() > 0.2 ? 'Inbound' : 'Outbound',
        outcome: Math.random() > 0.8 ? 'Transferred' : Math.random() > 0.9 ? 'Escalated' : 'Resolved'
      });
    }

    return history.reverse();
  }, [agentData]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Online':
        return 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400';
      case 'In Call':
        return 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400';
      case 'Break':
        return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-400';
      case 'Offline':
        return 'bg-slate-100 dark:bg-slate-700/30 text-slate-800 dark:text-slate-400';
      default:
        return 'bg-slate-100 dark:bg-slate-700/30 text-slate-800 dark:text-slate-400';
    }
  };

  const getOutcomeColor = (outcome: string) => {
    switch (outcome) {
      case 'Resolved':
        return 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400';
      case 'Transferred':
        return 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400';
      case 'Escalated':
        return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-400';
      default:
        return 'bg-slate-100 dark:bg-slate-700/30 text-slate-800 dark:text-slate-400';
    }
  };

  // Calculate stats
  const resolvedCount = callHistory.filter(c => c.outcome === 'Resolved').length;
  const resolutionRate = callHistory.length > 0 ? ((resolvedCount / callHistory.length) * 100).toFixed(1) : '0';
  const avgCallDuration = callHistory.length > 0
    ? callHistory.reduce((sum, c) => sum + c.durationSeconds, 0) / callHistory.length
    : 0;
  const avgCallMin = Math.floor(avgCallDuration / 60);
  const avgCallSec = Math.floor(avgCallDuration % 60);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 p-6 md:p-8 transition-colors duration-300">
      <style>
        {`
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
          * {
            scrollbar-width: thin;
            scrollbar-color: rgba(156, 163, 175, 0.5) transparent;
          }
        `}
      </style>

      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button onClick={handleBackClick} className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div className="flex-1">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-lg font-bold text-blue-600 dark:text-blue-400">
                  {agentData.name.split(' ').map(n => n[0]).join('')}
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-slate-900 dark:text-white">{agentData.name}</h1>
                  <div className="flex items-center space-x-3 mt-1">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(agentData.status)}`}>
                      {agentData.status}
                    </span>
                    <span className="text-sm text-slate-500 dark:text-slate-400">{agentData.email}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <ThemeToggle />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Calls Today */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Phone className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Calls Today</span>
              </div>
              <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{agentData.calls}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">+8% from yesterday</p>
          </div>

          {/* Average Handle Time */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Avg Handle Time</span>
              </div>
              <TrendingDown className="w-4 h-4 text-green-600 dark:text-green-400" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{agentData.aht}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">-5% from yesterday</p>
          </div>

          {/* Utilization Rate */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Utilization</span>
              </div>
              <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{agentData.utilization}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">+3% from yesterday</p>
          </div>

          {/* Resolution Rate */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Resolution Rate</span>
              </div>
              <Award className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{resolutionRate}%</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{resolvedCount}/{callHistory.length} resolved</p>
          </div>
        </div>

        {/* Agent Info */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white mb-4">Agent Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Team</p>
              <p className="text-base font-medium text-slate-900 dark:text-white">{agentData.team}</p>
            </div>
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">Weekly Schedule</p>
              <div className="space-y-2">
                {weeklySchedule.map((day, idx) => (
                  <div key={idx} className="flex items-center justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">{day.day} ({day.date})</span>
                    <span className={`font-medium ${
                      day.isWorking
                        ? 'text-green-700 dark:text-green-400'
                        : 'text-slate-500 dark:text-slate-500'
                    }`}>
                      {day.shift}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Performance Trend Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Calls Trend */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">7-Day Call Volume</h2>
              <div className="text-sm text-slate-500 dark:text-slate-400">
                Avg: <span className="font-bold text-blue-600 dark:text-blue-400">
                  {Math.round(performanceTrend.reduce((sum, d) => sum + d.calls, 0) / performanceTrend.length)}
                </span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={250} minWidth={300} minHeight={250}>
              <LineChart data={performanceTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} />
                <XAxis dataKey="date" stroke={isDark ? '#94a3b8' : '#64748b'} style={{ fontSize: '12px' }} />
                <YAxis stroke={isDark ? '#94a3b8' : '#64748b'} style={{ fontSize: '12px' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isDark ? '#1e293b' : '#ffffff',
                    border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`,
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
                <ReferenceLine
                  y={performanceTrend.reduce((sum, d) => sum + d.calls, 0) / performanceTrend.length}
                  stroke={isDark ? '#60a5fa' : '#3b82f6'}
                  strokeDasharray="5 5"
                  label={{ value: 'Avg', position: 'right', fill: isDark ? '#60a5fa' : '#3b82f6', fontSize: 12 }}
                />
                <Line type="monotone" dataKey="calls" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6', r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Utilization Trend */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">7-Day Utilization Rate</h2>
              <div className="text-sm text-slate-500 dark:text-slate-400">
                Avg: <span className="font-bold text-green-600 dark:text-green-400">
                  {Math.round(performanceTrend.reduce((sum, d) => sum + d.utilization, 0) / performanceTrend.length)}%
                </span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={250} minWidth={300} minHeight={250}>
              <BarChart data={performanceTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} />
                <XAxis dataKey="date" stroke={isDark ? '#94a3b8' : '#64748b'} style={{ fontSize: '12px' }} />
                <YAxis stroke={isDark ? '#94a3b8' : '#64748b'} style={{ fontSize: '12px' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isDark ? '#1e293b' : '#ffffff',
                    border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`,
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                  formatter={(value: number) => `${value}%`}
                />
                <ReferenceLine
                  y={performanceTrend.reduce((sum, d) => sum + d.utilization, 0) / performanceTrend.length}
                  stroke={isDark ? '#34d399' : '#10b981'}
                  strokeDasharray="5 5"
                  label={{ value: 'Avg', position: 'right', fill: isDark ? '#34d399' : '#10b981', fontSize: 12 }}
                />
                <Bar dataKey="utilization" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Call History */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700">
          <div className="p-6 border-b border-slate-100 dark:border-slate-700">
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Today's Call History</h2>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              Showing {callHistory.length} most recent calls • Average: {avgCallMin}m {avgCallSec}s
            </p>
          </div>
          <div className="overflow-auto max-h-[500px]">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 dark:bg-slate-900 text-slate-500 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-700 sticky top-0">
                <tr>
                  <th className="px-6 py-3">
                    <div className="flex items-center space-x-2">
                      <Clock className="w-4 h-4" />
                      <span>Time</span>
                    </div>
                  </th>
                  <th className="px-6 py-3">Type</th>
                  <th className="px-6 py-3">Duration</th>
                  <th className="px-6 py-3">Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                {callHistory.map((call, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors">
                    <td className="px-6 py-3 font-medium text-slate-900 dark:text-white">{call.time}</td>
                    <td className="px-6 py-3 text-slate-600 dark:text-slate-400">{call.type}</td>
                    <td className="px-6 py-3 text-slate-600 dark:text-slate-400">{call.duration}</td>
                    <td className="px-6 py-3">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getOutcomeColor(call.outcome)}`}>
                        {call.outcome}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
