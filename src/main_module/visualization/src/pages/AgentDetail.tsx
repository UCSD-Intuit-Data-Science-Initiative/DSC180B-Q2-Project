import React, { useMemo, useState, useEffect } from 'react';
import { ArrowLeft, Phone, Clock, TrendingUp, TrendingDown, Target, Award, Activity, Loader2 } from 'lucide-react';
import { useParams, useNavigate } from 'react-router';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useTheme } from '../context/ThemeContext';
import { ThemeToggle } from '../components/ThemeToggle';

interface AgentProfile {
  expert_id: string;
  name: string;
  segment: string;
  business_segment: string;
  status: string;
  contacts: number;
  answered_contacts: number;
  resolution_rate: number;
  transfer_rate: number;
  aht: string;
  aht_seconds: number;
  hold_time_seconds: number;
  composite_score: number;
  fcr_rate: number;
  utilization: number;
  mean_occupancy: number;
  answer_rate: number;
  median_hold: number;
  mean_hold: number;
}

const API_BASE = import.meta.env.VITE_API_URL || '';

export default function AgentDetail() {
  const { agentName } = useParams<{ agentName: string }>();
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const navigate = useNavigate();

  const [agent, setAgent] = useState<AgentProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const expertId = agentName ? decodeURIComponent(agentName) : '';

  useEffect(() => {
    const fetchAgent = async () => {
      if (!expertId) return;

      try {
        setLoading(true);
        const response = await fetch(`${API_BASE}/api/agents/${encodeURIComponent(expertId)}`);
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('Agent not found');
          }
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAgent(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch agent:', err);
        setError(err instanceof Error ? err.message : 'Failed to load agent data');
      } finally {
        setLoading(false);
      }
    };

    fetchAgent();
  }, [expertId]);

  const handleBackClick = () => {
    navigate(-1);
  };

  const performanceMetrics = useMemo(() => {
    if (!agent) return [];

    return [
      { metric: 'Resolution Rate', value: agent.resolution_rate, target: 85 },
      { metric: 'FCR Rate', value: agent.fcr_rate, target: 80 },
      { metric: 'Utilization', value: agent.utilization, target: 75 },
      { metric: 'Answer Rate', value: agent.answer_rate, target: 90 },
    ];
  }, [agent]);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-400">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Loading agent profile...</span>
        </div>
      </div>
    );
  }

  if (error || !agent) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 p-6 md:p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center space-x-4 mb-8">
            <button onClick={handleBackClick} className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </button>
            <h1 className="text-2xl font-bold">Agent Not Found</h1>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            <p className="text-red-700 dark:text-red-300">{error || 'Agent not found'}</p>
          </div>
        </div>
      </div>
    );
  }

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
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button onClick={handleBackClick} className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div className="flex-1">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-lg font-bold text-blue-600 dark:text-blue-400">
                  {agent.expert_id.slice(-2).toUpperCase()}
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-slate-900 dark:text-white">{agent.expert_id}</h1>
                  <div className="flex items-center space-x-3 mt-1">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400">
                      {agent.segment}
                    </span>
                    <span className="text-sm text-slate-500 dark:text-slate-400">{agent.business_segment}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <ThemeToggle />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Phone className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Total Contacts</span>
              </div>
              <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{agent.contacts.toLocaleString()}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{agent.answered_contacts.toLocaleString()} answered</p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Avg Handle Time</span>
              </div>
              <TrendingDown className="w-4 h-4 text-green-600 dark:text-green-400" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{agent.aht}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Hold: {Math.round(agent.mean_hold)}s avg</p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Utilization</span>
              </div>
              <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{agent.utilization.toFixed(1)}%</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Occupancy: {agent.mean_occupancy.toFixed(1)}%</p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Resolution Rate</span>
              </div>
              <Award className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{agent.resolution_rate.toFixed(1)}%</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">FCR: {agent.fcr_rate.toFixed(1)}%</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
            <h2 className="text-lg font-bold text-slate-900 dark:text-white mb-4">Agent Information</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Expert ID</p>
                <p className="text-base font-medium text-slate-900 dark:text-white">{agent.expert_id}</p>
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Segment</p>
                <p className="text-base font-medium text-slate-900 dark:text-white">{agent.segment}</p>
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Business Segment</p>
                <p className="text-base font-medium text-slate-900 dark:text-white">{agent.business_segment}</p>
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Resolution Rate</p>
                <p className={`text-base font-bold ${
                  agent.resolution_rate >= 90 ? 'text-green-600 dark:text-green-400' :
                  agent.resolution_rate >= 80 ? 'text-yellow-600 dark:text-yellow-400' :
                  'text-red-600 dark:text-red-400'
                }`}>
                  {agent.resolution_rate.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Transfer Rate</p>
                <p className="text-base font-medium text-slate-900 dark:text-white">{agent.transfer_rate.toFixed(1)}%</p>
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Answer Rate</p>
                <p className="text-base font-medium text-slate-900 dark:text-white">{agent.answer_rate.toFixed(1)}%</p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">Performance vs Target</h2>
            </div>
            <ResponsiveContainer width="100%" height={250} minWidth={300} minHeight={250}>
              <BarChart data={performanceMetrics} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} />
                <XAxis type="number" domain={[0, 100]} stroke={isDark ? '#94a3b8' : '#64748b'} style={{ fontSize: '12px' }} />
                <YAxis dataKey="metric" type="category" stroke={isDark ? '#94a3b8' : '#64748b'} style={{ fontSize: '12px' }} width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isDark ? '#1e293b' : '#ffffff',
                    border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`,
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                />
                <ReferenceLine x={80} stroke={isDark ? '#60a5fa' : '#3b82f6'} strokeDasharray="5 5" />
                <Bar
                  dataKey="value"
                  fill="#10b981"
                  radius={[0, 4, 4, 0]}
                  name="Actual"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white mb-4">Detailed Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-slate-50 dark:bg-slate-700/30 rounded-lg">
              <p className="text-2xl font-bold text-slate-900 dark:text-white">{agent.contacts.toLocaleString()}</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">Total Contacts</p>
            </div>
            <div className="text-center p-4 bg-slate-50 dark:bg-slate-700/30 rounded-lg">
              <p className="text-2xl font-bold text-slate-900 dark:text-white">{agent.answered_contacts.toLocaleString()}</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">Answered</p>
            </div>
            <div className="text-center p-4 bg-slate-50 dark:bg-slate-700/30 rounded-lg">
              <p className="text-2xl font-bold text-slate-900 dark:text-white">{Math.round(agent.aht_seconds)}s</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">Avg Handle Time</p>
            </div>
            <div className="text-center p-4 bg-slate-50 dark:bg-slate-700/30 rounded-lg">
              <p className="text-2xl font-bold text-slate-900 dark:text-white">{Math.round(agent.hold_time_seconds)}s</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">Avg Hold Time</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
