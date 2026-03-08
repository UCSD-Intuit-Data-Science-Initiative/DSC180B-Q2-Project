import React, { useMemo, useState, useEffect } from 'react';
import { ArrowLeft, User, Phone, Clock, TrendingUp, ArrowUp, ArrowDown, Loader2 } from 'lucide-react';
import { Link } from 'react-router';
import { ThemeToggle } from '../components/ThemeToggle';

type SortKey = 'name' | 'segment' | 'contacts' | 'aht' | 'utilization' | 'composite_score';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  key: SortKey;
  direction: SortDirection;
}

interface Agent {
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
}

const API_BASE = 'http://localhost:8000';

export default function AllAgents() {
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'composite_score', direction: 'desc' });
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE}/api/agents?n=100&sort_by=composite_score`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAgents(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch agents:', err);
        setError('Failed to load agent data');
      } finally {
        setLoading(false);
      }
    };

    fetchAgents();
  }, []);

  const sortedAgents = useMemo(() => {
    return [...agents].sort((a, b) => {
      let aValue: any = a[sortConfig.key as keyof Agent];
      let bValue: any = b[sortConfig.key as keyof Agent];

      if (sortConfig.key === 'aht') {
        aValue = a.aht_seconds;
        bValue = b.aht_seconds;
      }

      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [agents, sortConfig]);

  const handleSort = (key: SortKey) => {
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const SortIcon = ({ columnKey }: { columnKey: SortKey }) => {
    if (sortConfig.key !== columnKey) return <div className="w-4 h-4" />;
    return sortConfig.direction === 'asc' ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />;
  };

  const getSegmentColor = (segment: string) => {
    const colors: Record<string, string> = {
      'Premium': 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-400',
      'Standard': 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400',
      'Basic': 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400',
    };
    return colors[segment] || 'bg-slate-100 dark:bg-slate-700/30 text-slate-800 dark:text-slate-400';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-400">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Loading agents...</span>
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
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/" className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">All Agents</h1>
              <p className="text-slate-500 dark:text-slate-400 text-sm">
                {agents.length} agents with real performance metrics. Click column headers to sort.
              </p>
            </div>
          </div>
          <ThemeToggle />
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <p className="text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 flex flex-col h-[80vh]">
            <div className="overflow-auto flex-1 rounded-xl custom-scrollbar">
                <table className="w-full text-left text-sm relative">
                    <thead className="bg-slate-50 dark:bg-slate-900 text-slate-500 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-700 sticky top-0 z-10 shadow-sm">
                        <tr>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('name')}
                            >
                                <div className="flex items-center space-x-2">
                                    <User className="w-4 h-4" />
                                    <span>Agent ID</span>
                                    <SortIcon columnKey="name" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('segment')}
                            >
                                <div className="flex items-center space-x-2">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Segment</span>
                                    <SortIcon columnKey="segment" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('contacts')}
                            >
                                <div className="flex items-center space-x-2">
                                    <Phone className="w-4 h-4" />
                                    <span>Contacts</span>
                                    <SortIcon columnKey="contacts" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('aht')}
                            >
                                <div className="flex items-center space-x-2">
                                    <Clock className="w-4 h-4" />
                                    <span>Avg Handle Time</span>
                                    <SortIcon columnKey="aht" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('utilization')}
                            >
                                <div className="flex items-center space-x-2">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Utilization</span>
                                    <SortIcon columnKey="utilization" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('composite_score')}
                            >
                                <div className="flex items-center space-x-2">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Score</span>
                                    <SortIcon columnKey="composite_score" />
                                </div>
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                        {sortedAgents.map((agent) => (
                            <tr key={agent.expert_id} className="hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors cursor-pointer">
                                <td className="px-6 py-4">
                                    <Link
                                        to={`/agent/${encodeURIComponent(agent.expert_id)}`}
                                        className="font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:underline"
                                    >
                                        {agent.expert_id}
                                    </Link>
                                </td>
                                <td className="px-6 py-4">
                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSegmentColor(agent.segment)}`}>
                                        {agent.segment}
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.contacts.toLocaleString()}</td>
                                <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.aht}</td>
                                <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.utilization.toFixed(1)}%</td>
                                <td className="px-6 py-4">
                                    <span className={`font-medium ${
                                        agent.composite_score >= 80 ? 'text-green-600 dark:text-green-400' :
                                        agent.composite_score >= 60 ? 'text-yellow-600 dark:text-yellow-400' :
                                        'text-red-600 dark:text-red-400'
                                    }`}>
                                        {agent.composite_score.toFixed(1)}
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
