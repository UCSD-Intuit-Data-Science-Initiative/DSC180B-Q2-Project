import React, { useState, useEffect } from 'react';
import { ArrowLeft, User, Phone, Clock, TrendingUp, ArrowUp, ArrowDown, Loader2, Info } from 'lucide-react';
import { Link } from 'react-router';
import { ThemeToggle } from '../components/ThemeToggle';
import { API_BASE } from '../lib/api';

type SortKey = 'name' | 'segment' | 'contacts' | 'aht' | 'utilization' | 'resolution_rate' | 'composite_score';
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

export default function AllAgents() {
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'resolution_rate', direction: 'desc' });
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setLoading(true);
        const asc = sortConfig.direction === 'asc';
        const params = new URLSearchParams({
          n: '300',
          sort_by: sortConfig.key,
          ascending: String(asc),
        });
        const response = await fetch(`${API_BASE}/api/agents?${params}`);
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
  }, [sortConfig]);

  const displayAgents = agents;

  const handleSort = (e: React.MouseEvent, key: SortKey) => {
    e.preventDefault();
    e.stopPropagation();
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const SortIcon = ({ columnKey }: { columnKey: SortKey }) => {
    if (sortConfig.key !== columnKey) return <span className="w-4 h-4 inline-block" />;
    return sortConfig.direction === 'asc' ? <ArrowUp className="w-4 h-4 inline" /> : <ArrowDown className="w-4 h-4 inline" />;
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
                {agents.length} agents. Click any column header to sort.
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

        <div className="bg-slate-100 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
          <div className="flex items-start gap-2">
            <Info className="w-5 h-5 text-slate-500 dark:text-slate-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-slate-700 dark:text-slate-300">Composite Score Formula</p>
              <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                30% Resolution Rate + 20% First-Call Resolution + 20% Efficiency (handle time) + 15% Occupancy + 15% Volume. Each component is 0–100; composite is 0–100.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 flex flex-col h-[80vh]">
            <div className="overflow-auto flex-1 rounded-xl custom-scrollbar">
                <table className="w-full text-left text-sm relative">
                    <thead className="bg-slate-50 dark:bg-slate-900 text-slate-500 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-700 sticky top-0 z-10 shadow-sm">
                        <tr>
                            <th className="px-6 py-4">
                                <button type="button" onClick={(e) => handleSort(e, 'name')} className="flex items-center space-x-2 w-full text-left font-medium text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none py-1 -my-1 rounded cursor-pointer bg-transparent border-0">
                                    <User className="w-4 h-4" />
                                    <span>Agent ID</span>
                                    <SortIcon columnKey="name" />
                                </button>
                            </th>
                            <th className="px-6 py-4">
                                <button type="button" onClick={(e) => handleSort(e, 'segment')} className="flex items-center space-x-2 w-full text-left font-medium text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none py-1 -my-1 rounded cursor-pointer bg-transparent border-0">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Segment</span>
                                    <SortIcon columnKey="segment" />
                                </button>
                            </th>
                            <th className="px-6 py-4">
                                <button type="button" onClick={(e) => handleSort(e, 'contacts')} className="flex items-center space-x-2 w-full text-left font-medium text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none py-1 -my-1 rounded cursor-pointer bg-transparent border-0">
                                    <Phone className="w-4 h-4" />
                                    <span>Contacts</span>
                                    <SortIcon columnKey="contacts" />
                                </button>
                            </th>
                            <th className="px-6 py-4">
                                <button type="button" onClick={(e) => handleSort(e, 'aht')} className="flex items-center space-x-2 w-full text-left font-medium text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none py-1 -my-1 rounded cursor-pointer bg-transparent border-0">
                                    <Clock className="w-4 h-4" />
                                    <span>Avg Handle Time</span>
                                    <SortIcon columnKey="aht" />
                                </button>
                            </th>
                            <th className="px-6 py-4">
                                <button type="button" onClick={(e) => handleSort(e, 'utilization')} className="flex items-center space-x-2 w-full text-left font-medium text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none py-1 -my-1 rounded cursor-pointer bg-transparent border-0">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Utilization</span>
                                    <SortIcon columnKey="utilization" />
                                </button>
                            </th>
                            <th className="px-6 py-4">
                                <button type="button" onClick={(e) => handleSort(e, 'resolution_rate')} className="flex items-center space-x-2 w-full text-left font-medium text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none py-1 -my-1 rounded cursor-pointer bg-transparent border-0">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Resolution Rate</span>
                                    <SortIcon columnKey="resolution_rate" />
                                </button>
                            </th>
                            <th className="px-6 py-4">
                                <button type="button" onClick={(e) => handleSort(e, 'composite_score')} className="flex items-center space-x-2 w-full text-left font-medium text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none py-1 -my-1 rounded cursor-pointer bg-transparent border-0">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Composite</span>
                                    <SortIcon columnKey="composite_score" />
                                </button>
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                        {displayAgents.map((agent) => (
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
                                        agent.resolution_rate >= 90 ? 'text-green-600 dark:text-green-400' :
                                        agent.resolution_rate >= 80 ? 'text-yellow-600 dark:text-yellow-400' :
                                        'text-red-600 dark:text-red-400'
                                    }`}>
                                        {agent.resolution_rate.toFixed(1)}%
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-slate-600 dark:text-slate-400">
                                    {agent.composite_score.toFixed(1)}
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
