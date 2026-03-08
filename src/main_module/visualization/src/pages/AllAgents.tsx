import React, { useMemo, useState } from 'react';
import { ArrowLeft, User, Phone, Clock, TrendingUp, ArrowUp, ArrowDown } from 'lucide-react';
import { Link } from 'react-router';
import { ThemeToggle } from '../components/ThemeToggle';

type SortKey = 'name' | 'status' | 'calls' | 'aht' | 'utilization';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  key: SortKey;
  direction: SortDirection;
}

interface Agent {
  name: string;
  status: 'Online' | 'In Call' | 'Break' | 'Offline';
  calls: number;
  aht: string;
  ahtSeconds: number;
  utilization: string;
  utilizationPercent: number;
}

export default function AllAgents() {
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'calls', direction: 'desc' });

  // Generate full agent list
  const agentsData = useMemo((): Agent[] => {
    const agents: Agent[] = [
      { name: 'Jackie Wang', status: 'Online', calls: 45, aht: '3m 12s', ahtSeconds: 192, utilization: '87%', utilizationPercent: 87 },
      { name: 'Sarah He', status: 'In Call', calls: 38, aht: '2m 55s', ahtSeconds: 175, utilization: '84%', utilizationPercent: 84 },
      { name: 'Hao Zhang', status: 'Break', calls: 41, aht: '3m 05s', ahtSeconds: 185, utilization: '86%', utilizationPercent: 86 },
      { name: 'Sophia Fang', status: 'Online', calls: 32, aht: '3m 40s', ahtSeconds: 220, utilization: '82%', utilizationPercent: 82 },
      { name: 'Michael Chen', status: 'In Call', calls: 43, aht: '2m 48s', ahtSeconds: 168, utilization: '89%', utilizationPercent: 89 },
      { name: 'Lisa Park', status: 'Online', calls: 39, aht: '3m 20s', ahtSeconds: 200, utilization: '85%', utilizationPercent: 85 },
      { name: 'David Liu', status: 'Break', calls: 36, aht: '3m 15s', ahtSeconds: 195, utilization: '83%', utilizationPercent: 83 },
      { name: 'Emily Zhang', status: 'Online', calls: 44, aht: '2m 52s', ahtSeconds: 172, utilization: '88%', utilizationPercent: 88 },
      { name: 'Ryan Kim', status: 'In Call', calls: 40, aht: '3m 08s', ahtSeconds: 188, utilization: '86%', utilizationPercent: 86 },
      { name: 'Jessica Wu', status: 'Online', calls: 37, aht: '3m 25s', ahtSeconds: 205, utilization: '84%', utilizationPercent: 84 },
      { name: 'Kevin Zhao', status: 'Break', calls: 35, aht: '3m 18s', ahtSeconds: 198, utilization: '81%', utilizationPercent: 81 },
      { name: 'Amy Lin', status: 'Online', calls: 42, aht: '2m 58s', ahtSeconds: 178, utilization: '87%', utilizationPercent: 87 },
      { name: 'Tom Martinez', status: 'In Call', calls: 38, aht: '3m 10s', ahtSeconds: 190, utilization: '85%', utilizationPercent: 85 },
      { name: 'Rachel Chang', status: 'Online', calls: 41, aht: '3m 02s', ahtSeconds: 182, utilization: '86%', utilizationPercent: 86 },
      { name: 'James Lee', status: 'Break', calls: 34, aht: '3m 28s', ahtSeconds: 208, utilization: '80%', utilizationPercent: 80 },
      { name: 'Anna Chen', status: 'Offline', calls: 30, aht: '3m 35s', ahtSeconds: 215, utilization: '78%', utilizationPercent: 78 },
      { name: 'Mark Johnson', status: 'Online', calls: 46, aht: '2m 45s', ahtSeconds: 165, utilization: '90%', utilizationPercent: 90 },
      { name: 'Lily Wang', status: 'In Call', calls: 39, aht: '3m 05s', ahtSeconds: 185, utilization: '85%', utilizationPercent: 85 },
      { name: 'Chris Lee', status: 'Break', calls: 33, aht: '3m 22s', ahtSeconds: 202, utilization: '79%', utilizationPercent: 79 },
      { name: 'Michelle Kim', status: 'Online', calls: 40, aht: '3m 12s', ahtSeconds: 192, utilization: '86%', utilizationPercent: 86 },
      { name: 'Daniel Park', status: 'Offline', calls: 28, aht: '3m 40s', ahtSeconds: 220, utilization: '75%', utilizationPercent: 75 },
      { name: 'Sophie Zhang', status: 'In Call', calls: 41, aht: '2m 58s', ahtSeconds: 178, utilization: '87%', utilizationPercent: 87 },
      { name: 'Alex Wu', status: 'Online', calls: 44, aht: '3m 00s', ahtSeconds: 180, utilization: '88%', utilizationPercent: 88 },
      { name: 'Grace Liu', status: 'Break', calls: 36, aht: '3m 18s', ahtSeconds: 198, utilization: '83%', utilizationPercent: 83 },
      { name: 'Eric Chen', status: 'Online', calls: 42, aht: '2m 52s', ahtSeconds: 172, utilization: '87%', utilizationPercent: 87 },
    ];

    // Sort based on current config
    return [...agents].sort((a, b) => {
      let aValue: any = a[sortConfig.key];
      let bValue: any = b[sortConfig.key];

      if (sortConfig.key === 'aht') {
        aValue = a.ahtSeconds;
        bValue = b.ahtSeconds;
      } else if (sortConfig.key === 'utilization') {
        aValue = a.utilizationPercent;
        bValue = b.utilizationPercent;
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
    if (sortConfig.key !== columnKey) return <div className="w-4 h-4" />;
    return sortConfig.direction === 'asc' ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />;
  };

  const getStatusColor = (status: Agent['status']) => {
    switch (status) {
      case 'Online':
        return 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400';
      case 'In Call':
        return 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400';
      case 'Break':
        return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-400';
      case 'Offline':
        return 'bg-slate-100 dark:bg-slate-700/30 text-slate-800 dark:text-slate-400';
    }
  };

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
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/" className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">All Agents</h1>
              <p className="text-slate-500 dark:text-slate-400 text-sm">Complete agent roster with performance metrics. Click column headers to sort.</p>
            </div>
          </div>
          <ThemeToggle />
        </div>

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
                                    <span>Agent Name</span>
                                    <SortIcon columnKey="name" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('status')}
                            >
                                <div className="flex items-center space-x-2">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Status</span>
                                    <SortIcon columnKey="status" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('calls')}
                            >
                                <div className="flex items-center space-x-2">
                                    <Phone className="w-4 h-4" />
                                    <span>Calls Taken</span>
                                    <SortIcon columnKey="calls" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('aht')}
                            >
                                <div className="flex items-center space-x-2">
                                    <Clock className="w-4 h-4" />
                                    <span>Average Handle Time</span>
                                    <SortIcon columnKey="aht" />
                                </div>
                            </th>
                            <th
                              className="px-6 py-4 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors select-none"
                              onClick={() => handleSort('utilization')}
                            >
                                <div className="flex items-center space-x-2">
                                    <TrendingUp className="w-4 h-4" />
                                    <span>Utilization Rate</span>
                                    <SortIcon columnKey="utilization" />
                                </div>
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                        {agentsData.map((agent, idx) => (
                            <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors cursor-pointer">
                                <td className="px-6 py-4">
                                    <Link
                                        to={`/agent/${encodeURIComponent(agent.name)}`}
                                        className="font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:underline"
                                    >
                                        {agent.name}
                                    </Link>
                                </td>
                                <td className="px-6 py-4">
                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
                                        {agent.status}
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.calls}</td>
                                <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.aht}</td>
                                <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{agent.utilization}</td>
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
