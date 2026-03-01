import React, { useMemo, useState } from 'react';
import { ArrowLeft, Calendar, Clock, Target, Phone, ArrowUp, ArrowDown } from 'lucide-react';
import { Link } from 'react-router';

type SortKey = 'date' | 'sla' | 'waitTime' | 'calls';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  key: SortKey;
  direction: SortDirection;
}

export default function MetricsHistory() {
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'date', direction: 'desc' });

  const historyData = useMemo(() => {
    const data = [];
    const startDate = new Date('2023-01-01');
    const endDate = new Date('2025-12-31');

    for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
        // Generate pseudo-random realistic data
        const isWeekend = d.getDay() === 0 || d.getDay() === 6;
        const baseCalls = isWeekend ? 4000 : 9000;
        const calls = Math.floor(baseCalls + Math.random() * 3000);

        const baseSla = 92;
        const sla = Math.min(100, (baseSla + (Math.random() * 10 - 5)));

        const baseWait = 40;
        const wait = Math.floor(baseWait + (Math.random() * 20 - 10));

        data.push({
            date: new Date(d),
            sla: Number(sla.toFixed(1)),
            waitTime: wait,
            calls: calls
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
             <p className="text-slate-500 dark:text-slate-400 text-sm">Detailed performance data from 2023 to 2025</p>
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
                        {historyData.slice(0, 500).map((row, idx) => (
                            <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors">
                                <td className="px-6 py-4 font-medium text-slate-900 dark:text-slate-100">
                                    {row.date.toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' })}
                                </td>
                                <td className="px-6 py-4">
                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                        Number(row.sla) >= 90 ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300' :
                                        Number(row.sla) >= 80 ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300' :
                                        'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                                    }`}>
                                        {row.sla}%
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-slate-600 dark:text-slate-400">
                                    {row.waitTime}s
                                </td>
                                <td className="px-6 py-4 text-slate-900 dark:text-slate-100 font-bold">
                                    {row.calls.toLocaleString()}
                                </td>
                            </tr>
                        ))}
                        {historyData.length > 500 && (
                            <tr>
                                <td colSpan={4} className="px-6 py-4 text-center text-slate-500 dark:text-slate-400 italic">
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
