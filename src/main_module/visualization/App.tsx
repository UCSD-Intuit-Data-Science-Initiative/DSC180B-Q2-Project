import React from 'react';
import { MetricCard } from './components/MetricCard';
import { DemandChart } from './components/DemandChart';
import { SimulationPanel } from './components/SimulationPanel';
import { Smile, Clock, Target, TrendingUp } from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      
      <div className="transition-all duration-300">
        <main className="p-6 md:p-8 max-w-7xl mx-auto space-y-8">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Dashboard Overview</h1>
              <p className="text-slate-500 mt-1">Real-time insights and workforce performance metrics.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard 
              title="Customer Satisfaction (CSAT)" 
              value="4.8/5.0" 
              change="+12%" 
              isPositive={true}
              icon={Smile} 
            />
            <MetricCard 
              title="Service Level (SLA)" 
              value="94.2%" 
              change="+2.4%" 
              isPositive={true}
              icon={Target} 
            />
            <MetricCard 
              title="Avg. Waiting Time" 
              value="45s" 
              change="-8s" 
              isPositive={true}
              icon={Clock} 
            />
            <MetricCard 
              title="Total Calls Processed" 
              value="12,450" 
              change="+5.1%" 
              isPositive={true}
              icon={TrendingUp} 
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 min-w-0">
              <DemandChart />
            </div>
            
            <div className="h-full min-w-0">
              <SimulationPanel />
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden">
             <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                <h2 className="text-lg font-bold text-slate-900">Top Performing Agents</h2>
                <button className="text-sm text-blue-600 font-medium hover:underline">View All</button>
             </div>
             <div className="overflow-x-auto">
               <table className="w-full text-left text-sm">
                 <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                   <tr>
                     <th className="px-6 py-4">Agent Name</th>
                     <th className="px-6 py-4">Status</th>
                     <th className="px-6 py-4">Calls Taken</th>
                     <th className="px-6 py-4">Avg Handle Time</th>
                     <th className="px-6 py-4">CSAT Score</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-slate-100">
                   {[
                     { name: 'Jackie Wang', status: 'Online', calls: 45, aht: '3m 12s', csat: '4.9' },
                     { name: 'Sarah He', status: 'In Call', calls: 38, aht: '2m 55s', csat: '4.8' },
                     { name: 'Hao Zhang', status: 'Break', calls: 41, aht: '3m 05s', csat: '4.7' },
                     { name: 'Sophia Fang', status: 'Online', calls: 32, aht: '3m 40s', csat: '4.9' },
                   ].map((agent, idx) => (
                     <tr key={idx} className="hover:bg-slate-50 transition-colors">
                       <td className="px-6 py-4 font-medium text-slate-900">{agent.name}</td>
                       <td className="px-6 py-4">
                         <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                           agent.status === 'Online' ? 'bg-green-100 text-green-800' :
                           agent.status === 'In Call' ? 'bg-blue-100 text-blue-800' :
                           'bg-yellow-100 text-yellow-800'
                         }`}>
                           {agent.status}
                         </span>
                       </td>
                       <td className="px-6 py-4 text-slate-600">{agent.calls}</td>
                       <td className="px-6 py-4 text-slate-600">{agent.aht}</td>
                       <td className="px-6 py-4 text-slate-900 font-bold">{agent.csat}</td>
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
