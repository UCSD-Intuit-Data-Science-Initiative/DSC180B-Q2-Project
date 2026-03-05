import React, { useState, useCallback } from 'react';
import { DailyBreakdownChart } from './DailyBreakdownChart';
import { SimulationPanel } from './SimulationPanel';
import { Activity } from 'lucide-react';

interface WorkforceManagementPanelProps {
  selectedDate: Date;
  onPrevDay: () => void;
  onNextDay: () => void;
  initialCallVolume: number | undefined;
  onReset: () => void;
  dailyBreakdownData: Array<{ time: string; calls: number }>;
  onJumpToToday: () => void;
}

export function WorkforceManagementPanel({
  selectedDate,
  onPrevDay,
  onNextDay,
  initialCallVolume,
  onReset,
  dailyBreakdownData,
  onJumpToToday
}: WorkforceManagementPanelProps) {
  // State to hold staffing data from API (via SimulationPanel)
  const [staffingData, setStaffingData] = useState<Array<{ time: string; calls: number; agents: number }>>([]);

  // State to hold current stats from SimulationPanel
  const [currentStats, setCurrentStats] = useState<{
    currentCallVolume: number;
    currentRequiredAgents: number;
    peakCallVolume: number;
    peakRequiredAgents: number;
    currentTimeSlot: string;
    isToday: boolean;
  } | null>(null);

  const handleStaffingDataChange = useCallback((data: Array<{ time: string; calls: number; agents: number }>) => {
    setStaffingData(data);
  }, []);

  const handleCurrentStatsChange = useCallback((stats: {
    currentCallVolume: number;
    currentRequiredAgents: number;
    peakCallVolume: number;
    peakRequiredAgents: number;
    currentTimeSlot: string;
    isToday: boolean;
  }) => {
    setCurrentStats(stats);
  }, []);

  return (
    <div
      style={{
        background: 'rgba(255, 255, 255, 0.10)',
        boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.10), 0 1px 2px -1px rgba(0, 0, 0, 0.10)'
      }}
      className="backdrop-blur-xl rounded-xl border border-slate-100 dark:border-slate-700 dark:bg-slate-800/50 overflow-hidden"
    >
      {/* Header */}
      <div className="p-6 border-b border-slate-200/50 dark:border-slate-700/50">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-gradient-to-br from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-700 rounded-lg shadow-sm">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900 dark:text-white">Performance Simulator</h2>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">Demand forecasting and staffing optimization</p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6 space-y-8">
        {/* Performance Simulator at top */}
        <SimulationPanel
          initialCallVolume={initialCallVolume}
          onReset={onReset}
          selectedDate={selectedDate}
          dailyBreakdownData={dailyBreakdownData}
          onStaffingDataChange={handleStaffingDataChange}
          onCurrentStatsChange={handleCurrentStatsChange}
        />

        {/* Divider */}
        <div className="border-t border-slate-200/50 dark:border-slate-700/50"></div>

        {/* Charts below */}
        <div className="space-y-6">
          <div className="min-h-[400px]">
            <DailyBreakdownChart
              selectedDate={selectedDate}
              onPrevDay={onPrevDay}
              onNextDay={onNextDay}
              onJumpToToday={onJumpToToday}
              staffingData={staffingData}
              currentCallVolume={currentStats?.currentCallVolume}
              currentRequiredAgents={currentStats?.currentRequiredAgents}
              peakCallVolume={currentStats?.peakCallVolume}
              peakRequiredAgents={currentStats?.peakRequiredAgents}
              currentTimeSlot={currentStats?.currentTimeSlot}
              isToday={currentStats?.isToday}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
