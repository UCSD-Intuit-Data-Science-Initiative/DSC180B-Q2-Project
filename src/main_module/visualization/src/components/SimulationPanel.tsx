import React, { useState, useEffect, useMemo } from 'react';
import { Calculator, RefreshCw, Target, Clock, Users, Phone } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { fetchStaffing, StaffingSlot } from '../lib/api';

interface SimulationPanelProps {
  initialCallVolume?: number;
  onReset?: () => void;
  selectedDate?: Date;
  dailyBreakdownData?: Array<{ time: string; calls: number }>;
  onStaffingDataChange?: (data: Array<{ time: string; calls: number; agents: number }>) => void;
  onCurrentStatsChange?: (stats: { currentCallVolume: number; currentRequiredAgents: number; peakCallVolume: number; peakRequiredAgents: number; currentTimeSlot: string; isToday: boolean }) => void;
}

// Calculate required agents for a given call volume and targets
function calculateRequiredAgents(callsPerHalfHour: number, targetSLA: number, targetWaitTime: number, targetOccupancy: number): number {
  const callsPerHour = callsPerHalfHour * 2;

  // Base agents needed for the workload (assuming 3-min average handle time = 20 calls/hr per agent)
  const avgHandleTimeHours = 3 / 60;
  const baseAgents = (callsPerHour * avgHandleTimeHours);

  // Factor for SLA target
  const slaFactor = targetSLA / 90;
  const agentsForSLA = baseAgents * slaFactor * 1.15;

  // Factor for wait time target
  const waitTimeFactor = 30 / Math.max(targetWaitTime, 5);
  const agentsForWaitTime = baseAgents * waitTimeFactor * 1.1;

  // Factor for occupancy target
  const occupancyFactor = 85 / Math.max(targetOccupancy, 50);
  const agentsForOccupancy = baseAgents * occupancyFactor;

  // Take the maximum to ensure all targets are met
  const maxRequired = Math.max(agentsForSLA, agentsForWaitTime, agentsForOccupancy);

  // Add a small buffer for variability
  const finalAgents = Math.ceil(maxRequired * 1.05);

  return Math.max(1, finalAgents);
}

// Get current time slot label
function getCurrentTimeSlot() {
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
    return '23:30';
  }

  return `${finalHour.toString().padStart(2, '0')}:${finalMinute.toString().padStart(2, '0')}`;
}

export function SimulationPanel({ initialCallVolume, onReset, selectedDate, dailyBreakdownData, onStaffingDataChange, onCurrentStatsChange }: SimulationPanelProps) {
  // Target inputs (user controlled)
  const [targetSLA, setTargetSLA] = useState(90);
  const [targetWaitTime, setTargetWaitTime] = useState(30);
  const [targetOccupancy, setTargetOccupancy] = useState(80);

  // Input field values (for typing)
  const [slaInput, setSlaInput] = useState('90');
  const [waitTimeInput, setWaitTimeInput] = useState('30');
  const [occupancyInput, setOccupancyInput] = useState('80');

  // Staffing slots from API
  const [staffingSlots, setStaffingSlots] = useState<StaffingSlot[]>([]);

  // Check if selected date is today
  const isToday = useMemo(() => {
    if (!selectedDate) return true;
    const today = new Date();
    return selectedDate.getDate() === today.getDate() &&
           selectedDate.getMonth() === today.getMonth() &&
           selectedDate.getFullYear() === today.getFullYear();
  }, [selectedDate]);

  // Current time slot (updates every minute if viewing today)
  const [currentTimeSlot, setCurrentTimeSlot] = useState(getCurrentTimeSlot());

  useEffect(() => {
    if (isToday) {
      const updateTime = () => {
        setCurrentTimeSlot(getCurrentTimeSlot());
      };

      updateTime();
      const interval = setInterval(updateTime, 60000);

      return () => clearInterval(interval);
    }
  }, [isToday]);

  // OLD USEEFFECT — using seeded-random dailyBreakdownData + local calculateRequiredAgents formula (replaced with API call)
  // useEffect(() => {
  //   if (!selectedDate) return;
  //   if (dailyBreakdownData && dailyBreakdownData.length > 0) {
  //     if (onStaffingDataChange) {
  //       const staffingData = dailyBreakdownData.map(slot => ({
  //         time: slot.time,
  //         calls: slot.calls,
  //         agents: calculateRequiredAgents(slot.calls, targetSLA, targetWaitTime, targetOccupancy)
  //       }));
  //       onStaffingDataChange(staffingData);
  //     }
  //     fetchStaffing(selectedDate, targetSLA, targetWaitTime, targetOccupancy)
  //       .then(slots => {
  //         const slot = slots.find(s => s.time === currentTimeSlot) ?? slots[0];
  //       })
  //       .catch(console.error);
  //     if (onCurrentStatsChange) {
  //       const currentSlot = dailyBreakdownData.find(slot => slot.time === currentTimeSlot);
  //       const currentCalls = currentSlot ? currentSlot.calls : 0;
  //       const currentAgents = calculateRequiredAgents(currentCalls, targetSLA, targetWaitTime, targetOccupancy);
  //       const peak = Math.max(...dailyBreakdownData.map(slot => slot.calls));
  //       const peakAgents = calculateRequiredAgents(peak, targetSLA, targetWaitTime, targetOccupancy);
  //       onCurrentStatsChange({
  //         currentCallVolume: currentCalls,
  //         currentRequiredAgents: currentAgents,
  //         peakCallVolume: peak,
  //         peakRequiredAgents: peakAgents,
  //         currentTimeSlot,
  //         isToday
  //       });
  //     }
  //   }
  // }, [selectedDate, targetSLA, targetWaitTime, targetOccupancy, dailyBreakdownData, currentTimeSlot, isToday, onStaffingDataChange, onCurrentStatsChange]);

  // NEW USEEFFECT — Fetch staffing data from backend API whenever date or slider targets change
  useEffect(() => {
    if (!selectedDate) return;

    fetchStaffing(selectedDate, targetSLA, targetWaitTime, targetOccupancy)
      .then(slots => {
        setStaffingSlots(slots);

        // Map API slots to the shape DailyBreakdownChart expects
        if (onStaffingDataChange) {
          onStaffingDataChange(slots.map(s => ({
            time: s.time,
            calls: s.predicted_calls,
            agents: s.agents,
          })));
        }

        // Current time slot stats for the stat cards
        if (onCurrentStatsChange) {
          const slot = slots.find(s => s.time === currentTimeSlot) ?? slots[0];
          const peak = Math.max(...slots.map(s => s.predicted_calls));
          const peakAgents = Math.max(...slots.map(s => s.agents));
          onCurrentStatsChange({
            currentCallVolume: slot?.predicted_calls ?? 0,
            currentRequiredAgents: slot?.agents ?? 0,
            peakCallVolume: peak,
            peakRequiredAgents: peakAgents,
            currentTimeSlot,
            isToday,
          });
        }
      })
      .catch(console.error);
  }, [selectedDate, targetSLA, targetWaitTime, targetOccupancy, currentTimeSlot, isToday, onStaffingDataChange, onCurrentStatsChange]);

  const handleReset = () => {
    setTargetSLA(90);
    setTargetWaitTime(30);
    setTargetOccupancy(80);
    setSlaInput('90');
    setWaitTimeInput('30');
    setOccupancyInput('80');

    if (onReset) {
      onReset();
    }
  };

  // Handle input changes for SLA
  const handleSlaInputChange = (value: string) => {
    setSlaInput(value);
    const num = Number(value);
    if (!isNaN(num) && num >= 70 && num <= 99) {
      setTargetSLA(num);
    }
  };

  const handleSlaInputBlur = () => {
    const num = Number(slaInput);
    if (isNaN(num) || num < 70) {
      setSlaInput('70');
      setTargetSLA(70);
    } else if (num > 99) {
      setSlaInput('99');
      setTargetSLA(99);
    }
  };

  // Handle input changes for Wait Time
  const handleWaitTimeInputChange = (value: string) => {
    setWaitTimeInput(value);
    const num = Number(value);
    if (!isNaN(num) && num >= 10 && num <= 60) {
      setTargetWaitTime(num);
    }
  };

  const handleWaitTimeInputBlur = () => {
    const num = Number(waitTimeInput);
    if (isNaN(num) || num < 10) {
      setWaitTimeInput('10');
      setTargetWaitTime(10);
    } else if (num > 60) {
      setWaitTimeInput('60');
      setTargetWaitTime(60);
    }
  };

  // Handle input changes for Occupancy
  const handleOccupancyInputChange = (value: string) => {
    setOccupancyInput(value);
    const num = Number(value);
    if (!isNaN(num) && num >= 60 && num <= 95) {
      setTargetOccupancy(num);
    }
  };

  const handleOccupancyInputBlur = () => {
    const num = Number(occupancyInput);
    if (isNaN(num) || num < 60) {
      setOccupancyInput('60');
      setTargetOccupancy(60);
    } else if (num > 95) {
      setOccupancyInput('95');
      setTargetOccupancy(95);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
            <Calculator className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">Target Metrics</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">Adjust targets to calculate staffing needs</p>
          </div>
        </div>
        <button
          onClick={handleReset}
          className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-all"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Reset</span>
        </button>
      </div>

      {/* Target Metrics - Horizontal */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Target SLA */}
        <div className="bg-slate-50/50 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200/50 dark:border-slate-700/50">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Target className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Target SLA</span>
            </div>
            <div className="flex items-center space-x-1">
              <input
                type="number"
                min="70"
                max="99"
                value={slaInput}
                onChange={(e) => handleSlaInputChange(e.target.value)}
                onBlur={handleSlaInputBlur}
                className="w-12 text-right text-lg font-bold text-blue-600 dark:text-blue-400 bg-transparent border-b border-blue-600/30 dark:border-blue-400/30 focus:outline-none focus:border-blue-600 dark:focus:border-blue-400"
              />
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400">%</span>
            </div>
          </div>
          <input
            type="range"
            min="70"
            max="99"
            value={targetSLA}
            onChange={(e) => {
              const val = Number(e.target.value);
              setTargetSLA(val);
              setSlaInput(val.toString());
            }}
            className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
            <span>70%</span>
            <span>99%</span>
          </div>
        </div>

        {/* Target Wait Time */}
        <div className="bg-slate-50/50 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200/50 dark:border-slate-700/50">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4 text-green-600 dark:text-green-400" />
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Wait Time</span>
            </div>
            <div className="flex items-center space-x-1">
              <input
                type="number"
                min="10"
                max="60"
                value={waitTimeInput}
                onChange={(e) => handleWaitTimeInputChange(e.target.value)}
                onBlur={handleWaitTimeInputBlur}
                className="w-12 text-right text-lg font-bold text-green-600 dark:text-green-400 bg-transparent border-b border-green-600/30 dark:border-green-400/30 focus:outline-none focus:border-green-600 dark:focus:border-green-400"
              />
              <span className="text-lg font-bold text-green-600 dark:text-green-400">s</span>
            </div>
          </div>
          <input
            type="range"
            min="10"
            max="60"
            value={targetWaitTime}
            onChange={(e) => {
              const val = Number(e.target.value);
              setTargetWaitTime(val);
              setWaitTimeInput(val.toString());
            }}
            className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-600"
          />
          <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
            <span>10s</span>
            <span>60s</span>
          </div>
        </div>

        {/* Target Occupancy */}
        <div className="bg-slate-50/50 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200/50 dark:border-slate-700/50">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Users className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Occupancy</span>
            </div>
            <div className="flex items-center space-x-1">
              <input
                type="number"
                min="60"
                max="95"
                value={occupancyInput}
                onChange={(e) => handleOccupancyInputChange(e.target.value)}
                onBlur={handleOccupancyInputBlur}
                className="w-12 text-right text-lg font-bold text-purple-600 dark:text-purple-400 bg-transparent border-b border-purple-600/30 dark:border-purple-400/30 focus:outline-none focus:border-purple-600 dark:focus:border-purple-400"
              />
              <span className="text-lg font-bold text-purple-600 dark:text-purple-400">%</span>
            </div>
          </div>
          <input
            type="range"
            min="60"
            max="95"
            value={targetOccupancy}
            onChange={(e) => {
              const val = Number(e.target.value);
              setTargetOccupancy(val);
              setOccupancyInput(val.toString());
            }}
            className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
          />
          <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
            <span>60%</span>
            <span>95%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
