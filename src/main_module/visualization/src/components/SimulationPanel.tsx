import React, { useState, useEffect } from 'react';
import { Users, Calculator, ArrowRight, RefreshCw, AlertTriangle, TrendingUp, TrendingDown, Target, Clock, Activity } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { fetchStaffing } from '../lib/api';

interface SimulationPanelProps {
  initialCallVolume?: number;
  onReset?: () => void;
  selectedDate?: Date;
  dailyBreakdownData?: Array<{ time: string; calls: number }>;
}

export function SimulationPanel({ initialCallVolume, onReset, selectedDate, dailyBreakdownData }: SimulationPanelProps) {
  // Target inputs (user controlled)
  const [targetSLA, setTargetSLA] = useState(90);
  const [targetWaitTime, setTargetWaitTime] = useState(30);
  const [targetOccupancy, setTargetOccupancy] = useState(80);

  // Current time slot and call volume (read-only, derived from data)
  const [currentTimeSlot, setCurrentTimeSlot] = useState<string>('12:00');
  const [callsPerHalfHour, setCallsPerHalfHour] = useState(60);

  const [isCalculating, setIsCalculating] = useState(false);

  // Output: Required agent supply
  const [requiredAgents, setRequiredAgents] = useState(45);

  // Update current time slot based on selected date and time
  useEffect(() => {
    if (dailyBreakdownData && dailyBreakdownData.length > 0) {
      const now = new Date();
      const isToday = selectedDate &&
                      selectedDate.getDate() === now.getDate() &&
                      selectedDate.getMonth() === now.getMonth() &&
                      selectedDate.getFullYear() === now.getFullYear();

      let timeSlotIndex = 24; // Default to noon (12:00)

      if (isToday) {
        // Use current time to find the slot
        const hour = now.getHours();
        const minute = now.getMinutes();
        timeSlotIndex = hour * 2 + (minute >= 30 ? 1 : 0);
        // Ensure we don't go beyond the array
        timeSlotIndex = Math.min(timeSlotIndex, dailyBreakdownData.length - 1);
      }

      const slot = dailyBreakdownData[timeSlotIndex];
      setCurrentTimeSlot(slot.time);
      setCallsPerHalfHour(slot.calls);
    }
  }, [dailyBreakdownData, selectedDate]);

  // Fetch required agents from backend whenever date or slider targets change
  useEffect(() => {
    if (!selectedDate) return;
    setIsCalculating(true);
    const timer = setTimeout(() => {
      fetchStaffing(selectedDate, targetSLA, targetWaitTime, targetOccupancy)
        .then(slots => {
          const slot = slots.find(s => s.time === currentTimeSlot) ?? slots[0];
          if (slot) setRequiredAgents(slot.agents);
        })
        .catch(console.error)
        .finally(() => setIsCalculating(false));
    }, 400);
    return () => clearTimeout(timer);
  }, [selectedDate, targetSLA, targetWaitTime, targetOccupancy, currentTimeSlot]);

  const handleReset = () => {
    // Reset to default targets
    setTargetSLA(90);
    setTargetWaitTime(30);
    setTargetOccupancy(80);

    // Notify parent to reset date/charts
    if (onReset) {
      onReset();
    }
  };

  const getAgentSupplyColor = () => {
    // Color based on reasonableness of required agents
    if (requiredAgents > 100) return 'text-red-600 dark:text-red-400';
    if (requiredAgents > 70) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-green-600 dark:text-green-400';
  };

  return (
    <div
      style={{
        background: 'rgba(255, 255, 255, 0.10)',
        boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.10), 0 1px 2px -1px rgba(0, 0, 0, 0.10)'
      }}
      className="backdrop-blur-xl p-6 rounded-xl h-full flex flex-col border border-slate-100 dark:border-slate-700 dark:bg-slate-800/50"
    >
      <div className="flex items-center space-x-2 mb-6">
        <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
          <Calculator className="w-5 h-5 text-purple-600 dark:text-purple-400" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">Performance Simulator</h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">Calculate required staffing</p>
        </div>
      </div>

      <div className="space-y-6 flex-1">
        {/* Call Volume Display (Read-only) */}
        <div className="bg-blue-50/50 dark:bg-blue-900/20 backdrop-blur-sm p-4 rounded-lg border border-blue-100 dark:border-blue-800">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Call Volume</span>
            </div>
            <span className="text-xs text-slate-500 dark:text-slate-400">{currentTimeSlot}</span>
          </div>
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {callsPerHalfHour}
            <span className="text-sm font-normal text-slate-600 dark:text-slate-400 ml-1">per 30 min</span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">From daily demand forecast</p>
        </div>

        <div className="border-t border-slate-100 dark:border-slate-700"></div>

        {/* Target Inputs */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-1">
            <Target className="w-4 h-4" />
            <span>Target Metrics</span>
          </h3>

          {/* Target SLA */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Target SLA</label>
              <div className="flex items-center space-x-2">
                <input
                  type="number"
                  min="50"
                  max="99"
                  value={targetSLA}
                  onChange={(e) => setTargetSLA(Number(e.target.value))}
                  className="w-16 px-2 py-1 text-sm font-bold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-sm font-bold text-blue-600 dark:text-blue-400">%</span>
              </div>
            </div>
            <input
              type="range"
              min="50"
              max="99"
              value={targetSLA}
              onChange={(e) => setTargetSLA(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-600 dark:accent-blue-500"
            />
            <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500 mt-1">
              <span>50%</span>
              <span>99%</span>
            </div>
          </div>

          {/* Target Wait Time */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Target Wait Time</label>
              <div className="flex items-center space-x-2">
                <input
                  type="number"
                  min="5"
                  max="120"
                  value={targetWaitTime}
                  onChange={(e) => setTargetWaitTime(Number(e.target.value))}
                  className="w-16 px-2 py-1 text-sm font-bold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-sm font-bold text-blue-600 dark:text-blue-400">s</span>
              </div>
            </div>
            <input
              type="range"
              min="5"
              max="120"
              value={targetWaitTime}
              onChange={(e) => setTargetWaitTime(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-600 dark:accent-blue-500"
            />
            <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500 mt-1">
              <span>5s</span>
              <span>120s</span>
            </div>
          </div>

          {/* Target Agent Occupancy */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Target Occupancy</label>
              <div className="flex items-center space-x-2">
                <input
                  type="number"
                  min="50"
                  max="95"
                  value={targetOccupancy}
                  onChange={(e) => setTargetOccupancy(Number(e.target.value))}
                  className="w-16 px-2 py-1 text-sm font-bold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-sm font-bold text-blue-600 dark:text-blue-400">%</span>
              </div>
            </div>
            <input
              type="range"
              min="50"
              max="95"
              value={targetOccupancy}
              onChange={(e) => setTargetOccupancy(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-600 dark:accent-blue-500"
            />
            <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500 mt-1">
              <span>50%</span>
              <span>95%</span>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-100 dark:border-slate-700"></div>

        {/* Output: Required Agent Supply */}
        <div className="bg-slate-50/50 dark:bg-slate-900/30 backdrop-blur-sm p-5 rounded-lg border-2 border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-2 mb-3">
            <Users className="w-5 h-5 text-slate-600 dark:text-slate-400" />
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">Required Agent Supply</span>
          </div>
          <AnimatePresence mode="wait">
            <motion.div
              key={requiredAgents}
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className={`text-4xl font-bold ${getAgentSupplyColor()}`}
            >
              {requiredAgents}
              <span className="text-lg font-normal text-slate-600 dark:text-slate-400 ml-2">agents</span>
            </motion.div>
          </AnimatePresence>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
            Minimum staffing to meet all targets
          </p>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-slate-100 dark:border-slate-700">
        <button
          onClick={handleReset}
          className="w-full py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-blue-600 dark:text-blue-400 font-medium rounded-lg hover:bg-blue-50 dark:hover:bg-slate-700 transition-colors flex items-center justify-center space-x-2"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Reset to Current</span>
        </button>
      </div>
    </div>
  );
}
