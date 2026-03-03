import React from 'react';
import { LucideIcon } from 'lucide-react';
import { motion } from 'motion/react';

interface MetricCardProps {
  title: string;
  value: string;
  change: string;
  isPositive: boolean;
  icon: LucideIcon;
  timeFrame?: string;
  targetText?: string;
  prevWeekValue?: string;
}

export function MetricCard({ title, value, change, isPositive, icon: Icon, timeFrame, targetText, prevWeekValue }: MetricCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="backdrop-blur-xl p-6 rounded-xl flex flex-col justify-between border border-slate-200/50 dark:border-slate-700/50 bg-slate-50/50 dark:bg-slate-800/50 h-[190px]"
    >
      <div className="flex justify-between items-start mb-4">
        <div className="p-2 bg-blue-50/80 dark:bg-blue-900/30 rounded-lg">
          <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
        </div>
        <span className={`text-sm font-medium px-2 py-1 rounded-full ${isPositive ? 'bg-green-100/80 dark:bg-green-900/30 text-green-700 dark:text-green-400' : 'bg-red-100/80 dark:bg-red-900/30 text-red-700 dark:text-red-400'}`}>
          {change}
        </span>
      </div>
      <div>
        <h3 className="text-slate-500 dark:text-slate-400 text-sm font-medium mb-1">{title}</h3>
        <div className="text-2xl font-bold text-slate-900 dark:text-white">{value}</div>
        {targetText && (
          <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">{targetText}</p>
        )}
        {prevWeekValue && (
          <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Previous Week: {prevWeekValue}</p>
        )}
      </div>
    </motion.div>
  );
}
