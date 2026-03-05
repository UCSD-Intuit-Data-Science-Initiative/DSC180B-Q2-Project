import React, { useState } from 'react';
import { DemandForecastChart } from './DemandChart';
import { DailyBreakdownChart } from './DailyBreakdownChart';

interface ForecastingSectionProps {
  weekStart: Date;
  onPrevWeek: () => void;
  onNextWeek: () => void;
  selectedDate: Date;
  onSelectDate: (date: Date, volume: number) => void;
  onJumpToToday: () => void;
  onPrevDay: () => void;
  onNextDay: () => void;
}

export function ForecastingSection({
  weekStart,
  onPrevWeek,
  onNextWeek,
  selectedDate,
  onSelectDate,
  onJumpToToday,
  onPrevDay,
  onNextDay
}: ForecastingSectionProps) {

  return (
    <div className="flex flex-col gap-6">
      <div className="min-h-[500px]">
        <DailyBreakdownChart
          selectedDate={selectedDate}
          onPrevDay={onPrevDay}
          onNextDay={onNextDay}
          onJumpToToday={onJumpToToday}
        />
      </div>
      <DemandForecastChart
        weekStart={weekStart}
        onPrevWeek={onPrevWeek}
        onNextWeek={onNextWeek}
        selectedDate={selectedDate}
        onSelectDate={onSelectDate}
        onJumpToToday={onJumpToToday}
      />
    </div>
  );
}
