import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const sampleData = [
  { time: "6:00", actual: 120, predicted: 115 },
  { time: "7:00", actual: 180, predicted: 175 },
  { time: "8:00", actual: 280, predicted: 290 },
  { time: "9:00", actual: 350, predicted: 340 },
  { time: "10:00", actual: 420, predicted: 410 },
  { time: "11:00", actual: 380, predicted: 395 },
  { time: "12:00", actual: 300, predicted: 310 },
  { time: "13:00", actual: 340, predicted: 330 },
  { time: "14:00", actual: 390, predicted: 385 },
  { time: "15:00", actual: 360, predicted: 370 },
  { time: "16:00", actual: 280, predicted: 275 },
  { time: "17:00", actual: 180, predicted: 190 },
];

export function DemandChart() {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-6 h-full">
      <h2 className="text-lg font-bold text-slate-900 mb-4">Call Demand Forecast</h2>
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={sampleData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#94a3b8" />
          <YAxis tick={{ fontSize: 12 }} stroke="#94a3b8" />
          <Tooltip />
          <Area
            type="monotone"
            dataKey="actual"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.15}
            name="Actual Calls"
          />
          <Area
            type="monotone"
            dataKey="predicted"
            stroke="#8b5cf6"
            fill="#8b5cf6"
            fillOpacity={0.1}
            strokeDasharray="5 5"
            name="Predicted Calls"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}