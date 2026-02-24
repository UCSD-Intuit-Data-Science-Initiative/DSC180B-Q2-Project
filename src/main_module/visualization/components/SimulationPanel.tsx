import React, { useState } from "react";

export function SimulationPanel() {
  const [slaTarget, setSlaTarget] = useState(80);
  const [maxWait, setMaxWait] = useState(60);
  const [maxOccupancy, setMaxOccupancy] = useState(85);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-6 h-full flex flex-col">
      <h2 className="text-lg font-bold text-slate-900 mb-4">Simulation Controls</h2>
      <div className="space-y-5 flex-1">
        <div>
          <label className="text-sm font-medium text-slate-600 flex justify-between">
            SLA Target
            <span className="text-slate-900 font-bold">{slaTarget}%</span>
          </label>
          <input
            type="range"
            min={50}
            max={100}
            value={slaTarget}
            onChange={(e) => setSlaTarget(Number(e.target.value))}
            className="w-full mt-2 accent-blue-600"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-slate-600 flex justify-between">
            Max Wait Time
            <span className="text-slate-900 font-bold">{maxWait}s</span>
          </label>
          <input
            type="range"
            min={10}
            max={300}
            value={maxWait}
            onChange={(e) => setMaxWait(Number(e.target.value))}
            className="w-full mt-2 accent-blue-600"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-slate-600 flex justify-between">
            Max Occupancy
            <span className="text-slate-900 font-bold">{maxOccupancy}%</span>
          </label>
          <input
            type="range"
            min={50}
            max={100}
            value={maxOccupancy}
            onChange={(e) => setMaxOccupancy(Number(e.target.value))}
            className="w-full mt-2 accent-blue-600"
          />
        </div>
      </div>
      <button className="mt-6 w-full bg-blue-600 text-white font-medium py-2.5 rounded-lg hover:bg-blue-700 transition-colors">
        Run Simulation
      </button>
    </div>
  );
}