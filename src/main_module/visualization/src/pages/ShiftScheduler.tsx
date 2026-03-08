import React, { useMemo, useState, useEffect } from 'react';
import { ArrowLeft, Calendar, Users, Clock, AlertCircle, CheckCircle, User, ChevronLeft, ChevronRight, Lock, Save, Target, Activity, X, Plus, Minus } from 'lucide-react';
import { Link } from 'react-router';
import { useTheme } from '../context/ThemeContext';
import { ThemeToggle } from '../components/ThemeToggle';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface Agent {
  name: string;
  team: string;
  availability: {
    [day: string]: { start: string; end: string } | null;
  };
  assignedShift?: { day: string; start: string; end: string };
}

interface TimeSlot {
  time: string;
  required: number;
  scheduled: number;
}

// Seeded random function for consistent data
function seededRandom(seed: number) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

export default function ShiftScheduler() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const [selectedWeekOffset, setSelectedWeekOffset] = useState(0);

  // Default to today's day of week
  const getTodayDayIndex = () => {
    const today = new Date();
    const day = today.getDay();
    return day === 0 ? 6 : day - 1; // Convert to 0=Monday format
  };

  const [selectedDay, setSelectedDay] = useState(getTodayDayIndex());
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [expandedTimeSlot, setExpandedTimeSlot] = useState<string | null>(null);
  const [completedDays, setCompletedDays] = useState<Set<string>>(new Set());
  const [editMode, setEditMode] = useState(false);
  const [showCompleteDialog, setShowCompleteDialog] = useState(false);
  const [showShiftPicker, setShowShiftPicker] = useState(false);
  const [agentToSchedule, setAgentToSchedule] = useState<string | null>(null);
  const [showAgentProfile, setShowAgentProfile] = useState<string | null>(null);
  const [hoveredAgent, setHoveredAgent] = useState<string | null>(null);
  const [hoverTimeout, setHoverTimeout] = useState<NodeJS.Timeout | null>(null);

  // Track assigned shifts: { dateKey: { agentName: { start, end } } }
  const [assignedShifts, setAssignedShifts] = useState<{
    [dateKey: string]: { [agentName: string]: { start: string; end: string } };
  }>({});

  // Track saved state for cancel functionality
  const [savedShifts, setSavedShifts] = useState<{
    [dateKey: string]: { [agentName: string]: { start: string; end: string } };
  }>({});

  // Drag state
  const [draggingAgent, setDraggingAgent] = useState<string | null>(null);
  const [dragStartX, setDragStartX] = useState<number | null>(null);
  const [dragStartTime, setDragStartTime] = useState<{ start: string; end: string } | null>(null);
  const [resizingAgent, setResizingAgent] = useState<string | null>(null);
  const [resizingSide, setResizingSide] = useState<'start' | 'end' | null>(null);
  const [tempSortOrder, setTempSortOrder] = useState<string[]>([]);

  // Get current week dates
  const weekDates = useMemo(() => {
    const today = new Date();
    const monday = new Date(today);
    const day = monday.getDay();
    const diff = monday.getDate() - day + (day === 0 ? -6 : 1); // Adjust for Sunday
    monday.setDate(diff + selectedWeekOffset * 7);
    monday.setHours(0, 0, 0, 0);

    const dates = [];
    for (let i = 0; i < 7; i++) {
      const date = new Date(monday);
      date.setDate(monday.getDate() + i);
      dates.push(date);
    }
    return dates;
  }, [selectedWeekOffset]);

  // Check if selected date is in the past
  const selectedDateIsPast = useMemo(() => {
    const selectedDate = weekDates[selectedDay];
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return selectedDate < today;
  }, [weekDates, selectedDay]);

  // Warn user before leaving page in edit mode
  React.useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (editMode) {
        e.preventDefault();
        e.returnValue = '';
        return '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [editMode]);

  // Check if selected day is completed or locked
  const isDayLocked = useMemo(() => {
    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    return selectedDateIsPast || completedDays.has(dateKey);
  }, [weekDates, selectedDay, selectedDateIsPast, completedDays]);

  const handleCompleteDay = () => {
    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    setCompletedDays(prev => new Set([...prev, dateKey]));
    setEditMode(false);
    setShowCompleteDialog(false);
  };

  const handleCancelComplete = () => {
    setShowCompleteDialog(false);
  };

  // Convert time string to minutes from midnight (00:00)
  const timeToMinutes = (time: string): number => {
    const [hours, mins] = time.split(':').map(Number);
    return hours * 60 + mins;
  };

  // Convert minutes from midnight (00:00) to time string
  const minutesToTime = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
  };

  // Convert 24-hour format to 12-hour format with am/pm
  const formatTime12Hour = (time24: string): string => {
    const [hourStr, minuteStr] = time24.split(':');
    const hour = parseInt(hourStr);
    const minute = parseInt(minuteStr);

    // Handle 24:00 as 12am (midnight of next day)
    if (hour === 24 && minute === 0) return '12am';
    if (hour === 0 && minute === 0) return '12am';
    if (hour === 12 && minute === 0) return '12pm';

    const period = hour >= 12 ? 'pm' : 'am';
    const hour12 = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour;

    if (minute === 0) {
      return `${hour12}${period}`;
    }
    return `${hour12}:${minuteStr}${period}`;
  };

  // Snap to 30-minute intervals
  const snapToInterval = (minutes: number): number => {
    return Math.round(minutes / 30) * 30;
  };

  // Get agent availability in minutes
  const getAgentAvailability = (agentName: string): { start: number; end: number } | null => {
    const agent = agents.find(a => a.name === agentName);
    if (!agent) return null;

    const dayName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selectedDay];
    const availability = agent.availability[dayName];
    if (!availability) return null;

    const [startHour, startMin] = availability.start.split(':').map(Number);
    const [endHour, endMin] = availability.end.split(':').map(Number);

    return {
      start: startHour * 60 + startMin,
      end: endHour * 60 + endMin
    };
  };

  // Handle drag start
  const handleDragStart = (agentName: string, shift: { start: string; end: string }, e: React.MouseEvent) => {
    if (!editMode || isDayLocked) return;

    e.preventDefault();
    setDraggingAgent(agentName);
    setDragStartX(e.clientX);
    setDragStartTime(shift);
  };

  // Handle resize start
  const handleResizeStart = (agentName: string, side: 'start' | 'end', e: React.MouseEvent) => {
    if (!editMode || isDayLocked) return;

    e.preventDefault();
    e.stopPropagation();
    setResizingAgent(agentName);
    setResizingSide(side);
    setDragStartX(e.clientX);

    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    const shift = assignedShifts[dateKey]?.[agentName];
    if (shift) {
      setDragStartTime(shift);
    }
  };

  // Handle mouse move
  const handleMouseMove = (e: React.MouseEvent, containerWidth: number) => {
    if (!editMode || isDayLocked) return;

    if (draggingAgent && dragStartX !== null && dragStartTime) {
      const deltaX = e.clientX - dragStartX;
      const deltaMinutes = Math.round((deltaX / containerWidth) * 24 * 60);

      const currentStartMinutes = timeToMinutes(dragStartTime.start);
      const currentEndMinutes = timeToMinutes(dragStartTime.end);
      const duration = currentEndMinutes - currentStartMinutes;

      let newStartMinutes = snapToInterval(currentStartMinutes + deltaMinutes);
      let newEndMinutes = newStartMinutes + duration;

      // Check availability bounds
      const availability = getAgentAvailability(draggingAgent);
      if (availability) {
        if (newStartMinutes < availability.start) {
          newStartMinutes = availability.start;
          newEndMinutes = newStartMinutes + duration;
        }
        if (newEndMinutes > availability.end) {
          newEndMinutes = availability.end;
          newStartMinutes = newEndMinutes - duration;
        }
      }

      // Ensure within 00:00-24:00 (0 to 1440 minutes)
      newStartMinutes = Math.max(0, Math.min(newStartMinutes, 24 * 60 - duration));
      newEndMinutes = newStartMinutes + duration;

      const newStart = minutesToTime(newStartMinutes);
      const newEnd = minutesToTime(newEndMinutes);

      const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
      setAssignedShifts(prev => ({
        ...prev,
        [dateKey]: {
          ...prev[dateKey],
          [draggingAgent]: { start: newStart, end: newEnd }
        }
      }));
    } else if (resizingAgent && resizingSide && dragStartX !== null && dragStartTime) {
      const deltaX = e.clientX - dragStartX;
      const deltaMinutes = Math.round((deltaX / containerWidth) * 24 * 60);

      const currentStartMinutes = timeToMinutes(dragStartTime.start);
      const currentEndMinutes = timeToMinutes(dragStartTime.end);

      let newStartMinutes = currentStartMinutes;
      let newEndMinutes = currentEndMinutes;

      if (resizingSide === 'start') {
        newStartMinutes = snapToInterval(currentStartMinutes + deltaMinutes);
        // Minimum shift duration: 2 hours (120 minutes)
        if (newStartMinutes >= currentEndMinutes - 120) {
          newStartMinutes = currentEndMinutes - 120;
        }
      } else {
        newEndMinutes = snapToInterval(currentEndMinutes + deltaMinutes);
        // Minimum shift duration: 2 hours (120 minutes)
        if (newEndMinutes <= currentStartMinutes + 120) {
          newEndMinutes = currentStartMinutes + 120;
        }
      }

      // Check availability bounds
      const availability = getAgentAvailability(resizingAgent);
      if (availability) {
        newStartMinutes = Math.max(availability.start, newStartMinutes);
        newEndMinutes = Math.min(availability.end, newEndMinutes);
      }

      // Ensure within 00:00-24:00
      newStartMinutes = Math.max(0, newStartMinutes);
      newEndMinutes = Math.min(24 * 60, newEndMinutes);

      const newStart = minutesToTime(newStartMinutes);
      const newEnd = minutesToTime(newEndMinutes);

      const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
      setAssignedShifts(prev => ({
        ...prev,
        [dateKey]: {
          ...prev[dateKey],
          [resizingAgent]: { start: newStart, end: newEnd }
        }
      }));
    }
  };

  // Handle mouse up
  const handleMouseUp = () => {
    if (draggingAgent) {
      setDraggingAgent(null);
      setDragStartX(null);
      setDragStartTime(null);
      setTempSortOrder([]); // Clear temp sort order
    }
    if (resizingAgent) {
      setResizingAgent(null);
      setResizingSide(null);
      setDragStartX(null);
      setDragStartTime(null);
    }
  };

  // Generate agent availability
  const agents = useMemo((): Agent[] => {
    const agentNames = [
      { name: 'Jackie Wang', team: 'Tax Expert' },
      { name: 'Sarah He', team: 'Payroll Specialist' },
      { name: 'Hao Zhang', team: 'Tax Expert' },
      { name: 'Sophia Fang', team: 'Bookkeeping Advisor' },
      { name: 'Michael Chen', team: 'Tax Expert' },
      { name: 'Lisa Park', team: 'Payroll Specialist' },
      { name: 'David Liu', team: 'Bookkeeping Advisor' },
      { name: 'Emily Zhang', team: 'Tax Expert' },
      { name: 'Ryan Kim', team: 'Payroll Specialist' },
      { name: 'Jessica Wu', team: 'Bookkeeping Advisor' },
      { name: 'Kevin Zhao', team: 'Tax Expert' },
      { name: 'Amy Lin', team: 'Payroll Specialist' },
      { name: 'Tom Martinez', team: 'Bookkeeping Advisor' },
      { name: 'Rachel Chang', team: 'Tax Expert' },
      { name: 'James Lee', team: 'Payroll Specialist' },
      { name: 'Anna Chen', team: 'Bookkeeping Advisor' },
      { name: 'Mark Johnson', team: 'Tax Expert' },
      { name: 'Lily Wang', team: 'Payroll Specialist' },
      { name: 'Chris Lee', team: 'Bookkeeping Advisor' },
      { name: 'Michelle Kim', team: 'Tax Expert' },
      { name: 'Daniel Park', team: 'Payroll Specialist' },
      { name: 'Sophie Zhang', team: 'Bookkeeping Advisor' },
      { name: 'Alex Wu', team: 'Tax Expert' },
      { name: 'Grace Liu', team: 'Payroll Specialist' },
      { name: 'Eric Chen', team: 'Tax Expert' },
    ];

    return agentNames.map((agent, idx) => {
      const availability: Agent['availability'] = {};
      const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

      // Generate availability for each day (some agents might not be available certain days)
      dayNames.forEach((day, dayIdx) => {
        const seed = idx * 7 + dayIdx + weekDates[0].getTime();
        const isAvailable = seededRandom(seed) > 0.15; // 85% chance available

        if (isAvailable) {
          const shiftType = seededRandom(seed + 1);
          // All shifts are 8 hours, spread across 24-hour day
          if (shiftType < 0.2) {
            availability[day] = { start: '00:00', end: '08:00' }; // Night shift
          } else if (shiftType < 0.4) {
            availability[day] = { start: '06:00', end: '14:00' }; // Early morning shift
          } else if (shiftType < 0.6) {
            availability[day] = { start: '08:00', end: '16:00' }; // Early shift
          } else if (shiftType < 0.75) {
            availability[day] = { start: '12:00', end: '20:00' }; // Mid shift
          } else if (shiftType < 0.9) {
            availability[day] = { start: '14:00', end: '22:00' }; // Evening shift
          } else {
            availability[day] = { start: '16:00', end: '23:59' }; // Late shift
          }
        } else {
          availability[day] = null; // Not available
        }
      });

      return {
        ...agent,
        availability
      };
    });
  }, [weekDates]);

  // Calculate required staffing by time slot for selected day
  const requiredStaffing = useMemo(() => {
    const slots: TimeSlot[] = [];
    const selectedDate = weekDates[selectedDay];
    const dateKey = selectedDate.toISOString().split('T')[0];
    const seedBase = selectedDate.getFullYear() * 10000 + (selectedDate.getMonth() + 1) * 100 + selectedDate.getDate();
    const dayName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selectedDay];

    // Generate for 24 hours (00:00 to 23:30 = 48 half-hour slots)
    for (let i = 0; i < 48; i++) {
      const hour = Math.floor(i / 2);
      const minute = (i % 2) * 30;
      const timeLabel = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

      // Calculate required agents (demand pattern throughout the day)
      const base = 3;
      const peak1 = 10 * Math.exp(-Math.pow(i - 16, 2) / 30); // Morning peak around 8:00
      const peak2 = 8 * Math.exp(-Math.pow(i - 32, 2) / 30); // Afternoon peak around 16:00
      const nightDip = i < 12 || i > 40 ? -2 : 0; // Lower demand at night
      const randomVal = seededRandom(seedBase + i * 13);
      const noise = randomVal * 2;

      const required = Math.max(2, Math.floor(base + peak1 + peak2 + nightDip + noise));

      // Count how many agents are ASSIGNED for this time
      const dayShifts = assignedShifts[dateKey] || {};
      const scheduled = Object.entries(dayShifts).filter(([agentName, shift]) => {
        const [startHour, startMin] = shift.start.split(':').map(Number);
        const [endHour, endMin] = shift.end.split(':').map(Number);
        const startMinutes = startHour * 60 + startMin;
        const endMinutes = endHour * 60 + endMin;
        const slotMinutes = hour * 60 + minute;

        return slotMinutes >= startMinutes && slotMinutes < endMinutes;
      }).length;

      slots.push({
        time: timeLabel,
        required,
        scheduled
      });
    }

    return slots;
  }, [selectedDay, weekDates, assignedShifts]);

  // Get agents scheduled for a specific time slot
  const getAgentsForTimeSlot = (timeSlot: string) => {
    const [hour, minute] = timeSlot.split(':').map(Number);
    const slotMinutes = hour * 60 + minute;
    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    const dayShifts = assignedShifts[dateKey] || {};

    return Object.entries(dayShifts)
      .filter(([agentName, shift]) => {
        const [startHour, startMin] = shift.start.split(':').map(Number);
        const [endHour, endMin] = shift.end.split(':').map(Number);
        const startMinutes = startHour * 60 + startMin;
        const endMinutes = endHour * 60 + endMin;

        return slotMinutes >= startMinutes && slotMinutes < endMinutes;
      })
      .map(([agentName]) => agents.find(a => a.name === agentName)!)
      .filter(Boolean);
  };

  // Assign shift to an agent
  const assignShift = (agentName: string, start: string, end: string) => {
    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    setAssignedShifts(prev => ({
      ...prev,
      [dateKey]: {
        ...(prev[dateKey] || {}),
        [agentName]: { start, end }
      }
    }));
  };

  // Unassign shift from an agent
  const unassignShift = (agentName: string) => {
    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    setAssignedShifts(prev => {
      const newShifts = { ...prev };
      if (newShifts[dateKey]) {
        const dayShifts = { ...newShifts[dateKey] };
        delete dayShifts[agentName];
        newShifts[dateKey] = dayShifts;
      }
      return newShifts;
    });
  };

  // Check if agent is assigned a shift for current day
  const isAgentAssigned = (agentName: string) => {
    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    const dayShifts = assignedShifts[dateKey] || {};
    return agentName in dayShifts;
  };

  // Get assigned shift for agent
  const getAgentShift = (agentName: string) => {
    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    const dayShifts = assignedShifts[dateKey] || {};
    return dayShifts[agentName];
  };

  // Get agents available for selected day
  const availableAgents = useMemo(() => {
    const dayName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selectedDay];
    const available = agents.filter(agent => agent.availability[dayName] !== null);

    // Sort by earliest available time, then by name
    return available.sort((a, b) => {
      const aTime = a.availability[dayName]!.start;
      const bTime = b.availability[dayName]!.start;

      if (aTime !== bTime) {
        return aTime.localeCompare(bTime);
      }

      return a.name.localeCompare(b.name);
    });
  }, [selectedDay, agents]);

  const unavailableAgents = useMemo(() => {
    const dayName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selectedDay];
    return agents.filter(agent => agent.availability[dayName] === null);
  }, [selectedDay, agents]);

  const coverageStatus = useMemo(() => {
    const understaffed = requiredStaffing.filter(slot => slot.scheduled < slot.required).length;
    const adequate = requiredStaffing.filter(slot => slot.scheduled >= slot.required).length;
    const totalSlots = requiredStaffing.length;
    const coveragePercent = totalSlots > 0 ? Math.round((adequate / totalSlots) * 100) : 0;

    // Past dates should not show "needs attention"
    const showWarning = !selectedDateIsPast && understaffed > 0;

    return { understaffed, adequate, coveragePercent, showWarning };
  }, [requiredStaffing, selectedDateIsPast]);

  // Calculate occupancy and utilization metrics
  const staffingMetrics = useMemo(() => {
    const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
    const dayShifts = assignedShifts[dateKey] || {};
    const assignedCount = Object.keys(dayShifts).length;

    // Calculate average occupancy (how busy agents are expected to be)
    // Based on call demand vs scheduled agents
    let totalOccupancy = 0;
    let occupancySlots = 0;

    requiredStaffing.forEach(slot => {
      if (slot.scheduled > 0) {
        // Occupancy = (required / scheduled) - capped at 100%
        const occupancy = Math.min(100, (slot.required / slot.scheduled) * 100);
        totalOccupancy += occupancy;
        occupancySlots++;
      }
    });

    const avgOccupancy = occupancySlots > 0 ? totalOccupancy / occupancySlots : 0;

    // Calculate utilization (scheduled vs available)
    const availableCount = availableAgents.length;
    const utilization = availableCount > 0 ? (assignedCount / availableCount) * 100 : 0;

    // Target ranges
    const occupancyTarget = { min: 80, max: 90 };
    const utilizationTarget = { min: 75, max: 95 };

    return {
      assignedCount,
      availableCount,
      avgOccupancy: Math.round(avgOccupancy),
      utilization: Math.round(utilization),
      occupancyTarget,
      utilizationTarget,
      occupancyStatus: avgOccupancy < occupancyTarget.min ? 'low' : avgOccupancy > occupancyTarget.max ? 'high' : 'optimal',
      utilizationStatus: utilization < utilizationTarget.min ? 'low' : utilization > utilizationTarget.max ? 'high' : 'optimal'
    };
  }, [requiredStaffing, assignedShifts, weekDates, selectedDay, availableAgents]);

  const handlePrevWeek = () => setSelectedWeekOffset(prev => prev - 1);
  const handleNextWeek = () => setSelectedWeekOffset(prev => prev + 1);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 p-6 md:p-8 transition-colors duration-300">
      <style>
        {`
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
          * {
            scrollbar-width: thin;
            scrollbar-color: rgba(156, 163, 175, 0.5) transparent;
          }
        `}
      </style>

      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/" className="p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Shift Scheduler</h1>
              <p className="text-slate-500 dark:text-slate-400 text-sm">Match agent availability with required staffing levels</p>
            </div>
          </div>
          <ThemeToggle />
        </div>

        {/* Week Selector */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <Calendar className="w-5 h-5 text-slate-600 dark:text-slate-400" />
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">
                Week of {weekDates[0].toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
              </h2>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={handlePrevWeek}
                className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 transition-colors"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button
                onClick={() => setSelectedWeekOffset(0)}
                className="px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors text-sm font-medium"
              >
                This Week
              </button>
              <button
                onClick={handleNextWeek}
                className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 transition-colors"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Day Selector */}
          <div className="grid grid-cols-7 gap-2">
            {weekDates.map((date, idx) => {
              const isToday = date.toDateString() === new Date().toDateString();
              const isSelected = selectedDay === idx;
              const isPast = (() => {
                const today = new Date();
                today.setHours(0, 0, 0, 0);
                return date < today;
              })();

              return (
                <button
                  key={idx}
                  onClick={() => setSelectedDay(idx)}
                  className={`p-4 rounded-lg transition-all h-24 flex flex-col items-center justify-center ${
                    isSelected
                      ? 'bg-blue-600 text-white shadow-lg'
                      : isToday
                      ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300'
                      : isPast
                      ? 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-700'
                      : 'bg-slate-50 dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300'
                  }`}
                >
                  <div className="text-xs font-medium mb-1">
                    {date.toLocaleDateString('en-US', { weekday: 'short' })}
                  </div>
                  <div className="text-lg font-bold">
                    {date.getDate()}
                  </div>
                  {isToday && !isSelected && (
                    <div className="text-xs mt-1">Today</div>
                  )}
                  {isPast && !isToday && !isSelected && (
                    <div className="text-xs mt-1">Past</div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Coverage Summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5">
            <div className="flex items-center space-x-2 mb-2">
              <Users className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Available Agents</span>
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{availableAgents.length}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{unavailableAgents.length} unavailable</p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5">
            <div className="flex items-center space-x-2 mb-2">
              <Clock className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Time Slots Coverage</span>
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">{coverageStatus.coveragePercent}%</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              {coverageStatus.adequate} adequate, {coverageStatus.understaffed} understaffed
            </p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Agent Occupancy</span>
            </div>
            <p className={`text-3xl font-bold ${
              staffingMetrics.occupancyStatus === 'low' ? 'text-yellow-600 dark:text-yellow-400' :
              staffingMetrics.occupancyStatus === 'high' ? 'text-red-600 dark:text-red-400' :
              'text-green-600 dark:text-green-400'
            }`}>
              {staffingMetrics.avgOccupancy}%
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Target: {staffingMetrics.occupancyTarget.min}-{staffingMetrics.occupancyTarget.max}%
            </p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-5">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
              <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Staff Utilization</span>
            </div>
            <p className={`text-3xl font-bold ${
              staffingMetrics.utilizationStatus === 'low' ? 'text-yellow-600 dark:text-yellow-400' :
              staffingMetrics.utilizationStatus === 'high' ? 'text-red-600 dark:text-red-400' :
              'text-green-600 dark:text-green-400'
            }`}>
              {staffingMetrics.utilization}%
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              {staffingMetrics.assignedCount} of {staffingMetrics.availableCount} scheduled
            </p>
          </div>

          <div className={`rounded-xl shadow-sm border p-5 ${
            coverageStatus.showWarning
              ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-700'
              : isDayLocked
              ? 'bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600'
              : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700'
          }`}>
            <div className="flex items-center space-x-2 mb-2">
              {coverageStatus.showWarning ? (
                <AlertCircle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
              ) : isDayLocked ? (
                <Lock className="w-4 h-4 text-slate-600 dark:text-slate-400" />
              ) : (
                <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
              )}
              <span className={`text-sm font-medium ${
                coverageStatus.showWarning
                  ? 'text-yellow-700 dark:text-yellow-300'
                  : isDayLocked
                  ? 'text-slate-700 dark:text-slate-300'
                  : 'text-green-700 dark:text-green-300'
              }`}>
                Schedule Status
              </span>
            </div>
            <p className={`text-3xl font-bold ${
              coverageStatus.showWarning
                ? 'text-yellow-900 dark:text-yellow-200'
                : isDayLocked
                ? 'text-slate-900 dark:text-slate-200'
                : 'text-green-900 dark:text-green-200'
            }`}>
              {coverageStatus.showWarning ? 'Needs Attention' : isDayLocked ? 'Locked' : 'Fully Covered'}
            </p>
          </div>
        </div>

        {/* Status Banners */}
        {selectedDateIsPast && (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 text-blue-700 dark:text-blue-300">
              <AlertCircle className="w-5 h-5" />
              <p className="text-sm font-medium">
                Viewing historical schedule for {weekDates[selectedDay].toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
              </p>
            </div>
          </div>
        )}

        {!selectedDateIsPast && !isDayLocked && (
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Calendar className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <div>
                  <p className="text-sm font-medium text-slate-900 dark:text-white">
                    Schedule for {weekDates[selectedDay].toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                    {editMode ? 'Edit mode active - Click Complete when finished' : 'Click Edit Schedule to make changes'}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {!editMode ? (
                  <button
                    onClick={() => {
                      // Save current state before entering edit mode
                      setSavedShifts(JSON.parse(JSON.stringify(assignedShifts)));
                      setEditMode(true);
                    }}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium flex items-center space-x-2"
                  >
                    <Save className="w-4 h-4" />
                    <span>Edit Schedule</span>
                  </button>
                ) : (
                  <>
                    <button
                      onClick={() => {
                        // Revert to saved state
                        setAssignedShifts(JSON.parse(JSON.stringify(savedShifts)));
                        setEditMode(false);
                      }}
                      className="px-4 py-2 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors text-sm font-medium flex items-center space-x-2"
                    >
                      <X className="w-4 h-4" />
                      <span>Cancel</span>
                    </button>
                    <button
                      onClick={() => {
                        // Save the current state
                        setSavedShifts(JSON.parse(JSON.stringify(assignedShifts)));
                        setEditMode(false);
                      }}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium flex items-center space-x-2"
                    >
                      <Save className="w-4 h-4" />
                      <span>Save</span>
                    </button>
                    <button
                      onClick={() => setShowCompleteDialog(true)}
                      className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium flex items-center space-x-2"
                    >
                      <CheckCircle className="w-4 h-4" />
                      <span>Complete</span>
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Staffing Visualization and Agent Availability */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Staffing Visualization */}
          <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700">
            <div className="p-6 border-b border-slate-100 dark:border-slate-700">
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">
                Staffing Visualization - {weekDates[selectedDay].toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Visual representation of agent shifts and staffing requirements</p>
            </div>

            <div className="p-6">
              {(() => {
                const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
                const dayShifts = assignedShifts[dateKey] || {};
                const hasScheduledAgents = Object.keys(dayShifts).length > 0;

                if (!hasScheduledAgents) {
                  return (
                    <div className="text-center py-12 text-slate-500 dark:text-slate-400">
                      <Users className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p className="text-sm font-medium">No agents scheduled yet</p>
                      {editMode && !isDayLocked && <p className="text-xs mt-1">Click + button in Agent Availability to add agents to schedule</p>}
                    </div>
                  );
                }

                return (
                  <div className="flex">
                    {/* Fixed name column */}
                    <div className="w-40 flex-shrink-0 pr-4">
                      <div className="h-8 mb-4"></div> {/* Spacer for time axis */}
                      <div className="space-y-3">
                        {(() => {
                      const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
                      const dayShifts = assignedShifts[dateKey] || {};

                      // Sort by start time, then end time, then name (but keep original order during drag)
                      let assignedAgents = Object.entries(dayShifts)
                        .sort(([nameA, shiftA], [nameB, shiftB]) => {
                          const startCompare = shiftA.start.localeCompare(shiftB.start);
                          if (startCompare !== 0) return startCompare;

                          const endCompare = shiftA.end.localeCompare(shiftB.end);
                          if (endCompare !== 0) return endCompare;

                          return nameA.localeCompare(nameB);
                        });

                      // Keep positions stable during drag/resize
                      if ((draggingAgent || resizingAgent) && tempSortOrder.length > 0) {
                        assignedAgents = assignedAgents.sort(([nameA], [nameB]) => {
                          const indexA = tempSortOrder.indexOf(nameA);
                          const indexB = tempSortOrder.indexOf(nameB);
                          if (indexA === -1) return 1;
                          if (indexB === -1) return -1;
                          return indexA - indexB;
                        });
                      } else if (!draggingAgent && !resizingAgent) {
                        // Update order when not actively dragging
                        const currentOrder = assignedAgents.map(([name]) => name);
                        if (currentOrder.join(',') !== tempSortOrder.join(',')) {
                          setTempSortOrder(currentOrder);
                        }
                      }

                      if (assignedAgents.length === 0) {
                        return null;
                      }

                      return assignedAgents.map(([agentName], idx) => {
                        const agent = agents.find(a => a.name === agentName);
                        return (
                          <div
                            key={agentName}
                            className="h-10 flex items-center transition-all duration-300 ease-in-out"
                            style={{ order: idx }}
                          >
                            <div
                              className="flex items-center space-x-2 w-full hover:opacity-80 transition-opacity cursor-pointer"
                              onClick={() => setShowAgentProfile(agentName)}
                              onMouseEnter={() => {
                                const timeout = setTimeout(() => setHoveredAgent(agentName), 500);
                                setHoverTimeout(timeout);
                              }}
                              onMouseLeave={() => {
                                if (hoverTimeout) clearTimeout(hoverTimeout);
                                setHoverTimeout(null);
                                setHoveredAgent(null);
                              }}
                            >
                              <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-xs font-bold text-blue-600 dark:text-blue-400 flex-shrink-0">
                                {agentName.split(' ').map(n => n[0]).join('')}
                              </div>
                              <div className="min-w-0 flex-1">
                                <p className="text-xs font-medium text-slate-900 dark:text-white truncate">{agentName}</p>
                                <p className="text-xs text-slate-500 dark:text-slate-400 truncate">{agent?.team || 'Agent'}</p>
                              </div>
                            </div>
                          </div>
                        );
                        });
                      })()}
                    </div>
                  </div>

                  {/* Timeline with horizontal scroll */}
                <div className="flex-1 overflow-x-auto pb-2">
                  <div className="min-w-[600px] pr-4">{/* Added pr-4 for padding on the right */}
                    {/* Time axis */}
                    <div className="flex mb-4 text-xs text-slate-500 dark:text-slate-400 h-12">
                      {Array.from({ length: 25 }, (_, i) => {
                        const time24 = `${i.toString().padStart(2, '0')}:00`;
                        return (
                          <div key={i} className="flex-1 flex justify-center">
                            <span className="transform -rotate-45 origin-center whitespace-nowrap">
                              {formatTime12Hour(time24)}
                            </span>
                          </div>
                        );
                      })}
                    </div>

                    {/* Agent shift bars */}
                    <div
                      className="space-y-3 select-none"
                      onMouseMove={(e) => {
                        const container = e.currentTarget.querySelector('.shift-timeline');
                        if (container) {
                          handleMouseMove(e, container.getBoundingClientRect().width);
                        }
                      }}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseUp}
                    >
                      {(() => {
                        const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
                        const dayShifts = assignedShifts[dateKey] || {};

                        // Sort by start time, then end time, then name (but keep original order during drag)
                        let assignedAgents = Object.entries(dayShifts)
                          .sort(([nameA, shiftA], [nameB, shiftB]) => {
                            const startCompare = shiftA.start.localeCompare(shiftB.start);
                            if (startCompare !== 0) return startCompare;

                            const endCompare = shiftA.end.localeCompare(shiftB.end);
                            if (endCompare !== 0) return endCompare;

                            return nameA.localeCompare(nameB);
                          });

                        // Keep positions stable during drag/resize
                        if ((draggingAgent || resizingAgent) && tempSortOrder.length > 0) {
                          assignedAgents = assignedAgents.sort(([nameA], [nameB]) => {
                            const indexA = tempSortOrder.indexOf(nameA);
                            const indexB = tempSortOrder.indexOf(nameB);
                            if (indexA === -1) return 1;
                            if (indexB === -1) return -1;
                            return indexA - indexB;
                          });
                        } else if (!draggingAgent && !resizingAgent) {
                          // Update order when not actively dragging
                          const currentOrder = assignedAgents.map(([name]) => name);
                          if (currentOrder.join(',') !== tempSortOrder.join(',')) {
                            setTempSortOrder(currentOrder);
                          }
                        }

                        if (assignedAgents.length === 0) {
                          return (
                            <div className="text-center py-8 text-slate-500 dark:text-slate-400">
                              <Users className="w-8 h-8 mx-auto mb-2 opacity-50" />
                              <p className="text-sm">No agents scheduled yet</p>
                              {editMode && <p className="text-xs mt-1">Click + button below to add agents to schedule</p>}
                            </div>
                          );
                        }

                        return assignedAgents.map(([agentName, shift], idx) => {
                          const agent = agents.find(a => a.name === agentName);
                          const [startHour, startMin] = shift.start.split(':').map(Number);
                          const [endHour, endMin] = shift.end.split(':').map(Number);

                          // Calculate position (00:00 = 0%, 24:00 = 100%)
                          const startPercent = ((startHour + startMin / 60) / 24) * 100;
                          const endPercent = ((endHour + endMin / 60) / 24) * 100;
                          const width = endPercent - startPercent;

                          // Calculate shift duration
                          const durationMinutes = (endHour * 60 + endMin) - (startHour * 60 + startMin);
                          const durationHours = Math.floor(durationMinutes / 60);
                          const durationMins = durationMinutes % 60;
                          const durationLabel = durationMins > 0 ? `${durationHours}h ${durationMins}m` : `${durationHours}h`;

                          // Get agent availability for background
                          const dayName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selectedDay];
                          const availability = agent?.availability[dayName];
                          let availStartPercent = 0;
                          let availWidth = 100;

                          if (availability) {
                            const [availStartHour, availStartMin] = availability.start.split(':').map(Number);
                            const [availEndHour, availEndMin] = availability.end.split(':').map(Number);
                            availStartPercent = ((availStartHour + availStartMin / 60) / 24) * 100;
                            const availEndPercent = ((availEndHour + availEndMin / 60) / 24) * 100;
                            availWidth = availEndPercent - availStartPercent;
                          }

                          const isDragging = draggingAgent === agentName;
                          const isResizing = resizingAgent === agentName;

                          return (
                            <div
                              key={agentName}
                              className="h-10 transition-all duration-300 ease-in-out"
                              style={{ order: idx }}
                            >
                              <div className="relative h-10 bg-slate-100/50 dark:bg-slate-700/30 rounded-md border border-slate-200/50 dark:border-slate-600/50 shift-timeline">
                                {/* Availability background */}
                                {availability && (
                                  <div
                                    className="absolute h-full bg-green-100/40 dark:bg-green-900/20 border-l-2 border-r-2 border-green-300/50 dark:border-green-700/50"
                                    style={{
                                      left: `${availStartPercent}%`,
                                      width: `${availWidth}%`
                                    }}
                                    title={`Available: ${availability.start} - ${availability.end}`}
                                  />
                                )}

                                {/* Shift bar */}
                                <div
                                  className={`absolute h-full bg-gradient-to-r from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-700 rounded-md flex items-center justify-center text-white text-xs font-medium shadow-sm transition-shadow border border-blue-600 dark:border-blue-500 group ${
                                    editMode && !isDayLocked ? 'cursor-move hover:shadow-md' : ''
                                  } ${isDragging || isResizing ? 'shadow-lg ring-2 ring-blue-400 dark:ring-blue-500' : ''}`}
                                  style={{
                                    left: `${startPercent}%`,
                                    width: `${width}%`
                                  }}
                                  onMouseDown={(e) => handleDragStart(agentName, shift, e)}
                                  title={`${agentName}: ${shift.start} - ${shift.end} (${durationLabel})`}
                                >
                                  {/* Resize handle - left */}
                                  {editMode && !isDayLocked && (
                                    <div
                                      className="absolute left-0 top-0 bottom-0 w-2 bg-blue-700/50 dark:bg-blue-500/50 hover:bg-blue-700 dark:hover:bg-blue-500 cursor-ew-resize opacity-0 group-hover:opacity-100 transition-opacity rounded-l-md"
                                      onMouseDown={(e) => handleResizeStart(agentName, 'start', e)}
                                      title="Drag to adjust start time"
                                    />
                                  )}

                                  <span className="truncate px-2 pointer-events-none">{shift.start} - {shift.end}</span>

                                  {/* Resize handle - right */}
                                  {editMode && !isDayLocked && (
                                    <div
                                      className="absolute right-0 top-0 bottom-0 w-2 bg-blue-700/50 dark:bg-blue-500/50 hover:bg-blue-700 dark:hover:bg-blue-500 cursor-ew-resize opacity-0 group-hover:opacity-100 transition-opacity rounded-r-md"
                                      onMouseDown={(e) => handleResizeStart(agentName, 'end', e)}
                                      title="Drag to adjust end time"
                                    />
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        });
                      })()}
                    </div>
                  </div>
                </div>
              </div>
                );
              })()}

              {/* Staffing Requirements Chart - Always visible */}
              <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Staffing Requirements</h3>
                  <div className="flex items-center space-x-4 text-xs">
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 bg-red-500 rounded"></div>
                    <span className="text-slate-600 dark:text-slate-400">Understaffed</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 bg-green-500 rounded"></div>
                    <span className="text-slate-600 dark:text-slate-400">Optimal</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 bg-blue-500 rounded"></div>
                    <span className="text-slate-600 dark:text-slate-400">Overstaffed</span>
                  </div>
                </div>
              </div>
              <div style={{ height: '192px', minHeight: '192px', minWidth: '300px' }}>
                {isMounted ? (
                <ResponsiveContainer width="100%" height="100%" minHeight={192} minWidth={300}>
                  <BarChart data={requiredStaffing.filter((_, idx) => idx % 2 === 0).map(slot => ({
                    ...slot,
                    gap: slot.scheduled < slot.required ? slot.required - slot.scheduled : 0,
                    excess: slot.scheduled > slot.required ? slot.scheduled - slot.required : 0,
                    optimal: Math.min(slot.scheduled, slot.required)
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#374151' : '#e2e8f0'} />
                    <XAxis
                      dataKey="time"
                      stroke={isDark ? '#94a3b8' : '#64748b'}
                      tick={{ fontSize: 11 }}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis
                      stroke={isDark ? '#94a3b8' : '#64748b'}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: isDark ? '#1e293b' : '#ffffff',
                        border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`,
                        borderRadius: '8px',
                        color: isDark ? '#f1f5f9' : '#0f172a'
                      }}
                      formatter={(value: any, name: string) => {
                        if (name === 'gap') return [value, 'Need to Add'];
                        if (name === 'excess') return [value, 'Can Reduce'];
                        if (name === 'optimal') return [value, 'Optimal'];
                        return [value, name];
                      }}
                    />
                    <Legend
                      formatter={(value: string) => {
                        if (value === 'gap') return 'Need to Add';
                        if (value === 'excess') return 'Can Reduce';
                        if (value === 'optimal') return 'Optimal';
                        return value;
                      }}
                    />
                    <Bar dataKey="optimal" stackId="a" fill="#10b981" name="optimal" radius={[0, 0, 0, 0]} />
                    <Bar dataKey="gap" stackId="a" fill="#ef4444" name="gap" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="excess" stackId="a" fill="#3b82f6" name="excess" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-400">
                    Loading chart...
                  </div>
                )}
              </div>
            </div>
            </div>
          </div>

          {/* Agent Availability */}
          <div className="lg:col-span-1 bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 p-6">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-4">Agent Availability</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
              {availableAgents.length} agents available on {weekDates[selectedDay].toLocaleDateString('en-US', { weekday: 'long' })}
            </p>
            <div className="space-y-3 overflow-auto max-h-[600px]">
                <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-3">Available ({availableAgents.length})</h4>
                {availableAgents.map((agent, idx) => {
                  const dayName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selectedDay];
                  const availability = agent.availability[dayName]!;
                  const assigned = isAgentAssigned(agent.name);
                  const assignedShift = getAgentShift(agent.name);

                  return (
                    <div
                      key={idx}
                      className={`flex items-center justify-between p-5 rounded-lg border transition-colors ${
                        assigned
                          ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700'
                          : 'bg-slate-50 dark:bg-slate-700/50 border-slate-200 dark:border-slate-600'
                      }`}
                    >
                      <div
                        className="flex items-center space-x-4 flex-1 hover:opacity-80 transition-opacity cursor-pointer"
                        onClick={() => setShowAgentProfile(agent.name)}
                        onMouseEnter={() => {
                          const timeout = setTimeout(() => setHoveredAgent(agent.name), 500);
                          setHoverTimeout(timeout);
                        }}
                        onMouseLeave={() => {
                          if (hoverTimeout) clearTimeout(hoverTimeout);
                          setHoverTimeout(null);
                          setHoveredAgent(null);
                        }}
                      >
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center text-sm font-bold ${
                          assigned
                            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                            : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
                        }`}>
                          {agent.name.split(' ').map(n => n[0]).join('')}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-900 dark:text-white">{agent.name}</p>
                          <p className="text-xs text-slate-500 dark:text-slate-400">{agent.team}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          {assigned && assignedShift ? (
                            <>
                              <div className="flex items-center space-x-1 text-sm font-medium text-blue-700 dark:text-blue-300">
                                <CheckCircle className="w-3 h-3" />
                                <span>{assignedShift.start} - {assignedShift.end}</span>
                              </div>
                              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                                Scheduled
                              </p>
                            </>
                          ) : (
                            <>
                              <div className="flex items-center space-x-1 text-sm font-medium text-slate-700 dark:text-slate-300">
                                <Clock className="w-3 h-3" />
                                <span>{availability.start} - {availability.end}</span>
                              </div>
                              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                                Available
                              </p>
                            </>
                          )}
                        </div>
                        {editMode && !isDayLocked && (
                          <div className="flex items-center space-x-2">
                            {assigned ? (
                              <button
                                onClick={() => unassignShift(agent.name)}
                                className="p-2 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-lg hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors"
                                title="Remove from schedule"
                              >
                                <Minus className="w-4 h-4" />
                              </button>
                            ) : (
                              <button
                                onClick={() => {
                                  setAgentToSchedule(agent.name);
                                  setShowShiftPicker(true);
                                }}
                                className="p-2 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-lg hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
                                title="Add to schedule"
                              >
                                <Plus className="w-4 h-4" />
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Unavailable Agents */}
              {unavailableAgents.length > 0 && (
                <div className="pt-3 space-y-3 border-t border-slate-100 dark:border-slate-700 mt-6">
                  <h4 className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-3">Unavailable ({unavailableAgents.length})</h4>
                  {unavailableAgents.map((agent, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-4 bg-slate-100/50 dark:bg-slate-800/50 rounded-lg opacity-60 border border-slate-200 dark:border-slate-700"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-sm font-bold text-slate-600 dark:text-slate-400">
                          {agent.name.split(' ').map(n => n[0]).join('')}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-700 dark:text-slate-400">{agent.name}</p>
                          <p className="text-xs text-slate-500 dark:text-slate-500">{agent.team}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-400">
                          Not Available
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Confirmation Dialog */}
        {showCompleteDialog && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full border border-slate-200 dark:border-slate-700 animate-in fade-in zoom-in duration-200">
              <div className="p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-12 h-12 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                    <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-slate-900 dark:text-white">Complete Schedule?</h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      {weekDates[selectedDay].toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
                    </p>
                  </div>
                </div>

                <p className="text-sm text-slate-600 dark:text-slate-300 mb-6">
                  Once completed, this day's schedule will be locked and cannot be edited. Are you sure you want to continue?
                </p>

                <div className="flex items-center space-x-3">
                  <button
                    onClick={handleCancelComplete}
                    className="flex-1 px-4 py-2 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors text-sm font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCompleteDay}
                    className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
                  >
                    Complete Schedule
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Shift Picker Dialog */}
        {showShiftPicker && agentToSchedule && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full border border-slate-200 dark:border-slate-700 animate-in fade-in zoom-in duration-200">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-bold text-slate-900 dark:text-white">Schedule Shift</h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{agentToSchedule}</p>
                  </div>
                  <button
                    onClick={() => {
                      setShowShiftPicker(false);
                      setAgentToSchedule(null);
                    }}
                    className="p-2 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5 text-slate-500 dark:text-slate-400" />
                  </button>
                </div>

                <div className="mb-6">
                  <p className="text-sm text-slate-600 dark:text-slate-300 mb-4">
                    Select shift duration and start time:
                  </p>

                  {(() => {
                    const agent = agents.find(a => a.name === agentToSchedule);
                    if (!agent) return null;

                    const dayName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selectedDay];
                    const availability = agent.availability[dayName];
                    if (!availability) return null;

                    const [availStartHour, availStartMin] = availability.start.split(':').map(Number);
                    const [availEndHour, availEndMin] = availability.end.split(':').map(Number);
                    const availStartMinutes = availStartHour * 60 + availStartMin;
                    const availEndMinutes = availEndHour * 60 + availEndMin;
                    const availDuration = availEndMinutes - availStartMinutes;

                    // Show availability window
                    const availHours = Math.floor(availDuration / 60);
                    const availMins = availDuration % 60;
                    const availLabel = availMins > 0 ? `${availHours}h ${availMins}m` : `${availHours}h`;

                    // Generate shifts for different durations (4h, 6h, 8h, 10h, 12h, or full availability)
                    const shiftDurations = [240, 360, 480, 600, 720]; // in minutes
                    const allPossibleShifts: Array<{start: string, end: string, duration: number, label: string}> = [];

                    // Add shifts for each duration
                    shiftDurations.forEach(duration => {
                      if (duration <= availDuration) {
                        for (let startMin = availStartMinutes; startMin + duration <= availEndMinutes; startMin += 30) {
                          const startHour = Math.floor(startMin / 60);
                          const startMinute = startMin % 60;
                          const endMin = startMin + duration;
                          const endHour = Math.floor(endMin / 60);
                          const endMinute = endMin % 60;

                          const durationHours = Math.floor(duration / 60);
                          const durationMins = duration % 60;
                          const durationLabel = durationMins > 0 ? `${durationHours}h ${durationMins}m` : `${durationHours}h`;

                          allPossibleShifts.push({
                            start: `${startHour.toString().padStart(2, '0')}:${startMinute.toString().padStart(2, '0')}`,
                            end: `${endHour.toString().padStart(2, '0')}:${endMinute.toString().padStart(2, '0')}`,
                            duration,
                            label: durationLabel
                          });
                        }
                      }
                    });

                    // Group by duration for better display
                    const shiftsByDuration = allPossibleShifts.reduce((acc, shift) => {
                      if (!acc[shift.duration]) {
                        acc[shift.duration] = [];
                      }
                      acc[shift.duration].push(shift);
                      return acc;
                    }, {} as Record<number, typeof allPossibleShifts>);

                    if (Object.keys(shiftsByDuration).length === 0) {
                      return (
                        <div className="text-center py-8 text-slate-500 dark:text-slate-400">
                          <AlertCircle className="w-8 h-8 mx-auto mb-2" />
                          <p className="text-sm">No shifts available for this agent's availability window.</p>
                          <p className="text-xs mt-2">Available: {availability.start} - {availability.end} ({availLabel})</p>
                        </div>
                      );
                    }

                    return (
                      <div className="space-y-4">
                        <div className="text-xs text-slate-500 dark:text-slate-400 p-3 bg-slate-50 dark:bg-slate-700/30 rounded-lg border border-slate-200 dark:border-slate-600">
                          <strong>Agent Availability:</strong> {availability.start} - {availability.end} ({availLabel})
                        </div>

                        <div className="max-h-96 overflow-y-auto space-y-4">
                          {Object.entries(shiftsByDuration)
                            .sort(([a], [b]) => Number(b) - Number(a)) // Sort by duration descending
                            .map(([duration, shifts]) => (
                              <div key={duration} className="space-y-2">
                                <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
                                  {shifts[0].label} Shifts
                                </h4>
                                {shifts.map((shift, idx) => (
                                  <button
                                    key={idx}
                                    onClick={() => {
                                      assignShift(agentToSchedule, shift.start, shift.end);
                                      setShowShiftPicker(false);
                                      setAgentToSchedule(null);
                                    }}
                                    className="w-full p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 border border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-600 transition-colors text-left"
                                  >
                                    <div className="flex items-center justify-between">
                                      <div>
                                        <p className="text-sm font-medium text-slate-900 dark:text-white">
                                          {shift.start} - {shift.end}
                                        </p>
                                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{shift.label} shift</p>
                                      </div>
                                      <Clock className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                                    </div>
                                  </button>
                                ))}
                              </div>
                            ))}
                        </div>
                      </div>
                    );
                  })()}
                </div>

                <button
                  onClick={() => {
                    setShowShiftPicker(false);
                    setAgentToSchedule(null);
                  }}
                  className="w-full px-4 py-2 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors text-sm font-medium"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Agent Profile Popup */}
      {(showAgentProfile || hoveredAgent) && (
        <div
          className={`fixed inset-0 ${showAgentProfile ? 'bg-black/50 z-50' : 'pointer-events-none z-40'} flex items-center justify-center p-4`}
          onClick={() => showAgentProfile && setShowAgentProfile(null)}
        >
          <div
            className={`bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full border border-slate-200 dark:border-slate-700 pointer-events-auto ${
              showAgentProfile ? 'animate-in fade-in zoom-in duration-200' : hoveredAgent ? 'animate-in fade-in slide-in-from-bottom-2 duration-150' : ''
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            {(() => {
              const agentName = showAgentProfile || hoveredAgent;
              if (!agentName) return null;

              const agent = agents.find(a => a.name === agentName);
              if (!agent) return null;

              // Get agent shift for selected day
              const dateKey = weekDates[selectedDay].toISOString().split('T')[0];
              const dayShift = assignedShifts[dateKey]?.[agentName];
              const dayName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][selectedDay];
              const availability = agent.availability[dayName];

              return (
                <>
                  <div className="p-6 border-b border-slate-100 dark:border-slate-700 flex items-start justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="w-16 h-16 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-xl font-bold text-blue-600 dark:text-blue-400">
                        {agentName.split(' ').map(n => n[0]).join('')}
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-slate-900 dark:text-white">{agentName}</h3>
                        <p className="text-sm text-slate-500 dark:text-slate-400">{agent.team}</p>
                      </div>
                    </div>
                    {showAgentProfile && (
                      <button
                        onClick={() => setShowAgentProfile(null)}
                        className="p-2 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
                      >
                        <X className="w-5 h-5 text-slate-500 dark:text-slate-400" />
                      </button>
                    )}
                  </div>

                  <div className="p-6 space-y-4">
                    <div>
                      <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                        {dayName} - {weekDates[selectedDay].toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      </h4>

                      {availability ? (
                        <div className="space-y-2">
                          <div className="flex items-center space-x-2 text-sm">
                            <Clock className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                            <span className="text-slate-600 dark:text-slate-400">Available:</span>
                            <span className="font-medium text-slate-900 dark:text-white">
                              {availability.start} - {availability.end}
                            </span>
                          </div>

                          {dayShift ? (
                            <div className="flex items-center space-x-2 text-sm">
                              <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                              <span className="text-slate-600 dark:text-slate-400">Scheduled:</span>
                              <span className="font-medium text-green-700 dark:text-green-300">
                                {dayShift.start} - {dayShift.end}
                              </span>
                            </div>
                          ) : (
                            <div className="flex items-center space-x-2 text-sm">
                              <AlertCircle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                              <span className="text-slate-600 dark:text-slate-400">Not scheduled yet</span>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="flex items-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
                          <X className="w-4 h-4" />
                          <span>Not available on {dayName}</span>
                        </div>
                      )}
                    </div>

                    <div className="pt-4 border-t border-slate-100 dark:border-slate-700">
                      <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">Weekly Availability</h4>
                      <div className="space-y-1">
                        {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map((day, idx) => {
                          const dayAvail = agent.availability[day];
                          return (
                            <div key={day} className={`flex items-center justify-between text-xs py-1 px-2 rounded ${
                              idx === selectedDay ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                            }`}>
                              <span className={`font-medium ${
                                idx === selectedDay
                                  ? 'text-blue-700 dark:text-blue-300'
                                  : 'text-slate-600 dark:text-slate-400'
                              }`}>
                                {day.substring(0, 3)}
                              </span>
                              {dayAvail ? (
                                <span className={`${
                                  idx === selectedDay
                                    ? 'text-blue-600 dark:text-blue-400'
                                    : 'text-slate-500 dark:text-slate-500'
                                }`}>
                                  {dayAvail.start} - {dayAvail.end}
                                </span>
                              ) : (
                                <span className="text-slate-400 dark:text-slate-600">—</span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
