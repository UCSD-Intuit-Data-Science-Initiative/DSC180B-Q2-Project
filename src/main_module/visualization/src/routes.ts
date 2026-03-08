import { createBrowserRouter } from "react-router";
import Dashboard from "./pages/Dashboard";
import MetricsHistory from "./pages/MetricsHistory";
import AllAgents from "./pages/AllAgents";
import AgentDetail from "./pages/AgentDetail";
import ShiftScheduler from "./pages/ShiftScheduler";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Dashboard,
  },
  {
    path: "/metrics-history",
    Component: MetricsHistory,
  },
  {
    path: "/all-agents",
    Component: AllAgents,
  },
  {
    path: "/agent/:agentName",
    Component: AgentDetail,
  },
  {
    path: "/shift-scheduler",
    Component: ShiftScheduler,
  },
]);
