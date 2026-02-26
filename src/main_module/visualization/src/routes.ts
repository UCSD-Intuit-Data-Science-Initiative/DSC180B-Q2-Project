import { createBrowserRouter } from "react-router";
import Dashboard from "./pages/Dashboard";
import MetricsHistory from "./pages/MetricsHistory";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Dashboard,
  },
  {
    path: "/metrics-history",
    Component: MetricsHistory,
  },
]);
