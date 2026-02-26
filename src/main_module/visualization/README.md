# Workforce Dashboard — Frontend

React dashboard for the Workforce Optimization project. Built with Vite + React 19 + Tailwind 4 + shadcn/ui.

Runs at **`localhost:5173`** with hot reload.

## Prerequisites

Only [Docker Desktop](https://www.docker.com/products/docker-desktop/) is required — no Node or npm needed locally.

## Running the dashboard

From the **project root**:

```bash
make frontend-up    # build image + start dev server
make frontend-open  # open localhost:5173 in browser
make frontend-logs  # tail Vite output / errors
make frontend-down  # stop the container
```

Hot reload is enabled — edits to any `.tsx` file update the browser instantly without restarting.

## Project structure

```
visualization/
├── App.tsx                     # main layout and page structure
├── main.tsx                    # React entry point
├── index.html                  # HTML shell
├── components/
│   ├── MetricCard.tsx          # KPI cards (CSAT, SLA, wait time, etc.)
│   ├── DemandChart.tsx         # call volume chart (Recharts)
│   ├── SimulationPanel.tsx     # staffing simulation controls
│   ├── figma/
│   │   └── ImageWithFallback.tsx  # image component with error fallback
│   └── ui/                     # shadcn/ui primitives (button, dialog, etc.)
├── styles/
│   └── globals.css             # Tailwind base styles
├── DockerFile                  # dev + production Docker stages
├── vite.config.ts              # Vite config
├── tsconfig.json               # TypeScript config
└── package.json                # npm dependencies
```

## Tech stack

| Tool | Version | Purpose |
|---|---|---|
| React | 19 | UI framework |
| Vite | 6 | Dev server + bundler |
| Tailwind CSS | 4 | Styling |
| shadcn/ui | latest | Component primitives |
| Recharts | 2 | Charts |
| TypeScript | 5.7 | Type safety |
| Docker | any | Containerized dev environment |

## Docker details

The `DockerFile` has two stages:

- **`dev`** (used by `make frontend-up`) — runs Vite dev server with hot reload
- **`production`** — builds static files and serves via nginx on port 80

The `docker-compose.yml` at the project root mounts your local files into the container via a volume, which is what enables hot reload without rebuilding the image.
