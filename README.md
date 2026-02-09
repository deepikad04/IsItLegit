# IsItLegit

**A simulation-based decision training platform that helps you practice financial decision-making under pressure — and learn from your mistakes with AI-powered feedback.**

Built with React, FastAPI, PostgreSQL, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

---

## What is IsItLegit?

Ever wondered if you'd fall for a pump-and-dump scheme? Or panic-sell during a flash crash? IsItLegit puts you in realistic market scenarios and tracks *how* you make decisions — not just the outcome.

After each simulation, Google Gemini analyzes your behavior for cognitive biases like FOMO, loss aversion, and anchoring, then gives you personalized coaching to improve.

### How It Works

1. **Pick a scenario** — Choose from 7 built-in scenarios (or let AI generate one targeting your weaknesses)
2. **Trade under pressure** — Make buy/sell/hold decisions in real-time with a ticking clock, breaking news, and social media noise
3. **Get AI feedback** — Gemini analyzes your biases, compares you to expert traders, and coaches you on what to improve
4. **Track your growth** — Your behavior profile evolves across simulations, showing strengths and areas to work on

---

## Features

### Simulation Engine
- **14 market realism features** — bid-ask spreads, transaction fees, GARCH volatility clustering, circuit breaker halts, liquidity constraints, margin/leverage, correlated assets, crowd behavior model, and more
- **Real-time SSE streaming** — price ticks, news events, and social signals delivered via Server-Sent Events
- **Progressive difficulty** — easier scenarios use fewer features; harder ones stack them all

### AI Analysis (Google Gemini)
- **Bias detection** — identifies FOMO, loss aversion, anchoring, impulsivity, and social proof patterns
- **"Why did I do that?"** — explains what likely drove each decision using behavioral psychology
- **"What would a pro do?"** — side-by-side comparison with an expert decision path
- **Counterfactual timelines** — "what if you had sold 30 seconds earlier?"
- **Adaptive coaching** — personalized tips based on your evolving behavior profile
- **AI-generated scenarios** — Gemini creates custom scenarios targeting your specific weaknesses

### Frontend
- **Briefing screen** before each simulation showing active market features
- **Enhanced news panel** with source attribution, "BREAKING" tags, and relative timestamps
- **Social media feed** with fake usernames, avatars, engagement metrics, and platform icons
- **Bias heatmap**, calibration charts, and outcome distribution visualizations
- **Learning module** with bite-sized cards on cognitive biases

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | React 18, Vite, Tailwind CSS, Recharts, Lucide Icons |
| Backend | FastAPI, SQLAlchemy, Pydantic, Python 3.11 |
| Database | PostgreSQL with JSONB columns |
| AI | Google Gemini API (`google-genai` SDK) |
| Infra | Docker Compose, Nginx |

---

## Getting Started

### With Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/deepikad04/IsItLegit.git
cd IsItLegit

# Create your env file
cp backend/.env.example backend/.env
# Edit backend/.env and add your GEMINI_API_KEY

# Start everything
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- PostgreSQL: localhost:5432

### Without Docker

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:5173 and proxies API calls to the backend.

---

## Environment Variables

Create `backend/.env` with these values:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/isitlegit
SECRET_KEY=your-secret-key
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3-pro-preview
USE_MOCK_GEMINI=false
```

Set `USE_MOCK_GEMINI=true` to run without a Gemini API key (uses heuristic fallbacks).

---

## Architecture

```
┌─────────────┐     JWT Auth     ┌──────────────┐     Structured JSON     ┌─────────────┐
│  React UI   │ ───────────────> │  FastAPI API  │ ────────────────────── > │ Google      │
│  (Vite)     │ <─── SSE Stream  │              │ <── Schema-validated ── │ Gemini API  │
└─────────────┘                  └──────┬───────┘                         └─────────────┘
                                        │
                                        v
                                 ┌──────────────┐
                                 │  PostgreSQL   │
                                 │  (JSONB)      │
                                 └──────────────┘
```

---

## Project Structure

```
IsItLegit/
├── backend/
│   ├── routers/          # API endpoints (auth, scenarios, simulations, reflection, profile, learning)
│   ├── models/           # SQLAlchemy models (User, Scenario, Simulation, Decision, BehaviorProfile)
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/
│   │   ├── gemini_service.py       # All Gemini API calls with retries, caching, and fallbacks
│   │   └── simulation_engine.py    # Deterministic market simulation (1070+ lines)
│   ├── data/             # Scenario JSON templates and learning cards
│   └── tests/            # pytest suite for schemas and simulation engine
├── frontend/
│   ├── src/pages/        # Dashboard, Simulation, Reflection, Profile, Learning, Home
│   ├── src/components/   # BiasHeatmap, ProReplayChart, CoachNudge, Layout
│   └── src/api/          # Axios API client
└── docker-compose.yml
```

---

## Running Tests

```bash
cd backend
pytest
```

---

## Acknowledgments

Built as a portfolio project to explore the intersection of behavioral psychology and AI-powered feedback systems.
