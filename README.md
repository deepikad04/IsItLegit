# IsItLegit

**A simulation-based decision training platform that teaches you how to make better financial decisions through experience, not advice.**

Built with React, FastAPI, PostgreSQL, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Tests](https://img.shields.io/badge/Tests-158%20passing-brightgreen)

---

## What is IsItLegit?

Ever wondered if you'd fall for a pump-and-dump scheme? Or panic-sell during a flash crash? IsItLegit puts you in realistic market scenarios and tracks *how* you make decisions — not just the outcome.

After each simulation, Google Gemini analyzes your behavior for cognitive biases like FOMO, loss aversion, and anchoring, then gives you personalized coaching to improve.

### The Learning Loop

```
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
┌────────┐     ┌────────┐     ┌──────────┐     ┌──────┴───┐
│SIMULATE│────▶│ DECIDE │────▶│ REFLECT  │────▶│  COACH   │
│        │     │        │     │          │     │          │
│Pick a  │     │Buy/Sell│     │Gemini    │     │Personal  │
│scenario│     │/Hold in│     │analyzes  │     │tips +    │
│with 14 │     │real-   │     │biases &  │     │profile   │
│market  │     │time    │     │counter-  │     │update    │
│realism │     │under   │     │factuals  │     │          │
│features│     │pressure│     │          │     │          │
└────────┘     └────────┘     └──────────┘     └──────────┘
    ▲                                                  │
    │           ┌──────────┐                           │
    └───────────│  ADAPT   │◀──────────────────────────┘
                │          │
                │AI creates│
                │scenarios │
                │targeting │
                │YOUR weak │
                │spots     │
                └──────────┘
```

Each cycle updates your behavior profile. Your score starts at 0 and grows as Gemini identifies improving decision patterns across simulations.

---

## Design Philosophy

### Why not just use Investopedia Simulator?

Investopedia Simulator is excellent for learning the mechanics of trading and tracking outcomes. IsItLegit targets a different layer of the problem: **how people make decisions under uncertainty**.

Instead of optimizing for profit or competition, we instrument the reasoning process itself — timing, confidence, evidence use, and response to stress — and use AI to reflect on those patterns. This allows us to **separate good decisions from lucky outcomes**, which traditional simulators cannot do.

We intentionally avoided competitive elements because they reinforce outcome bias. Our focus is **self-calibration and decision quality** rather than leaderboard performance, which aligns more closely with how real expertise is built.

### Why single-asset scenarios?

The simulations are simplified by design. By constraining complexity to single-asset scenarios, we can isolate cognitive patterns like timing, confidence, and evidence usage without noise from market microstructure. This allows us to study **decision quality rather than mechanical skill**.

### What drives retention without a leaderboard?

Retention is driven by adaptive scenarios, reflection, and visible improvement in decision calibration. The platform isn't meant to replace professional trading tools, but to **train judgment under uncertainty** — particularly for users before complexity masks their behavior.

---

## Architecture

```mermaid
graph TB
    subgraph Frontend ["Frontend (React 18 + Vite)"]
        UI[Simulation UI<br/>Briefing → Trading → Reflection]
        SSE[SSE Stream Client]
        Charts[Recharts Visualizations]
    end

    subgraph Backend ["Backend (FastAPI)"]
        Auth[JWT Auth]
        SimRouter[Simulation Router<br/>SSE streaming, decisions]
        RefRouter[Reflection Router<br/>5 analysis endpoints]
        ProfRouter[Profile Router<br/>Behavior tracking]

        subgraph Engine ["Simulation Engine (1070+ lines)"]
            Price[Deterministic Price<br/>Timeline Generator]
            Market[14 Market Realism<br/>Features]
            Portfolio[Portfolio & Order<br/>Processing]
        end

        subgraph GeminiSvc ["Gemini Service"]
            Core[Structured JSON Output<br/>+ Schema Validation]
            Think[Thinking Levels<br/>low/high per call type]
            Cache[Context Caching<br/>shared scenario prefix]
            Ground[Search Grounding<br/>+ URL Context]
            Retry[Retries + Rate Limits<br/>+ TTL Cache]
        end
    end

    subgraph Data ["PostgreSQL"]
        Users[(Users)]
        Sims[(Simulations<br/>+ gemini_cache JSONB)]
        Decisions[(Decisions<br/>+ Snapshots)]
        Profiles[(Behavior Profiles)]
    end

    subgraph Gemini ["Google Gemini API"]
        Model[gemini-3-pro]
    end

    UI -->|JWT| Auth
    UI -->|Start/Decide| SimRouter
    SSE <-.->|Price ticks, events| SimRouter
    UI -->|Post-sim analysis| RefRouter
    UI -->|Score & profile| ProfRouter

    SimRouter --> Engine
    RefRouter --> GeminiSvc
    ProfRouter --> GeminiSvc

    GeminiSvc -->|Structured prompts| Model
    Model -->|Schema-validated JSON| GeminiSvc

    SimRouter --> Sims
    SimRouter --> Decisions
    RefRouter --> Sims
    ProfRouter --> Profiles
    Auth --> Users
```

---

## Gemini API Features Used

This project uses **6 distinct Gemini API capabilities** — not just chat completions:

| Feature | How It's Used | Call Types |
|---------|---------------|------------|
| **Structured Output** | Every Gemini response is schema-validated via Pydantic before reaching clients. No raw text — all structured JSON. | All 15+ call types |
| **Thinking Levels** | `low` for real-time nudges/coaching (latency-sensitive), `high` for deep counterfactual analysis and adaptive scenario generation | Configured per call type in `THINKING_LEVELS` dict |
| **Context Caching** | Shared scenario+decision prefix cached across related calls (reflection, counterfactuals, coaching) to reduce token costs and latency | `_get_or_create_context_cache()` — 7 call types share cached prefix |
| **Search Grounding** | `GoogleSearch` tool verifies claim credibility with real web data. Extracts `groundingChunks` (source URIs), `groundingSupports` (citation segments), `web_search_queries` | `verify_claim_credibility()` |
| **URL Context** | `UrlContext` tool reads article URLs to generate custom scenarios from real financial news. Extracts `url_context_metadata` with retrieval statuses | `generate_scenario_from_url()` |
| **Mock Fallbacks** | Every Gemini call has a deterministic heuristic fallback (`USE_MOCK_GEMINI=true`) so the app runs without an API key | All call types |

### All 15+ Gemini Call Types

| Endpoint | Call Type | Thinking | What It Does |
|----------|-----------|----------|--------------|
| SSE stream | `nudge` | low | Real-time coaching nudge during simulation |
| SSE stream | `challenge` | low | "Challenge my reasoning" — scores rationale live |
| POST complete | `reflection` | low | Full post-sim bias analysis |
| POST complete | `counterfactuals` | high | "What if you sold 30s earlier?" alternate timelines |
| GET /why | `why` | low | "Why did I do that?" — behavioral psychology explainer |
| GET /pro-comparison | `pro` | high | Side-by-side expert vs your decisions |
| GET /coaching | `coaching` | low | Personalized tips + behavior profile update |
| GET /bias-heatmap | `bias_heatmap` | low | Time-series bias intensity data for heatmap |
| GET /rationale-review | `rationale_review` | low | Reviews quality of user's stated reasoning |
| POST generate | `adaptive_scenario` | high | AI generates scenario targeting your weaknesses |
| POST /verify-credibility | grounding | low | Fact-checks a claim against real web sources |
| POST /generate-from-url | url_context | high | Creates scenario from a news article URL |
| Profile update | `profile_update` | low | Updates persistent behavior profile |
| Batch analysis | `batch` | high | Multi-simulation pattern analysis |
| Isolation | `isolate` | high | Single-bias deep dive |

---

## 14 Market Realism Features

The simulation engine (`simulation_engine.py`, 1070+ lines) implements these features progressively — easier scenarios use 2-3, the hardest ("The Perfect Storm") uses all 14:

1. **Bid-Ask Spreads** — realistic price friction
2. **Transaction Fees** — fixed + percentage costs
3. **GARCH Volatility Clustering** — vol surges and calm periods
4. **Circuit Breaker Halts** — trading suspended on 7%+ moves
5. **Liquidity Constraints** — partial fills on large orders
6. **Order Types** — limit and stop orders (not just market)
7. **News Latency** — breaking news arrives with realistic delays
8. **Time Pressure Fills** — prices move while you deliberate
9. **Crowd Behavior Model** — herd sentiment affects prices
10. **Margin/Leverage** — 2.5x leverage with margin calls
11. **Correlated Assets** — secondary asset moves in tandem
12. **Macro Indicators** — VIX, interest rates, market breadth
13. **Risk Limits** — max drawdown enforcement
14. **Historical Price Context** — pre-simulation price history for chart analysis

---

## Features

### Simulation Engine
- **14 market realism features** — bid-ask spreads, transaction fees, GARCH volatility clustering, circuit breaker halts, liquidity constraints, margin/leverage, correlated assets, crowd behavior model, and more
- **Real-time SSE streaming** — price ticks, news events, and social signals delivered via Server-Sent Events
- **Progressive difficulty** — easier scenarios use fewer features; harder ones stack them all
- **7 built-in scenarios** including "The Perfect Storm" (difficulty 5, all 14 features active)

### AI Analysis (Google Gemini)
- **Bias detection** — identifies FOMO, loss aversion, anchoring, impulsivity, and social proof patterns
- **"Why did I do that?"** — explains what likely drove each decision using behavioral psychology
- **"What would a pro do?"** — side-by-side comparison with an expert decision path
- **Counterfactual timelines** — "what if you had sold 30 seconds earlier?"
- **Adaptive coaching** — personalized tips based on your evolving behavior profile
- **AI-generated scenarios** — Gemini creates custom scenarios targeting your specific weaknesses
- **Claim verification** — search grounding checks news credibility against real web sources
- **URL-based scenarios** — paste a news article URL and Gemini creates a scenario from it

### Frontend
- **Briefing screen** before each simulation showing active market features
- **Enhanced news panel** with source attribution (Reuters, Bloomberg), "BREAKING" tags, and relative timestamps
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

### Seed Demo Data (optional)

Pre-populate the database with two demo accounts that have full simulation history, AI analysis, and behavior profiles — so you can explore the complete experience immediately:

```bash
cd backend && python seed_demo.py
```

| Account | Email | Password | Simulations | Story Arc |
|---------|-------|----------|-------------|-----------|
| **demo_trader** | demo@isitlegit.com | demo1234 | 20 sims, 30 days | FOMO losses → patience → contrarian mastery (25 → 88 process score) |
| **alex_novice** | alex@isitlegit.com | alex1234 | 12 sims, 10 days | Beginner mistakes → slowly improving (28 → 58 process score) |

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
│   └── tests/            # pytest suite (158 tests)
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

158 backend tests + 29 frontend tests covering schema validation, simulation engine determinism, Gemini advanced features, and UI components.

---

## Roadmap

Planned features for future development:

- **Multi-asset portfolio simulation** — manage a portfolio of 3-5 correlated assets simultaneously, with sector rotation and diversification analysis
- **Team / classroom mode** — instructors create cohorts, assign scenarios, and compare aggregate bias patterns across students
- **Real-time market data integration** — connect to live market feeds to generate scenarios from actual price action as it happens
- **Mobile-native app** — React Native version with push notifications for daily bias training exercises
- **Spaced repetition for bias training** — schedule scenario replays targeting biases that are fading from memory
- **Community scenario marketplace** — users create and share custom scenarios with difficulty ratings and bias tags
- **Export to portfolio tracker** — integrate with brokerage APIs to compare simulated decisions with real trading behavior

---
