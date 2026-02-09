# IsItLegit

**A simulation-based decision training platform that teaches you how to make better financial decisions through experience, not advice.**

Built with React, FastAPI, PostgreSQL, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Tests](https://img.shields.io/badge/Tests-167%20passing-brightgreen)

---

## Why This Matters

Financial literacy programs teach *what* to do but not *how you actually behave* under pressure. Research shows that **cognitive biases cause 80% of retail investor losses** (Barber & Odean, 2000; Kahneman & Tversky, 1979) — individual investors underperform by 1.5% annually due to overtrading driven by overconfidence alone. Yet no existing tool measures or trains against these biases using actual behavioral data.

IsItLegit is the first platform that:
- **Instruments the decision process itself** — not just outcomes
- **Uses AI to discover behavioral patterns** — biases are emergent from data, not pre-programmed
- **Separates luck from skill** — so users learn from good decisions, not lucky outcomes
- **Adapts to each user** — AI generates personalized scenarios targeting individual weak spots

### Who Benefits
- **Individual investors** learning to recognize their own biases before trading real money
- **Finance educators** who need a tool to demonstrate behavioral finance concepts experientially
- **Financial advisors** who want clients to understand their own risk behavior
- **Anyone making high-stakes decisions** — the cognitive patterns transfer to hiring, negotiation, and strategy

---

## Data Flow: How Decisions Become Insights

```
 USER ACTIONS                    BACKEND PIPELINE                   AI ANALYSIS
 ───────────                    ─────────────────                  ────────────

 ┌──────────┐    SSE stream     ┌─────────────────┐
 │  Trade   │ ──────────────▶  │ Simulation       │
 │  Buy/Sell│    decisions      │ Engine           │
 │  /Hold   │    + rationales   │ (deterministic)  │
 └──────────┘                   │                  │
      │                         │ ● Price timeline │
      │                         │ ● 14 realism     │
      │                         │   features       │
      │                         │ ● Portfolio math  │
      │                         └────────┬─────────┘
      │                                  │
      │         ┌────────────────────────┼────────────────────────┐
      │         │     PostgreSQL         │                        │
      │         │                        ▼                        │
      │         │  ┌──────────┐  ┌──────────────┐  ┌──────────┐  │
      └────────▶│  │Decisions │  │ Simulations  │  │Behavior  │  │
                │  │          │  │              │  │Profiles  │  │
                │  │● type    │  │● final_outcome│ │          │  │
                │  │● timing  │  │● gemini_cache │ │● biases  │  │
                │  │● conf.   │  │● analysis    │  │● style   │  │
                │  │● rationale│ │              │  │● history │  │
                │  └─────┬────┘  └──────┬───────┘  └────┬─────┘  │
                │        │              │               │         │
                └────────┼──────────────┼───────────────┼─────────┘
                         │              │               │
                         ▼              ▼               ▼
                  ┌─────────────────────────────────────────┐
                  │         Gemini Service (18+ calls)      │
                  │                                         │
                  │  Decision trace + features ──▶ Gemini   │
                  │                                  │      │
                  │  Schema-validated JSON ◀──────────┘     │
                  │                                         │
                  │  ● Bias detection    ● Counterfactuals  │
                  │  ● Pro comparison    ● Coaching         │
                  │  ● Confidence cal.   ● History analysis │
                  └──────────────┬──────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────────┐
                  │         Persisted to gemini_cache       │
                  │         (JSONB column on Simulation)    │
                  │                                         │
                  │  First request:  Gemini API call        │
                  │  Subsequent:     Instant DB lookup       │
                  └─────────────────────────────────────────┘
```

---

## Testing Instructions (for Judges)

**Live App:** https://isitlegit-frontend.onrender.com

> Note: The backend is hosted on Render's free tier, so the first request may take ~30s to cold-start. After that, everything is fast.

> **Gemini API auto-fallback:** This app uses the Gemini free tier. If the daily quota is exhausted, the backend automatically detects the 429 error and switches to deterministic heuristic fallbacks for 5 minutes before retrying Gemini. The app remains fully functional either way — AI-generated analysis is richer, but heuristic results still demonstrate the full data pipeline and all UI features. You will never see an error page due to quota limits.

### Pre-seeded Demo Accounts

Two accounts are pre-loaded with raw behavioral data (decisions, rationales, outcomes). AI analysis is generated on-demand by Gemini when you view the reflection page — demonstrating the data-to-insight pipeline live:

| Account | Email | Password | What to See |
|---------|-------|----------|-------------|
| **demo_trader** | `demo@isitlegit.com` | `demo1234` | 19 completed simulations, full progress chart, impact metrics, behavior profile with bias patterns, all reflection tabs |
| **alex_novice** | `alex@isitlegit.com` | `alex1234` | 12 simulations, beginner-level profile, improvement trajectory |

### Suggested Walkthrough (5 min)

1. **Log in** with `demo@isitlegit.com` / `demo1234`
2. **Dashboard** — see progress chart, impact metrics (process quality trend), quick links, and recent simulations
3. **Scenarios page** — browse all scenarios with category/difficulty filters, see AI Challenge and recommended scenario cards
4. **Click any completed simulation** under "Recent Simulations" to view the **Reflection page**
   - Try each tab: Bias Analysis, Counterfactuals ("what if?"), Why Decisions, Pro Comparison, Coaching
   - These are all powered by Gemini with structured JSON output
4. **View Full Profile** — see bias patterns, strengths/weaknesses, improvement trajectory
5. **Start a new simulation** — go to Scenarios and pick any scenario (try "The Perfect Storm" for all 14 market features)
   - Read the **briefing screen** showing active market features
   - Click "Begin Simulation" and trade for 30-60 seconds
   - Notice: real-time price chart, news with source attribution, social feed with avatars, live bias detector alerts
   - Try the "Challenge My Reasoning" button after writing a rationale
6. **End the simulation** and view your AI-generated reflection
7. **Learning tab** — browse bite-sized bias education cards

### Or Register a Fresh Account

Register with any email to see the **new user experience** — empty dashboard, score starts at 0, guided first simulation flow.

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
| **Thinking** | Extended thinking enabled for all calls — Gemini uses internal reasoning before producing structured output | `include_thoughts=True` on all calls |
| **Context Caching** | Shared scenario+decision prefix cached across related calls (reflection, counterfactuals, coaching) to reduce token costs and latency | `_get_or_create_context_cache()` — 7 call types share cached prefix |
| **Search Grounding** | `GoogleSearch` tool verifies claim credibility with real web data. Extracts `groundingChunks` (source URIs), `groundingSupports` (citation segments), `web_search_queries` | `verify_claim_credibility()` |
| **URL Context** | `UrlContext` tool reads article URLs to generate custom scenarios from real financial news. Extracts `url_context_metadata` with retrieval statuses | `generate_scenario_from_url()` |
| **Mock Fallbacks** | Every Gemini call has a deterministic heuristic fallback (`USE_MOCK_GEMINI=true`) so the app runs without an API key | All call types |

### All 18+ Gemini Call Types

| Endpoint | Call Type | What It Does |
|----------|-----------|--------------|
| SSE stream | `nudge` | Real-time coaching nudge during simulation |
| SSE stream | `challenge` | "Challenge my reasoning" — scores rationale live |
| POST complete | `reflection` | Full post-sim bias analysis |
| POST complete | `counterfactuals` | "What if you sold 30s earlier?" alternate timelines |
| GET /why | `why` | "Why did I do that?" — behavioral psychology explainer |
| GET /pro-comparison | `pro` | Side-by-side expert vs your decisions |
| GET /coaching | `coaching` | Personalized tips + behavior profile update |
| GET /bias-heatmap | `bias_heatmap` | Time-series bias intensity data for heatmap |
| GET /rationale-review | `rationale_review` | Reviews quality of user's stated reasoning |
| POST generate | `adaptive_scenario` | AI generates scenario targeting your weaknesses |
| POST /verify-credibility | grounding | Fact-checks a claim against real web sources |
| POST /generate-from-url | url_context | Creates scenario from a news article URL |
| Profile update | `profile_update` | Updates persistent behavior profile |
| GET /bias-classifier | `bias_classifier` | AI classifies biases from qualitative trace + quantitative features |
| GET /confidence-calibration | `confidence_calibration` | AI self-evaluates pattern detections against evidence |
| GET /behavior-history | `behavior_history` | Longitudinal analysis of user's full simulation history |
| Batch analysis | `batch` | Multi-simulation pattern analysis |
| Isolation | `isolate` | Single-bias deep dive |

---

## Market Realism Features

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

### AI Analysis (Google Gemini — Reasoning Layer, Not a Wrapper)

Gemini acts as the **reasoning layer** that discovers patterns from raw behavioral data. We supply structured decision traces and numerical features; Gemini performs the actual analysis. Insights are emergent, not hand-authored.

- **Bias detection** — identifies FOMO, loss aversion, anchoring, impulsivity, and social proof patterns
- **AI bias classifier** — Gemini analyzes both qualitative decision traces AND quantitative features (timing, price context, position sizing, info usage) to produce per-decision bias scores with evidenced reasoning
- **Confidence calibration** — Gemini self-evaluates its own pattern detections against independently gathered observable evidence, producing High/Medium/Low/Insufficient confidence labels
- **Behavioral history analysis** — Gemini analyzes a user's full simulation history to discover emerging patterns, learning trajectories, and decision style — patterns are discovered, not pre-computed
- **"Why did I do that?"** — explains what likely drove each decision using behavioral psychology
- **"What would a pro do?"** — side-by-side comparison with an expert + algorithmic baseline (rule-based momentum/stop-loss trader replaying the same price timeline)
- **Counterfactual timelines** — "what if you had sold 30 seconds earlier?"
- **Adaptive coaching** — personalized tips based on your evolving behavior profile
- **AI-generated scenarios** — Gemini creates custom scenarios targeting your specific weaknesses
- **Claim verification** — search grounding checks news credibility against real web sources
- **URL-based scenarios** — paste a news article URL and Gemini creates a scenario from it

### Frontend
- **Collapsible sidebar navigation** — desktop sidebar collapses to icon-only mode; mobile uses slide-down menu
- **Dedicated Scenarios page** — browse, filter by category/difficulty, see recommended and AI Challenge cards
- **Dashboard with impact metrics** — process quality trend, quick links, progress gated on minimum simulations
- **Briefing screen** before each simulation showing active market features
- **Enhanced news panel** with source attribution (Reuters, Bloomberg), "BREAKING" tags, and relative timestamps
- **Social media feed** with fake usernames, avatars, engagement metrics, and platform icons
- **Share results card** — save reflection results as image or copy as text
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

Pre-populate the database with two demo accounts containing raw behavioral data (decisions, rationales, outcomes, timestamps):

```bash
cd backend && python seed_demo.py
```

| Account | Email | Password | Simulations | Story Arc |
|---------|-------|----------|-------------|-----------|
| **demo_trader** | demo@isitlegit.com | demo1234 | 20 sims, 30 days | FOMO losses → patience → contrarian mastery (25 → 88 process score) |
| **alex_novice** | alex@isitlegit.com | alex1234 | 12 sims, 10 days | Beginner mistakes → slowly improving (28 → 58 process score) |

### Pre-compute AI Analysis (optional, requires GEMINI_API_KEY)

After seeding demo data, run Gemini analysis on the behavioral data and persist results to DB. This ensures instant reflection results without waiting for API calls:

```bash
cd backend && python seed_analysis.py
```

This runs 8 Gemini analysis types (reflection, counterfactuals, why, pro-comparison, bias heatmap, rationale review, bias classifier, coaching) on the 5 most recent simulations per user, plus behavior history analysis. Results are stored in the `gemini_cache` JSONB column.

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
│   ├── routers/              # API endpoints (auth, scenarios, simulations, reflection, profile, learning)
│   ├── models/               # SQLAlchemy models (User, Scenario, Simulation, Decision, BehaviorProfile)
│   ├── schemas/              # Pydantic request/response schemas
│   ├── services/
│   │   ├── gemini/           # Gemini AI package (split from monolith)
│   │   │   ├── service.py    #   GeminiService class — 18+ call types (~3600 lines)
│   │   │   ├── schemas.py    #   Pydantic output schemas for structured JSON (~350 lines)
│   │   │   └── helpers.py    #   Module-level caches, constants, helpers (~135 lines)
│   │   ├── gemini_service.py # Backward-compatible re-export (thin wrapper)
│   │   ├── simulation_engine.py    # Deterministic market simulation + algorithmic pro trader
│   │   ├── bias_classifier.py      # Feature extraction for bias classification
│   │   └── confidence_calibrator.py # Evidence signal gathering for calibration
│   ├── data/                 # Scenario JSON templates and learning cards
│   ├── tests/                # pytest suite (167 tests)
│   ├── seed_demo.py          # Seeds demo users with raw behavioral data
│   └── seed_analysis.py      # Runs Gemini on seeded data, persists AI analysis to DB
├── frontend/
│   ├── src/pages/            # Dashboard, Scenarios, Simulation, Reflection, Profile, Learning, Home
│   ├── src/components/
│   │   ├── reflection/       # Extracted: ProcessGauge, PatternCard, CalibrationCard, etc.
│   │   ├── simulation/       # SimulationBriefing, InfoPanels, etc.
│   │   └── ...               # BiasHeatmap, ProReplayChart, CoachNudge, Layout
│   └── src/api/              # Axios API client
└── docker-compose.yml
```

---

## Running Tests

```bash
cd backend
pytest
```

167 tests covering schema validation, simulation engine determinism, Gemini advanced features, bias classifier, and confidence calibration.

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