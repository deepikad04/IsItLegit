"""
IsItLegit - Decision Training Platform
A simulation-based decision training platform powered by Gemini 3
that teaches how to think under uncertainty through behavioral analysis.
"""
import logging
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database import init_db
from routers import (
    auth_router,
    scenarios_router,
    simulations_router,
    reflection_router,
    learning_router,
    profile_router
)

settings = get_settings()


# ── Structured JSON Logging ─────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""
    def format(self, record):
        log = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log["request_id"] = record.request_id
        return json.dumps(log)


def setup_logging():
    """Configure structured logging for all app loggers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


logger = logging.getLogger("isitlegit")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    setup_logging()
    logger.info("Starting IsItLegit API", extra={"gemini_model": settings.gemini_model})
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down IsItLegit API")


# ── OpenAPI metadata ────────────────────────────────────────────────
OPENAPI_TAGS = [
    {"name": "authentication", "description": "User registration, login, and JWT token management"},
    {"name": "scenarios", "description": "Browse and generate simulation scenarios (including AI-adaptive)"},
    {"name": "simulations", "description": "Run simulations: start, stream (SSE), make decisions, challenge mode"},
    {"name": "reflection", "description": "Post-simulation AI analysis: bias detection, pro comparison, calibration, Monte Carlo"},
    {"name": "learning", "description": "Behavioral learning cards based on detected patterns"},
    {"name": "profile", "description": "User behavior profile, playbook, and improvement history"},
    {"name": "ops", "description": "Health checks and operational endpoints"},
]

app = FastAPI(
    title="IsItLegit API",
    description=(
        "# IsItLegit — Behavioral Finance Decision Trainer\n\n"
        "A simulation platform powered by **Gemini 3** that teaches users to recognize "
        "cognitive biases in financial decisions through:\n\n"
        "- Real-time market simulations with live AI coaching\n"
        "- Post-simulation bias analysis & counterfactual timelines\n"
        "- Confidence calibration & Monte Carlo outcome distributions\n"
        "- Adaptive scenarios that target user-specific weaknesses\n\n"
        "**Tech:** FastAPI + PostgreSQL + Gemini 3 Pro + React 18"
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    contact={"name": "IsItLegit Team"},
    license_info={"name": "MIT"},
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter state (required by slowapi)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from routers.auth import limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include routers
app.include_router(auth_router)
app.include_router(scenarios_router)
app.include_router(simulations_router)
app.include_router(reflection_router)
app.include_router(learning_router)
app.include_router(profile_router)


# ── Request logging middleware ───────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, and response time."""
    import time
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    if not request.url.path.startswith("/health"):
        logger.info(
            "%s %s %d %.0fms",
            request.method, request.url.path, response.status_code, duration_ms,
        )
    return response


@app.get("/", tags=["ops"])
async def root():
    """Root endpoint with API discovery links."""
    return {
        "name": "IsItLegit API",
        "version": "1.0.0",
        "description": "Behavioral Finance Decision Training Platform — powered by Gemini 3",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["ops"])
async def health_check():
    """Health check for Docker/k8s readiness probes. Returns DB and Gemini status."""
    from sqlalchemy import text
    from database import SessionLocal
    checks = {"api": "ok"}
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"
    checks["gemini_mode"] = "mock" if settings.use_mock_gemini else "live"
    checks["gemini_model"] = settings.gemini_model
    overall = "healthy" if checks["database"] == "ok" else "degraded"
    return {"status": overall, "service": "isitlegit-api", "version": "1.0.0", "checks": checks}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
