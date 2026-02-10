"""Module-level helpers, caches, and constants for the Gemini service."""
"""
Gemini API service — real Gemini calls with structured JSON output,
retries, timeouts, rate-limit handling, TTL caching, thinking-level control,
context caching, Google Search grounding, and URL context support.
Falls back to deterministic heuristics only when use_mock_gemini=True.
"""

import json
import hashlib
import logging
import asyncio
from datetime import datetime
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types
from cachetools import TTLCache
from pydantic import BaseModel, Field, ValidationError

from config import get_settings
from models.simulation import Simulation
from models.scenario import Scenario
from models.decision import Decision
from schemas.reflection import (
    ReflectionResponse,
    ProcessQuality,
    PatternDetection,
    Counterfactual,
    ActionableInsight,
    DecisionExplanation,
    WhyThisDecisionResponse,
    ProDecision,
    ProComparisonResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Module-level response cache keyed on (simulation_id, call_type) ──────────
_cache: TTLCache = TTLCache(maxsize=256, ttl=settings.gemini_cache_ttl_seconds)

# ── Context cache store: maps simulation_id → Gemini cached_content name ─────
_context_cache_store: dict[str, str] = {}

# ── Thinking budget presets per call type (token count) ────────────────────────
# Controls how many tokens Gemini allocates for internal reasoning.
# Higher budgets = deeper reasoning = more latency. Min for Pro = 128.
# -1 = dynamic (model decides based on complexity).
THINKING_BUDGETS = {
    # Live/real-time calls — minimize latency (128 = minimum allowed)
    "nudge": 128,
    "challenge": 256,
    # Post-sim analysis — moderate thinking for balanced quality/speed
    "reflection": 1024,
    "why": 512,
    "coaching": 512,
    "bias_heatmap": 512,
    "rationale_review": 512,
    "profile_update": 512,
    "playbook": 512,
    "adherence": 256,
    "learning_modules": 512,
    # Deep analysis — maximum reasoning depth
    "counterfactuals": 4096,
    "pro": 4096,
    "batch": 8192,
    "isolate": 4096,
    "adaptive_scenario": 4096,
    "bias_classifier": 2048,
    "confidence_calibration": 2048,
    "behavior_history": 4096,
    "chart_analysis": 1024,
}

# Legacy name kept for backward compat (maps budget to level label for logging)
THINKING_LEVELS = {k: ("low" if v <= 512 else "high") for k, v in THINKING_BUDGETS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a compact decision trace string from DB decision rows
# ─────────────────────────────────────────────────────────────────────────────

def _get_market_state(d: Decision) -> dict:
    """Get market state from either the inline blob or the snapshot reference."""
    ms = d.market_state_at_decision
    if not ms and hasattr(d, 'snapshot') and d.snapshot:
        ms = d.snapshot.data or {}
    return ms or {}


def _decision_trace(decisions: list[Decision]) -> str:
    """Serialize decisions into a compact text trace for the prompt."""
    lines: list[str] = []
    for i, d in enumerate(decisions):
        info_viewed = d.info_viewed or []
        panels = [iv.get("panel", "?") if isinstance(iv, dict) else str(iv) for iv in info_viewed]
        market = _get_market_state(d)
        sentiment = market.get("available_info", {}).get("market_sentiment", "unknown")
        events = d.events_since_last or []
        event_str = "; ".join(e.get("content", "") for e in events) if events else "none"

        lines.append(
            f"#{i+1} | t={d.simulation_time}s | {d.decision_type.upper()} "
            f"amt={d.amount or 0} | conf={d.confidence_level or '?'}/5 | "
            f"deliberation={d.time_spent_seconds or 0:.1f}s | "
            f"price=${d.price_at_decision or 0:.2f} | sentiment={sentiment} | "
            f"info_viewed=[{', '.join(panels)}] | events_since=[{event_str}]"
        )
    return "\n".join(lines)


def _scenario_summary(scenario: Scenario) -> str:
    """One-paragraph scenario summary for prompts."""
    init = scenario.initial_data or {}
    events = scenario.events or []
    event_text = "\n".join(
        f"  t={e.get('time',0)}s  {e.get('type','?')}: {e.get('content', e.get('change',''))}"
        for e in events
    )
    return (
        f"Scenario: {scenario.name}\n"
        f"Category: {scenario.category} | Difficulty: {scenario.difficulty}/5\n"
        f"Asset: {init.get('asset','?')} | Start price: ${init.get('price',0)} | "
        f"Starting cash: ${init.get('your_balance',10000)}\n"
        f"Time limit: {scenario.time_pressure_seconds}s | "
        f"Initial sentiment: {init.get('market_sentiment','neutral')}\n"
        f"Events timeline:\n{event_text}"
    )


def _cache_key(simulation_id: str, call_type: str) -> str:
    return f"{simulation_id}:{call_type}"


# ─────────────────────────────────────────────────────────────────────────────
# Main service
# ─────────────────────────────────────────────────────────────────────────────

