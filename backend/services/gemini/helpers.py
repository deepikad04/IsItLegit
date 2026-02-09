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

# ── Thinking level presets per call type ──────────────────────────────────────
# Gemini 3 Pro supports: "low" and "high" only (no medium/minimal).
# Gemini 3 Flash also supports "minimal" and "medium".
# We use Pro-compatible values since gemini_model defaults to gemini-3-pro.
THINKING_LEVELS = {
    # Live/real-time calls — minimize latency
    "nudge": "low",
    "challenge": "low",
    # Post-sim analysis — low thinking for speed (Pro has no "medium")
    "reflection": "low",
    "why": "low",
    "coaching": "low",
    "bias_heatmap": "low",
    "rationale_review": "low",
    "profile_update": "low",
    "playbook": "low",
    "adherence": "low",
    "learning_modules": "low",
    # Deep analysis — maximum reasoning depth
    "counterfactuals": "high",
    "pro": "high",
    "batch": "high",
    "isolate": "high",
    "adaptive_scenario": "high",
    "bias_classifier": "high",
    "confidence_calibration": "high",
    "behavior_history": "high",
}


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

