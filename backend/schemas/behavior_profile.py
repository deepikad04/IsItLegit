from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


class BiasPattern(BaseModel):
    """A detected bias pattern with score."""
    name: str  # loss_aversion, fomo, anchoring, etc.
    score: float = Field(..., ge=0, le=1)  # 0 = not present, 1 = very strong
    description: str
    trend: str  # "improving", "stable", "worsening"


class ImprovementPoint(BaseModel):
    """A point in the improvement trajectory."""
    date: datetime
    overall_score: float
    simulations_count: int
    key_changes: list[str]


class BehaviorProfileResponse(BaseModel):
    """Full behavior profile for a user."""
    user_id: UUID

    # Strengths and weaknesses
    strengths: list[str]
    weaknesses: list[str]

    # Detailed bias patterns
    bias_patterns: list[BiasPattern]

    # Decision style
    decision_style: str  # "reactive", "analytical", "balanced"
    stress_response: str  # "impulsive", "cautious", "steady"

    # Improvement over time
    overall_score: float = Field(..., ge=0, le=100)
    improvement_trajectory: list[ImprovementPoint]

    # Stats
    total_simulations_analyzed: int
    last_updated: datetime


class ProfileSummary(BaseModel):
    """Quick summary of profile for dashboard."""
    overall_score: float
    top_strength: Optional[str]
    top_weakness: Optional[str]
    recent_trend: str  # "improving", "stable", "declining"
    simulations_this_week: int
    current_streak: int = 0
