from pydantic import BaseModel, Field
from uuid import UUID
from typing import Optional


class ProcessQuality(BaseModel):
    """Assessment of decision-making process quality."""
    score: float = Field(..., ge=0, le=100)
    factors: dict[str, float]  # timing, information_usage, risk_sizing, emotional_indicators
    summary: str


class PatternDetection(BaseModel):
    """Detected behavioral pattern."""
    pattern_name: str  # fomo, loss_aversion, anchoring, etc.
    confidence: float = Field(..., ge=0, le=1)
    evidence: list[str]  # citations from decision logs
    description: str


class Counterfactual(BaseModel):
    """An alternate timeline showing different outcome."""
    timeline_name: str  # "Market Crash", "Extended Rally", etc.
    description: str
    market_changes: str  # what would have been different
    outcome: dict[str, float]  # profit_loss, final_value
    lesson: str  # what this teaches about luck vs skill


class ActionableInsight(BaseModel):
    """Specific recommendation for improvement."""
    title: str
    description: str
    related_pattern: Optional[str] = None
    recommended_card_id: Optional[str] = None  # link to learning card


class ReflectionResponse(BaseModel):
    """Full reflection analysis for a simulation."""
    simulation_id: UUID

    # Outcome (what happened)
    outcome_summary: str  # profit/loss amount
    outcome_type: str  # "profit", "loss", "break_even"

    # Process quality (how well they decided)
    process_quality: ProcessQuality

    # Behavioral analysis
    patterns_detected: list[PatternDetection]

    # Luck vs skill analysis
    luck_factor: float = Field(..., ge=0, le=1)  # 0 = all skill, 1 = all luck
    skill_factor: float = Field(..., ge=0, le=1)
    luck_skill_explanation: str

    # Counterfactual timelines
    counterfactuals: list[Counterfactual]

    # Recommendations
    insights: list[ActionableInsight]

    # Key message
    key_takeaway: str

    # Personalized coaching
    coaching_message: Optional[str] = None


class EvidenceTimestamp(BaseModel):
    """A specific event/data point that influenced a decision."""
    time: int
    event: str
    relevance: str


class DecisionExplanation(BaseModel):
    """Why Gemini detected a specific bias in a specific decision."""
    decision_index: int
    decision_type: str
    timestamp_seconds: int
    detected_bias: str
    explanation: str  # Gemini explains *why* using the user's actual actions
    evidence_from_actions: list[str]
    severity: str  # "minor", "moderate", "significant"
    evidence_timestamps: list[EvidenceTimestamp] | None = None  # Phase 3.2: causal evidence


class WhyThisDecisionResponse(BaseModel):
    """Full 'Why this decision?' breakdown for a simulation."""
    simulation_id: UUID
    explanations: list[DecisionExplanation]
    overall_narrative: str  # Gemini's cohesive story of the user's decision journey


class ProDecision(BaseModel):
    """What a professional trader would have done differently."""
    at_timestamp: int  # seconds into simulation
    user_action: str
    pro_action: str
    pro_reasoning: str
    outcome_difference: str  # how this would have changed the result
    skill_demonstrated: str  # what cognitive skill the pro uses


class ProComparisonResponse(BaseModel):
    """Side-by-side comparison with a pro's decision path."""
    simulation_id: UUID
    pro_decisions: list[ProDecision]
    pro_final_outcome: dict[str, float]  # profit_loss, final_value
    user_final_outcome: dict[str, float]
    key_differences: list[str]
    what_to_practice: list[str]  # specific skills to work on
