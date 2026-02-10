"""Pydantic schemas for Gemini structured output validation."""
from typing import Optional

from pydantic import BaseModel, Field

from schemas.reflection import (
    ProcessQuality,
    PatternDetection,
    Counterfactual,
    ActionableInsight,
    DecisionExplanation,
    ProDecision,
)




class _ReflectionGeminiOutput(BaseModel):
    outcome_summary: str
    outcome_type: str
    process_quality: ProcessQuality
    patterns_detected: list[PatternDetection]
    luck_factor: float = Field(ge=0, le=1)
    skill_factor: float = Field(ge=0, le=1)
    luck_skill_explanation: str
    insights: list[ActionableInsight]
    key_takeaway: str
    coaching_message: Optional[str] = None


class _CounterfactualGeminiOutput(BaseModel):
    counterfactuals: list[Counterfactual]


class _WhyGeminiOutput(BaseModel):
    explanations: list[DecisionExplanation]
    overall_narrative: str


class _ProGeminiOutput(BaseModel):
    pro_decisions: list[ProDecision]
    pro_final_outcome: dict[str, float]
    user_final_outcome: dict[str, float]
    key_differences: list[str]
    what_to_practice: list[str]


class _CoachingGeminiOutput(BaseModel):
    coaching_message: str
    persona: Optional[str] = None


class _ProfileUpdateGeminiOutput(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    bias_patterns: dict[str, float]
    decision_style: str
    stress_response: str
    improvement_notes: Optional[str] = None


# ── Batch analysis output ─────────────────────────────────────────────────
class _BatchAnalysisGeminiOutput(BaseModel):
    """Combined schema for reflection + counterfactuals + coaching in one call."""
    outcome_summary: str
    outcome_type: str
    process_quality: ProcessQuality
    patterns_detected: list[PatternDetection]
    luck_factor: float = Field(ge=0, le=1)
    skill_factor: float = Field(ge=0, le=1)
    luck_skill_explanation: str
    insights: list[ActionableInsight]
    key_takeaway: str
    coaching_message: str
    counterfactuals: list[Counterfactual]


# ── Bias heatmap output ───────────────────────────────────────────────────
class _BiasHeatmapEntry(BaseModel):
    timestamp_seconds: int
    decision_index: int
    biases: dict[str, float]
    evidence: str
    intensity: str  # "low", "medium", "high"


class _BiasHeatmapGeminiOutput(BaseModel):
    timeline: list[_BiasHeatmapEntry]
    peak_bias_moment: int
    dominant_bias: str


# ── Rationale review output ───────────────────────────────────────────────
class _RationaleCritique(BaseModel):
    decision_index: int
    user_rationale: str
    critique: str
    quality_score: int = Field(ge=1, le=5)
    missed_factors: list[str]
    reasoning_bias: Optional[str] = None


class _RationaleReviewOutput(BaseModel):
    reviews: list[_RationaleCritique]
    summary: str
    overall_reasoning_quality: int = Field(ge=1, le=5)


# ── Counterfactual isolation output ───────────────────────────────────────
class _IsolatedCounterfactualOutput(BaseModel):
    original_decision: str
    alternative_decision: str
    ripple_effects: list[str]
    original_outcome: dict[str, float]
    alternative_outcome: dict[str, float]
    causal_impact: float
    lesson: str


# ── Playbook output ───────────────────────────────────────────────────────
class _PlaybookOutput(BaseModel):
    dos: list[str]
    donts: list[str]
    key_rules: list[str]
    generated_from: int


class _PlaybookAdherenceOutput(BaseModel):
    adherence_score: float = Field(ge=0, le=100)
    followed: list[str]
    violated: list[str]
    specific_examples: list[str]


# ── Live nudge output ────────────────────────────────────────────────────
class _LiveNudgeOutput(BaseModel):
    message: str
    bias: str


# ── Challenge output ─────────────────────────────────────────────────────
class _ChallengeOutput(BaseModel):
    reasoning_score: int = Field(ge=1, le=5)
    feedback: str


# ── Adaptive scenario output ────────────────────────────────────────────
class _AdaptiveScenarioOutput(BaseModel):
    name: str
    description: str
    difficulty: int = Field(ge=1, le=5)
    category: str
    time_pressure_seconds: int
    initial_data: dict  # includes market_params with realism features
    events: list[dict]
    target_bias: str


# ── Learning module generation output ──────────────────────────────────
class _GeneratedLessonOutput(BaseModel):
    id: str
    title: str
    content: str
    key_insight: Optional[str] = None


class _GeneratedQuizOutput(BaseModel):
    question: str
    options: list[str]
    correct: int = Field(ge=0, le=3)
    explanation: str


class _GeneratedModuleOutput(BaseModel):
    id: str
    title: str
    description: str
    category: str
    icon: str
    lessons: list[_GeneratedLessonOutput]
    quiz: list[_GeneratedQuizOutput]


class _GeneratedModulesGeminiOutput(BaseModel):
    modules: list[_GeneratedModuleOutput]


# ── Search grounding credibility output ──────────────────────────────────
class _CredibilityCheckOutput(BaseModel):
    credibility_score: float = Field(ge=0, le=1)
    verdict: str  # "verified", "partially_verified", "unverified", "likely_false"
    supporting_sources: list[str]
    contradicting_sources: list[str]
    key_finding: str
    risk_level: str  # "low", "medium", "high"
    grounding_source_urls: list[dict] = Field(
        default_factory=list,
        description="Real source URLs from Google Search grounding [{uri, title}]",
    )
    search_queries_used: list[str] = Field(
        default_factory=list,
        description="Search queries Gemini executed for grounding",
    )


# ── URL context scenario output ──────────────────────────────────────────
class _URLScenarioOutput(BaseModel):
    name: str
    description: str
    category: str
    difficulty: int = Field(ge=1, le=5)
    time_pressure_seconds: int
    source_url: str
    source_summary: str
    initial_data: dict
    events: list[dict]
    url_retrieval_metadata: list[dict] = Field(
        default_factory=list,
        description="URL retrieval statuses from Gemini URL context [{retrieved_url, status}]",
    )


# ── Bias classifier output ────────────────────────────────────────────

class _PerDecisionBiasScore(BaseModel):
    decision_index: int
    decision_type: str
    timestamp_seconds: int
    bias_scores: dict[str, float]
    primary_bias: str
    qualitative_evidence: str
    quantitative_evidence: str


class _TopBiasEntry(BaseModel):
    bias: str
    score: float
    rank: int
    reasoning: str
    qualitative_strength: str  # "strong", "moderate", "weak"
    quantitative_strength: str


class _FeatureImportanceEntry(BaseModel):
    feature: str
    importance: float
    explanation: str


class _BiasClassifierGeminiOutput(BaseModel):
    per_decision_scores: list[_PerDecisionBiasScore]
    aggregate_scores: dict[str, float]
    top_biases: list[_TopBiasEntry]
    feature_importance: dict[str, list[_FeatureImportanceEntry]]
    gemini_reasoning: str


# ── Confidence calibration output ──────────────────────────────────────

class _SupportingSignal(BaseModel):
    signal: str
    persuasiveness: str  # "high", "medium", "low"
    explanation: str


class _CalibratedPattern(BaseModel):
    pattern_name: str
    original_confidence: float
    evidence_strength: float
    calibrated_confidence: float
    confidence_label: str  # "High", "Medium", "Low", "Insufficient"
    supporting_signals: list[_SupportingSignal]
    contradicting_signals: list[_SupportingSignal]
    calibration_reasoning: str
    should_retract: bool = False


class _AbstainedPattern(BaseModel):
    pattern_name: str
    original_confidence: float
    reason: str
    missing_evidence: list[str]


class _ConfidenceCalibrationGeminiOutput(BaseModel):
    calibrated_patterns: list[_CalibratedPattern]
    abstained_patterns: list[_AbstainedPattern]
    overall_evidence_quality: str  # "strong", "moderate", "weak"
    calibration_summary: str


# ── Behavior history output ────────────────────────────────────────────

class _EmergingPattern(BaseModel):
    pattern_name: str
    persistence: str  # "persistent", "situational"
    frequency: float
    trend: str  # "increasing", "stable", "decreasing"
    evidence_across_sims: list[str]
    situational_triggers: list[str] = Field(default_factory=list)


class _PQTrendEntry(BaseModel):
    sim_number: int
    score: int
    note: str = ""


class _MomentEntry(BaseModel):
    sim_number: int
    description: str


class _LearningTrajectory(BaseModel):
    overall_direction: str  # "improving", "stagnating", "regressing"
    process_quality_trend: list[_PQTrendEntry]
    breakthrough_moments: list[_MomentEntry]
    regression_points: list[_MomentEntry]
    trajectory_summary: str


class _ScenarioPerformance(BaseModel):
    best_category: str
    worst_category: str
    difficulty_impact: str


class _ImprovementRec(BaseModel):
    priority: int
    area: str
    recommendation: str
    expected_impact: str
    based_on: str


class _BehaviorHistoryGeminiOutput(BaseModel):
    emerging_patterns: list[_EmergingPattern]
    learning_trajectory: _LearningTrajectory
    strengths: list[str]
    weaknesses: list[str]
    scenario_performance: _ScenarioPerformance
    decision_style: str  # "reactive", "analytical", "balanced"
    decision_style_evidence: str
    stress_response: str  # "impulsive", "cautious", "steady"
    stress_response_evidence: str
    improvement_recommendations: list[_ImprovementRec]
    history_analysis_reasoning: str


# ── Chart analysis (multimodal vision) output ────────────────────────

class _ChartBiasWarning(BaseModel):
    bias: str
    explanation: str
    risk_level: str  # "low", "medium", "high"


class _ChartAnalysisGeminiOutput(BaseModel):
    chart_type: str  # "candlestick", "line", "bar", "unknown"
    trend_summary: str
    key_patterns: list[str]
    support_resistance: list[str]
    bias_warnings: list[_ChartBiasWarning]
    recommended_action: str  # "buy", "sell", "hold", "wait"
    confidence: float = Field(ge=0, le=1)
    reasoning: str
