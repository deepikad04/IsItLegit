"""
Snapshot tests for Gemini output schemas.
Verifies that heuristic fallbacks produce valid, schema-compliant JSON.
"""
import uuid
import pytest
from unittest.mock import MagicMock
from datetime import datetime

from pydantic import ValidationError

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


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_scenario():
    s = MagicMock()
    s.id = uuid.uuid4()
    s.name = "Crypto Hype Trap"
    s.category = "fomo_trap"
    s.difficulty = 2
    s.time_pressure_seconds = 180
    s.initial_data = {
        "asset": "HYPECOIN",
        "price": 0.50,
        "your_balance": 10000,
        "market_sentiment": "bullish",
    }
    s.events = [
        {"time": 15, "type": "social", "content": "Celebrity tweeted about HYPECOIN!"},
        {"time": 30, "type": "price", "change": 0.25},
        {"time": 60, "type": "news", "content": "BREAKING: Exchange listing rumored"},
        {"time": 120, "type": "price", "change": -0.40},
    ]
    return s


def _make_simulation(scenario_id):
    s = MagicMock()
    s.id = uuid.uuid4()
    s.user_id = uuid.uuid4()
    s.scenario_id = scenario_id
    s.status = "completed"
    s.process_quality_score = 55.0
    s.final_outcome = {
        "profit_loss": 250.0,
        "final_value": 10250.0,
        "profit_loss_percent": 2.5,
    }
    s.current_time_elapsed = 160
    return s


def _make_decisions(simulation_id, count=4):
    decisions = []
    for i in range(count):
        d = MagicMock()
        d.id = uuid.uuid4()
        d.simulation_id = simulation_id
        d.simulation_time = 10 + i * 30
        d.decision_type = ["buy", "hold", "buy", "sell"][i % 4]
        d.amount = [500, None, 300, 200][i % 4]
        d.confidence_level = [4, 3, 5, 2][i % 4]
        d.time_spent_seconds = [3.0, 8.0, 2.5, 12.0][i % 4]
        d.price_at_decision = 0.50 + i * 0.1
        d.info_viewed = [{"panel": "news", "view_duration_seconds": 2, "timestamp": d.simulation_time}]
        d.info_ignored = []
        d.market_state_at_decision = {
            "current_price": d.price_at_decision,
            "available_info": {
                "market_sentiment": ["bullish", "bullish", "bearish", "bearish"][i % 4]
            }
        }
        d.events_since_last = []
        decisions.append(d)
    return decisions


# ── Schema validation tests ──────────────────────────────────────────────

class TestReflectionSchema:
    """Test that ReflectionResponse schema validates correctly."""

    def test_valid_reflection_response(self):
        data = {
            "simulation_id": str(uuid.uuid4()),
            "outcome_summary": "+$250.00",
            "outcome_type": "profit",
            "process_quality": {
                "score": 65.0,
                "factors": {"timing": 0.7, "information_usage": 0.6, "risk_sizing": 0.8, "emotional_control": 0.5},
                "summary": "Decent process overall."
            },
            "patterns_detected": [
                {
                    "pattern_name": "fomo",
                    "confidence": 0.8,
                    "evidence": ["Bought during bullish run", "Quick decisions"],
                    "description": "Fear of missing out"
                }
            ],
            "luck_factor": 0.6,
            "skill_factor": 0.4,
            "luck_skill_explanation": "Mixed result.",
            "counterfactuals": [],
            "insights": [
                {"title": "Slow down", "description": "Take more time.", "related_pattern": "fomo", "recommended_card_id": "card_fomo_01"}
            ],
            "key_takeaway": "Process over outcome.",
            "coaching_message": "Good effort, keep practicing."
        }
        result = ReflectionResponse.model_validate(data)
        assert result.outcome_type == "profit"
        assert result.process_quality.score == 65.0
        assert len(result.patterns_detected) == 1
        assert result.coaching_message is not None

    def test_luck_factor_bounds(self):
        """luck_factor must be 0-1."""
        with pytest.raises(ValidationError):
            ReflectionResponse.model_validate({
                "simulation_id": str(uuid.uuid4()),
                "outcome_summary": "+$100",
                "outcome_type": "profit",
                "process_quality": {"score": 50, "factors": {}, "summary": "ok"},
                "patterns_detected": [],
                "luck_factor": 1.5,  # Invalid
                "skill_factor": 0.5,
                "luck_skill_explanation": "test",
                "counterfactuals": [],
                "insights": [],
                "key_takeaway": "test"
            })

    def test_process_quality_score_bounds(self):
        """process_quality.score must be 0-100."""
        with pytest.raises(ValidationError):
            ProcessQuality.model_validate({
                "score": 150,  # Invalid
                "factors": {},
                "summary": "test"
            })


class TestWhyDecisionSchema:
    """Test WhyThisDecisionResponse schema."""

    def test_valid_why_response(self):
        data = {
            "simulation_id": str(uuid.uuid4()),
            "explanations": [
                {
                    "decision_index": 0,
                    "decision_type": "buy",
                    "timestamp_seconds": 10,
                    "detected_bias": "fomo",
                    "explanation": "Bought during hype.",
                    "evidence_from_actions": ["Bullish sentiment", "Quick decision"],
                    "severity": "moderate"
                }
            ],
            "overall_narrative": "The user chased rising prices."
        }
        result = WhyThisDecisionResponse.model_validate(data)
        assert len(result.explanations) == 1
        assert result.explanations[0].detected_bias == "fomo"

    def test_empty_explanations_valid(self):
        """No biases detected should still be valid."""
        data = {
            "simulation_id": str(uuid.uuid4()),
            "explanations": [],
            "overall_narrative": "No strong patterns detected."
        }
        result = WhyThisDecisionResponse.model_validate(data)
        assert len(result.explanations) == 0


class TestProComparisonSchema:
    """Test ProComparisonResponse schema."""

    def test_valid_pro_response(self):
        data = {
            "simulation_id": str(uuid.uuid4()),
            "pro_decisions": [
                {
                    "at_timestamp": 10,
                    "user_action": "buy at t=10s",
                    "pro_action": "Wait and observe",
                    "pro_reasoning": "Gather information first.",
                    "outcome_difference": "Avoids premature entry.",
                    "skill_demonstrated": "Patience"
                }
            ],
            "pro_final_outcome": {"profit_loss": 150.0, "final_value": 10150.0},
            "user_final_outcome": {"profit_loss": 250.0, "final_value": 10250.0},
            "key_differences": ["Pro waits", "Pro sizes smaller"],
            "what_to_practice": ["Wait 20s before first trade"]
        }
        result = ProComparisonResponse.model_validate(data)
        assert len(result.pro_decisions) == 1
        assert result.pro_final_outcome["profit_loss"] == 150.0


class TestCounterfactualSchema:
    """Test Counterfactual schema."""

    def test_valid_counterfactual(self):
        data = {
            "timeline_name": "Market Crash",
            "description": "What if the market crashed?",
            "market_changes": "40% decline",
            "outcome": {"profit_loss": -3000.0, "final_value": 7000.0},
            "lesson": "Position sizing matters."
        }
        result = Counterfactual.model_validate(data)
        assert result.timeline_name == "Market Crash"
        assert result.outcome["profit_loss"] == -3000.0


# ── Heuristic output tests ──────────────────────────────────────────────

class TestHeuristicOutputs:
    """Test that heuristic fallbacks produce schema-valid output."""

    def setup_method(self):
        """Set up test fixtures."""
        # Force mock mode
        import config
        original = config.get_settings
        settings = original()
        settings.use_mock_gemini = True
        self.scenario = _make_scenario()
        self.simulation = _make_simulation(self.scenario.id)
        self.decisions = _make_decisions(self.simulation.id)

    def test_heuristic_analyze_produces_valid_schema(self):
        from services.gemini_service import GeminiService
        svc = GeminiService()
        svc.use_mock = True
        result = svc._heuristic_analyze(self.simulation, self.decisions, self.scenario)
        # Validate by serializing and deserializing
        data = result.model_dump()
        validated = ReflectionResponse.model_validate(data)
        assert 0 <= validated.process_quality.score <= 100
        assert 0 <= validated.luck_factor <= 1
        assert len(validated.patterns_detected) >= 1
        assert validated.coaching_message is not None

    def test_heuristic_counterfactuals_valid(self):
        from services.gemini_service import GeminiService
        svc = GeminiService()
        svc.use_mock = True
        result = svc._heuristic_counterfactuals(self.simulation, self.decisions, self.scenario)
        assert len(result) == 3
        for cf in result:
            data = cf.model_dump()
            validated = Counterfactual.model_validate(data)
            assert validated.timeline_name
            assert "profit_loss" in validated.outcome

    def test_heuristic_explain_decisions_valid(self):
        from services.gemini_service import GeminiService
        svc = GeminiService()
        svc.use_mock = True
        result = svc._heuristic_explain_decisions(self.simulation, self.decisions, self.scenario)
        data = result.model_dump()
        validated = WhyThisDecisionResponse.model_validate(data)
        assert validated.overall_narrative
        for exp in validated.explanations:
            assert exp.severity in ("minor", "moderate", "significant")
            assert len(exp.evidence_from_actions) > 0

    def test_heuristic_pro_comparison_valid(self):
        from services.gemini_service import GeminiService
        svc = GeminiService()
        svc.use_mock = True
        result = svc._heuristic_pro_comparison(self.simulation, self.decisions, self.scenario)
        data = result.model_dump()
        validated = ProComparisonResponse.model_validate(data)
        assert len(validated.pro_decisions) >= 1
        assert len(validated.what_to_practice) >= 1
        assert "profit_loss" in validated.pro_final_outcome

    def test_heuristic_coaching_valid(self):
        from services.gemini_service import GeminiService
        svc = GeminiService()
        svc.use_mock = True
        result = svc._heuristic_coaching(self.simulation, self.decisions, self.scenario, None)
        assert isinstance(result, str)
        assert len(result) > 20  # Not empty

    def test_heuristic_coaching_with_profile(self):
        from services.gemini_service import GeminiService
        svc = GeminiService()
        svc.use_mock = True
        profile = {
            "strengths": ["patience"],
            "weaknesses": ["fomo_susceptibility"],
            "bias_patterns": {"fomo": 0.7, "impulsivity": 0.3},
            "decision_style": "reactive",
            "total_simulations_analyzed": 5,
        }
        result = svc._heuristic_coaching(self.simulation, self.decisions, self.scenario, profile)
        assert isinstance(result, str)
        assert len(result) > 20

    def test_heuristic_profile_update_valid(self):
        from services.gemini_service import GeminiService
        svc = GeminiService()
        svc.use_mock = True
        result = svc._heuristic_profile_update(None, self.decisions)
        assert "bias_patterns" in result
        assert "strengths" in result
        assert isinstance(result["bias_patterns"], dict)
