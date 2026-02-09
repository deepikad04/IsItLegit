"""
Tests for hackathon polish features:
- Simulation Engine: calibration, Monte Carlo, timeline cache
- Gemini Service: live nudge, challenge reasoning, adaptive scenario
- Schema validation for new output types
"""
import uuid
import pytest
from unittest.mock import MagicMock

from services.simulation_engine import SimulationEngine, _timeline_cache
from services.gemini_service import GeminiService, _get_market_state


# ── Shared Fixtures ──────────────────────────────────────────────────────


def _make_scenario(name="Test Scenario", time_limit=60):
    s = MagicMock()
    s.id = uuid.uuid4()
    s.name = name
    s.category = "fomo_trap"
    s.difficulty = 2
    s.time_pressure_seconds = time_limit
    s.initial_data = {
        "asset": "TESTCOIN",
        "price": 100.0,
        "your_balance": 10000.0,
        "market_sentiment": "neutral",
    }
    s.events = [
        {"time": 10, "type": "news", "content": "Breaking news!"},
        {"time": 20, "type": "price", "change": 0.10},
        {"time": 30, "type": "social", "content": "Everyone's buying!"},
        {"time": 40, "type": "price", "change": -0.15},
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
    s.current_time_elapsed = 55
    return s


def _make_decisions(simulation_id, count=4):
    decisions = []
    for i in range(count):
        d = MagicMock()
        d.id = uuid.uuid4()
        d.simulation_id = simulation_id
        d.simulation_time = 10 + i * 15
        d.decision_type = ["buy", "hold", "buy", "sell"][i % 4]
        d.amount = [500, None, 300, 200][i % 4]
        d.confidence_level = [4, 3, 5, 2][i % 4]
        d.time_spent_seconds = [3.0, 8.0, 2.5, 12.0][i % 4]
        d.price_at_decision = 100.0 + i * 10
        d.rationale = ["Looks good", None, "Everyone is buying!", "Cutting losses"][i % 4]
        d.info_viewed = [{"panel": "news", "view_duration_seconds": 2, "timestamp": d.simulation_time}]
        d.info_ignored = []
        d.market_state_at_decision = {
            "current_price": d.price_at_decision,
            "available_info": {
                "market_sentiment": ["bullish", "bullish", "bearish", "bearish"][i % 4]
            },
        }
        d.snapshot = None  # No snapshot reference in unit tests
        d.events_since_last = []
        decisions.append(d)
    return decisions


# ── Simulation Engine: Calibration ────────────────────────────────────


class TestCalibration:
    def setup_method(self):
        # Clear cache between tests
        _timeline_cache.clear()
        self.scenario = _make_scenario()
        self.engine = SimulationEngine(self.scenario)

    def test_evaluate_decision_outcome_buy_favorable(self):
        """High confidence buy followed by price increase = well_calibrated."""
        d = MagicMock()
        d.decision_type = "buy"
        d.price_at_decision = 100.0
        d.confidence_level = 4
        result = self.engine.evaluate_decision_outcome(d, 112.0)
        assert result["calibration"] == "well_calibrated"
        assert result["favorable"] is True
        assert result["price_change_pct"] == pytest.approx(12.0, rel=0.01)

    def test_evaluate_decision_outcome_buy_overconfident(self):
        """High confidence buy followed by price drop = overconfident."""
        d = MagicMock()
        d.decision_type = "buy"
        d.price_at_decision = 100.0
        d.confidence_level = 5
        result = self.engine.evaluate_decision_outcome(d, 90.0)
        assert result["calibration"] == "overconfident"
        assert result["favorable"] is False

    def test_evaluate_decision_outcome_sell_favorable(self):
        """Sell followed by price drop = favorable."""
        d = MagicMock()
        d.decision_type = "sell"
        d.price_at_decision = 100.0
        d.confidence_level = 4
        result = self.engine.evaluate_decision_outcome(d, 95.0)
        assert result["calibration"] == "well_calibrated"
        assert result["favorable"] is True

    def test_evaluate_decision_outcome_underconfident(self):
        """Low confidence but favorable outcome = underconfident."""
        d = MagicMock()
        d.decision_type = "buy"
        d.price_at_decision = 100.0
        d.confidence_level = 1
        result = self.engine.evaluate_decision_outcome(d, 115.0)
        assert result["calibration"] == "underconfident"

    def test_evaluate_decision_outcome_hold(self):
        """Hold is favorable when price stays flat."""
        d = MagicMock()
        d.decision_type = "hold"
        d.price_at_decision = 100.0
        d.confidence_level = 3
        result = self.engine.evaluate_decision_outcome(d, 100.5)
        assert result["favorable"] is True
        assert result["calibration"] == "well_calibrated"

    def test_get_calibration_report_structure(self):
        """Calibration report has required fields and valid values."""
        decisions = _make_decisions(uuid.uuid4())
        report = self.engine.get_calibration_report(decisions)

        assert "calibration_score" in report
        assert "overconfident_count" in report
        assert "underconfident_count" in report
        assert "well_calibrated_count" in report
        assert "total_decisions" in report
        assert "details" in report

        assert 0 <= report["calibration_score"] <= 100
        total = report["overconfident_count"] + report["underconfident_count"] + report["well_calibrated_count"]
        assert total == report["total_decisions"]
        assert len(report["details"]) == report["total_decisions"]

    def test_get_calibration_report_empty_decisions(self):
        """Empty decision list returns 0% calibration (no data)."""
        report = self.engine.get_calibration_report([])
        assert report["calibration_score"] == 0  # 0 well_calibrated / max(0,1) = 0
        assert report["total_decisions"] == 0

    def test_calibration_detail_fields(self):
        """Each detail entry has all required fields."""
        decisions = _make_decisions(uuid.uuid4(), count=2)
        report = self.engine.get_calibration_report(decisions)

        for detail in report["details"]:
            assert "decision_index" in detail
            assert "decision_type" in detail
            assert "confidence" in detail
            assert "price_at_decision" in detail
            assert "price_after" in detail
            assert "price_change_pct" in detail
            assert "favorable" in detail
            assert "calibration" in detail
            assert detail["calibration"] in ("overconfident", "underconfident", "well_calibrated")


# ── Simulation Engine: Monte Carlo ────────────────────────────────────


class TestMonteCarlo:
    def setup_method(self):
        _timeline_cache.clear()
        self.scenario = _make_scenario()
        self.engine = SimulationEngine(self.scenario)

    def test_monte_carlo_structure(self):
        """Monte Carlo output has all required fields."""
        decisions = _make_decisions(uuid.uuid4(), count=2)
        result = self.engine.monte_carlo_outcomes(decisions, n=20)

        assert "simulations_run" in result
        assert "outcomes" in result
        assert "buckets" in result
        assert "actual_outcome" in result
        assert "percentile" in result
        assert "median_outcome" in result
        assert "best_outcome" in result
        assert "worst_outcome" in result

        assert result["simulations_run"] == 20
        assert len(result["outcomes"]) == 20

    def test_monte_carlo_outcomes_sorted(self):
        """Outcomes list is sorted ascending."""
        decisions = _make_decisions(uuid.uuid4(), count=2)
        result = self.engine.monte_carlo_outcomes(decisions, n=50)
        for i in range(1, len(result["outcomes"])):
            assert result["outcomes"][i] >= result["outcomes"][i - 1]

    def test_monte_carlo_worst_best_consistent(self):
        """Worst <= median <= best."""
        decisions = _make_decisions(uuid.uuid4(), count=3)
        result = self.engine.monte_carlo_outcomes(decisions, n=50)
        assert result["worst_outcome"] <= result["median_outcome"]
        assert result["median_outcome"] <= result["best_outcome"]

    def test_monte_carlo_percentile_range(self):
        """Percentile is between 0 and 100."""
        decisions = _make_decisions(uuid.uuid4(), count=2)
        result = self.engine.monte_carlo_outcomes(decisions, n=30)
        assert 0 <= result["percentile"] <= 100

    def test_monte_carlo_buckets(self):
        """Buckets cover the full range and have valid counts."""
        decisions = _make_decisions(uuid.uuid4(), count=2)
        result = self.engine.monte_carlo_outcomes(decisions, n=50)
        assert len(result["buckets"]) == 20
        for bucket in result["buckets"]:
            assert "range_low" in bucket
            assert "range_high" in bucket
            assert "count" in bucket
            assert bucket["count"] >= 0

    def test_monte_carlo_no_decisions(self):
        """With no decisions, all outcomes should be ~0 (cash unchanged)."""
        result = self.engine.monte_carlo_outcomes([], n=10)
        assert result["simulations_run"] == 10
        # All outcomes should be 0 since no trades
        for o in result["outcomes"]:
            assert o == 0.0

    def test_monte_carlo_deterministic_same_seed(self):
        """Same scenario + decisions = same Monte Carlo results."""
        _timeline_cache.clear()
        decisions = _make_decisions(uuid.uuid4(), count=2)
        r1 = self.engine.monte_carlo_outcomes(decisions, n=20)
        r2 = self.engine.monte_carlo_outcomes(decisions, n=20)
        assert r1["outcomes"] == r2["outcomes"]


# ── Simulation Engine: Timeline Cache ─────────────────────────────────


class TestTimelineCache:
    def setup_method(self):
        _timeline_cache.clear()

    def test_cache_hit(self):
        """Second engine creation for same scenario uses cached timeline."""
        scenario = _make_scenario()
        engine1 = SimulationEngine(scenario)
        cache_key = str(scenario.id)
        assert cache_key in _timeline_cache

        engine2 = SimulationEngine(scenario)
        # Both engines should share the same timeline dict
        assert engine1.price_timeline is engine2.price_timeline

    def test_cache_miss_different_scenario(self):
        """Different scenario IDs create different cache entries."""
        s1 = _make_scenario("Scenario A")
        s2 = _make_scenario("Scenario B")
        e1 = SimulationEngine(s1)
        e2 = SimulationEngine(s2)
        assert str(s1.id) in _timeline_cache
        assert str(s2.id) in _timeline_cache
        assert e1.price_timeline is not e2.price_timeline


# ── Gemini Service: _get_market_state helper ──────────────────────────


class TestGetMarketState:
    def test_inline_state(self):
        """Returns inline market_state_at_decision when available."""
        d = MagicMock()
        d.market_state_at_decision = {"current_price": 100}
        d.snapshot = None
        assert _get_market_state(d) == {"current_price": 100}

    def test_snapshot_fallback(self):
        """Falls back to snapshot.data when inline is None."""
        d = MagicMock()
        d.market_state_at_decision = None
        d.snapshot = MagicMock()
        d.snapshot.data = {"current_price": 200}
        assert _get_market_state(d) == {"current_price": 200}

    def test_both_none(self):
        """Returns empty dict when both are None."""
        d = MagicMock()
        d.market_state_at_decision = None
        d.snapshot = None
        assert _get_market_state(d) == {}

    def test_inline_preferred_over_snapshot(self):
        """When both exist, inline is preferred."""
        d = MagicMock()
        d.market_state_at_decision = {"current_price": 100}
        d.snapshot = MagicMock()
        d.snapshot.data = {"current_price": 200}
        assert _get_market_state(d)["current_price"] == 100


# ── Gemini Service: Heuristic Live Nudge ──────────────────────────────


class TestLiveNudge:
    def setup_method(self):
        _timeline_cache.clear()
        self.svc = GeminiService()
        self.svc.use_mock = True
        self.scenario = _make_scenario()

    @pytest.mark.asyncio
    async def test_no_nudge_for_no_decisions(self):
        result = await self.svc.generate_live_nudge([], self.scenario, 30)
        assert result is None

    @pytest.mark.asyncio
    async def test_nudge_for_fomo_decisions(self):
        """FOMO-pattern decisions should produce a nudge."""
        decisions = _make_decisions(uuid.uuid4(), count=3)
        # All buys during bullish sentiment with quick timing → strong FOMO
        for d in decisions:
            d.decision_type = "buy"
            d.time_spent_seconds = 2.0
            d.confidence_level = 5
            d.market_state_at_decision = {
                "current_price": 100,
                "available_info": {"market_sentiment": "bullish"},
            }
        result = await self.svc.generate_live_nudge(decisions, self.scenario, 45)
        assert result is not None
        assert "message" in result
        assert "bias" in result
        assert len(result["message"]) > 10

    @pytest.mark.asyncio
    async def test_nudge_contains_valid_bias(self):
        """Nudge bias field should be a recognized bias name."""
        decisions = _make_decisions(uuid.uuid4(), count=2)
        for d in decisions:
            d.decision_type = "buy"
            d.time_spent_seconds = 1.5
            d.confidence_level = 5
            d.market_state_at_decision = {
                "current_price": 100,
                "available_info": {"market_sentiment": "bullish"},
            }
        result = await self.svc.generate_live_nudge(decisions, self.scenario, 30)
        if result:
            valid_biases = {"fomo", "impulsivity", "loss_aversion", "overconfidence", "anchoring", "social_proof_reliance"}
            assert result["bias"] in valid_biases

    @pytest.mark.asyncio
    async def test_no_nudge_for_calm_decisions(self):
        """Low-bias decisions should not produce a nudge."""
        decisions = _make_decisions(uuid.uuid4(), count=2)
        for d in decisions:
            d.decision_type = "hold"
            d.time_spent_seconds = 15.0
            d.confidence_level = 3
            d.market_state_at_decision = {
                "current_price": 100,
                "available_info": {"market_sentiment": "neutral"},
            }
        result = await self.svc.generate_live_nudge(decisions, self.scenario, 50)
        assert result is None


# ── Gemini Service: Heuristic Challenge Reasoning ─────────────────────


class TestChallengeReasoning:
    def setup_method(self):
        self.svc = GeminiService()
        self.svc.use_mock = True
        self.scenario = _make_scenario()
        self.state = {"current_price": 105.0, "available_info": {"market_sentiment": "neutral"}}

    @pytest.mark.asyncio
    async def test_challenge_returns_score_and_feedback(self):
        result = await self.svc.challenge_reasoning(
            "buy", 100, "Based on trend analysis and news data", self.scenario, self.state, []
        )
        assert "reasoning_score" in result
        assert "feedback" in result
        assert 1 <= result["reasoning_score"] <= 5
        assert len(result["feedback"]) > 10

    @pytest.mark.asyncio
    async def test_analytical_rationale_scores_higher(self):
        r1 = await self.svc.challenge_reasoning(
            "buy", 100, "Based on data analysis and risk assessment", self.scenario, self.state, []
        )
        r2 = await self.svc.challenge_reasoning(
            "buy", 100, "moon hype fomo", self.scenario, self.state, []
        )
        assert r1["reasoning_score"] > r2["reasoning_score"]

    @pytest.mark.asyncio
    async def test_short_rationale_low_score(self):
        result = await self.svc.challenge_reasoning(
            "buy", 100, "yolo", self.scenario, self.state, []
        )
        assert result["reasoning_score"] <= 2

    @pytest.mark.asyncio
    async def test_challenge_with_existing_decisions(self):
        decisions = _make_decisions(uuid.uuid4(), count=2)
        result = await self.svc.challenge_reasoning(
            "sell", 50, "Trend reversal based on news data", self.scenario, self.state, decisions
        )
        assert 1 <= result["reasoning_score"] <= 5


# ── Gemini Service: Heuristic Adaptive Scenario ──────────────────────


class TestAdaptiveScenario:
    def setup_method(self):
        self.svc = GeminiService()
        self.svc.use_mock = True

    @pytest.mark.asyncio
    async def test_adaptive_scenario_fomo_profile(self):
        profile = {"bias_patterns": {"fomo": 0.8, "impulsivity": 0.3}, "weaknesses": ["fomo_susceptibility"]}
        result = await self.svc.generate_adaptive_scenario(profile)

        assert "name" in result
        assert "description" in result
        assert "initial_data" in result
        assert "events" in result
        assert "target_bias" in result
        assert result["target_bias"] == "fomo"
        assert result["category"] == "fomo_trap"
        assert len(result["events"]) >= 4

    @pytest.mark.asyncio
    async def test_adaptive_scenario_impulsivity_profile(self):
        profile = {"bias_patterns": {"impulsivity": 0.9, "fomo": 0.2}}
        result = await self.svc.generate_adaptive_scenario(profile)
        assert result["target_bias"] == "impulsivity"
        assert result["category"] == "patience_test"

    @pytest.mark.asyncio
    async def test_adaptive_scenario_loss_aversion_profile(self):
        profile = {"bias_patterns": {"loss_aversion": 0.7, "fomo": 0.1}}
        result = await self.svc.generate_adaptive_scenario(profile)
        assert result["target_bias"] == "loss_aversion"

    @pytest.mark.asyncio
    async def test_adaptive_scenario_has_valid_initial_data(self):
        profile = {"bias_patterns": {"fomo": 0.5}}
        result = await self.svc.generate_adaptive_scenario(profile)
        init = result["initial_data"]
        assert "asset" in init
        assert "price" in init
        assert "your_balance" in init
        assert init["price"] > 0
        assert init["your_balance"] > 0

    @pytest.mark.asyncio
    async def test_adaptive_scenario_empty_profile(self):
        """Empty bias patterns should default to fomo template."""
        result = await self.svc.generate_adaptive_scenario({})
        assert result["target_bias"] == "fomo"

    @pytest.mark.asyncio
    async def test_adaptive_scenario_has_time_pressure(self):
        profile = {"bias_patterns": {"fomo": 0.5}}
        result = await self.svc.generate_adaptive_scenario(profile)
        assert result["time_pressure_seconds"] == 180
        assert 1 <= result["difficulty"] <= 5


# ── Gemini Service: Heuristic Nudge Messages ─────────────────────────


class TestHeuristicNudge:
    def setup_method(self):
        self.svc = GeminiService()

    def test_nudge_messages_for_all_biases(self):
        """Each known bias type should produce a distinct nudge message."""
        biases = ["fomo", "impulsivity", "loss_aversion", "overconfidence", "anchoring", "social_proof_reliance"]
        messages = set()
        for bias in biases:
            entry = {"biases": {bias: 0.8}, "evidence": "test"}
            result = self.svc._heuristic_nudge(bias, entry)
            assert "message" in result
            assert result["bias"] == bias
            messages.add(result["message"])
        # All messages should be different
        assert len(messages) == len(biases)

    def test_unknown_bias_gets_fallback(self):
        result = self.svc._heuristic_nudge("unknown_bias", {"biases": {}, "evidence": "test"})
        assert "message" in result
        assert "breath" in result["message"].lower()


# ── Schema Validation: EvidenceTimestamp ──────────────────────────────


class TestEvidenceTimestampSchema:
    def test_valid_evidence_timestamp(self):
        from schemas.reflection import EvidenceTimestamp
        et = EvidenceTimestamp(time=30, event="Price spike", relevance="Triggered FOMO")
        assert et.time == 30
        assert et.event == "Price spike"

    def test_decision_explanation_with_evidence_timestamps(self):
        from schemas.reflection import DecisionExplanation, EvidenceTimestamp
        exp = DecisionExplanation(
            decision_index=0,
            decision_type="buy",
            timestamp_seconds=15,
            detected_bias="fomo",
            explanation="Bought during hype",
            evidence_from_actions=["Bullish sentiment"],
            severity="moderate",
            evidence_timestamps=[
                EvidenceTimestamp(time=10, event="Celebrity tweet", relevance="Social pressure"),
                EvidenceTimestamp(time=14, event="Price +10%", relevance="Rising price"),
            ],
        )
        assert len(exp.evidence_timestamps) == 2
        assert exp.evidence_timestamps[0].time == 10

    def test_decision_explanation_without_evidence_timestamps(self):
        from schemas.reflection import DecisionExplanation
        exp = DecisionExplanation(
            decision_index=0,
            decision_type="buy",
            timestamp_seconds=15,
            detected_bias="fomo",
            explanation="Bought during hype",
            evidence_from_actions=["Bullish sentiment"],
            severity="moderate",
        )
        assert exp.evidence_timestamps is None


# ── Gemini Pydantic Output Schemas ────────────────────────────────────


class TestNewOutputSchemas:
    def test_live_nudge_output_valid(self):
        from services.gemini_service import _LiveNudgeOutput
        o = _LiveNudgeOutput(message="Slow down!", bias="impulsivity")
        assert o.message == "Slow down!"

    def test_challenge_output_valid(self):
        from services.gemini_service import _ChallengeOutput
        o = _ChallengeOutput(reasoning_score=4, feedback="Good reasoning")
        assert o.reasoning_score == 4

    def test_challenge_output_bounds(self):
        from services.gemini_service import _ChallengeOutput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _ChallengeOutput(reasoning_score=0, feedback="Bad")  # min is 1
        with pytest.raises(ValidationError):
            _ChallengeOutput(reasoning_score=6, feedback="Too high")  # max is 5

    def test_adaptive_scenario_output_valid(self):
        from services.gemini_service import _AdaptiveScenarioOutput
        o = _AdaptiveScenarioOutput(
            name="Test",
            description="desc",
            difficulty=3,
            category="fomo_trap",
            time_pressure_seconds=180,
            initial_data={"asset": "X", "price": 100, "your_balance": 10000},
            events=[{"time": 30, "type": "news", "content": "test"}],
            target_bias="fomo",
        )
        assert o.difficulty == 3

    def test_adaptive_scenario_output_difficulty_bounds(self):
        from services.gemini_service import _AdaptiveScenarioOutput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _AdaptiveScenarioOutput(
                name="X", description="X", difficulty=6,
                category="x", time_pressure_seconds=180,
                initial_data={}, events=[], target_bias="fomo"
            )


# ── Integration: Calibration + Monte Carlo with same decisions ────────


class TestCalibrationMonteCarlo:
    """Ensure calibration and Monte Carlo work together on the same decision set."""

    def setup_method(self):
        _timeline_cache.clear()
        self.scenario = _make_scenario()
        self.engine = SimulationEngine(self.scenario)
        self.decisions = _make_decisions(uuid.uuid4(), count=4)

    def test_both_produce_valid_output(self):
        cal = self.engine.get_calibration_report(self.decisions)
        mc = self.engine.monte_carlo_outcomes(self.decisions, n=20)

        assert cal["total_decisions"] == 4
        assert mc["simulations_run"] == 20
        assert isinstance(mc["actual_outcome"], float)
        assert isinstance(cal["calibration_score"], int)

    def test_monte_carlo_actual_matches_engine_calculation(self):
        """The actual_outcome from Monte Carlo should match replaying decisions on the real timeline."""
        mc = self.engine.monte_carlo_outcomes(self.decisions, n=10)
        # Manually replay
        portfolio = {"cash": 10000.0, "holdings": {}}
        for d in self.decisions:
            price = self.engine.price_timeline.get(d.simulation_time, 100.0)
            if d.decision_type == "buy" and d.amount:
                cost = d.amount * price
                if cost <= portfolio["cash"]:
                    portfolio["cash"] -= cost
                    portfolio["holdings"]["TESTCOIN"] = portfolio["holdings"].get("TESTCOIN", 0) + d.amount
            elif d.decision_type == "sell" and d.amount:
                held = portfolio["holdings"].get("TESTCOIN", 0)
                sell_amt = min(d.amount, held)
                if sell_amt > 0:
                    portfolio["cash"] += sell_amt * price
                    portfolio["holdings"]["TESTCOIN"] = held - sell_amt

        final_price = self.engine.price_timeline.get(self.engine.time_limit, 100.0)
        hv = portfolio["holdings"].get("TESTCOIN", 0) * final_price
        expected_pl = round(portfolio["cash"] + hv - 10000.0, 2)

        assert mc["actual_outcome"] == expected_pl
