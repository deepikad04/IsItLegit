"""
Integration tests for the full simulation flow:
- End-to-end simulation lifecycle (create → decisions → complete → quality score)
- Market realism features (fees, spreads, halts, margin, volatility)
- Gemini mock fallbacks produce valid, parseable responses
"""
import uuid
import pytest
from unittest.mock import MagicMock

from services.simulation_engine import SimulationEngine
from services.gemini_service import GeminiService
from schemas.decision import DecisionCreate
from schemas.reflection import (
    ReflectionResponse,
    WhyThisDecisionResponse,
    ProComparisonResponse,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_scenario(market_params=None, events=None, time_limit=120):
    s = MagicMock()
    s.id = uuid.uuid4()
    s.name = "Integration Test Scenario"
    s.category = "fomo_trap"
    s.difficulty = 3
    s.time_pressure_seconds = time_limit
    s.initial_data = {
        "asset": "INTCOIN",
        "price": 50.0,
        "your_balance": 5000.0,
        "market_sentiment": "neutral",
    }
    if market_params:
        s.initial_data["market_params"] = market_params
    s.events = events or [
        {"time": 15, "type": "news", "content": "Analyst upgrade"},
        {"time": 30, "type": "price", "change": 0.08},
        {"time": 50, "type": "social", "content": "Hype building!"},
        {"time": 70, "type": "price", "change": -0.12},
        {"time": 90, "type": "news", "content": "CEO resigns"},
    ]
    return s


def _make_simulation():
    s = MagicMock()
    s.id = uuid.uuid4()
    s.user_id = uuid.uuid4()
    s.status = "completed"
    s.process_quality_score = 60.0
    s.final_outcome = {
        "profit_loss": 150.0,
        "profit_loss_percent": 3.0,
        "final_value": 5150.0,
    }
    s.current_time_elapsed = 55
    return s


def _make_decisions_data(count=4):
    decisions = []
    for i in range(count):
        d = MagicMock()
        d.id = uuid.uuid4()
        d.simulation_id = uuid.uuid4()
        d.decision_type = ["buy", "hold", "sell", "hold"][i % 4]
        d.amount = 100.0 if d.decision_type in ("buy", "sell") else None
        d.confidence_level = [4, 3, 2, 5][i % 4]
        d.simulation_time = 15 + i * 25
        d.time_spent_seconds = 5.0 + i
        d.rationale = f"Testing decision {i}"
        d.info_viewed = [{"panel": "news"}]
        d.order_type = "market"
        d.limit_price = None
        d.stop_price = None
        d.created_at = None
        decisions.append(d)
    return decisions


# ── Full Simulation Lifecycle ──────────────────────────────────────


class TestSimulationLifecycle:
    """Test the complete simulation flow from start to finish."""

    def test_full_buy_hold_sell_flow(self):
        """Simulate a full buy → hold → sell sequence and verify portfolio."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        initial = engine.get_initial_state()

        portfolio = initial["portfolio"]
        assert portfolio["cash"] == 5000.0
        assert portfolio["total_value"] == 5000.0

        # Buy at time 0
        state_t0 = engine.get_state_at_time(0, portfolio)
        buy_decision = DecisionCreate(
            decision_type="buy", amount=10.0, confidence_level=4,
        )
        portfolio = engine.process_decision(buy_decision, state_t0, portfolio)
        assert portfolio["holdings"]["INTCOIN"] == 10.0
        assert portfolio["cash"] < 5000.0

        # Hold at time 30
        state_t30 = engine.get_state_at_time(30, portfolio)
        hold_decision = DecisionCreate(
            decision_type="hold", confidence_level=3,
        )
        portfolio_after_hold = engine.process_decision(hold_decision, state_t30, portfolio)
        assert portfolio_after_hold["cash"] == portfolio["cash"]

        # Sell at time 60
        state_t60 = engine.get_state_at_time(60, portfolio_after_hold)
        sell_decision = DecisionCreate(
            decision_type="sell", amount=5.0, confidence_level=3,
        )
        portfolio = engine.process_decision(sell_decision, state_t60, portfolio_after_hold)
        assert portfolio["holdings"]["INTCOIN"] == 5.0
        assert portfolio["cash"] > portfolio_after_hold["cash"]

        # Final state
        final = engine.get_final_state(portfolio)
        assert "final_price" in final
        assert final["total_value"] > 0
        assert final["cash"] >= 0

    def test_process_quality_with_varied_decisions(self):
        """Process quality should reward good patterns."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        decisions = _make_decisions_data(4)

        score = engine.calculate_process_quality(decisions, scenario)
        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_process_quality_single_decision(self):
        """Single decision should still produce valid score."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        decisions = _make_decisions_data(1)

        score = engine.calculate_process_quality(decisions, scenario)
        assert 0 <= score <= 100

    def test_multiple_buys_accumulate_holdings(self):
        """Multiple buy decisions should accumulate holdings."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]

        for i in range(3):
            state = engine.get_state_at_time(i * 10, portfolio)
            buy = DecisionCreate(decision_type="buy", amount=5.0, confidence_level=3)
            portfolio = engine.process_decision(buy, state, portfolio)

        assert portfolio["holdings"]["INTCOIN"] == 15.0

    def test_sell_all_holdings(self):
        """Selling all holdings should leave zero."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]

        # Buy 10
        state = engine.get_state_at_time(0, portfolio)
        portfolio = engine.process_decision(
            DecisionCreate(decision_type="buy", amount=10.0, confidence_level=3),
            state, portfolio,
        )

        # Sell 10
        state = engine.get_state_at_time(30, portfolio)
        portfolio = engine.process_decision(
            DecisionCreate(decision_type="sell", amount=10.0, confidence_level=3),
            state, portfolio,
        )

        assert portfolio["holdings"]["INTCOIN"] == 0

    def test_events_appear_at_correct_times(self):
        """Events should only appear when their timestamp has passed."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]

        state_10 = engine.get_state_at_time(10, portfolio)
        news_10 = state_10["available_info"]["news"]

        state_60 = engine.get_state_at_time(60, portfolio)
        news_60 = state_60["available_info"]["news"]

        assert len(news_60) >= len(news_10)

    def test_price_timeline_is_complete(self):
        """Every second should have a price entry."""
        scenario = _make_scenario(time_limit=60)
        engine = SimulationEngine(scenario)
        for t in range(61):
            assert t in engine.price_timeline, f"Missing price at t={t}"


# ── Market Realism Features ────────────────────────────────────────


class TestMarketRealism:
    """Test market realism features in the simulation engine."""

    def test_fees_reduce_cash_on_trade(self):
        """Transaction fees should reduce available cash beyond the trade amount."""
        scenario = _make_scenario(market_params={
            "fixed_fee": 5.0,
            "pct_fee": 0.01,
        })
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]

        state = engine.get_state_at_time(0, portfolio)
        buy = DecisionCreate(decision_type="buy", amount=10.0, confidence_level=3)
        new_portfolio = engine.process_decision(buy, state, portfolio)

        # With fees: should be less than 4500 (5000 - 10*50)
        assert new_portfolio["cash"] < 4500.0

    def test_spread_affects_price(self):
        """Bid-ask spread should result in different buy/sell prices."""
        scenario = _make_scenario(market_params={
            "base_spread_pct": 0.02,
        })
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]

        state = engine.get_state_at_time(0, portfolio)
        mc = state.get("market_conditions", {})

        if mc.get("bid") is not None and mc.get("ask") is not None:
            assert mc["ask"] > mc["bid"]
            assert mc["spread_pct"] > 0

    def test_halt_detection_in_market_conditions(self):
        """Large price moves should not crash the engine."""
        scenario = _make_scenario(
            market_params={
                "halts_enabled": True,
                "halt_threshold_pct": 0.05,
                "halt_duration_ticks": 10,
            },
            events=[
                {"time": 10, "type": "price", "change": 0.20},
            ],
        )
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]

        # Should not crash
        state = engine.get_state_at_time(15, portfolio)
        assert state["current_price"] > 0

    def test_volatility_clustering_params(self):
        """GARCH volatility params should not crash the engine."""
        scenario = _make_scenario(market_params={
            "volatility_clustering": True,
            "garch_base_vol": 0.02,
            "garch_persistence": 0.90,
            "vol_shock_probability": 0.05,
            "vol_shock_multiplier": 3.0,
        })
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]

        for t in range(0, 60, 10):
            state = engine.get_state_at_time(t, portfolio)
            assert state["current_price"] > 0

    def test_margin_params_accepted(self):
        """Margin trading params should be accepted without errors."""
        scenario = _make_scenario(market_params={
            "margin_enabled": True,
            "leverage_ratio": 2.0,
            "max_drawdown_pct": 0.20,
        })
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]
        state = engine.get_state_at_time(0, portfolio)
        assert state["current_price"] > 0

    def test_crowd_model_params_accepted(self):
        """Crowd behavior model params should not crash the engine."""
        scenario = _make_scenario(market_params={
            "crowd_model_enabled": True,
        })
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]
        state = engine.get_state_at_time(30, portfolio)
        assert state["current_price"] > 0


# ── Gemini Mock Fallbacks ──────────────────────────────────────────


class TestGeminiMockFallbacks:
    """Test that all Gemini heuristic fallbacks produce valid responses."""

    def setup_method(self):
        self.gemini = GeminiService()
        self.scenario = _make_scenario()
        self.simulation = _make_simulation()
        self.decisions = _make_decisions_data(3)

    @pytest.mark.asyncio
    async def test_reflection_fallback_valid_schema(self):
        """Heuristic reflection should produce a valid ReflectionResponse."""
        result = await self.gemini.analyze_simulation(
            self.simulation, self.decisions, self.scenario
        )
        assert result is not None
        if isinstance(result, dict):
            parsed = ReflectionResponse(**result)
            assert parsed.outcome_type in ("profit", "loss")
            assert parsed.process_quality is not None

    @pytest.mark.asyncio
    async def test_why_decisions_fallback_valid(self):
        """Heuristic why-decisions should produce valid response."""
        result = await self.gemini.explain_decisions(
            self.simulation, self.decisions, self.scenario
        )
        assert result is not None
        if isinstance(result, dict):
            parsed = WhyThisDecisionResponse(**result)
            assert len(parsed.explanations) > 0
            assert parsed.overall_narrative

    @pytest.mark.asyncio
    async def test_pro_comparison_fallback_valid(self):
        """Heuristic pro-comparison should produce valid response."""
        result = await self.gemini.compare_with_pro(
            self.simulation, self.decisions, self.scenario
        )
        assert result is not None
        if isinstance(result, dict):
            parsed = ProComparisonResponse(**result)
            assert len(parsed.pro_decisions) > 0

    @pytest.mark.asyncio
    async def test_coaching_fallback_valid(self):
        """Heuristic coaching should produce valid coaching data."""
        profile_data = {
            "bias_patterns": {"fomo": 0.6, "impulsivity": 0.4},
            "total_simulations_analyzed": 3,
        }
        result = await self.gemini.generate_coaching(
            self.simulation, self.decisions, self.scenario, profile_data
        )
        assert result is not None
        # Coaching may return a dict or a string depending on the fallback path
        if isinstance(result, dict):
            assert "coaching_message" in result
            assert "persona" in result
        else:
            # String coaching message is also valid
            assert isinstance(result, str)
            assert len(result) > 10

    @pytest.mark.asyncio
    async def test_bias_heatmap_fallback_valid(self):
        """Heuristic bias heatmap should produce valid heatmap data."""
        result = await self.gemini.analyze_bias_timeline(
            self.simulation, self.decisions, self.scenario
        )
        assert result is not None
        assert "timeline" in result
        assert len(result["timeline"]) > 0
        assert "dominant_bias" in result

    @pytest.mark.asyncio
    async def test_rationale_review_fallback_valid(self):
        """Heuristic rationale review should produce valid review data."""
        result = await self.gemini.review_rationales(
            self.simulation, self.decisions, self.scenario
        )
        assert result is not None
        assert "reviews" in result
        assert "overall_reasoning_quality" in result

    @pytest.mark.asyncio
    async def test_adaptive_scenario_fallback_valid(self):
        """Heuristic adaptive scenario should produce valid scenario data."""
        profile_data = {
            "bias_patterns": {"fomo": 0.7, "loss_aversion": 0.5},
            "weaknesses": ["fomo_susceptibility"],
            "strengths": ["patience"],
            "total_simulations_analyzed": 5,
        }
        result = await self.gemini.generate_adaptive_scenario(profile_data)
        assert result is not None
        assert "name" in result
        assert "initial_data" in result
        assert "events" in result
        assert len(result["events"]) > 0

    @pytest.mark.asyncio
    async def test_counterfactuals_fallback_valid(self):
        """Heuristic counterfactuals should produce valid list."""
        result = await self.gemini.generate_counterfactuals(
            self.simulation, self.decisions, self.scenario
        )
        assert result is not None
        assert isinstance(result, list)


# ── Edge Cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_balance_cannot_buy(self):
        """Cannot buy with zero cash."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        portfolio = {"cash": 0.0, "holdings": {}, "total_value": 0.0}
        state = engine.get_state_at_time(0, portfolio)
        buy = DecisionCreate(decision_type="buy", amount=1.0, confidence_level=3)
        result = engine.process_decision(buy, state, portfolio)
        assert result["cash"] == 0.0

    def test_wait_decision_no_change(self):
        """Wait decision should not modify portfolio."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        portfolio = {"cash": 5000.0, "holdings": {"INTCOIN": 10.0}, "total_value": 5500.0}
        state = engine.get_state_at_time(0, portfolio)
        wait = DecisionCreate(decision_type="wait", confidence_level=3)
        result = engine.process_decision(wait, state, portfolio)
        assert result["cash"] == 5000.0
        assert result["holdings"]["INTCOIN"] == 10.0

    def test_negative_price_change_event(self):
        """Large negative price events should not make price negative."""
        scenario = _make_scenario(events=[
            {"time": 5, "type": "price", "change": -0.50},
        ])
        engine = SimulationEngine(scenario)
        for t in range(61):
            assert engine.price_timeline[t] > 0, f"Price went non-positive at t={t}"

    def test_many_rapid_decisions(self):
        """Engine should handle many rapid decisions without crashing."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]

        for i in range(20):
            state = engine.get_state_at_time(i * 3, portfolio)
            decision_type = "buy" if i % 2 == 0 else "sell"
            d = DecisionCreate(decision_type=decision_type, amount=1.0, confidence_level=3)
            portfolio = engine.process_decision(d, state, portfolio)

        assert portfolio["cash"] >= 0

    def test_state_at_time_zero(self):
        """State at time 0 should return valid initial conditions."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]
        state = engine.get_state_at_time(0, portfolio)

        assert state["current_price"] > 0
        assert "available_info" in state
        assert "news" in state["available_info"]
        assert "social" in state["available_info"]

    def test_state_at_max_time(self):
        """State at the final second should return valid data."""
        scenario = _make_scenario(time_limit=60)
        engine = SimulationEngine(scenario)
        portfolio = engine.get_initial_state()["portfolio"]
        state = engine.get_state_at_time(60, portfolio)

        assert state["current_price"] > 0

    def test_final_state_has_required_fields(self):
        """Final state should contain all required financial data."""
        scenario = _make_scenario()
        engine = SimulationEngine(scenario)
        portfolio = {"cash": 3000.0, "holdings": {"INTCOIN": 20.0}, "total_value": 4000.0}
        final = engine.get_final_state(portfolio)

        assert "final_price" in final
        assert "cash" in final
        assert "total_value" in final
        assert final["total_value"] >= 0
        assert final["final_price"] > 0
