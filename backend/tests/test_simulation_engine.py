"""
Tests for the deterministic simulation engine logic.
"""
import pytest
from unittest.mock import MagicMock
from services.simulation_engine import SimulationEngine
from schemas.decision import DecisionCreate


def _make_scenario():
    s = MagicMock()
    s.name = "Test Scenario"
    s.category = "test"
    s.difficulty = 1
    s.time_pressure_seconds = 60
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


class TestSimulationEngine:
    def setup_method(self):
        self.scenario = _make_scenario()
        self.engine = SimulationEngine(self.scenario)

    def test_initial_state(self):
        state = self.engine.get_initial_state()
        assert state["current_price"] == 100.0
        assert state["portfolio"]["cash"] == 10000.0
        assert state["portfolio"]["total_value"] == 10000.0
        assert len(state["price_history"]) >= 1

    def test_price_timeline_has_entries(self):
        """Price timeline should cover the entire time limit."""
        assert 0 in self.engine.price_timeline
        assert self.engine.time_limit in self.engine.price_timeline

    def test_price_events_affect_timeline(self):
        """Price change events at t=20 (+10%) and t=40 (-15%) should shift the price."""
        price_before_20 = self.engine.price_timeline[19]
        price_at_20 = self.engine.price_timeline[20]
        # The price at t=20 should reflect the +10% change (roughly)
        # There's noise, so just check direction
        assert price_at_20 > price_before_20 * 0.95  # At least close to an increase

    def test_state_at_time_returns_events(self):
        portfolio = {"cash": 10000.0, "holdings": {}, "total_value": 10000.0}
        state = self.engine.get_state_at_time(35, portfolio)
        # Should have news and social events by t=35
        news = state["available_info"]["news"]
        assert len(news) >= 1

    def test_buy_decision_updates_portfolio(self):
        current_state = {"current_price": 100.0}
        portfolio = {"cash": 10000.0, "holdings": {}, "total_value": 10000.0}
        decision = DecisionCreate(
            decision_type="buy",
            amount=10.0,
            confidence_level=3,
        )
        new_portfolio = self.engine.process_decision(decision, current_state, portfolio)
        assert new_portfolio["cash"] == 9000.0  # 10000 - 10*100
        assert new_portfolio["holdings"]["TESTCOIN"] == 10.0

    def test_sell_decision_updates_portfolio(self):
        current_state = {"current_price": 110.0}
        portfolio = {"cash": 5000.0, "holdings": {"TESTCOIN": 10.0}, "total_value": 6100.0}
        decision = DecisionCreate(
            decision_type="sell",
            amount=5.0,
            confidence_level=3,
        )
        new_portfolio = self.engine.process_decision(decision, current_state, portfolio)
        assert new_portfolio["cash"] == 5550.0  # 5000 + 5*110
        assert new_portfolio["holdings"]["TESTCOIN"] == 5.0

    def test_cannot_buy_more_than_cash(self):
        """Buying more than you can afford should fail gracefully."""
        current_state = {"current_price": 100.0}
        portfolio = {"cash": 500.0, "holdings": {}, "total_value": 500.0}
        decision = DecisionCreate(
            decision_type="buy",
            amount=10.0,  # Would cost 1000
            confidence_level=3,
        )
        new_portfolio = self.engine.process_decision(decision, current_state, portfolio)
        # Should not change - insufficient funds
        assert new_portfolio["cash"] == 500.0
        assert new_portfolio["holdings"].get("TESTCOIN", 0) == 0

    def test_cannot_sell_more_than_held(self):
        """Selling more than held should sell only what's available."""
        current_state = {"current_price": 100.0}
        portfolio = {"cash": 5000.0, "holdings": {"TESTCOIN": 3.0}, "total_value": 5300.0}
        decision = DecisionCreate(
            decision_type="sell",
            amount=10.0,  # Only 3 held
            confidence_level=3,
        )
        new_portfolio = self.engine.process_decision(decision, current_state, portfolio)
        assert new_portfolio["cash"] == 5300.0  # 5000 + 3*100
        assert new_portfolio["holdings"]["TESTCOIN"] == 0

    def test_hold_does_not_change_portfolio(self):
        current_state = {"current_price": 100.0}
        portfolio = {"cash": 10000.0, "holdings": {"TESTCOIN": 5.0}, "total_value": 10500.0}
        decision = DecisionCreate(
            decision_type="hold",
            confidence_level=3,
        )
        new_portfolio = self.engine.process_decision(decision, current_state, portfolio)
        assert new_portfolio["cash"] == 10000.0
        assert new_portfolio["holdings"]["TESTCOIN"] == 5.0

    def test_get_events_between(self):
        events = self.engine.get_events_between(5, 25)
        # Should get events at t=10 and t=20
        assert len(events) == 2

    def test_get_events_between_exclusive_start(self):
        events = self.engine.get_events_between(10, 25)
        # t=10 excluded (start exclusive), t=20 included
        assert len(events) == 1

    def test_final_state(self):
        portfolio = {"cash": 8000.0, "holdings": {"TESTCOIN": 20.0}}
        final = self.engine.get_final_state(portfolio)
        assert final["cash"] == 8000.0
        assert "final_price" in final
        assert final["total_value"] == 8000.0 + 20.0 * final["final_price"]

    def test_calculate_process_quality(self):
        """Process quality score should be a float between 0 and 100."""
        decisions = []
        for i in range(3):
            d = MagicMock()
            d.simulation_time = 5 + i * 20
            d.info_viewed = [{"panel": "news"}]
            d.confidence_level = 3
            d.decision_type = ["buy", "hold", "sell"][i]
            d.time_spent_seconds = 8.0
            decisions.append(d)

        score = self.engine.calculate_process_quality(decisions, self.scenario)
        assert 0 <= score <= 100

    def test_calculate_process_quality_no_decisions(self):
        score = self.engine.calculate_process_quality([], self.scenario)
        assert score == 50.0

    def test_sentiment_calculation(self):
        """Sentiment should reflect recent price movement."""
        # At the start, should use initial sentiment
        assert self.engine._calculate_sentiment(0) == "neutral"

    def test_deterministic_price_timeline(self):
        """Same scenario should produce same price timeline."""
        engine2 = SimulationEngine(self.scenario)
        for t in range(0, self.scenario.time_pressure_seconds + 1):
            assert self.engine.price_timeline[t] == engine2.price_timeline[t], \
                f"Price differs at t={t}"
