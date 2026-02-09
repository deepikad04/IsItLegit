"""
Tests for the 3 new features:
1. Algorithmic Pro Trader (#5)
2. Bias Classifier (#2)
3. Confidence Calibrator (#3)
"""
import pytest
from unittest.mock import MagicMock
from services.bias_classifier import (
    extract_decision_features, classify_decision, classify_simulation_biases,
    get_feature_importance, BIAS_LABELS, _evaluate_rule,
)
from services.confidence_calibrator import (
    calibrate_pattern_confidence, calibrate_all_patterns, EVIDENCE_RULES,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _mock_decision(
    decision_type="buy", simulation_time=30, amount=10, confidence_level=3,
    time_spent_seconds=5, info_viewed=None, price_at_decision=100, rationale=None,
):
    d = MagicMock()
    d.decision_type = decision_type
    d.simulation_time = simulation_time
    d.amount = amount
    d.confidence_level = confidence_level
    d.time_spent_seconds = time_spent_seconds
    d.info_viewed = info_viewed or []
    d.price_at_decision = price_at_decision
    d.rationale = rationale
    return d


def _simple_timeline(initial=100, duration=120):
    """Flat price timeline for testing."""
    return {t: initial for t in range(duration + 1)}


def _rising_timeline(initial=100, duration=120, rate=0.001):
    """Steadily rising price timeline."""
    return {t: initial * (1 + rate * t) for t in range(duration + 1)}


def _spike_timeline(initial=100, duration=120, spike_at=25, spike_pct=0.08):
    """Timeline with a spike at a specific time."""
    tl = {}
    for t in range(duration + 1):
        if t >= spike_at:
            tl[t] = initial * (1 + spike_pct)
        else:
            tl[t] = initial
    return tl


# ═══════════════════════════════════════════════════════════════════════
# FEATURE #5: Algorithmic Pro Trader
# ═══════════════════════════════════════════════════════════════════════


class TestAlgorithmicTrader:
    """Test the run_algorithmic_trader() method on SimulationEngine."""

    def _make_engine(self, price_fn=None, time_limit=120):
        """Create a minimal SimulationEngine for testing."""
        from services.simulation_engine import SimulationEngine

        scenario = MagicMock()
        scenario.name = "test_algo"
        scenario.time_pressure_seconds = time_limit
        scenario.initial_data = {
            "asset": "TEST",
            "price": 100,
            "your_balance": 10000,
            "holdings": {},
            "market_params": {},
        }
        scenario.events = []
        scenario.id = "test-id-123"

        engine = SimulationEngine(scenario)
        if price_fn:
            engine.price_timeline = price_fn
        return engine

    def test_returns_expected_keys(self):
        engine = self._make_engine()
        result = engine.run_algorithmic_trader([])
        assert "algo_decisions" in result
        assert "algo_final_outcome" in result
        assert "algo_portfolio_timeline" in result
        assert "strategy_description" in result
        assert "rules" in result

    def test_no_decisions_with_flat_market(self):
        """Flat market = no momentum signal = no trades."""
        engine = self._make_engine(_simple_timeline(100, 120), 120)
        result = engine.run_algorithmic_trader([])
        assert result["algo_final_outcome"]["profit_loss"] == 0

    def test_algo_trades_on_rising_market(self):
        """Rising market should trigger momentum buy."""
        tl = _rising_timeline(100, 120, rate=0.005)
        engine = self._make_engine(tl, 120)
        result = engine.run_algorithmic_trader([])
        buys = [d for d in result["algo_decisions"] if d["action"] == "buy"]
        assert len(buys) >= 1, "Expected at least one buy in rising market"

    def test_algo_respects_20s_wait(self):
        """No trades before t=20."""
        tl = _rising_timeline(100, 120, rate=0.01)
        engine = self._make_engine(tl, 120)
        result = engine.run_algorithmic_trader([])
        early_trades = [d for d in result["algo_decisions"] if d["time"] < 20]
        assert len(early_trades) == 0, "Should not trade before t=20"

    def test_algo_outcome_has_correct_structure(self):
        engine = self._make_engine()
        result = engine.run_algorithmic_trader([])
        outcome = result["algo_final_outcome"]
        assert "profit_loss" in outcome
        assert "final_value" in outcome
        assert "cumulative_fees" in outcome
        assert "total_trades" in outcome

    def test_algo_portfolio_timeline_has_entries(self):
        engine = self._make_engine()
        result = engine.run_algorithmic_trader([])
        assert len(result["algo_portfolio_timeline"]) > 0
        assert "time" in result["algo_portfolio_timeline"][0]
        assert "value" in result["algo_portfolio_timeline"][0]

    def test_rules_list_complete(self):
        engine = self._make_engine()
        result = engine.run_algorithmic_trader([])
        rule_names = {r["rule"] for r in result["rules"]}
        assert "patience" in rule_names
        assert "stop_loss" in rule_names
        assert "take_profit" in rule_names
        assert "momentum_entry" in rule_names
        assert "position_size" in rule_names

    def test_position_size_capped_at_5pct(self):
        """Each buy should be ≤5% of portfolio value."""
        tl = _rising_timeline(100, 120, rate=0.005)
        engine = self._make_engine(tl, 120)
        result = engine.run_algorithmic_trader([])
        for d in result["algo_decisions"]:
            if d["action"] == "buy":
                trade_value = d["amount"] * d["price"]
                assert trade_value < 600, f"Trade value {trade_value} exceeds 5% cap"


# ═══════════════════════════════════════════════════════════════════════
# FEATURE #2: Bias Classifier
# ═══════════════════════════════════════════════════════════════════════


class TestBiasClassifierFeatures:
    def test_extract_features_returns_all_keys(self):
        d = _mock_decision()
        features = extract_decision_features(
            d, _simple_timeline(), 120, 100, 10000, [d], 0
        )
        expected_keys = {
            "time_spent_seconds", "time_pct_in_sim", "is_first_decision",
            "price_change_10s", "price_change_30s", "price_vs_initial",
            "future_change_30s", "info_panels_viewed", "confidence_normalized",
            "position_pct_of_capital", "is_buy", "is_sell", "is_hold",
            "local_volatility", "time_since_last_decision", "direction_consistency",
        }
        assert set(features.keys()) == expected_keys

    def test_buy_decision_flags(self):
        d = _mock_decision(decision_type="buy")
        features = extract_decision_features(d, _simple_timeline(), 120, 100, 10000, [d], 0)
        assert features["is_buy"] == 1.0
        assert features["is_sell"] == 0.0
        assert features["is_hold"] == 0.0

    def test_sell_decision_flags(self):
        d = _mock_decision(decision_type="sell")
        features = extract_decision_features(d, _simple_timeline(), 120, 100, 10000, [d], 0)
        assert features["is_buy"] == 0.0
        assert features["is_sell"] == 1.0

    def test_first_decision_flag(self):
        d1 = _mock_decision(simulation_time=10)
        d2 = _mock_decision(simulation_time=30)
        f1 = extract_decision_features(d1, _simple_timeline(), 120, 100, 10000, [d1, d2], 0)
        f2 = extract_decision_features(d2, _simple_timeline(), 120, 100, 10000, [d1, d2], 1)
        assert f1["is_first_decision"] == 1.0
        assert f2["is_first_decision"] == 0.0

    def test_price_change_calculated(self):
        tl = _spike_timeline(100, 120, spike_at=20, spike_pct=0.10)
        d = _mock_decision(simulation_time=25)
        features = extract_decision_features(d, tl, 120, 100, 10000, [d], 0)
        assert features["price_change_10s"] > 0.05

    def test_position_pct_calculated(self):
        d = _mock_decision(amount=50, price_at_decision=100)
        features = extract_decision_features(d, _simple_timeline(), 120, 100, 10000, [d], 0)
        assert features["position_pct_of_capital"] == 0.5


class TestBiasClassifierRules:
    def test_evaluate_rule_comparisons(self):
        assert _evaluate_rule(5, ">", 3) is True
        assert _evaluate_rule(2, "<", 3) is True
        assert _evaluate_rule(3, "==", 3) is True
        assert _evaluate_rule(0.05, "abs<", 0.1) is True
        assert _evaluate_rule(-0.05, "abs<", 0.1) is True
        assert _evaluate_rule(0.15, "abs>", 0.1) is True

    def test_classify_decision_returns_all_biases(self):
        features = {
            "time_spent_seconds": 2, "time_pct_in_sim": 0.5, "is_first_decision": 1.0,
            "price_change_10s": 0.06, "price_change_30s": 0.1, "price_vs_initial": 0.05,
            "future_change_30s": -0.02, "info_panels_viewed": 0, "confidence_normalized": 0.9,
            "position_pct_of_capital": 0.15, "is_buy": 1.0, "is_sell": 0.0, "is_hold": 0.0,
            "local_volatility": 0.03, "time_since_last_decision": 5, "direction_consistency": 1.0,
        }
        scores = classify_decision(features)
        assert set(scores.keys()) == set(BIAS_LABELS)
        assert scores["fomo"] > 0.3
        assert scores["impulsivity"] > 0.3

    def test_classify_calm_decision_low_scores(self):
        """A calm, well-researched decision should have low bias scores."""
        features = {
            "time_spent_seconds": 15, "time_pct_in_sim": 0.3, "is_first_decision": 0.0,
            "price_change_10s": 0.001, "price_change_30s": 0.002, "price_vs_initial": 0.01,
            "future_change_30s": 0.01, "info_panels_viewed": 4, "confidence_normalized": 0.6,
            "position_pct_of_capital": 0.03, "is_buy": 1.0, "is_sell": 0.0, "is_hold": 0.0,
            "local_volatility": 0.01, "time_since_last_decision": 40, "direction_consistency": 0.5,
        }
        scores = classify_decision(features)
        max_score = max(scores.values())
        assert max_score <= 0.5, f"Expected low bias scores, got max {max_score}"


class TestBiasClassifierPipeline:
    def test_empty_decisions(self):
        result = classify_simulation_biases([], _simple_timeline(), 120, 100, 10000)
        assert result["per_decision"] == []
        assert all(v == 0.0 for v in result["aggregate_scores"].values())

    def test_full_pipeline_returns_expected_keys(self):
        decisions = [_mock_decision(simulation_time=30), _mock_decision(decision_type="sell", simulation_time=60)]
        result = classify_simulation_biases(decisions, _simple_timeline(), 120, 100, 10000)
        assert "per_decision" in result
        assert "aggregate_scores" in result
        assert "feature_importance" in result
        assert "top_biases" in result
        assert len(result["per_decision"]) == 2

    def test_per_decision_has_required_fields(self):
        d = _mock_decision()
        result = classify_simulation_biases([d], _simple_timeline(), 120, 100, 10000)
        pd = result["per_decision"][0]
        assert "decision_index" in pd
        assert "features" in pd
        assert "bias_scores" in pd
        assert "primary_bias" in pd

    def test_gemini_comparison_when_provided(self):
        decisions = [_mock_decision(time_spent_seconds=2, info_viewed=[], confidence_level=5)]
        gemini_patterns = [
            {"pattern_name": "fomo", "confidence": 0.8},
            {"pattern_name": "impulsivity", "confidence": 0.7},
        ]
        result = classify_simulation_biases(
            decisions, _simple_timeline(), 120, 100, 10000,
            gemini_patterns=gemini_patterns,
        )
        assert result["gemini_comparison"] is not None
        assert "agreement_rate" in result["gemini_comparison"]
        assert "details" in result["gemini_comparison"]

    def test_gemini_comparison_not_present_when_none(self):
        decisions = [_mock_decision()]
        result = classify_simulation_biases(decisions, _simple_timeline(), 120, 100, 10000)
        assert result["gemini_comparison"] is None

    def test_feature_importance_has_entries(self):
        decisions = [
            _mock_decision(simulation_time=20, time_spent_seconds=2),
            _mock_decision(simulation_time=40, time_spent_seconds=10),
            _mock_decision(simulation_time=60, time_spent_seconds=3),
        ]
        result = classify_simulation_biases(decisions, _simple_timeline(), 120, 100, 10000)
        assert len(result["feature_importance"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# FEATURE #3: Confidence Calibrator
# ═══════════════════════════════════════════════════════════════════════


class TestConfidenceCalibrator:
    def test_unknown_pattern_returns_low_confidence(self):
        result = calibrate_pattern_confidence(
            "totally_unknown_bias", 0.9, [], {}, [], 120, 100, 10000,
        )
        assert result["confidence_level"] == "low"
        assert result["calibrated_confidence"] < 0.5

    def test_all_known_biases_have_rules(self):
        known_biases = [
            "fomo", "loss_aversion", "anchoring", "overconfidence",
            "impulsivity", "recency_bias", "confirmation_bias", "social_proof_reliance",
        ]
        for bias in known_biases:
            assert bias in EVIDENCE_RULES, f"No rules for {bias}"
            assert len(EVIDENCE_RULES[bias]["signals"]) >= 3

    def test_fomo_detected_with_evidence(self):
        """FOMO should be high-confidence when there's a spike + fast buy + no info."""
        tl = _spike_timeline(100, 120, spike_at=20, spike_pct=0.08)
        decisions = [
            _mock_decision(decision_type="buy", simulation_time=25, time_spent_seconds=2,
                           info_viewed=[], confidence_level=5, amount=20, price_at_decision=108),
        ]
        result = calibrate_pattern_confidence(
            "fomo", 0.8, decisions, tl, [], 120, 100, 10000,
        )
        matched = sum(1 for e in result["evidence_details"] if e["matched"])
        assert matched >= 2, f"Expected ≥2 matched signals, got {matched}"
        assert result["confidence_level"] in ("high", "medium")

    def test_no_evidence_returns_insufficient(self):
        """Calm decisions should produce 'insufficient' for FOMO."""
        decisions = [
            _mock_decision(decision_type="hold", simulation_time=60, time_spent_seconds=15,
                           info_viewed=["news", "social", "chart"], confidence_level=3),
        ]
        result = calibrate_pattern_confidence(
            "fomo", 0.5, decisions, _simple_timeline(), [], 120, 100, 10000,
        )
        assert result["confidence_level"] in ("insufficient", "low")

    def test_calibrate_all_patterns_structure(self):
        patterns = [
            {"pattern_name": "fomo", "confidence": 0.7},
            {"pattern_name": "overconfidence", "confidence": 0.6},
        ]
        result = calibrate_all_patterns(
            patterns, [_mock_decision()], _simple_timeline(), [], 120, 100, 10000,
        )
        assert "calibrated_patterns" in result
        assert "abstained_patterns" in result
        assert "overall_evidence_quality" in result
        assert "summary" in result

    def test_empty_patterns(self):
        result = calibrate_all_patterns([], [], {}, [], 120, 100, 10000)
        assert result["overall_evidence_quality"] == "weak"
        assert result["calibrated_patterns"] == []

    def test_evidence_details_present(self):
        decisions = [_mock_decision(time_spent_seconds=2, info_viewed=[])]
        result = calibrate_pattern_confidence(
            "impulsivity", 0.7, decisions, _simple_timeline(), [], 120, 100, 10000,
        )
        assert len(result["evidence_details"]) > 0
        for detail in result["evidence_details"]:
            assert "signal" in detail
            assert "matched" in detail
            assert "weight" in detail

    def test_calibrated_confidence_blends_evidence_and_gemini(self):
        decisions = [_mock_decision(time_spent_seconds=1, info_viewed=[])]
        result = calibrate_pattern_confidence(
            "impulsivity", 0.9, decisions, _simple_timeline(), [], 120, 100, 10000,
        )
        assert 0 <= result["calibrated_confidence"] <= 1
        assert result["calibrated_confidence"] > 0.2

    def test_loss_aversion_with_held_through_drop(self):
        """Loss aversion should detect when user holds through a price drop."""
        tl = {}
        for t in range(121):
            if t < 30:
                tl[t] = 100
            elif t < 60:
                tl[t] = 100 - (t - 30) * 0.5
            else:
                tl[t] = 85
        decisions = [
            _mock_decision(decision_type="buy", simulation_time=25, amount=10, price_at_decision=100),
        ]
        result = calibrate_pattern_confidence(
            "loss_aversion", 0.7, decisions, tl, [], 120, 100, 10000,
        )
        matched = sum(1 for e in result["evidence_details"] if e["matched"])
        assert matched >= 1, "Should detect held_through_drop"


class TestConfidenceCalibratorEdgeCases:
    def test_no_decisions(self):
        result = calibrate_pattern_confidence(
            "fomo", 0.5, [], _simple_timeline(), [], 120, 100, 10000,
        )
        matched = sum(1 for e in result["evidence_details"] if e["matched"])
        assert matched == 0

    def test_weights_sum_to_one(self):
        """All signal weights for each bias should sum to approximately 1.0."""
        for bias, rules in EVIDENCE_RULES.items():
            total = sum(s["weight"] for s in rules["signals"])
            assert abs(total - 1.0) < 0.01, f"{bias} weights sum to {total}, expected ~1.0"
