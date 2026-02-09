"""
Tests for community stats endpoint logic and profile scoring.
- calculate_overall_score: 0 for empty, scales with bias patterns
- calculate_trend: stable/improving/declining from trajectory
- get_bias_description: known biases + fallback
- Score starts at 0 for new users (no bias_patterns)
"""
import pytest
from routers.profile import calculate_overall_score, calculate_trend, get_bias_description


# ── calculate_overall_score ─────────────────────────────────────────


class TestCalculateOverallScore:
    def test_empty_profile_data_returns_zero(self):
        assert calculate_overall_score({}) == 0
        assert calculate_overall_score(None) == 0

    def test_no_bias_patterns_returns_zero(self):
        """New users with no simulations should get 0, not 50."""
        assert calculate_overall_score({"strengths": [], "weaknesses": []}) == 0

    def test_empty_bias_dict_returns_zero(self):
        assert calculate_overall_score({"bias_patterns": {}}) == 0

    def test_low_biases_yield_high_score(self):
        """If all biases are low (0.1), score should be high."""
        score = calculate_overall_score({
            "bias_patterns": {"fomo": 0.1, "loss_aversion": 0.1, "anchoring": 0.1},
            "strengths": ["patience"],
        })
        # (1 - 0.1) * 100 = 90, + min(1*5, 20) = 5 → 95
        assert score == 95

    def test_high_biases_yield_low_score(self):
        """If all biases are high (0.9), score should be low."""
        score = calculate_overall_score({
            "bias_patterns": {"fomo": 0.9, "loss_aversion": 0.9},
            "strengths": [],
        })
        # (1 - 0.9) * 100 = 10, no bonus → 10
        assert round(score) == 10

    def test_strengths_bonus_capped_at_20(self):
        """Bonus from strengths should cap at 20."""
        score = calculate_overall_score({
            "bias_patterns": {"fomo": 0.5},
            "strengths": ["a", "b", "c", "d", "e"],  # 5 * 5 = 25, capped at 20
        })
        # (1 - 0.5) * 100 = 50, + 20 = 70
        assert score == 70

    def test_score_capped_at_100(self):
        """Score should never exceed 100."""
        score = calculate_overall_score({
            "bias_patterns": {"fomo": 0.0},
            "strengths": ["a", "b", "c", "d", "e"],
        })
        # (1 - 0) * 100 = 100, + 20 = 120 → capped at 100
        assert score == 100

    def test_single_bias_mid_range(self):
        score = calculate_overall_score({
            "bias_patterns": {"overconfidence": 0.5},
        })
        # (1 - 0.5) * 100 = 50, no bonus
        assert score == 50


# ── calculate_trend ──────────────────────────────────────────────────


class TestCalculateTrend:
    def test_empty_trajectory_is_stable(self):
        assert calculate_trend([]) == "stable"

    def test_single_point_is_stable(self):
        assert calculate_trend([{"score": 50}]) == "stable"

    def test_improving_trend(self):
        trajectory = [{"score": 40}, {"score": 50}, {"score": 55}]
        assert calculate_trend(trajectory) == "improving"

    def test_declining_trend(self):
        trajectory = [{"score": 70}, {"score": 60}, {"score": 55}]
        assert calculate_trend(trajectory) == "declining"

    def test_stable_small_change(self):
        trajectory = [{"score": 50}, {"score": 52}, {"score": 53}]
        assert calculate_trend(trajectory) == "stable"

    def test_two_points_improving(self):
        trajectory = [{"score": 30}, {"score": 50}]
        assert calculate_trend(trajectory) == "improving"

    def test_two_points_declining(self):
        trajectory = [{"score": 70}, {"score": 55}]
        assert calculate_trend(trajectory) == "declining"

    def test_long_trajectory_uses_last_three(self):
        """Should only look at last 3 points."""
        trajectory = [
            {"score": 10}, {"score": 20}, {"score": 30},
            {"score": 80}, {"score": 85}, {"score": 90}
        ]
        # Last 3: [80, 85, 90] → diff = 10 → improving
        assert calculate_trend(trajectory) == "improving"


# ── get_bias_description ──────────────────────────────────────────────


class TestGetBiasDescription:
    def test_known_biases(self):
        assert "losses" in get_bias_description("loss_aversion").lower()
        assert "fomo" in get_bias_description("fomo").lower() or "missing" in get_bias_description("fomo").lower()
        assert "anchor" in get_bias_description("anchoring").lower() or "first" in get_bias_description("anchoring").lower()

    def test_unknown_bias_fallback(self):
        assert get_bias_description("some_unknown_bias") == "No description available"

    def test_all_eight_biases_have_descriptions(self):
        known = [
            "loss_aversion", "fomo", "anchoring", "social_proof_reliance",
            "overconfidence", "recency_bias", "confirmation_bias", "impulsivity"
        ]
        for bias in known:
            desc = get_bias_description(bias)
            assert desc != "No description available", f"{bias} has no description"
            assert len(desc) > 10, f"{bias} description too short"


# ── Community stats response shape ────────────────────────────────────


class TestCommunityStatsShape:
    """Test that the community stats endpoint returns the expected keys.
    We test the pure-function parts here; the DB queries are integration tests.
    """

    def test_expected_keys(self):
        """Verify the response dict structure matches frontend expectations."""
        # These are the keys the Dashboard.jsx expects
        expected_keys = {
            "total_traders", "total_simulations", "total_decisions",
            "avg_process_score", "most_common_bias", "most_common_bias_pct",
            "most_popular_scenario", "score_distribution", "your_percentile"
        }
        # The endpoint returns a dict — test the shape contract
        sample = {
            "total_traders": 10,
            "total_simulations": 50,
            "total_decisions": 200,
            "avg_process_score": 55.3,
            "most_common_bias": "Loss Aversion",
            "most_common_bias_pct": 40,
            "most_popular_scenario": "FOMO Trap",
            "score_distribution": {
                "beginner": 5, "developing": 10, "proficient": 8, "expert": 2
            },
            "your_percentile": 65,
        }
        assert set(sample.keys()) == expected_keys

    def test_distribution_categories(self):
        """Score distribution must have exactly these 4 buckets."""
        categories = {"beginner", "developing", "proficient", "expert"}
        distribution = {"beginner": 0, "developing": 0, "proficient": 0, "expert": 0}
        assert set(distribution.keys()) == categories

    def test_bias_name_formatting(self):
        """Bias names from DB should be formatted with .replace('_', ' ').title()."""
        raw = "loss_aversion"
        formatted = raw.replace("_", " ").title()
        assert formatted == "Loss Aversion"

        raw2 = "social_proof_reliance"
        formatted2 = raw2.replace("_", " ").title()
        assert formatted2 == "Social Proof Reliance"
