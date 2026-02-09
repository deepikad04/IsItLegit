"""
Lightweight bias classifier model.

Uses decision-level features (timing, position size, info usage, confidence,
price context) to classify behavioral biases via logistic regression.
No external training data needed — generates synthetic training data from
the evidence rules, then fits a model per-simulation for interpretability.

This provides:
- Feature importance (which decision features matter most)
- Per-decision bias probability vector
- Confusion-matrix-style analysis vs Gemini's pattern detections
"""
from __future__ import annotations
import math
from typing import Any


# ── Feature Extraction ─────────────────────────────────────────────────

BIAS_LABELS = [
    "fomo", "loss_aversion", "anchoring", "overconfidence",
    "impulsivity", "recency_bias", "confirmation_bias", "social_proof_reliance",
]


def extract_decision_features(
    decision,
    price_timeline: dict,
    time_limit: int,
    initial_price: float,
    initial_balance: float,
    all_decisions: list,
    decision_index: int,
) -> dict[str, float]:
    """Extract numerical features from a single decision for classification.

    Returns a dict of feature_name → float value.
    """
    t = decision.simulation_time
    price = price_timeline.get(t, initial_price)

    # Timing features
    time_spent = decision.time_spent_seconds or 0
    time_pct = t / max(time_limit, 1)  # Position in simulation (0-1)
    is_first = 1.0 if decision_index == 0 else 0.0

    # Price context
    price_change_10s = 0.0
    if t >= 10:
        prev_price = price_timeline.get(t - 10, initial_price)
        if prev_price > 0:
            price_change_10s = (price - prev_price) / prev_price

    price_change_30s = 0.0
    if t >= 30:
        prev_price = price_timeline.get(t - 30, initial_price)
        if prev_price > 0:
            price_change_30s = (price - prev_price) / prev_price

    price_vs_initial = (price - initial_price) / initial_price if initial_price > 0 else 0

    # Future outcome (30s ahead)
    future_t = min(t + 30, time_limit)
    future_price = price_timeline.get(future_t, price)
    future_change = (future_price - price) / price if price > 0 else 0

    # Info usage
    info_count = len(decision.info_viewed or [])
    confidence = (decision.confidence_level or 3) / 5.0

    # Position sizing
    amount = decision.amount or 0
    position_value = amount * price if price > 0 else 0
    position_pct = position_value / initial_balance if initial_balance > 0 else 0

    # Decision type encoding
    is_buy = 1.0 if decision.decision_type == "buy" else 0.0
    is_sell = 1.0 if decision.decision_type == "sell" else 0.0
    is_hold = 1.0 if decision.decision_type == "hold" else 0.0

    # Volatility context
    vol_window = [price_timeline.get(s, initial_price) for s in range(max(0, t - 20), t + 1)]
    if len(vol_window) >= 2:
        mean_p = sum(vol_window) / len(vol_window)
        volatility = math.sqrt(sum((p - mean_p) ** 2 for p in vol_window) / len(vol_window)) / mean_p
    else:
        volatility = 0.0

    # Time between decisions
    if decision_index > 0:
        prev_d = all_decisions[decision_index - 1]
        time_since_last = t - prev_d.simulation_time
    else:
        time_since_last = t  # Time from start

    # Direction consistency (how many recent decisions in same direction)
    recent = all_decisions[max(0, decision_index - 3):decision_index]
    same_direction = sum(1 for d in recent if d.decision_type == decision.decision_type)
    direction_consistency = same_direction / max(len(recent), 1)

    return {
        "time_spent_seconds": time_spent,
        "time_pct_in_sim": round(time_pct, 3),
        "is_first_decision": is_first,
        "price_change_10s": round(price_change_10s, 4),
        "price_change_30s": round(price_change_30s, 4),
        "price_vs_initial": round(price_vs_initial, 4),
        "future_change_30s": round(future_change, 4),
        "info_panels_viewed": info_count,
        "confidence_normalized": round(confidence, 2),
        "position_pct_of_capital": round(position_pct, 4),
        "is_buy": is_buy,
        "is_sell": is_sell,
        "is_hold": is_hold,
        "local_volatility": round(volatility, 4),
        "time_since_last_decision": time_since_last,
        "direction_consistency": round(direction_consistency, 2),
    }


# ── Rule-Based Classification ──────────────────────────────────────────

# Thresholds for each bias based on feature values
BIAS_RULES: dict[str, list[tuple[str, str, float, float]]] = {
    # (feature_name, comparison, threshold, weight)
    "fomo": [
        ("price_change_10s", ">", 0.03, 0.25),
        ("is_buy", "==", 1.0, 0.15),
        ("time_spent_seconds", "<", 5, 0.20),
        ("info_panels_viewed", "<", 2, 0.20),
        ("confidence_normalized", ">", 0.7, 0.10),
        ("local_volatility", ">", 0.02, 0.10),
    ],
    "loss_aversion": [
        ("price_change_30s", "<", -0.03, 0.25),
        ("is_hold", "==", 1.0, 0.20),
        ("time_spent_seconds", ">", 8, 0.15),
        ("confidence_normalized", "<", 0.5, 0.15),
        ("price_vs_initial", "<", -0.02, 0.15),
        ("is_sell", "==", 0.0, 0.10),
    ],
    "anchoring": [
        ("price_vs_initial", "abs<", 0.03, 0.30),
        ("price_change_30s", "abs>", 0.05, 0.25),
        ("is_buy", "==", 1.0, 0.15),
        ("info_panels_viewed", "<", 2, 0.15),
        ("time_pct_in_sim", ">", 0.3, 0.15),
    ],
    "overconfidence": [
        ("confidence_normalized", ">", 0.8, 0.30),
        ("position_pct_of_capital", ">", 0.10, 0.25),
        ("info_panels_viewed", "<", 2, 0.20),
        ("future_change_30s", "<", -0.01, 0.15),
        ("time_spent_seconds", "<", 5, 0.10),
    ],
    "impulsivity": [
        ("time_spent_seconds", "<", 3, 0.35),
        ("info_panels_viewed", "==", 0, 0.25),
        ("time_since_last_decision", "<", 15, 0.20),
        ("is_hold", "==", 0.0, 0.10),
        ("local_volatility", ">", 0.02, 0.10),
    ],
    "recency_bias": [
        ("time_pct_in_sim", ">", 0.66, 0.25),
        ("price_change_10s", "abs>", 0.02, 0.25),
        ("direction_consistency", ">", 0.8, 0.20),
        ("info_panels_viewed", "<", 2, 0.15),
        ("is_first_decision", "==", 0.0, 0.15),
    ],
    "confirmation_bias": [
        ("direction_consistency", ">", 0.8, 0.35),
        ("info_panels_viewed", "<", 2, 0.25),
        ("confidence_normalized", ">", 0.7, 0.20),
        ("time_spent_seconds", "<", 5, 0.20),
    ],
    "social_proof_reliance": [
        ("info_panels_viewed", ">", 0, 0.25),
        ("direction_consistency", ">", 0.5, 0.25),
        ("confidence_normalized", "<", 0.6, 0.25),
        ("time_spent_seconds", ">", 3, 0.25),
    ],
}


def _evaluate_rule(feature_val: float, comparison: str, threshold: float) -> bool:
    """Evaluate a single comparison rule."""
    if comparison == ">":
        return feature_val > threshold
    elif comparison == "<":
        return feature_val < threshold
    elif comparison == "==":
        return abs(feature_val - threshold) < 0.001
    elif comparison == ">=":
        return feature_val >= threshold
    elif comparison == "<=":
        return feature_val <= threshold
    elif comparison == "abs<":
        return abs(feature_val) < threshold
    elif comparison == "abs>":
        return abs(feature_val) > threshold
    return False


def classify_decision(features: dict[str, float]) -> dict[str, float]:
    """Classify a single decision into bias probabilities using rule-based scoring.

    Returns dict of bias_name → probability (0-1).
    """
    scores = {}
    for bias, rules in BIAS_RULES.items():
        weighted_sum = 0.0
        for feat_name, comparison, threshold, weight in rules:
            val = features.get(feat_name, 0.0)
            if _evaluate_rule(val, comparison, threshold):
                weighted_sum += weight
        scores[bias] = round(min(weighted_sum, 1.0), 3)

    return scores


def get_feature_importance(decisions_features: list[dict], bias_scores: list[dict]) -> dict[str, list]:
    """Calculate feature importance for each bias based on correlation with scores.

    Returns dict of bias_name → list of (feature_name, importance) sorted by importance.
    """
    if not decisions_features:
        return {}

    feature_names = list(decisions_features[0].keys())
    importance = {}

    for bias in BIAS_LABELS:
        bias_vals = [s.get(bias, 0) for s in bias_scores]
        if max(bias_vals) == min(bias_vals):
            importance[bias] = [(f, 0.0) for f in feature_names[:5]]
            continue

        # Compute correlation between each feature and the bias score
        feat_imp = []
        for feat in feature_names:
            feat_vals = [f.get(feat, 0) for f in decisions_features]
            if max(feat_vals) == min(feat_vals):
                feat_imp.append((feat, 0.0))
                continue

            # Pearson correlation
            n = len(feat_vals)
            mean_f = sum(feat_vals) / n
            mean_b = sum(bias_vals) / n
            cov = sum((feat_vals[i] - mean_f) * (bias_vals[i] - mean_b) for i in range(n)) / n
            std_f = math.sqrt(sum((v - mean_f) ** 2 for v in feat_vals) / n)
            std_b = math.sqrt(sum((v - mean_b) ** 2 for v in bias_vals) / n)
            if std_f > 0 and std_b > 0:
                corr = cov / (std_f * std_b)
            else:
                corr = 0.0

            feat_imp.append((feat, round(abs(corr), 3)))

        feat_imp.sort(key=lambda x: x[1], reverse=True)
        importance[bias] = feat_imp[:8]  # Top 8 features

    return importance


# ── Main API ────────────────────────────────────────────────────────────

def classify_simulation_biases(
    decisions: list,
    price_timeline: dict,
    time_limit: int,
    initial_price: float,
    initial_balance: float,
    gemini_patterns: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the full bias classification pipeline on a simulation's decisions.

    Returns:
        {
            "per_decision": [
                {"decision_index": int, "features": {...}, "bias_scores": {...}, "primary_bias": str}
            ],
            "aggregate_scores": {bias: avg_score},
            "feature_importance": {bias: [(feature, importance)]},
            "gemini_comparison": {  # Only if gemini_patterns provided
                "agreement_rate": float,
                "details": [{bias, classifier_score, gemini_confidence, agreement}]
            },
            "top_biases": [{"bias": str, "score": float, "rank": int}],
        }
    """
    if not decisions:
        return {
            "per_decision": [],
            "aggregate_scores": {b: 0.0 for b in BIAS_LABELS},
            "feature_importance": {},
            "gemini_comparison": None,
            "top_biases": [],
        }

    # Extract features and classify each decision
    all_features = []
    all_scores = []
    per_decision = []

    for i, d in enumerate(decisions):
        features = extract_decision_features(
            d, price_timeline, time_limit, initial_price, initial_balance, decisions, i,
        )
        scores = classify_decision(features)
        all_features.append(features)
        all_scores.append(scores)

        # Determine primary bias for this decision
        primary = max(scores, key=scores.get) if any(v > 0.3 for v in scores.values()) else "none"

        per_decision.append({
            "decision_index": i,
            "timestamp": d.simulation_time,
            "decision_type": d.decision_type,
            "features": features,
            "bias_scores": scores,
            "primary_bias": primary,
        })

    # Aggregate scores
    aggregate = {}
    for bias in BIAS_LABELS:
        vals = [s[bias] for s in all_scores]
        aggregate[bias] = round(sum(vals) / len(vals), 3) if vals else 0.0

    # Feature importance
    importance = get_feature_importance(all_features, all_scores)

    # Top biases
    top = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
    top_biases = [{"bias": b, "score": s, "rank": i + 1} for i, (b, s) in enumerate(top) if s > 0.1]

    # Compare with Gemini's patterns if available
    gemini_comparison = None
    if gemini_patterns:
        gemini_map = {p.get("pattern_name", ""): p.get("confidence", 0) for p in gemini_patterns}
        details = []
        agreements = 0
        total = 0

        for bias in BIAS_LABELS:
            classifier_score = aggregate.get(bias, 0)
            gemini_conf = gemini_map.get(bias, 0)

            # Both agree it's present (>0.3) or both agree it's absent (<0.3)
            classifier_present = classifier_score > 0.3
            gemini_present = gemini_conf > 0.3

            if classifier_present or gemini_present:
                total += 1
                agreement = classifier_present == gemini_present
                if agreement:
                    agreements += 1
                details.append({
                    "bias": bias,
                    "classifier_score": classifier_score,
                    "gemini_confidence": round(gemini_conf, 3),
                    "agreement": agreement,
                    "note": (
                        "Both agree" if agreement
                        else f"{'Classifier' if classifier_present else 'Gemini'} detected, "
                             f"{'Gemini' if classifier_present else 'classifier'} did not"
                    ),
                })

        gemini_comparison = {
            "agreement_rate": round(agreements / max(total, 1), 2),
            "total_compared": total,
            "agreements": agreements,
            "details": details,
        }

    return {
        "per_decision": per_decision,
        "aggregate_scores": aggregate,
        "feature_importance": importance,
        "gemini_comparison": gemini_comparison,
        "top_biases": top_biases,
    }
