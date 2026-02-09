"""
Confidence calibration for AI outputs.

Scores Gemini's pattern detections against observable evidence from the
simulation timeline, providing High/Medium/Low confidence labels with
reasoning. This allows the system to "abstain" when evidence is thin.
"""
from __future__ import annotations
from typing import Any


# Bias → what observable signals we'd expect in the decision data
EVIDENCE_RULES: dict[str, dict[str, Any]] = {
    "fomo": {
        "label": "FOMO",
        "signals": [
            {"check": "rapid_buy_after_spike", "weight": 0.3,
             "desc": "Bought within 10s of a >5% price increase"},
            {"check": "ignored_info", "weight": 0.2,
             "desc": "Traded without checking info panels"},
            {"check": "high_confidence_thin_info", "weight": 0.2,
             "desc": "High confidence (4-5) with <2 info sources viewed"},
            {"check": "large_position", "weight": 0.15,
             "desc": "Position size >10% of portfolio"},
            {"check": "fast_decision", "weight": 0.15,
             "desc": "Decision made in <5 seconds"},
        ],
    },
    "loss_aversion": {
        "label": "Loss Aversion",
        "signals": [
            {"check": "held_through_drop", "weight": 0.35,
             "desc": "Held position through >5% price drop without acting"},
            {"check": "panic_sell_bottom", "weight": 0.3,
             "desc": "Sold near the lowest price in the session"},
            {"check": "refused_to_sell_at_loss", "weight": 0.2,
             "desc": "Held losing position >30s after price dropped below entry"},
            {"check": "asymmetric_confidence", "weight": 0.15,
             "desc": "Higher confidence on buys than sells"},
        ],
    },
    "anchoring": {
        "label": "Anchoring",
        "signals": [
            {"check": "bought_near_initial", "weight": 0.35,
             "desc": "Bought when price was within 3% of initial price after large move"},
            {"check": "ignored_trend", "weight": 0.3,
             "desc": "Traded against a clear price trend (>5% move)"},
            {"check": "repeated_price_reference", "weight": 0.2,
             "desc": "Multiple trades at similar price levels"},
            {"check": "early_anchor", "weight": 0.15,
             "desc": "First price seen strongly influenced later decisions"},
        ],
    },
    "overconfidence": {
        "label": "Overconfidence",
        "signals": [
            {"check": "high_confidence_bad_outcome", "weight": 0.3,
             "desc": "Confidence ≥4 on decisions that went against the trader"},
            {"check": "large_positions", "weight": 0.25,
             "desc": "Position sizes >15% of portfolio"},
            {"check": "few_info_checks", "weight": 0.25,
             "desc": "Fewer than average info panel checks"},
            {"check": "fast_decisions", "weight": 0.2,
             "desc": "Average decision time <5 seconds"},
        ],
    },
    "impulsivity": {
        "label": "Impulsivity",
        "signals": [
            {"check": "very_fast_decision", "weight": 0.35,
             "desc": "Decision made in <3 seconds"},
            {"check": "no_info_viewed", "weight": 0.3,
             "desc": "No information panels viewed before trade"},
            {"check": "rapid_reversal", "weight": 0.2,
             "desc": "Buy then sell (or reverse) within 15 seconds"},
            {"check": "many_trades", "weight": 0.15,
             "desc": "More than 5 trades in a short simulation"},
        ],
    },
    "recency_bias": {
        "label": "Recency Bias",
        "signals": [
            {"check": "trades_follow_recent_move", "weight": 0.4,
             "desc": "Trades aligned with the last 10s price direction"},
            {"check": "ignored_earlier_data", "weight": 0.3,
             "desc": "Didn't check historical prices or earlier news"},
            {"check": "late_decisions", "weight": 0.3,
             "desc": "Most decisions in the last third of the simulation"},
        ],
    },
    "confirmation_bias": {
        "label": "Confirmation Bias",
        "signals": [
            {"check": "selective_info", "weight": 0.4,
             "desc": "Only viewed info that confirms the position direction"},
            {"check": "ignored_contrary_news", "weight": 0.35,
             "desc": "Contrary news/social appeared but was not reflected in decisions"},
            {"check": "consistent_direction", "weight": 0.25,
             "desc": "All trades in the same direction despite mixed signals"},
        ],
    },
    "social_proof_reliance": {
        "label": "Social Proof",
        "signals": [
            {"check": "traded_with_crowd", "weight": 0.4,
             "desc": "Trades aligned with crowd sentiment direction"},
            {"check": "viewed_social_before_trade", "weight": 0.3,
             "desc": "Checked social panel right before trading"},
            {"check": "followed_social_sentiment", "weight": 0.3,
             "desc": "Trade direction matched social sentiment polarity"},
        ],
    },
}


def _check_signal(check_name: str, decisions: list, price_timeline: dict,
                  events: list, time_limit: int, initial_price: float,
                  initial_balance: float) -> tuple[bool, str | None]:
    """Evaluate a single evidence signal against the decision data.

    Returns (matched: bool, detail: str | None).
    """
    if not decisions:
        return False, None

    trade_decisions = [d for d in decisions if d.decision_type in ("buy", "sell")]

    # ── FOMO signals ──
    if check_name == "rapid_buy_after_spike":
        for d in trade_decisions:
            if d.decision_type != "buy":
                continue
            t = d.simulation_time
            prev_price = price_timeline.get(max(0, t - 10), initial_price)
            cur_price = price_timeline.get(t, initial_price)
            if prev_price > 0 and (cur_price - prev_price) / prev_price > 0.05:
                return True, f"Bought at t={t}s after {((cur_price-prev_price)/prev_price)*100:.1f}% spike"
        return False, None

    if check_name == "ignored_info":
        for d in trade_decisions:
            if not d.info_viewed or len(d.info_viewed) == 0:
                return True, f"{d.decision_type} at t={d.simulation_time}s with no info viewed"
        return False, None

    if check_name == "high_confidence_thin_info":
        for d in trade_decisions:
            conf = d.confidence_level or 3
            info_count = len(d.info_viewed or [])
            if conf >= 4 and info_count < 2:
                return True, f"Confidence {conf}/5 but only {info_count} info sources at t={d.simulation_time}s"
        return False, None

    if check_name == "large_position":
        for d in trade_decisions:
            if d.decision_type == "buy" and d.amount and d.price_at_decision:
                position_value = d.amount * d.price_at_decision
                if position_value > initial_balance * 0.10:
                    return True, f"Position {position_value:.0f} is {position_value/initial_balance*100:.0f}% of capital"
        return False, None

    if check_name == "fast_decision":
        for d in trade_decisions:
            if (d.time_spent_seconds or 0) < 5:
                return True, f"{d.decision_type} in {d.time_spent_seconds}s at t={d.simulation_time}s"
        return False, None

    # ── Loss aversion signals ──
    if check_name == "held_through_drop":
        # Check if there's a >5% drop where user had holdings but didn't sell
        buy_times = [d.simulation_time for d in trade_decisions if d.decision_type == "buy"]
        sell_times = [d.simulation_time for d in trade_decisions if d.decision_type == "sell"]
        for bt in buy_times:
            buy_price = price_timeline.get(bt, initial_price)
            # Check next 60s for a 5% drop
            for t in range(bt + 1, min(bt + 60, time_limit + 1)):
                p = price_timeline.get(t, initial_price)
                if buy_price > 0 and (p - buy_price) / buy_price < -0.05:
                    # Did they sell during this drop?
                    sold = any(bt < st <= t + 5 for st in sell_times)
                    if not sold:
                        return True, f"Held through {((p-buy_price)/buy_price)*100:.1f}% drop at t={t}s"
        return False, None

    if check_name == "panic_sell_bottom":
        if not trade_decisions:
            return False, None
        sell_decisions = [d for d in trade_decisions if d.decision_type == "sell"]
        if not sell_decisions:
            return False, None
        prices = [price_timeline.get(t, initial_price) for t in range(time_limit + 1)]
        if not prices:
            return False, None
        min_price = min(prices)
        for d in sell_decisions:
            p = price_timeline.get(d.simulation_time, initial_price)
            if min_price > 0 and (p - min_price) / min_price < 0.03:
                return True, f"Sold at t={d.simulation_time}s near session low"
        return False, None

    if check_name == "refused_to_sell_at_loss":
        # Similar to held_through_drop but longer duration
        return False, None  # Covered by held_through_drop

    if check_name == "asymmetric_confidence":
        buys = [d for d in trade_decisions if d.decision_type == "buy"]
        sells = [d for d in trade_decisions if d.decision_type == "sell"]
        if buys and sells:
            avg_buy_conf = sum(d.confidence_level or 3 for d in buys) / len(buys)
            avg_sell_conf = sum(d.confidence_level or 3 for d in sells) / len(sells)
            if avg_buy_conf - avg_sell_conf > 0.5:
                return True, f"Avg buy confidence {avg_buy_conf:.1f} vs sell {avg_sell_conf:.1f}"
        return False, None

    # ── Anchoring signals ──
    if check_name == "bought_near_initial":
        for d in trade_decisions:
            if d.decision_type == "buy":
                p = price_timeline.get(d.simulation_time, initial_price)
                if initial_price > 0 and abs(p - initial_price) / initial_price < 0.03:
                    # Was there a large move at some point?
                    max_move = max(abs(price_timeline.get(t, initial_price) - initial_price) / initial_price
                                   for t in range(min(d.simulation_time, time_limit + 1)))
                    if max_move > 0.05:
                        return True, f"Bought near initial price ${initial_price:.0f} despite {max_move*100:.0f}% swing"
        return False, None

    if check_name == "ignored_trend":
        for d in trade_decisions:
            t = d.simulation_time
            if t < 10:
                continue
            recent = price_timeline.get(t, initial_price)
            past = price_timeline.get(max(0, t - 20), initial_price)
            if past > 0:
                trend = (recent - past) / past
                if d.decision_type == "buy" and trend < -0.05:
                    return True, f"Bought during {trend*100:.1f}% downtrend at t={t}s"
                if d.decision_type == "sell" and trend > 0.05:
                    return True, f"Sold during {trend*100:.1f}% uptrend at t={t}s"
        return False, None

    if check_name in ("repeated_price_reference", "early_anchor"):
        return False, None  # Hard to detect from decision data alone

    # ── Overconfidence signals ──
    if check_name == "high_confidence_bad_outcome":
        for d in trade_decisions:
            conf = d.confidence_level or 3
            if conf >= 4:
                t = d.simulation_time
                future_t = min(t + 30, time_limit)
                p_now = price_timeline.get(t, initial_price)
                p_future = price_timeline.get(future_t, initial_price)
                if p_now > 0:
                    change = (p_future - p_now) / p_now
                    if d.decision_type == "buy" and change < -0.02:
                        return True, f"Confidence {conf}/5 on buy, price dropped {change*100:.1f}%"
                    if d.decision_type == "sell" and change > 0.02:
                        return True, f"Confidence {conf}/5 on sell, price rose {change*100:.1f}%"
        return False, None

    if check_name == "large_positions":
        for d in trade_decisions:
            if d.amount and d.price_at_decision:
                val = d.amount * d.price_at_decision
                if val > initial_balance * 0.15:
                    return True, f"Position {val:.0f} = {val/initial_balance*100:.0f}% of capital"
        return False, None

    if check_name == "few_info_checks":
        total_info = sum(len(d.info_viewed or []) for d in trade_decisions)
        avg = total_info / max(len(trade_decisions), 1)
        if avg < 1.5:
            return True, f"Average {avg:.1f} info checks per trade"
        return False, None

    if check_name == "fast_decisions":
        avg_time = sum(d.time_spent_seconds or 0 for d in trade_decisions) / max(len(trade_decisions), 1)
        if avg_time < 5:
            return True, f"Average decision time {avg_time:.1f}s"
        return False, None

    # ── Impulsivity signals ──
    if check_name == "very_fast_decision":
        for d in trade_decisions:
            if (d.time_spent_seconds or 0) < 3:
                return True, f"{d.decision_type} in {d.time_spent_seconds}s at t={d.simulation_time}s"
        return False, None

    if check_name == "no_info_viewed":
        for d in trade_decisions:
            if not d.info_viewed or len(d.info_viewed) == 0:
                return True, f"No info viewed before {d.decision_type} at t={d.simulation_time}s"
        return False, None

    if check_name == "rapid_reversal":
        for i, d in enumerate(trade_decisions[:-1]):
            next_d = trade_decisions[i + 1]
            if (d.decision_type != next_d.decision_type and
                    next_d.simulation_time - d.simulation_time < 15):
                return True, f"{d.decision_type}→{next_d.decision_type} in {next_d.simulation_time - d.simulation_time}s"
        return False, None

    if check_name == "many_trades":
        if len(trade_decisions) > 5:
            return True, f"{len(trade_decisions)} trades in {time_limit}s"
        return False, None

    # ── Recency bias signals ──
    if check_name == "trades_follow_recent_move":
        following = 0
        for d in trade_decisions:
            t = d.simulation_time
            if t < 5:
                continue
            recent = price_timeline.get(t, initial_price)
            past = price_timeline.get(max(0, t - 10), initial_price)
            if past > 0:
                move = (recent - past) / past
                if (d.decision_type == "buy" and move > 0.01) or (d.decision_type == "sell" and move < -0.01):
                    following += 1
        if following > len(trade_decisions) * 0.6 and following > 1:
            return True, f"{following}/{len(trade_decisions)} trades followed recent price direction"
        return False, None

    if check_name == "ignored_earlier_data":
        for d in trade_decisions:
            info = d.info_viewed or []
            if "historical" not in " ".join(info).lower() and len(info) < 2:
                return True, "Didn't review historical data"
        return False, None

    if check_name == "late_decisions":
        if not trade_decisions:
            return False, None
        late = sum(1 for d in trade_decisions if d.simulation_time > time_limit * 0.66)
        if late > len(trade_decisions) * 0.6:
            return True, f"{late}/{len(trade_decisions)} trades in final third"
        return False, None

    # ── Confirmation bias signals ──
    if check_name == "selective_info":
        return False, None  # Would need content analysis of info panels

    if check_name == "ignored_contrary_news":
        return False, None  # Would need content analysis

    if check_name == "consistent_direction":
        if len(trade_decisions) >= 3:
            types = set(d.decision_type for d in trade_decisions)
            if len(types) == 1:
                return True, f"All {len(trade_decisions)} trades were {trade_decisions[0].decision_type}"
        return False, None

    # ── Social proof signals ──
    if check_name == "traded_with_crowd":
        return False, None  # Would need crowd data from simulation state

    if check_name == "viewed_social_before_trade":
        for d in trade_decisions:
            if d.info_viewed and "social" in " ".join(d.info_viewed).lower():
                return True, f"Viewed social before {d.decision_type} at t={d.simulation_time}s"
        return False, None

    if check_name == "followed_social_sentiment":
        return False, None  # Would need social content analysis

    return False, None


def calibrate_pattern_confidence(
    pattern_name: str,
    gemini_confidence: float,
    decisions: list,
    price_timeline: dict,
    events: list,
    time_limit: int,
    initial_price: float,
    initial_balance: float,
) -> dict:
    """Score a single Gemini pattern detection against observable evidence.

    Returns:
        {
            "pattern": str,
            "gemini_confidence": float,
            "evidence_score": float (0-1),
            "calibrated_confidence": float (0-1),
            "confidence_level": "high" | "medium" | "low" | "insufficient",
            "evidence_details": [{"signal": str, "matched": bool, "detail": str}],
            "reasoning": str,
        }
    """
    rules = EVIDENCE_RULES.get(pattern_name, None)
    if not rules:
        return {
            "pattern": pattern_name,
            "gemini_confidence": gemini_confidence,
            "evidence_score": 0,
            "calibrated_confidence": gemini_confidence * 0.5,
            "confidence_level": "low",
            "evidence_details": [],
            "reasoning": f"No evidence rules defined for '{pattern_name}' — treating Gemini's assessment with caution.",
        }

    evidence_details = []
    weighted_score = 0.0

    for signal in rules["signals"]:
        matched, detail = _check_signal(
            signal["check"], decisions, price_timeline, events,
            time_limit, initial_price, initial_balance,
        )
        evidence_details.append({
            "signal": signal["desc"],
            "matched": matched,
            "detail": detail or "",
            "weight": signal["weight"],
        })
        if matched:
            weighted_score += signal["weight"]

    # Calibrated confidence: blend Gemini's confidence with evidence score
    # Evidence has 60% weight, Gemini has 40% weight
    calibrated = 0.6 * weighted_score + 0.4 * gemini_confidence

    # Determine confidence level
    matched_count = sum(1 for e in evidence_details if e["matched"])
    total_signals = len(evidence_details)

    if calibrated >= 0.7 and matched_count >= 2:
        level = "high"
    elif calibrated >= 0.4 and matched_count >= 1:
        level = "medium"
    elif matched_count == 0:
        level = "insufficient"
    else:
        level = "low"

    # Build reasoning
    if level == "high":
        matched_descs = [e["detail"] for e in evidence_details if e["matched"] and e["detail"]]
        reasoning = f"Strong evidence supports this pattern: {'; '.join(matched_descs[:3])}"
    elif level == "medium":
        reasoning = f"Some evidence found ({matched_count}/{total_signals} signals matched), but not conclusive."
    elif level == "insufficient":
        reasoning = (
            "No observable evidence found in the decision data to support this pattern. "
            "The AI may be over-interpreting limited data."
        )
    else:
        reasoning = f"Weak evidence ({matched_count}/{total_signals} signals). The pattern may be present but data is inconclusive."

    return {
        "pattern": pattern_name,
        "gemini_confidence": round(gemini_confidence, 3),
        "evidence_score": round(weighted_score, 3),
        "calibrated_confidence": round(calibrated, 3),
        "confidence_level": level,
        "evidence_details": evidence_details,
        "reasoning": reasoning,
    }


def _gather_evidence_signals(
    patterns: list[dict],
    decisions: list,
    price_timeline: dict,
    events: list,
    time_limit: int,
    initial_price: float,
    initial_balance: float,
) -> dict[str, list]:
    """Gather raw evidence signals for each detected pattern.

    Returns a dict of pattern_name → list of signal dicts with:
        {check, desc, weight, matched, detail}

    This is used to feed evidence data to Gemini for self-evaluation,
    separate from the calibration scoring logic.
    """
    result = {}
    for p in patterns:
        name = p.get("pattern_name", "")
        rules = EVIDENCE_RULES.get(name)
        if not rules:
            result[name] = []
            continue

        signals = []
        for signal in rules["signals"]:
            matched, detail = _check_signal(
                signal["check"], decisions, price_timeline, events,
                time_limit, initial_price, initial_balance,
            )
            signals.append({
                "check": signal["check"],
                "desc": signal["desc"],
                "weight": signal["weight"],
                "matched": matched,
                "detail": detail or "",
            })
        result[name] = signals

    return result


def calibrate_all_patterns(
    patterns: list[dict],
    decisions: list,
    price_timeline: dict,
    events: list,
    time_limit: int,
    initial_price: float,
    initial_balance: float,
) -> dict:
    """Calibrate confidence for all detected patterns.

    Args:
        patterns: List of dicts with 'pattern_name' and 'confidence' keys
                  (from Gemini's PatternDetection output).

    Returns:
        {
            "calibrated_patterns": [...],
            "overall_evidence_quality": "strong" | "moderate" | "weak",
            "abstained_patterns": [...],  # patterns with insufficient evidence
            "summary": str,
        }
    """
    calibrated = []
    abstained = []

    for p in patterns:
        name = p.get("pattern_name", "")
        conf = p.get("confidence", 0.5)

        result = calibrate_pattern_confidence(
            name, conf, decisions, price_timeline, events,
            time_limit, initial_price, initial_balance,
        )
        if result["confidence_level"] == "insufficient":
            abstained.append(result)
        else:
            calibrated.append(result)

    # Overall quality
    if not calibrated and not abstained:
        quality = "weak"
    elif len(abstained) > len(calibrated):
        quality = "weak"
    elif any(c["confidence_level"] == "high" for c in calibrated):
        quality = "strong"
    else:
        quality = "moderate"

    high_count = sum(1 for c in calibrated if c["confidence_level"] == "high")
    med_count = sum(1 for c in calibrated if c["confidence_level"] == "medium")

    summary_parts = []
    if high_count:
        summary_parts.append(f"{high_count} pattern(s) with strong evidence")
    if med_count:
        summary_parts.append(f"{med_count} with moderate evidence")
    if abstained:
        summary_parts.append(f"{len(abstained)} with insufficient evidence (abstained)")

    return {
        "calibrated_patterns": calibrated,
        "overall_evidence_quality": quality,
        "abstained_patterns": abstained,
        "summary": ". ".join(summary_parts) if summary_parts else "No patterns detected.",
    }
