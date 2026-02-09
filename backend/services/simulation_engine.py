"""
Deterministic simulation engine with full market realism.
Features: spread/slippage, fees, liquidity, volatility clustering, gaps/halts,
order types, time-pressure fills, correlated assets, micro-structure noise,
margin/leverage, drawdown limits, news latency, macro indicators, crowd model.
"""
import math
import random
from typing import Any
from models.scenario import Scenario
from schemas.decision import DecisionCreate

# Module-level cache keyed on (scenario.id, market_params_hash)
_timeline_cache: dict[str, dict] = {}


class SimulationEngine:
    """Handles simulation state and deterministic calculations."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.initial_data = scenario.initial_data
        self.events = scenario.events
        self.time_limit = scenario.time_pressure_seconds

        self.asset = self.initial_data.get("asset", "ASSET")
        self.initial_price = self.initial_data.get("price", 100)
        self.initial_balance = self.initial_data.get("your_balance", 10000)

        # Load market realism parameters (all backward-compatible defaults)
        self.market_params = self._load_market_params()

        # Secondary asset for correlated trading
        self.secondary_asset = self.initial_data.get("secondary_asset", None)
        self.correlation = self.initial_data.get("correlation", 0.7)
        self.secondary_price = self.initial_data.get("secondary_price", None)

        # Build timelines (cached)
        cache_key = f"{scenario.id}:{hash(str(self.market_params))}"
        if cache_key in _timeline_cache:
            cached = _timeline_cache[cache_key]
            self.price_timeline = cached["price"]
            self.volatility_timeline = cached["vol"]
            self.halt_periods = cached["halts"]
            self.secondary_timeline = cached.get("secondary", {})
        else:
            self._build_price_timeline()
            _timeline_cache[cache_key] = {
                "price": self.price_timeline,
                "vol": self.volatility_timeline,
                "halts": self.halt_periods,
                "secondary": getattr(self, "secondary_timeline", {}),
            }

    def _load_market_params(self) -> dict:
        """Extract market parameters from scenario with backward-compatible defaults."""
        mp = self.initial_data.get("market_params", {})
        return {
            # Feature 2: Transaction costs
            "fixed_fee": mp.get("fixed_fee", 0.0),
            "pct_fee": mp.get("pct_fee", 0.0),
            # Feature 1 & 3: Spread + liquidity
            "base_spread_pct": mp.get("base_spread_pct", 0.0),
            "volume_per_tick": mp.get("volume_per_tick", float("inf")),
            "max_trade_per_tick": mp.get("max_trade_per_tick", float("inf")),
            # Feature 5: Regime shifts
            "volatility_clustering": mp.get("volatility_clustering", False),
            "vol_params": mp.get("vol_params", {}),
            # Feature 6: Gaps and halts
            "halts_enabled": mp.get("halts_enabled", False),
            "halt_threshold_pct": mp.get("halt_threshold_pct", 0.10),
            "halt_duration": mp.get("halt_duration", 10),
            # Feature 4: News latency
            "news_latency_enabled": mp.get("news_latency_enabled", False),
            # Feature 7: Order types
            "order_types_enabled": mp.get("order_types_enabled", False),
            # Feature 8: Time pressure fills
            "time_pressure_fills": mp.get("time_pressure_fills", False),
            # Feature 10: Market microstructure
            "microstructure_noise": mp.get("microstructure_noise", False),
            # Feature 11: Margin/leverage
            "margin_enabled": mp.get("margin_enabled", False),
            "max_leverage": mp.get("max_leverage", 1.0),
            "maintenance_margin": mp.get("maintenance_margin", 0.25),
            # Feature 12: Risk constraints
            "max_drawdown_pct": mp.get("max_drawdown_pct", None),
            # Feature 14: Crowd model
            "crowd_model_enabled": mp.get("crowd_model_enabled", False),
        }

    # ── PRICE TIMELINE GENERATION ──────────────────────────────────────

    def _build_price_timeline(self, seed_override: int | None = None):
        """Pre-calculate price + volatility timelines with optional GARCH clustering."""
        self.price_timeline = {}
        self.volatility_timeline = {}
        self.halt_periods: set[int] = set()
        self.secondary_timeline = {}

        current_price = self.initial_price
        seed = seed_override if seed_override is not None else hash(self.scenario.name) % 10000
        rng = random.Random(seed)

        use_garch = self.market_params["volatility_clustering"]
        vp = self.market_params["vol_params"]
        base_vol = vp.get("base_vol", 0.02)
        persistence = vp.get("persistence", 0.85)
        reversion = vp.get("reversion", 0.10)
        shock_prob = vp.get("shock_prob", 0.01)
        shock_mult = vp.get("shock_multiplier", 3.0)

        current_vol = base_vol
        halts_enabled = self.market_params["halts_enabled"]
        halt_threshold = self.market_params["halt_threshold_pct"]
        halt_dur = self.market_params["halt_duration"]
        use_micro = self.market_params.get("microstructure_noise", False)

        # Secondary asset setup
        sec_price = self.secondary_price or (self.initial_price * 0.5 if self.secondary_asset else None)
        sec_rng = random.Random(seed + 7777) if self.secondary_asset else None
        rho = self.correlation

        halt_until = -1  # when current halt expires

        for t in range(self.time_limit + 1):
            # During halt: freeze price
            if t <= halt_until:
                self.halt_periods.add(t)
                self.price_timeline[t] = self.price_timeline.get(t - 1, current_price)
                self.volatility_timeline[t] = current_vol
                if self.secondary_asset and sec_price is not None:
                    self.secondary_timeline[t] = self.secondary_timeline.get(t - 1, sec_price)
                continue

            if use_garch:
                # GARCH-like volatility process
                if rng.random() < shock_prob:
                    current_vol *= shock_mult
                vol_noise = rng.gauss(0, base_vol * 0.1)
                current_vol = persistence * current_vol + reversion * base_vol + abs(vol_noise)
                current_vol = max(base_vol * 0.2, min(current_vol, base_vol * 10))

                noise = rng.gauss(0, current_vol) * current_price

                # Feature 10: microstructure — occasional extra jumps
                if use_micro and rng.random() < 0.05:
                    noise += rng.gauss(0, current_vol * 2) * current_price
            else:
                # Legacy path: exact old behavior
                noise = rng.uniform(-0.02, 0.02) * current_price
                current_vol = 0.02

            current_price += noise

            # Apply event effects
            for event in self.events:
                if event.get("time") == t and event.get("type") == "price":
                    change = event.get("change", 0)
                    current_price = current_price * (1 + change)

            current_price = max(0.01, current_price)

            # Feature 6: Check for halt trigger
            if halts_enabled and t > 0:
                prev = self.price_timeline.get(t - 1, self.initial_price)
                move_pct = abs(current_price - prev) / prev
                if move_pct > halt_threshold:
                    halt_until = t + halt_dur
                    self.halt_periods.add(t)

            self.price_timeline[t] = current_price
            self.volatility_timeline[t] = current_vol

            # Feature 9: Correlated secondary asset
            if self.secondary_asset and sec_price is not None and sec_rng is not None:
                shared_z = rng.gauss(0, 1)
                indep_z = sec_rng.gauss(0, 1)
                sec_z = rho * shared_z + math.sqrt(1 - rho ** 2) * indep_z
                sec_noise = sec_z * current_vol * sec_price
                sec_price += sec_noise
                # Cross-news impact (secondary gets 50% of primary event effects)
                for event in self.events:
                    if event.get("time") == t and event.get("type") == "price":
                        change = event.get("change", 0)
                        sec_price = sec_price * (1 + change * 0.5)
                sec_price = max(0.01, sec_price)
                self.secondary_timeline[t] = sec_price

    # ── SPREAD, SLIPPAGE, LIQUIDITY ────────────────────────────────────

    def _calculate_spread_and_slippage(self, time_elapsed: int, order_size: float) -> tuple[float, float]:
        """Return (spread_pct, slippage_pct) based on volatility and order size."""
        base_spread = self.market_params["base_spread_pct"]
        if base_spread == 0:
            return 0.0, 0.0

        # Use pre-computed volatility
        vol = self.volatility_timeline.get(time_elapsed, 0.02)

        # Spread widens with volatility (1x at low vol, up to 5x at high vol)
        vol_multiplier = 1.0 + min(4.0, vol / 0.02 * 2.0)
        spread = base_spread * vol_multiplier

        # Slippage proportional to order_size / available volume
        volume = self.market_params["volume_per_tick"]
        if volume < float("inf") and order_size > 0:
            fill_ratio = order_size / volume
            slippage = spread * 0.5 * min(fill_ratio, 1.0)
        else:
            slippage = 0.0

        return spread, slippage

    # ── CROWD MODEL ────────────────────────────────────────────────────

    def _simulate_crowd(self, time_elapsed: int) -> dict:
        """Simulate behavioral crowd sentiment at a given time."""
        if not self.market_params["crowd_model_enabled"] or time_elapsed < 5:
            return {"crowd_buying_pct": 50, "crowd_action": "mixed", "pressure": "none"}

        rng = random.Random(hash(self.scenario.name) % 10000 + time_elapsed)

        price_now = self.price_timeline.get(time_elapsed, self.initial_price)
        price_ago = self.price_timeline.get(max(0, time_elapsed - 10), self.initial_price)
        price_change = (price_now - price_ago) / price_ago

        vol = self.volatility_timeline.get(time_elapsed, 0.02)
        high_vol = vol > 0.04

        # Momentum traders (30%): follow trend
        momentum_buy = 1 if price_change > 0.01 else (0 if price_change < -0.01 else 0.5)
        # Panic sellers (20%): sell on drops + high vol
        panic_sell = 1 if (price_change < -0.03 and high_vol) else 0
        # FOMO buyers (20%): buy after big moves up
        fomo_buy = 1 if price_change > 0.05 else 0
        # Contrarians (15%): buy dips, sell rallies
        contrarian_buy = 1 if price_change < -0.03 else (0 if price_change > 0.03 else 0.5)
        # Random (15%)
        rand_buy = rng.random()

        buying_pct = (
            0.30 * momentum_buy +
            0.20 * (1 - panic_sell) +
            0.20 * fomo_buy +
            0.15 * contrarian_buy +
            0.15 * rand_buy
        ) * 100

        buying_pct = max(5, min(95, buying_pct + rng.gauss(0, 5)))

        if buying_pct > 65:
            action = "mostly buying"
            pressure = "bullish"
        elif buying_pct < 35:
            action = "mostly selling"
            pressure = "bearish"
        else:
            action = "mixed"
            pressure = "none"

        return {
            "crowd_buying_pct": round(buying_pct),
            "crowd_action": action,
            "pressure": pressure,
        }

    # ── MACRO INDICATORS ───────────────────────────────────────────────

    def _generate_macro_indicators(self, time_elapsed: int) -> list[dict]:
        """Generate macro economic indicators at a given time."""
        rng = random.Random(hash(self.scenario.name) % 10000 + time_elapsed // 30)

        price_now = self.price_timeline.get(time_elapsed, self.initial_price)
        price_start = self.price_timeline.get(0, self.initial_price)
        overall_trend = (price_now - price_start) / price_start

        # Interest rate direction (shifts every ~60s)
        rate_phase = time_elapsed // 60
        rate_rng = random.Random(hash(self.scenario.name) + rate_phase)
        rate_direction = rate_rng.choice(["rising", "stable", "falling"])

        # Market breadth
        breadth_pct = 50 + overall_trend * 200 + rng.gauss(0, 10)
        breadth_pct = max(15, min(85, breadth_pct))
        breadth = "advancing" if breadth_pct > 60 else ("declining" if breadth_pct < 40 else "mixed")

        # Volatility index
        vol = self.volatility_timeline.get(time_elapsed, 0.02)
        vix = vol * 1000
        vix_label = "elevated" if vix > 30 else ("normal" if vix > 15 else "low")

        return [
            {"name": "Interest Rates", "value": rate_direction, "icon": "percent"},
            {"name": "Market Breadth", "value": breadth, "detail": f"{breadth_pct:.0f}% advancing", "icon": "bar-chart"},
            {"name": "Volatility Index", "value": vix_label, "detail": f"VIX: {vix:.1f}", "icon": "activity"},
        ]

    # ── STATE QUERIES ──────────────────────────────────────────────────

    def get_initial_state(self) -> dict[str, Any]:
        """Get the initial simulation state."""
        initial_holdings = self.initial_data.get("holdings", {})
        holdings_value = initial_holdings.get(self.asset, 0) * self.initial_price
        return {
            "current_price": self.initial_price,
            "price_history": [self.initial_price],
            "historical_prices": self.initial_data.get("price_history", []),
            "portfolio": {
                "cash": self.initial_balance,
                "holdings": initial_holdings,
                "total_value": self.initial_balance + holdings_value,
                "holdings_value": holdings_value,
                "cumulative_fees": 0.0,
                "pending_orders": [],
                "peak_value": self.initial_balance + holdings_value,
            },
            "available_info": {
                "news": self._get_initial_news(),
                "social": self._get_initial_social(),
                "market_sentiment": self.initial_data.get("market_sentiment", "neutral"),
            },
            "market_conditions": self._get_market_conditions(0),
        }

    def _get_initial_news(self) -> list[dict]:
        return self.initial_data.get("news_headlines", [
            {"time": 0, "content": "Markets open normally today", "type": "neutral"}
        ])

    def _get_initial_social(self) -> list[dict]:
        return self.initial_data.get("social_signals", [
            {"time": 0, "content": "Watching the market closely today", "sentiment": "neutral"}
        ])

    def _score_credibility(self, item: dict) -> float:
        """Heuristic credibility score for a news/social item."""
        if item.get("credibility_override") is not None:
            return item["credibility_override"]

        content = (item.get("content", "") or "").lower()
        if any(w in content for w in ["sec filing", "quarterly", "report", "official", "ceo", "announces", "confirms"]):
            return 0.9
        if any(w in content for w in ["analyst", "upgrade", "downgrade", "earnings"]):
            return 0.75
        if any(w in content for w in ["rumor", "my cousin", "trust me", "guaranteed", "to the moon", "🚀", "moon", "heard something"]):
            return 0.2
        if any(w in content for w in ["everyone", "don't miss", "fomo", "easiest money"]):
            return 0.3
        if item.get("sentiment") or item.get("type") == "social":
            return 0.4
        return 0.6

    def _get_market_conditions(self, time_elapsed: int) -> dict:
        """Build market conditions snapshot at a given time."""
        price = self.price_timeline.get(time_elapsed, self.initial_price)
        spread, _ = self._calculate_spread_and_slippage(time_elapsed, 0)
        vol = self.volatility_timeline.get(time_elapsed, 0.02)
        halted = time_elapsed in self.halt_periods
        crowd = self._simulate_crowd(time_elapsed)

        return {
            "bid": round(price * (1 - spread / 2), 4) if spread > 0 else round(price, 4),
            "ask": round(price * (1 + spread / 2), 4) if spread > 0 else round(price, 4),
            "spread_pct": round(spread * 100, 3),
            "volatility": round(vol, 5),
            "vol_regime": "high" if vol > 0.04 else ("low" if vol < 0.01 else "normal"),
            "halted": halted,
            **crowd,
            "macro": self._generate_macro_indicators(time_elapsed),
        }

    def get_state_at_time(self, time_elapsed: int, current_portfolio: dict) -> dict[str, Any]:
        """Get simulation state at a specific time."""
        current_price = self.price_timeline.get(time_elapsed, self.initial_price)

        # Build price history (~60 points max)
        step = max(1, time_elapsed // 60)
        price_history = [
            self.price_timeline.get(t, self.initial_price)
            for t in range(0, time_elapsed + 1, step)
        ]

        # Feature 4: News latency
        news_latency = self.market_params["news_latency_enabled"]
        occurred_events = []
        for e in self.events:
            event_time = e.get("time", 0)
            latency = e.get("latency", 0) if news_latency else 0
            visible_time = event_time + latency
            if visible_time <= time_elapsed:
                event_copy = dict(e)
                if e.get("unverified"):
                    event_copy["tag"] = "unverified"
                if e.get("false_rumor"):
                    event_copy["tag"] = "unverified"
                occurred_events.append(event_copy)

        news = [e for e in occurred_events if e.get("type") == "news"]
        social = [e for e in occurred_events if e.get("type") == "social"]

        # Calculate portfolio value
        holdings = current_portfolio.get("holdings", {})
        holdings_value = holdings.get(self.asset, 0) * current_price
        if self.secondary_asset:
            sec_price = self.secondary_timeline.get(time_elapsed, 0)
            holdings_value += holdings.get(self.secondary_asset, 0) * sec_price
        cash = current_portfolio.get("cash", self.initial_balance)
        total_value = cash + holdings_value

        peak_value = max(current_portfolio.get("peak_value", total_value), total_value)

        # Annotate info with credibility
        initial_news = [{**n, "credibility": self._score_credibility(n)} for n in self._get_initial_news()]
        event_news = [
            {"time": e["time"], "content": e["content"], "type": "event",
             "credibility": self._score_credibility(e), "tag": e.get("tag")}
            for e in news
        ]
        initial_social = [{**s, "credibility": self._score_credibility(s)} for s in self._get_initial_social()]
        event_social = [
            {"time": e["time"], "content": e["content"], "sentiment": e.get("sentiment", "event"),
             "credibility": self._score_credibility(e), "tag": e.get("tag")}
            for e in social
        ]

        # Check pending orders
        portfolio_updates = self._check_pending_orders(time_elapsed, current_price, current_portfolio)

        market_conditions = self._get_market_conditions(time_elapsed)

        # Drawdown
        drawdown_pct = (peak_value - total_value) / peak_value * 100 if peak_value > 0 else 0

        # Margin status
        margin_status = "ok"
        if self.market_params["margin_enabled"]:
            margin_used = current_portfolio.get("margin_used", 0)
            if margin_used > 0:
                equity_ratio = total_value / (total_value + margin_used)
                if equity_ratio < self.market_params["maintenance_margin"]:
                    margin_status = "margin_call"
                elif equity_ratio < self.market_params["maintenance_margin"] * 1.5:
                    margin_status = "warning"

        market_conditions["margin_status"] = margin_status
        market_conditions["drawdown_pct"] = round(drawdown_pct, 2)

        # Secondary asset
        secondary_info = {}
        if self.secondary_asset:
            sec_price = self.secondary_timeline.get(time_elapsed, 0)
            secondary_info = {
                "secondary_asset": self.secondary_asset,
                "secondary_price": round(sec_price, 2),
                "secondary_history": [
                    self.secondary_timeline.get(t, sec_price)
                    for t in range(0, time_elapsed + 1, step)
                ],
            }

        return {
            "current_price": current_price,
            "price_history": price_history,
            "historical_prices": self.initial_data.get("price_history", []),
            "portfolio": {
                **current_portfolio,
                **portfolio_updates,
                "total_value": total_value,
                "holdings_value": holdings_value,
                "peak_value": peak_value,
            },
            "available_info": {
                "news": initial_news + event_news,
                "social": initial_social + event_social,
                "market_sentiment": self._calculate_sentiment(time_elapsed),
            },
            "market_conditions": market_conditions,
            "recent_events": [
                e for e in occurred_events if e.get("time", 0) > time_elapsed - 30
            ],
            **secondary_info,
        }

    def _calculate_sentiment(self, time_elapsed: int) -> str:
        if time_elapsed < 10:
            return self.initial_data.get("market_sentiment", "neutral")
        recent_prices = [
            self.price_timeline.get(t, self.initial_price)
            for t in range(max(0, time_elapsed - 30), time_elapsed + 1)
        ]
        if len(recent_prices) < 2:
            return "neutral"
        change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        if change > 0.05:
            return "bullish"
        elif change < -0.05:
            return "bearish"
        return "neutral"

    # ── PENDING ORDERS (LIMIT / STOP) ──────────────────────────────────

    def _check_pending_orders(self, time_elapsed: int, current_price: float,
                              current_portfolio: dict) -> dict:
        """Check and execute pending limit/stop orders."""
        pending = current_portfolio.get("pending_orders", [])
        if not pending:
            return {}

        executed = []
        remaining = []
        cash = current_portfolio.get("cash", 0)
        holdings = current_portfolio.get("holdings", {}).copy()

        for order in pending:
            triggered = False
            if order["order_type"] == "limit":
                if order["side"] == "buy" and current_price <= order["price"]:
                    triggered = True
                elif order["side"] == "sell" and current_price >= order["price"]:
                    triggered = True
            elif order["order_type"] == "stop":
                if order["side"] == "sell" and current_price <= order["price"]:
                    triggered = True
                elif order["side"] == "buy" and current_price >= order["price"]:
                    triggered = True

            if triggered:
                exec_price = order["price"]
                amount = order["amount"]
                if order["side"] == "buy":
                    cost = amount * exec_price
                    if cost <= cash:
                        cash -= cost
                        asset = order.get("asset", self.asset)
                        holdings[asset] = holdings.get(asset, 0) + amount
                        executed.append({**order, "executed_at": time_elapsed, "exec_price": exec_price})
                elif order["side"] == "sell":
                    asset = order.get("asset", self.asset)
                    held = holdings.get(asset, 0)
                    sell_amt = min(amount, held)
                    if sell_amt > 0:
                        cash += sell_amt * exec_price
                        holdings[asset] = held - sell_amt
                        executed.append({**order, "executed_at": time_elapsed, "exec_price": exec_price})
            else:
                remaining.append(order)

        if executed:
            return {
                "cash": cash,
                "holdings": holdings,
                "pending_orders": remaining,
                "executed_orders": executed,
            }
        return {}

    # ── DECISION PROCESSING ────────────────────────────────────────────

    def process_decision(
        self,
        decision: DecisionCreate,
        current_state: dict,
        current_portfolio: dict,
        time_elapsed: int = 0,
    ) -> dict:
        """Process a decision with full market realism and return updated portfolio."""
        portfolio = current_portfolio.copy()
        holdings = portfolio.get("holdings", {}).copy()
        cash = portfolio.get("cash", self.initial_balance)
        cumulative_fees = portfolio.get("cumulative_fees", 0.0)
        pending_orders = list(portfolio.get("pending_orders", []))
        peak_value = portfolio.get("peak_value", self.initial_balance)

        mid_price = current_state["current_price"]
        asset = decision.asset or self.asset

        # Feature 6: Halt check
        if time_elapsed in self.halt_periods and decision.decision_type in ("buy", "sell"):
            holdings_value = sum(
                holdings.get(a, 0) * self.price_timeline.get(time_elapsed, mid_price)
                for a in [self.asset] + ([self.secondary_asset] if self.secondary_asset else [])
            )
            return {
                "cash": cash,
                "holdings": holdings,
                "total_value": cash + holdings_value,
                "holdings_value": holdings_value,
                "cumulative_fees": cumulative_fees,
                "pending_orders": pending_orders,
                "peak_value": peak_value,
                "margin_used": portfolio.get("margin_used", 0),
                "last_execution": {
                    "halted": True,
                    "halt_reason": "Circuit breaker active",
                    "fees_paid": 0,
                    "execution_price": mid_price,
                    "requested_amount": decision.amount or 0,
                    "filled_amount": 0,
                    "slippage": 0,
                    "spread": 0,
                    "partial_fill": False,
                },
            }

        # Feature 7: Order types (limit/stop) — store as pending
        order_type = getattr(decision, "order_type", "market") or "market"
        if self.market_params["order_types_enabled"] and order_type in ("limit", "stop"):
            limit_price = getattr(decision, "limit_price", None)
            stop_price = getattr(decision, "stop_price", None)
            trigger_price = limit_price if order_type == "limit" else stop_price

            if trigger_price and decision.amount:
                pending_orders.append({
                    "order_type": order_type,
                    "side": decision.decision_type,
                    "amount": decision.amount,
                    "price": trigger_price,
                    "asset": asset,
                    "placed_at": time_elapsed,
                })

            holdings_value = holdings.get(self.asset, 0) * mid_price
            return {
                "cash": cash,
                "holdings": holdings,
                "total_value": cash + holdings_value,
                "holdings_value": holdings_value,
                "cumulative_fees": cumulative_fees,
                "pending_orders": pending_orders,
                "peak_value": peak_value,
                "margin_used": portfolio.get("margin_used", 0),
                "last_execution": {
                    "halted": False,
                    "fees_paid": 0,
                    "execution_price": 0,
                    "requested_amount": decision.amount or 0,
                    "filled_amount": 0,
                    "slippage": 0,
                    "spread": 0,
                    "partial_fill": False,
                    "order_placed": True,
                    "order_type": order_type,
                    "trigger_price": trigger_price,
                },
            }

        # Calculate spread and slippage
        order_size = decision.amount or 0
        spread, slippage = self._calculate_spread_and_slippage(time_elapsed, order_size)

        # Feature 8: Time pressure trade-off
        time_spent = decision.time_spent_seconds or 0
        if self.market_params["time_pressure_fills"] and time_spent < 5:
            slippage += 0.005

        # Feature 3: Liquidity constraints
        max_trade = self.market_params["max_trade_per_tick"]
        requested_amount = decision.amount or 0
        filled_amount = min(requested_amount, max_trade)
        partial_fill = filled_amount < requested_amount and requested_amount > 0

        # Price impact for large orders
        volume = self.market_params["volume_per_tick"]
        if volume < float("inf") and filled_amount > 0:
            impact = (filled_amount / volume) * 0.01
            slippage += impact

        fees = 0.0
        execution_price = mid_price

        if decision.decision_type == "buy" and filled_amount > 0:
            execution_price = mid_price * (1 + spread / 2 + slippage)
            cost = filled_amount * execution_price

            buying_power = cash
            if self.market_params["margin_enabled"]:
                buying_power = cash * self.market_params["max_leverage"]

            if cost <= buying_power:
                fees = self.market_params["fixed_fee"] + cost * self.market_params["pct_fee"]
                actual_cost = cost + fees
                if actual_cost <= buying_power:
                    margin_used = max(0, actual_cost - cash)
                    cash -= min(actual_cost, cash)
                    holdings[asset] = holdings.get(asset, 0) + filled_amount
                    cumulative_fees += fees
                    portfolio["margin_used"] = portfolio.get("margin_used", 0) + margin_used
                else:
                    filled_amount = 0
                    fees = 0
                    execution_price = mid_price
            else:
                filled_amount = 0
                fees = 0
                execution_price = mid_price

        elif decision.decision_type == "sell" and filled_amount > 0:
            current_holding = holdings.get(asset, 0)
            sell_amount = min(filled_amount, current_holding)
            if sell_amount > 0:
                execution_price = mid_price * (1 - spread / 2 - slippage)
                revenue = sell_amount * execution_price
                fees = self.market_params["fixed_fee"] + revenue * self.market_params["pct_fee"]
                cash += revenue - fees
                holdings[asset] = current_holding - sell_amount
                cumulative_fees += fees
                filled_amount = sell_amount
            else:
                filled_amount = 0
                fees = 0
                execution_price = mid_price
        else:
            filled_amount = 0

        holdings_value = holdings.get(self.asset, 0) * mid_price
        if self.secondary_asset:
            sec_price = self.secondary_timeline.get(time_elapsed, 0)
            holdings_value += holdings.get(self.secondary_asset, 0) * sec_price
        total_value = cash + holdings_value
        peak_value = max(peak_value, total_value)

        # Feature 12: Auto de-risk on excessive drawdown
        max_dd = self.market_params["max_drawdown_pct"]
        drawdown_triggered = False
        if max_dd and peak_value > 0:
            drawdown = (peak_value - total_value) / peak_value
            if drawdown > max_dd and holdings.get(self.asset, 0) > 0:
                force_sell = holdings[self.asset] * 0.5
                force_revenue = force_sell * mid_price * (1 - spread / 2)
                force_fees = self.market_params["fixed_fee"] + force_revenue * self.market_params["pct_fee"]
                cash += force_revenue - force_fees
                holdings[self.asset] -= force_sell
                cumulative_fees += force_fees
                drawdown_triggered = True
                holdings_value = holdings.get(self.asset, 0) * mid_price
                total_value = cash + holdings_value

        return {
            "cash": cash,
            "holdings": holdings,
            "total_value": cash + holdings_value,
            "holdings_value": holdings_value,
            "cumulative_fees": round(cumulative_fees, 4),
            "pending_orders": pending_orders,
            "peak_value": peak_value,
            "margin_used": portfolio.get("margin_used", 0),
            "last_execution": {
                "halted": False,
                "fees_paid": round(fees, 4),
                "execution_price": round(execution_price, 4),
                "requested_amount": requested_amount,
                "filled_amount": round(filled_amount, 4),
                "slippage": round(slippage * 100, 3),
                "spread": round(spread * 100, 3),
                "partial_fill": partial_fill,
                "drawdown_triggered": drawdown_triggered,
            },
        }

    # ── UTILITY METHODS ────────────────────────────────────────────────

    def get_events_between(self, start_time: int, end_time: int) -> list[dict]:
        return [
            e for e in self.events
            if start_time < e.get("time", 0) <= end_time
        ]

    def get_final_state(self, current_portfolio: dict) -> dict:
        final_price = self.price_timeline.get(self.time_limit, self.initial_price)
        holdings = current_portfolio.get("holdings", {})
        holdings_value = holdings.get(self.asset, 0) * final_price
        if self.secondary_asset:
            sec_price = self.secondary_timeline.get(self.time_limit, 0)
            holdings_value += holdings.get(self.secondary_asset, 0) * sec_price
        cash = current_portfolio.get("cash", 0)

        return {
            "final_price": final_price,
            "cash": cash,
            "holdings": holdings,
            "holdings_value": holdings_value,
            "total_value": cash + holdings_value,
            "cumulative_fees": current_portfolio.get("cumulative_fees", 0),
        }

    # ── PROCESS QUALITY SCORING ────────────────────────────────────────

    def calculate_process_quality(self, decisions: list, scenario: Scenario,
                                  portfolio: dict | None = None) -> float:
        """Calculate process quality score with market realism factors."""
        if not decisions:
            return 50.0

        score = 50.0

        # Factor 1: Decision timing
        first_decision_time = decisions[0].simulation_time if decisions else 0
        if first_decision_time >= 15:
            score += 10
        elif first_decision_time < 5:
            score -= 15

        # Factor 2: Information usage
        total_info_viewed = sum(len(d.info_viewed or []) for d in decisions)
        if total_info_viewed >= len(decisions) * 2:
            score += 15
        elif total_info_viewed == 0:
            score -= 20

        # Factor 3: Confidence calibration
        high_confidence = [d for d in decisions if (d.confidence_level or 3) >= 4]
        if len(high_confidence) < len(decisions) * 0.8:
            score += 5

        # Factor 4: Decision diversity
        if len(set(d.decision_type for d in decisions)) >= 2:
            score += 10

        # Factor 5: Deliberation time
        avg_time = sum(d.time_spent_seconds or 0 for d in decisions) / max(len(decisions), 1)
        if avg_time >= 5:
            score += 10
        elif avg_time < 2:
            score -= 10

        # ── MARKET REALISM FACTORS ─────────────────────────────────

        trade_decisions = [d for d in decisions if d.decision_type in ("buy", "sell")]

        # Factor 6: Spread awareness
        if self.market_params.get("base_spread_pct", 0) > 0 and trade_decisions:
            high_spread_trades = sum(
                1 for d in trade_decisions
                if self.volatility_timeline.get(d.simulation_time, 0.02) > 0.04
            )
            score -= (high_spread_trades / len(trade_decisions)) * 15

        # Factor 7: Halt respect
        if self.halt_periods:
            halt_attempts = sum(
                1 for d in trade_decisions if d.simulation_time in self.halt_periods
            )
            if halt_attempts == 0:
                score += 5
            else:
                score -= halt_attempts * 5

        # Factor 8: Fee efficiency
        if portfolio:
            cumulative_fees = portfolio.get("cumulative_fees", 0)
            if cumulative_fees > 0:
                fee_ratio = cumulative_fees / self.initial_balance
                if fee_ratio > 0.02:
                    score -= 10
                elif fee_ratio < 0.005:
                    score += 5

        # Factor 9: Credibility filtering
        for d in trade_decisions:
            recent_unreliable = [
                e for e in self.events
                if e.get("time", 0) <= d.simulation_time
                and e.get("time", 0) > d.simulation_time - 15
                and (e.get("false_rumor") or e.get("unverified"))
            ]
            if recent_unreliable:
                score -= 3

        # Factor 10: Crowd independence
        if self.market_params["crowd_model_enabled"] and trade_decisions:
            following = sum(
                1 for d in trade_decisions
                if (d.decision_type == "buy" and self._simulate_crowd(d.simulation_time)["crowd_buying_pct"] > 60)
                or (d.decision_type == "sell" and self._simulate_crowd(d.simulation_time)["crowd_buying_pct"] < 40)
            )
            if following < len(trade_decisions) * 0.5:
                score += 10

        return max(0, min(100, score))

    # ── CONFIDENCE CALIBRATION ─────────────────────────────────────────

    def evaluate_decision_outcome(self, decision, next_price: float) -> dict:
        price_at = decision.price_at_decision or self.initial_price
        price_change_pct = ((next_price - price_at) / price_at) * 100

        if decision.decision_type == "buy":
            favorable = price_change_pct > 1.0
        elif decision.decision_type == "sell":
            favorable = price_change_pct < -1.0
        else:
            favorable = abs(price_change_pct) < 2.0

        confidence = decision.confidence_level or 3
        high_confidence = confidence >= 4
        low_confidence = confidence <= 2

        if high_confidence and favorable:
            calibration = "well_calibrated"
        elif high_confidence and not favorable:
            calibration = "overconfident"
        elif low_confidence and favorable:
            calibration = "underconfident"
        elif low_confidence and not favorable:
            calibration = "well_calibrated"
        else:
            calibration = "well_calibrated"

        return {
            "decision_index": None,
            "decision_type": decision.decision_type,
            "confidence": confidence,
            "price_at_decision": price_at,
            "price_after": next_price,
            "price_change_pct": round(price_change_pct, 2),
            "favorable": favorable,
            "calibration": calibration,
        }

    def get_calibration_report(self, decisions: list) -> dict:
        details = []
        overconfident = underconfident = well_calibrated = 0

        for i, d in enumerate(decisions):
            next_time = min(d.simulation_time + 30, self.time_limit)
            next_price = self.price_timeline.get(next_time, self.initial_price)
            result = self.evaluate_decision_outcome(d, next_price)
            result["decision_index"] = i
            if result["calibration"] == "overconfident":
                overconfident += 1
            elif result["calibration"] == "underconfident":
                underconfident += 1
            else:
                well_calibrated += 1
            details.append(result)

        total = max(len(decisions), 1)
        return {
            "calibration_score": round((well_calibrated / total) * 100),
            "overconfident_count": overconfident,
            "underconfident_count": underconfident,
            "well_calibrated_count": well_calibrated,
            "total_decisions": len(decisions),
            "details": details,
        }

    # ── MONTE CARLO ────────────────────────────────────────────────────

    def monte_carlo_outcomes(self, decisions: list, n: int = 100) -> dict:
        """Replay decisions against n different price timelines with market realism."""
        base_seed = hash(self.scenario.name) % 10000
        outcomes = []

        use_garch = self.market_params["volatility_clustering"]
        vp = self.market_params["vol_params"]
        base_vol = vp.get("base_vol", 0.02)
        persistence = vp.get("persistence", 0.85)
        reversion = vp.get("reversion", 0.10)
        shock_prob = vp.get("shock_prob", 0.01)
        shock_mult = vp.get("shock_multiplier", 3.0)

        fixed_fee = self.market_params["fixed_fee"]
        pct_fee = self.market_params["pct_fee"]
        base_spread = self.market_params["base_spread_pct"]

        for trial in range(n):
            alt_timeline = {}
            current_price = self.initial_price
            rng = random.Random(base_seed + trial + 1)
            current_vol = base_vol

            for t in range(self.time_limit + 1):
                if use_garch:
                    if rng.random() < shock_prob:
                        current_vol *= shock_mult
                    vol_noise = rng.gauss(0, base_vol * 0.1)
                    current_vol = persistence * current_vol + reversion * base_vol + abs(vol_noise)
                    current_vol = max(base_vol * 0.2, min(current_vol, base_vol * 10))
                    noise = rng.gauss(0, current_vol) * current_price
                else:
                    noise = rng.uniform(-0.02, 0.02) * current_price

                current_price += noise
                for event in self.events:
                    if event.get("time") == t and event.get("type") == "price":
                        current_price = current_price * (1 + event.get("change", 0))
                alt_timeline[t] = max(0.01, current_price)

            # Replay decisions with fees/spread
            portfolio_cash = self.initial_balance
            portfolio_holdings = dict(self.initial_data.get("holdings", {}))

            for d in decisions:
                price = alt_timeline.get(d.simulation_time, self.initial_price)
                spread_adj = base_spread * 0.5

                if d.decision_type == "buy" and d.amount:
                    exec_price = price * (1 + spread_adj)
                    cost = d.amount * exec_price
                    fees = fixed_fee + cost * pct_fee
                    if cost + fees <= portfolio_cash:
                        portfolio_cash -= cost + fees
                        portfolio_holdings[self.asset] = portfolio_holdings.get(self.asset, 0) + d.amount
                elif d.decision_type == "sell" and d.amount:
                    held = portfolio_holdings.get(self.asset, 0)
                    sell_amt = min(d.amount, held)
                    if sell_amt > 0:
                        exec_price = price * (1 - spread_adj)
                        revenue = sell_amt * exec_price
                        fees = fixed_fee + revenue * pct_fee
                        portfolio_cash += revenue - fees
                        portfolio_holdings[self.asset] = held - sell_amt

            final_price = alt_timeline.get(self.time_limit, self.initial_price)
            hv = portfolio_holdings.get(self.asset, 0) * final_price
            outcomes.append(round(portfolio_cash + hv - self.initial_balance, 2))

        outcomes.sort()

        # Histogram
        if outcomes:
            min_val, max_val = outcomes[0], outcomes[-1]
            bucket_count = 20
            bucket_size = max((max_val - min_val) / bucket_count, 0.01)
            buckets = [
                {"range_low": round(min_val + i * bucket_size, 2),
                 "range_high": round(min_val + (i + 1) * bucket_size, 2),
                 "count": sum(1 for o in outcomes if min_val + i * bucket_size <= o < min_val + (i + 1) * bucket_size)}
                for i in range(bucket_count)
            ]
        else:
            buckets = []

        # Actual outcome
        actual_cash = self.initial_balance
        actual_holdings = dict(self.initial_data.get("holdings", {}))
        for d in decisions:
            price = self.price_timeline.get(d.simulation_time, self.initial_price)
            if d.decision_type == "buy" and d.amount:
                cost = d.amount * price * (1 + base_spread * 0.5)
                fees = fixed_fee + cost * pct_fee
                if cost + fees <= actual_cash:
                    actual_cash -= cost + fees
                    actual_holdings[self.asset] = actual_holdings.get(self.asset, 0) + d.amount
            elif d.decision_type == "sell" and d.amount:
                held = actual_holdings.get(self.asset, 0)
                sell_amt = min(d.amount, held)
                if sell_amt > 0:
                    revenue = sell_amt * price * (1 - base_spread * 0.5)
                    fees = fixed_fee + revenue * pct_fee
                    actual_cash += revenue - fees
                    actual_holdings[self.asset] = held - sell_amt

        actual_final = self.price_timeline.get(self.time_limit, self.initial_price)
        actual_hv = actual_holdings.get(self.asset, 0) * actual_final
        actual_pl = actual_cash + actual_hv - self.initial_balance

        percentile = sum(1 for o in outcomes if o <= actual_pl) / max(len(outcomes), 1) * 100

        return {
            "simulations_run": n,
            "outcomes": outcomes,
            "buckets": buckets,
            "actual_outcome": round(actual_pl, 2),
            "percentile": round(percentile, 1),
            "median_outcome": round(outcomes[len(outcomes) // 2], 2) if outcomes else 0,
            "best_outcome": round(outcomes[-1], 2) if outcomes else 0,
            "worst_outcome": round(outcomes[0], 2) if outcomes else 0,
        }
