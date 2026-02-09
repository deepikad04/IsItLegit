"""
Seed script: creates two demo users with rich, varied simulation history,
decisions, behavior profiles, and learning progress — ready for hackathon demo.

Usage:
    cd backend && python seed_demo.py

Login credentials:
    Email:    demo@isitlegit.com   Password: demo1234  (20 sims, 30-day arc)
    Email:    alex@isitlegit.com   Password: alex1234  (12 sims, beginner arc)
"""
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("DATABASE_URL", "postgresql://isitlegit:isitlegit_secret@localhost:5432/isitlegit")

from database import SessionLocal, init_db
from models.user import User
from models.scenario import Scenario
from models.simulation import Simulation
from models.decision import Decision
from models.simulation_snapshot import SimulationSnapshot
from models.behavior_profile import BehaviorProfile, LearningProgress
from routers.auth import get_password_hash

NOW = datetime.utcnow()


# ─────────────────────────────────────────────────────────────────────
# Simulation data: (scenario_index, days_ago, P&L, process_score, decisions)
# Scenarios: 0=HYPECOIN(fomo), 1=TECHGROW(patience), 2=SAFECO(loss_aversion),
#            3=SECRETCO(social_proof), 4=MEGACORP(risk_mgmt), 5=FAILCO(contrarian)
# ─────────────────────────────────────────────────────────────────────

DEMO_SIMS = [
    # === WEEK 1: Beginner mistakes, emotional trading ===
    (0, 30, -1800.00, 25, [  # HYPECOIN — pure FOMO, worst sim
        {"t": 5, "type": "buy", "amt": 7000, "conf": 5, "price": 0.52, "rationale": "Everyone's talking about it, can't miss out"},
        {"t": 18, "type": "buy", "amt": 3000, "conf": 5, "price": 0.70, "rationale": "It's pumping! Putting everything in"},
        {"t": 90, "type": "hold", "amt": None, "conf": 3, "price": 0.40, "rationale": "It has to come back, right?"},
        {"t": 140, "type": "hold", "amt": None, "conf": 2, "price": 0.32, "rationale": "I can't sell at this loss"},
        {"t": 170, "type": "sell", "amt": 10000, "conf": 1, "price": 0.28, "rationale": "Panic selling, I've lost too much"},
    ]),
    (3, 28, -900.00, 30, [  # SECRETCO — fell for insider tip
        {"t": 8, "type": "buy", "amt": 6000, "conf": 4, "price": 25.20, "rationale": "My friend says something big is coming"},
        {"t": 30, "type": "buy", "amt": 2000, "conf": 4, "price": 26.00, "rationale": "More people are confirming the rumor"},
        {"t": 100, "type": "hold", "amt": None, "conf": 3, "price": 24.50, "rationale": "The announcement should come any day"},
        {"t": 145, "type": "sell", "amt": 8000, "conf": 2, "price": 23.50, "rationale": "No announcement. It was all fake."},
    ]),
    (2, 27, -1400.00, 20, [  # SAFECO — panic sold at the bottom
        {"t": 12, "type": "hold", "amt": None, "conf": 3, "price": 48.00, "rationale": "Regulatory news is scary but maybe temporary"},
        {"t": 30, "type": "sell", "amt": 5000, "conf": 2, "price": 42.00, "rationale": "Can't take this anymore, selling half"},
        {"t": 50, "type": "sell", "amt": 5000, "conf": 1, "price": 38.00, "rationale": "Get me out! It keeps dropping!"},
        {"t": 120, "type": "hold", "amt": None, "conf": 1, "price": 47.00, "rationale": "...it recovered. I sold at the bottom."},
    ]),
    (1, 25, 200.00, 50, [  # TECHGROW — first patient sim, small win
        {"t": 25, "type": "hold", "amt": None, "conf": 3, "price": 45.50, "rationale": "Watching for now, no rush"},
        {"t": 55, "type": "buy", "amt": 3000, "conf": 3, "price": 46.50, "rationale": "Looks steady, small position"},
        {"t": 120, "type": "hold", "amt": None, "conf": 3, "price": 47.50, "rationale": "Holding, trend looks fine"},
        {"t": 170, "type": "sell", "amt": 3000, "conf": 3, "price": 48.50, "rationale": "Taking a modest profit"},
    ]),

    # === WEEK 2: Starting to learn, mixed results ===
    (0, 22, -450.00, 55, [  # HYPECOIN retry — smaller position, quicker exit
        {"t": 15, "type": "hold", "amt": None, "conf": 3, "price": 0.55, "rationale": "Not jumping in this time, watching first"},
        {"t": 45, "type": "buy", "amt": 2000, "conf": 3, "price": 0.68, "rationale": "Small test position only"},
        {"t": 95, "type": "sell", "amt": 2000, "conf": 3, "price": 0.48, "rationale": "Cut early, lesson learned from last time"},
    ]),
    (4, 20, -2500.00, 35, [  # MEGACORP — overconfidence on all-time-high
        {"t": 10, "type": "buy", "amt": 25000, "conf": 5, "price": 510.00, "rationale": "ATH means momentum, going big"},
        {"t": 40, "type": "buy", "amt": 15000, "conf": 4, "price": 530.00, "rationale": "Analyst says $600 target, adding more"},
        {"t": 80, "type": "hold", "amt": None, "conf": 3, "price": 520.00, "rationale": "Just a pullback, fundamentals are strong"},
        {"t": 130, "type": "hold", "amt": None, "conf": 2, "price": 480.00, "rationale": "Can't sell now, too far underwater"},
        {"t": 165, "type": "sell", "amt": 40000, "conf": 1, "price": 460.00, "rationale": "Congress testimony scares me, exiting"},
    ]),
    (5, 18, 800.00, 65, [  # FAILCO — first contrarian win
        {"t": 20, "type": "hold", "amt": None, "conf": 2, "price": 28.00, "rationale": "Everyone's panicking but the drop seems overdone"},
        {"t": 55, "type": "hold", "amt": None, "conf": 2, "price": 24.00, "rationale": "Still dropping, but earnings miss was already priced in"},
        {"t": 90, "type": "buy", "amt": 3000, "conf": 3, "price": 22.50, "rationale": "CEO turnaround plan sounds credible, small position"},
        {"t": 130, "type": "hold", "amt": None, "conf": 3, "price": 28.00, "rationale": "Recovering, holding for more"},
        {"t": 165, "type": "sell", "amt": 3000, "conf": 4, "price": 33.00, "rationale": "Nice bounce, taking profits while they're there"},
    ]),
    (1, 16, 650.00, 70, [  # TECHGROW — patient, systematic
        {"t": 30, "type": "buy", "amt": 4000, "conf": 4, "price": 45.50, "rationale": "Steady uptrend confirmed with earnings report"},
        {"t": 70, "type": "hold", "amt": None, "conf": 4, "price": 47.00, "rationale": "Trend intact, analyst upgrade is a plus"},
        {"t": 130, "type": "buy", "amt": 1500, "conf": 3, "price": 48.50, "rationale": "Adding on strength"},
        {"t": 170, "type": "sell", "amt": 2500, "conf": 4, "price": 50.00, "rationale": "Partial exit at round number target"},
    ]),

    # === WEEK 3: Real improvement, hitting stride ===
    (3, 13, 50.00, 72, [  # SECRETCO — resisted social proof
        {"t": 15, "type": "hold", "amt": None, "conf": 3, "price": 25.10, "rationale": "Rumors aren't evidence. Waiting for real news"},
        {"t": 50, "type": "hold", "amt": None, "conf": 3, "price": 26.50, "rationale": "Price rising on hype alone, still no fundamentals"},
        {"t": 85, "type": "buy", "amt": 1500, "conf": 3, "price": 27.00, "rationale": "If it holds here after rumor hype, maybe real support"},
        {"t": 135, "type": "sell", "amt": 1500, "conf": 3, "price": 25.20, "rationale": "No announcement came, exiting quickly at small loss"},
    ]),
    (2, 11, 400.00, 75, [  # SAFECO — held through panic, bought the dip
        {"t": 10, "type": "hold", "amt": None, "conf": 3, "price": 46.00, "rationale": "Bad news but let's see if it's actually serious"},
        {"t": 35, "type": "hold", "amt": None, "conf": 3, "price": 40.00, "rationale": "Painful drop but analyst says investigation is routine"},
        {"t": 60, "type": "buy", "amt": 3000, "conf": 3, "price": 38.00, "rationale": "Everyone panicking, but fundamentals unchanged. Buying dip"},
        {"t": 100, "type": "hold", "amt": None, "conf": 4, "price": 46.00, "rationale": "Recovery happening as expected"},
        {"t": 140, "type": "sell", "amt": 3000, "conf": 4, "price": 49.00, "rationale": "Sold at a profit, didn't panic this time"},
    ]),
    (4, 9, -800.00, 60, [  # MEGACORP — better risk management
        {"t": 20, "type": "buy", "amt": 10000, "conf": 3, "price": 505.00, "rationale": "Bullish but keeping position size reasonable"},
        {"t": 60, "type": "hold", "amt": None, "conf": 3, "price": 525.00, "rationale": "Up nicely, but won't add more at this level"},
        {"t": 100, "type": "hold", "amt": None, "conf": 3, "price": 510.00, "rationale": "Small pullback, within normal range"},
        {"t": 130, "type": "sell", "amt": 10000, "conf": 3, "price": 480.00, "rationale": "Congress hearing risk too high, cutting before worse news"},
    ]),
    (0, 7, 350.00, 78, [  # HYPECOIN — mastered the FOMO trap
        {"t": 20, "type": "hold", "amt": None, "conf": 3, "price": 0.55, "rationale": "Hype cycle starting again. I know this pattern now"},
        {"t": 55, "type": "hold", "amt": None, "conf": 4, "price": 0.72, "rationale": "Resisting FOMO. No real catalysts, just social buzz"},
        {"t": 100, "type": "buy", "amt": 1500, "conf": 3, "price": 0.45, "rationale": "Buying after the crash, not before. Contrarian entry"},
        {"t": 150, "type": "sell", "amt": 1500, "conf": 4, "price": 0.58, "rationale": "Taking profit on the bounce, disciplined exit"},
    ]),

    # === WEEK 4: Consistent, diverse, confident ===
    (5, 5, 1200.00, 82, [  # FAILCO — strong contrarian conviction
        {"t": 15, "type": "hold", "amt": None, "conf": 3, "price": 27.50, "rationale": "Blood in the streets. Let's see if it's justified"},
        {"t": 45, "type": "hold", "amt": None, "conf": 3, "price": 23.00, "rationale": "Still falling but below book value now"},
        {"t": 70, "type": "buy", "amt": 4000, "conf": 4, "price": 21.00, "rationale": "Oversold on emotion, turnaround plan + hedge fund buying"},
        {"t": 110, "type": "buy", "amt": 2000, "conf": 4, "price": 26.00, "rationale": "Scaling in, recovery confirmed"},
        {"t": 155, "type": "sell", "amt": 6000, "conf": 4, "price": 35.00, "rationale": "Taking profits at 66% gain, excellent risk/reward"},
    ]),
    (1, 4, 1100.00, 88, [  # TECHGROW — best process yet
        {"t": 25, "type": "hold", "amt": None, "conf": 3, "price": 45.50, "rationale": "Analyzing before acting, reviewing quarterly numbers"},
        {"t": 50, "type": "buy", "amt": 5000, "conf": 4, "price": 46.00, "rationale": "Clear uptrend, low volatility, meets all my criteria"},
        {"t": 95, "type": "buy", "amt": 2000, "conf": 4, "price": 47.50, "rationale": "Scaling in on confirmed trend, still room to run"},
        {"t": 140, "type": "sell", "amt": 3000, "conf": 4, "price": 50.50, "rationale": "Partial exit at +9%. Letting winners run but locking some in"},
        {"t": 170, "type": "sell", "amt": 4000, "conf": 4, "price": 51.00, "rationale": "Full exit, disciplined profit-taking"},
    ]),
    (3, 3, 300.00, 80, [  # SECRETCO — smart social proof navigation
        {"t": 12, "type": "hold", "amt": None, "conf": 3, "price": 25.05, "rationale": "Tip came in. Going to verify before acting"},
        {"t": 40, "type": "hold", "amt": None, "conf": 3, "price": 25.50, "rationale": "Price moving on rumors only. Need real evidence"},
        {"t": 75, "type": "buy", "amt": 2000, "conf": 3, "price": 26.80, "rationale": "Official news report gives it some credibility. Small bet"},
        {"t": 115, "type": "sell", "amt": 2000, "conf": 4, "price": 28.50, "rationale": "Taking 6% profit before the inevitable 'buy the rumor sell the news'"},
    ]),
    (2, 2, 550.00, 85, [  # SAFECO — cool-headed in panic
        {"t": 8, "type": "hold", "amt": None, "conf": 4, "price": 47.00, "rationale": "Regulatory investigation. Checking analyst reactions first"},
        {"t": 30, "type": "hold", "amt": None, "conf": 3, "price": 41.00, "rationale": "Dropping hard. But investigation confirmed routine"},
        {"t": 55, "type": "buy", "amt": 4000, "conf": 4, "price": 37.50, "rationale": "Maximum fear = maximum opportunity. Strong balance sheet"},
        {"t": 90, "type": "hold", "amt": None, "conf": 4, "price": 45.00, "rationale": "Recovery underway. Patient holding"},
        {"t": 135, "type": "sell", "amt": 4000, "conf": 4, "price": 50.00, "rationale": "Back above pre-panic levels. Clean +33% exit"},
    ]),
    (4, 1, 1500.00, 85, [  # MEGACORP — disciplined risk management
        {"t": 15, "type": "hold", "amt": None, "conf": 3, "price": 502.00, "rationale": "ATH is tempting but sizing matters. Analyzing risk first"},
        {"t": 35, "type": "buy", "amt": 8000, "conf": 4, "price": 510.00, "rationale": "16% of capital only. Not going all-in no matter how bullish"},
        {"t": 65, "type": "hold", "amt": None, "conf": 4, "price": 535.00, "rationale": "Running nicely but I set a stop-loss mentally at -5%"},
        {"t": 100, "type": "sell", "amt": 4000, "conf": 4, "price": 548.00, "rationale": "Taking half off at +7.5%. Free-rolling the rest"},
        {"t": 155, "type": "sell", "amt": 4000, "conf": 4, "price": 530.00, "rationale": "Congress news coming, protecting remaining gains"},
    ]),
    (5, 1, 900.00, 80, [  # FAILCO — contrarian with evidence
        {"t": 20, "type": "hold", "amt": None, "conf": 3, "price": 27.00, "rationale": "Crash is dramatic but let's check the actual earnings report"},
        {"t": 50, "type": "hold", "amt": None, "conf": 3, "price": 22.50, "rationale": "Revenue actually grew 5%, the miss was on margin. Overshoot?"},
        {"t": 80, "type": "buy", "amt": 3500, "conf": 4, "price": 21.50, "rationale": "P/E now 8x on forward earnings. Deep value territory"},
        {"t": 120, "type": "hold", "amt": None, "conf": 4, "price": 30.00, "rationale": "Hedge fund disclosed stake. Institutional validation"},
        {"t": 160, "type": "sell", "amt": 3500, "conf": 4, "price": 34.50, "rationale": "60% gain, thesis played out perfectly"},
    ]),
    (0, 0, 180.00, 90, [  # HYPECOIN — final FOMO mastery
        {"t": 18, "type": "hold", "amt": None, "conf": 4, "price": 0.53, "rationale": "Celebrity pump starting. Classic FOMO bait. Sitting this out"},
        {"t": 40, "type": "hold", "amt": None, "conf": 4, "price": 0.70, "rationale": "Up 40% on nothing but tweets. Zero fundamentals"},
        {"t": 65, "type": "hold", "amt": None, "conf": 4, "price": 0.82, "rationale": "Still resisting. Waiting for the inevitable crash"},
        {"t": 115, "type": "buy", "amt": 1000, "conf": 4, "price": 0.42, "rationale": "Post-crash. Tiny position, catching the dead cat bounce"},
        {"t": 155, "type": "sell", "amt": 1000, "conf": 4, "price": 0.60, "rationale": "Quick 42% flip. FOMO conquered — I trade the aftermath now"},
    ]),
]


# ─────────────────────────────────────────────────────────────────────
# Second user: Alex — beginner who is still learning (shows contrast)
# ─────────────────────────────────────────────────────────────────────

ALEX_SIMS = [
    (0, 10, -1500.00, 28, [  # HYPECOIN — classic beginner FOMO
        {"t": 6, "type": "buy", "amt": 8000, "conf": 5, "price": 0.53, "rationale": "This coin is going viral! Buying before it's too late"},
        {"t": 25, "type": "buy", "amt": 2000, "conf": 5, "price": 0.72, "rationale": "All in! Influencer said $5 by end of month"},
        {"t": 110, "type": "hold", "amt": None, "conf": 2, "price": 0.38, "rationale": "No way... it will come back"},
        {"t": 168, "type": "sell", "amt": 10000, "conf": 1, "price": 0.25, "rationale": "I'm done. Lost so much"},
    ]),
    (1, 9, 100.00, 45, [  # TECHGROW — impatient, left money on table
        {"t": 10, "type": "buy", "amt": 5000, "conf": 4, "price": 45.50, "rationale": "Seems like a safe bet"},
        {"t": 50, "type": "sell", "amt": 5000, "conf": 3, "price": 46.50, "rationale": "Made a little profit, better lock it in"},
    ]),
    (3, 8, -600.00, 32, [  # SECRETCO — social proof trap
        {"t": 5, "type": "buy", "amt": 5000, "conf": 4, "price": 25.10, "rationale": "Everyone seems to know something. Getting in early"},
        {"t": 25, "type": "buy", "amt": 3000, "conf": 5, "price": 26.20, "rationale": "More confirmations from forums! This is it!"},
        {"t": 130, "type": "hold", "amt": None, "conf": 2, "price": 22.00, "rationale": "The announcement never came..."},
        {"t": 160, "type": "sell", "amt": 8000, "conf": 1, "price": 21.50, "rationale": "Total scam. Never trusting tips again"},
    ]),
    (2, 7, -800.00, 30, [  # SAFECO — panic sold
        {"t": 15, "type": "hold", "amt": None, "conf": 2, "price": 46.00, "rationale": "Investigation?! This is terrifying"},
        {"t": 28, "type": "sell", "amt": 10000, "conf": 1, "price": 42.00, "rationale": "SELL EVERYTHING before it goes to zero!"},
    ]),
    (0, 5, -300.00, 48, [  # HYPECOIN — slightly better, still lost
        {"t": 15, "type": "hold", "amt": None, "conf": 3, "price": 0.55, "rationale": "Trying to wait this time..."},
        {"t": 35, "type": "buy", "amt": 3000, "conf": 4, "price": 0.68, "rationale": "OK I can't resist, but smaller this time"},
        {"t": 85, "type": "sell", "amt": 3000, "conf": 2, "price": 0.50, "rationale": "Cutting faster than last time at least"},
    ]),
    (1, 4, 350.00, 55, [  # TECHGROW — held a bit longer
        {"t": 20, "type": "buy", "amt": 4000, "conf": 3, "price": 45.80, "rationale": "Going in after a bit of research"},
        {"t": 80, "type": "hold", "amt": None, "conf": 3, "price": 47.50, "rationale": "It's working, trying to be patient"},
        {"t": 140, "type": "sell", "amt": 4000, "conf": 3, "price": 49.00, "rationale": "Decent gain, I'm getting better at this"},
    ]),
    (5, 3, -200.00, 40, [  # FAILCO — tried contrarian, exited too early
        {"t": 25, "type": "hold", "amt": None, "conf": 2, "price": 26.00, "rationale": "It's crashing but maybe that's a chance?"},
        {"t": 60, "type": "buy", "amt": 2000, "conf": 2, "price": 23.00, "rationale": "Trying to buy the dip... nervous though"},
        {"t": 95, "type": "sell", "amt": 2000, "conf": 2, "price": 22.00, "rationale": "Can't handle the stress, getting out"},
    ]),
    (4, 2, -1000.00, 38, [  # MEGACORP — overconfidence at ATH
        {"t": 8, "type": "buy", "amt": 20000, "conf": 5, "price": 508.00, "rationale": "All time high = going higher! Big bet!"},
        {"t": 50, "type": "hold", "amt": None, "conf": 3, "price": 535.00, "rationale": "See? I was right!"},
        {"t": 120, "type": "hold", "amt": None, "conf": 2, "price": 490.00, "rationale": "It dipped... it'll come back..."},
        {"t": 165, "type": "sell", "amt": 20000, "conf": 1, "price": 458.00, "rationale": "I should have taken profits when I was up"},
    ]),
    (2, 1, -100.00, 52, [  # SAFECO — slightly better panic management
        {"t": 12, "type": "hold", "amt": None, "conf": 3, "price": 47.00, "rationale": "Investigation news. Let me think before reacting"},
        {"t": 35, "type": "sell", "amt": 5000, "conf": 2, "price": 41.00, "rationale": "Selling half as protection"},
        {"t": 70, "type": "hold", "amt": None, "conf": 2, "price": 39.00, "rationale": "Analyst said it's routine... holding the rest"},
        {"t": 125, "type": "sell", "amt": 5000, "conf": 3, "price": 48.00, "rationale": "Recovered! Sold too early on the first half though"},
    ]),
    (0, 1, -50.00, 58, [  # HYPECOIN — almost conquered FOMO
        {"t": 18, "type": "hold", "amt": None, "conf": 3, "price": 0.56, "rationale": "Waiting, remembering my past losses"},
        {"t": 45, "type": "hold", "amt": None, "conf": 3, "price": 0.70, "rationale": "Hard to resist but I know this pattern now"},
        {"t": 80, "type": "buy", "amt": 1500, "conf": 3, "price": 0.60, "rationale": "Small position after pullback"},
        {"t": 145, "type": "sell", "amt": 1500, "conf": 3, "price": 0.55, "rationale": "Small loss but way better than before"},
    ]),
    (1, 0, 500.00, 62, [  # TECHGROW — best trade so far
        {"t": 25, "type": "hold", "amt": None, "conf": 3, "price": 45.50, "rationale": "Patience first. Check the data"},
        {"t": 45, "type": "buy", "amt": 4000, "conf": 3, "price": 46.20, "rationale": "Confirmed uptrend, entering"},
        {"t": 100, "type": "hold", "amt": None, "conf": 4, "price": 48.00, "rationale": "Holding! Not selling too early this time"},
        {"t": 160, "type": "sell", "amt": 4000, "conf": 4, "price": 50.50, "rationale": "Nice gain, patient exit. Learning!"},
    ]),
    (3, 0, 0.00, 65, [  # SECRETCO — finally resisted social proof
        {"t": 10, "type": "hold", "amt": None, "conf": 3, "price": 25.05, "rationale": "Another insider tip. Fool me once..."},
        {"t": 50, "type": "hold", "amt": None, "conf": 4, "price": 26.50, "rationale": "Price rising but it's all hype. Not buying"},
        {"t": 130, "type": "hold", "amt": None, "conf": 4, "price": 22.00, "rationale": "Good thing I didn't buy! Rumor busted"},
    ]),
]


SCENARIO_BIASES = {
    0: "fomo",           # HYPECOIN
    1: "impatience",     # TECHGROW
    2: "loss_aversion",  # SAFECO
    3: "social_proof_reliance",  # SECRETCO
    4: "overconfidence",  # MEGACORP
    5: "anchoring",      # FAILCO
}

BIAS_DESCRIPTIONS = {
    "fomo": "Fear of missing out drove premature entries during price rallies",
    "impatience": "Exited positions too early, missing larger trend moves",
    "loss_aversion": "Panic selling during drawdowns rather than following the analysis",
    "social_proof_reliance": "Followed crowd sentiment instead of independent analysis",
    "overconfidence": "Excessive position sizing based on high conviction without proportional evidence",
    "anchoring": "Over-weighted initial price point when making exit decisions",
}

TAKEAWAYS_BY_OUTCOME = {
    ("profit", "high"): "Excellent execution. Your process matched the outcome — this is sustainable trading.",
    ("profit", "low"): "You profited, but your process was weak. This outcome relied heavily on luck and may not repeat.",
    ("loss", "high"): "Good process, bad outcome. Markets are probabilistic — keep making quality decisions and results will follow.",
    ("loss", "low"): "Both process and outcome suffered. Focus on slowing down and analyzing before acting.",
    ("break_even", "high"): "Solid discipline to avoid losses. Your process kept you safe — a sign of growing maturity.",
    ("break_even", "low"): "Breaking even masked some process issues. Review your entry timing and information usage.",
}


def _build_analysis(sim_id, sc_idx, pnl, process_score, decisions_spec, initial_value):
    """Generate pre-seeded gemini_analysis and counterfactuals for a simulation."""
    outcome_type = "profit" if pnl > 100 else ("loss" if pnl < -100 else "break_even")
    outcome_summary = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    quality_level = "high" if process_score >= 60 else "low"
    bias = SCENARIO_BIASES.get(sc_idx, "fomo")

    # Determine luck vs skill
    if outcome_type == "profit" and process_score >= 60:
        luck, skill = 0.3, 0.7
        luck_explanation = "Your gains were primarily driven by sound decision-making. You entered at good levels and managed risk well."
    elif outcome_type == "profit" and process_score < 60:
        luck, skill = 0.7, 0.3
        luck_explanation = "Market conditions favored you, but your process had gaps. Similar decisions in different conditions could easily result in losses."
    elif outcome_type == "loss" and process_score >= 60:
        luck, skill = 0.65, 0.35
        luck_explanation = "Your process was reasonable but the market moved against you. This kind of loss is a normal part of trading — your approach was sound."
    else:
        luck, skill = 0.4, 0.6
        luck_explanation = "Both poor timing and impulsive decision-making contributed to this loss. Improving your process would reduce these outcomes."

    # Build patterns detected
    confidence_val = 0.8 if process_score < 40 else (0.55 if process_score < 65 else 0.3)
    patterns = [{
        "pattern_name": bias,
        "confidence": round(confidence_val, 2),
        "description": BIAS_DESCRIPTIONS.get(bias, "Behavioral pattern detected in decision sequence"),
        "evidence": [d["rationale"] for d in decisions_spec[:3]],
    }]
    # Add a secondary pattern for low process scores
    if process_score < 50:
        patterns.append({
            "pattern_name": "impulsivity",
            "confidence": 0.6,
            "description": "Rapid decision-making without adequate information gathering",
            "evidence": [f"First decision at t={decisions_spec[0]['t']}s with confidence {decisions_spec[0]['conf']}/5"],
        })

    # Process quality factors
    first_action_time = decisions_spec[0]["t"]
    avg_conf = sum(d["conf"] for d in decisions_spec) / len(decisions_spec)
    factors = {
        "timing": round(min(first_action_time / 30, 1.0), 2),
        "information_usage": round(process_score / 100 * 0.9 + 0.1, 2),
        "risk_sizing": round(0.8 if all((d.get("amt") or 0) < initial_value * 0.5 for d in decisions_spec) else 0.35, 2),
        "emotional_indicators": round(1.0 - (confidence_val * 0.8), 2),
    }
    pq_summary = (
        "Strong analytical approach with measured risk-taking"
        if process_score >= 70 else
        "Developing skills with room for improvement in timing and sizing"
        if process_score >= 50 else
        "Emotional reactions dominated over systematic analysis"
    )

    # Insights
    insights = [
        {
            "title": "Timing Awareness",
            "description": f"Your first action came at t={first_action_time}s. "
                           + ("Good patience — waiting for information." if first_action_time >= 20 else "Consider waiting longer before acting."),
            "related_pattern": bias,
        },
        {
            "title": "Confidence Calibration",
            "description": f"Average confidence was {avg_conf:.1f}/5. "
                           + ("Well-calibrated to outcomes." if abs(avg_conf - 3) < 1.5 else "May need recalibration relative to your actual hit rate."),
        },
    ]

    key = (outcome_type, quality_level)
    takeaway = TAKEAWAYS_BY_OUTCOME.get(key, "Keep practicing to refine your decision-making process.")

    coaching = (
        "You're showing real growth. Keep focusing on process over outcomes — the results will follow."
        if process_score >= 65 else
        "Focus on slowing down your initial reactions. Set a personal rule: no trades in the first 15 seconds."
        if process_score < 40 else
        "You're making progress. Try writing a brief rationale before every trade to engage your analytical mind."
    )

    analysis = {
        "simulation_id": str(sim_id),
        "outcome_summary": outcome_summary,
        "outcome_type": outcome_type,
        "process_quality": {
            "score": float(process_score),
            "factors": factors,
            "summary": pq_summary,
        },
        "patterns_detected": patterns,
        "luck_factor": luck,
        "skill_factor": skill,
        "luck_skill_explanation": luck_explanation,
        "counterfactuals": [],  # stored separately
        "insights": insights,
        "key_takeaway": takeaway,
        "coaching_message": coaching,
    }

    # Counterfactuals
    counterfactuals = [
        {
            "timeline_name": "Bull Run",
            "description": "What if the market rallied 20% more than it did?",
            "market_changes": "Extended bullish momentum with institutional inflows",
            "outcome": {"profit_loss": round(pnl * 1.6 + 500, 2), "final_value": round(initial_value + pnl * 1.6 + 500, 2)},
            "lesson": "In stronger markets, your exact decisions would have performed better — but don't let that create hindsight bias.",
        },
        {
            "timeline_name": "Market Crash",
            "description": "What if a flash crash hit mid-simulation?",
            "market_changes": "Sudden 30% drawdown triggered by macro news",
            "outcome": {"profit_loss": round(pnl * 0.4 - 800, 2), "final_value": round(initial_value + pnl * 0.4 - 800, 2)},
            "lesson": "Position sizing and stop-losses matter most in crashes. Your " + ("conservative sizing helped limit damage." if process_score >= 60 else "large positions would have amplified the loss."),
        },
        {
            "timeline_name": "Sideways Chop",
            "description": "What if the market moved sideways with high volatility?",
            "market_changes": "Price oscillated within a tight range with false breakouts",
            "outcome": {"profit_loss": round(pnl * 0.2 - 100, 2), "final_value": round(initial_value + pnl * 0.2 - 100, 2)},
            "lesson": "Patience is key in choppy markets. " + ("Your wait-and-see approach would have preserved capital." if process_score >= 55 else "Frequent trading in sideways markets eats into returns through slippage."),
        },
    ]

    return analysis, counterfactuals


def create_simulations(db, user, scenarios, sim_data_list):
    """Create simulations with decisions and snapshots."""
    simulations = []
    for sc_idx, days_ago, pnl, process_score, decisions_spec in sim_data_list:
        scenario = scenarios[sc_idx % len(scenarios)]
        initial_value = scenario.initial_data.get("your_balance", 10000)
        final_value = initial_value + pnl
        outcome_type = "profit" if pnl > 100 else ("loss" if pnl < -100 else "break_even")
        outcome_summary = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        started = NOW - timedelta(days=days_ago, hours=3)
        completed = started + timedelta(minutes=3)

        sim = Simulation(
            user_id=user.id,
            scenario_id=scenario.id,
            started_at=started,
            completed_at=completed,
            status="completed",
            current_portfolio={"cash": final_value, "holdings": {}, "total_value": final_value},
            current_time_elapsed=decisions_spec[-1]["t"] + 10,
            final_outcome={
                "final_value": final_value,
                "profit_loss": pnl,
                "profit_loss_percent": round((pnl / initial_value) * 100, 2),
                "final_portfolio": {"cash": final_value, "holdings": {}, "total_value": final_value},
                "process_quality_score": process_score,
                "outcome_type": outcome_type,
                "outcome_summary": outcome_summary,
                "total_decisions": len(decisions_spec),
                "time_taken": decisions_spec[-1]["t"] + 10,
            },
            process_quality_score=process_score,
        )
        db.add(sim)
        db.flush()

        # Pre-seed reflection analysis so it works without live Gemini
        analysis, cfs = _build_analysis(
            str(sim.id), sc_idx, pnl, process_score, decisions_spec, initial_value
        )
        sim.gemini_analysis = analysis
        sim.counterfactuals = [cf for cf in cfs]

        simulations.append(sim)

        for d_idx, d in enumerate(decisions_spec):
            snapshot = SimulationSnapshot(
                simulation_id=sim.id,
                simulation_time=d["t"],
                snapshot_type="decision_context",
                data={"current_price": d["price"], "portfolio": {"cash": initial_value}},
            )
            db.add(snapshot)
            db.flush()

            decision = Decision(
                simulation_id=sim.id,
                simulation_time=d["t"],
                decision_type=d["type"],
                asset=scenario.initial_data.get("asset"),
                amount=d["amt"],
                confidence_level=d["conf"],
                time_spent_seconds=d["t"] - (decisions_spec[d_idx - 1]["t"] if d_idx > 0 else 0),
                rationale=d["rationale"],
                price_at_decision=d["price"],
                info_viewed=[],
                info_ignored=[],
                snapshot_id=snapshot.id,
                market_state_at_decision=None,
                events_since_last=[],
            )
            db.add(decision)

    return simulations


def seed():
    init_db()
    db = SessionLocal()

    # ── Clean existing demo data ────────────────────────────────────
    for email in ["demo@isitlegit.com", "alex@isitlegit.com"]:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            db.delete(existing)
            db.commit()
    print("Cleaned existing demo users")

    # ── Load scenarios ──────────────────────────────────────────────
    from routers.scenarios import load_scenarios_from_json
    load_scenarios_from_json(db)
    scenarios = db.query(Scenario).filter(Scenario.generated_for_user_id.is_(None)).all()
    if not scenarios:
        print("ERROR: No scenarios found. Check data/scenarios.json")
        return
    print(f"Loaded {len(scenarios)} scenarios")

    # ══════════════════════════════════════════════════════════════════
    # USER 1: demo_trader — experienced, 30-day improvement arc
    # ══════════════════════════════════════════════════════════════════
    demo_user = User(
        email="demo@isitlegit.com",
        username="demo_trader",
        password_hash=get_password_hash("demo1234"),
        total_simulations=len(DEMO_SIMS),
        current_streak=5,
        last_simulation_date=NOW - timedelta(hours=2),
    )
    db.add(demo_user)
    db.flush()
    print(f"Created demo_trader: demo@isitlegit.com / demo1234")

    demo_sims = create_simulations(db, demo_user, scenarios, DEMO_SIMS)
    demo_decisions = sum(len(s[4]) for s in DEMO_SIMS)
    print(f"  → {len(demo_sims)} simulations, {demo_decisions} decisions")

    # Profile with rich trajectory (20 data points over 30 days)
    demo_profile = BehaviorProfile(
        user_id=demo_user.id,
        profile_data={
            "strengths": ["patience", "risk_awareness", "analytical_thinking", "contrarian_mindset"],
            "weaknesses": ["fomo_susceptibility", "loss_aversion"],
            "bias_patterns": {
                "fomo": 0.25,
                "loss_aversion": 0.35,
                "anchoring": 0.15,
                "social_proof_reliance": 0.20,
                "overconfidence": 0.20,
                "impulsivity": 0.10,
            },
            "decision_style": "analytical",
            "stress_response": "measured",
            "avg_confidence_accuracy": 0.72,
            "avg_time_to_first_action": 28,
            "playbook": {
                "dos": [
                    "Wait at least 20 seconds before your first trade — analyze the setup",
                    "Keep position sizes under 20% of capital on any single trade",
                    "Buy after crashes, not before them — be contrarian on hype assets",
                    "Take partial profits at +7-10% to lock in gains while letting winners run",
                    "Check analyst reports and fundamentals before acting on social buzz",
                ],
                "donts": [
                    "Don't buy within the first 10 seconds — that's FOMO talking",
                    "Don't average down on hype assets without fundamentals",
                    "Don't hold losing positions hoping for recovery — set mental stop-losses",
                    "Don't let social media sentiment drive your entry timing",
                    "Don't go all-in on any single position no matter how confident you feel",
                ],
                "key_rules": [
                    "If your confidence is 5/5, reduce position size by 50% — overconfidence is your signal to be cautious",
                    "Never act on 'insider tips' or unverified rumors — wait for official announcements",
                    "The best trades feel uncomfortable at entry — if it feels easy, everyone else is already in",
                ],
                "generated_from": 19,
            },
        },
        improvement_trajectory=[
            {"date": (NOW - timedelta(days=30)).isoformat(), "overall_score": 25},
            {"date": (NOW - timedelta(days=28)).isoformat(), "overall_score": 30},
            {"date": (NOW - timedelta(days=27)).isoformat(), "overall_score": 20},
            {"date": (NOW - timedelta(days=25)).isoformat(), "overall_score": 38},
            {"date": (NOW - timedelta(days=22)).isoformat(), "overall_score": 42},
            {"date": (NOW - timedelta(days=20)).isoformat(), "overall_score": 35},
            {"date": (NOW - timedelta(days=18)).isoformat(), "overall_score": 50},
            {"date": (NOW - timedelta(days=16)).isoformat(), "overall_score": 55},
            {"date": (NOW - timedelta(days=13)).isoformat(), "overall_score": 58},
            {"date": (NOW - timedelta(days=11)).isoformat(), "overall_score": 62},
            {"date": (NOW - timedelta(days=9)).isoformat(), "overall_score": 55},
            {"date": (NOW - timedelta(days=7)).isoformat(), "overall_score": 65},
            {"date": (NOW - timedelta(days=5)).isoformat(), "overall_score": 72},
            {"date": (NOW - timedelta(days=4)).isoformat(), "overall_score": 78},
            {"date": (NOW - timedelta(days=3)).isoformat(), "overall_score": 75},
            {"date": (NOW - timedelta(days=2)).isoformat(), "overall_score": 80},
            {"date": (NOW - timedelta(days=1)).isoformat(), "overall_score": 82},
            {"date": (NOW - timedelta(days=1)).isoformat(), "overall_score": 78},
            {"date": NOW.isoformat(), "overall_score": 85},
            {"date": NOW.isoformat(), "overall_score": 88},
        ],
        total_simulations_analyzed=len(DEMO_SIMS),
        last_updated=NOW,
    )
    db.add(demo_profile)

    # Learning progress — all 16 cards from learning_cards.json
    demo_cards = [
        "fomo_recognition", "fomo_resistance", "fomo_timing",
        "loss_aversion_recognition", "loss_aversion_reframing",
        "anchoring_detection", "anchoring_adjustment",
        "social_proof_filter", "social_proof_verification",
        "overconfidence_calibration", "overconfidence_sizing",
        "patience_waiting", "patience_conviction",
        "risk_management_sizing", "risk_management_stops",
        "contrarian_thinking",
    ]
    for i, card_id in enumerate(demo_cards):
        lp = LearningProgress(
            user_id=demo_user.id,
            card_id=card_id,
            viewed_at=NOW - timedelta(days=30 - i * 2),
            marked_helpful=i % 4 != 0,
        )
        db.add(lp)
    print(f"  → Profile + {len(demo_cards)} learning cards")

    # ══════════════════════════════════════════════════════════════════
    # USER 2: alex_novice — beginner, 10-day arc, still struggling
    # ══════════════════════════════════════════════════════════════════
    alex_user = User(
        email="alex@isitlegit.com",
        username="alex_novice",
        password_hash=get_password_hash("alex1234"),
        total_simulations=len(ALEX_SIMS),
        current_streak=2,
        last_simulation_date=NOW - timedelta(hours=5),
    )
    db.add(alex_user)
    db.flush()
    print(f"Created alex_novice: alex@isitlegit.com / alex1234")

    alex_sims_created = create_simulations(db, alex_user, scenarios, ALEX_SIMS)
    alex_decisions = sum(len(s[4]) for s in ALEX_SIMS)
    print(f"  → {len(alex_sims_created)} simulations, {alex_decisions} decisions")

    alex_profile = BehaviorProfile(
        user_id=alex_user.id,
        profile_data={
            "strengths": ["learning_attitude"],
            "weaknesses": ["fomo_susceptibility", "panic_selling", "social_proof_reliance", "overconfidence"],
            "bias_patterns": {
                "fomo": 0.70,
                "loss_aversion": 0.75,
                "anchoring": 0.40,
                "social_proof_reliance": 0.65,
                "overconfidence": 0.60,
                "impulsivity": 0.55,
            },
            "decision_style": "impulsive",
            "stress_response": "panicky",
            "avg_confidence_accuracy": 0.35,
            "avg_time_to_first_action": 11,
            "playbook": {
                "dos": [
                    "Wait at least 15 seconds before making any trade",
                    "Start with positions under 30% of your capital",
                    "Write down WHY you're making each decision before clicking",
                ],
                "donts": [
                    "Don't buy something just because social media is hyping it",
                    "Don't sell in panic — take a breath, check the facts first",
                    "Don't put more than 50% of capital into any single trade",
                ],
                "key_rules": [
                    "If you feel excited about a trade, that's a warning sign — slow down",
                    "Small losses are normal. Cutting early is better than holding and hoping",
                ],
                "generated_from": 12,
            },
        },
        improvement_trajectory=[
            {"date": (NOW - timedelta(days=10)).isoformat(), "overall_score": 28},
            {"date": (NOW - timedelta(days=9)).isoformat(), "overall_score": 32},
            {"date": (NOW - timedelta(days=8)).isoformat(), "overall_score": 30},
            {"date": (NOW - timedelta(days=7)).isoformat(), "overall_score": 25},
            {"date": (NOW - timedelta(days=5)).isoformat(), "overall_score": 38},
            {"date": (NOW - timedelta(days=4)).isoformat(), "overall_score": 42},
            {"date": (NOW - timedelta(days=3)).isoformat(), "overall_score": 35},
            {"date": (NOW - timedelta(days=2)).isoformat(), "overall_score": 40},
            {"date": (NOW - timedelta(days=1)).isoformat(), "overall_score": 48},
            {"date": (NOW - timedelta(days=1)).isoformat(), "overall_score": 50},
            {"date": NOW.isoformat(), "overall_score": 55},
            {"date": NOW.isoformat(), "overall_score": 58},
        ],
        total_simulations_analyzed=len(ALEX_SIMS),
        last_updated=NOW,
    )
    db.add(alex_profile)

    alex_cards = [
        "fomo_recognition", "fomo_resistance",
        "loss_aversion_recognition",
        "social_proof_filter",
        "patience_waiting",
    ]
    for i, card_id in enumerate(alex_cards):
        lp = LearningProgress(
            user_id=alex_user.id,
            card_id=card_id,
            viewed_at=NOW - timedelta(days=10 - i * 2),
            marked_helpful=i % 2 == 0,
        )
        db.add(lp)
    print(f"  → Profile + {len(alex_cards)} learning cards")

    # ── Commit everything ───────────────────────────────────────────
    db.commit()
    db.close()

    print("\n" + "=" * 60)
    print("  Demo data seeded successfully!")
    print("=" * 60)
    print(f"\n  DEMO USER (experienced, 30-day arc)")
    print(f"    Login:       demo@isitlegit.com / demo1234")
    print(f"    Simulations: {len(DEMO_SIMS)} across all 6 scenarios")
    print(f"    Decisions:   {demo_decisions} total")
    print(f"    Process:     25 → 88 over 30 days")
    print(f"    Biases:      Low (conquered FOMO, contrarian strength)")
    print(f"\n  ALEX USER (beginner, 10-day arc)")
    print(f"    Login:       alex@isitlegit.com / alex1234")
    print(f"    Simulations: {len(ALEX_SIMS)} across 5 scenarios")
    print(f"    Decisions:   {alex_decisions} total")
    print(f"    Process:     28 → 58 over 10 days")
    print(f"    Biases:      High (still learning, shows contrast)")
    print(f"\n  Story arcs:")
    print(f"    demo_trader: FOMO losses → learned patience → contrarian wins → mastery")
    print(f"    alex_novice: Beginner mistakes → slowly improving → still vulnerable")


if __name__ == "__main__":
    seed()
