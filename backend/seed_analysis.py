"""
Seed Analysis: Runs Gemini on seeded demo data and persists all AI analysis to DB.

This ensures that judges logging in as demo users see instant reflection results
without waiting for Gemini API calls. Run AFTER seed_demo.py.

Usage:
    cd backend && python seed_analysis.py

Prerequisites:
    - seed_demo.py has been run (demo users + simulations exist)
    - GEMINI_API_KEY environment variable is set
    - Database is running and accessible

What it does:
    For each demo user's simulations (most recent N):
      1. analyze_simulation → gemini_analysis column
      2. generate_counterfactuals → counterfactuals column
      3. explain_decisions (why) → gemini_cache["why"]
      4. compare_with_pro → gemini_cache["pro_comparison"]
      5. analyze_bias_timeline → gemini_cache["bias_heatmap"]
      6. review_rationales → gemini_cache["rationale_review"]
      7. classify_biases_with_gemini → gemini_cache["bias_classifier"]
      8. generate_coaching + update_behavior_profile → gemini_cache["coaching"] + profile update
    Then for each user:
      9. analyze_user_behavior_history → behavior history patterns
"""
import sys
import os
import asyncio
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("DATABASE_URL", "postgresql://isitlegit:isitlegit_secret@localhost:5432/isitlegit")

from database import SessionLocal, init_db
from models.user import User
from models.simulation import Simulation
from models.decision import Decision
from models.behavior_profile import BehaviorProfile
from sqlalchemy.orm.attributes import flag_modified
from services.gemini_service import GeminiService


# How many simulations per user to fully analyze (most recent first)
# Keeps Gemini API usage reasonable while giving judges plenty to explore
SIMS_PER_USER = 5

# Delay between Gemini calls to avoid rate limits
CALL_DELAY = 2.0


def _set_cached(simulation, key, data, db):
    """Persist a Gemini result to the gemini_cache JSONB column."""
    safe_data = json.loads(json.dumps(data, default=str))
    if not simulation.gemini_cache:
        simulation.gemini_cache = {}
    simulation.gemini_cache[key] = safe_data
    flag_modified(simulation, "gemini_cache")
    db.commit()


async def analyze_single_simulation(gemini, sim, decisions, db, sim_index, total):
    """Run all Gemini analysis types for a single simulation."""
    scenario = sim.scenario
    label = f"[{sim_index + 1}/{total}] {scenario.name}"

    # 1. Core reflection analysis
    if not sim.gemini_analysis:
        print(f"  {label}: Generating reflection analysis...")
        try:
            analysis = await gemini.analyze_simulation(
                simulation=sim, decisions=decisions, scenario=scenario
            )
            sim.gemini_analysis = analysis.model_dump(mode='json')
            db.commit()
            print(f"    ✓ Reflection analysis stored")
            await asyncio.sleep(CALL_DELAY)
        except Exception as e:
            print(f"    ✗ Reflection failed: {e}")
    else:
        print(f"  {label}: Reflection already cached")

    # 2. Counterfactuals
    if not sim.counterfactuals:
        print(f"  {label}: Generating counterfactuals...")
        try:
            cfs = await gemini.generate_counterfactuals(
                simulation=sim, decisions=decisions, scenario=scenario
            )
            sim.counterfactuals = [cf.model_dump(mode='json') for cf in cfs]
            db.commit()
            print(f"    ✓ {len(cfs)} counterfactuals stored")
            await asyncio.sleep(CALL_DELAY)
        except Exception as e:
            print(f"    ✗ Counterfactuals failed: {e}")
    else:
        print(f"  {label}: Counterfactuals already cached")

    # 3. Why this decision
    cache = sim.gemini_cache or {}
    if "why" not in cache:
        print(f"  {label}: Generating 'why this decision'...")
        try:
            result = await gemini.explain_decisions(
                simulation=sim, decisions=decisions, scenario=scenario
            )
            _set_cached(sim, "why", result.model_dump(mode='json'), db)
            print(f"    ✓ Why analysis stored")
            await asyncio.sleep(CALL_DELAY)
        except Exception as e:
            print(f"    ✗ Why failed: {e}")

    # 4. Pro comparison
    cache = sim.gemini_cache or {}
    if "pro_comparison" not in cache:
        print(f"  {label}: Generating pro comparison...")
        try:
            result = await gemini.compare_with_pro(
                simulation=sim, decisions=decisions, scenario=scenario
            )
            from services.simulation_engine import SimulationEngine
            engine = SimulationEngine(scenario)
            time_limit = scenario.time_pressure_seconds
            price_history = [
                {"time": t, "price": round(engine.price_timeline.get(t, 0), 2)}
                for t in range(0, time_limit + 1, max(1, time_limit // 60))
            ]
            algo_result = engine.run_algorithmic_trader(decisions)
            data = result.model_dump(mode='json')
            data["price_history"] = price_history
            data["algorithmic_baseline"] = algo_result
            _set_cached(sim, "pro_comparison", data, db)
            print(f"    ✓ Pro comparison stored")
            await asyncio.sleep(CALL_DELAY)
        except Exception as e:
            print(f"    ✗ Pro comparison failed: {e}")

    # 5. Bias heatmap
    cache = sim.gemini_cache or {}
    if "bias_heatmap" not in cache:
        print(f"  {label}: Generating bias heatmap...")
        try:
            result = await gemini.analyze_bias_timeline(
                simulation=sim, decisions=decisions, scenario=scenario
            )
            data = result.model_dump(mode='json') if hasattr(result, 'model_dump') else result
            _set_cached(sim, "bias_heatmap", data, db)
            print(f"    ✓ Bias heatmap stored")
            await asyncio.sleep(CALL_DELAY)
        except Exception as e:
            print(f"    ✗ Bias heatmap failed: {e}")

    # 6. Rationale review
    cache = sim.gemini_cache or {}
    if "rationale_review" not in cache:
        decisions_with_rationale = [d for d in decisions if d.rationale]
        if decisions_with_rationale:
            print(f"  {label}: Generating rationale review...")
            try:
                result = await gemini.review_rationales(
                    simulation=sim, decisions=decisions_with_rationale, scenario=scenario
                )
                data = result.model_dump(mode='json') if hasattr(result, 'model_dump') else result
                _set_cached(sim, "rationale_review", data, db)
                print(f"    ✓ Rationale review stored")
                await asyncio.sleep(CALL_DELAY)
            except Exception as e:
                print(f"    ✗ Rationale review failed: {e}")

    # 7. Bias classifier
    cache = sim.gemini_cache or {}
    if "bias_classifier" not in cache:
        print(f"  {label}: Generating bias classification...")
        try:
            from services.simulation_engine import SimulationEngine
            from services.bias_classifier import extract_decision_features
            engine = SimulationEngine(scenario)
            decision_features = []
            for i, d in enumerate(decisions):
                features = extract_decision_features(
                    d, engine.price_timeline, engine.time_limit,
                    engine.initial_price, engine.initial_balance, decisions, i,
                )
                decision_features.append(features)
            result = await gemini.classify_biases_with_gemini(
                simulation=sim, decisions=decisions, scenario=scenario,
                decision_features=decision_features,
            )
            _set_cached(sim, "bias_classifier", result, db)
            print(f"    ✓ Bias classifier stored")
            await asyncio.sleep(CALL_DELAY)
        except Exception as e:
            print(f"    ✗ Bias classifier failed: {e}")

    print(f"  {label}: Done\n")


async def analyze_user(gemini, user, db):
    """Run all Gemini analysis for a user's simulations."""
    print(f"\n{'=' * 60}")
    print(f"  Analyzing: {user.username} ({user.email})")
    print(f"{'=' * 60}")

    # Get completed simulations, most recent first
    simulations = db.query(Simulation).filter(
        Simulation.user_id == user.id,
        Simulation.status == "completed",
    ).order_by(Simulation.completed_at.desc()).all()

    if not simulations:
        print(f"  No completed simulations found for {user.username}")
        return

    # Analyze the most recent N simulations
    to_analyze = simulations[:SIMS_PER_USER]
    print(f"  Found {len(simulations)} simulations, analyzing {len(to_analyze)}")

    for i, sim in enumerate(to_analyze):
        decisions = db.query(Decision).filter(
            Decision.simulation_id == sim.id
        ).order_by(Decision.simulation_time).all()

        await analyze_single_simulation(gemini, sim, decisions, db, i, len(to_analyze))

    # Generate coaching + update profile for the most recent simulation
    most_recent = to_analyze[0]
    cache = most_recent.gemini_cache or {}
    if "coaching" not in cache:
        print(f"  Generating coaching for most recent simulation...")
        decisions = db.query(Decision).filter(
            Decision.simulation_id == most_recent.id
        ).order_by(Decision.simulation_time).all()

        profile = db.query(BehaviorProfile).filter(
            BehaviorProfile.user_id == user.id
        ).first()
        profile_data = profile.profile_data if profile else None

        try:
            persona, _ = gemini._determine_persona(profile_data)
            coaching = await gemini.generate_coaching(
                simulation=most_recent, decisions=decisions,
                scenario=most_recent.scenario, behavior_profile=profile_data,
            )
            updated_profile = await gemini.update_behavior_profile(
                user_id=str(user.id), simulation=most_recent,
                decisions=decisions, scenario=most_recent.scenario,
                existing_profile=profile_data,
            )

            # Update profile in DB
            if profile:
                profile.profile_data = updated_profile
                profile.total_simulations_analyzed += 1
                flag_modified(profile, "profile_data")
            else:
                from datetime import datetime
                profile = BehaviorProfile(
                    user_id=user.id,
                    profile_data=updated_profile,
                    total_simulations_analyzed=1,
                    last_updated=datetime.utcnow(),
                )
                db.add(profile)

            result = {
                "coaching_message": coaching,
                "persona": persona,
                "profile_updated": True,
            }
            _set_cached(most_recent, "coaching", result, db)
            print(f"    ✓ Coaching + profile update stored")
            await asyncio.sleep(CALL_DELAY)
        except Exception as e:
            print(f"    ✗ Coaching failed: {e}")

    # Run behavior history analysis
    print(f"\n  Running behavior history analysis for {user.username}...")
    history_summary = []
    for sim in simulations:
        outcome = sim.final_outcome or {}
        top_biases = []
        if sim.gemini_analysis:
            for pat in sim.gemini_analysis.get("patterns_detected", [])[:3]:
                top_biases.append({
                    "bias": pat.get("pattern_name", ""),
                    "score": pat.get("confidence", 0),
                })
        history_summary.append({
            "simulation_id": str(sim.id),
            "scenario_name": sim.scenario.name if sim.scenario else "Unknown",
            "category": sim.scenario.category if sim.scenario else "unknown",
            "difficulty": sim.scenario.difficulty if sim.scenario else 3,
            "profit_loss": outcome.get("profit_loss", 0),
            "process_quality_score": sim.process_quality_score,
            "decisions_count": outcome.get("total_decisions", 0),
            "top_biases": top_biases,
            "timestamp": sim.started_at.isoformat() if sim.started_at else "unknown",
        })

    try:
        profile = db.query(BehaviorProfile).filter(
            BehaviorProfile.user_id == user.id
        ).first()
        existing_profile = profile.profile_data if profile else None

        history_result = await gemini.analyze_user_behavior_history(
            history_summary=history_summary,
            existing_profile=existing_profile,
        )
        print(f"    ✓ Behavior history analysis complete")
        # Store the result in the user's most recent simulation's gemini_cache
        # so the profile page can load it instantly
        if profile:
            if not profile.profile_data:
                profile.profile_data = {}
            profile.profile_data["behavior_history_analysis"] = (
                history_result.model_dump(mode='json')
                if hasattr(history_result, 'model_dump')
                else json.loads(json.dumps(history_result, default=str))
            )
            flag_modified(profile, "profile_data")
            db.commit()
            print(f"    ✓ Behavior history persisted to profile")
    except Exception as e:
        print(f"    ✗ Behavior history failed: {e}")


async def main():
    print("=" * 60)
    print("  IsItLegit — Seed Analysis Runner")
    print("  Runs Gemini on demo data and persists to DB")
    print("=" * 60)

    init_db()
    db = SessionLocal()
    gemini = GeminiService()

    start = time.time()

    # Find demo users
    demo_emails = ["demo@isitlegit.com", "alex@isitlegit.com"]
    for email in demo_emails:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"\n  ✗ User {email} not found. Run seed_demo.py first.")
            continue
        await analyze_user(gemini, user, db)

    elapsed = time.time() - start
    db.close()

    print(f"\n{'=' * 60}")
    print(f"  Seed analysis complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"\n  Demo users now have pre-computed AI analysis.")
    print(f"  Judges will see instant reflection results on login.")
    print(f"\n  To verify: log in as demo@isitlegit.com / demo1234")
    print(f"  and navigate to any completed simulation's reflection.")


if __name__ == "__main__":
    asyncio.run(main())
