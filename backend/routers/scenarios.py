import json
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from models.scenario import Scenario
from models.simulation import Simulation
from models.behavior_profile import BehaviorProfile
from routers.auth import get_current_user
from schemas.scenario import ScenarioResponse, ScenarioDetailResponse

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])


def load_scenarios_from_json(db: Session):
    """Load scenarios from JSON file, adding any missing ones by name."""
    data_path = Path(__file__).parent.parent / "data" / "scenarios.json"
    if not data_path.exists():
        return

    with open(data_path) as f:
        scenarios_data = json.load(f)

    existing_names = {s.name for s in db.query(Scenario.name).all()}
    added = 0

    for scenario_data in scenarios_data:
        if scenario_data["name"] in existing_names:
            continue
        scenario = Scenario(
            name=scenario_data["name"],
            description=scenario_data["description"],
            difficulty=scenario_data["difficulty"],
            category=scenario_data["category"],
            initial_data=scenario_data["initial_data"],
            events=scenario_data["events"],
            time_pressure_seconds=scenario_data.get("time_pressure_seconds", 180),
            unlock_requirements=scenario_data.get("unlock_requirements"),
        )
        db.add(scenario)
        added += 1

    if added:
        db.commit()


@router.get("/", response_model=list[ScenarioResponse])
async def list_scenarios(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    category: str | None = None,
    difficulty: int | None = None
):
    """List all available scenarios."""
    # Ensure scenarios are loaded
    load_scenarios_from_json(db)

    query = db.query(Scenario).filter(
        Scenario.is_active == 1,
        # Only show built-in scenarios OR AI scenarios owned by this user
        (Scenario.generated_for_user_id == None) | (Scenario.generated_for_user_id == current_user.id),  # noqa: E711
    )

    if category:
        query = query.filter(Scenario.category == category)
    if difficulty:
        query = query.filter(Scenario.difficulty == difficulty)

    return query.all()


@router.get("/unlocked")
async def list_unlocked_scenarios(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """List scenarios with lock status based on user's progress."""
    load_scenarios_from_json(db)
    scenarios = db.query(Scenario).filter(
        Scenario.is_active == 1,
        # Only show built-in scenarios (no owner) OR AI scenarios owned by this user
        (Scenario.generated_for_user_id == None) | (Scenario.generated_for_user_id == current_user.id),  # noqa: E711
    ).all()

    completed_sims = db.query(Simulation).filter(
        Simulation.user_id == current_user.id,
        Simulation.status == "completed",
    ).all()

    result = []
    for s in scenarios:
        unlocked = _check_unlock(s, completed_sims, current_user)
        result.append({
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "difficulty": s.difficulty,
            "category": s.category,
            "time_pressure_seconds": s.time_pressure_seconds,
            "created_at": s.created_at,
            "is_locked": not unlocked,
            "unlock_requirements": s.unlock_requirements,
            "is_ai_generated": s.generated_for_user_id is not None,
        })

    # Sort: unlocked first, then by difficulty ascending
    result.sort(key=lambda x: (x["is_locked"], x["difficulty"]))
    return result


def _check_unlock(scenario: Scenario, completed_sims: list, user: User) -> bool:
    """Check if a scenario is unlocked for this user."""
    reqs = scenario.unlock_requirements
    if not reqs:
        return True

    if reqs.get("min_simulations") and user.total_simulations < reqs["min_simulations"]:
        return False

    if reqs.get("min_process_score") and completed_sims:
        avg_score = sum(s.process_quality_score or 0 for s in completed_sims) / len(completed_sims)
        if avg_score < reqs["min_process_score"]:
            return False

    return True


@router.post("/generate-adaptive")
async def generate_adaptive_scenario(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Generate an AI-tailored scenario targeting the user's weakest bias."""
    from services.gemini_service import GeminiService

    profile = db.query(BehaviorProfile).filter(
        BehaviorProfile.user_id == current_user.id
    ).first()

    profile_data = profile.profile_data if profile else {
        "bias_patterns": {"fomo": 0.5, "impulsivity": 0.4},
        "weaknesses": ["fomo_susceptibility"],
        "strengths": [],
        "total_simulations_analyzed": 0,
    }

    gemini = GeminiService()
    scenario_data = await gemini.generate_adaptive_scenario(profile_data)

    # Persist the generated scenario
    scenario = Scenario(
        name=scenario_data["name"],
        description=scenario_data["description"],
        difficulty=scenario_data.get("difficulty", 3),
        category=scenario_data.get("category", "fomo_trap"),
        initial_data=scenario_data["initial_data"],
        events=scenario_data["events"],
        time_pressure_seconds=scenario_data.get("time_pressure_seconds", 180),
        generated_for_user_id=current_user.id,
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)

    return {
        "id": scenario.id,
        "name": scenario.name,
        "description": scenario.description,
        "difficulty": scenario.difficulty,
        "category": scenario.category,
        "time_pressure_seconds": scenario.time_pressure_seconds,
        "target_bias": scenario_data.get("target_bias", "general"),
    }


@router.post("/generate-from-url")
async def generate_scenario_from_url(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    url: str = "",
    difficulty: int = 3,
):
    """Generate a scenario from a real article URL using Gemini URL Context."""
    from services.gemini_service import GeminiService

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    gemini = GeminiService()
    scenario_data = await gemini.generate_scenario_from_url(url, difficulty)

    scenario = Scenario(
        name=scenario_data["name"],
        description=scenario_data["description"],
        difficulty=scenario_data.get("difficulty", difficulty),
        category=scenario_data.get("category", "risk_management"),
        initial_data=scenario_data["initial_data"],
        events=scenario_data["events"],
        time_pressure_seconds=scenario_data.get("time_pressure_seconds", 180),
        generated_for_user_id=current_user.id,
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)

    return {
        "id": scenario.id,
        "name": scenario.name,
        "description": scenario.description,
        "difficulty": scenario.difficulty,
        "category": scenario.category,
        "time_pressure_seconds": scenario.time_pressure_seconds,
        "source_url": scenario_data.get("source_url", url),
        "source_summary": scenario_data.get("source_summary", ""),
    }


@router.get("/categories")
async def list_categories(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """List all scenario categories."""
    categories = db.query(Scenario.category).distinct().all()
    return [c[0] for c in categories]


@router.get("/{scenario_id}", response_model=ScenarioDetailResponse)
async def get_scenario(
    scenario_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Get detailed scenario information."""
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()

    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenario not found"
        )

    # Block access to AI scenarios owned by other users
    if scenario.generated_for_user_id and scenario.generated_for_user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenario not found"
        )

    return scenario
