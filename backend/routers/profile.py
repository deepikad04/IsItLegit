from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from uuid import UUID

from database import get_db
from models.user import User
from models.simulation import Simulation
from models.decision import Decision
from models.behavior_profile import BehaviorProfile
from routers.auth import get_current_user
from services.gemini_service import GeminiService
from schemas.behavior_profile import (
    BehaviorProfileResponse,
    BiasPattern,
    ImprovementPoint,
    ProfileSummary
)

router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("/", response_model=BehaviorProfileResponse, summary="Get full behavior profile")
async def get_behavior_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Get user's full behavior profile."""
    profile = db.query(BehaviorProfile).filter(
        BehaviorProfile.user_id == current_user.id
    ).first()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )

    profile_data = profile.profile_data or {}

    # Convert bias patterns to list format
    bias_patterns = []
    for name, score in profile_data.get("bias_patterns", {}).items():
        bias_patterns.append(BiasPattern(
            name=name,
            score=score,
            description=get_bias_description(name),
            trend="stable"  # Would be calculated from history
        ))

    # Convert improvement trajectory
    trajectory = []
    for point in profile.improvement_trajectory or []:
        trajectory.append(ImprovementPoint(
            date=datetime.fromisoformat(point["date"]) if isinstance(point["date"], str) else point["date"],
            overall_score=point.get("overall_score", point.get("score", 50)),
            simulations_count=point.get("simulations_count", 0),
            key_changes=point.get("key_changes", [])
        ))

    return BehaviorProfileResponse(
        user_id=profile.user_id,
        strengths=profile_data.get("strengths", []),
        weaknesses=profile_data.get("weaknesses", []),
        bias_patterns=bias_patterns,
        decision_style=profile_data.get("decision_style", "unknown"),
        stress_response=profile_data.get("stress_response", "unknown"),
        overall_score=calculate_overall_score(profile_data),
        improvement_trajectory=trajectory,
        total_simulations_analyzed=profile.total_simulations_analyzed,
        last_updated=profile.last_updated
    )


@router.get("/summary", response_model=ProfileSummary, summary="Dashboard profile summary")
async def get_profile_summary(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Get quick profile summary for dashboard."""
    profile = db.query(BehaviorProfile).filter(
        BehaviorProfile.user_id == current_user.id
    ).first()

    # Count simulations this week
    week_ago = datetime.utcnow() - timedelta(days=7)
    simulations_this_week = db.query(Simulation).filter(
        Simulation.user_id == current_user.id,
        Simulation.started_at >= week_ago,
        Simulation.status == "completed"
    ).count()

    if not profile or not profile.profile_data:
        return ProfileSummary(
            overall_score=0,
            top_strength=None,
            top_weakness=None,
            recent_trend="stable",
            simulations_this_week=simulations_this_week,
            current_streak=current_user.current_streak or 0
        )

    profile_data = profile.profile_data
    strengths = profile_data.get("strengths", [])
    weaknesses = profile_data.get("weaknesses", [])

    return ProfileSummary(
        overall_score=calculate_overall_score(profile_data),
        top_strength=strengths[0] if strengths else None,
        top_weakness=weaknesses[0] if weaknesses else None,
        recent_trend=calculate_trend(profile.improvement_trajectory or []),
        simulations_this_week=simulations_this_week,
        current_streak=current_user.current_streak or 0
    )


@router.get("/history", summary="Simulation history with scores")
async def get_simulation_history(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    limit: int = 20
):
    """Get user's simulation history with scores."""
    simulations = db.query(Simulation).filter(
        Simulation.user_id == current_user.id,
        Simulation.status == "completed"
    ).order_by(Simulation.completed_at.desc()).limit(limit).all()

    return [
        {
            "id": s.id,
            "scenario_name": s.scenario.name,
            "completed_at": s.completed_at,
            "profit_loss": s.final_outcome.get("profit_loss", 0) if s.final_outcome else 0,
            "process_quality_score": s.process_quality_score
        }
        for s in simulations
    ]


@router.get("/playbook", summary="Get personal do/don't playbook")
async def get_playbook(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Get or generate the user's personal do/don't playbook."""
    profile = db.query(BehaviorProfile).filter(
        BehaviorProfile.user_id == current_user.id
    ).first()

    if not profile:
        raise HTTPException(status_code=404, detail="No behavior profile yet. Complete a simulation first.")

    profile_data = profile.profile_data or {}
    playbook = profile_data.get("playbook")

    if playbook:
        return playbook

    # Generate via Gemini or heuristic
    gemini = GeminiService()
    playbook = await gemini.generate_playbook(profile_data)

    # Store in profile
    profile_data["playbook"] = playbook
    profile.profile_data = profile_data
    db.commit()

    return playbook


@router.post("/playbook/track", summary="Check playbook adherence for a simulation")
async def track_playbook_adherence(
    simulation_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Check how well the user followed their playbook in a simulation."""
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id,
        Simulation.status == "completed",
    ).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Completed simulation not found")

    decisions = db.query(Decision).filter(
        Decision.simulation_id == simulation_id
    ).order_by(Decision.simulation_time).all()

    profile = db.query(BehaviorProfile).filter(
        BehaviorProfile.user_id == current_user.id
    ).first()

    playbook = (profile.profile_data or {}).get("playbook", {}) if profile else {}
    if not playbook:
        return {"adherence_score": None, "message": "No playbook generated yet"}

    gemini = GeminiService()
    return await gemini.check_playbook_adherence(playbook, decisions, simulation)


def get_bias_description(bias_name: str) -> str:
    """Get description for a bias pattern."""
    descriptions = {
        "loss_aversion": "Tendency to prefer avoiding losses over acquiring equivalent gains",
        "fomo": "Fear of missing out - making impulsive decisions to not miss opportunities",
        "anchoring": "Over-relying on the first piece of information encountered",
        "social_proof_reliance": "Following the crowd instead of independent analysis",
        "overconfidence": "Excessive confidence in one's own predictions and abilities",
        "recency_bias": "Giving more weight to recent events than historical data",
        "confirmation_bias": "Seeking information that confirms existing beliefs",
        "impulsivity": "Making quick decisions without adequate analysis"
    }
    return descriptions.get(bias_name, "No description available")


def calculate_overall_score(profile_data: dict) -> float:
    """Calculate overall profile score from 0-100."""
    if not profile_data:
        return 0

    bias_patterns = profile_data.get("bias_patterns", {})
    if not bias_patterns:
        return 0  # No score until first simulation analyzed

    # Lower bias scores = higher overall score
    avg_bias = sum(bias_patterns.values()) / len(bias_patterns) if bias_patterns else 0.5
    base_score = (1 - avg_bias) * 100

    # Bonus for strengths
    strengths = profile_data.get("strengths", [])
    bonus = min(len(strengths) * 5, 20)

    return min(base_score + bonus, 100)


def calculate_trend(trajectory: list) -> str:
    """Calculate recent trend from improvement trajectory."""
    if len(trajectory) < 2:
        return "stable"

    recent = trajectory[-3:] if len(trajectory) >= 3 else trajectory
    scores = [p.get("score", 50) if isinstance(p, dict) else p.overall_score for p in recent]

    if len(scores) < 2:
        return "stable"

    diff = scores[-1] - scores[0]

    if diff > 5:
        return "improving"
    elif diff < -5:
        return "declining"
    return "stable"
