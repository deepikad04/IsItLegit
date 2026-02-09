import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from models.behavior_profile import BehaviorProfile, LearningProgress
from routers.auth import get_current_user
from schemas.learning import (
    LearningCard,
    LearningCardFeedback,
    LearningProgressResponse,
    PersonalizedCards
)
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning", tags=["learning"])

# Load learning cards from JSON
_learning_cards = None
_learning_modules = None


def get_learning_cards() -> list[dict]:
    global _learning_cards
    if _learning_cards is None:
        data_path = Path(__file__).parent.parent / "data" / "learning_cards.json"
        if data_path.exists():
            with open(data_path) as f:
                _learning_cards = json.load(f)
        else:
            _learning_cards = []
    return _learning_cards


def get_learning_modules() -> list[dict]:
    global _learning_modules
    if _learning_modules is None:
        data_path = Path(__file__).parent.parent / "data" / "learning_modules.json"
        if data_path.exists():
            with open(data_path) as f:
                _learning_modules = json.load(f)
        else:
            _learning_modules = []
    return _learning_modules


@router.get("/cards", response_model=PersonalizedCards)
async def get_personalized_cards(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    limit: int = 5
):
    """Get personalized learning cards based on behavior profile."""
    profile = db.query(BehaviorProfile).filter(
        BehaviorProfile.user_id == current_user.id
    ).first()

    all_cards = get_learning_cards()

    if not profile or not profile.profile_data.get("weaknesses"):
        # Return general beginner cards
        cards = [c for c in all_cards if c.get("difficulty", 1) == 1][:limit]
        return PersonalizedCards(
            cards=[LearningCard(**c) for c in cards],
            reason="Starting with foundational concepts",
            based_on_patterns=[]
        )

    # Get user's weaknesses and bias patterns
    weaknesses = profile.profile_data.get("weaknesses", [])
    bias_patterns = profile.profile_data.get("bias_patterns", {})

    # Get cards user hasn't seen recently
    viewed_ids = {
        lp.card_id for lp in db.query(LearningProgress).filter(
            LearningProgress.user_id == current_user.id
        ).all()
    }

    # Score cards by relevance to weaknesses
    scored_cards = []
    for card in all_cards:
        if card["id"] in viewed_ids:
            continue

        score = 0
        related = card.get("related_patterns", [])

        # Score based on weakness match
        for weakness in weaknesses:
            if weakness in related:
                score += 3

        # Score based on bias pattern scores
        for pattern, strength in bias_patterns.items():
            if pattern in related:
                score += strength * 2

        if score > 0:
            scored_cards.append((card, score))

    # Sort by score and take top cards
    scored_cards.sort(key=lambda x: x[1], reverse=True)
    selected = [c[0] for c in scored_cards[:limit]]

    # If not enough, add some general cards
    if len(selected) < limit:
        for card in all_cards:
            if card["id"] not in viewed_ids and card not in selected:
                selected.append(card)
                if len(selected) >= limit:
                    break

    return PersonalizedCards(
        cards=[LearningCard(**c) for c in selected],
        reason=f"Selected based on your patterns: {', '.join(weaknesses[:3])}",
        based_on_patterns=weaknesses
    )


@router.post("/cards/feedback")
async def submit_card_feedback(
    feedback: LearningCardFeedback,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Submit feedback on a learning card."""
    # Check if card exists
    all_cards = get_learning_cards()
    card_exists = any(c["id"] == feedback.card_id for c in all_cards)

    if not card_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Card not found"
        )

    # Update or create progress
    progress = db.query(LearningProgress).filter(
        LearningProgress.user_id == current_user.id,
        LearningProgress.card_id == feedback.card_id
    ).first()

    if progress:
        progress.marked_helpful = feedback.marked_helpful
    else:
        progress = LearningProgress(
            user_id=current_user.id,
            card_id=feedback.card_id,
            marked_helpful=feedback.marked_helpful
        )
        db.add(progress)

    db.commit()

    return {"message": "Feedback recorded"}


@router.get("/progress", response_model=LearningProgressResponse)
async def get_learning_progress(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Get user's learning progress summary."""
    progress = db.query(LearningProgress).filter(
        LearningProgress.user_id == current_user.id
    ).all()

    all_cards = get_learning_cards()

    # Get categories covered
    viewed_ids = {p.card_id for p in progress}
    categories = set()
    for card in all_cards:
        if card["id"] in viewed_ids:
            categories.add(card["category"])

    # Get helpful count
    helpful_count = sum(1 for p in progress if p.marked_helpful)

    # Get recommended next (cards not yet seen)
    not_seen = [c["id"] for c in all_cards if c["id"] not in viewed_ids]

    return LearningProgressResponse(
        total_cards_viewed=len(progress),
        cards_marked_helpful=helpful_count,
        categories_covered=list(categories),
        recommended_next=not_seen[:5]
    )


@router.get("/cards/all", response_model=list[LearningCard])
async def get_all_cards(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Get all learning cards (for browsing)."""
    return [LearningCard(**c) for c in get_learning_cards()]


@router.get("/modules")
async def list_modules(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Get all learning modules with user completion status. Uses Gemini to personalize when a behavior profile exists."""
    # Check if user has a behavior profile with weaknesses
    profile = db.query(BehaviorProfile).filter(
        BehaviorProfile.user_id == current_user.id
    ).first()

    if profile and profile.profile_data.get("weaknesses"):
        # Try Gemini-generated personalized modules
        try:
            gemini = GeminiService()
            generated = await gemini.generate_learning_modules(profile.profile_data)
            if generated:
                modules = generated
            else:
                modules = get_learning_modules()
        except Exception as e:
            logger.warning("Gemini learning modules failed, using static: %s", e)
            modules = get_learning_modules()
    else:
        modules = get_learning_modules()

    # Get completed modules from learning progress
    progress = db.query(LearningProgress).filter(
        LearningProgress.user_id == current_user.id
    ).all()
    completed_ids = {p.card_id for p in progress}  # card_id stores module_id for quiz completions

    result = []
    for mod in modules:
        result.append({
            **mod,
            "completed": mod["id"] in completed_ids,
            "lesson_count": len(mod.get("lessons", [])),
            "quiz_count": len(mod.get("quiz", [])),
        })

    return result


@router.get("/modules/{module_id}")
async def get_module(
    module_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Get a single module by ID with full content."""
    # Check static modules first
    modules = get_learning_modules()
    module = next((m for m in modules if m["id"] == module_id), None)
    if module:
        return module

    # Check Gemini-generated modules (cached) for gen_ prefixed IDs
    if module_id.startswith("gen_"):
        profile = db.query(BehaviorProfile).filter(
            BehaviorProfile.user_id == current_user.id
        ).first()
        if profile and profile.profile_data.get("weaknesses"):
            try:
                gemini = GeminiService()
                generated = await gemini.generate_learning_modules(profile.profile_data)
                module = next((m for m in generated if m["id"] == module_id), None)
                if module:
                    return module
            except Exception:
                pass

    raise HTTPException(status_code=404, detail="Module not found")


@router.post("/modules/{module_id}/complete")
async def complete_module_quiz(
    module_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Record that a user completed a module's quiz."""
    # Accept both static (mod_*) and generated (gen_*) module IDs
    if not module_id.startswith(("mod_", "gen_")):
        raise HTTPException(status_code=404, detail="Module not found")

    # Validate static modules exist
    if module_id.startswith("mod_"):
        modules = get_learning_modules()
        module = next((m for m in modules if m["id"] == module_id), None)
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")

    # Check if already recorded
    existing = db.query(LearningProgress).filter(
        LearningProgress.user_id == current_user.id,
        LearningProgress.card_id == module_id,
    ).first()

    if not existing:
        progress = LearningProgress(
            user_id=current_user.id,
            card_id=module_id,
            marked_helpful=True,
        )
        db.add(progress)
        db.commit()

    return {"message": "Module completed", "module_id": module_id}
