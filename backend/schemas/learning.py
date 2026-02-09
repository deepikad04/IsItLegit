from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


class LearningCard(BaseModel):
    """A single learning card."""
    id: str
    title: str
    content: str
    category: str  # fomo, risk_management, emotional_control, etc.
    difficulty: int = Field(..., ge=1, le=3)  # 1=beginner, 2=intermediate, 3=advanced
    example: Optional[str] = None
    key_takeaway: str
    related_patterns: list[str]  # which bias patterns this addresses


class LearningCardFeedback(BaseModel):
    """User feedback on a learning card."""
    card_id: str
    marked_helpful: bool


class LearningProgressResponse(BaseModel):
    """User's learning progress summary."""
    total_cards_viewed: int
    cards_marked_helpful: int
    categories_covered: list[str]
    recommended_next: list[str]  # card IDs


class PersonalizedCards(BaseModel):
    """Personalized card recommendations."""
    cards: list[LearningCard]
    reason: str  # why these cards were selected
    based_on_patterns: list[str]  # which detected patterns drove selection
