import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from database import Base


class BehaviorProfile(Base):
    """AI-compressed memory of user's behavioral patterns."""
    __tablename__ = "behavior_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)

    # Compressed profile data
    profile_data = Column(JSONB, default=dict)
    # Example structure:
    # {
    #     "strengths": ["pattern_recognition", "risk_awareness"],
    #     "weaknesses": ["fomo_susceptibility", "overconfidence"],
    #     "bias_patterns": {
    #         "loss_aversion": 0.7,
    #         "anchoring": 0.4,
    #         "social_proof_reliance": 0.8,
    #         "impulsivity": 0.6
    #     },
    #     "decision_style": "reactive",  # or "analytical", "balanced"
    #     "stress_response": "impulsive"  # or "cautious", "steady"
    # }

    # Improvement tracking
    improvement_trajectory = Column(JSONB, default=list)
    # Example: [{"date": "2024-01-01", "score": 45}, {"date": "2024-01-15", "score": 52}]

    last_updated = Column(DateTime, default=datetime.utcnow)
    total_simulations_analyzed = Column(Integer, default=0)

    # Relationship
    user = relationship("User", back_populates="behavior_profile")


class LearningProgress(Base):
    """Tracks which learning cards the user has viewed."""
    __tablename__ = "learning_progress"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    card_id = Column(String(100), nullable=False)
    viewed_at = Column(DateTime, default=datetime.utcnow)
    marked_helpful = Column(Boolean, nullable=True)

    # Relationship
    user = relationship("User", back_populates="learning_progress")
