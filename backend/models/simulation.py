import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from database import Base


class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    scenario_id = Column(UUID(as_uuid=True), ForeignKey("scenarios.id"), nullable=False)

    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True, index=True)

    # Status: pending, in_progress, completed, abandoned
    status = Column(String(50), default="pending", index=True)

    # Simulation state
    current_portfolio = Column(JSONB, default=dict)  # cash, holdings, etc.
    current_time_elapsed = Column(Integer, default=0)  # seconds elapsed

    # Final results (populated on completion)
    final_outcome = Column(JSONB, nullable=True)  # profit/loss, final portfolio
    process_quality_score = Column(Float, nullable=True)  # 0-100

    # Gemini analysis (populated after completion)
    gemini_analysis = Column(JSONB, nullable=True)
    counterfactuals = Column(JSONB, nullable=True)
    gemini_cache = Column(JSONB, nullable=True, default=dict)  # Cache for why, pro_comparison, coaching, bias_heatmap, rationale_review

    # Relationships
    user = relationship("User", back_populates="simulations")
    scenario = relationship("Scenario", back_populates="simulations")
    decisions = relationship("Decision", back_populates="simulation", cascade="all, delete-orphan")
    snapshots = relationship("SimulationSnapshot", back_populates="simulation", cascade="all, delete-orphan")
