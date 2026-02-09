import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from database import Base


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    difficulty = Column(Integer, nullable=False)  # 1-5
    category = Column(String(100), nullable=False)  # crypto_hype, market_crash, fomo_trap, etc.

    # JSON fields for complex data
    initial_data = Column(JSONB, nullable=False)  # price history, news, social signals
    events = Column(JSONB, nullable=False)  # timed events during simulation

    time_pressure_seconds = Column(Integer, nullable=False, default=180)
    unlock_requirements = Column(JSONB, nullable=True)  # {min_simulations, min_process_score, ...}
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)  # 1=active, 0=inactive

    # AI-generated scenario for a specific user (Phase 4: Adaptive Scenario Generator)
    generated_for_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Relationships
    simulations = relationship("Simulation", back_populates="scenario")
