import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_simulations = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    last_simulation_date = Column(DateTime, nullable=True)

    # Relationships
    simulations = relationship("Simulation", back_populates="user", cascade="all, delete-orphan")
    behavior_profile = relationship("BehaviorProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    learning_progress = relationship("LearningProgress", back_populates="user", cascade="all, delete-orphan")
