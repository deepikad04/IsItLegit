import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Float, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from database import Base


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = (
        Index('ix_decisions_sim_time', 'simulation_id', 'simulation_time'),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simulation_id = Column(UUID(as_uuid=True), ForeignKey("simulations.id"), nullable=False, index=True)

    timestamp = Column(DateTime, default=datetime.utcnow)
    simulation_time = Column(Integer, nullable=False)  # seconds into simulation

    # Decision details
    decision_type = Column(String(50), nullable=False)  # buy, sell, hold, wait
    asset = Column(String(100), nullable=True)  # which asset
    amount = Column(Float, nullable=True)  # how much
    price_at_decision = Column(Float, nullable=True)

    # User state at decision time
    confidence_level = Column(Integer, nullable=True)  # 1-5 user-reported
    time_spent_seconds = Column(Float, nullable=True)  # time deliberating

    # Information tracking
    info_viewed = Column(JSONB, default=list)  # what data points user looked at
    info_ignored = Column(JSONB, default=list)  # available but not viewed
    info_view_times = Column(JSONB, default=dict)  # time spent on each info panel

    # Market state snapshot (legacy — new decisions use snapshot_id reference)
    market_state_at_decision = Column(JSONB, nullable=True)

    # Normalized snapshot reference (deduplicates state blob storage)
    snapshot_id = Column(UUID(as_uuid=True), ForeignKey("simulation_snapshots.id"), nullable=True)

    # User's stated reasoning for the decision
    rationale = Column(Text, nullable=True)

    # Events that occurred since last decision
    events_since_last = Column(JSONB, default=list)

    # Relationships
    simulation = relationship("Simulation", back_populates="decisions")
    snapshot = relationship("SimulationSnapshot", foreign_keys=[snapshot_id])
