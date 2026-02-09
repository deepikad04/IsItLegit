import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from database import Base


class SimulationSnapshot(Base):
    """Normalized storage for large simulation state snapshots."""
    __tablename__ = "simulation_snapshots"
    __table_args__ = (
        Index('ix_snapshots_sim_type', 'simulation_id', 'snapshot_type'),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simulation_id = Column(UUID(as_uuid=True), ForeignKey("simulations.id"), nullable=False, index=True)
    simulation_time = Column(Integer, nullable=False)
    snapshot_type = Column(String(50), nullable=False)  # "decision_context", "completion"
    data = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    simulation = relationship("Simulation", back_populates="snapshots")
