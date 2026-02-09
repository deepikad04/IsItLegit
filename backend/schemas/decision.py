from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional, Any


class InfoViewed(BaseModel):
    """Track what information the user viewed."""
    panel: str  # "price_chart", "news", "social", "portfolio"
    view_duration_seconds: float
    timestamp: int  # simulation time


class DecisionCreate(BaseModel):
    """Request to log a decision."""
    decision_type: str = Field(..., pattern="^(buy|sell|hold|wait)$")
    asset: Optional[str] = None
    amount: Optional[float] = Field(None, ge=0)
    confidence_level: Optional[int] = Field(None, ge=1, le=5)
    time_spent_seconds: Optional[float] = None
    rationale: Optional[str] = Field(None, max_length=500, description="Why are you making this decision?")
    info_viewed: Optional[list[InfoViewed]] = None
    order_type: str = Field("market", pattern="^(market|limit|stop)$")
    limit_price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    time_elapsed: Optional[int] = Field(None, ge=0, description="Client-reported simulation time in seconds")


class DecisionResponse(BaseModel):
    id: UUID
    simulation_id: UUID
    timestamp: datetime
    simulation_time: int
    decision_type: str
    asset: Optional[str]
    amount: Optional[float]
    confidence_level: Optional[int]
    price_at_decision: Optional[float]
    rationale: Optional[str] = None
    market_state_at_decision: Optional[dict[str, Any]]

    class Config:
        from_attributes = True


class DecisionSummary(BaseModel):
    """Summary of decisions for reflection."""
    total_decisions: int
    buy_count: int
    sell_count: int
    hold_count: int
    average_confidence: Optional[float]
    average_decision_time: Optional[float]
    most_viewed_info: list[str]
    least_viewed_info: list[str]
