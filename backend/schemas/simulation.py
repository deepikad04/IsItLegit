from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional, Any


class SimulationCreate(BaseModel):
    scenario_id: UUID


class SimulationState(BaseModel):
    """Current state of an active simulation."""
    id: UUID
    scenario_id: UUID
    scenario_name: str
    status: str
    time_elapsed: int  # seconds
    time_remaining: int  # seconds
    current_price: float
    price_history: list[float]
    portfolio: dict[str, Any]  # cash, holdings, total_value
    available_info: dict[str, Any]  # news, social signals, etc.
    recent_events: list[dict[str, Any]]
    market_conditions: Optional[dict[str, Any]] = None  # bid, ask, spread, vol, halt, crowd, macro
    last_execution: Optional[dict[str, Any]] = None  # fees_paid, execution_price, filled_amount, slippage, etc.
    pending_orders: Optional[list[dict[str, Any]]] = None  # limit/stop orders awaiting fill


class ScenarioBrief(BaseModel):
    id: UUID
    name: str
    category: str
    difficulty: int

    class Config:
        from_attributes = True


class SimulationResponse(BaseModel):
    id: UUID
    user_id: UUID
    scenario_id: UUID
    started_at: datetime
    completed_at: Optional[datetime]
    status: str
    process_quality_score: Optional[float]
    final_outcome: Optional[dict[str, Any]]
    scenario: Optional[ScenarioBrief] = None

    class Config:
        from_attributes = True


class SimulationComplete(BaseModel):
    """Request to complete a simulation."""
    final_decision: Optional[str] = None  # optional final action


class SimulationOutcome(BaseModel):
    """Result of a completed simulation."""
    simulation_id: UUID
    profit_loss: float
    profit_loss_percent: float
    final_portfolio_value: float
    process_quality_score: float
    total_decisions: int
    time_taken: int  # seconds
    outcome_type: str  # "profit", "loss", "break_even"
    outcome_summary: str  # dollar-formatted string, e.g. "+$150.00"
    total_fees_paid: Optional[float] = None
