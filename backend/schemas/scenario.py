from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional, Any


class ScenarioEvent(BaseModel):
    time: int  # seconds into simulation
    type: str  # "price", "news", "social"
    content: Optional[str] = None
    change: Optional[float] = None  # for price events


class ScenarioInitialData(BaseModel):
    asset: str
    price: float
    your_balance: float
    market_sentiment: str  # bullish, bearish, neutral
    price_history: Optional[list[float]] = None
    news_headlines: Optional[list[str]] = None
    social_signals: Optional[list[dict]] = None


class ScenarioCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: str
    difficulty: int = Field(..., ge=1, le=5)
    category: str = Field(..., max_length=100)
    initial_data: dict[str, Any]
    events: list[dict[str, Any]]
    time_pressure_seconds: int = Field(default=180, ge=30, le=600)


class ScenarioResponse(BaseModel):
    id: UUID
    name: str
    description: str
    difficulty: int
    category: str
    time_pressure_seconds: int
    created_at: datetime
    is_locked: Optional[bool] = False
    unlock_requirements: Optional[dict[str, Any]] = None

    class Config:
        from_attributes = True


class ScenarioDetailResponse(ScenarioResponse):
    initial_data: dict[str, Any]
    events: list[dict[str, Any]]
