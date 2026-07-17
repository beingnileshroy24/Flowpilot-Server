from datetime import datetime, timezone
from beanie import Document
from pydantic import Field
from typing import List, Optional

class SprintPrediction(Document):
    project_id: str
    sprint_id: str
    historical_velocity_drift: float = 0.0
    unplanned_scope_creep_points: float = 0.0
    comment_sentiment_volatility: float = 0.0
    failure_rate: float = 0.0
    burndown_trajectory: List[float] = []
    task_completion_max: float = 0.0
    explanation: Optional[str] = ""
    risk_score: float = 0.0
    status: str = "HEALTHY"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "sprint_predictions"
