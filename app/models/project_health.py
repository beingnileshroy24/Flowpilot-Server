from datetime import datetime, timezone
from beanie import Document
from pydantic import Field
from typing import List, Optional, Dict, Any

class ProjectHealth(Document):
    project_id: str
    health_score: float = 100.0
    status: str = "HEALTHY"
    active_sprint_id: Optional[str] = None
    task_delay_risks: List[Dict[str, Any]] = []
    assignee_burnout_risks: List[Dict[str, Any]] = []
    explanation: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "project_health"
