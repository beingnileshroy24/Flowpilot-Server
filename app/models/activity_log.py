from datetime import datetime, timezone
from beanie import Document
from pydantic import Field
from typing import Optional


class ActivityLog(Document):
    task_id: Optional[str] = None
    project_id: str
    user_id: str
    user_name: str
    action: str  # e.g. "status_change", "assignment_change", "comment_added", "task_created"
    detail: str  # e.g. "Changed status from TODO to IN_PROGRESS"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "activity_logs"
