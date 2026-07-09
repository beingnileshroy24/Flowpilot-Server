from datetime import datetime, timezone
from beanie import Document
from pydantic import Field
from typing import List

class Notification(Document):
    project_id: str
    message: str
    created_by_name: str
    read_by_user_ids: List[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "notifications"
