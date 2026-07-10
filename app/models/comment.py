from datetime import datetime, timezone
from beanie import Document
from pydantic import Field


class Comment(Document):
    task_id: str
    author_id: str
    author_name: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "comments"
