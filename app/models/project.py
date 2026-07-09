from datetime import datetime, timezone
from beanie import Document
from pydantic import Field

from typing import Optional, List

class Project(Document):
    name: str
    description: str
    owner_id: str  # User ID of creator
    developer_ids: List[str] = []
    lead_developer_id: Optional[str] = None
    github_frontend: Optional[str] = None
    github_backend: Optional[str] = None
    test_server: Optional[str] = None
    prod_server: Optional[str] = None
    test_mongodb_url: Optional[str] = None
    prod_mongodb_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "projects"
