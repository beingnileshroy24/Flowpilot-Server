from datetime import datetime, timezone
from beanie import Document
from pydantic import BaseModel, Field

from typing import Optional, List

class Milestone(BaseModel):
    id: str
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = None
    status: str = "PLANNED"  # PLANNED, IN_PROGRESS, COMPLETED, CANCELLED

class Release(BaseModel):
    id: str
    version: str
    release_date: Optional[str] = None
    status: str = "DRAFT"  # DRAFT, STAGING, PRODUCTION
    notes: Optional[str] = ""

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
    tech_stack: List[str] = []
    requirements: Optional[str] = ""
    milestones: List[Milestone] = []
    releases: List[Release] = []
    retrospective: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "projects"

