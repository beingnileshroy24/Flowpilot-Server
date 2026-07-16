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
    linked_task_ids: List[str] = []
    checklist: List[dict] = []  # [{text: str, done: bool}]


class Sprint(BaseModel):
    id: str
    title: str
    goal: Optional[str] = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: str = "PLANNING"  # PLANNING, ACTIVE, COMPLETED, REVIEWED
    capacity_hours: float = 0.0


class ActionItem(BaseModel):
    id: str
    text: str
    assignee_name: Optional[str] = ""
    done: bool = False


class RetroEntry(BaseModel):
    id: str
    sprint_id: Optional[str] = None
    sprint_title: Optional[str] = ""
    went_well: List[str] = []
    improvements: List[str] = []
    action_items: List[ActionItem] = []
    created_at: Optional[str] = None


class DecisionEntry(BaseModel):
    id: str
    title: str
    context: Optional[str] = ""
    decision: Optional[str] = ""
    alternatives: Optional[str] = ""
    decided_by: Optional[str] = ""
    decided_date: Optional[str] = None
    status: str = "PROPOSED"  # PROPOSED, ACCEPTED, SUPERSEDED, DEPRECATED


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
    backend_secrets: Optional[str] = None
    frontend_secrets: Optional[str] = None
    tech_stack: List[str] = []
    requirements: Optional[str] = ""
    milestones: List[Milestone] = []
    releases: List[Release] = []
    retrospective: Optional[str] = ""
    sprints: List[Sprint] = []
    decisions: List[DecisionEntry] = []
    retro_entries: List[RetroEntry] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "projects"
