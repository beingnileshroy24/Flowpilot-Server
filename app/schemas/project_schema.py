from pydantic import BaseModel, Field, ConfigDict
from beanie import PydanticObjectId
from datetime import datetime
from typing import Optional, List


class MilestoneSchema(BaseModel):
    id: str
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = None
    status: str = "PLANNED"  # PLANNED, IN_PROGRESS, COMPLETED, CANCELLED


class ReleaseSchema(BaseModel):
    id: str
    version: str
    release_date: Optional[str] = None
    status: str = "DRAFT"  # DRAFT, STAGING, PRODUCTION
    notes: Optional[str] = ""
    linked_task_ids: List[str] = Field(default_factory=list)
    checklist: List[dict] = Field(default_factory=list)


class SprintSchema(BaseModel):
    id: str
    title: str
    goal: Optional[str] = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: str = "PLANNING"  # PLANNING, ACTIVE, COMPLETED, REVIEWED
    capacity_hours: float = 0.0


class ActionItemSchema(BaseModel):
    id: str
    text: str
    assignee_name: Optional[str] = ""
    done: bool = False


class RetroEntrySchema(BaseModel):
    id: str
    sprint_id: Optional[str] = None
    sprint_title: Optional[str] = ""
    went_well: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    action_items: List[ActionItemSchema] = Field(default_factory=list)
    created_at: Optional[str] = None


class DecisionEntrySchema(BaseModel):
    id: str
    title: str
    context: Optional[str] = ""
    decision: Optional[str] = ""
    alternatives: Optional[str] = ""
    decided_by: Optional[str] = ""
    decided_date: Optional[str] = None
    status: str = "PROPOSED"  # PROPOSED, ACCEPTED, SUPERSEDED, DEPRECATED


class ProjectCreateSchema(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: str = Field("", max_length=500)
    developer_ids: List[str] = Field(default_factory=list)
    lead_developer_id: Optional[str] = None
    github_frontend: Optional[str] = None
    github_backend: Optional[str] = None
    test_server: Optional[str] = None
    prod_server: Optional[str] = None
    test_mongodb_url: Optional[str] = None
    prod_mongodb_url: Optional[str] = None
    backend_secrets: Optional[str] = None
    frontend_secrets: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)
    requirements: Optional[str] = ""
    milestones: List[MilestoneSchema] = Field(default_factory=list)
    releases: List[ReleaseSchema] = Field(default_factory=list)
    retrospective: Optional[str] = ""
    sprints: List[SprintSchema] = Field(default_factory=list)
    decisions: List[DecisionEntrySchema] = Field(default_factory=list)
    retro_entries: List[RetroEntrySchema] = Field(default_factory=list)


class ProjectUpdateSchema(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    developer_ids: Optional[List[str]] = None
    lead_developer_id: Optional[str] = None
    github_frontend: Optional[str] = None
    github_backend: Optional[str] = None
    test_server: Optional[str] = None
    prod_server: Optional[str] = None
    test_mongodb_url: Optional[str] = None
    prod_mongodb_url: Optional[str] = None
    backend_secrets: Optional[str] = None
    frontend_secrets: Optional[str] = None
    tech_stack: Optional[List[str]] = None
    requirements: Optional[str] = None
    milestones: Optional[List[MilestoneSchema]] = None
    releases: Optional[List[ReleaseSchema]] = None
    retrospective: Optional[str] = None
    sprints: Optional[List[SprintSchema]] = None
    decisions: Optional[List[DecisionEntrySchema]] = None
    retro_entries: Optional[List[RetroEntrySchema]] = None


class ProjectResponseSchema(BaseModel):
    id: PydanticObjectId
    name: str
    description: str
    owner_id: str
    developer_ids: List[str]
    lead_developer_id: Optional[str] = None
    github_frontend: Optional[str] = None
    github_backend: Optional[str] = None
    test_server: Optional[str] = None
    prod_server: Optional[str] = None
    test_mongodb_url: Optional[str] = None
    prod_mongodb_url: Optional[str] = None
    backend_secrets: Optional[str] = None
    frontend_secrets: Optional[str] = None
    tech_stack: List[str]
    requirements: Optional[str] = ""
    milestones: List[MilestoneSchema]
    releases: List[ReleaseSchema]
    retrospective: Optional[str] = ""
    sprints: List[SprintSchema]
    decisions: List[DecisionEntrySchema]
    retro_entries: List[RetroEntrySchema]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
