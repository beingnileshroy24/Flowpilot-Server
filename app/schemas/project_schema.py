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
    tech_stack: List[str] = Field(default_factory=list)
    requirements: Optional[str] = ""
    milestones: List[MilestoneSchema] = Field(default_factory=list)
    releases: List[ReleaseSchema] = Field(default_factory=list)
    retrospective: Optional[str] = ""

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
    tech_stack: Optional[List[str]] = None
    requirements: Optional[str] = None
    milestones: Optional[List[MilestoneSchema]] = None
    releases: Optional[List[ReleaseSchema]] = None
    retrospective: Optional[str] = None

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
    tech_stack: List[str]
    requirements: Optional[str] = ""
    milestones: List[MilestoneSchema]
    releases: List[ReleaseSchema]
    retrospective: Optional[str] = ""
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

