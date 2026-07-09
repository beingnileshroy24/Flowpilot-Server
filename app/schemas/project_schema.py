from pydantic import BaseModel, Field, ConfigDict
from beanie import PydanticObjectId
from datetime import datetime
from typing import Optional, List

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
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
