from pydantic import BaseModel, Field, ConfigDict
from beanie import PydanticObjectId
from typing import Optional, List
from datetime import datetime
from app.models.task import TaskType, TaskStatus, TaskPriority
from app.schemas.user_schema import UserResponseSchema

class TaskCreateSchema(BaseModel):
    project_id: str = Field(..., description="Project ID this task belongs to")
    type: TaskType = Field(TaskType.TASK)
    parent_id: Optional[str] = Field(None, description="Self-referencing parent task ID for subtasks")
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field("")
    status: TaskStatus = Field(TaskStatus.TODO)
    priority: TaskPriority = Field(TaskPriority.MEDIUM)
    assigned_to_id: Optional[str] = Field(None, description="User ID of the assignee")
    estimated_hours: float = Field(0.0, ge=0.0)
    tags: List[str] = Field(default_factory=list)
    attachment_url: Optional[str] = None

class TaskUpdateSchema(BaseModel):
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = None
    type: Optional[TaskType] = None
    parent_id: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assigned_to_id: Optional[str] = None
    estimated_hours: Optional[float] = Field(None, ge=0.0)
    actual_hours: Optional[float] = Field(None, ge=0.0)
    tags: Optional[List[str]] = None
    attachment_url: Optional[str] = None

class TaskResponseSchema(BaseModel):
    id: PydanticObjectId
    project_id: str
    type: TaskType
    parent_id: Optional[str] = None
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    assigned_to: Optional[UserResponseSchema] = None
    estimated_hours: float
    actual_hours: float
    tags: List[str]
    attachment_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
