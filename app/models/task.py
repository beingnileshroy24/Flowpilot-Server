from enum import Enum
from datetime import datetime, timezone
from typing import Optional, List
from beanie import Document
from pydantic import BaseModel, Field

class TaskType(str, Enum):
    EPIC = "EPIC"
    TASK = "TASK"
    SUBTASK = "SUBTASK"
    BUG = "BUG"

class TaskStatus(str, Enum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    IN_REVIEW = "IN_REVIEW"
    DONE = "DONE"

class TaskPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ChecklistItem(BaseModel):
    text: str
    done: bool = False

class Task(Document):
    project_id: str
    type: TaskType
    parent_id: Optional[str] = None  # Self-referencing ID for SUBTASK or BUG hierarchy
    title: str
    description: str
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_to_id: Optional[str] = None  # Direct string representation of User ID
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    dependency_ids: List[str] = []
    blocked_hours: float = 0.0
    tags: List[str] = []
    attachment_url: Optional[str] = None
    due_date: Optional[str] = None
    sprint_id: Optional[str] = None
    release_id: Optional[str] = None
    checklist_items: List[ChecklistItem] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "tasks"
