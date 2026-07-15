from pydantic import BaseModel, Field
from typing import List, Optional
from app.models.task import TaskType, TaskPriority

class WbsTaskCreate(BaseModel):
    title: str = Field(..., description="The title of the task")
    description: str = Field(..., description="Detailed description of the task")
    type: TaskType = Field(default=TaskType.TASK, description="Type of the task: EPIC, TASK, BUG, or SUBTASK")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Priority: LOW, MEDIUM, HIGH, CRITICAL")
    estimated_hours: float = Field(default=0.0, description="Estimated effort in hours")
    checklist_items: List[str] = Field(default_factory=list, description="Actionable checklist items")

class WbsCommitRequest(BaseModel):
    project_id: str = Field(..., description="The ID of the project")
    tasks: List[WbsTaskCreate] = Field(..., description="The list of tasks to create")
