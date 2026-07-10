from pydantic import BaseModel, ConfigDict
from beanie import PydanticObjectId
from datetime import datetime
from typing import Optional


class ActivityLogResponseSchema(BaseModel):
    id: PydanticObjectId
    task_id: Optional[str] = None
    project_id: str
    user_id: str
    user_name: str
    action: str
    detail: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
