from pydantic import BaseModel, ConfigDict
from beanie import PydanticObjectId
from datetime import datetime
from typing import List

class NotificationResponseSchema(BaseModel):
    id: PydanticObjectId
    project_id: str
    message: str
    created_by_name: str
    read_by_user_ids: List[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
