from pydantic import BaseModel, Field, ConfigDict
from beanie import PydanticObjectId
from datetime import datetime


class CommentCreateSchema(BaseModel):
    task_id: str = Field(..., description="The task this comment belongs to")
    content: str = Field(..., min_length=1, max_length=5000)


class CommentResponseSchema(BaseModel):
    id: PydanticObjectId
    task_id: str
    author_id: str
    author_name: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
