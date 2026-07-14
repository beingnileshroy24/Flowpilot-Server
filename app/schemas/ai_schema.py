from pydantic import BaseModel, Field
from typing import List, Optional

class DuplicateCheckRequest(BaseModel):
    project_id: str
    title: str = Field(..., min_length=1)
    description: str

class DuplicateMatch(BaseModel):
    task_id: str
    title: str
    status: str
    similarity: float

class DuplicateCheckResponse(BaseModel):
    is_potential_duplicate: bool
    max_similarity_score: float = 0.0
    matches: List[DuplicateMatch]
