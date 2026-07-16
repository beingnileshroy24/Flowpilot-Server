from datetime import datetime, timezone
from typing import List, Optional
from beanie import Document
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    sender: str  # "user" or "bot"
    text: str
    thoughts: Optional[str] = ""
    sources: Optional[List[dict]] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CopilotChat(Document):
    project_id: str
    user_id: str
    title: str
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "copilot_chats"
