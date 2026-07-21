from enum import Enum
from datetime import datetime, timezone
from beanie import Document
from pydantic import EmailStr, Field

class UserRole(str, Enum):
    ADMIN = "ADMIN"
    MANAGER = "MANAGER"
    DEVELOPER = "DEVELOPER"
    CLIENT = "CLIENT"

class UserStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ON_LEAVE = "ON_LEAVE"
    DEPARTED = "DEPARTED"

class User(Document):
    name: str
    email: EmailStr
    hashed_password: str
    role: UserRole
    status: UserStatus = UserStatus.ACTIVE
    skills: list[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "users"  # Collection name
