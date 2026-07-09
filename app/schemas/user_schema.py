from pydantic import BaseModel, EmailStr, Field, ConfigDict
from beanie import PydanticObjectId
from datetime import datetime
from app.models.user import UserRole, UserStatus

class UserSignupSchema(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    role: UserRole = UserRole.DEVELOPER

class UserLoginSchema(BaseModel):
    username: EmailStr = Field(..., alias="email")  # fastapi OAuth2 password flow uses "username"
    password: str

    model_config = ConfigDict(populate_by_name=True)

class UserResponseSchema(BaseModel):
    id: PydanticObjectId
    name: str
    email: EmailStr
    role: UserRole
    status: UserStatus
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class TokenResponseSchema(BaseModel):
    access_token: str
    token_type: str = "bearer"
