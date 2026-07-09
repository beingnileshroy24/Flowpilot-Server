from fastapi import APIRouter, Depends
from typing import List
from app.models.user import User
from app.schemas.user_schema import UserResponseSchema
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/users", tags=["Users"])

@router.get("/me", response_model=UserResponseSchema)
async def get_my_profile(current_user: User = Depends(get_current_user)):
    """Retrieves the profile of the currently authenticated user."""
    return current_user

@router.get("/", response_model=List[UserResponseSchema])
async def list_users(current_user: User = Depends(get_current_user)):
    """Lists all registered users (used for assigning tasks in the UI)."""
    users = await User.find_all().to_list()
    return users
