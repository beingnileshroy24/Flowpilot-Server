from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any, List, Optional
from datetime import datetime
from app.models.activity_log import ActivityLog
from app.models.user import User, UserRole
from app.schemas.activity_log_schema import ActivityLogResponseSchema
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/activity", tags=["Activity Logs"])


@router.get("/", response_model=List[ActivityLogResponseSchema])
async def get_activity_logs(
    task_id: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
    action: Optional[str] = None,
    query: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
):
    """
    Fetches activity logs with advanced search capabilities.
    Filter by task_id, project_id, user_id, user_name, action, datetime range, or general query.
    Returns most recent first, capped at limit.
    """
    # Authorization checks: managers/admins can query anything.
    # Standard developers/clients can query if scoped to a task, project, or their own user ID/user name.
    is_privileged = current_user.role in [UserRole.MANAGER, UserRole.ADMIN]
    is_filtering_by_target = bool(task_id or project_id)
    is_self_filter = bool(
        (user_id and user_id == str(current_user.id)) or
        (user_name and user_name.lower() == current_user.name.lower())
    )

    if not is_privileged and not is_filtering_by_target and not is_self_filter:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Non-privileged users can only view activity logs scoped to a task, project, or themselves."
        )

    filters: dict[str, Any] = {}
    if task_id:
        filters["task_id"] = task_id
    if project_id:
        filters["project_id"] = project_id
    if user_id:
        filters["user_id"] = user_id
    if action:
        filters["action"] = action

    if user_name:
        filters["user_name"] = {"$regex": user_name, "$options": "i"}

    if query:
        filters["$or"] = [
            {"user_name": {"$regex": query, "$options": "i"}},
            {"action": {"$regex": query, "$options": "i"}},
            {"detail": {"$regex": query, "$options": "i"}}
        ]

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
        filters["created_at"] = date_filter

    logs = await ActivityLog.find(filters).sort("-created_at").limit(limit).to_list()
    return logs
