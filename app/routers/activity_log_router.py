from fastapi import APIRouter, Depends
from typing import List, Optional
from app.models.activity_log import ActivityLog
from app.models.user import User
from app.schemas.activity_log_schema import ActivityLogResponseSchema
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/activity", tags=["Activity Logs"])


@router.get("/", response_model=List[ActivityLogResponseSchema])
async def get_activity_logs(
    task_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """Fetches activity logs. Filter by task_id or project_id. Returns most recent first, capped at limit."""
    if task_id:
        logs = await ActivityLog.find(
            ActivityLog.task_id == task_id
        ).sort("-created_at").limit(limit).to_list()
    elif project_id:
        logs = await ActivityLog.find(
            ActivityLog.project_id == project_id
        ).sort("-created_at").limit(limit).to_list()
    else:
        # If no filter, return recent activity for all projects (admin/manager view)
        logs = await ActivityLog.find_all().sort("-created_at").limit(limit).to_list()

    return logs
