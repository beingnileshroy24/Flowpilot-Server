from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.models.notification import Notification
from app.models.project import Project
from app.models.user import User, UserRole
from app.schemas.notification_schema import NotificationResponseSchema
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/notifications", tags=["Notifications"])

@router.get("/", response_model=List[NotificationResponseSchema])
async def list_notifications(current_user: User = Depends(get_current_user)):
    """Lists notifications relevant to the current user's assigned projects."""
    if current_user.role in [UserRole.MANAGER, UserRole.ADMIN]:
        # Managers and Admins get all notifications
        notifications = await Notification.find_all().sort("-created_at").to_list()
    else:
        # Developers get notifications for projects they are assigned to
        user_id_str = str(current_user.id)
        assigned_projects = await Project.find({"developer_ids": user_id_str}).to_list()
        project_ids = [str(p.id) for p in assigned_projects]
        
        notifications = await Notification.find(
            {"project_id": {"$in": project_ids}}
        ).sort("-created_at").to_list()
        
    return notifications

@router.post("/{notification_id}/read", response_model=NotificationResponseSchema)
async def mark_as_read(notification_id: str, current_user: User = Depends(get_current_user)):
    """Marks a notification as read/dismissed by adding the user ID to the list."""
    notification = await Notification.get(notification_id)
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )
    
    user_id_str = str(current_user.id)
    if user_id_str not in notification.read_by_user_ids:
        notification.read_by_user_ids.append(user_id_str)
        await notification.save()
        
    return notification
