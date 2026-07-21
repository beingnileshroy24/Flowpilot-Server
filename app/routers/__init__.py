from app.routers.auth_router import router as auth_router
from app.routers.user_router import router as user_router
from app.routers.task_router import router as task_router
from app.routers.project_router import router as project_router
from app.routers.notification_router import router as notification_router
from app.routers.comment_router import router as comment_router
from app.routers.activity_log_router import router as activity_log_router
from app.routers.ai_router import router as ai_router
from app.routers.copilot_router import router as copilot_router
from app.routers.health_router import router as health_router
from app.routers.planner_router import router as planner_router

__all__ = [
    "auth_router",
    "user_router",
    "task_router",
    "project_router",
    "notification_router",
    "comment_router",
    "activity_log_router",
    "ai_router",
    "copilot_router",
    "health_router",
    "planner_router",
]
