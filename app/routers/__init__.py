from app.routers.auth_router import router as auth_router
from app.routers.user_router import router as user_router
from app.routers.task_router import router as task_router
from app.routers.project_router import router as project_router

__all__ = [
    "auth_router",
    "user_router",
    "task_router",
    "project_router",
]
