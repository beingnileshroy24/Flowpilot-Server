from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.routers import auth_router, user_router, task_router, project_router, notification_router, comment_router, activity_log_router, ai_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database on startup
    await init_db()
    yield
    # Clean up on shutdown if necessary

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Configure CORS for integration with frontend React client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set origins restriction in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import os

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Wire up routers
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(task_router)
app.include_router(project_router)
app.include_router(notification_router)
app.include_router(comment_router)
app.include_router(activity_log_router)
app.include_router(ai_router)

@app.get("/")
async def root():
    """Welcome and API documentation quicklinks."""
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME} API",
        "docs": "/docs",
        "status": "healthy"
    }
