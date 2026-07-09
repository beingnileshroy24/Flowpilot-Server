from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.config import settings
from app.models.user import User
from app.models.task import Task
from app.models.project import Project
from app.models.notification import Notification

async def init_db():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    await init_beanie(
        database=client[settings.DATABASE_NAME],  # type: ignore
        document_models=[
            User,
            Task,
            Project,
            Notification,
        ]
    )
