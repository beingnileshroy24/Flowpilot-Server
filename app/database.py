import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4', '1.1.1.1']

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.config import settings
from app.models.user import User
from app.models.task import Task
from app.models.project import Project
from app.models.notification import Notification
from app.models.comment import Comment
from app.models.activity_log import ActivityLog
from app.models.copilot_chat import CopilotChat

async def init_db():
    client = AsyncIOMotorClient(settings.MONGODB_URL, tz_aware=True)
    await init_beanie(
        database=client[settings.DATABASE_NAME],  # type: ignore
        document_models=[
            User,
            Task,
            Project,
            Notification,
            Comment,
            ActivityLog,
            CopilotChat,
        ]
    )
