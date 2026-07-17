import pytest
import os
import asyncio

# Set test environment database name before loading any app modules
os.environ["DATABASE_NAME"] = "flowpilot_test"
os.environ["USE_MOCK_LLM"] = "True"

from fastapi.testclient import TestClient
from app.main import app
from app.models.user import User
from app.models.task import Task

@pytest.fixture(scope="function")
def client():
    """Function fixture that runs the app lifespan (connecting to MongoDB) and yields the test client."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="function", autouse=True)
def clean_database():
    """Autouse fixture that deletes all users and tasks before each test function using a synchronous pymongo connection."""
    from pymongo import MongoClient
    from app.config import settings
    
    client = MongoClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    db["users"].delete_many({})
    db["tasks"].delete_many({})
    db["projects"].delete_many({})
    db["copilot_chats"].delete_many({})
    db["sprint_predictions"].delete_many({})
    db["project_health"].delete_many({})
    client.close()
