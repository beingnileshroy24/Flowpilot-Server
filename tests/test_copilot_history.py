import pytest
from bson import ObjectId
from app.models.user import UserRole
from app.auth.utils import get_password_hash
from pymongo import MongoClient
from app.config import settings

@pytest.mark.anyio
async def test_copilot_chat_history_endpoints(client):
    # 1. Setup - use pymongo synchronously to avoid Beanie/Starlette loop conflicts
    mongo_client = MongoClient(settings.MONGODB_URL)
    db = mongo_client[settings.DATABASE_NAME]
    
    # Hash password correctly
    hashed_pw = get_password_hash("securepassword123")
    
    user_id = ObjectId()
    user_doc = {
        "_id": user_id,
        "email": "history_test@flowpilot.ai",
        "hashed_password": hashed_pw,
        "name": "History Reviewer",
        "role": UserRole.DEVELOPER.value,
        "status": "ACTIVE"
    }
    db["users"].insert_one(user_doc)
    
    # Create project
    proj_id = ObjectId()
    proj_doc = {
        "_id": proj_id,
        "name": "History Test Project",
        "description": "Verify Chat History",
        "owner_id": str(user_id),
        "developer_ids": [],
        "milestones": [],
        "releases": [],
        "retro_entries": [],
        "sprints": [],
        "decisions": []
    }
    db["projects"].insert_one(proj_doc)
    
    mongo_client.close()
    
    # Authenticate
    login_resp = client.post("/api/v1/auth/login", data={
        "username": "history_test@flowpilot.ai",
        "password": "securepassword123"
    })
    assert login_resp.status_code == 200, f"Auth login failed: {login_resp.json()}"
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    str_proj_id = str(proj_id)
    
    # 2. Verify List Chats is initially empty
    list_resp = client.get(f"/api/v1/copilot/chats?project_id={str_proj_id}", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 0
    
    # 3. Create a chat session
    create_payload = {
        "project_id": str_proj_id,
        "title": "Blockers review session"
    }
    create_resp = client.post("/api/v1/copilot/chats", json=create_payload, headers=headers)
    assert create_resp.status_code == 200
    chat_data = create_resp.json()
    assert chat_data["title"] == "Blockers review session"
    chat_id = chat_data["_id"]
    
    # 4. Verify list now has 1 chat
    list_resp = client.get(f"/api/v1/copilot/chats?project_id={str_proj_id}", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1
    assert list_resp.json()[0]["id"] == chat_id
    
    # 5. Query inside chat session (stream response)
    query_payload = {"prompt": "What are the blockers in Sprint 8?"}
    response = client.post(f"/api/v1/copilot/chats/{chat_id}/query", json=query_payload, headers=headers)
    assert response.status_code == 200
    # Consume the stream to trigger DB message append inside background task
    lines = list(response.iter_lines())
    assert len(lines) > 0
    
    # Check that at least some data: packets are emitted
    data_lines = [l for l in lines if l.startswith("data:")]
    assert len(data_lines) > 0
    
    # 6. Fetch details and verify history has been saved
    detail_resp = client.get(f"/api/v1/copilot/chats/{chat_id}", headers=headers)
    assert detail_resp.status_code == 200
    chat_details = detail_resp.json()
    assert len(chat_details["messages"]) == 2  # User query + Bot response
    assert chat_details["messages"][0]["sender"] == "user"
    assert chat_details["messages"][0]["text"] == "What are the blockers in Sprint 8?"
    assert chat_details["messages"][1]["sender"] == "bot"
    assert "sprint 8" in chat_details["messages"][1]["text"].lower() or "no relevant data" in chat_details["messages"][1]["text"].lower()
