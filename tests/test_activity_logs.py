import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient
from app.config import settings
from app.models.user import UserRole
from app.auth.utils import get_password_hash
from datetime import datetime, timezone, timedelta

def test_activity_logs_advanced_search(client: TestClient):
    # 1. Setup - create a Manager user and a Developer user in DB synchronously
    mongo_client = MongoClient(settings.MONGODB_URL)
    db = mongo_client[settings.DATABASE_NAME]

    manager_pw = get_password_hash("managerpass")
    dev_pw = get_password_hash("devpass")

    manager_id = "60c72b2f9b1d8b2bad123456"
    dev_id = "60c72b2f9b1d8b2bad654321"

    db["users"].insert_one({
        "_id": manager_id,
        "email": "manager@flowpilot.ai",
        "hashed_password": manager_pw,
        "name": "Manager Bob",
        "role": UserRole.MANAGER.value,
        "status": "ACTIVE",
        "created_at": datetime.now(timezone.utc)
    })

    db["users"].insert_one({
        "_id": dev_id,
        "email": "developer@flowpilot.ai",
        "hashed_password": dev_pw,
        "name": "Dev Alice",
        "role": UserRole.DEVELOPER.value,
        "status": "ACTIVE",
        "created_at": datetime.now(timezone.utc)
    })

    # Add activity logs with different dates/users/actions
    base_time = datetime.now(timezone.utc)
    
    # 3 logs
    db["activity_logs"].insert_many([
        {
            "task_id": "task-1",
            "project_id": "proj-1",
            "user_id": manager_id,
            "user_name": "Manager Bob",
            "action": "task_created",
            "detail": "Manager Bob created task-1 in proj-1",
            "created_at": base_time - timedelta(days=2)
        },
        {
            "task_id": "task-2",
            "project_id": "proj-1",
            "user_id": dev_id,
            "user_name": "Dev Alice",
            "action": "status_change",
            "detail": "Dev Alice changed status of task-2 to IN_PROGRESS",
            "created_at": base_time - timedelta(days=1)
        },
        {
            "task_id": "task-3",
            "project_id": "proj-2",
            "user_id": dev_id,
            "user_name": "Dev Alice",
            "action": "comment_added",
            "detail": "Dev Alice added comment: Looks good",
            "created_at": base_time
        }
    ])

    mongo_client.close()

    # 2. Log in as Manager
    login_resp = client.post("/api/v1/auth/login", data={"username": "manager@flowpilot.ai", "password": "managerpass"})
    assert login_resp.status_code == 200
    manager_token = login_resp.json()["access_token"]
    manager_headers = {"Authorization": f"Bearer {manager_token}"}

    # 3. Test global advanced search: user_id
    resp = client.get(f"/api/v1/activity/?user_id={dev_id}", headers=manager_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert all(d["user_id"] == dev_id for d in data)

    # 4. Test global advanced search: user_name (regex case insensitive)
    resp = client.get("/api/v1/activity/?user_name=bob", headers=manager_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert all(d["user_name"] == "Manager Bob" for d in data)

    # 5. Test global advanced search: action
    resp = client.get("/api/v1/activity/?action=status_change", headers=manager_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["action"] == "status_change"

    # 6. Test global advanced search: query (should match detail)
    resp = client.get("/api/v1/activity/?query=looks%20good", headers=manager_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert "Looks good" in data[0]["detail"]

    # 7. Test global advanced search: start_date & end_date
    start = (base_time - timedelta(days=1, hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (base_time - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = client.get(f"/api/v1/activity/?start_date={start}&end_date={end}", headers=manager_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["task_id"] == "task-2"

    # 8. Log in as Developer
    login_resp = client.post("/api/v1/auth/login", data={"username": "developer@flowpilot.ai", "password": "devpass"})
    assert login_resp.status_code == 200
    dev_token = login_resp.json()["access_token"]
    dev_headers = {"Authorization": f"Bearer {dev_token}"}

    # 9. Non-privileged user: can query scoping to a project
    resp = client.get("/api/v1/activity/?project_id=proj-1", headers=dev_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    # 10. Non-privileged user: can query scoping to themselves
    resp = client.get(f"/api/v1/activity/?user_id={dev_id}", headers=dev_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 2
    assert all(d["user_id"] == dev_id for d in data)

    # 11. Non-privileged user: Access denied for global queries without project or self-filter
    resp = client.get("/api/v1/activity/", headers=dev_headers)
    assert resp.status_code == 403
