import pytest
from fastapi.testclient import TestClient

def test_auth_and_user_endpoints(client: TestClient):
    # 1. Sign up a new user
    signup_payload = {
        "name": "Nilesh Roy",
        "email": "nilesh@example.com",
        "password": "securepassword123",
        "role": "MANAGER"
    }
    response = client.post("/api/v1/auth/signup", json=signup_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == signup_payload["name"]
    assert data["email"] == signup_payload["email"]
    assert data["role"] == "MANAGER"
    assert data["status"] == "ACTIVE"
    assert "id" in data
    assert "hashed_password" not in data

    # 2. Login with form-data credentials
    login_data = {
        "username": "nilesh@example.com",
        "password": "securepassword123"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    token = token_data["access_token"]

    # 3. Retrieve personal profile info with token
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/v1/users/me", headers=headers)
    assert response.status_code == 200
    profile_data = response.json()
    assert profile_data["email"] == "nilesh@example.com"
    assert profile_data["role"] == "MANAGER"

    # 4. List all registered users
    response = client.get("/api/v1/users/", headers=headers)
    assert response.status_code == 200
    users_list = response.json()
    assert len(users_list) == 1
    assert users_list[0]["email"] == "nilesh@example.com"

def test_task_operations_and_kanban_flow(client: TestClient):
    # Setup - Register manager and developer users
    client.post("/api/v1/auth/signup", json={
        "name": "Manager Bob",
        "email": "bob@example.com",
        "password": "bobpassword",
        "role": "MANAGER"
    })
    
    dev_response = client.post("/api/v1/auth/signup", json={
        "name": "Developer Alice",
        "email": "alice@example.com",
        "password": "alicepassword",
        "role": "DEVELOPER"
    })
    dev_id = dev_response.json()["id"]

    # Log in as Manager Bob to perform operations
    login_response = client.post("/api/v1/auth/login", data={
        "username": "bob@example.com",
        "password": "bobpassword"
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create a task and assign it to Developer Alice
    task_payload = {
        "project_id": "proj-100",
        "type": "BUG",
        "title": "Fix login button click crash",
        "description": "Clicking the login button twice in fast succession causes app to crash.",
        "status": "TODO",
        "priority": "HIGH",
        "assigned_to_id": dev_id,
        "estimated_hours": 3.5,
        "tags": ["frontend", "bug", "auth"]
    }
    
    response = client.post("/api/v1/tasks/", json=task_payload, headers=headers)
    assert response.status_code == 201
    task_data = response.json()
    assert task_data["title"] == task_payload["title"]
    assert task_data["project_id"] == "proj-100"
    assert task_data["status"] == "TODO"
    assert task_data["priority"] == "HIGH"
    assert task_data["assigned_to"]["id"] == dev_id
    assert task_data["assigned_to"]["name"] == "Developer Alice"
    task_id = task_data["id"]

    # 2. Get all tasks for Project ID: "proj-100"
    response = client.get("/api/v1/tasks/?project_id=proj-100", headers=headers)
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 1
    assert tasks[0]["id"] == task_id

    # 3. Retrieve single task details
    response = client.get(f"/api/v1/tasks/{task_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["title"] == "Fix login button click crash"

    # 4. Progress Task Status (Simulate dragging card to IN_PROGRESS on Kanban)
    response = client.patch(f"/api/v1/tasks/{task_id}/status?current_status=IN_PROGRESS", headers=headers)
    assert response.status_code == 200
    updated_task = response.json()
    assert updated_task["status"] == "IN_PROGRESS"

    # 5. Perform general updates (modify description and log actual hours)
    update_payload = {
        "description": "Updated crash notes: root cause identified as race condition.",
        "actual_hours": 4.0,
        "status": "IN_REVIEW"
    }
    response = client.patch(f"/api/v1/tasks/{task_id}", json=update_payload, headers=headers)
    assert response.status_code == 200
    final_task = response.json()
    assert final_task["description"] == update_payload["description"]
    assert final_task["actual_hours"] == 4.0
    assert final_task["status"] == "IN_REVIEW"
