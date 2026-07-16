import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

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

    # 0. Create a project first
    project_payload = {
        "name": "Project 100",
        "description": "Flowpilot Main Project",
        "developer_ids": [dev_id],
        "lead_developer_id": dev_id
    }
    project_response = client.post("/api/v1/projects/", json=project_payload, headers=headers)
    assert project_response.status_code == 201
    project_data = project_response.json()
    project_id = project_data["id"]

    # 1. Create a task and assign it to Developer Alice
    task_payload = {
        "project_id": project_id,
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
    assert task_data["project_id"] == project_id
    assert task_data["status"] == "TODO"
    assert task_data["priority"] == "HIGH"
    assert task_data["assigned_to"]["id"] == dev_id
    assert task_data["assigned_to"]["name"] == "Developer Alice"
    task_id = task_data["id"]

    # 2. Get all tasks for Project ID
    response = client.get(f"/api/v1/tasks/?project_id={project_id}", headers=headers)
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


def test_duplicate_check_endpoint(client: TestClient):
    # 1. Test unauthorized duplicate check
    response = client.post("/api/v1/tasks/check-duplicates", json={
        "title": "Validate OAuth Token flow",
        "description": "Short description"
    })
    assert response.status_code == 401

    # 2. Login to get token
    signup_payload = {
        "name": "Tester Person",
        "email": "tester@example.com",
        "password": "testerpassword",
        "role": "DEVELOPER"
    }
    client.post("/api/v1/auth/signup", json=signup_payload)
    login_data = {
        "username": "tester@example.com",
        "password": "testerpassword"
    }
    login_response = client.post("/api/v1/auth/login", data=login_data)
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Test validation constraint: title too short (<= 5 characters)
    short_title_payload = {
        "title": "Short",  # 5 chars
        "description": "Valid description"
    }
    response = client.post("/api/v1/tasks/check-duplicates", json=short_title_payload, headers=headers)
    assert response.status_code == 422

    # 4. Test valid payload (since AI engine isn't running, it should handle connection error/timeout gracefully and return empty list)
    valid_payload = {
        "title": "My custom valid title checking",  # > 5 chars
        "description": "This is a detailed description of some random problem"
    }
    response = client.post("/api/v1/tasks/check-duplicates", json=valid_payload, headers=headers)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["is_potential_duplicate"] is False
    assert res_data["max_similarity_score"] == 0.0
    assert res_data["matches"] == []


@patch("app.services.sync_service.sync_task_event", new_callable=AsyncMock)
def test_task_mutations_and_sync_dispatcher(mock_sync, client: TestClient):
    # Register manager user (so they can perform create, update, delete)
    signup_payload = {
        "name": "Sync Manager",
        "email": "sync_mgr@example.com",
        "password": "mgrpassword",
        "role": "MANAGER"
    }
    client.post("/api/v1/auth/signup", json=signup_payload)
    login_response = client.post("/api/v1/auth/login", data={
        "username": "sync_mgr@example.com",
        "password": "mgrpassword"
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Register developer user to assign tasks to
    dev_response = client.post("/api/v1/auth/signup", json={
        "name": "Sync Dev",
        "email": "sync_dev@example.com",
        "password": "devpassword",
        "role": "DEVELOPER"
    })
    dev_id = dev_response.json()["id"]

    # Create a project first
    project_payload = {
        "name": "Project for Sync Test",
        "description": "Sync Verification Project",
        "developer_ids": [dev_id],
        "lead_developer_id": dev_id
    }
    project_res = client.post("/api/v1/projects/", json=project_payload, headers=headers)
    project_id = project_res.json()["id"]

    # 1. Test CREATE task event
    task_payload = {
        "project_id": project_id,
        "type": "TASK",
        "title": "Task for sync checks",
        "description": "Verify LanceDB sync dispatcher",
        "assigned_to_id": dev_id
    }
    res = client.post("/api/v1/tasks/", json=task_payload, headers=headers)
    assert res.status_code == 201
    task_id = res.json()["id"]
    
    assert mock_sync.call_count == 1
    assert mock_sync.call_args[0][0] == "create"
    created_task = mock_sync.call_args[0][1]
    assert created_task["id"] == task_id
    assert created_task["project_id"] == project_id
    assert created_task["title"] == "Task for sync checks"
    assert created_task["description"] == "Verify LanceDB sync dispatcher"
    assert created_task["assigned_to_id"] == dev_id
    
    # Reset mock call history
    mock_sync.reset_mock()

    # 2. Test STATUS UPDATE task event
    res = client.patch(f"/api/v1/tasks/{task_id}/status?current_status=IN_PROGRESS", headers=headers)
    assert res.status_code == 200
    assert mock_sync.call_count == 1
    assert mock_sync.call_args[0][0] == "update"
    assert mock_sync.call_args[0][1]["status"] == "IN_PROGRESS"

    mock_sync.reset_mock()

    # 3. Test GENERAL UPDATE task event
    update_payload = {
        "title": "Updated Task for sync checks",
        "description": "Updated description"
    }
    res = client.patch(f"/api/v1/tasks/{task_id}", json=update_payload, headers=headers)
    assert res.status_code == 200
    assert mock_sync.call_count == 1
    assert mock_sync.call_args[0][0] == "update"
    assert mock_sync.call_args[0][1]["title"] == "Updated Task for sync checks"
    assert mock_sync.call_args[0][1]["description"] == "Updated description"

    mock_sync.reset_mock()

    # 4. Test DELETE task event
    # First test unauthorized delete as developer (DEVELOPER role)
    dev_login = client.post("/api/v1/auth/login", data={
        "username": "sync_dev@example.com",
        "password": "devpassword"
    })
    dev_token = dev_login.json()["access_token"]
    dev_headers = {"Authorization": f"Bearer {dev_token}"}
    
    del_res = client.delete(f"/api/v1/tasks/{task_id}", headers=dev_headers)
    assert del_res.status_code == 403
    assert mock_sync.call_count == 0

    # Delete as Manager
    del_res = client.delete(f"/api/v1/tasks/{task_id}", headers=headers)
    assert del_res.status_code == 204
    assert mock_sync.call_count == 1
    assert mock_sync.call_args[0][0] == "delete"
    assert mock_sync.call_args[0][1]["id"] == task_id

    # Verify task is actually deleted from DB
    get_res = client.get(f"/api/v1/tasks/{task_id}", headers=headers)
    assert get_res.status_code == 404


def test_project_names_population(client: TestClient):
    # Setup - Register manager and developer users
    client.post("/api/v1/auth/signup", json={
        "name": "Manager Charlie",
        "email": "charlie@example.com",
        "password": "charliepassword",
        "role": "MANAGER"
    })
    
    dev_response = client.post("/api/v1/auth/signup", json={
        "name": "Developer Dave",
        "email": "dave@example.com",
        "password": "davepassword",
        "role": "DEVELOPER"
    })
    dev_id = dev_response.json()["id"]

    # Log in as Manager Charlie
    login_response = client.post("/api/v1/auth/login", data={
        "username": "charlie@example.com",
        "password": "charliepassword"
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create a project
    project_payload = {
        "name": "Project Charlie",
        "description": "Charlie's test project",
        "developer_ids": [dev_id],
        "lead_developer_id": dev_id
    }
    project_response = client.post("/api/v1/projects/", json=project_payload, headers=headers)
    assert project_response.status_code == 201
    project_data = project_response.json()
    assert project_data["owner_name"] == "Manager Charlie"
    assert project_data["developer_names"] == ["Developer Dave"]
    project_id = project_data["id"]

    # 2. Get project by ID and verify names are populated
    get_response = client.get(f"/api/v1/projects/{project_id}", headers=headers)
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["owner_name"] == "Manager Charlie"
    assert get_data["developer_names"] == ["Developer Dave"]

    # 3. List projects and verify names are populated
    list_response = client.get("/api/v1/projects/", headers=headers)
    assert list_response.status_code == 200
    list_data = list_response.json()
    project_in_list = next(p for p in list_data if p["id"] == project_id)
    assert project_in_list["owner_name"] == "Manager Charlie"
    assert project_in_list["developer_names"] == ["Developer Dave"]

    # 4. Update project and verify names are populated
    update_payload = {
        "description": "Updated description"
    }
    update_response = client.patch(f"/api/v1/projects/{project_id}", json=update_payload, headers=headers)
    assert update_response.status_code == 200
    update_data = update_response.json()
    assert update_data["owner_name"] == "Manager Charlie"
    assert update_data["developer_names"] == ["Developer Dave"]


