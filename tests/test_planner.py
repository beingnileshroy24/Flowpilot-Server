import pytest
from fastapi.testclient import TestClient

def test_generate_sprint_plan(client: TestClient):
    # 1. Create Developer
    dev_response = client.post("/api/v1/auth/signup", json={
        "name": "Test Dev",
        "email": "planner_dev@example.com",
        "password": "devpassword",
        "role": "MANAGER"
    })
    assert dev_response.status_code == 201
    dev_id = dev_response.json()["id"]

    # 2. Login
    login_response = client.post("/api/v1/auth/login", data={
        "username": "planner_dev@example.com",
        "password": "devpassword"
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Create Project
    project_payload = {
        "name": "Planner Project",
        "description": "Test planner",
        "developer_ids": [dev_id],
        "lead_developer_id": dev_id
    }
    project_res = client.post("/api/v1/projects/", json=project_payload, headers=headers)
    assert project_res.status_code == 201
    project_id = project_res.json()["id"]

    # 4. Create tasks
    t1_payload = {
        "project_id": project_id,
        "type": "TASK",
        "title": "Task 1",
        "description": "Critical task",
        "priority": "CRITICAL",
        "estimated_hours": 5.0,
        "tags": ["python"]
    }
    t1_res = client.post("/api/v1/tasks/", json=t1_payload, headers=headers)
    assert t1_res.status_code == 201
    
    t2_payload = {
        "project_id": project_id,
        "type": "TASK",
        "title": "Task 2",
        "description": "Low priority task",
        "priority": "LOW",
        "estimated_hours": 40.0,
        "tags": ["python"]
    }
    t2_res = client.post("/api/v1/tasks/", json=t2_payload, headers=headers)
    assert t2_res.status_code == 201

    # 5. Trigger Planner API
    payload = {
        "project_id": project_id,
        "target_sprint_id": "sprint-123",
        "capacity_override": 10.0
    }
    response = client.post("/api/v1/planner/generate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "assigned_tasks" in data
    assert "dropped_tasks" in data
    assert "explanation" in data
    assert "solver_status" in data
    
    # T1 should be assigned (5h <= 10h), T2 dropped (40h > 10h remaining)
    assigned_titles = [t["title"] for t in data["assigned_tasks"]]
    dropped_titles = [t["title"] for t in data["dropped_tasks"]]
    
    assert "Task 1" in assigned_titles
    assert "Task 2" in dropped_titles
    assert data["solver_status"] in ["OPTIMAL", "FEASIBLE"]
