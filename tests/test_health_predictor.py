import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timezone

def test_health_predictor_flow(client: TestClient):
    # 1. Sign up manager
    signup_payload = {
        "name": "Project Lead Nilesh",
        "email": "lead@example.com",
        "password": "securepassword123",
        "role": "MANAGER"
    }
    response = client.post("/api/v1/auth/signup", json=signup_payload)
    assert response.status_code == 201
    lead_id = response.json()["id"]

    # 2. Login
    login_data = {
        "username": "lead@example.com",
        "password": "securepassword123"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Create a Project
    project_payload = {
        "name": "Next-Gen AI Core",
        "description": "Building LLM agent pipeline",
        "lead_developer_id": lead_id,
        "developer_ids": [lead_id],
        "sprints": [
            {
                "id": "sprint-1",
                "title": "Sprint 1: Architecture & Pipelines",
                "goal": "Get basic ML flow working",
                "start_date": datetime.now(timezone.utc).isoformat(),
                "end_date": datetime.now(timezone.utc).isoformat(),
                "status": "ACTIVE",
                "capacity_hours": 80.0
            }
        ],
        "infras": [],
        "secrets": [],
        "tech_stacks": ["Python", "FastAPI", "MongoDB"]
    }
    response = client.post("/api/v1/projects/", json=project_payload, headers=headers)
    assert response.status_code == 201
    project_id = response.json()["id"]

    # 4. Create tasks inside project active sprint
    task1_payload = {
        "project_id": project_id,
        "type": "TASK",
        "title": "Setup ONNX lightgbm regression model",
        "description": "Implement feature calculation and scoring",
        "status": "TODO",
        "priority": "HIGH",
        "assigned_to_id": lead_id,
        "estimated_hours": 8.0,
        "tags": ["ml-model"],
        "sprint_id": "sprint-1",
        "dependency_ids": [],
        "blocked_hours": 12.0
    }
    response = client.post("/api/v1/tasks/", json=task1_payload, headers=headers)
    assert response.status_code == 201
    task1_id = response.json()["id"]

    task2_payload = {
        "project_id": project_id,
        "type": "BUG",
        "title": "Fix context switching memory leak in MLX",
        "description": "Fix background thread resource exhaustion",
        "status": "IN_PROGRESS",
        "priority": "CRITICAL",
        "assigned_to_id": lead_id,
        "estimated_hours": 5.0,
        "tags": ["bugfix", "mlx"],
        "sprint_id": "sprint-1",
        "dependency_ids": [task1_id],
        "blocked_hours": 0.0
    }
    response = client.post("/api/v1/tasks/", json=task2_payload, headers=headers)
    assert response.status_code == 201

    # 5. Trigger predictions manually
    response = client.post(f"/api/v1/health/predict/{project_id}", headers=headers)
    assert response.status_code == 200
    health_data = response.json()
    assert "health_score" in health_data
    assert "status" in health_data
    assert len(health_data["task_delay_risks"]) > 0
    assert len(health_data["assignee_burnout_risks"]) > 0

    # 6. Fetch project health
    response = client.get(f"/api/v1/health/project/{project_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["project_id"] == project_id

    # 7. Fetch sprint prediction
    response = client.get(f"/api/v1/health/sprint/sprint-1", headers=headers)
    assert response.status_code == 200
    sprint_data = response.json()
    assert sprint_data["project_id"] == project_id
    assert sprint_data["sprint_id"] == "sprint-1"
    assert "failure_rate" in sprint_data
    assert len(sprint_data["burndown_trajectory"]) > 0

    # 8. Test SSE events endpoint
    response = client.get("/api/v1/health/events", headers=headers)
    # Stream endpoints can be tested by making a request and closing or reading headers
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
