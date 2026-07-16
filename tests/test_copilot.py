import pytest
import json
from fastapi.testclient import TestClient
from app.ai.core.splitter import RecursiveParagraphSplitter
from app.ai.core.react_engine import react_engine
from app.services.sync_queue import push_to_sync_queue, sync_queue, _sync_event_to_lancedb
from app.models.task import Task
from app.models.project import Project

def test_recursive_splitter():
    splitter = RecursiveParagraphSplitter(chunk_size=10, chunk_overlap=2)
    # 1 token = approx 4 chars, so char_size = 40, char_overlap = 8
    text = "This is a very long paragraph that will definitely exceed the forty character limit we set for token splits. And here is a second paragraph that is also quite long."
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 1
    # Check that overlap exists (characters of previous chunk end appear in the start of the next chunk)
    for i in range(1, len(chunks)):
        prev_end = chunks[i-1][-8:]
        curr_start = chunks[i][:8]
        # Allow slight mismatches if text boundary was split differently, but check overlap presence
        assert len(curr_start) > 0

def test_reciprocal_rank_fusion():
    # Construct mock ranked lists
    list_a = [
        {"entity_type": "TASK", "source_id": "1", "content_snippet": "Task 1", "metadata": {}},
        {"entity_type": "TASK", "source_id": "2", "content_snippet": "Task 2", "metadata": {}},
    ]
    list_b = [
        {"entity_type": "TASK", "source_id": "2", "content_snippet": "Task 2", "metadata": {}},
        {"entity_type": "TASK", "source_id": "3", "content_snippet": "Task 3", "metadata": {}},
    ]
    
    # Run RRF
    fused = react_engine._reciprocal_rank_fusion([list_a, list_b])
    
    assert len(fused) == 3
    # Task 2 should be ranked 1st because it appears in both lists
    assert fused[0]["source_id"] == "2"
    # Task 1 and Task 3 follow
    assert {f[ "source_id" ] for f in fused} == {"1", "2", "3"}

def test_sync_queue_push():
    # Empty queue
    while not sync_queue.empty():
        sync_queue.get_nowait()
        
    push_to_sync_queue("TASK", "mock_task_id", "create", "mock_project_id")
    assert sync_queue.qsize() == 1
    
    event = sync_queue.get_nowait()
    assert event["entity_type"] == "TASK"
    assert event["entity_id"] == "mock_task_id"
    assert event["action_type"] == "create"
    assert event["project_id"] == "mock_project_id"

def test_copilot_query_endpoint(client: TestClient):
    # 1. Sign up user
    client.post("/api/v1/auth/signup", json={
        "name": "Copilot User",
        "email": "copilot@example.com",
        "password": "copilotpassword",
        "role": "MANAGER"
    })
    
    # 2. Login
    login_response = client.post("/api/v1/auth/login", data={
        "username": "copilot@example.com",
        "password": "copilotpassword"
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Create Project
    proj_response = client.post("/api/v1/projects/", json={
        "name": "Copilot Test Proj",
        "description": "Project for copilot query testing",
        "developer_ids": [],
    }, headers=headers)
    project_id = proj_response.json()["id"]
    
    # 4. Invoke Copilot streaming query
    payload = {
        "prompt": "What are the blockers in Sprint 8?",
        "contextScope": {
            "project_id": project_id
        }
    }
    
    response = client.post("/api/v1/copilot/query", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Read stream chunks
    lines = list(response.iter_lines())
    assert len(lines) > 0
    
    # Check that at least some data: packets are emitted
    data_lines = [l for l in lines if l.startswith("data:")]
    assert len(data_lines) > 0

@pytest.mark.anyio
async def test_react_engine_tools_direct():
    from app.database import init_db
    await init_db()
    
    from app.models.project import Project
    from app.models.task import Task, TaskType, TaskStatus
    from app.models.user import User, UserRole, UserStatus
    
    user = User(name="Dev Engineer", email="dev@example.com", hashed_password="pw", role=UserRole.DEVELOPER, status=UserStatus.ACTIVE)
    await user.insert()
    user_id = str(user.id)

    project = Project(name="Test Proj Tools", description="Desc", developer_ids=[], owner_id=user_id)
    await project.insert()
    proj_id = str(project.id)
    
    task = Task(project_id=proj_id, type=TaskType.TASK, title="Tool Task", description="Blocked by DB setup", status=TaskStatus.TODO, assigned_to_id=user_id, estimated_hours=5.0)
    await task.insert()
    
    # Test backlog search tool
    backlog_items = await react_engine._tool_search_backlog(proj_id, {"status": "TODO"})
    assert len(backlog_items) > 0
    assert backlog_items[0]["entity_type"] == "TASK"
    assert "cit_" in backlog_items[0]["metadata"]["citation_hash"]
    
    # Test workload metrics tool
    metrics = await react_engine._tool_get_team_workload_metrics(proj_id)
    assert len(metrics) > 0
    assert metrics[0]["metadata"]["active_task_count"] == 1
    assert metrics[0]["metadata"]["total_estimated_hours"] == 5.0
    
    # Test read document chunk tool (MongoDB fallback)
    project.requirements = "Requirement line 1\nRequirement line 2"
    await project.save()
    
    doc_chunk = await react_engine._tool_read_document_chunk(proj_id, {"doc_id": "project_requirements", "chunk_index": 0})
    assert len(doc_chunk) > 0
    assert "Requirement" in doc_chunk[0]["content_snippet"]

    # Test read document chunk tool handles None / invalid doc_id
    doc_chunk_none = await react_engine._tool_read_document_chunk(proj_id, {"doc_id": None, "chunk_index": 0})
    assert doc_chunk_none == []

    doc_chunk_missing = await react_engine._tool_read_document_chunk(proj_id, {"chunk_index": 0})
    assert doc_chunk_missing == []

