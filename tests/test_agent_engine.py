import pytest
import asyncio
import json
import os
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from app.ai.core.agent_engine import AgentEngine

def test_token_counter_and_pruning():
    engine = AgentEngine()
    
    # Verify token counter works
    text = "Hello world from flowpilot server!"
    token_count = engine.count_tokens(text)
    assert token_count > 0

    # Verify context assembly and pruning at 100 tokens (approx 2 blocks)
    mock_items = [
        {
            "entity_type": "TASK",
            "source_id": "task_1",
            "content_snippet": "Setup staging database credentials.",
            "metadata": {"title": "Staging DB", "citation_hash": "cit_1"}
        },
        {
            "entity_type": "TASK",
            "source_id": "task_2",
            "content_snippet": "Configure oauth SSO setup.",
            "metadata": {"title": "SSO Setup", "citation_hash": "cit_2"}
        },
        {
            "entity_type": "TASK",
            "source_id": "task_3",
            "content_snippet": "Unrelated backlog task content.",
            "metadata": {"title": "Backlog Task", "citation_hash": "cit_3"}
        }
    ]

    # Prune at 100 tokens
    pruned = engine._assemble_and_prune_context(mock_items, max_tokens=100)
    assert "cit_1" in pruned
    assert "cit_2" in pruned
    assert "cit_3" not in pruned  # should be pruned out

@pytest.mark.anyio
async def test_agent_engine_process_query_flow():
    engine = AgentEngine()
    
    # Clean logs before test by truncating the file to keep file descriptors intact
    log_path = "logs/copilot_agent.log"
    if os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.truncate(0)

    # Mock LLM manager generate and stream_generate
    mock_generate = MagicMock(return_value="<think>Planning to search backlog</think>\nAction: search_backlog(query=\"sprint 8\")")
    
    async def mock_stream(prompt, max_tokens=2048):
        yield "thought", "Fusing task details"
        yield "chunk", "Response chunk 1"
        yield "chunk", " Response chunk 2"

    # Mock tools
    mock_search_backlog = AsyncMock(return_value=[
        {
            "entity_type": "TASK",
            "source_id": "task_123",
            "content_snippet": "Task description blocker.",
            "metadata": {"citation_hash": "cit_abc"}
        }
    ])

    with patch("app.ai.core.agent_engine.llm_manager.generate", mock_generate), \
         patch("app.ai.core.agent_engine.llm_manager.stream_generate", mock_stream), \
         patch.object(engine, "_tool_search_backlog", mock_search_backlog):
         
        # Run query process
        stream_chunks = []
        async for chunk in engine.process_query("Find blockers in sprint 8", "proj_xyz"):
            stream_chunks.append(chunk)

        # Verify stream content
        assert len(stream_chunks) > 0
        
        # Verify SSE packaging
        assert any("thought" in c for c in stream_chunks)
        assert any("chunk" in c for c in stream_chunks)
        assert stream_chunks[-1] == "data: [DONE]\n\n"

        # Verify history caching
        assert "proj_xyz" in engine.conversation_history
        assert len(engine.conversation_history["proj_xyz"]) == 1
        cached_turn = engine.conversation_history["proj_xyz"][0]
        assert cached_turn["query"] == "Find blockers in sprint 8"
        assert "Response chunk 1" in cached_turn["response"]

        # Flush any logger handlers to make sure log entries are written
        for handler in logging.getLogger("copilot_agent").handlers:
            handler.flush()

        # Verify file logging exists and captured execution
        assert os.path.exists(log_path)
        with open(log_path, "r") as f:
            log_content = f.read()
            assert "Starting process_query" in log_content
            assert "Executed tool pattern" in log_content
            assert "Total process_query timeframe" in log_content

def test_route_query_mutations():
    engine = AgentEngine()
    
    # Test case 1: status change
    actions = engine._route_query("mark task 'Login UI' as DONE", "proj_123")
    assert len(actions) == 1
    assert actions[0]["name"] == "modify_tasks"
    assert actions[0]["args"]["target_status"] == "DONE"
    assert actions[0]["args"]["task_title_or_id"] == "Login UI"

    # Test case 2: priority change
    actions = engine._route_query("set priority of task 'SSO' to high", "proj_123")
    assert len(actions) == 1
    assert actions[0]["name"] == "modify_tasks"
    assert actions[0]["args"]["target_priority"] == "HIGH"
    assert actions[0]["args"]["task_title_or_id"] == "SSO"

    # Test case 3: assign task
    actions = engine._route_query("assign task 'Update docs' to roy", "proj_123")
    assert len(actions) == 1
    assert actions[0]["name"] == "modify_tasks"
    assert actions[0]["args"]["target_assignee"] == "roy"
    assert actions[0]["args"]["task_title_or_id"] == "Update docs"

    # Test case 4: bulk update
    actions = engine._route_query("transition tasks in sprint 8 from todo to done", "proj_123")
    assert len(actions) == 1
    assert actions[0]["name"] == "modify_tasks"
    assert actions[0]["args"]["target_status"] == "DONE"
    assert actions[0]["args"]["filter_status"] == "TODO"
    assert actions[0]["args"]["filter_sprint"] == "Sprint 8"

    # Test case 5: query that looks like mutation but starts with question word should NOT trigger mutation
    actions_q = engine._route_query("what tasks were changed to done in sprint 8", "proj_123")
    assert all(a["name"] != "modify_tasks" for a in actions_q)

@pytest.mark.anyio
async def test_tool_modify_tasks_execution(client):
    engine = AgentEngine()
    
    class MockStatus:
        def __init__(self, value):
            self.value = value
    class MockPriority:
        def __init__(self, value):
            self.value = value

    mock_task = MagicMock()
    mock_task.id = "task_id_123"
    mock_task.project_id = "proj_123"
    mock_task.title = "Test Task"
    mock_task.status = MockStatus("TODO")
    mock_task.priority = MockPriority("MEDIUM")
    mock_task.assigned_to_id = None
    mock_task.model_dump = MagicMock(return_value={"id": "task_id_123", "title": "Test Task"})
    mock_task.save = AsyncMock()

    mock_user = MagicMock()
    mock_user.id = "user_id_roy"
    mock_user.name = "roy"
    
    mock_task_find = MagicMock()
    mock_task_find.to_list = AsyncMock(return_value=[mock_task])
    
    with patch("app.models.task.Task.find", return_value=mock_task_find), \
         patch("app.models.user.User.find_one", AsyncMock(return_value=mock_user)), \
         patch("app.models.user.User.get", AsyncMock(return_value=mock_user)), \
         patch("app.models.activity_log.ActivityLog.insert", AsyncMock()) as mock_activity_insert, \
         patch("app.routers.task_router.emit_sync_event", AsyncMock()) as mock_emit_sync:
         
        args = {
            "target_status": "DONE",
            "target_priority": "HIGH",
            "target_assignee": "roy",
            "task_title_or_id": "Test Task"
        }
        
        results = await engine._tool_modify_tasks("proj_123", args, user_id="operator_id")
        
        assert len(results) == 1
        assert results[0]["entity_type"] == "TASK"
        assert results[0]["source_id"] == "task_id_123"
        assert "status changed from" in results[0]["content_snippet"]
        assert "priority changed from" in results[0]["content_snippet"]
        assert "assigned to" in results[0]["content_snippet"]
        
        mock_task.save.assert_called_once()
        mock_activity_insert.assert_called_once()
        mock_emit_sync.assert_called_once_with("update", {"id": "task_id_123", "title": "Test Task"})

