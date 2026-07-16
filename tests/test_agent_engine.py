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
