import pytest
import os
import shutil
from unittest.mock import patch
from app.database import init_db
from app.ai.storage.lancedb_client import LanceDBManager
from app.services import sync_service
from app.models.task import Task, TaskType, TaskStatus
from app.models.project import Project

TEST_DB_PATH = "./.lancedb_sync_test_store"

@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)

@pytest.mark.anyio
async def test_rebuild_vector_index():
    # Initialize Beanie database in the same loop
    await init_db()
    
    # Instantiate a clean LanceDBManager pointing to the test DB path
    test_manager = LanceDBManager(db_path=TEST_DB_PATH)
    
    # Patch the singleton manager in sync_service
    with patch("app.services.sync_service.lancedb_manager", test_manager):
        # Insert a mock task in MongoDB
        # Note: clean_database autouse fixture in conftest.py clears MongoDB tasks table
        task = Task(
            project_id="test_project_id",
            type=TaskType.TASK,
            title="Integrate LanceDB index rebuild",
            description="Fixing typo in recovery method",
            status=TaskStatus.TODO,
            assigned_to_id="some_user_id",
            estimated_hours=2.0
        )
        await task.insert()
        
        # Verify the database has the task
        tasks_in_db = await Task.find_all().to_list()
        assert len(tasks_in_db) == 1
        
        # Verify LanceDB currently has 0 tasks
        assert test_manager.table.count_rows() == 0
        
        # Execute the rebuild index pipeline
        await sync_service.rebuild_vector_index()
        
        # Verify LanceDB has the reconstructed task
        assert test_manager.table.count_rows() == 1
        results = test_manager.search_similar(vector=[0.1]*768, project_id="test_project_id", limit=1)
        assert len(results) == 1
        assert results[0]["task_id"] == str(task.id)
        assert results[0]["title"] == "Integrate LanceDB index rebuild"
        assert results[0]["status"] == "TODO"
