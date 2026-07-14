import pytest
import os
import shutil
from app.ai.storage.lancedb_client import LanceDBManager

TEST_DB_PATH = "./.lancedb_test_store"

@pytest.fixture(autouse=True)
def clean_db():
    # Clean up test database directory before and after tests
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)

def test_lancedb_manager_lifecycle():
    manager = LanceDBManager(db_path=TEST_DB_PATH)
    
    # Generate mock 768-dim vectors
    vector_1 = [0.1] * 768
    vector_2 = [0.2] * 768
    
    # 1. Insertion Test
    manager.insert_task(
        vector=vector_1,
        task_id="task_1",
        project_id="proj_A",
        title="First Task",
        status="TODO"
    )
    
    # Verify count and existence
    assert manager.table.count_rows() == 1
    
    # 2. Search Similar Test
    results = manager.search_similar(vector=vector_1, project_id="proj_A", limit=5)
    assert len(results) == 1
    assert results[0]["task_id"] == "task_1"
    assert results[0]["project_id"] == "proj_A"
    assert results[0]["title"] == "First Task"
    
    # Verify separation constraints via metadata filtering rules (searching with different project_id should return 0 results)
    empty_results = manager.search_similar(vector=vector_1, project_id="proj_B", limit=5)
    assert len(empty_results) == 0
    
    # 3. Update (Upsert) Test
    # Try updating the title and status of task_1
    updated_vector = [0.15] * 768
    manager.upsert_task(
        vector=updated_vector,
        task_id="task_1",
        project_id="proj_A",
        title="First Task Updated",
        status="IN_PROGRESS"
    )
    
    # Count should still be 1 (upsert did not insert a new row)
    assert manager.table.count_rows() == 1
    
    # Retrieve and check updated values
    results = manager.search_similar(vector=updated_vector, project_id="proj_A", limit=5)
    assert len(results) == 1
    assert results[0]["title"] == "First Task Updated"
    assert results[0]["status"] == "IN_PROGRESS"
    
    # Upsert inserting a new task
    manager.upsert_task(
        vector=vector_2,
        task_id="task_2",
        project_id="proj_A",
        title="Second Task",
        status="TODO"
    )
    
    # Count should now be 2
    assert manager.table.count_rows() == 2
    
    # 4. Deletion Test
    manager.delete_task(task_id="task_1")
    
    # Count should be 1
    assert manager.table.count_rows() == 1
    
    remaining_results = manager.search_similar(vector=vector_2, project_id="proj_A", limit=5)
    assert len(remaining_results) == 1
    assert remaining_results[0]["task_id"] == "task_2"

def test_validation_constraints():
    manager = LanceDBManager(db_path=TEST_DB_PATH)
    
    # Invalid vector size
    with pytest.raises(ValueError, match="Vector dimensions must be exactly 768"):
        manager.insert_task(
            vector=[0.1] * 100,
            task_id="task_fail",
            project_id="proj_A",
            title="Fail Task",
            status="TODO"
        )
        
    # Empty task_id
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        manager.insert_task(
            vector=[0.1] * 768,
            task_id="",
            project_id="proj_A",
            title="Fail Task",
            status="TODO"
        )
