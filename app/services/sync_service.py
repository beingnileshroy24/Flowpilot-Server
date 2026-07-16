import logging
from typing import Any
from app.ai.core.embedder import ModernBertEmbedderSingleton
from app.ai.storage.lancedb_client import LanceDBManager
from app.models.task import Task

logger = logging.getLogger(__name__)

# Singletons lazy init
embedder = ModernBertEmbedderSingleton()
lancedb_manager = LanceDBManager()

async def sync_task_event(action_type: str, task_document: dict):
    """
    MongoDB-to-LanceDB Mirror Sync Manager.
    Maps source writes to the vector store index.
    """
    task_id = str(task_document.get("id") or task_document.get("_id", ""))
    project_id = str(task_document.get("project_id", ""))
    title = task_document.get("title", "")
    description = task_document.get("description", "")
    status = task_document.get("status", "TODO")
    
    logger.info(f"[SYNC SERVICE] Processing '{action_type}' event for task ID: {task_id}")
    
    try:
        if action_type == "create":
            vector = embedder.compute_embedding(title, description)
            lancedb_manager.insert_task(vector, task_id, project_id, title, status)
        elif action_type == "update":
            vector = embedder.compute_embedding(title, description)
            lancedb_manager.upsert_task(vector, task_id, project_id, title, status)
        elif action_type == "delete":
            lancedb_manager.delete_task(task_id)
        else:
            logger.warning(f"Unknown action_type {action_type} for task {task_id}")
    except Exception as e:
        logger.error(f"[SYNC SERVICE] Failed to sync '{action_type}' event for task {task_id}: {str(e)}")

async def cleanup_project_vectors(project_id: str):
    """
    Project/Workspace Cleanup Transformation:
    Cleans up orphaned vector fields by executing batch deletion sweeps.
    """
    try:
        logger.info(f"[SYNC SERVICE] Sweeping orphaned vectors for project: {project_id}")
        escaped_project_id = project_id.replace("'", "''")
        lancedb_manager.table.delete(f"project_id = '{escaped_project_id}'")
    except Exception as e:
        logger.error(f"[SYNC SERVICE] Cleanup sweep failed for project {project_id}: {str(e)}")

async def rebuild_vector_index():
    """
    Recovery Fail-Safe:
    Administrative recovery pipeline that rebuilds cleanly aligned vector tables from scratch.
    """
    try:
        logger.info("[SYNC SERVICE] Starting administrative recovery pipeline...")
        # Drop the existing table and recreate it
        try:
            lancedb_manager.db.drop_table(lancedb_manager.table_name)
        except Exception:
            pass # Table might not exist yet
        lancedb_manager._get_or_create_tables()
        
        # Iteratively fetch and reconstruct vectors
        async for task in Task.find_all():
            task_id = str(task.id)
            project_id = task.project_id
            title = task.title
            description = task.description or ""
            status = task.status.value if hasattr(task.status, 'value') else str(task.status)
            
            vector = embedder.compute_embedding(title, description)
            lancedb_manager.insert_task(vector, task_id, project_id, title, status)
            
        logger.info("[SYNC SERVICE] Recovery pipeline completed successfully.")
    except Exception as e:
        logger.error(f"[SYNC SERVICE] Recovery pipeline failed: {str(e)}")
