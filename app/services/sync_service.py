import logging

logger = logging.getLogger(__name__)

async def sync_task_event(action_type: str, task_document: dict):
    """
    Placeholder/mock for MongoDB-to-LanceDB Mirror Sync Manager.
    Logs the action type and task details, simulating downstream synchronization.
    """
    task_id = task_document.get("id") or task_document.get("_id")
    title = task_document.get("title", "Unknown Title")
    logger.info(f"[SYNC SERVICE] Received '{action_type}' event for task ID: {task_id} ('{title}')")
