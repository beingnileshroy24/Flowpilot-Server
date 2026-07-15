import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
import uuid


from app.ai.core.embedder import ModernBertEmbedderSingleton
from app.ai.storage.lancedb_client import LanceDBManager
from app.ai.core.splitter import RecursiveParagraphSplitter
from app.models.task import Task
from app.models.comment import Comment
from app.models.project import Project
from app.models.user import User

logger = logging.getLogger(__name__)

# Singletons
embedder = ModernBertEmbedderSingleton()
lancedb_manager = LanceDBManager()
splitter = RecursiveParagraphSplitter(chunk_size=256, chunk_overlap=32)

# Global async queue for memory synchronization
sync_queue: asyncio.Queue = asyncio.Queue()
_worker_task: asyncio.Task[Any] | None = None

def push_to_sync_queue(entity_type: str, entity_id: str, action_type: str, project_id: str):
    """
    Pushes an entity write/update event to the sync queue.
    """
    event = {
        "entity_type": entity_type,  # "TASK", "COMMENT", "SPRINT", "DOCUMENT"
        "entity_id": entity_id,
        "action_type": action_type,  # "create", "update", "delete"
        "project_id": project_id
    }
    sync_queue.put_nowait(event)
    logger.debug(f"[SYNC QUEUE] Enqueued event: {event}")

async def start_sync_queue_worker():
    """
    Starts the background worker to batch-process events from the queue.
    """
    global _worker_task
    if _worker_task is not None:
        return
    _worker_task = asyncio.create_task(_worker_loop())
    logger.info("[SYNC QUEUE] Background worker task started.")

async def stop_sync_queue_worker():
    """
    Stops the background worker.
    """
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    _worker_task = None
    logger.info("[SYNC QUEUE] Background worker task stopped.")

async def _worker_loop():
    while True:
        try:
            # Sleep 5 seconds to batch incoming updates
            await asyncio.sleep(5)
            
            # Retrieve all currently queued items
            events = []
            while not sync_queue.empty():
                try:
                    events.append(sync_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
                    
            if not events:
                continue
                
            logger.info(f"[SYNC QUEUE] Processing batch of {len(events)} events...")
            
            # Deduplicate events in the batch: keep the latest action for each (entity_type, entity_id)
            deduped = {}
            for ev in events:
                key = (ev["entity_type"], ev["entity_id"])
                deduped[key] = ev
                
            for key, ev in deduped.items():
                try:
                    await _sync_event_to_lancedb(ev)
                except Exception as ex:
                    logger.error(f"[SYNC QUEUE] Failed to sync event {ev}: {str(ex)}")
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[SYNC QUEUE] Error in worker loop: {str(e)}")

async def _sync_event_to_lancedb(event: dict):
    entity_type = event["entity_type"]
    entity_id = event["entity_id"]
    action_type = event["action_type"]
    project_id = event["project_id"]
    
    # Handle deletions
    if action_type == "delete":
        if entity_type == "TASK":
            lancedb_manager.delete_task(entity_id) # Legacy
            lancedb_manager.delete_knowledge(entity_id) # Unified
        else:
            lancedb_manager.delete_knowledge(entity_id)
        logger.info(f"[SYNC QUEUE] Handled DELETE event for {entity_type} ID: {entity_id}")
        return

    # Handle creation and updates
    if entity_type == "TASK":
        task = await Task.get(entity_id)
        if not task:
            return
            
        # Get assignee details
        assignee_role = "UNASSIGNED"
        if task.assigned_to_id:
            user = await User.get(task.assigned_to_id)
            if user:
                assignee_role = user.role.value if hasattr(user.role, 'value') else str(user.role)
                
        # Vectorize legacy task registry
        legacy_vector = embedder.compute_embedding(task.title, task.description or "")
        lancedb_manager.upsert_task(
            vector=legacy_vector,
            task_id=entity_id,
            project_id=project_id,
            title=task.title,
            status=task.status.value if hasattr(task.status, 'value') else str(task.status)
        )
        
        # Vectorize unified knowledge registry (Do not split tasks/epics/stories)
        canonical_text = f"Title: {task.title} | Description: {task.description or ''} | Status: {task.status.value} | Assignee: {assignee_role}"
        unified_vector = embedder.compute_embedding(canonical_text, "")
        
        chunk = {
            "vector": unified_vector,
            "created_at": task.created_at.isoformat() if task.created_at else datetime.now(timezone.utc).isoformat(),
            "content_snippet": canonical_text,
            "metadata_struct": {
                "title": task.title,
                "sprint": task.sprint_id or "",
                "owner": assignee_role,
                "code": task.priority.value if hasattr(task.priority, 'value') else str(task.priority),
                "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
            }
        }
        lancedb_manager.upsert_knowledge_batch(
            project_id=project_id,
            entity_type="TASK",
            source_id=entity_id,
            chunks=[chunk]
        )
        logger.info(f"[SYNC QUEUE] Synced TASK {entity_id} to LanceDB.")

    elif entity_type == "COMMENT":
        comment = await Comment.get(entity_id)
        if not comment:
            return
            
        task = await Task.get(comment.task_id)
        task_title = task.title if task else "Unknown Task"
        
        canonical_text = f"Author: {comment.author_name} | Comment: {comment.content} | Task: {task_title}"
        vector = embedder.compute_embedding(canonical_text, "")
        
        chunk = {
            "vector": vector,
            "created_at": comment.created_at.isoformat() if comment.created_at else datetime.now(timezone.utc).isoformat(),
            "content_snippet": canonical_text,
            "metadata_struct": {
                "author": comment.author_name,
                "task_ref": task_title,
                "timestamp": comment.created_at.isoformat() if comment.created_at else "",
                "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
            }
        }
        lancedb_manager.upsert_knowledge_batch(
            project_id=project_id,
            entity_type="COMMENT",
            source_id=entity_id,
            chunks=[chunk]
        )
        logger.info(f"[SYNC QUEUE] Synced COMMENT {entity_id} to LanceDB.")

    elif entity_type == "SPRINT" or entity_type == "DOCUMENT":
        # Index project document subsets: Sprints, Retros, and Requirements
        project = await Project.get(project_id)
        if not project:
            return
            
        # Re-index sprints as a document source
        sprint_texts = []
        for s in project.sprints:
            sprint_texts.append(f"Sprint Title: {s.title} | Goal: {s.goal or ''} | Status: {s.status}")
            
        if sprint_texts:
            full_sprint_text = "\n\n".join(sprint_texts)
            chunks = splitter.split_text(full_sprint_text)
            
            chunk_records = []
            for idx, text in enumerate(chunks):
                vector = embedder.compute_embedding(text, "")
                chunk_records.append({
                    "vector": vector,
                    "created_at": project.created_at.isoformat() if project.created_at else datetime.now(timezone.utc).isoformat(),
                    "content_snippet": text,
                    "metadata_struct": {
                        "filename": "project_sprints.txt",
                        "chunk_idx": idx,
                        "section": "Sprints",
                        "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
                    }
                })
            
            lancedb_manager.upsert_knowledge_batch(
                project_id=project_id,
                entity_type="DOCUMENT",
                source_id=f"{project_id}_sprints",
                chunks=chunk_records
            )
            logger.info(f"[SYNC QUEUE] Synced {len(chunk_records)} SPRINT chunks for project {project_id}.")

        # Re-index project requirements
        if project.requirements:
            chunks = splitter.split_text(project.requirements)
            chunk_records = []
            for idx, text in enumerate(chunks):
                vector = embedder.compute_embedding(text, "")
                chunk_records.append({
                    "vector": vector,
                    "created_at": project.created_at.isoformat() if project.created_at else datetime.now(timezone.utc).isoformat(),
                    "content_snippet": text,
                    "metadata_struct": {
                        "filename": "project_requirements.txt",
                        "chunk_idx": idx,
                        "section": "Requirements",
                        "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
                    }
                })
            lancedb_manager.upsert_knowledge_batch(
                project_id=project_id,
                entity_type="DOCUMENT",
                source_id=f"{project_id}_requirements",
                chunks=chunk_records
            )
            logger.info(f"[SYNC QUEUE] Synced {len(chunk_records)} REQUIREMENT chunks for project {project_id}.")

        # Re-index project retrospective retro entries
        retro_texts = []
        for r in project.retro_entries:
            retro_texts.append(f"Retro for Sprint: {r.sprint_title or 'unknown'} | Went Well: {', '.join(r.went_well)} | Improvements: {', '.join(r.improvements)}")
            
        if retro_texts:
            full_retro_text = "\n\n".join(retro_texts)
            chunks = splitter.split_text(full_retro_text)
            
            chunk_records = []
            for idx, text in enumerate(chunks):
                vector = embedder.compute_embedding(text, "")
                chunk_records.append({
                    "vector": vector,
                    "created_at": project.created_at.isoformat() if project.created_at else datetime.now(timezone.utc).isoformat(),
                    "content_snippet": text,
                    "metadata_struct": {
                        "filename": "project_retrospectives.txt",
                        "chunk_idx": idx,
                        "section": "Retrospectives",
                        "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
                    }
                })
            
            lancedb_manager.upsert_knowledge_batch(
                project_id=project_id,
                entity_type="DOCUMENT",
                source_id=f"{project_id}_retrospectives",
                chunks=chunk_records
            )
            logger.info(f"[SYNC QUEUE] Synced {len(chunk_records)} RETRO chunks for project {project_id}.")
