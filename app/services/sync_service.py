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
    from app.models.project import Project
    from app.models.comment import Comment
    from app.ai.core.splitter import RecursiveParagraphSplitter
    
    try:
        logger.info("[SYNC SERVICE] Starting administrative recovery pipeline...")
        # 1. Drop existing tables to start fresh
        try:
            lancedb_manager.db.drop_table(lancedb_manager.table_name)
        except Exception:
            pass
        try:
            lancedb_manager.db.drop_table(lancedb_manager.knowledge_table_name)
        except Exception:
            pass
            
        lancedb_manager._get_or_create_tables()
        splitter = RecursiveParagraphSplitter(chunk_size=256, chunk_overlap=32)
        
        # 2. Sync all Task documents
        logger.info("[SYNC SERVICE] Rebuilding Task vectors...")
        from app.models.user import User
        async for task in Task.find_all():
            task_id = str(task.id)
            project_id = task.project_id
            title = task.title
            description = task.description or ""
            status = task.status.value if hasattr(task.status, 'value') else str(task.status)
            priority = task.priority.value if hasattr(task.priority, 'value') else str(task.priority)
            sprint_id = task.sprint_id or ""
            created_at = task.created_at.isoformat() if task.created_at else ""

            # Resolve assignee name
            assignee_name = "Unassigned"
            if task.assigned_to_id:
                try:
                    user = await User.get(task.assigned_to_id)
                    if user:
                        assignee_name = user.name if hasattr(user, 'name') and user.name else (
                            user.role.value if hasattr(user.role, 'value') else str(user.role)
                        )
                except Exception:
                    pass

            # Legacy tasks table sync
            vector = embedder.compute_embedding(title, description)
            lancedb_manager.insert_task(vector, task_id, project_id, title, status)
            
            # Unified knowledge table sync — use the same rich canonical text as sync_queue
            snippet = (
                f"Title: {title} | "
                f"Status: {status} | "
                f"Priority: {priority} | "
                f"Assignee: {assignee_name} | "
                f"Sprint: {sprint_id} | "
                f"Description: {description[:200]}"
            )
            canonical_vector = embedder.compute_embedding(snippet, "")
            lancedb_manager.insert_knowledge(
                vector=canonical_vector,
                entity_type="TASK",
                source_id=task_id,
                project_id=project_id,
                created_at=created_at,
                content_snippet=snippet,
                metadata_struct={
                    "title": title,
                    "status": status,
                    "priority": priority,
                    "sprint": sprint_id,
                    "owner": assignee_name,
                    "code": priority
                }
            )


        # 3. Sync all Project document chunks (Requirements, Sprints, Retrospectives)
        logger.info("[SYNC SERVICE] Rebuilding Project document vectors...")
        async for project in Project.find_all():
            project_id = str(project.id)
            created_at_str = project.created_at.isoformat() if project.created_at else ""
            
            # --- Index Project Requirements ---
            reqs_text = project.requirements or ""
            if reqs_text.strip():
                reqs_chunks = splitter.split_text(reqs_text)
                for idx, chunk in enumerate(reqs_chunks):
                    vector = embedder.compute_embedding(chunk, "")
                    lancedb_manager.insert_knowledge(
                        vector=vector,
                        entity_type="DOCUMENT",
                        source_id="requirements",
                        project_id=project_id,
                        created_at=created_at_str,
                        content_snippet=chunk,
                        metadata_struct={
                            "filename": "project_requirements.txt",
                            "chunk_idx": idx,
                            "section": "Requirements"
                        }
                    )
                    
            # --- Index Project Sprints ---
            if project.sprints:
                sprint_texts = [
                    f"Sprint Title: {s.title} | Goal: {s.goal or ''} | Status: {s.status} | Capacity: {s.capacity_hours} hrs"
                    for s in project.sprints
                ]
                sprint_combined = "\n\n".join(sprint_texts)
                sprint_chunks = splitter.split_text(sprint_combined)
                for idx, chunk in enumerate(sprint_chunks):
                    vector = embedder.compute_embedding(chunk, "")
                    lancedb_manager.insert_knowledge(
                        vector=vector,
                        entity_type="DOCUMENT",
                        source_id="sprints",
                        project_id=project_id,
                        created_at=created_at_str,
                        content_snippet=chunk,
                        metadata_struct={
                            "filename": "project_sprints.txt",
                            "chunk_idx": idx,
                            "section": "Sprints"
                        }
                    )
                    
            # --- Index Project Retrospectives ---
            if project.retro_entries:
                retro_texts = [
                    f"Retro for Sprint: {r.sprint_title or 'unknown'} | Went Well: {', '.join(r.went_well)} | Improvements: {', '.join(r.improvements)}"
                    for r in project.retro_entries
                ]
                retro_combined = "\n\n".join(retro_texts)
                retro_chunks = splitter.split_text(retro_combined)
                for idx, chunk in enumerate(retro_chunks):
                    vector = embedder.compute_embedding(chunk, "")
                    lancedb_manager.insert_knowledge(
                        vector=vector,
                        entity_type="DOCUMENT",
                        source_id="retrospectives",
                        project_id=project_id,
                        created_at=created_at_str,
                        content_snippet=chunk,
                        metadata_struct={
                            "filename": "project_retrospectives.txt",
                            "chunk_idx": idx,
                            "section": "Retrospectives"
                        }
                    )

        # 4. Sync all Task Comments
        logger.info("[SYNC SERVICE] Rebuilding Comment vectors...")
        async for comment in Comment.find_all():
            comment_id = str(comment.id)
            task_id = comment.task_id
            content = comment.content
            author_name = comment.author_name
            created_at = comment.created_at.isoformat() if comment.created_at else ""
            
            # Lookup task to resolve project_id scope
            task = await Task.get(task_id)
            if task:
                project_id = task.project_id
                vector = embedder.compute_embedding(content, "")
                lancedb_manager.insert_knowledge(
                    vector=vector,
                    entity_type="COMMENT",
                    source_id=comment_id,
                    project_id=project_id,
                    created_at=created_at,
                    content_snippet=f"Comment by {author_name}: {content}",
                    metadata_struct={
                        "task_id": task_id,
                        "author_name": author_name
                    }
                )

        logger.info("[SYNC SERVICE] Recovery pipeline completed successfully.")
    except Exception as e:
        logger.error(f"[SYNC SERVICE] Recovery pipeline failed: {str(e)}")
