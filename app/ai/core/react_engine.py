import re
import json
import logging
import numpy as np
import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, AsyncGenerator
from beanie.operators import In

from app.ai.core.llm_manager import llm_manager
from app.ai.core.embedder import ModernBertEmbedderSingleton
from app.ai.storage.lancedb_client import LanceDBManager
from app.ai.prompts.nwbe_templates import REACT_PLANNING_TEMPLATE, GROUNDED_SYNTHESIS_TEMPLATE
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.project import Project
from app.models.comment import Comment
from app.models.activity_log import ActivityLog
from app.models.user import User, UserRole

logger = logging.getLogger(__name__)

# Singletons lazy init
embedder = ModernBertEmbedderSingleton()
lancedb_manager = LanceDBManager()


class ReActEngine:
    def __init__(self):
        pass

    async def execute_query(self, query: str, context_scope: dict) -> AsyncGenerator[str, None]:
        """
        Executes the ReAct loop:
        1. Planning/Tool Choice (up to 3 iterations)
        2. Tool Execution (search_backlog, read_document_chunk, get_team_workload_metrics)
        3. Rank Fusion (RRF) & Pruning
        4. Grounded Synthesis Stream with cit_[short-uuid] citations
        """
        project_id = context_scope.get("project_id", "")
        if not project_id:
            yield "data: " + json.dumps({"error": "project_id is required in contextScope."}) + "\n\n"
            return

        conversation_log = ""
        all_retrieved_items = []

        # --- Phase 1 & 2: ReAct Planning Loop (Bounded by max 3 iterations) ---
        for iteration in range(3):
            planning_prompt = REACT_PLANNING_TEMPLATE.format(
                query=query,
                context_scope=json.dumps(context_scope),
                conversation_log=conversation_log or "No actions taken yet."
            )
            
            logger.info(f"[REACT ENGINE] Running planning step (iteration {iteration + 1})...")
            llm_response = llm_manager.generate(planning_prompt, max_tokens=512)
            logger.info(f"[REACT ENGINE] Planning response: {llm_response}")

            # Parse planning response to extract think block and tool choice
            think_content = ""
            think_match = re.search(r"<think>(.*?)</think>", llm_response, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                # Stream the planning thought chain to the gateway
                yield f"data: {json.dumps({'thought': f'[Step {iteration + 1}] ' + think_content})}\n\n"

            tool_action = self._parse_tool_action(llm_response)
            action_name = tool_action.get("name")
            args = tool_action.get("args", {})
            
            logger.info(f"[REACT ENGINE] Decoded tool action: {action_name}({args})")
            
            if action_name == "finalize" or not action_name:
                logger.info("[REACT ENGINE] Agent requested finalize or no action found. Breaking loop.")
                break

            # Execute tool
            retrieved_items = []
            error_msg = None
            try:
                if action_name == "search_backlog":
                    retrieved_items = await self._tool_search_backlog(project_id, args)
                elif action_name == "read_document_chunk":
                    retrieved_items = await self._tool_read_document_chunk(project_id, args)
                elif action_name == "get_team_workload_metrics":
                    t_project_id = args.get("project_id") or project_id
                    retrieved_items = await self._tool_get_team_workload_metrics(t_project_id)
                else:
                    error_msg = f"Unknown tool: {action_name}"
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[REACT ENGINE] Tool execution failed: {error_msg}")

            if error_msg:
                observation = f"Error executing tool: {error_msg}"
            else:
                observation = f"Executed {action_name} successfully. Gathered {len(retrieved_items)} items."
                all_retrieved_items.extend(retrieved_items)
                
            # Inject: Append action and outcome to conversation log
            conversation_log += f"Thought: {think_content}\nAction: {action_name}({args})\nObservation: {observation}\n"

        # Also, perform a quick vector search to get a secondary candidate list for hybrid RRF fusion
        secondary_items = []
        try:
            secondary_items = await self._tool_vector_search(project_id, query, limit=5)
        except Exception:
            pass

        # --- Phase 3: Reciprocal Rank Fusion (RRF) & Cosine threshold cutoff ---
        fused_items = self._reciprocal_rank_fusion([all_retrieved_items, secondary_items])
        
        # Prune context to maximum 4000 tokens (~16,000 characters)
        pruned_context = self._prune_context(fused_items, max_chars=16000)

        # --- Phase 4: Grounded Synthesis & Streaming ---
        # If nothing was retrieved, pass an explicit NO_DATA marker so the model
        # gives a clean one-sentence reply rather than hallucinating context.
        context_block = pruned_context if pruned_context.strip() else "NO_DATA — No relevant workspace data was found for this query."
        synthesis_prompt = GROUNDED_SYNTHESIS_TEMPLATE.format(
            context=context_block,
            query=query
        )

        logger.info("[REACT ENGINE] Starting synthesis streaming generation...")
        async for chunk_type, token_text in llm_manager.stream_generate(synthesis_prompt):
            # Route thought tokens and answer tokens to separate SSE event types
            if chunk_type == "thought":
                yield f"data: {json.dumps({'thought': token_text})}\n\n"
            else:
                yield f"data: {json.dumps({'chunk': token_text})}\n\n"

        # Signal completion
        yield "data: [DONE]\n\n"

    def _parse_tool_action(self, response_text: str) -> dict:
        """
        Parses text like 'Action: search_backlog(sprint="Sprint 8", status="BLOCKED")'
        Supports both quoted strings and numeric parameters (like chunk_index).
        """
        # Find the Action line
        action_match = re.search(r"Action:\s*(\w+)\((.*)\)", response_text)
        if not action_match:
            return {"name": "search_backlog", "args": {}}

        name = action_match.group(1)
        args_str = action_match.group(2)
        
        args: Dict[str, Any] = {}
        # Matches key="val", key='val', or key=123 (integers)
        kwarg_matches = re.finditer(r"(\w+)\s*=\s*(?:[\"'](.*?)[\"']|(\d+))", args_str)
        for m in kwarg_matches:
            key = m.group(1)
            val_str = m.group(2)
            val_num = m.group(3)
            if val_num is not None:
                args[key] = int(val_num)
            else:
                args[key] = val_str
                
        return {"name": name, "args": args}

    # ================= Tool Implementations =================

    async def _tool_vector_search(self, project_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Vector search tool querying LanceDB workspace_knowledge_index table.
        Applies Cosine Threshold Strict Cutoff (Drop all hits scoring < 0.72)
        """
        logger.info(f"[TOOL] Vector Search on project_id={project_id}, query='{query}'")
        vector = embedder.compute_embedding(query, "")
        results = lancedb_manager.search_knowledge_similar(vector, project_id, limit=limit)
        
        items = []
        for r in results:
            distance = r.get('_distance', 2.0)
            similarity = 1.0 - (distance / 2.0)
            if similarity < 0.72:
                logger.info(f"[TOOL] Vector search item skipped due to low similarity: {similarity:.4f}")
                continue

            metadata_str = r.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str)
            except Exception:
                metadata = {}
                
            if "citation_hash" not in metadata:
                metadata["citation_hash"] = f"cit_{uuid.uuid4().hex[:5]}"
                
            items.append({
                "entity_type": r.get("entity_type", "UNKNOWN"),
                "source_id": r.get("source_id", ""),
                "project_id": r.get("project_id", ""),
                "created_at": r.get("created_at", ""),
                "content_snippet": r.get("content_snippet", ""),
                "metadata": metadata,
                "similarity": similarity
            })
        return items

    async def _tool_search_backlog(self, project_id: str, args: dict) -> List[Dict[str, Any]]:
        """
        Queries LanceDB (vector search) and MongoDB (exact task parameters) and fuses the results.
        """
        query = args.get("query")
        status_val = args.get("status")
        sprint_title = args.get("sprint")
        
        vector_results = []
        mongo_results = []
        
        if query:
            vector_results = await self._tool_vector_search(project_id, query)
            
        if status_val or sprint_title:
            filters: List[Any] = [Task.project_id == project_id]
            if sprint_title:
                project = await Project.get(project_id)
                sprint_id = None
                if project and project.sprints:
                    for s in project.sprints:
                        if s.title.lower() == sprint_title.lower():
                            sprint_id = s.id
                            break
                if sprint_id:
                    filters.append(Task.sprint_id == sprint_id)
                else:
                    filters.append(Task.sprint_id == sprint_title)
                    
            if status_val:
                for enum_val in TaskStatus:
                    if enum_val.value == status_val.upper():
                        filters.append(Task.status == enum_val)
                        break
                        
            tasks = await Task.find(*filters).to_list()
            for t in tasks:
                assignee_role = "UNKNOWN"
                if t.assigned_to_id:
                    user = await User.get(t.assigned_to_id)
                    if user:
                        assignee_role = user.role.value if hasattr(user.role, 'value') else str(user.role)
                
                content = f"Title: {t.title} | Description: {t.description or ''} | Status: {t.status.value} | Assignee: {assignee_role} | Priority: {t.priority.value}"
                mongo_results.append({
                    "entity_type": "TASK",
                    "source_id": str(t.id),
                    "project_id": project_id,
                    "created_at": t.created_at.isoformat() if t.created_at else "",
                    "content_snippet": content,
                    "metadata": {
                        "title": t.title,
                        "sprint": sprint_title or t.sprint_id or "",
                        "owner": assignee_role,
                        "status": t.status.value if hasattr(t.status, 'value') else str(t.status),
                        "code": t.priority.value,
                        "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
                    }
                })
                
        if vector_results and mongo_results:
            return self._reciprocal_rank_fusion([vector_results, mongo_results])
        elif vector_results:
            return vector_results
        else:
            return mongo_results

    async def _tool_read_document_chunk(self, project_id: str, args: dict) -> List[Dict[str, Any]]:
        """
        Pulls explicit source lines from system documents (requirements, retrospectives, sprints) at a given chunk index.
        """
        doc_id = args.get("doc_id")
        if not doc_id or not isinstance(doc_id, str):
            return []
        chunk_idx = args.get("chunk_index")
        if chunk_idx is None:
            chunk_idx = 0
        else:
            try:
                chunk_idx = int(chunk_idx)
            except ValueError:
                chunk_idx = 0
                
        logger.info(f"[TOOL] Reading document chunk for doc_id={doc_id}, chunk_index={chunk_idx}")
        
        # 1. Try to fetch from LanceDB
        try:
            results = lancedb_manager.knowledge_table.search().where(f"project_id = '{project_id}' AND source_id = '{doc_id}'").to_list()
            for r in results:
                metadata_str = r.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str)
                except Exception:
                    metadata = {}
                    
                if metadata.get("chunk_idx") == chunk_idx:
                    if "citation_hash" not in metadata:
                        metadata["citation_hash"] = f"cit_{uuid.uuid4().hex[:5]}"
                    return [{
                        "entity_type": r.get("entity_type", "DOCUMENT"),
                        "source_id": r.get("source_id", ""),
                        "project_id": r.get("project_id", ""),
                        "created_at": r.get("created_at", ""),
                        "content_snippet": r.get("content_snippet", ""),
                        "metadata": metadata
                    }]
        except Exception as e:
            logger.error(f"[TOOL] Failed to fetch document chunk from LanceDB: {str(e)}")
            
        # 2. Fallback: split project fields from MongoDB
        project = await Project.get(project_id)
        if not project:
            return []
            
        text = ""
        filename = "document.txt"
        section = "Content"
        
        if "requirements" in doc_id.lower():
            text = project.requirements or ""
            filename = "project_requirements.txt"
            section = "Requirements"
        elif "retro" in doc_id.lower():
            retro_texts = [f"Retro for Sprint: {r.sprint_title or 'unknown'} | Went Well: {', '.join(r.went_well)} | Improvements: {', '.join(r.improvements)}" for r in project.retro_entries]
            text = "\n\n".join(retro_texts)
            filename = "project_retrospectives.txt"
            section = "Retrospectives"
        elif "sprint" in doc_id.lower():
            sprint_texts = [f"Sprint Title: {s.title} | Goal: {s.goal or ''} | Status: {s.status}" for s in project.sprints]
            text = "\n\n".join(sprint_texts)
            filename = "project_sprints.txt"
            section = "Sprints"
            
        if text:
            from app.ai.core.splitter import RecursiveParagraphSplitter
            splitter = RecursiveParagraphSplitter(chunk_size=256, chunk_overlap=32)
            chunks = splitter.split_text(text)
            if 0 <= chunk_idx < len(chunks):
                return [{
                    "entity_type": "DOCUMENT",
                    "source_id": doc_id,
                    "project_id": project_id,
                    "created_at": project.created_at.isoformat() if project.created_at else "",
                    "content_snippet": chunks[chunk_idx],
                    "metadata": {
                        "filename": filename,
                        "chunk_idx": chunk_idx,
                        "section": section,
                        "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
                    }
                }]
                
        return []

    async def _tool_get_team_workload_metrics(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Aggregates active (non-DONE) task points (estimated hours) and counts per engineer.
        """
        logger.info(f"[TOOL] Getting team workload metrics for project_id={project_id}")
        tasks = await Task.find(Task.project_id == project_id).to_list()
        
        from collections import defaultdict
        metrics_by_user = defaultdict(lambda: {"task_count": 0, "estimated_hours": 0.0})
        
        for t in tasks:
            if t.status != TaskStatus.DONE:
                user_key = t.assigned_to_id or "Unassigned"
                metrics_by_user[user_key]["task_count"] += 1
                metrics_by_user[user_key]["estimated_hours"] += t.estimated_hours
                
        items = []
        for eng_id, stats in metrics_by_user.items():
            name = "Unassigned"
            email = "N/A"
            role = "N/A"
            
            if eng_id != "Unassigned":
                user = await User.get(eng_id)
                if user:
                    name = user.name
                    email = user.email
                    role = user.role.value if hasattr(user.role, 'value') else str(user.role)
                    
            active_task_count = stats["task_count"]
            total_estimated_hours = stats["estimated_hours"]
            
            content = f"Engineer: {name} ({email}) | Role: {role} | Active Tasks: {active_task_count} | Total Estimated Hours (Points): {total_estimated_hours}"
            items.append({
                "entity_type": "METRIC",
                "source_id": f"{project_id}_workload_metrics",
                "project_id": project_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content_snippet": content,
                "metadata": {
                    "engineer_id": eng_id,
                    "name": name,
                    "email": email,
                    "role": role,
                    "active_task_count": active_task_count,
                    "total_estimated_hours": total_estimated_hours,
                    "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
                }
            })
            
        return items

    # ================= RRF & Context Pruning =================

    def _reciprocal_rank_fusion(self, rank_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Combines multiple ranked retrieval results using Reciprocal Rank Fusion (RRF).
        Uses numpy arrays for internal indexing and scoring.
        """
        if not rank_lists:
            return []

        # Find all unique items based on (entity_type, source_id)
        unique_keys = []
        key_to_item = {}
        
        for lst in rank_lists:
            for item in lst:
                if not item:
                    continue
                key = (item["entity_type"], item["source_id"])
                if key not in key_to_item:
                    unique_keys.append(key)
                    key_to_item[key] = item

        num_items = len(unique_keys)
        if num_items == 0:
            return []

        rrf_scores = np.zeros(num_items)
        
        # Calculate scores
        for lst in rank_lists:
            for rank_idx, item in enumerate(lst):
                if not item:
                    continue
                key = (item["entity_type"], item["source_id"])
                item_idx = unique_keys.index(key)
                rank = rank_idx + 1
                rrf_scores[item_idx] += 1.0 / (k + rank)

        # Sort item indices by score in descending order
        sorted_indices = np.argsort(-rrf_scores)
        
        sorted_items = []
        for idx in sorted_indices:
            key = unique_keys[idx]
            sorted_items.append(key_to_item[key])
            
        return sorted_items

    def _prune_context(self, items: List[Dict[str, Any]], max_chars: int = 16000) -> str:
        """
        Formats items into a string context matching strict citation boundaries,
        capping characters to stay within approximate 4000 max tokens.
        """
        context_blocks = []
        total_len = 0
        
        for item in items:
            metadata = item.get("metadata", {})
            citation_hash = metadata.get("citation_hash", "")
            if not citation_hash:
                citation_hash = f"cit_{uuid.uuid4().hex[:5]}"
                metadata["citation_hash"] = citation_hash

            entity_type = item.get("entity_type", "UNKNOWN")
            title = metadata.get("title") or metadata.get("filename") or metadata.get("task_ref") or "N/A"
            status = metadata.get("status") or "N/A"
            
            if status == "N/A" and entity_type == "TASK":
                status_match = re.search(r"Status:\s*(\w+)", item.get("content_snippet", ""))
                if status_match:
                    status = status_match.group(1)
            
            sprint = metadata.get("sprint") or "N/A"
            if sprint == "N/A" and entity_type == "TASK":
                sprint_match = re.search(r"Sprint:\s*([^\s|]+)", item.get("content_snippet", ""))
                if sprint_match:
                    sprint = sprint_match.group(1)
            
            content_snippet = item.get("content_snippet", "")
            
            block = (
                f"=========================================\n"
                f"RETRIEVED WORKSPACE CONTEXT ELEMENT: [ID: {citation_hash}]\n"
                f"Type: {entity_type} | Title: {title}\n"
                f"Status: {status} | Sprint: {sprint}\n"
                f"Content: {content_snippet}\n"
                f"=========================================\n"
            )
            
            if total_len + len(block) > max_chars:
                break
                
            context_blocks.append(block)
            total_len += len(block)
            
        return "".join(context_blocks)


react_engine = ReActEngine()
