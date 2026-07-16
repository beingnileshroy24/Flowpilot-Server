import re
import json
import logging
import time
import asyncio
import uuid
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any, AsyncGenerator, Optional

from app.ai.core.llm_manager import llm_manager
from app.ai.core.embedder import ModernBertEmbedderSingleton
from app.ai.storage.lancedb_client import LanceDBManager
from app.ai.storage.hybrid_retriever import HybridRetriever
from app.ai.prompts.nwbe_templates import REACT_PLANNING_TEMPLATE, GROUNDED_SYNTHESIS_TEMPLATE
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.project import Project
from app.models.user import User, UserRole
from app.models.copilot_chat import CopilotChat, ChatMessage

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Dedicated file logger for AgentEngine
agent_logger = logging.getLogger("copilot_agent")
agent_logger.setLevel(logging.INFO)
if agent_logger.hasHandlers():
    agent_logger.handlers.clear()

file_handler = logging.FileHandler("logs/copilot_agent.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
agent_logger.addHandler(file_handler)

class AgentEngine:
    def __init__(self):
        self.lancedb_manager = LanceDBManager()
        self.embedder = ModernBertEmbedderSingleton()
        self.hybrid_retriever = HybridRetriever(self.lancedb_manager, self.embedder)
        self.conversation_history = defaultdict(list)

    def count_tokens(self, text: str) -> int:
        if llm_manager.tokenizer is not None:
            try:
                return len(llm_manager.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback estimation
        return len(text.split()) + len(text) // 4

    async def process_query(self, user_query: str, project_id: str, chat_id: Optional[str] = None, user_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Agentic Orchestration and Tools Routing Loop.
        Evaluates incoming queries, references history, executes tools, and streams response.
        """
        start_time = time.time()
        agent_logger.info(f"Starting process_query for project_id={project_id}, query='{user_query}', chat_id={chat_id}, user_id={user_id}")
        
        # 1. Reference conversation history cache (DB-persisted if chat_id provided, otherwise local in-memory)
        conversation_log = ""
        chat_document = None
        if chat_id:
            try:
                chat_document = await CopilotChat.get(chat_id)
                if chat_document and chat_document.messages:
                    conversation_log += "=== Prior Conversation History ===\n"
                    turn_idx = 1
                    for msg in chat_document.messages:
                        sender_label = "User" if msg.sender == "user" else "Agent Response"
                        conversation_log += f"Turn {turn_idx} - {sender_label}: {msg.text}\n"
                        if msg.sender == "bot":
                            turn_idx += 1
                    conversation_log += "==================================\n\n"
            except Exception as e:
                agent_logger.error(f"Failed to fetch copilot chat history from DB: {str(e)}")
        else:
            history = self.conversation_history[project_id]
            if history:
                conversation_log += "=== Prior Conversation History ===\n"
                for idx, turn in enumerate(history):
                    conversation_log += f"Turn {idx + 1} - User: {turn['query']}\nTurn {idx + 1} - Agent Response: {turn['response']}\n"
                conversation_log += "==================================\n\n"
            
        all_retrieved_items = []
        
        # 2. Tool routing — deterministic keyword classifier replaces the LLM planning call.
        #    The real DeepSeek-R1 model does NOT reliably follow the strict Action: format,
        #    leading to wrong tools ("search_backlog(query='test query')") for every query type.
        #    Keyword routing is fast, correct, and doesn't require any LLM inference.
        planned_actions = self._route_query(user_query, project_id)
        
        for step_idx, tool_action in enumerate(planned_actions):
            action_name = tool_action.get("name")
            args = tool_action.get("args", {})
            thought_msg = tool_action.get("thought", f"Executing {action_name}...")
            
            yield f"data: {json.dumps({'thought': f'[Step {step_idx + 1}] {thought_msg}\n'})}\n\n"
            agent_logger.info(f"Executed tool pattern: {action_name}({args})")
            
            retrieved_items = []
            error_msg = None
            tool_start_time = time.time()
            
            try:
                if action_name == "search_backlog":
                    retrieved_items = await self._tool_search_backlog(project_id, args)
                elif action_name == "read_document_chunk":
                    retrieved_items = await self._tool_read_document_chunk(project_id, args)
                elif action_name == "get_team_workload_metrics":
                    t_project_id = args.get("project_id") or project_id
                    retrieved_items = await self._tool_get_team_workload_metrics(t_project_id)
                elif action_name == "modify_tasks":
                    retrieved_items = await self._tool_modify_tasks(project_id, args, user_id=user_id)
            except Exception as e:
                error_msg = str(e)
                agent_logger.error(f"Tool execution failed: {error_msg}")
                
            tool_duration = time.time() - tool_start_time
            agent_logger.info(f"Tool execution timeframe: {tool_duration:.4f}s")
            
            if not error_msg:
                all_retrieved_items.extend(retrieved_items)

        # Secondary semantic search — only run if the primary tool was NOT search_backlog
        # (backlog tool already blends MongoDB + vector internally, so a second broad search
        # would contaminate context with DOCUMENTs and COMMENTs)
        routed_tool_names = [a.get("name") for a in planned_actions]
        if "search_backlog" not in routed_tool_names and "modify_tasks" not in routed_tool_names:
            try:
                secondary_items = self.hybrid_retriever.search_hybrid(
                    query=user_query,
                    project_id=project_id,
                    limit=5,
                    similarity_threshold=0.45
                )
                for item in secondary_items:
                    if "similarity" in item:
                        distance = 2.0 * (1.0 - item["similarity"])
                        agent_logger.info(f"Vector distance score: {distance:.4f} (similarity: {item['similarity']:.4f})")
                all_retrieved_items.extend(secondary_items)
            except Exception as e:
                agent_logger.error(f"Secondary hybrid search failed: {str(e)}")

        # Deduplicate and rank items
        fused_items = self._deduplicate_and_rank(all_retrieved_items)
        
        # Yield sources metadata packet to client
        sources_payload = []
        for item in fused_items:
            sources_payload.append({
                "entity_type": item.get("entity_type"),
                "source_id": item.get("source_id"),
                "content_snippet": item.get("content_snippet"),
                "metadata": item.get("metadata")
            })
        yield f"data: {json.dumps({'sources': sources_payload})}\n\n"
        
        # 3. Assemble and prune context
        pruned_context = self._assemble_and_prune_context(fused_items, max_tokens=6000)
        
        # 4. Stream response
        # When no data was retrieved, skip the LLM entirely and reply directly.
        # This avoids hallucination and is much faster.
        if not pruned_context.strip():
            no_data_msg = "No relevant data was found in the workspace database for this query. Please ensure workspace data has been synced, or try rephrasing your question."
            for token in no_data_msg.split():
                yield f"data: {json.dumps({'chunk': token + ' '})}\n\n"
                await asyncio.sleep(0.01)
            if chat_document:
                try:
                    user_msg = ChatMessage(sender="user", text=user_query)
                    bot_msg = ChatMessage(sender="bot", text=no_data_msg)
                    chat_document.messages.append(user_msg)
                    chat_document.messages.append(bot_msg)
                    chat_document.updated_at = datetime.now(timezone.utc)
                    await chat_document.save()
                except Exception as e:
                    agent_logger.error(f"Failed to persist early-exit chat message to MongoDB: {str(e)}")
            else:
                self.conversation_history[project_id].append({
                    "query": user_query,
                    "response": no_data_msg
                })
            total_duration = time.time() - start_time
            agent_logger.info(f"Total process_query timeframe: {total_duration:.4f}s (no-data fast path)")
            yield "data: [DONE]\n\n"
            return

        synthesis_prompt = GROUNDED_SYNTHESIS_TEMPLATE.format(
            context=pruned_context,
            query=user_query
        )
        # Add a newline spacer in the thoughts panel before streaming synthesis thoughts
        yield f"data: {json.dumps({'thought': '\n'})}\n\n"
        
        full_response = ""
        accumulated_thoughts = ""
        async for chunk_type, token_text in llm_manager.stream_generate(synthesis_prompt):
            if chunk_type == "thought":
                accumulated_thoughts += token_text
                yield f"data: {json.dumps({'thought': token_text})}\n\n"
            else:
                full_response += token_text
                yield f"data: {json.dumps({'chunk': token_text})}\n\n"

        # Fallback: If the LLM produced no answer tokens (put everything in <think>),
        # stream a concise summary directly from the retrieved context.
        if not full_response.strip():
            agent_logger.warning("LLM synthesis produced no chunk tokens. Falling back to context summary.")
            summary_lines = ["Based on the retrieved workspace data:\n\n"]
            for item in fused_items[:4]:
                snippet = item.get("content_snippet", "").strip()
                meta = item.get("metadata", {})
                citation = meta.get("citation_hash", "")
                if snippet:
                    line = f"- {snippet[:200]}"
                    if citation:
                        line += f" [{citation}]"
                    summary_lines.append(line + "\n")
            fallback_text = "".join(summary_lines)
            full_response = fallback_text
            for token in fallback_text.split():
                yield f"data: {json.dumps({'chunk': token + ' '})}\n\n"
                await asyncio.sleep(0.01)
            
        # Cache and/or persist conversation messages
        cleaned_response = full_response.strip()
        if chat_document:
            try:
                # Add user prompt and bot response
                user_msg = ChatMessage(sender="user", text=user_query)
                bot_msg = ChatMessage(
                    sender="bot",
                    text=cleaned_response,
                    thoughts=accumulated_thoughts,
                    sources=sources_payload
                )
                chat_document.messages.append(user_msg)
                chat_document.messages.append(bot_msg)
                chat_document.updated_at = datetime.now(timezone.utc)
                await chat_document.save()
            except Exception as e:
                agent_logger.error(f"Failed to persist chat messages to MongoDB: {str(e)}")
        else:
            self.conversation_history[project_id].append({
                "query": user_query,
                "response": cleaned_response
            })
            
        total_duration = time.time() - start_time
        agent_logger.info(f"Total process_query timeframe: {total_duration:.4f}s")
        yield "data: [DONE]\n\n"

    def _route_query(self, query: str, project_id: str) -> list:
        """
        Deterministic keyword-based tool router.
        Returns an ordered list of tool action dicts to execute.
        
        This replaces the LLM-based planning loop which was unreliable:
        the real DeepSeek-R1 model doesn't follow the strict Action: format
        and was generating wrong actions (e.g. search_backlog(query='test query')).
        """
        q = query.lower()
        actions = []

        # --- Task Mutation / Modification check ---
        mutation_keywords = ["change", "update", "set", "mark", "move", "assign", "transition"]
        is_mutation = any(kw in q for kw in mutation_keywords)
        question_words = ["what", "show", "list", "who", "which", "how", "find", "search", "get"]
        starts_with_question = any(q.startswith(qw) for qw in question_words)

        if is_mutation and not starts_with_question:
            target_status = None
            target_priority = None
            target_assignee = None

            # Detect target status
            if any(kw in q for kw in ["to done", "as done", "status done", "move to done", "change to done", "mark done", "mark as done", "completed", "complete"]):
                target_status = "DONE"
            elif any(kw in q for kw in ["to in progress", "to in_progress", "as in progress", "status in progress", "mark as in progress", "mark in progress", "move to in progress", "wip"]):
                target_status = "IN_PROGRESS"
            elif any(kw in q for kw in ["to in review", "to in_review", "as in review", "status in review", "mark as in review", "mark in review", "move to in review"]):
                target_status = "IN_REVIEW"
            elif any(kw in q for kw in ["to todo", "to to do", "as todo", "status todo", "mark as todo", "mark todo", "move to todo"]):
                target_status = "TODO"
            else:
                if "done" in q or "completed" in q or "complete" in q:
                    target_status = "DONE"
                elif "in progress" in q or "in_progress" in q:
                    target_status = "IN_PROGRESS"
                elif "in review" in q or "in_review" in q:
                    target_status = "IN_REVIEW"
                elif "todo" in q or "to do" in q:
                    target_status = "TODO"

            # Detect target priority
            if "critical" in q:
                target_priority = "CRITICAL"
            elif "high" in q:
                target_priority = "HIGH"
            elif "medium" in q:
                target_priority = "MEDIUM"
            elif "low" in q:
                target_priority = "LOW"

            # Detect target assignee
            if "assign" in q:
                assign_match = re.search(r"to\s+([a-zA-Z0-9_\-]+)", query, re.IGNORECASE)
                if assign_match:
                    possible_assignee = assign_match.group(1)
                    if possible_assignee.upper() not in ["TODO", "DONE", "IN_PROGRESS", "IN_REVIEW", "IN-PROGRESS", "IN-REVIEW", "REVIEW", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                        target_assignee = possible_assignee
                if not target_assignee:
                    assign_match_2 = re.search(r"assign\s+([a-zA-Z0-9_\-]+)", query, re.IGNORECASE)
                    if assign_match_2:
                        possible_assignee = assign_match_2.group(1)
                        if possible_assignee.lower() not in ["task", "ticket", "issue"]:
                            target_assignee = possible_assignee

            # Detect task title or ID
            task_title_or_id = None
            is_bulk_update = any(phrase in q for phrase in ["all task", "all the task", "every task", "all ticket", "all issue"])
            
            if not is_bulk_update:
                quote_match = re.search(r"(?:task|ticket|issue)\s+['\"]([^'\"]+)['\"]", query, re.IGNORECASE)
                if quote_match:
                    task_title_or_id = quote_match.group(1)
                else:
                    unquote_match = re.search(r"(?:task|ticket|issue)\s+(.+?)\s+(?:to|as|status|priority)", query, re.IGNORECASE)
                    if unquote_match:
                        potential_title = unquote_match.group(1).strip()
                        if potential_title.lower() not in ["from", "all"]:
                            task_title_or_id = potential_title

            # Detect filters for bulk update
            filter_status = None
            try:
                # Find indexes to distinguish filter status from target status
                if "todo" in q and target_status != "TODO":
                    filter_status = "TODO"
                elif "to do" in q and target_status != "TODO":
                    filter_status = "TODO"
                elif "in progress" in q and target_status != "IN_PROGRESS":
                    filter_status = "IN_PROGRESS"
                elif "in review" in q and target_status != "IN_REVIEW":
                    filter_status = "IN_REVIEW"
            except Exception:
                pass

            filter_sprint = None
            sprint_match = re.search(r"sprint\s+(\w+)", q)
            if sprint_match:
                filter_sprint = f"Sprint {sprint_match.group(1).capitalize()}"

            if target_status or target_priority or target_assignee:
                actions.append({
                    "name": "modify_tasks",
                    "args": {
                        "target_status": target_status,
                        "target_priority": target_priority,
                        "target_assignee": target_assignee,
                        "task_title_or_id": task_title_or_id,
                        "filter_status": filter_status,
                        "filter_sprint": filter_sprint,
                        "original_query": query
                    },
                    "thought": "Detected task modification request. Routing to modify_tasks tool."
                })
                return actions

        # --- Document / Requirements queries ---
        doc_keywords = ["requirements", "requirement", "prd", "scope", "spec", "specification",
                        "document", "documentation", "feature", "user story"]
        retro_keywords = ["retrospective", "retro", "went well", "improvement"]
        sprint_doc_keywords = ["sprint goal", "sprint scope", "sprint document"]

        if any(kw in q for kw in retro_keywords):
            actions.append({
                "name": "read_document_chunk",
                "args": {"doc_id": "retro", "chunk_index": 0},
                "thought": "Query relates to retrospectives. Retrieving retrospective document chunks from the workspace database."
            })
        elif any(kw in q for kw in sprint_doc_keywords):
            actions.append({
                "name": "read_document_chunk",
                "args": {"doc_id": "sprint", "chunk_index": 0},
                "thought": "Query relates to sprint scope or goals. Retrieving sprint document data."
            })
        elif any(kw in q for kw in doc_keywords):
            actions.append({
                "name": "read_document_chunk",
                "args": {"doc_id": "requirements", "chunk_index": 0},
                "thought": "Query relates to project requirements or documentation. Retrieving requirement document chunks."
            })

        # --- Workload / Team metrics queries ---
        workload_keywords = ["workload", "metrics", "capacity", "engineer", "team load",
                             "who is working", "who has", "assignee", "assigned to", "task count"]
        if any(kw in q for kw in workload_keywords):
            actions.append({
                "name": "get_team_workload_metrics",
                "args": {"project_id": project_id},
                "thought": "Query relates to team workload or task distribution. Aggregating task metrics per team member."
            })

        # --- Backlog / Task search queries ---
        # Build smart args from query keywords
        backlog_keywords = ["blocker", "blocked", "task", "issue", "ticket", "sprint",
                            "progress", "status", "in progress", "done", "todo", "backlog",
                            "priority", "high priority", "overdue", "deadline"]
        needs_backlog = any(kw in q for kw in backlog_keywords) or not actions

        if needs_backlog:
            status_val = None
            sprint_val = None

            # Detect status filters
            if "block" in q:
                status_val = "BLOCKED"
            elif "in progress" in q or "in_progress" in q or "wip" in q:
                status_val = "IN_PROGRESS"
            elif "done" in q or "complete" in q or "finish" in q:
                status_val = "DONE"
            elif "todo" in q or "not started" in q:
                status_val = "TODO"

            # Detect sprint name (e.g. "Sprint 8", "sprint 2")
            sprint_match = re.search(r"sprint\s+(\w+)", q)
            if sprint_match:
                sprint_val = f"Sprint {sprint_match.group(1).capitalize()}"

            search_args: Dict[str, Any] = {"query": query}
            if status_val:
                search_args["status"] = status_val
            if sprint_val:
                search_args["sprint"] = sprint_val

            actions.append({
                "name": "search_backlog",
                "args": search_args,
                "thought": f"Searching workspace backlog for relevant tasks{' with status ' + status_val if status_val else ''}{' in ' + sprint_val if sprint_val else ''}."
            })

        return actions

    def _parse_tool_action(self, response_text: str) -> dict:
        action_match = re.search(r"Action:\s*(\w+)\((.*)\)", response_text)
        if not action_match:
            return {"name": "search_backlog", "args": {}}

        name = action_match.group(1)
        args_str = action_match.group(2)
        args: Dict[str, Any] = {}
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

    def _deduplicate_and_rank(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for item in items:
            if not item:
                continue
            key = (item.get("entity_type"), item.get("source_id"))
            if key not in seen:
                seen.add(key)
                metadata = item.get("metadata")
                if metadata is None:
                    metadata = {}
                if not metadata.get("citation_hash"):
                    metadata["citation_hash"] = f"cit_{uuid.uuid4().hex[:5]}"
                item["metadata"] = metadata
                deduped.append(item)
        return deduped

    def _assemble_and_prune_context(self, items: List[Dict[str, Any]], max_tokens: int = 4000) -> str:
        context_blocks = []
        current_tokens = 0
        
        for item in items:
            metadata = item.get("metadata", {})
            citation_hash = metadata.get("citation_hash", "")
            if not citation_hash:
                citation_hash = f"cit_{uuid.uuid4().hex[:5]}"
                metadata["citation_hash"] = citation_hash

            entity_type = item.get("entity_type", "UNKNOWN")
            title = metadata.get("title") or metadata.get("filename") or metadata.get("task_ref") or "N/A"
            status = metadata.get("status") or "N/A"
            sprint = metadata.get("sprint") or "N/A"
            content_snippet = item.get("content_snippet", "")
            
            block = (
                f"=========================================\n"
                f"RETRIEVED WORKSPACE CONTEXT ELEMENT: [ID: {citation_hash}]\n"
                f"Type: {entity_type} | Title: {title}\n"
                f"Status: {status} | Sprint: {sprint}\n"
                f"Content: {content_snippet}\n"
                f"=========================================\n"
            )
            
            block_tokens = self.count_tokens(block)
            if current_tokens + block_tokens > max_tokens:
                agent_logger.info(f"Context pruned: reached token limit at {current_tokens} tokens.")
                break
                
            context_blocks.append(block)
            current_tokens += block_tokens
            
        return "".join(context_blocks)

    async def _tool_search_backlog(self, project_id: str, args: dict) -> List[Dict[str, Any]]:
        """
        Primary backlog search tool.
        Strategy: MongoDB is ALWAYS the authoritative source for task counts and lists.
        Vector/keyword hybrid search provides relevance ranking signal.
        Results are blended via Reciprocal Rank Fusion (RRF).
        """
        query = args.get("query")
        status_val = args.get("status")
        sprint_title = args.get("sprint")

        # --- Step 1: Vector + keyword hybrid search (TASK-scoped, low threshold for inclusivity) ---
        vector_results = []
        if query:
            try:
                vector_results = self.hybrid_retriever.search_hybrid(
                    query=query,
                    project_id=project_id,
                    limit=20,
                    entity_type="TASK",
                    similarity_threshold=0.35
                )
                agent_logger.info(f"Vector/keyword search returned {len(vector_results)} TASK results for query='{query}'")
            except Exception as e:
                agent_logger.error(f"Hybrid search error in _tool_search_backlog: {str(e)}")

        # --- Step 2: MongoDB direct fetch (always authoritative for task list and counts) ---
        # Build filters: always start with project_id scope
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

        mongo_tasks = await Task.find(*filters).to_list()
        agent_logger.info(f"MongoDB fetch returned {len(mongo_tasks)} tasks for project={project_id}, status={status_val}, sprint={sprint_title}")

        mongo_results = []
        for t in mongo_tasks:
            assignee_name = "Unassigned"
            if t.assigned_to_id:
                try:
                    user = await User.get(t.assigned_to_id)
                    if user:
                        assignee_name = user.name if hasattr(user, 'name') and user.name else (
                            user.role.value if hasattr(user.role, 'value') else str(user.role)
                        )
                except Exception:
                    pass

            status_str = t.status.value if hasattr(t.status, 'value') else str(t.status)
            priority_str = t.priority.value if hasattr(t.priority, 'value') else str(t.priority)
            sprint_str = t.sprint_id or ""
            description_str = (t.description or "")[:200]  # cap to avoid token bloat

            content = (
                f"Title: {t.title} | "
                f"Status: {status_str} | "
                f"Priority: {priority_str} | "
                f"Assignee: {assignee_name} | "
                f"Sprint: {sprint_str} | "
                f"Description: {description_str}"
            )
            mongo_results.append({
                "entity_type": "TASK",
                "source_id": str(t.id),
                "project_id": project_id,
                "created_at": t.created_at.isoformat() if t.created_at else "",
                "content_snippet": content,
                "metadata": {
                    "title": t.title,
                    "status": status_str,
                    "priority": priority_str,
                    "sprint": sprint_title or sprint_str,
                    "owner": assignee_name,
                    "code": priority_str,
                    "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
                }
            })

        # --- Step 3: Blend MongoDB results with vector results via RRF ---
        # MongoDB is always used; vector adds relevance-based re-ranking on top.
        if vector_results and mongo_results:
            blended = self.hybrid_retriever._reciprocal_rank_fusion([mongo_results, vector_results])
            agent_logger.info(f"RRF blended result count: {len(blended)}")
            return blended
        elif mongo_results:
            return mongo_results
        else:
            return vector_results

    async def _tool_read_document_chunk(self, project_id: str, args: dict) -> List[Dict[str, Any]]:
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
                
        # Fetch from LanceDB
        try:
            results = self.lancedb_manager.knowledge_table.search().where(f"project_id = '{project_id}' AND source_id = '{doc_id}'").to_list()
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
            agent_logger.error(f"Failed to fetch document chunk from LanceDB: {str(e)}")
            
        # Fallback to MongoDB
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

    async def _tool_modify_tasks(self, project_id: str, args: dict, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        from app.models.task import Task, TaskStatus, TaskPriority
        from app.models.project import Project
        from app.models.user import User
        from app.models.activity_log import ActivityLog
        from app.routers.task_router import emit_sync_event
        from beanie import PydanticObjectId

        target_status = args.get("target_status")
        target_priority = args.get("target_priority")
        target_assignee = args.get("target_assignee")
        task_title_or_id = args.get("task_title_or_id")
        filter_status = args.get("filter_status")
        filter_sprint = args.get("filter_sprint")

        # 1. Resolve assignee if target_assignee is provided
        assignee_id = None
        assignee_name = None
        if target_assignee:
            if target_assignee.lower() == "unassigned":
                assignee_id = None
                assignee_name = "Unassigned"
            else:
                user = await User.find_one({"name": {"$regex": target_assignee, "$options": "i"}})
                if not user:
                    user = await User.find_one({"email": {"$regex": target_assignee, "$options": "i"}})
                if user:
                    assignee_id = str(user.id)
                    assignee_name = user.name
                else:
                    agent_logger.warning(f"Assignee target '{target_assignee}' not found in database.")
                    return [{
                        "entity_type": "MUTATION_ERROR",
                        "source_id": "error",
                        "content_snippet": f"Could not find a user matching the assignee name '{target_assignee}'. No tasks were updated.",
                        "metadata": {"title": "Error"}
                    }]

        # 2. Get sprint ID if filter_sprint is provided
        sprint_id = None
        if filter_sprint:
            project = await Project.get(project_id)
            if project and project.sprints:
                for s in project.sprints:
                    if s.title.lower() == filter_sprint.lower():
                        sprint_id = s.id
                        break
            if not sprint_id:
                sprint_id = filter_sprint

        # 3. Build MongoDB filters
        filters: List[Any] = [Task.project_id == project_id]

        if task_title_or_id:
            if PydanticObjectId.is_valid(task_title_or_id):
                filters.append(Task.id == PydanticObjectId(task_title_or_id))
            else:
                filters.append({"title": {"$regex": task_title_or_id, "$options": "i"}})
        else:
            original_q = args.get("original_query", "").lower()
            if not filter_status and not filter_sprint and "all" not in original_q and "every" not in original_q:
                return [{
                    "entity_type": "MUTATION_ERROR",
                    "source_id": "error",
                    "content_snippet": "Please specify which task you want to modify (e.g. by title, ID, or say 'all tasks').",
                    "metadata": {"title": "Error"}
                }]
            if filter_status:
                for enum_val in TaskStatus:
                    if enum_val.value == filter_status.upper():
                        filters.append(Task.status == enum_val)
                        break
            if sprint_id:
                filters.append(Task.sprint_id == sprint_id)

        tasks = await Task.find(*filters).to_list()
        if not tasks and task_title_or_id:
            try:
                closest_results = self.hybrid_retriever.search_hybrid(
                    query=task_title_or_id,
                    project_id=project_id,
                    limit=1,
                    entity_type="TASK",
                    similarity_threshold=0.35
                )
                if closest_results:
                    closest_task_id = closest_results[0].get("source_id")
                    if closest_task_id:
                        task = await Task.get(closest_task_id)
                        if task:
                            tasks = [task]
            except Exception as e:
                agent_logger.error(f"Fallback hybrid search failed for modify task: {str(e)}")

        if not tasks:
            return [{
                "entity_type": "MUTATION_RESULT",
                "source_id": "no_tasks",
                "content_snippet": "No matching tasks were found to modify.",
                "metadata": {"title": "No Tasks Modified"}
            }]

        operator_name = "Copilot"
        if user_id:
            try:
                operator = await User.get(user_id)
                if operator:
                    operator_name = operator.name
            except Exception:
                pass

        results = []
        for task in tasks:
            old_status = task.status.value if hasattr(task.status, 'value') else str(task.status)
            old_priority = task.priority.value if hasattr(task.priority, 'value') else str(task.priority)
            
            changes = []
            if target_status:
                for enum_val in TaskStatus:
                    if enum_val.value == target_status.upper():
                        task.status = enum_val
                        changes.append(f"status changed from {old_status} to {enum_val.value}")
                        break

            if target_priority:
                for enum_val in TaskPriority:
                    if enum_val.value == target_priority.upper():
                        task.priority = enum_val
                        changes.append(f"priority changed from {old_priority} to {enum_val.value}")
                        break

            if target_assignee:
                task.assigned_to_id = assignee_id
                changes.append(f"assigned to {assignee_name or 'Unassigned'}")

            if changes:
                task.updated_at = datetime.now(timezone.utc)
                await task.save()

                detail_str = f"Updated task '{task.title}': " + ", ".join(changes)
                activity = ActivityLog(
                    task_id=str(task.id),
                    project_id=task.project_id,
                    user_id=user_id or "copilot",
                    user_name=operator_name,
                    action="task_updated",
                    detail=detail_str
                )
                await activity.insert()

                await emit_sync_event("update", task.model_dump())

                results.append({
                    "entity_type": "TASK",
                    "source_id": str(task.id),
                    "content_snippet": f"Successfully updated task '{task.title}' (ID: {task.id}): " + ", ".join(changes),
                    "metadata": {
                        "title": task.title,
                        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                        "priority": task.priority.value if hasattr(task.priority, 'value') else str(task.priority),
                        "owner": assignee_name or (await self._get_user_name(task.assigned_to_id)),
                        "citation_hash": f"cit_{uuid.uuid4().hex[:5]}"
                    }
                })

        return results

    async def _get_user_name(self, user_id: Optional[str]) -> str:
        if not user_id:
            return "Unassigned"
        try:
            from app.models.user import User
            user = await User.get(user_id)
            if user:
                return user.name
        except Exception:
            pass
        return "Unknown"
