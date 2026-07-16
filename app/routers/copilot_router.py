from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.copilot_chat import CopilotChat
from app.ai.core.agent_engine import AgentEngine

router = APIRouter(
    prefix="/api/v1/copilot",
    tags=["Copilot"]
)

class CopilotQueryRequest(BaseModel):
    prompt: str
    contextScope: dict

class CreateChatRequest(BaseModel):
    project_id: str
    title: str

class ChatQueryRequest(BaseModel):
    prompt: str

agent_engine = AgentEngine()

@router.post("/query")
async def copilot_query(
    payload: CopilotQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Main endpoint for the Agentic Workspace Copilot.
    Runs the agent orchestration process query loop.
    """
    if not payload.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt cannot be empty."
        )
        
    project_id = payload.contextScope.get("project_id", "")
    return StreamingResponse(
        agent_engine.process_query(payload.prompt, project_id, user_id=str(current_user.id)),
        media_type="text/event-stream"
    )

@router.get("/chats")
async def get_chats(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Lists all prior Copilot chat sessions for the current user in this project.
    """
    try:
        chats = await CopilotChat.find(
            CopilotChat.project_id == project_id,
            CopilotChat.user_id == str(current_user.id)
        ).sort("-updated_at").to_list()
        
        # Format the list output cleanly
        return [{
            "id": str(c.id),
            "project_id": c.project_id,
            "title": c.title,
            "updated_at": c.updated_at.isoformat(),
            "created_at": c.created_at.isoformat()
        } for c in chats]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load chats: {str(e)}"
        )

@router.get("/chats/{chat_id}")
async def get_chat_details(
    chat_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves full message details of a specific chat session.
    """
    chat = await CopilotChat.get(chat_id)
    if not chat or chat.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat history session not found."
        )
    return chat

@router.post("/chats")
async def create_chat(
    payload: CreateChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Creates a new empty Copilot chat session document.
    """
    try:
        chat = CopilotChat(
            project_id=payload.project_id,
            user_id=str(current_user.id),
            title=payload.title
        )
        await chat.insert()
        return chat
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat session: {str(e)}"
        )

@router.post("/chats/{chat_id}/query")
async def copilot_chat_query(
    chat_id: str,
    payload: ChatQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Streams a prompt query inside a specific, active Copilot chat session.
    """
    chat = await CopilotChat.get(chat_id)
    if not chat or chat.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat history session not found."
        )
        
    if not payload.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt cannot be empty."
        )
        
    return StreamingResponse(
        agent_engine.process_query(payload.prompt, chat.project_id, chat_id=str(chat.id), user_id=str(current_user.id)),
        media_type="text/event-stream"
    )
