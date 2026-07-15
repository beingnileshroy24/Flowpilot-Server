from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.models.user import User
from app.ai.core.agent_engine import AgentEngine

router = APIRouter(
    prefix="/api/v1/copilot",
    tags=["Copilot"]
)

class CopilotQueryRequest(BaseModel):
    prompt: str
    contextScope: dict

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
        agent_engine.process_query(payload.prompt, project_id),
        media_type="text/event-stream"
    )
