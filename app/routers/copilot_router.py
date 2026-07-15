from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.models.user import User
from app.ai.core.react_engine import react_engine

router = APIRouter(
    prefix="/api/v1/copilot",
    tags=["Copilot"]
)

class CopilotQueryRequest(BaseModel):
    prompt: str
    contextScope: dict

@router.post("/query")
async def copilot_query(
    payload: CopilotQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Main endpoint for the Agentic Workspace Copilot.
    Executes a ReAct loop over local vector and MongoDB indexes, fusses results,
    and returns a Server-Sent Events (SSE) token stream.
    """
    if not payload.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt cannot be empty."
        )
        
    return StreamingResponse(
        react_engine.execute_query(payload.prompt, payload.contextScope),
        media_type="text/event-stream"
    )
