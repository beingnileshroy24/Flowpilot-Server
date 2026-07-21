from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Any, Dict
from app.services.planner_service import PlannerService

router = APIRouter(prefix="/api/v1/planner", tags=["Planner"])

class GeneratePlanRequest(BaseModel):
    project_id: str
    target_sprint_id: str
    capacity_override: Optional[float] = None

@router.post("/generate")
async def generate_sprint_plan(req: GeneratePlanRequest):
    """
    Generates an optimized sprint plan using Google OR-Tools CP-SAT Solver 
    and returns MLX-generated explanations for the trade-offs.
    """
    try:
        plan = await PlannerService.generate_sprint_plan(
            project_id=req.project_id,
            target_sprint_id=req.target_sprint_id,
            capacity_override=req.capacity_override
        )
        return plan
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
