from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Any, Dict
from app.services.planner_service import PlannerService
from app.models.task import Task, TaskStatus

router = APIRouter(prefix="/api/v1/planner", tags=["Planner"])

class GeneratePlanRequest(BaseModel):
    project_id: str
    target_sprint_id: str
    capacity_override: Optional[float] = None

class CommitPlanRequest(BaseModel):
    project_id: str
    target_sprint_id: str
    task_assignments: Dict[str, str]  # mapping task_id -> assignee_id

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

@router.post("/commit")
async def commit_sprint_plan(req: CommitPlanRequest):
    """
    Commits an AI-generated sprint plan, applying it directly to the Task collections.
    """
    try:
        updated_count = 0
        for task_id, assignee_id in req.task_assignments.items():
            task = await Task.get(task_id)
            if task and task.project_id == req.project_id:
                task.sprint_id = req.target_sprint_id
                task.assigned_to_id = assignee_id
                if task.status == TaskStatus.TODO:
                    # Explicitly ensure status is TODO (or we can just keep it whatever it is, unless DONE)
                    pass 
                await task.save()
                updated_count += 1
        return {"status": "success", "updated_tasks": updated_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
