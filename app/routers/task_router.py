# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from datetime import datetime, timezone
from app.models.task import Task, TaskStatus
from app.models.user import User
from app.schemas.task_schema import TaskCreateSchema, TaskUpdateSchema, TaskResponseSchema
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])

def make_task_response(task: Task, assigned_to_res: Optional[User]) -> TaskResponseSchema:
    """Helper to convert Beanie Task to TaskResponseSchema safely."""
    task_dict = task.model_dump()
    task_dict.pop("id", None)
    return TaskResponseSchema(
        **task_dict,
        id=task.id,
        assigned_to=assigned_to_res
    )

@router.post("/", response_model=TaskResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_new_task(payload: TaskCreateSchema, current_user: User = Depends(get_current_user)):
    """Allows clients and managers (or developers) to raise an issue or create a task."""
    # Check if assigned_to_id is valid
    if payload.assigned_to_id:
        assignee = await User.get(payload.assigned_to_id)
        if not assignee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assignee user with ID {payload.assigned_to_id} not found"
            )
            
    new_task = Task(**payload.model_dump())
    await new_task.insert()
    
    # Construct task response manual mapping
    assigned_to_res = None
    if new_task.assigned_to_id:
        user = await User.get(new_task.assigned_to_id)
        if user:
            assigned_to_res = user
            
    return make_task_response(new_task, assigned_to_res)

@router.get("/", response_model=List[TaskResponseSchema])
async def get_all_tasks(project_id: str, current_user: User = Depends(get_current_user)):
    """Fetches all tasks/bugs for a specific project to display on the Kanban board."""
    tasks = await Task.find(Task.project_id == project_id).to_list()
    
    response_list = []
    for task in tasks:
        assigned_to_res = None
        if task.assigned_to_id:
            user = await User.get(task.assigned_to_id)
            if user:
                assigned_to_res = user
                
        response_list.append(make_task_response(task, assigned_to_res))
    return response_list

@router.get("/{task_id}", response_model=TaskResponseSchema)
async def get_task_by_id(task_id: str, current_user: User = Depends(get_current_user)):
    """Retrieves a single task by its ID."""
    task = await Task.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
        
    assigned_to_res = None
    if task.assigned_to_id:
        user = await User.get(task.assigned_to_id)
        if user:
            assigned_to_res = user
            
    return make_task_response(task, assigned_to_res)

@router.patch("/{task_id}/status", response_model=TaskResponseSchema)
async def update_task_status(task_id: str, current_status: TaskStatus, current_user: User = Depends(get_current_user)):
    """Allows developers or managers to progress a ticket through the workflow stages."""
    task = await Task.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task.status = current_status
    task.updated_at = datetime.now(timezone.utc)
    await task.save()
    
    assigned_to_res = None
    if task.assigned_to_id:
        user = await User.get(task.assigned_to_id)
        if user:
            assigned_to_res = user
            
    return make_task_response(task, assigned_to_res)

@router.patch("/{task_id}", response_model=TaskResponseSchema)
async def update_task(task_id: str, payload: TaskUpdateSchema, current_user: User = Depends(get_current_user)):
    """Allows updates to details like title, description, estimates, and assignees."""
    task = await Task.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    update_data = payload.model_dump(exclude_unset=True)
    
    # Validate assignee if update_data has assigned_to_id
    if "assigned_to_id" in update_data:
        assigned_to_id = update_data.get("assigned_to_id")
        if assigned_to_id:
            assignee = await User.get(assigned_to_id)
            if not assignee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Assignee user with ID {assigned_to_id} not found"
                )
                
    for field, value in update_data.items():
        setattr(task, field, value)
        
    task.updated_at = datetime.now(timezone.utc)
    await task.save()
    
    assigned_to_res = None
    if task.assigned_to_id:
        user = await User.get(task.assigned_to_id)
        if user:
            assigned_to_res = user
            
    return make_task_response(task, assigned_to_res)
