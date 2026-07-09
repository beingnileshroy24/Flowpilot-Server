from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import List, Optional
from datetime import datetime, timezone
import shutil
import uuid
import os
from app.models.task import Task, TaskStatus
from app.models.user import User, UserRole
from app.schemas.task_schema import TaskCreateSchema, TaskUpdateSchema, TaskResponseSchema
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])

@router.post("/upload")
async def upload_task_media(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """Handles uploading task attachments (images or videos)."""
    os.makedirs("uploads", exist_ok=True)
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing or invalid."
        )
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = f"uploads/{unique_filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"url": f"/uploads/{unique_filename}"}

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
    # Retrieve project
    from app.models.project import Project
    project = await Project.get(payload.project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {payload.project_id} not found"
        )

    if current_user.role in [UserRole.CLIENT, UserRole.MANAGER]:
        # Client and Manager cannot assign anyone. It automatically assigns to project lead.
        if not project.lead_developer_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This project does not have a lead developer assigned. A project lead is required to raise tickets."
            )
        payload.assigned_to_id = project.lead_developer_id
    else:
        # Developer or Admin is creating
        if payload.assigned_to_id:
            # If the creator is not the project lead and not admin, check if they are assigning to themselves
            if current_user.role != UserRole.ADMIN and str(current_user.id) != project.lead_developer_id:
                # If they are a developer in the project, they can only assign to themselves.
                if str(current_user.id) in project.developer_ids:
                    if payload.assigned_to_id != str(current_user.id):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Developers can only assign tasks to themselves."
                        )
                else:
                    # Developer is not in the project at all
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You are not a member of this project."
                    )
            else:
                # Admin or Project Lead is creating
                # Ensure assignee is in project.developer_ids or is project.lead_developer_id
                if payload.assigned_to_id != project.lead_developer_id and payload.assigned_to_id not in project.developer_ids:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Assignee must be a developer assigned to this project."
                    )

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
    
    # Create notification for workspace/assigned developers
    from app.models.notification import Notification
    notification = Notification(
        project_id=new_task.project_id,
        message=f"{current_user.name} raised a new {new_task.type.value.lower()}: '{new_task.title}'",
        created_by_name=current_user.name
    )
    await notification.insert()
    
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
    
    from app.models.project import Project
    project = await Project.get(task.project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project associated with this task was not found."
        )
    
    update_data = payload.model_dump(exclude_unset=True)
    
    # Validate assignee if update_data has assigned_to_id
    if "assigned_to_id" in update_data:
        new_assignee_id = update_data.get("assigned_to_id")
        
        if current_user.role == UserRole.ADMIN:
            # Admins can assign to anyone
            pass
        elif str(current_user.id) == project.lead_developer_id:
            # Project Lead can assign to any developer in the project or to themselves
            if new_assignee_id:
                if new_assignee_id != project.lead_developer_id and new_assignee_id not in project.developer_ids:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Assignee must be a developer assigned to this project."
                    )
        elif str(current_user.id) in project.developer_ids:
            # Developers in the project can only assign the task to themselves (or unassign themselves)
            if new_assignee_id and new_assignee_id != str(current_user.id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Developers can only assign tasks to themselves."
                )
        else:
            # Clients, Managers, or other developers not on the project cannot assign tasks
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to assign this task."
            )
            
        if new_assignee_id:
            assignee = await User.get(new_assignee_id)
            if not assignee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Assignee user with ID {new_assignee_id} not found"
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
