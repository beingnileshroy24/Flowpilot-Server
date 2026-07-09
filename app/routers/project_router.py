from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.models.project import Project
from app.models.user import User
from app.schemas.project_schema import ProjectCreateSchema, ProjectUpdateSchema, ProjectResponseSchema
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/projects", tags=["Projects"])

@router.post("/", response_model=ProjectResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreateSchema, current_user: User = Depends(get_current_user)):
    """Creates a new project owned by the current user."""
    new_project = Project(
        name=payload.name,
        description=payload.description,
        github_frontend=payload.github_frontend,
        github_backend=payload.github_backend,
        test_server=payload.test_server,
        prod_server=payload.prod_server,
        test_mongodb_url=payload.test_mongodb_url,
        prod_mongodb_url=payload.prod_mongodb_url,
        owner_id=str(current_user.id)
    )
    await new_project.insert()
    return new_project

@router.get("/", response_model=List[ProjectResponseSchema])
async def list_projects(current_user: User = Depends(get_current_user)):
    """Lists all projects in the workspace."""
    projects = await Project.find_all().to_list()
    return projects

@router.get("/{project_id}", response_model=ProjectResponseSchema)
async def get_project(project_id: str, current_user: User = Depends(get_current_user)):
    """Gets a project by its ID."""
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    return project

@router.patch("/{project_id}", response_model=ProjectResponseSchema)
async def update_project(project_id: str, payload: ProjectUpdateSchema, current_user: User = Depends(get_current_user)):
    """Updates details of an existing project (e.g. repo links, servers, MongoDB connection string)."""
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    
    await project.save()
    return project
