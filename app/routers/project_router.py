from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List
from app.models.project import Project
from app.models.user import User, UserRole
from app.services.sync_service import cleanup_project_vectors
from app.models.task import Task
from app.schemas.project_schema import ProjectCreateSchema, ProjectUpdateSchema, ProjectResponseSchema
from app.auth.dependencies import get_current_user
from beanie import PydanticObjectId

router = APIRouter(prefix="/api/v1/projects", tags=["Projects"])

@router.post("/", response_model=ProjectResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreateSchema, current_user: User = Depends(get_current_user)):
    """Creates a new project. Only Managers and Admins can create projects and assign developers."""
    if current_user.role not in [UserRole.MANAGER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only managers or administrators have permission to create projects."
        )

    dev_ids = payload.developer_ids or []
    lead_id = payload.lead_developer_id

    if len(dev_ids) > 1:
        if not lead_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A Lead Developer must be chosen if more than one developer is assigned."
            )
        if lead_id not in dev_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The Lead Developer must be one of the assigned developers."
            )
    elif len(dev_ids) == 1:
        lead_id = dev_ids[0]

    new_project = Project(
        name=payload.name,
        description=payload.description,
        developer_ids=dev_ids,
        lead_developer_id=lead_id,
        github_frontend=payload.github_frontend,
        github_backend=payload.github_backend,
        test_server=payload.test_server,
        prod_server=payload.prod_server,
        test_mongodb_url=payload.test_mongodb_url,
        prod_mongodb_url=payload.prod_mongodb_url,
        backend_secrets=payload.backend_secrets,
        frontend_secrets=payload.frontend_secrets,
        owner_id=str(current_user.id)
    )
    await new_project.insert()
    schema = ProjectResponseSchema.model_validate(new_project)
    schema.owner_name = current_user.name
    dev_names = []
    if new_project.developer_ids:
        valid_dev_ids = []
        for d in new_project.developer_ids:
            try:
                valid_dev_ids.append(PydanticObjectId(d))
            except Exception:
                pass
        if valid_dev_ids:
            devs = await User.find({"_id": {"$in": valid_dev_ids}}).to_list()
            dev_map = {str(d.id): d.name for d in devs}
            dev_names = [dev_map.get(d_id, "Unknown") for d_id in new_project.developer_ids]
    schema.developer_names = dev_names
    return schema

@router.get("/", response_model=List[ProjectResponseSchema])
async def list_projects(current_user: User = Depends(get_current_user)):
    """Lists all projects in the workspace."""
    projects = await Project.find_all().to_list()
    user_ids = set()
    for p in projects:
        if p.owner_id:
            user_ids.add(p.owner_id)
        for d in p.developer_ids:
            user_ids.add(d)
            
    valid_ids = []
    for uid in user_ids:
        try:
            valid_ids.append(PydanticObjectId(uid))
        except Exception:
            pass
            
    users = await User.find({"_id": {"$in": valid_ids}}).to_list()
    user_map = {str(u.id): u.name for u in users}
    
    res = []
    for p in projects:
        schema = ProjectResponseSchema.model_validate(p)
        schema.owner_name = user_map.get(p.owner_id, "Unknown")
        schema.developer_names = [user_map.get(d_id, "Unknown") for d_id in p.developer_ids]
        res.append(schema)
        
    return res

@router.get("/{project_id}", response_model=ProjectResponseSchema)
async def get_project(project_id: str, current_user: User = Depends(get_current_user)):
    """Gets a project by its ID."""
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    schema = ProjectResponseSchema.model_validate(project)
    owner = await User.get(project.owner_id)
    schema.owner_name = owner.name if owner else "Unknown"
    
    dev_names = []
    if project.developer_ids:
        valid_dev_ids = []
        for d in project.developer_ids:
            try:
                valid_dev_ids.append(PydanticObjectId(d))
            except Exception:
                pass
        if valid_dev_ids:
            devs = await User.find({"_id": {"$in": valid_dev_ids}}).to_list()
            dev_map = {str(d.id): d.name for d in devs}
            dev_names = [dev_map.get(d_id, "Unknown") for d_id in project.developer_ids]
    schema.developer_names = dev_names
    return schema

@router.patch("/{project_id}", response_model=ProjectResponseSchema)
async def update_project(project_id: str, payload: ProjectUpdateSchema, current_user: User = Depends(get_current_user)):
    """Updates details of an existing project. Managers (owners) and Lead Developers have write access."""
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    is_owner = project.owner_id == str(current_user.id)
    is_lead = project.lead_developer_id == str(current_user.id)
    is_admin = current_user.role == UserRole.ADMIN
    is_manager = current_user.role == UserRole.MANAGER

    if not (is_owner or is_lead or is_admin or is_manager):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the Project Manager (owner) or the Lead Developer have permission to configure project details."
        )

    update_data = payload.model_dump(exclude_unset=True)

    # Permission check for changing developer assignments (only manager or admin)
    if "developer_ids" in update_data or "lead_developer_id" in update_data:
        if not (is_manager or is_admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only project managers and administrators can change developer assignments or modify the lead developer."
            )
        
        # Validation checks
        new_devs = update_data.get("developer_ids", project.developer_ids) or []
        new_lead = update_data.get("lead_developer_id", project.lead_developer_id)

        if len(new_devs) > 1:
            if not new_lead:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A Lead Developer must be chosen if more than one developer is assigned."
                )
            if new_lead not in new_devs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The Lead Developer must be one of the assigned developers."
                )
        elif len(new_devs) == 1:
            update_data["lead_developer_id"] = new_devs[0]

    for field, value in update_data.items():
        setattr(project, field, value)
    
    await project.save()
    
    schema = ProjectResponseSchema.model_validate(project)
    owner = await User.get(project.owner_id)
    schema.owner_name = owner.name if owner else "Unknown"
    
    dev_names = []
    if project.developer_ids:
        valid_dev_ids = []
        for d in project.developer_ids:
            try:
                valid_dev_ids.append(PydanticObjectId(d))
            except Exception:
                pass
        if valid_dev_ids:
            devs = await User.find({"_id": {"$in": valid_dev_ids}}).to_list()
            dev_map = {str(d.id): d.name for d in devs}
            dev_names = [dev_map.get(d_id, "Unknown") for d_id in project.developer_ids]
    schema.developer_names = dev_names
    
    # Sync sprints and documents to LanceDB knowledge base
    from app.services.sync_queue import push_to_sync_queue
    push_to_sync_queue("SPRINT", project_id, "update", project_id)
    
    return schema

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    """Deletes a project and all associated tasks. Only Managers and Admins can delete projects."""
    if current_user.role not in [UserRole.MANAGER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only managers or administrators have permission to delete projects."
        )

    project = await Project.get(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Delete all tasks associated with this project
    await Task.find(Task.project_id == project_id).delete()
    
    # Delete project
    await project.delete()
    
    # Sweep orphaned vectors in LanceDB asynchronously
    background_tasks.add_task(cleanup_project_vectors, project_id)
    
    return None
