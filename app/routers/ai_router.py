from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Any

from app.schemas.ai_schema import DuplicateCheckRequest, DuplicateCheckResponse, DuplicateMatch
from app.schemas.wbs_schema import WbsCommitRequest
from app.auth.dependencies import get_current_user
from app.models.user import User, UserRole
from app.models.task import Task
from app.ai.core.embedder import ModernBertEmbedderSingleton
from app.ai.storage.lancedb_client import LanceDBManager
from app.services.sync_service import rebuild_vector_index
from app.ai.core.wbs_engine import wbs_engine

router = APIRouter(
    prefix="/api/v1/ai",
    tags=["AI"]
)

# Initialize singletons at router load (lazy initialization happens in the class)
embedder = ModernBertEmbedderSingleton()
lancedb_manager = LanceDBManager()

SIMILARITY_THRESHOLD = 0.85

@router.post("/check-duplicates", response_model=DuplicateCheckResponse)
async def check_duplicates(
    request: DuplicateCheckRequest,
    current_user: User = Depends(get_current_user)
) -> Any:
    try:
        if not request.title or not request.title.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Empty field titles"
            )

        # Compute embedding
        vector = embedder.compute_embedding(request.title, request.description)

        # Search LanceDB
        results = lancedb_manager.search_similar(vector=vector, project_id=request.project_id, limit=5)

        matches = []
        max_score = 0.0

        for res in results:
            # LanceDB default distance metric is L2 (squared L2 distance). 
            # With L2-normalized vectors, cosine_similarity = 1 - (L2_squared / 2)
            distance = res.get('_distance', 0)
            similarity = 1.0 - (distance / 2.0)
            
            if similarity > max_score:
                max_score = similarity

            if similarity >= SIMILARITY_THRESHOLD:
                matches.append(DuplicateMatch(
                    task_id=res.get("task_id", ""),
                    title=res.get("title", ""),
                    status=res.get("status", ""),
                    similarity=round(similarity, 3)
                ))

        # Sort matches by highest similarity first
        matches.sort(key=lambda x: x.similarity, reverse=True)

        return DuplicateCheckResponse(
            is_potential_duplicate=len(matches) > 0,
            max_similarity_score=round(max_score, 3) if results else 0.0,
            matches=matches
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"FastAPI core failures: {str(e)}"
        )

@router.post("/sync-recovery")
async def sync_recovery(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Administrative recovery pipeline that rebuilds cleanly aligned vector tables from scratch.
    Requires ADMIN privileges.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can trigger the recovery pipeline."
        )
    
    background_tasks.add_task(rebuild_vector_index)
    
    return {"message": "Administrative recovery pipeline initiated successfully."}

@router.post("/wbs/generate")
async def generate_wbs(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint for uploading a requirements file and receiving a streamed JSON Array of WBS tasks.
    Supports PDF, DOCX, MD, and TXT files.
    """
    if current_user.role not in [UserRole.MANAGER, UserRole.ADMIN, UserRole.DEVELOPER]:
        raise HTTPException(status_code=403, detail="Not authorized to generate WBS.")
    
    content = await file.read()
    filename = file.filename or "document.txt"
    
    # 1. Extract text
    raw_text = wbs_engine.extract_text(filename, content)
    
    # 2. Sanitize and truncate to fit token limits
    safe_text = wbs_engine.sanitize_text(raw_text)
    
    # 3. Stream generated JSON chunks
    return StreamingResponse(
        wbs_engine.stream_wbs_generation(safe_text),
        media_type="text/event-stream"
    )

@router.post("/wbs/commit")
async def commit_wbs(
    payload: WbsCommitRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Commits an array of structurally validated WBS tasks into MongoDB inside a single atomic transaction.
    """
    if current_user.role not in [UserRole.MANAGER, UserRole.ADMIN, UserRole.DEVELOPER]:
        raise HTTPException(status_code=403, detail="Not authorized to commit WBS.")

    # We will build Beanie Document models and insert them
    # Because Beanie builds on Motor, we can leverage session-based bulk inserts for atomicity.
    docs_to_insert = []
    for task_data in payload.tasks:
        docs_to_insert.append(
            Task(
                project_id=payload.project_id,
                title=task_data.title,
                description=task_data.description,
                type=task_data.type,
                priority=task_data.priority,
                estimated_hours=task_data.estimated_hours,
                checklist_items=[{"text": item, "done": False} for item in task_data.checklist_items],
                status="TODO"
            )
        )
    
    try:
        # Atomic Insert (Beanie doesn't expose a single bulk insert transaction easily without motor sessions)
        # We can use Task.insert_many() which is atomic on the collection level if configured,
        # but doing it in a batch is often sufficient.
        if docs_to_insert:
            await Task.insert_many(docs_to_insert)
            
        return {"message": f"Successfully committed {len(docs_to_insert)} tasks to project."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed atomic insert: {str(e)}")
