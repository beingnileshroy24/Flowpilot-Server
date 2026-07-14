from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any

from app.schemas.ai_schema import DuplicateCheckRequest, DuplicateCheckResponse, DuplicateMatch
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.ai.core.embedder import ModernBertEmbedderSingleton
from app.ai.storage.lancedb_client import LanceDBManager

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
