from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.models.comment import Comment
from app.models.activity_log import ActivityLog
from app.models.user import User
from app.schemas.comment_schema import CommentCreateSchema, CommentResponseSchema
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/comments", tags=["Comments"])


@router.post("/", response_model=CommentResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_comment(payload: CommentCreateSchema, current_user: User = Depends(get_current_user)):
    """Creates a new comment on a task."""
    from app.models.task import Task

    task = await Task.get(payload.task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    comment = Comment(
        task_id=payload.task_id,
        author_id=str(current_user.id),
        author_name=current_user.name,
        content=payload.content,
    )
    await comment.insert()

    # Sync to LanceDB knowledge base
    from app.services.sync_queue import push_to_sync_queue
    push_to_sync_queue("COMMENT", str(comment.id), "create", task.project_id)

    # Create activity log entry
    activity = ActivityLog(
        task_id=payload.task_id,
        project_id=task.project_id,
        user_id=str(current_user.id),
        user_name=current_user.name,
        action="comment_added",
        detail=f"Added a comment on '{task.title}'"
    )
    await activity.insert()

    return comment


@router.get("/", response_model=List[CommentResponseSchema])
async def get_comments(task_id: str, current_user: User = Depends(get_current_user)):
    """Fetches all comments for a specific task, ordered by creation time."""
    comments = await Comment.find(Comment.task_id == task_id).sort("+created_at").to_list()
    return comments


@router.delete("/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comment(comment_id: str, current_user: User = Depends(get_current_user)):
    """Deletes a comment. Only the author can delete their own comment."""
    comment = await Comment.get(comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found"
        )

    if comment.author_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own comments."
        )

    from app.services.sync_queue import push_to_sync_queue
    from app.models.task import Task
    task = await Task.get(comment.task_id)
    project_id = task.project_id if task else ""

    await comment.delete()
    
    # Sync deletion to LanceDB knowledge base
    push_to_sync_queue("COMMENT", comment_id, "delete", project_id)
    return None
