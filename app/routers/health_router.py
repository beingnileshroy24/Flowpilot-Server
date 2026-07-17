import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, TypedDict

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from beanie import PydanticObjectId

from app.auth.dependencies import get_current_user
from app.models.user import User, UserRole
from app.models.project import Project, Sprint
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.comment import Comment
from app.models.activity_log import ActivityLog
from app.models.sprint_prediction import SprintPrediction
from app.models.project_health import ProjectHealth
from app.models.notification import Notification
from app.ai.core.llm_manager import llm_manager

class TaskDelayRisk(TypedDict):
    task_id: str
    title: str
    delay_risk_score: float
    reasons: List[str]

class BurnoutRisk(TypedDict):
    developer_id: Any
    name: str
    workload_balance: float
    context_switching_count: int
    burnout_risk_level: str

logger = logging.getLogger("health_predictor")
logger.setLevel(logging.INFO)

# Router
router = APIRouter(prefix="/api/v1/health", tags=["Project Health"])

# SSE stream connections list
sse_connections: List[asyncio.Queue] = []

async def broadcast_health_alert(event_type: str, data: dict):
    logger.info(f"Broadcasting health event: {event_type}")
    import json
    # Format according to Server-Sent Events spec
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    for queue in list(sse_connections):
        try:
            await queue.put(payload)
        except Exception as e:
            logger.error(f"Failed to push SSE message to queue: {str(e)}")
            if queue in sse_connections:
                sse_connections.remove(queue)

async def run_project_prediction_pipeline(project_id: str, force_recalculate: bool = True) -> Optional[ProjectHealth]:
    """
    Executes the Complete Ingestion, Feature Engineering, Prediction, and Explainable AI Subsystem.
    Delegates to PredictionEngine and caches outputs in local MongoDB documents (`project_health` & `sprint_predictions`).
    """
    from app.services.prediction_engine import PredictionEngine
    return await PredictionEngine.run_project_prediction_pipeline(project_id, force_recalculate=force_recalculate)

# Background Cron Loop
health_worker_task = None
worker_running = False

async def health_predictor_worker_loop():
    global worker_running
    worker_running = True
    logger.info("Background Health Predictor Worker started.")
    await asyncio.sleep(5) # Initial warmup delay
    while worker_running:
        try:
            logger.info("Background Cron trigger: evaluating project health metrics...")
            projects = await Project.find_all().to_list()
            for p in projects:
                await run_project_prediction_pipeline(str(p.id), force_recalculate=False)
        except Exception as e:
            logger.error(f"Error in background health worker loop: {str(e)}")
            
        # Trigger every 2 hours (7200 seconds)
        for _ in range(720):
            if not worker_running:
                break
            await asyncio.sleep(10)

async def start_health_predictor_worker():
    global health_worker_task
    health_worker_task = asyncio.create_task(health_predictor_worker_loop())

async def stop_health_predictor_worker():
    global worker_running
    worker_running = False
    if health_worker_task:
        health_worker_task.cancel()
        try:
            await health_worker_task
        except asyncio.CancelledError:
            pass

# Endpoints
@router.get("/events")
async def health_events_stream():
    """
    SSE endpoint to stream health alerts in real-time.
    """
    async def event_generator():
        queue = asyncio.Queue()
        sse_connections.append(queue)
        try:
            while True:
                data = await queue.get()
                yield data
        except asyncio.CancelledError:
            pass
        finally:
            if queue in sse_connections:
                sse_connections.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/project/{project_id}")
async def get_latest_project_health(project_id: str, current_user: User = Depends(get_current_user)):
    health = await ProjectHealth.find(ProjectHealth.project_id == project_id).sort("-created_at").first_or_none()
    if not health:
        raise HTTPException(status_code=404, detail="No health metrics found for this project.")
    return health

@router.get("/sprint/{sprint_id}")
async def get_latest_sprint_prediction(sprint_id: str, current_user: User = Depends(get_current_user)):
    prediction = await SprintPrediction.find(SprintPrediction.sprint_id == sprint_id).sort("-created_at").first_or_none()
    if not prediction:
        raise HTTPException(status_code=404, detail="No prediction found for this sprint.")
    return prediction

@router.post("/predict/{project_id}")
async def trigger_prediction(project_id: str, current_user: User = Depends(get_current_user)):
    health = await run_project_prediction_pipeline(project_id)
    if not health:
        raise HTTPException(status_code=400, detail="Failed to calculate health prediction.")
    return health
