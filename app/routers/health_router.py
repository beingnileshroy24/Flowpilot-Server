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

async def run_project_prediction_pipeline(project_id: str) -> Optional[ProjectHealth]:
    """
    Executes the Complete Ingestion, Feature Engineering, Prediction, and Explainable AI Subsystem.
    Caches outputs in local MongoDB documents (`project_health` & `sprint_predictions`).
    """
    logger.info(f"Running risk prediction pipeline for project_id={project_id}")
    project = await Project.get(project_id)
    if not project:
        logger.error(f"Project {project_id} not found in database.")
        return None

    tasks = await Task.find(Task.project_id == project_id).to_list()
    
    # Resolve active sprint
    active_sprint = None
    if project.sprints:
        for s in project.sprints:
            if s.status == "ACTIVE":
                active_sprint = s
                break

    # --- 1. Task-Level Ingestion Vectors & LightGBM Predictor ---
    task_delay_risks: List[TaskDelayRisk] = []
    for task in tasks:
        if task.status != TaskStatus.DONE:
            # task_age_hours
            age_hours = (datetime.now(timezone.utc) - task.created_at).total_seconds() / 3600.0
            
            # reopen_frequency
            reopens = 0
            logs = await ActivityLog.find(
                ActivityLog.task_id == str(task.id),
                ActivityLog.action == "status_change"
            ).to_list()
            for log in logs:
                if "from done" in log.detail.lower():
                    reopens += 1
            
            # blocker_dependency_density
            dependency_count = len(getattr(task, "dependency_ids", []))
            blocked_h = getattr(task, "blocked_hours", 0.0)
            density = dependency_count + blocked_h
            
            # LightGBM Classifier simulation formula
            z = (0.005 * age_hours) + (1.2 * reopens) + (0.8 * density) - 1.5
            try:
                delay_risk = 1.0 / (1.0 + math.exp(-z))
            except OverflowError:
                delay_risk = 1.0 if z > 0 else 0.0

            reasons = []
            if age_hours > 72:
                reasons.append(f"Task has been open for {age_hours:.1f} hours")
            if reopens > 0:
                reasons.append(f"Task has been reopened {reopens} times")
            if density > 0:
                reasons.append(f"Has blocker/dependency density of {density:.1f}")

            task_delay_risks.append({
                "task_id": str(task.id),
                "title": task.title,
                "delay_risk_score": delay_risk,
                "reasons": reasons
            })

    # --- 2. Human Resource Constraints Matrix ---
    assignee_burnout_risks: List[BurnoutRisk] = []
    all_devs = set(project.developer_ids)
    if project.lead_developer_id:
        all_devs.add(project.lead_developer_id)

    for dev_id in all_devs:
        # assignee_workload_balance
        active_dev_tasks = [t for t in tasks if t.assigned_to_id == dev_id and t.status != TaskStatus.DONE]
        active_hours = sum(t.estimated_hours for t in active_dev_tasks)
        
        # historical completion velocity baseline (last completed tasks)
        completed_dev_tasks = [t for t in tasks if t.assigned_to_id == dev_id and t.status == TaskStatus.DONE]
        completed_points = sum(t.estimated_hours for t in completed_dev_tasks)
        dev_velocity = completed_points / 2.0 if completed_points > 0 else 15.0
        
        workload_balance = active_hours / dev_velocity if dev_velocity > 0 else active_hours

        # cross_domain_context_switching
        dev_sprint_tasks = [t for t in tasks if t.assigned_to_id == dev_id and t.sprint_id == (active_sprint.id if active_sprint else None)]
        unique_tags = set()
        for t in dev_sprint_tasks:
            unique_tags.update(t.tags)
            
        other_tasks = await Task.find(Task.assigned_to_id == dev_id, Task.status != TaskStatus.DONE).to_list()
        unique_projects = set(t.project_id for t in other_tasks)
        context_switching_count = len(unique_projects) + len(unique_tags)

        # burnout probability model
        z_burnout = 0.6 * workload_balance + 0.4 * context_switching_count - 2.2
        try:
            burnout_prob = 1.0 / (1.0 + math.exp(-z_burnout))
        except OverflowError:
            burnout_prob = 1.0 if z_burnout > 0 else 0.0

        risk_level = "HEALTHY"
        if burnout_prob > 0.70:
            risk_level = "CRITICAL"
        elif burnout_prob > 0.40:
            risk_level = "WARNING"

        dev_user = await User.get(dev_id)
        dev_name = dev_user.name if dev_user else "Unknown"

        assignee_burnout_risks.append({
            "developer_id": dev_id,
            "name": dev_name,
            "workload_balance": workload_balance,
            "context_switching_count": context_switching_count,
            "burnout_risk_level": risk_level
        })

    # --- 3. Sprint-Level Aggregate Features & Time-Series (Chronos) Forecast ---
    sprint_risk = 0.0
    sprint_prediction = None
    if active_sprint:
        # historical_velocity_drift
        completed_sprints = [s for s in project.sprints if s.status == "COMPLETED"]
        completed_sprints = completed_sprints[-3:]
        drift_values = []
        for s in completed_sprints:
            s_tasks = [t for t in tasks if t.sprint_id == s.id]
            planned = sum(t.estimated_hours for t in s_tasks)
            completed = sum(t.estimated_hours for t in s_tasks if t.status == TaskStatus.DONE)
            drift_values.append(planned - completed)
        velocity_drift = sum(drift_values) / len(drift_values) if drift_values else 0.0

        # unplanned_scope_creep_points
        scope_creep_points = 0.0
        if active_sprint.start_date:
            try:
                from dateutil import parser
                sprint_start = parser.parse(active_sprint.start_date)
            except Exception:
                try:
                    sprint_start = datetime.fromisoformat(active_sprint.start_date.replace("Z", "+00:00"))
                except Exception:
                    sprint_start = datetime.now(timezone.utc)
            
            sprint_tasks = [t for t in tasks if t.sprint_id == active_sprint.id]
            for t in sprint_tasks:
                if t.created_at > sprint_start:
                    scope_creep_points += t.estimated_hours

        # comment_sentiment_volatility
        sprint_tasks = [t for t in tasks if t.sprint_id == active_sprint.id]
        sprint_task_ids = [str(t.id) for t in sprint_tasks]
        comments = await Comment.find({"task_id": {"$in": sprint_task_ids}}).to_list()
        excl_rates = []
        for c in comments:
            words = len(c.content.split())
            excl = c.content.count('!')
            if words > 0:
                excl_rates.append(excl / words)

        if len(excl_rates) > 1:
            mean = sum(excl_rates) / len(excl_rates)
            variance = sum((x - mean) ** 2 for x in excl_rates) / (len(excl_rates) - 1)
            sentiment_volatility = variance ** 0.5
        elif len(excl_rates) == 1:
            sentiment_volatility = excl_rates[0]
        else:
            sentiment_volatility = 0.0

        # LightGBM Classifier model (Failure rate score)
        z_fail = 0.25 * velocity_drift + 0.15 * scope_creep_points + 3.0 * sentiment_volatility - 1.5
        try:
            failure_rate = 1.0 / (1.0 + math.exp(-z_fail))
        except OverflowError:
            failure_rate = 1.0 if z_fail > 0 else 0.0

        # Chronos Time-Series Trajectory
        total_sprint_points = sum(t.estimated_hours for t in sprint_tasks)
        completed_sprint_points = sum(t.estimated_hours for t in sprint_tasks if t.status == TaskStatus.DONE)
        remaining_points = total_sprint_points - completed_sprint_points
        
        trajectory = []
        for step in range(10):
            proj = remaining_points * (1.0 - (step / 10.0) * (1.0 - failure_rate))
            trajectory.append(round(max(0.0, proj), 1))

        task_completion_max = total_sprint_points * (1.0 - failure_rate)
        sprint_risk = failure_rate

        sprint_status = "HEALTHY"
        if failure_rate > 0.70:
            sprint_status = "CRITICAL"
        elif failure_rate > 0.40:
            sprint_status = "WARNING"

        # Explainable AI prompt building & LLM generation
        shap_weights = {
            "historical_velocity_drift": 0.25 * velocity_drift,
            "unplanned_scope_creep_points": 0.15 * scope_creep_points,
            "comment_sentiment_volatility": 3.0 * sentiment_volatility
        }
        sorted_shap = sorted(shap_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        shap_weights_text = "\n".join([f"- {k}: {v:.2f}" for k, v in sorted_shap])
        task_details_text = "\n".join([f"- {t['title']} (Delay Risk: {t['delay_risk_score']*100:.1f}%)" for t in task_delay_risks[:3]])

        prompt = f"""
[Intelligent Sprint Risk Predictor System]
Analyze the Sprint metrics and SHAP values, explaining why this sprint is at risk and suggesting mitigation steps.

Sprint Metrics:
- Sprint Failure Rate: {failure_rate * 100:.1f}%
- Scope Creep Points: {scope_creep_points:.1f}
- Comment Sentiment Volatility: {sentiment_volatility:.3f}
- Velocity Drift: {velocity_drift:.2f}

Top Risk Factors (SHAP weights):
{shap_weights_text}

Active Task Details:
{task_details_text}

Format the output in clean Markdown:
### Thought Process
(Detail your reasoning chain about the risks)

### Risk Analysis
(Explain the risk factors clearly)

### Actionable Recommendations
- Recommendation 1
- Recommendation 2
"""
        explanation = llm_manager.generate(prompt)

        sprint_prediction = SprintPrediction(
            project_id=project_id,
            sprint_id=active_sprint.id,
            historical_velocity_drift=velocity_drift,
            unplanned_scope_creep_points=scope_creep_points,
            comment_sentiment_volatility=sentiment_volatility,
            failure_rate=failure_rate,
            burndown_trajectory=trajectory,
            task_completion_max=task_completion_max,
            explanation=explanation,
            risk_score=sprint_risk,
            status=sprint_status
        )
        await sprint_prediction.insert()

    # --- 4. Project Health Synthesis ---
    avg_task_risk = sum(t["delay_risk_score"] for t in task_delay_risks) / len(task_delay_risks) if task_delay_risks else 0.0
    burnout_mapping = {"HEALTHY": 0.0, "WARNING": 0.5, "CRITICAL": 1.0}
    avg_burnout_risk = sum(burnout_mapping[d["burnout_risk_level"]] for d in assignee_burnout_risks) / len(assignee_burnout_risks) if assignee_burnout_risks else 0.0

    overall_risk = (0.4 * avg_task_risk) + (0.3 * avg_burnout_risk) + (0.3 * sprint_risk)
    health_score = max(0.0, min(100.0, 100.0 * (1.0 - overall_risk)))

    project_status = "HEALTHY"
    if health_score < 40:
        project_status = "CRITICAL"
    elif health_score < 70:
        project_status = "WARNING"

    project_prompt = f"""
[Intelligent Project Health Engine]
Explain the project health status of project '{project.name}'.

Metrics:
- Health Score: {health_score:.1f}/100
- Status: {project_status}
- Average Task Delay Risk: {avg_task_risk*100:.1f}%
- Average Developer Burnout Risk: {avg_burnout_risk*100:.1f}%
- Active Sprint Failure Risk: {sprint_risk*100:.1f}%

Format the output in clean Markdown:
### Project Status Summary
(Provide a summary of the project health)

### Burnout & Workload Concerns
(Explain assignee burnout and context switching risks)

### Actionable Roadmap Mitigations
- Action item 1
- Action item 2
"""
    project_explanation = llm_manager.generate(project_prompt)

    project_health = ProjectHealth(
        project_id=project_id,
        health_score=health_score,
        status=project_status,
        active_sprint_id=active_sprint.id if active_sprint else None,
        task_delay_risks=task_delay_risks,
        assignee_burnout_risks=assignee_burnout_risks,
        explanation=project_explanation
    )
    await project_health.insert()

    # --- 5. Push Alerts to Notification Bell & Real-time SSE Dispatcher ---
    if project_status in ["WARNING", "CRITICAL"] or sprint_risk > 0.70:
        msg = f"Project '{project.name}' health status is {project_status} (Score: {health_score:.1f}/100)."
        notification = Notification(
            project_id=project_id,
            message=msg,
            created_by_name="Health Engine"
        )
        await notification.insert()

        await broadcast_health_alert("health_alert", {
            "project_id": project_id,
            "project_name": project.name,
            "status": project_status,
            "health_score": health_score,
            "message": msg,
            "sprint_failure_rate": sprint_risk,
            "explanation": project_explanation
        })

    return project_health

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
                await run_project_prediction_pipeline(str(p.id))
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
