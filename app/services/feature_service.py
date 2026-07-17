import logging
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dateutil import parser as dateutil_parser

from app.models.project import Project, Sprint
from app.models.task import Task, TaskStatus
from app.models.comment import Comment
from app.models.activity_log import ActivityLog

logger = logging.getLogger("feature_service")
logger.setLevel(logging.INFO)

class FeatureService:
    # Baseline averages used for missing data
    VELOCITY_DRIFT_BASELINE = 5.0
    SCOPE_CREEP_POINTS_BASELINE = 2.0
    SENTINEL_VOLATILITY_BASELINE = 0.05
    BURNDOWN_HISTORY_BASELINE = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    @staticmethod
    async def build_sprint_vector(sprint_id: str) -> dict:
        """
        Aggregates raw historical database logs from MongoDB and builds tabular feature arrays for model inference.
        Missing values are replaced with historical baseline averages.
        Empty sprints default to zero-value tracking markers with logging.
        """
        logger.info(f"Building feature vector for sprint_id={sprint_id}")

        # Find project owning this sprint
        project = await Project.find_one({"sprints.id": sprint_id})
        if not project:
            logger.warning(f"Project not found for sprint_id={sprint_id}. Returning baselines.")
            return {
                "project_id": "",
                "sprint_id": sprint_id,
                "velocity_drift": FeatureService.VELOCITY_DRIFT_BASELINE,
                "scope_creep_points": FeatureService.SCOPE_CREEP_POINTS_BASELINE,
                "sentiment_volatility": FeatureService.SENTINEL_VOLATILITY_BASELINE,
                "burndown_history": FeatureService.BURNDOWN_HISTORY_BASELINE,
                "total_points": 0.0
            }

        # Locate the specific sprint
        sprint = None
        for s in project.sprints:
            if s.id == sprint_id:
                sprint = s
                break

        if not sprint:
            logger.warning(f"Sprint {sprint_id} not found in project {project.id}. Returning baselines.")
            return {
                "project_id": str(project.id),
                "sprint_id": sprint_id,
                "velocity_drift": FeatureService.VELOCITY_DRIFT_BASELINE,
                "scope_creep_points": FeatureService.SCOPE_CREEP_POINTS_BASELINE,
                "sentiment_volatility": FeatureService.SENTINEL_VOLATILITY_BASELINE,
                "burndown_history": FeatureService.BURNDOWN_HISTORY_BASELINE,
                "total_points": 0.0
            }

        # Fetch all tasks for project
        tasks = await Task.find(Task.project_id == str(project.id)).to_list()
        sprint_tasks = [t for t in tasks if t.sprint_id == sprint_id]

        # Case 1: Empty historical sprint data logs
        if not sprint_tasks:
            logger.warning(f"Empty historical sprint data logs for sprint_id={sprint_id}. Defaulting to zero-value tracking markers.")
            return {
                "project_id": str(project.id),
                "sprint_id": sprint_id,
                "velocity_drift": 0.0,
                "scope_creep_points": 0.0,
                "sentiment_volatility": 0.0,
                "burndown_history": [0.0] * 10,
                "total_points": 0.0
            }

        # --- 1. velocity_drift ---
        completed_sprints = [s for s in project.sprints if s.status == "COMPLETED"]
        completed_sprints = completed_sprints[-3:]
        drift_values = []
        for cs in completed_sprints:
            cs_tasks = [t for t in tasks if t.sprint_id == cs.id]
            planned = sum(t.estimated_hours for t in cs_tasks)
            completed = sum(t.estimated_hours for t in cs_tasks if t.status == TaskStatus.DONE)
            drift_values.append(planned - completed)

        if drift_values:
            velocity_drift = sum(drift_values) / len(drift_values)
        else:
            logger.warning(f"Missing historical velocity drift for sprint_id={sprint_id}. Replacing with baseline {FeatureService.VELOCITY_DRIFT_BASELINE}")
            velocity_drift = FeatureService.VELOCITY_DRIFT_BASELINE

        # --- 2. scope_creep_points ---
        scope_creep_points = 0.0
        if sprint.start_date:
            try:
                sprint_start = dateutil_parser.parse(sprint.start_date)
            except Exception:
                try:
                    sprint_start = datetime.fromisoformat(sprint.start_date.replace("Z", "+00:00"))
                except Exception:
                    sprint_start = datetime.now(timezone.utc)
            
            for t in sprint_tasks:
                if t.created_at > sprint_start:
                    scope_creep_points += t.estimated_hours
        else:
            logger.warning(f"Missing sprint start date for sprint_id={sprint_id}. Replacing scope creep with baseline {FeatureService.SCOPE_CREEP_POINTS_BASELINE}")
            scope_creep_points = FeatureService.SCOPE_CREEP_POINTS_BASELINE

        # --- 3. sentiment_volatility ---
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
            logger.warning(f"No comments / empty sentiment volatility for sprint_id={sprint_id}. Replacing with baseline {FeatureService.SENTINEL_VOLATILITY_BASELINE}")
            sentiment_volatility = FeatureService.SENTINEL_VOLATILITY_BASELINE

        # --- 4. burndown_history ---
        total_points = sum(t.estimated_hours for t in sprint_tasks)
        completed_points = sum(t.estimated_hours for t in sprint_tasks if t.status == TaskStatus.DONE)
        remaining_points = total_points - completed_points

        if total_points > 0:
            burndown_history = []
            for i in range(10):
                val = total_points - (total_points - remaining_points) * (i / 9.0)
                burndown_history.append(float(round(val, 2)))
        else:
            logger.warning(f"Total points is 0 for sprint_id={sprint_id}. Replacing burndown history with baseline.")
            burndown_history = FeatureService.BURNDOWN_HISTORY_BASELINE

        return {
            "project_id": str(project.id),
            "sprint_id": sprint_id,
            "velocity_drift": velocity_drift,
            "scope_creep_points": scope_creep_points,
            "sentiment_volatility": sentiment_volatility,
            "burndown_history": burndown_history,
            "total_points": total_points
        }

    @staticmethod
    async def build_task_vector(task_id: str) -> dict:
        """
        Builds feature vector for a task: [age_hours, reopens, density]
        """
        task = await Task.get(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found. Returning zero-value feature vector.")
            return {"age_hours": 0.0, "reopens": 0.0, "density": 0.0}

        age_hours = (datetime.now(timezone.utc) - task.created_at).total_seconds() / 3600.0
        
        reopens = 0
        logs = await ActivityLog.find(
            ActivityLog.task_id == task_id,
            ActivityLog.action == "status_change"
        ).to_list()
        for log in logs:
            if "from done" in log.detail.lower():
                reopens += 1

        dependency_count = len(getattr(task, "dependency_ids", []))
        blocked_h = getattr(task, "blocked_hours", 0.0)
        density = dependency_count + blocked_h

        return {
            "age_hours": age_hours,
            "reopens": float(reopens),
            "density": density
        }
