import os
import math
import logging
import numpy as np
import onnxruntime as ort
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from app.models.project import Project, Sprint
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.user import User
from app.models.comment import Comment
from app.models.sprint_prediction import SprintPrediction
from app.models.project_health import ProjectHealth
from app.models.notification import Notification
from app.services.feature_service import FeatureService
from app.ai.core.llm_manager import llm_manager

logger = logging.getLogger("prediction_engine")
logger.setLevel(logging.INFO)

# Root directory and ONNX registry setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_DIR = os.path.join(BASE_DIR, "ai", "models", "registry")

SPRINT_MODEL_PATH = os.path.join(REGISTRY_DIR, "sprint_failure_v1.0.onnx")
TASK_MODEL_PATH = os.path.join(REGISTRY_DIR, "task_delay_v1.0.onnx")
CHRONOS_MODEL_PATH = os.path.join(REGISTRY_DIR, "chronos_v1.0.onnx")

class PredictionEngine:
    _sprint_sess = None
    _task_sess = None
    _chronos_sess = None

    @classmethod
    def _get_sessions(cls):
        """
        Lazy-initialize ONNX runtime sessions with CPU provider and session options.
        """
        if cls._sprint_sess is None:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"]

            try:
                if os.path.exists(SPRINT_MODEL_PATH):
                    cls._sprint_sess = ort.InferenceSession(SPRINT_MODEL_PATH, sess_opts, providers=providers)
                    logger.info(f"Loaded Sprint Failure ONNX model from {SPRINT_MODEL_PATH}")
                else:
                    logger.error(f"Sprint Failure ONNX model not found at {SPRINT_MODEL_PATH}")

                if os.path.exists(TASK_MODEL_PATH):
                    cls._task_sess = ort.InferenceSession(TASK_MODEL_PATH, sess_opts, providers=providers)
                    logger.info(f"Loaded Task Delay ONNX model from {TASK_MODEL_PATH}")
                else:
                    logger.error(f"Task Delay ONNX model not found at {TASK_MODEL_PATH}")

                if os.path.exists(CHRONOS_MODEL_PATH):
                    cls._chronos_sess = ort.InferenceSession(CHRONOS_MODEL_PATH, sess_opts, providers=providers)
                    logger.info(f"Loaded Chronos ONNX model from {CHRONOS_MODEL_PATH}")
                else:
                    logger.error(f"Chronos ONNX model not found at {CHRONOS_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to initialize ONNX runtime sessions: {str(e)}")

        return cls._sprint_sess, cls._task_sess, cls._chronos_sess

    @classmethod
    def run_sprint_failure_prediction(cls, velocity_drift: float, scope_creep: float, sentiment_vol: float) -> float:
        """
        Runs LightGBM Classifier model (via ONNX) to predict sprint failure probability.
        Confirmed to output scores bounded safely between 0.0 and 1.0.
        """
        sprint_sess, _, _ = cls._get_sessions()
        if sprint_sess is not None:
            try:
                # Shape [1, 3]
                input_data = np.array([[velocity_drift, scope_creep, sentiment_vol]], dtype=np.float32)
                inputs = {sprint_sess.get_inputs()[0].name: input_data}
                outputs = sprint_sess.run(None, inputs)
                score = float(outputs[0][0][0])
                return max(0.0, min(1.0, score))
            except Exception as e:
                logger.error(f"Error running Sprint Failure ONNX inference: {str(e)}")
        
        # Fallback to analytical calculation
        z_fail = 0.25 * velocity_drift + 0.15 * scope_creep + 3.0 * sentiment_vol - 1.5
        try:
            return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-z_fail))))
        except OverflowError:
            return 1.0 if z_fail > 0 else 0.0

    @classmethod
    def run_task_delay_prediction(cls, age_hours: float, reopens: float, density: float) -> float:
        """
        Runs Task Delay Predictor (via ONNX) to calculate task delay risk probability.
        Confirmed to output scores bounded safely between 0.0 and 1.0.
        """
        _, task_sess, _ = cls._get_sessions()
        if task_sess is not None:
            try:
                # Shape [1, 3]
                input_data = np.array([[age_hours, reopens, density]], dtype=np.float32)
                inputs = {task_sess.get_inputs()[0].name: input_data}
                outputs = task_sess.run(None, inputs)
                score = float(outputs[0][0][0])
                return max(0.0, min(1.0, score))
            except Exception as e:
                logger.error(f"Error running Task Delay ONNX inference: {str(e)}")

        # Fallback to analytical calculation
        z = (0.005 * age_hours) + (1.2 * reopens) + (0.8 * density) - 1.5
        try:
            return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-z))))
        except OverflowError:
            return 1.0 if z > 0 else 0.0

    @classmethod
    def run_chronos_burndown_forecast(cls, burndown_history: List[float]) -> List[float]:
        """
        Runs Chronos-Tiny (via ONNX) using historical burndown track data.
        Returns forecasted burndown trajectory (10 steps).
        """
        _, _, chronos_sess = cls._get_sessions()
        if chronos_sess is not None:
            try:
                # Chronos model expects [1, 10]
                hist_arr = list(burndown_history)
                if len(hist_arr) < 10:
                    hist_arr = hist_arr + [hist_arr[-1]] * (10 - len(hist_arr))
                elif len(hist_arr) > 10:
                    hist_arr = hist_arr[:10]

                input_data = np.array([hist_arr], dtype=np.float32)
                inputs = {chronos_sess.get_inputs()[0].name: input_data}
                outputs = chronos_sess.run(None, inputs)
                trajectory = [float(x) for x in outputs[0][0]]
                return [round(max(0.0, val), 1) for val in trajectory]
            except Exception as e:
                logger.error(f"Error running Chronos-Tiny ONNX inference: {str(e)}")

        # Fallback burndown calculation
        if burndown_history:
            start_val = burndown_history[-1]
            return [round(max(0.0, start_val * (1.0 - (step / 10.0))), 1) for step in range(10)]
        return [0.0] * 10

    @classmethod
    async def run_project_prediction_pipeline(cls, project_id: str, force_recalculate: bool = False) -> Optional[ProjectHealth]:
        """
        Runs full predictive pipeline for a project. 
        Implements 2-hour caching window unless force_recalculate is True.
        """
        logger.info(f"Prediction Pipeline trigger for project_id={project_id} (force={force_recalculate})")

        # Caching logic
        if not force_recalculate:
            cached_health = await ProjectHealth.find(ProjectHealth.project_id == project_id).sort("-created_at").first_or_none()
            if cached_health:
                time_diff = datetime.now(timezone.utc) - cached_health.created_at
                if time_diff.total_seconds() < 7200: # 2 hours
                    logger.info(f"Cache HIT for project_id={project_id}. Returning cached ProjectHealth.")
                    return cached_health

        # Cache miss / Forced run
        project = await Project.get(project_id)
        if not project:
            logger.error(f"Project {project_id} not found in database.")
            return None

        # Resolve active sprint
        active_sprint = None
        for s in project.sprints:
            if s.status == "ACTIVE":
                active_sprint = s
                break

        tasks = await Task.find(Task.project_id == project_id).to_list()

        # 1. Task-Level Predictions
        task_delay_risks: List[Dict[str, Any]] = []
        for task in tasks:
            if task.status != TaskStatus.DONE:
                feats = await FeatureService.build_task_vector(str(task.id))
                delay_risk = cls.run_task_delay_prediction(
                    feats["age_hours"],
                    feats["reopens"],
                    feats["density"]
                )

                reasons = []
                if feats["age_hours"] > 72:
                    reasons.append(f"Task has been open for {feats['age_hours']:.1f} hours")
                if feats["reopens"] > 0:
                    reasons.append(f"Task has been reopened {int(feats['reopens'])} times")
                if feats["density"] > 0:
                    reasons.append(f"Has blocker/dependency density of {feats['density']:.1f}")

                task_delay_risks.append({
                    "task_id": str(task.id),
                    "title": task.title,
                    "delay_risk_score": delay_risk,
                    "reasons": reasons
                })

        # 2. Burnout Risk calculations (logic-driven)
        assignee_burnout_risks: List[Dict[str, Any]] = []
        all_devs = set(project.developer_ids)
        if project.lead_developer_id:
            all_devs.add(project.lead_developer_id)

        for dev_id in all_devs:
            active_dev_tasks = [t for t in tasks if t.assigned_to_id == dev_id and t.status != TaskStatus.DONE]
            active_hours = sum(t.estimated_hours for t in active_dev_tasks)
            
            completed_dev_tasks = [t for t in tasks if t.assigned_to_id == dev_id and t.status == TaskStatus.DONE]
            completed_points = sum(t.estimated_hours for t in completed_dev_tasks)
            dev_velocity = completed_points / 2.0 if completed_points > 0 else 15.0
            
            workload_balance = active_hours / dev_velocity if dev_velocity > 0 else active_hours

            dev_sprint_tasks = [t for t in tasks if t.assigned_to_id == dev_id and t.sprint_id == (active_sprint.id if active_sprint else None)]
            unique_tags = set()
            for t in dev_sprint_tasks:
                unique_tags.update(t.tags)
                
            other_tasks = await Task.find(Task.assigned_to_id == dev_id, Task.status != TaskStatus.DONE).to_list()
            unique_projects = set(t.project_id for t in other_tasks)
            context_switching_count = len(unique_projects) + len(unique_tags)

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

        # 3. Sprint failure rate model & Chronos forecasting
        sprint_risk = 0.0
        sprint_prediction = None
        sprint_status = "HEALTHY"

        if active_sprint:
            sprint_feats = await FeatureService.build_sprint_vector(active_sprint.id)
            failure_rate = cls.run_sprint_failure_prediction(
                sprint_feats["velocity_drift"],
                sprint_feats["scope_creep_points"],
                sprint_feats["sentiment_volatility"]
            )
            sprint_risk = failure_rate

            if failure_rate > 0.70:
                sprint_status = "CRITICAL"
            elif failure_rate > 0.40:
                sprint_status = "WARNING"

            trajectory = cls.run_chronos_burndown_forecast(sprint_feats["burndown_history"])
            task_completion_max = sprint_feats["total_points"] * (1.0 - failure_rate)

            # Explainable AI - DeepSeek R1 Trigger logic: runs only on risk alerts to preserve performance
            explanation = ""
            if sprint_status in ["WARNING", "CRITICAL"] or failure_rate > 0.70:
                shap_weights = {
                    "historical_velocity_drift": 0.25 * sprint_feats["velocity_drift"],
                    "unplanned_scope_creep_points": 0.15 * sprint_feats["scope_creep_points"],
                    "comment_sentiment_volatility": 3.0 * sprint_feats["sentiment_volatility"]
                }
                sorted_shap = sorted(shap_weights.items(), key=lambda x: abs(x[1]), reverse=True)
                shap_weights_text = "\n".join([f"- {k}: {v:.2f}" for k, v in sorted_shap])
                task_details_text = "\n".join([f"- {t['title']} (Delay Risk: {t['delay_risk_score']*100:.1f}%)" for t in task_delay_risks[:3]])

                prompt = f"""
[Intelligent Sprint Risk Predictor System]
Analyze the Sprint metrics and SHAP values, explaining why this sprint is at risk and suggesting mitigation steps.

Sprint Metrics:
- Sprint Failure Rate: {failure_rate * 100:.1f}%
- Scope Creep Points: {sprint_feats['scope_creep_points']:.1f}
- Comment Sentiment Volatility: {sprint_feats['sentiment_volatility']:.3f}
- Velocity Drift: {sprint_feats['velocity_drift']:.2f}

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
                logger.info(f"Triggering low-frequency DeepSeek-R1-7B Distill explainability prompt for sprint {active_sprint.id} status={sprint_status}")
                explanation = llm_manager.generate(prompt)
            else:
                explanation = "Status is HEALTHY. Sprint metrics are within normal thresholds; explainability triggers bypassed."

            sprint_prediction = SprintPrediction(
                project_id=project_id,
                sprint_id=active_sprint.id,
                historical_velocity_drift=sprint_feats["velocity_drift"],
                unplanned_scope_creep_points=sprint_feats["scope_creep_points"],
                comment_sentiment_volatility=sprint_feats["sentiment_volatility"],
                failure_rate=failure_rate,
                burndown_trajectory=trajectory,
                task_completion_max=task_completion_max,
                explanation=explanation,
                risk_score=sprint_risk,
                status=sprint_status
            )
            await sprint_prediction.insert()

        # 4. Project Health Synthesis
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

        # Explainable AI - DeepSeek R1 Trigger logic for project health
        project_explanation = ""
        if project_status in ["WARNING", "CRITICAL"]:
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
            logger.info(f"Triggering low-frequency DeepSeek-R1-7B Distill explainability prompt for project {project_id} status={project_status}")
            project_explanation = llm_manager.generate(project_prompt)
        else:
            project_explanation = "Project health status is HEALTHY. Metrics are within normal operating bounds."

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

        # 5. Push Alerts to Notification Bell & Real-time SSE Dispatcher
        if project_status in ["WARNING", "CRITICAL"] or sprint_risk > 0.70:
            msg = f"Project '{project.name}' health status is {project_status} (Score: {health_score:.1f}/100)."
            notification = Notification(
                project_id=project_id,
                message=msg,
                created_by_name="Health Engine"
            )
            await notification.insert()

            # Import the router broadcast function to avoid circular dependencies
            from app.routers.health_router import broadcast_health_alert
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
