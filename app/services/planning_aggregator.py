import logging
import asyncio
from typing import Optional, Dict, Any

from app.models.task import Task, TaskStatus
from app.models.user import User
from app.models.project import Project
from app.models.sprint_prediction import SprintPrediction
from app.services.feature_service import FeatureService
from app.services.prediction_engine import PredictionEngine

logger = logging.getLogger("planning_aggregator")

class PlanningAggregator:
    @classmethod
    async def gather_data(cls, project_id: str, target_sprint_id: str, capacity_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Pulls a flattened snapshot of the project state to feed the solver,
        incorporating Phase 4 Intelligence predictions.
        """
        project = await Project.get(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # 1. Fetch eligible tasks
        tasks = await Task.find(
            Task.project_id == project_id,
            Task.status != TaskStatus.DONE
        ).to_list()
        
        eligible_tasks = [t for t in tasks if t.sprint_id is None or t.sprint_id == target_sprint_id]
        
        # 2. Fetch developers
        developer_ids = project.developer_ids
        developers = []
        for dev_id in developer_ids:
            dev = await User.get(dev_id)
            if dev:
                developers.append(dev)

        if not developers:
            raise ValueError("No developers found in project to assign tasks.")

        # 3. Phase 4 - Developer Workload Scores & Capacity Constraints
        project_health = await PredictionEngine.run_project_prediction_pipeline(project_id, force_recalculate=False)
        
        dev_capacities = {}
        base_capacity = capacity_override if capacity_override is not None else 40.0
        
        if project_health and project_health.assignee_burnout_risks:
            for risk_data in project_health.assignee_burnout_risks:
                dev_id = risk_data["developer_id"]
                workload_balance = risk_data.get("workload_balance", 0.0)
                # Scale capacity inversely to workload balance to prevent burnout.
                # If workload_balance is 1.0 (at capacity), capacity gets reduced significantly.
                # We floor it at 10.0 hours.
                scaled_cap = max(10.0, base_capacity - (workload_balance * (base_capacity * 0.5)))
                dev_capacities[dev_id] = scaled_cap
        
        # Set defaults for devs not found in burnout risks
        for dev in developers:
            if str(dev.id) not in dev_capacities:
                dev_capacities[str(dev.id)] = base_capacity
                
        # 4. Phase 4 - Task Delay Predictions (Penalty Weights)
        task_risks = {}
        
        async def fetch_task_risk(task):
            feats = await FeatureService.build_task_vector(str(task.id))
            risk_score = PredictionEngine.run_task_delay_prediction(
                feats["age_hours"],
                feats["reopens"],
                feats["density"]
            )
            return str(task.id), risk_score
            
        risk_results = await asyncio.gather(*(fetch_task_risk(t) for t in eligible_tasks))
        for task_id, risk_score in risk_results:
            task_risks[task_id] = risk_score

        # 5. Fetch global sprint risk penalty
        risk_penalty_factor = 1.0
        sprint_pred = await SprintPrediction.find_one(SprintPrediction.sprint_id == target_sprint_id)
        if sprint_pred and sprint_pred.risk_score:
            risk_penalty_factor += (sprint_pred.risk_score / 100.0)

        return {
            "eligible_tasks": eligible_tasks,
            "developers": developers,
            "dev_capacities": dev_capacities,
            "task_risks": task_risks,
            "risk_penalty_factor": risk_penalty_factor
        }

