import logging
from typing import Optional, Dict, Any

from app.services.planning_aggregator import PlanningAggregator
from app.services.sprint_solver import SprintSolver
from app.services.plan_explainer import PlanExplainer

logger = logging.getLogger("planner_service")

class PlannerService:
    @classmethod
    async def generate_sprint_plan(cls, project_id: str, target_sprint_id: str, capacity_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Multi-stage pipeline:
        1. Data Aggregation (PlanningAggregator)
        2. CP-SAT Solver (SprintSolver)
        3. MLX Explanation Layer (PlanExplainer)
        """
        # 1. Data Aggregation
        aggregated_data = await PlanningAggregator.gather_data(project_id, target_sprint_id, capacity_override)
        
        # 2. CP-SAT Solver
        solver_result = SprintSolver.solve(aggregated_data)
        
        # 3. Explanation Layer
        explanation = PlanExplainer.explain(solver_result)
        
        return {
            "assigned_tasks": solver_result.get("assigned_tasks", []),
            "dropped_tasks": solver_result.get("dropped_tasks", []),
            "explanation": explanation,
            "solver_status": solver_result.get("solver_status")
        }
