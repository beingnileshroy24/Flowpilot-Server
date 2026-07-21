import logging
from typing import Dict, Any

from app.ai.core.llm_manager import llm_manager

logger = logging.getLogger("plan_explainer")

class PlanExplainer:
    @classmethod
    def explain(cls, solver_output: Dict[str, Any]) -> str:
        """
        Takes the solver output and feeds it into the MLX pipeline using a 
        strict schema prompt to generate explanatory text.
        """
        assigned_tasks = solver_output.get("assigned_tasks", [])
        dropped_tasks = solver_output.get("dropped_tasks", [])
        dev_capacity_hours = solver_output.get("dev_capacity_hours", 40.0)
        risk_penalty_factor = solver_output.get("risk_penalty_factor", 1.0)
        
        prompt = f"""[Sprint Planner - Optimization Explanation]
You are the MLX Explainability Layer for our Sprint Planner.
The Google OR-Tools CP-SAT Solver has generated a sprint plan.

Assigned Tasks:
{assigned_tasks}

Dropped/Deferred Tasks:
{dropped_tasks}

Capacity used per developer: {dev_capacity_hours} hours.
Risk Penalty Applied: {risk_penalty_factor:.2f}x.

Provide a short natural language summary explaining the trade-offs made, why certain tasks were dropped (e.g. capacity limits, dependencies, or lower priority), and how this plan maximizes business value while balancing workload.
"""
        try:
            explanation = llm_manager.generate(prompt, max_tokens=600)
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate plan explanation: {e}")
            return "Failed to generate explanation due to an internal error."
