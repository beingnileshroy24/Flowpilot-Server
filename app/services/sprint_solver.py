import logging
from typing import Dict, Any
from ortools.sat.python import cp_model

from app.models.task import TaskPriority

logger = logging.getLogger("sprint_solver")

class SprintSolver:
    PRIORITY_WEIGHTS = {
        TaskPriority.CRITICAL: 10,
        TaskPriority.HIGH: 5,
        TaskPriority.MEDIUM: 2,
        TaskPriority.LOW: 1
    }

    @classmethod
    def solve(cls, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates the aggregated data into OR-Tools boolean variables, applies constraints,
        and solves the constraint satisfaction problem using Phase 4 data.
        """
        eligible_tasks = aggregated_data.get("eligible_tasks", [])
        developers = aggregated_data.get("developers", [])
        dev_capacities = aggregated_data.get("dev_capacities", {})
        task_risks = aggregated_data.get("task_risks", {})
        risk_penalty_factor = aggregated_data.get("risk_penalty_factor", 1.0)

        if not eligible_tasks:
            return {
                "assigned_tasks": [],
                "dropped_tasks": [],
                "solver_status": "NO_TASKS",
                "dev_capacity_hours": "DYNAMIC",
                "risk_penalty_factor": risk_penalty_factor
            }

        model = cp_model.CpModel()
        assignments = {}
        task_assigned = {}
        
        SCALE = 10
        
        # Decision Variables
        for task in eligible_tasks:
            task_assigned[str(task.id)] = model.new_bool_var(f"assigned_{task.id}")
            for dev in developers:
                assignments[(str(task.id), str(dev.id))] = model.new_bool_var(f"assign_{task.id}_{dev.id}")

        # Unique Assignment Constraint
        for task in eligible_tasks:
            model.add(sum(assignments[(str(task.id), str(dev.id))] for dev in developers) == task_assigned[str(task.id)])

        # Capacity Constraint & Dev Load Tracking
        dev_loads = []
        max_possible_load = 0
        for dev in developers:
            dev_hours_assigned = []
            for task in eligible_tasks:
                task_weight = int((task.estimated_hours * SCALE) * risk_penalty_factor)
                if task_weight == 0:
                    task_weight = 1
                dev_hours_assigned.append(assignments[(str(task.id), str(dev.id))] * task_weight)
            
            dev_cap = dev_capacities.get(str(dev.id), 40.0)
            capacity_scaled = int(dev_cap * SCALE)
            max_possible_load += capacity_scaled
            
            dev_load = sum(dev_hours_assigned)
            model.add(dev_load <= capacity_scaled)
            
            # Create a variable to represent this developer's total assigned load
            dev_load_var = model.new_int_var(0, capacity_scaled, f"load_{dev.id}")
            model.add(dev_load_var == dev_load)
            dev_loads.append(dev_load_var)

        # Dependency Resolution (Precedence) Constraint - Driven by Phase 2 NWBE output
        for task in eligible_tasks:
            for dep_id in task.dependency_ids:
                if dep_id in task_assigned:
                    model.add_implication(task_assigned[str(task.id)], task_assigned[dep_id])

        # Skill Match Constraint
        for task in eligible_tasks:
            if task.tags:
                for dev in developers:
                    dev_skills = dev.skills if hasattr(dev, 'skills') else []
                    has_match = any(tag in dev_skills for tag in task.tags)
                    if not has_match and len(dev_skills) > 0:
                        model.add(assignments[(str(task.id), str(dev.id))] == 0)

        # Optimization Objective Function
        objective_terms = []
        
        # 1. Maximize Priority and 2. Minimize Task Delay Risk
        for task in eligible_tasks:
            priority_weight = cls.PRIORITY_WEIGHTS.get(task.priority, 1) * 100
            
            # Convert risk (0.0 - 1.0) to an integer penalty (0 - 50)
            risk = task_risks.get(str(task.id), 0.0)
            risk_penalty = int(risk * 50)
            
            net_weight = priority_weight - risk_penalty
            objective_terms.append(task_assigned[str(task.id)] * net_weight)
            
        # 3. Minimize Workload Variance (Min-Max formulation)
        if len(dev_loads) > 1:
            max_load = model.new_int_var(0, max_possible_load, "max_load")
            min_load = model.new_int_var(0, max_possible_load, "min_load")
            model.add_max_equality(max_load, dev_loads)
            model.add_min_equality(min_load, dev_loads)
            
            # Penalize the difference between the most loaded and least loaded dev.
            # Scale the penalty so it balances with the priority weights.
            VARIANCE_PENALTY_WEIGHT = 2 
            objective_terms.append(-(max_load - min_load) * VARIANCE_PENALTY_WEIGHT)
            
        model.maximize(sum(objective_terms))
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.solve(model)
        
        assigned_tasks = []
        dropped_tasks = []
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for task in eligible_tasks:
                is_assigned = False
                for dev in developers:
                    if solver.value(assignments[(str(task.id), str(dev.id))]):
                        assigned_tasks.append({
                            "task_id": str(task.id),
                            "title": task.title,
                            "assigned_to": str(dev.id),
                            "assignee_name": dev.name,
                            "estimated_hours": task.estimated_hours,
                            "priority": task.priority.value,
                            "delay_risk": task_risks.get(str(task.id), 0.0),
                            "dependency_ids": task.dependency_ids
                        })
                        is_assigned = True
                if not is_assigned:
                    dropped_tasks.append({
                        "task_id": str(task.id),
                        "title": task.title,
                        "estimated_hours": task.estimated_hours,
                        "priority": task.priority.value,
                        "delay_risk": task_risks.get(str(task.id), 0.0),
                        "dependency_ids": task.dependency_ids
                    })
        else:
            logger.warning("CP-SAT solver could not find a feasible solution.")
            # Gracefully handle unsatisfiable cases
            for task in eligible_tasks:
                dropped_tasks.append({
                    "task_id": str(task.id),
                    "title": task.title,
                    "estimated_hours": task.estimated_hours,
                    "priority": task.priority.value,
                    "delay_risk": task_risks.get(str(task.id), 0.0),
                    "dependency_ids": task.dependency_ids
                })

        return {
            "assigned_tasks": assigned_tasks,
            "dropped_tasks": dropped_tasks,
            "solver_status": solver.status_name(status),
            "dev_capacities": dev_capacities,
            "risk_penalty_factor": risk_penalty_factor
        }
