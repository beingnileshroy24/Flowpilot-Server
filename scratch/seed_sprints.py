import asyncio
from app.database import init_db
from app.models.project import Project, Sprint
from app.models.task import Task, TaskStatus

async def main():
    await init_db()
    
    project_id = "6a58c2e790271becd483c1f7"
    project = await Project.get(project_id)
    if not project:
        print(f"Project {project_id} not found!")
        return
        
    print(f"Found project: {project.name}")
    
    # 1. Define dummy sprints
    dummy_sprints = [
        Sprint(
            id="sprint-1-comp",
            title="Sprint 1: Alpha Core",
            goal="Establish DB schemas and user auth flows",
            start_date="2026-07-01",
            end_date="2026-07-14",
            status="COMPLETED",
            capacity_hours=80.0
        ),
        Sprint(
            id="sprint-2-active",
            title="Sprint 2: Beta Launch",
            goal="Add interactive boards and comments system",
            start_date="2026-07-15",
            end_date="2026-07-28",
            status="ACTIVE",
            capacity_hours=120.0
        ),
        Sprint(
            id="sprint-3-plan",
            title="Sprint 3: AI Integration",
            goal="Integrate CP-SAT Sprint Solver and MLX explanations",
            start_date="2026-07-29",
            end_date="2026-08-11",
            status="PLANNING",
            capacity_hours=120.0
        )
    ]
    
    project.sprints = dummy_sprints
    await project.save()
    print("Successfully seeded 3 dummy sprints into project!")
    
    # 2. Make tasks eligible
    tasks = await Task.find(Task.project_id == project_id).to_list()
    print(f"Updating {len(tasks)} tasks to be eligible for planning...")
    for task in tasks:
        # Clear sprint_id so they are considered backlog / eligible for selection
        task.sprint_id = None
        # Ensure status is TODO or IN_PROGRESS (DONE tasks are excluded)
        if task.status == TaskStatus.DONE:
            task.status = TaskStatus.TODO
        # Give them some realistic estimated hours if they are 0
        if not task.estimated_hours or task.estimated_hours == 0:
            task.estimated_hours = 8.0
        await task.save()
        print(f"  - Updated task: {task.title} (Est. Hours: {task.estimated_hours})")

    print("Database seeding completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
