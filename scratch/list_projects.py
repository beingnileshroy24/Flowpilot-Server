import asyncio
from app.database import init_db
from app.models.project import Project
from app.models.task import Task
from app.models.user import User

async def main():
    await init_db()
    projects = await Project.find_all().to_list()
    print(f"Found {len(projects)} projects:")
    for proj in projects:
        print(f"Project ID: {proj.id}")
        print(f"Name: {proj.name}")
        print(f"Developer IDs: {proj.developer_ids}")
        print(f"Lead Developer ID: {proj.lead_developer_id}")
        print(f"Sprints Count: {len(proj.sprints)}")
        for sprint in proj.sprints:
            print(f"  - Sprint: {sprint.title} (ID: {sprint.id}, Status: {sprint.status})")
        tasks = await Task.find(Task.project_id == str(proj.id)).to_list()
        print(f"Tasks Count: {len(tasks)}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
