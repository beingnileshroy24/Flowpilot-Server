import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timezone
from app.models.project import Project, Sprint
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.user import User, UserRole
from app.services.feature_service import FeatureService
from app.services.prediction_engine import PredictionEngine
from app.database import init_db
from app.models.project_health import ProjectHealth
from app.models.sprint_prediction import SprintPrediction

@pytest.mark.anyio
async def test_feature_extraction_empty_and_baselines(client: TestClient):
    await init_db()
    # Tests FeatureService empty fallback and missing value baselines.
    # 1. Create a dummy project
    sprint_id = "sprint-test-empty"
    project = Project(
        name="Empty Project",
        description="Testing feature extraction",
        owner_id="test_owner",
        sprints=[
            Sprint(
                id=sprint_id,
                title="Test Sprint",
                status="ACTIVE",
                start_date=None, # Missing start date to trigger baseline creep
                capacity_hours=40.0
            )
        ]
    )
    await project.insert()

    # 2. Build sprint vector with no tasks (empty sprint)
    empty_vector = await FeatureService.build_sprint_vector(sprint_id)
    assert empty_vector["velocity_drift"] == 0.0
    assert empty_vector["scope_creep_points"] == 0.0
    assert empty_vector["sentiment_volatility"] == 0.0
    assert empty_vector["burndown_history"] == [0.0] * 10

    # 3. Create a task in the sprint to verify baseline fallback
    task = Task(
        project_id=str(project.id),
        type="TASK",
        title="Test Task",
        description="Testing feature extraction task",
        status=TaskStatus.TODO,
        priority=TaskPriority.HIGH,
        sprint_id=sprint_id,
        estimated_hours=5.0
    )
    await task.insert()

    # Build vector now that there is a task, but missing velocity/comments/start_date
    partial_vector = await FeatureService.build_sprint_vector(sprint_id)
    assert partial_vector["velocity_drift"] == FeatureService.VELOCITY_DRIFT_BASELINE
    assert partial_vector["scope_creep_points"] == FeatureService.SCOPE_CREEP_POINTS_BASELINE
    assert partial_vector["sentiment_volatility"] == FeatureService.SENTINEL_VOLATILITY_BASELINE
    assert len(partial_vector["burndown_history"]) == 10

@pytest.mark.anyio
async def test_onnx_prediction_bounds_and_cache(client: TestClient):
    await init_db()
    # 1. Create a dummy user
    user = User(
        name="Test Lead",
        email="testlead@example.com",
        hashed_password="hashedpassword",
        role=UserRole.MANAGER
    )
    await user.insert()

    # 2. Create project
    sprint_id = "sprint-test-onnx"
    project = Project(
        name="ONNX Test Project",
        description="ONNX and Caching test",
        owner_id=str(user.id),
        developer_ids=[str(user.id)],
        lead_developer_id=str(user.id),
        sprints=[
            Sprint(
                id=sprint_id,
                title="Test Sprint",
                status="ACTIVE",
                start_date=datetime.now(timezone.utc).isoformat(),
                capacity_hours=40.0
            )
        ]
    )
    await project.insert()

    # 3. Create tasks
    task = Task(
        project_id=str(project.id),
        type="TASK",
        title="Task 1",
        description="ONNX test task",
        status=TaskStatus.TODO,
        priority=TaskPriority.HIGH,
        sprint_id=sprint_id,
        estimated_hours=8.0,
        assigned_to_id=str(user.id)
    )
    await task.insert()

    # Run predictions first time (cold run, cache miss)
    health1 = await PredictionEngine.run_project_prediction_pipeline(str(project.id), force_recalculate=True)
    assert health1 is not None
    assert 0.0 <= health1.health_score <= 100.0
    
    # Run predictions second time (should hit cache)
    health2 = await PredictionEngine.run_project_prediction_pipeline(str(project.id), force_recalculate=False)
    assert health2 is not None
    # Verify it returned the cached object by checking they have the same ID
    assert health1.id == health2.id

    # Run predictions third time with force_recalculate=True (should bypass cache)
    health3 = await PredictionEngine.run_project_prediction_pipeline(str(project.id), force_recalculate=True)
    assert health3 is not None
    # Should be a new calculation record
    assert health3.id != health1.id
