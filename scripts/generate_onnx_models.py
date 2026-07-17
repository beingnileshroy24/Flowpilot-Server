import os
import torch
import torch.nn as nn

class SprintFailureModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Features: [velocity_drift, scope_creep_points, sentiment_volatility]
        self.linear = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.25, 0.15, 3.0]]))
            self.linear.bias.copy_(torch.tensor([-1.5]))

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class TaskDelayModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Features: [age_hours, reopens, density]
        self.linear = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.005, 1.2, 0.8]]))
            self.linear.bias.copy_(torch.tensor([-1.5]))

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class ChronosTinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 10 daily burndown points
        # Output: 10 daily forecast points
        self.linear = nn.Linear(10, 10)
        with torch.no_grad():
            # Initializing with identity or down-trending weights to mock Chronos behavior
            self.linear.weight.copy_(torch.eye(10) * 0.9)
            self.linear.bias.copy_(torch.zeros(10))

    def forward(self, x):
        return self.linear(x)

def main():
    registry_dir = "app/ai/models/registry"
    os.makedirs(registry_dir, exist_ok=True)

    # 1. Export sprint failure model
    sprint_model = SprintFailureModel()
    sprint_dummy_in = torch.randn(1, 3)
    sprint_path = os.path.join(registry_dir, "sprint_failure_v1.0.onnx")
    torch.onnx.export(sprint_model, (sprint_dummy_in,), sprint_path, input_names=["input"], output_names=["output"])
    print(f"Exported {sprint_path}")

    # 2. Export task delay model
    task_model = TaskDelayModel()
    task_dummy_in = torch.randn(1, 3)
    task_path = os.path.join(registry_dir, "task_delay_v1.0.onnx")
    torch.onnx.export(task_model, (task_dummy_in,), task_path, input_names=["input"], output_names=["output"])
    print(f"Exported {task_path}")

    # 3. Export Chronos tiny model
    chronos_model = ChronosTinyModel()
    chronos_dummy_in = torch.randn(1, 10)
    chronos_path = os.path.join(registry_dir, "chronos_v1.0.onnx")
    torch.onnx.export(chronos_model, (chronos_dummy_in,), chronos_path, input_names=["input"], output_names=["output"])
    print(f"Exported {chronos_path}")

if __name__ == "__main__":
    main()
