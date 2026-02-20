"""
Gantt Chart Generator with Critical Path Method (CPM)

Demonstrates project scheduling concepts from software engineering:
- Task definition with dependencies
- Forward pass (earliest start/finish)
- Backward pass (latest start/finish)
- Critical path identification
- Slack (float) calculation
- ASCII Gantt chart rendering
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


@dataclass
class Task:
    name: str
    duration: int          # working days
    dependencies: list[str] = field(default_factory=list)
    # Computed by CPM
    early_start: int = 0
    early_finish: int = 0
    late_start: int = 0
    late_finish: int = 0

    @property
    def slack(self) -> int:
        return self.late_start - self.early_start

    @property
    def is_critical(self) -> bool:
        return self.slack == 0


def forward_pass(tasks: dict[str, Task]) -> None:
    """Calculate earliest start and finish times."""
    resolved: set[str] = set()

    def resolve(name: str) -> None:
        if name in resolved:
            return
        task = tasks[name]
        for dep in task.dependencies:
            resolve(dep)
        if task.dependencies:
            task.early_start = max(tasks[d].early_finish for d in task.dependencies)
        else:
            task.early_start = 0
        task.early_finish = task.early_start + task.duration
        resolved.add(name)

    for name in tasks:
        resolve(name)


def backward_pass(tasks: dict[str, Task], project_duration: int) -> None:
    """Calculate latest start and finish times."""
    # Initialize: tasks with no successors finish at project end
    successors: dict[str, list[str]] = {name: [] for name in tasks}
    for name, task in tasks.items():
        for dep in task.dependencies:
            successors[dep].append(name)

    resolved: set[str] = set()

    def resolve(name: str) -> None:
        if name in resolved:
            return
        task = tasks[name]
        succs = successors[name]
        for s in succs:
            resolve(s)
        if succs:
            task.late_finish = min(tasks[s].late_start for s in succs)
        else:
            task.late_finish = project_duration
        task.late_start = task.late_finish - task.duration
        resolved.add(name)

    # Resolve in reverse dependency order
    for name in reversed(list(tasks.keys())):
        resolve(name)


def find_critical_path(tasks: dict[str, Task]) -> list[str]:
    """Return task names on the critical path, in order."""
    critical = [name for name, t in tasks.items() if t.is_critical]
    # Sort by early_start to get execution order
    critical.sort(key=lambda n: tasks[n].early_start)
    return critical


def render_gantt(tasks: dict[str, Task], project_duration: int,
                 start_date: date, bar_width: int = 40) -> str:
    """Render an ASCII Gantt chart."""
    scale = bar_width / project_duration
    lines: list[str] = []

    # Header
    name_col = max(len(t.name) for t in tasks.values()) + 2
    lines.append(f"{'Task':<{name_col}} {'C':1} {'Slack':>5}  Timeline")
    lines.append("-" * (name_col + 8 + bar_width))

    # Date ruler (show week numbers)
    ruler = [" "] * bar_width
    for day in range(project_duration):
        if day % 5 == 0:
            label = str(day // 5 + 1)
            pos = int(day * scale)
            if pos < bar_width:
                ruler[pos] = label
    lines.append(f"{'':>{name_col}}   {'':>5}  {''.join(ruler)}")

    for name, task in tasks.items():
        bar = ["."] * bar_width
        start_pos = int(task.early_start * scale)
        end_pos = int(task.early_finish * scale)
        end_pos = min(end_pos, bar_width)
        for i in range(start_pos, end_pos):
            bar[i] = "#" if task.is_critical else "="
        critical_marker = "*" if task.is_critical else " "
        bar_str = "".join(bar)
        lines.append(
            f"{name:<{name_col}} {critical_marker} {task.slack:>5}  {bar_str}"
        )

    lines.append("-" * (name_col + 8 + bar_width))
    lines.append("Legend: # = critical path  = = non-critical  * = on critical path")

    # Date range
    end_date = start_date + timedelta(days=project_duration - 1)
    lines.append(f"Project: {start_date} → {end_date}  ({project_duration} working days)")
    return "\n".join(lines)


def schedule_project(tasks: dict[str, Task]) -> int:
    """Run CPM and return total project duration."""
    forward_pass(tasks)
    project_duration = max(t.early_finish for t in tasks.values())
    backward_pass(tasks, project_duration)
    return project_duration


if __name__ == "__main__":
    # --- Example: E-Commerce Platform Development ---
    tasks = {
        "Requirements":    Task("Requirements",    5, []),
        "UI Design":       Task("UI Design",       8, ["Requirements"]),
        "DB Schema":       Task("DB Schema",       4, ["Requirements"]),
        "Auth Service":    Task("Auth Service",    6, ["DB Schema"]),
        "Product API":     Task("Product API",     7, ["DB Schema"]),
        "Cart Service":    Task("Cart Service",    5, ["Auth Service", "Product API"]),
        "Payment Integ":   Task("Payment Integ",  10, ["Auth Service"]),
        "Frontend Dev":    Task("Frontend Dev",   12, ["UI Design", "Product API"]),
        "Integration QA":  Task("Integration QA",  6, ["Cart Service", "Payment Integ",
                                                        "Frontend Dev"]),
        "UAT & Launch":    Task("UAT & Launch",    4, ["Integration QA"]),
    }

    project_duration = schedule_project(tasks)
    critical_path = find_critical_path(tasks)
    start_date = date(2025, 3, 3)  # Monday

    print("=" * 65)
    print("  E-COMMERCE PLATFORM — PROJECT SCHEDULE (CPM)")
    print("=" * 65)

    print(f"\nProject Duration : {project_duration} working days")
    print(f"Critical Path    : {' → '.join(critical_path)}")
    print(f"Start Date       : {start_date}")

    print("\nTask Details:")
    header = f"{'Task':<18} {'Dur':>4} {'ES':>4} {'EF':>4} {'LS':>4} {'LF':>4} {'Slack':>5}  Critical"
    print(header)
    print("-" * len(header))
    for name, task in tasks.items():
        marker = "YES" if task.is_critical else "—"
        print(f"{name:<18} {task.duration:>4} {task.early_start:>4} "
              f"{task.early_finish:>4} {task.late_start:>4} "
              f"{task.late_finish:>4} {task.slack:>5}  {marker}")

    print("\n" + render_gantt(tasks, project_duration, start_date))
