"""
Software Estimation Calculator

Demonstrates three widely-used software estimation techniques:

1. COCOMO II Basic Model
   Effort (person-months) = a * (KLOC)^b
   Where a and b are mode-dependent constants:
     - Organic:      a=2.4, b=1.05  (small, familiar projects)
     - Semi-detached: a=3.0, b=1.12  (medium complexity)
     - Embedded:     a=3.6, b=1.20  (highly constrained)

2. Three-Point Estimation (PERT)
   Expected = (Optimistic + 4 * Most_Likely + Pessimistic) / 6
   Std_Dev  = (Pessimistic - Optimistic) / 6

3. Story Point Velocity Calculator
   Estimates remaining effort based on historical team velocity.

4. Function Point Counting (simplified)
   Counts inputs, outputs, queries, files, and interfaces
   to produce an unadjusted function point total.

Run:
    python 06_estimation_calculator.py
"""

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. COCOMO II Basic Model
# ---------------------------------------------------------------------------

COCOMO_MODES = {
    "organic":       (2.4, 1.05),
    "semi-detached": (3.0, 1.12),
    "embedded":      (3.6, 1.20),
}


def cocomo_basic(kloc: float, mode: str = "semi-detached") -> dict[str, float]:
    """
    Estimate effort and duration using COCOMO II Basic model.

    Args:
        kloc: Thousands of lines of code (estimated)
        mode: 'organic', 'semi-detached', or 'embedded'

    Returns:
        Dictionary with effort (person-months), duration (months),
        team size, and productivity.
    """
    if mode not in COCOMO_MODES:
        raise ValueError(f"Mode must be one of {list(COCOMO_MODES.keys())}")
    if kloc <= 0:
        raise ValueError("KLOC must be positive")

    a, b = COCOMO_MODES[mode]
    effort_pm = a * (kloc ** b)          # person-months
    duration_m = 2.5 * (effort_pm ** 0.38)  # calendar months
    team_size = effort_pm / duration_m
    productivity = (kloc * 1000) / effort_pm  # LOC per person-month

    return {
        "effort_person_months": round(effort_pm, 2),
        "duration_months": round(duration_m, 2),
        "team_size": round(team_size, 1),
        "productivity_loc_pm": round(productivity),
    }


# ---------------------------------------------------------------------------
# 2. Three-Point Estimation (PERT)
# ---------------------------------------------------------------------------

@dataclass
class ThreePointTask:
    name: str
    optimistic: float    # Best case (days)
    most_likely: float   # Most probable (days)
    pessimistic: float   # Worst case (days)

    @property
    def expected(self) -> float:
        return (self.optimistic + 4 * self.most_likely + self.pessimistic) / 6

    @property
    def std_dev(self) -> float:
        return (self.pessimistic - self.optimistic) / 6

    @property
    def variance(self) -> float:
        return self.std_dev ** 2


def pert_project(tasks: list[ThreePointTask], confidence: float = 0.90) -> dict:
    """
    Aggregate PERT estimates across tasks for a project-level estimate.

    Uses the Central Limit Theorem: the sum of independent normal distributions
    is also normal, with mean = sum of means, variance = sum of variances.

    Args:
        tasks: List of ThreePointTask instances.
        confidence: Desired confidence level (0.80, 0.90, or 0.95).

    Returns:
        Dictionary with expected total, std dev, and confidence interval.
    """
    z_scores = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}
    z = z_scores.get(confidence, 1.645)

    total_expected = sum(t.expected for t in tasks)
    total_variance = sum(t.variance for t in tasks)
    total_std_dev = math.sqrt(total_variance)

    return {
        "expected_days": round(total_expected, 1),
        "std_dev_days": round(total_std_dev, 1),
        "confidence_level": f"{int(confidence * 100)}%",
        "lower_bound": round(total_expected - z * total_std_dev, 1),
        "upper_bound": round(total_expected + z * total_std_dev, 1),
        "task_count": len(tasks),
    }


# ---------------------------------------------------------------------------
# 3. Story Point Velocity Calculator
# ---------------------------------------------------------------------------

def velocity_estimate(
    remaining_points: int,
    sprint_velocity: list[int],
    sprint_length_weeks: int = 2,
) -> dict:
    """
    Estimate time to complete remaining story points using historical velocity.

    Args:
        remaining_points: Total story points left in the backlog.
        sprint_velocity: Points completed per sprint in recent history.
        sprint_length_weeks: Length of one sprint in weeks.

    Returns:
        Dictionary with average velocity, sprints needed, and calendar weeks.
    """
    if not sprint_velocity:
        raise ValueError("Sprint velocity history must not be empty")

    avg_velocity = sum(sprint_velocity) / len(sprint_velocity)
    min_velocity = min(sprint_velocity)
    max_velocity = max(sprint_velocity)

    sprints_avg = math.ceil(remaining_points / avg_velocity)
    sprints_optimistic = math.ceil(remaining_points / max_velocity)
    sprints_pessimistic = math.ceil(remaining_points / min_velocity)

    return {
        "average_velocity": round(avg_velocity, 1),
        "sprints_needed_avg": sprints_avg,
        "sprints_needed_optimistic": sprints_optimistic,
        "sprints_needed_pessimistic": sprints_pessimistic,
        "weeks_avg": sprints_avg * sprint_length_weeks,
        "weeks_optimistic": sprints_optimistic * sprint_length_weeks,
        "weeks_pessimistic": sprints_pessimistic * sprint_length_weeks,
    }


# ---------------------------------------------------------------------------
# 4. Function Point Counting (simplified unadjusted)
# ---------------------------------------------------------------------------

# Complexity weights per IFPUG standard (simplified to average weights)
FP_WEIGHTS = {
    "external_inputs":   4,   # screens, forms
    "external_outputs":  5,   # reports, screens with derived data
    "external_queries":  4,   # online queries with no derived output
    "internal_files":    10,  # logical internal data groups
    "external_interfaces": 7, # data shared with other systems
}


def count_function_points(components: dict[str, int]) -> dict:
    """
    Compute Unadjusted Function Points (UFP) from component counts.

    Args:
        components: Dict mapping component type to count.
                    Keys: external_inputs, external_outputs, external_queries,
                          internal_files, external_interfaces

    Returns:
        Dictionary with individual FP contributions and total UFP.
    """
    breakdown = {}
    total = 0
    for key, weight in FP_WEIGHTS.items():
        count = components.get(key, 0)
        contribution = count * weight
        breakdown[key] = {"count": count, "weight": weight, "fp": contribution}
        total += contribution

    # Rough LOC conversion (language-dependent; Python ~50 LOC/FP)
    estimated_kloc_python = (total * 50) / 1000

    return {
        "breakdown": breakdown,
        "unadjusted_fp": total,
        "estimated_kloc_python": round(estimated_kloc_python, 2),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def demo_cocomo() -> None:
    print_section("1. COCOMO II Basic Estimation")
    project_kloc = 15.0
    print(f"\n  Project size: {project_kloc} KLOC\n")
    for mode in COCOMO_MODES:
        result = cocomo_basic(project_kloc, mode)
        print(f"  Mode: {mode.capitalize()}")
        print(f"    Effort:       {result['effort_person_months']} person-months")
        print(f"    Duration:     {result['duration_months']} months")
        print(f"    Team size:    {result['team_size']} people")
        print(f"    Productivity: {result['productivity_loc_pm']} LOC/person-month")
        print()


def demo_pert() -> None:
    print_section("2. Three-Point Estimation (PERT)")
    tasks = [
        ThreePointTask("Requirements analysis",   3,  5, 10),
        ThreePointTask("System design",           5,  8, 15),
        ThreePointTask("Backend development",    20, 30, 50),
        ThreePointTask("Frontend development",   15, 22, 40),
        ThreePointTask("Integration & testing",   8, 12, 20),
        ThreePointTask("Deployment & handover",   2,  4,  8),
    ]

    print(f"\n  {'Task':<30} {'Opt':>5} {'ML':>5} {'Pes':>5} {'Exp':>6} {'SD':>5}")
    print("  " + "-" * 58)
    for t in tasks:
        print(f"  {t.name:<30} {t.optimistic:>5.0f} {t.most_likely:>5.0f} "
              f"{t.pessimistic:>5.0f} {t.expected:>6.1f} {t.std_dev:>5.1f}")

    result = pert_project(tasks, confidence=0.90)
    print(f"\n  Project Total ({result['confidence_level']} confidence):")
    print(f"    Expected duration: {result['expected_days']} days")
    print(f"    Std deviation:     {result['std_dev_days']} days")
    print(f"    90% range:         {result['lower_bound']}–{result['upper_bound']} days")


def demo_velocity() -> None:
    print_section("3. Story Point Velocity Calculator")
    history = [38, 42, 35, 40, 44, 39]
    remaining = 320
    result = velocity_estimate(remaining, history, sprint_length_weeks=2)

    print(f"\n  Remaining backlog:  {remaining} story points")
    print(f"  Sprint history:     {history}")
    print(f"  Average velocity:   {result['average_velocity']} pts/sprint\n")
    print(f"  Scenario        Sprints  Weeks")
    print(f"  {'─'*35}")
    print(f"  Average         {result['sprints_needed_avg']:>7}  {result['weeks_avg']:>5}")
    print(f"  Optimistic      {result['sprints_needed_optimistic']:>7}  {result['weeks_optimistic']:>5}")
    print(f"  Pessimistic     {result['sprints_needed_pessimistic']:>7}  {result['weeks_pessimistic']:>5}")


def demo_function_points() -> None:
    print_section("4. Function Point Counting")
    components = {
        "external_inputs":    12,  # data entry forms
        "external_outputs":    8,  # reports
        "external_queries":    6,  # search/query screens
        "internal_files":      5,  # main data entities
        "external_interfaces": 3,  # third-party API integrations
    }
    result = count_function_points(components)

    print(f"\n  {'Component':<25} {'Count':>6} {'Weight':>7} {'FP':>6}")
    print("  " + "-" * 46)
    for key, data in result["breakdown"].items():
        label = key.replace("_", " ").title()
        print(f"  {label:<25} {data['count']:>6}   x {data['weight']:>3}  = {data['fp']:>4}")
    print("  " + "-" * 46)
    print(f"  {'Unadjusted Function Points':<25} {'':>6} {'':>7} {result['unadjusted_fp']:>6}")
    print(f"\n  Estimated size (Python, ~50 LOC/FP): {result['estimated_kloc_python']} KLOC")


if __name__ == "__main__":
    print("\nSoftware Estimation Calculator")
    print("Comparing COCOMO II, PERT, Velocity, and Function Points\n")
    demo_cocomo()
    demo_pert()
    demo_velocity()
    demo_function_points()
    print("\n" + "=" * 60)
    print("  Done. Use these estimates as starting points, not commitments.")
    print("=" * 60 + "\n")
