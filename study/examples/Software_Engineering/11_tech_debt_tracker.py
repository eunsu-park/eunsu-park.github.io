"""
Technical Debt Tracker

Demonstrates technical debt management concepts from software engineering:
- Debt categorization (code, architecture, test, documentation, security)
- Interest rate modelling: ongoing cost that grows each sprint
- ROI-based prioritization: benefit of paying off vs. effort required
- Debt report generation with actionable recommendations
- Sprint-by-sprint simulation of debt accumulation vs. payoff
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DebtType(Enum):
    CODE = "Code Quality"
    ARCHITECTURE = "Architecture"
    TEST = "Test Coverage"
    DOCUMENTATION = "Documentation"
    SECURITY = "Security"
    DEPENDENCY = "Dependency"


class Severity(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class DebtItem:
    """A single technical debt item."""
    id: str
    description: str
    debt_type: DebtType
    severity: Severity
    effort_days: float          # estimated days to fix
    interest_rate: float        # extra cost per sprint (in hours) if not fixed
    added_sprint: int           # sprint number when debt was introduced
    paid_off_sprint: Optional[int] = None

    @property
    def is_active(self) -> bool:
        return self.paid_off_sprint is None

    def accumulated_interest(self, current_sprint: int) -> float:
        """Total hours lost to this debt up to current_sprint."""
        if not self.is_active:
            end = self.paid_off_sprint
        else:
            end = current_sprint
        sprints_active = max(0, end - self.added_sprint)
        return sprints_active * self.interest_rate

    def roi_score(self, current_sprint: int) -> float:
        """
        ROI = (interest saved over next 10 sprints) / effort_days
        Higher score → pay off sooner.
        """
        if self.effort_days == 0:
            return float("inf")
        future_savings = self.interest_rate * 10  # hours saved
        return future_savings / self.effort_days


def generate_report(items: list[DebtItem], current_sprint: int) -> None:
    """Print a structured technical debt report."""
    active = [i for i in items if i.is_active]
    paid = [i for i in items if not i.is_active]

    total_interest = sum(i.accumulated_interest(current_sprint) for i in active)
    total_effort = sum(i.effort_days for i in active)

    print("=" * 68)
    print("  TECHNICAL DEBT REPORT")
    print(f"  Sprint {current_sprint}  |  Active items: {len(active)}  |  Resolved: {len(paid)}")
    print("=" * 68)

    print(f"\nDebt Summary:")
    print(f"  Total accumulated interest (active debt) : {total_interest:.1f} hours")
    print(f"  Estimated payoff effort                  : {total_effort:.1f} dev-days")
    print(f"  Interest / sprint (ongoing burn)         : "
          f"{sum(i.interest_rate for i in active):.1f} hours/sprint")

    # Breakdown by type
    print("\nDebt by Category:")
    by_type: dict[str, list[DebtItem]] = {}
    for item in active:
        by_type.setdefault(item.debt_type.value, []).append(item)
    for dtype, group in sorted(by_type.items()):
        hrs = sum(i.accumulated_interest(current_sprint) for i in group)
        print(f"  {dtype:<22} {len(group):>2} items   {hrs:>6.1f} h accumulated")

    # Prioritized list
    print("\nPrioritized Payoff List (by ROI):")
    print(f"  {'ID':<8} {'Description':<32} {'Sev':<9} {'Effort':>7} {'ROI':>6}  Recommendation")
    print("  " + "-" * 76)
    ranked = sorted(active, key=lambda i: i.roi_score(current_sprint), reverse=True)
    for rank, item in enumerate(ranked, 1):
        roi = item.roi_score(current_sprint)
        roi_str = f"{roi:.2f}" if roi != float("inf") else "∞"
        if rank <= 2:
            rec = ">>> FIX NOW"
        elif item.severity == Severity.CRITICAL:
            rec = ">> Urgent"
        elif roi > 2.0:
            rec = "> This sprint"
        else:
            rec = "  Backlog"
        print(f"  {item.id:<8} {item.description:<32} {item.severity.name:<9} "
              f"{item.effort_days:>5.1f}d  {roi_str:>6}  {rec}")

    # Resolved items
    if paid:
        print(f"\nResolved Debt ({len(paid)} items):")
        for item in paid:
            sprint_count = item.paid_off_sprint - item.added_sprint
            total = item.accumulated_interest(item.paid_off_sprint)
            print(f"  [{item.id}] {item.description[:40]}  "
                  f"(paid sprint {item.paid_off_sprint}, cost {total:.0f}h over {sprint_count} sprints)")

    print("=" * 68)


def simulate_sprints(items: list[DebtItem], total_sprints: int,
                     payoff_capacity_hours: float = 8.0) -> None:
    """
    Simulate debt accumulation over sprints.
    Each sprint: pay off highest-ROI items if capacity allows.
    """
    print("\nSPRINT SIMULATION (auto-payoff with 8h capacity/sprint)")
    print(f"{'Sprint':>7}  {'Active':>7}  {'Interest/sp':>11}  {'Accumulated':>12}  {'Action'}")
    print("-" * 70)

    payoff_sprint = {}
    paid_ids: set[str] = set()

    for sprint in range(1, total_sprints + 1):
        active = [i for i in items if i.added_sprint <= sprint and i.id not in paid_ids]
        interest_this_sprint = sum(i.interest_rate for i in active)
        accumulated = sum(i.accumulated_interest(sprint) for i in active)

        # Attempt payoff
        budget = payoff_capacity_hours
        action_parts = []
        for item in sorted(active, key=lambda i: i.roi_score(sprint), reverse=True):
            cost_hours = item.effort_days * 8
            if cost_hours <= budget:
                budget -= cost_hours
                paid_ids.add(item.id)
                payoff_sprint[item.id] = sprint
                action_parts.append(item.id)

        action = "Paid: " + ", ".join(action_parts) if action_parts else "—"
        print(f"{sprint:>7}  {len(active):>7}  {interest_this_sprint:>11.1f}  "
              f"{accumulated:>12.1f}  {action}")


if __name__ == "__main__":
    CURRENT_SPRINT = 8

    debt_items = [
        DebtItem("TD-001", "No unit tests for payment module",
                 DebtType.TEST, Severity.CRITICAL,
                 effort_days=3.0, interest_rate=4.0, added_sprint=2),

        DebtItem("TD-002", "Hardcoded API keys in config.py",
                 DebtType.SECURITY, Severity.CRITICAL,
                 effort_days=0.5, interest_rate=6.0, added_sprint=1),

        DebtItem("TD-003", "Monolithic UserService (God Object)",
                 DebtType.ARCHITECTURE, Severity.HIGH,
                 effort_days=5.0, interest_rate=3.0, added_sprint=3),

        DebtItem("TD-004", "SQLAlchemy 1.3 → 2.0 migration pending",
                 DebtType.DEPENDENCY, Severity.HIGH,
                 effort_days=2.0, interest_rate=1.5, added_sprint=4),

        DebtItem("TD-005", "Copy-paste logic in 6 report generators",
                 DebtType.CODE, Severity.MEDIUM,
                 effort_days=1.5, interest_rate=2.0, added_sprint=3),

        DebtItem("TD-006", "No API documentation (OpenAPI spec missing)",
                 DebtType.DOCUMENTATION, Severity.MEDIUM,
                 effort_days=2.0, interest_rate=1.0, added_sprint=5),

        DebtItem("TD-007", "N+1 query in product listing endpoint",
                 DebtType.CODE, Severity.HIGH,
                 effort_days=0.5, interest_rate=2.5, added_sprint=6),

        DebtItem("TD-008", "No retry/circuit-breaker on payment API calls",
                 DebtType.ARCHITECTURE, Severity.MEDIUM,
                 effort_days=1.5, interest_rate=1.0, added_sprint=7,
                 paid_off_sprint=8),   # already resolved
    ]

    generate_report(debt_items, CURRENT_SPRINT)
    simulate_sprints(debt_items, total_sprints=12, payoff_capacity_hours=8.0)
