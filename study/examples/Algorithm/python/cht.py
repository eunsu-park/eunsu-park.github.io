"""
Convex Hull Trick (CHT) - DP Optimization Technique

The Convex Hull Trick is used to optimize certain DP recurrences from O(n^2) to O(n log n).
It's applicable when the recurrence has the form:
    dp[i] = min/max(dp[j] + cost(j, i)) for all j < i
where cost(j, i) can be expressed as a linear function in terms of j.

Typical form: dp[i] = min(dp[j] + a[i] * b[j] + c[i])
This can be rewritten as: dp[i] = min(b[j] * a[i] + (dp[j] + c[i]))
Which is a line equation: y = mx + b, where m = b[j], b = dp[j] + c[i], x = a[i]

Time Complexity: O(n log n) with Li Chao Tree, O(n) if queries are monotonic
Space Complexity: O(n)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Line:
    """Represents a line y = m*x + c"""
    m: float  # slope
    c: float  # y-intercept

    def eval(self, x: float) -> float:
        """Evaluate line at position x"""
        return self.m * x + self.c

    def intersect_x(self, other: 'Line') -> float:
        """Find x-coordinate where this line intersects with other"""
        if self.m == other.m:
            return float('inf')
        return (other.c - self.c) / (self.m - other.m)


class ConvexHullTrick:
    """
    Convex Hull Trick for minimum queries.
    Maintains a lower envelope of lines.
    Assumes lines are added in decreasing order of slope (for online variant).
    """

    def __init__(self):
        self.lines: List[Line] = []

    def _bad_line(self, l1: Line, l2: Line, l3: Line) -> bool:
        """
        Check if line l2 is redundant (will never be optimal).
        l2 is bad if the intersection of (l1, l3) is left of intersection of (l1, l2).
        """
        # Cross product method to avoid division
        # (l3.c - l1.c) * (l1.m - l2.m) < (l2.c - l1.c) * (l1.m - l3.m)
        return (l3.c - l1.c) * (l1.m - l2.m) <= (l2.c - l1.c) * (l1.m - l3.m)

    def add_line(self, line: Line) -> None:
        """
        Add a line to the convex hull.
        Assumes lines are added in decreasing order of slope.
        """
        # Remove lines that become irrelevant
        while len(self.lines) >= 2:
            if self._bad_line(self.lines[-2], self.lines[-1], line):
                self.lines.pop()
            else:
                break
        self.lines.append(line)

    def query(self, x: float) -> float:
        """
        Find minimum value at position x.
        Uses binary search if queries are not monotonic.
        """
        if not self.lines:
            return float('inf')

        # Binary search for the best line
        left, right = 0, len(self.lines) - 1
        while left < right:
            mid = (left + right) // 2
            # Check if line at mid or mid+1 is better at x
            if self.lines[mid].eval(x) > self.lines[mid + 1].eval(x):
                left = mid + 1
            else:
                right = mid

        return self.lines[left].eval(x)


class ConvexHullTrickMax:
    """
    Convex Hull Trick for maximum queries.
    Maintains an upper envelope of lines.
    """

    def __init__(self):
        self.cht_min = ConvexHullTrick()

    def add_line(self, line: Line) -> None:
        """Add line by negating for maximum query"""
        self.cht_min.add_line(Line(-line.m, -line.c))

    def query(self, x: float) -> float:
        """Find maximum value at position x"""
        return -self.cht_min.query(x)


class LiChaoTree:
    """
    Li Chao Tree - Supports dynamic line insertion without slope restrictions.
    Works for any order of line insertion.
    Time Complexity: O(log n) per insertion and query
    """

    def __init__(self, x_min: int, x_max: int):
        self.x_min = x_min
        self.x_max = x_max
        self.tree: dict = {}  # node_id -> Line

    def _update(self, line: Line, node_id: int, left: int, right: int) -> None:
        """Recursively insert line into the tree"""
        mid = (left + right) // 2

        if node_id not in self.tree:
            self.tree[node_id] = line
            return

        cur_line = self.tree[node_id]

        # Determine which line is better at left, mid, right
        left_better = line.eval(left) < cur_line.eval(left)
        mid_better = line.eval(mid) < cur_line.eval(mid)

        if mid_better:
            line, self.tree[node_id] = self.tree[node_id], line

        if left == right:
            return

        if left_better != mid_better:
            self._update(line, 2 * node_id, left, mid)
        else:
            self._update(line, 2 * node_id + 1, mid + 1, right)

    def add_line(self, line: Line) -> None:
        """Add a line to the Li Chao Tree"""
        self._update(line, 1, self.x_min, self.x_max)

    def _query(self, x: int, node_id: int, left: int, right: int) -> float:
        """Recursively query minimum value at position x"""
        if node_id not in self.tree:
            return float('inf')

        if left == right:
            return self.tree[node_id].eval(x)

        mid = (left + right) // 2
        result = self.tree[node_id].eval(x)

        if x <= mid:
            result = min(result, self._query(x, 2 * node_id, left, mid))
        else:
            result = min(result, self._query(x, 2 * node_id + 1, mid + 1, right))

        return result

    def query(self, x: int) -> float:
        """Find minimum value at position x"""
        return self._query(x, 1, self.x_min, self.x_max)


def solve_machine_cost_problem(n: int, costs: List[int], machines: List[int]) -> int:
    """
    Example Problem: Machine Cost Minimization

    Problem: You have n tasks. For task i, you can:
    - Use a new machine with cost costs[i]
    - Use a previous machine j (j < i) with cost (machines[i] - machines[j])^2

    Find minimum total cost.

    DP recurrence: dp[i] = min(costs[i], min(dp[j] + (machines[i] - machines[j])^2))
    Expand: dp[i] = min(costs[i], min(dp[j] + machines[i]^2 - 2*machines[i]*machines[j] + machines[j]^2))
    Rewrite as line: y = mx + b where m = -2*machines[j], b = dp[j] + machines[j]^2
    Query at x = machines[i], then add machines[i]^2
    """
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    cht = ConvexHullTrick()
    # Initial line for dp[0]
    cht.add_line(Line(-2 * machines[0], dp[0] + machines[0] ** 2))

    for i in range(1, n + 1):
        # Option 1: Use new machine
        dp[i] = costs[i - 1]

        # Option 2: Use previous machine (query CHT)
        if machines[i - 1] >= machines[0]:  # Ensure valid query
            cost_from_prev = cht.query(machines[i - 1]) + machines[i - 1] ** 2
            dp[i] = min(dp[i], cost_from_prev)

        # Add current state as a line for future queries
        cht.add_line(Line(-2 * machines[i], dp[i] + machines[i] ** 2))

    return int(dp[n])


def solve_slope_optimization(n: int, a: List[int], b: List[int]) -> int:
    """
    Example Problem: Slope Optimization

    DP recurrence: dp[i] = min(dp[j] + a[i] * b[j]) for j < i
    This is a line equation: y = b[j] * x + dp[j], where x = a[i]
    """
    dp = [float('inf')] * n
    dp[0] = 0

    cht = ConvexHullTrick()
    cht.add_line(Line(b[0], dp[0]))

    for i in range(1, n):
        dp[i] = cht.query(a[i])
        cht.add_line(Line(b[i], dp[i]))

    return int(dp[n - 1])


if __name__ == "__main__":
    print("=== Convex Hull Trick Examples ===\n")

    # Test 1: Machine Cost Problem
    print("Test 1: Machine Cost Problem")
    n = 5
    costs = [10, 15, 20, 12, 18]
    machines = [1, 2, 3, 4, 5]
    result = solve_machine_cost_problem(n, costs, machines)
    print(f"Tasks: {n}, Costs: {costs}, Machines: {machines}")
    print(f"Minimum cost: {result}")
    print()

    # Test 2: Slope Optimization
    print("Test 2: Slope Optimization")
    n = 6
    a = [1, 2, 3, 4, 5, 6]
    b = [6, 5, 4, 3, 2, 1]
    result = solve_slope_optimization(n, a, b)
    print(f"n: {n}, a: {a}, b: {b}")
    print(f"Minimum dp value: {result}")
    print()

    # Test 3: Li Chao Tree
    print("Test 3: Li Chao Tree (dynamic line insertion)")
    li_chao = LiChaoTree(0, 100)

    # Add lines in arbitrary order
    lines = [
        Line(2, 5),    # y = 2x + 5
        Line(-1, 20),  # y = -x + 20
        Line(0.5, 10), # y = 0.5x + 10
    ]

    for line in lines:
        li_chao.add_line(line)
        print(f"Added line: y = {line.m}x + {line.c}")

    # Query at different points
    query_points = [0, 5, 10, 15, 20]
    print("\nQuery results:")
    for x in query_points:
        min_val = li_chao.query(x)
        print(f"  x = {x}: min value = {min_val:.2f}")
    print()

    # Test 4: ConvexHullTrickMax
    print("Test 4: Maximum Query with CHT")
    cht_max = ConvexHullTrickMax()

    # Add lines (must be in decreasing slope order for simple CHT)
    max_lines = [
        Line(5, 10),   # y = 5x + 10
        Line(3, 15),   # y = 3x + 15
        Line(1, 20),   # y = x + 20
    ]

    for line in max_lines:
        cht_max.add_line(line)
        print(f"Added line: y = {line.m}x + {line.c}")

    print("\nMaximum query results:")
    for x in [0, 2, 5, 10]:
        max_val = cht_max.query(x)
        print(f"  x = {x}: max value = {max_val:.2f}")
    print()

    print("=== Complexity Analysis ===")
    print("CHT with monotonic queries: O(n) total")
    print("CHT with binary search: O(n log n)")
    print("Li Chao Tree: O(n log n)")
    print("\nOptimization: Reduces O(n^2) DP to O(n log n)")
