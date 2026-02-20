"""
Divide and Conquer (D&C) Optimization for DP

D&C Optimization is used to optimize 2D DP from O(n^2 * k) to O(n * k * log n).
It's applicable when the DP satisfies the quadrangle inequality (monotonicity condition).

Condition: For dp[i][j] = min(dp[i-1][k] + cost(k, j)) for k < j,
if opt[i][j] <= opt[i][j+1], where opt[i][j] is the optimal k for dp[i][j].

This means the optimal split point is monotonic: as j increases, opt doesn't decrease.

Common applications:
- Splitting array into k groups to minimize cost
- Optimal Binary Search Tree
- Matrix Chain Multiplication variants

Time Complexity: O(k * n * log n) instead of O(k * n^2)
Space Complexity: O(k * n)
"""

from typing import List, Callable, Tuple
import sys


def divide_and_conquer_dp(
    n: int,
    k: int,
    cost: Callable[[int, int], float],
    prev_dp: List[float]
) -> List[float]:
    """
    Compute DP for one layer using divide and conquer optimization.

    Args:
        n: Number of elements
        k: Current layer/group number
        cost: Cost function cost(i, j) for range [i, j]
        prev_dp: DP values from previous layer

    Returns:
        List of DP values for current layer
    """
    dp = [float('inf')] * (n + 1)

    def solve(l: int, r: int, opt_l: int, opt_r: int) -> None:
        """
        Solve for positions [l, r] knowing optimal split is in [opt_l, opt_r].

        Args:
            l, r: Range of positions to compute
            opt_l, opt_r: Range where optimal split point must lie
        """
        if l > r:
            return

        mid = (l + r) // 2
        best_cost = float('inf')
        best_split = opt_l

        # Find optimal split point for position mid
        for split in range(opt_l, min(opt_r, mid) + 1):
            current_cost = prev_dp[split] + cost(split, mid)
            if current_cost < best_cost:
                best_cost = current_cost
                best_split = split

        dp[mid] = best_cost

        # Recursively solve left and right
        solve(l, mid - 1, opt_l, best_split)
        solve(mid + 1, r, best_split, opt_r)

    solve(1, n, 0, n)
    return dp


def split_array_min_cost(arr: List[int], k: int) -> Tuple[float, List[List[int]]]:
    """
    Split array into k groups to minimize sum of (group_sum)^2.

    Problem: Given array arr, split it into k non-empty contiguous groups
    such that sum of (sum of each group)^2 is minimized.

    DP: dp[i][j] = minimum cost to split first j elements into i groups
    Recurrence: dp[i][j] = min(dp[i-1][m] + cost(m, j)) for m in [i-1, j-1]
    where cost(m, j) = (sum(arr[m+1:j+1]))^2

    This satisfies quadrangle inequality, so D&C optimization applies.
    """
    n = len(arr)

    # Precompute prefix sums for O(1) range sum queries
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    def range_sum(i: int, j: int) -> int:
        """Sum of arr[i:j] (0-indexed, exclusive end)"""
        return prefix[j] - prefix[i]

    def cost(i: int, j: int) -> float:
        """Cost of grouping elements from index i+1 to j"""
        if i >= j:
            return float('inf')
        s = range_sum(i, j)
        return s * s

    # Initialize DP table
    dp = [[float('inf')] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0

    # Track splits for reconstruction
    split = [[0] * (n + 1) for _ in range(k + 1)]

    # Fill DP table using D&C optimization
    for i in range(1, k + 1):
        # Use D&C to compute dp[i][j] for all j
        prev_layer = dp[i - 1]

        def solve(l: int, r: int, opt_l: int, opt_r: int) -> None:
            if l > r:
                return

            mid = (l + r) // 2
            best_cost = float('inf')
            best_split = opt_l

            for s in range(opt_l, min(opt_r, mid) + 1):
                current_cost = prev_layer[s] + cost(s, mid)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_split = s

            dp[i][mid] = best_cost
            split[i][mid] = best_split

            solve(l, mid - 1, opt_l, best_split)
            solve(mid + 1, r, best_split, opt_r)

        solve(i, n, i - 1, n)

    # Reconstruct solution
    groups = []
    pos = n
    for i in range(k, 0, -1):
        start = split[i][pos]
        groups.append(arr[start:pos])
        pos = start
    groups.reverse()

    return dp[k][n], groups


def optimal_merge_cost(arr: List[int], k: int) -> Tuple[float, List[List[int]]]:
    """
    Merge elements with minimum cost, creating k groups.

    Problem: Given costs for each element, merge them into k groups.
    Cost of a group is the sum of all elements in it.
    Total cost is sum of all group costs.

    DP: dp[i][j] = minimum cost to merge first j elements into i groups
    """
    n = len(arr)

    # Prefix sum
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    def cost(i: int, j: int) -> float:
        """Cost of merging elements from i+1 to j into one group"""
        if i >= j:
            return 0
        return prefix[j] - prefix[i]

    # Initialize DP
    dp = [[float('inf')] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0

    split = [[0] * (n + 1) for _ in range(k + 1)]

    # Fill DP using D&C optimization
    for i in range(1, k + 1):
        prev_layer = dp[i - 1]

        def solve(l: int, r: int, opt_l: int, opt_r: int) -> None:
            if l > r:
                return

            mid = (l + r) // 2
            best_cost = float('inf')
            best_split = opt_l

            for s in range(opt_l, min(opt_r, mid) + 1):
                current_cost = prev_layer[s] + cost(s, mid)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_split = s

            dp[i][mid] = best_cost
            split[i][mid] = best_split

            solve(l, mid - 1, opt_l, best_split)
            solve(mid + 1, r, best_split, opt_r)

        solve(i, n, i - 1, n)

    # Reconstruct
    groups = []
    pos = n
    for i in range(k, 0, -1):
        start = split[i][pos]
        groups.append(arr[start:pos])
        pos = start
    groups.reverse()

    return dp[k][n], groups


def warehouse_allocation(n: int, k: int, positions: List[int], demands: List[int]) -> float:
    """
    Warehouse Allocation Problem with D&C Optimization.

    Problem: Place k warehouses to serve n locations.
    Cost of serving location i from warehouse at j is |positions[i] - positions[j]| * demands[i].

    DP: dp[w][i] = minimum cost to serve first i locations using w warehouses
    The last warehouse serves some contiguous range [j+1, i].

    This satisfies the monotonicity condition for D&C optimization.
    """
    # Precompute costs
    cost_cache = {}

    def compute_cost(start: int, end: int, warehouse_pos: int) -> float:
        """Cost to serve locations [start, end) from warehouse at warehouse_pos"""
        total = 0
        for i in range(start, end):
            total += abs(positions[i] - warehouse_pos) * demands[i]
        return total

    def cost(i: int, j: int) -> float:
        """
        Minimum cost to serve locations [i+1, j] with one warehouse.
        Optimal warehouse position is the weighted median.
        """
        if i >= j:
            return 0

        if (i, j) in cost_cache:
            return cost_cache[(i, j)]

        # For simplicity, try all positions in range as warehouse location
        min_cost = float('inf')
        for w in range(i, j):
            c = compute_cost(i, j, positions[w])
            min_cost = min(min_cost, c)

        cost_cache[(i, j)] = min_cost
        return min_cost

    # DP with D&C optimization
    dp = [[float('inf')] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0

    for w in range(1, k + 1):
        prev_layer = dp[w - 1]

        def solve(l: int, r: int, opt_l: int, opt_r: int) -> None:
            if l > r:
                return

            mid = (l + r) // 2
            best_cost = float('inf')
            best_split = opt_l

            for s in range(opt_l, min(opt_r, mid) + 1):
                current_cost = prev_layer[s] + cost(s, mid)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_split = s

            dp[w][mid] = best_cost

            solve(l, mid - 1, opt_l, best_split)
            solve(mid + 1, r, best_split, opt_r)

        solve(w, n, w - 1, n)

    return dp[k][n]


if __name__ == "__main__":
    print("=== Divide and Conquer DP Optimization Examples ===\n")

    # Test 1: Split Array Min Cost
    print("Test 1: Split Array to Minimize Sum of Squared Group Sums")
    arr1 = [1, 2, 3, 4, 5, 6]
    k1 = 3
    min_cost1, groups1 = split_array_min_cost(arr1, k1)
    print(f"Array: {arr1}")
    print(f"Number of groups: {k1}")
    print(f"Minimum cost: {min_cost1}")
    print(f"Groups: {groups1}")
    print(f"Verification: {sum(sum(g)**2 for g in groups1)}")
    print()

    # Test 2: Optimal Merge Cost
    print("Test 2: Optimal Merge Cost")
    arr2 = [3, 5, 8, 2, 4, 7, 1]
    k2 = 4
    min_cost2, groups2 = optimal_merge_cost(arr2, k2)
    print(f"Array: {arr2}")
    print(f"Number of groups: {k2}")
    print(f"Minimum total cost: {min_cost2}")
    print(f"Groups: {groups2}")
    print(f"Group costs: {[sum(g) for g in groups2]}")
    print()

    # Test 3: Warehouse Allocation
    print("Test 3: Warehouse Allocation Problem")
    n3 = 6
    k3 = 2
    positions3 = [1, 3, 5, 7, 9, 11]
    demands3 = [10, 5, 8, 12, 6, 9]
    min_cost3 = warehouse_allocation(n3, k3, positions3, demands3)
    print(f"Locations: {n3}")
    print(f"Warehouses: {k3}")
    print(f"Positions: {positions3}")
    print(f"Demands: {demands3}")
    print(f"Minimum cost: {min_cost3}")
    print()

    # Test 4: Large Array Performance Test
    print("Test 4: Performance on Larger Array")
    import time

    arr4 = list(range(1, 101))
    k4 = 10
    start_time = time.time()
    min_cost4, groups4 = split_array_min_cost(arr4, k4)
    end_time = time.time()

    print(f"Array size: {len(arr4)}")
    print(f"Number of groups: {k4}")
    print(f"Minimum cost: {min_cost4}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Group sizes: {[len(g) for g in groups4]}")
    print()

    # Test 5: Edge Cases
    print("Test 5: Edge Cases")

    # Single group
    arr5a = [1, 2, 3, 4, 5]
    k5a = 1
    cost5a, groups5a = split_array_min_cost(arr5a, k5a)
    print(f"Single group: {arr5a}, k={k5a}")
    print(f"Cost: {cost5a}, Groups: {groups5a}")

    # Each element in its own group
    arr5b = [1, 2, 3]
    k5b = 3
    cost5b, groups5b = split_array_min_cost(arr5b, k5b)
    print(f"Each in own group: {arr5b}, k={k5b}")
    print(f"Cost: {cost5b}, Groups: {groups5b}")
    print()

    print("=== Complexity Analysis ===")
    print("Without D&C Optimization: O(k * n^2)")
    print("With D&C Optimization: O(k * n * log n)")
    print()
    print("Speedup factor for n=1000, k=10:")
    print(f"  Without: ~{10 * 1000 * 1000:,} operations")
    print(f"  With: ~{10 * 1000 * 10:,} operations")
    print(f"  Speedup: ~{100}x faster")
