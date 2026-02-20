"""
실전 문제풀이 (Practice Problems)
Comprehensive Practice Problems

다양한 알고리즘을 조합하여 해결하는 종합 문제들입니다.
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop
from functools import lru_cache
from bisect import bisect_left, bisect_right
import sys


# =============================================================================
# 1. 투 포인터 + 이분탐색: 부분 배열 합
# =============================================================================

def subarray_sum_k(arr: List[int], k: int) -> int:
    """
    합이 k인 부분 배열의 개수
    시간복잡도: O(n)
    """
    count = 0
    prefix_sum = 0
    sum_count = defaultdict(int)
    sum_count[0] = 1

    for num in arr:
        prefix_sum += num
        # prefix_sum - k가 이전에 나왔다면
        count += sum_count[prefix_sum - k]
        sum_count[prefix_sum] += 1

    return count


def longest_subarray_sum_at_most_k(arr: List[int], k: int) -> int:
    """
    합이 k 이하인 가장 긴 부분 배열
    (양수만 있는 경우)
    시간복잡도: O(n)
    """
    n = len(arr)
    max_len = 0
    current_sum = 0
    left = 0

    for right in range(n):
        current_sum += arr[right]

        while current_sum > k and left <= right:
            current_sum -= arr[left]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len


# =============================================================================
# 2. BFS + DP: 최단 경로 + 비용
# =============================================================================

def shortest_path_with_fuel(n: int, edges: List[Tuple[int, int, int]],
                            start: int, end: int, fuel: int) -> int:
    """
    연료 제한이 있는 최단 경로
    edges: (u, v, fuel_cost)
    반환: 최소 거리 (불가능하면 -1)
    시간복잡도: O(V * F + E)
    """
    graph = defaultdict(list)
    for u, v, cost in edges:
        graph[u].append((v, cost))
        graph[v].append((u, cost))

    # BFS with state: (node, remaining_fuel)
    INF = float('inf')
    dist = [[INF] * (fuel + 1) for _ in range(n)]
    dist[start][fuel] = 0

    queue = deque([(start, fuel, 0)])  # (node, remaining_fuel, distance)

    while queue:
        node, remaining, d = queue.popleft()

        if node == end:
            return d

        if d > dist[node][remaining]:
            continue

        for neighbor, cost in graph[node]:
            if remaining >= cost:
                new_fuel = remaining - cost
                if d + 1 < dist[neighbor][new_fuel]:
                    dist[neighbor][new_fuel] = d + 1
                    queue.append((neighbor, new_fuel, d + 1))

    return -1


# =============================================================================
# 3. 그리디 + 우선순위 큐: 작업 스케줄링
# =============================================================================

def max_profit_scheduling(jobs: List[Tuple[int, int, int]]) -> int:
    """
    작업 스케줄링: 겹치지 않게 선택하여 최대 이익
    jobs: (시작, 종료, 이익)
    시간복잡도: O(n log n)
    """
    n = len(jobs)
    if n == 0:
        return 0

    # 종료 시간으로 정렬
    jobs = sorted(jobs, key=lambda x: x[1])
    end_times = [job[1] for job in jobs]

    dp = [0] * (n + 1)

    for i in range(n):
        start, end, profit = jobs[i]

        # 현재 작업 시작 전에 끝나는 마지막 작업
        j = bisect_right(end_times, start) - 1

        # 선택 vs 비선택
        dp[i + 1] = max(dp[i], (dp[j + 1] if j >= 0 else 0) + profit)

    return dp[n]


# =============================================================================
# 4. 문자열 + DP: 편집 거리 응용
# =============================================================================

def min_operations_to_palindrome(s: str) -> int:
    """
    문자열을 팰린드롬으로 만드는 최소 삽입/삭제
    = 길이 - LCS(s, reverse(s))
    """
    n = len(s)
    t = s[::-1]

    # LCS
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return n - dp[n][n]


def longest_palindromic_subsequence(s: str) -> int:
    """
    가장 긴 팰린드롬 부분 수열
    = LCS(s, reverse(s))
    """
    n = len(s)
    t = s[::-1]

    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][n]


# =============================================================================
# 5. 그래프 + Union-Find: 연결 요소
# =============================================================================

class DSU:
    """Disjoint Set Union"""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def min_cost_to_connect_all(n: int, edges: List[Tuple[int, int, int]]) -> int:
    """
    모든 노드를 연결하는 최소 비용 (MST)
    edges: (u, v, cost)
    """
    edges = sorted(edges, key=lambda x: x[2])
    dsu = DSU(n)
    total_cost = 0
    edges_used = 0

    for u, v, cost in edges:
        if dsu.union(u, v):
            total_cost += cost
            edges_used += 1
            if edges_used == n - 1:
                break

    return total_cost if edges_used == n - 1 else -1


def count_islands(grid: List[List[int]]) -> int:
    """
    섬의 개수 (Union-Find 버전)
    grid[i][j] = 1: 땅, 0: 물
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    dsu = DSU(rows * cols)
    count = 0

    def idx(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                count += 1

    directions = [(0, 1), (1, 0)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    if dsu.union(idx(r, c), idx(nr, nc)):
                        count -= 1

    return count


# =============================================================================
# 6. 세그먼트 트리 + 좌표 압축: 구간 쿼리
# =============================================================================

class SegmentTree:
    """구간 합 세그먼트 트리"""
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (4 * n)

    def update(self, node: int, start: int, end: int, idx: int, delta: int):
        if idx < start or idx > end:
            return
        if start == end:
            self.tree[node] += delta
            return
        mid = (start + end) // 2
        self.update(2 * node, start, mid, idx, delta)
        self.update(2 * node + 1, mid + 1, end, idx, delta)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return (self.query(2 * node, start, mid, l, r) +
                self.query(2 * node + 1, mid + 1, end, l, r))


def count_smaller_after_self(nums: List[int]) -> List[int]:
    """
    각 원소 뒤에 있는 더 작은 원소의 개수
    세그먼트 트리 + 좌표 압축
    시간복잡도: O(n log n)
    """
    # 좌표 압축
    sorted_nums = sorted(set(nums))
    rank = {v: i for i, v in enumerate(sorted_nums)}
    n = len(sorted_nums)

    result = []
    st = SegmentTree(n)

    # 역순으로 처리
    for num in reversed(nums):
        r = rank[num]
        # r보다 작은 인덱스의 합 (= 더 작은 원소의 개수)
        count = st.query(1, 0, n - 1, 0, r - 1) if r > 0 else 0
        result.append(count)
        st.update(1, 0, n - 1, r, 1)

    return result[::-1]


# =============================================================================
# 7. 백트래킹: 조합 최적화
# =============================================================================

def solve_sudoku(board: List[List[str]]) -> bool:
    """
    스도쿠 해결
    board: 9x9, '1'-'9' 또는 '.'
    """

    def is_valid(row: int, col: int, num: str) -> bool:
        # 행 검사
        if num in board[row]:
            return False
        # 열 검사
        for r in range(9):
            if board[r][col] == num:
                return False
        # 3x3 박스 검사
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False
        return True

    def solve() -> bool:
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num
                            if solve():
                                return True
                            board[row][col] = '.'
                    return False
        return True

    return solve()


# =============================================================================
# 8. 비트마스크 DP: 외판원 문제 (TSP)
# =============================================================================

def tsp(dist: List[List[int]]) -> int:
    """
    외판원 문제 (TSP)
    모든 도시를 방문하고 시작점으로 돌아오는 최소 비용
    시간복잡도: O(n² * 2^n)
    """
    n = len(dist)
    INF = float('inf')

    @lru_cache(maxsize=None)
    def dp(mask: int, last: int) -> int:
        if mask == (1 << n) - 1:  # 모든 도시 방문
            return dist[last][0] if dist[last][0] > 0 else INF

        result = INF
        for next_city in range(n):
            if mask & (1 << next_city):
                continue
            if dist[last][next_city] == 0:
                continue
            result = min(result, dist[last][next_city] + dp(mask | (1 << next_city), next_city))

        return result

    return dp(1, 0)  # 0번 도시에서 시작


# =============================================================================
# 9. 이분탐색 + 그리디: 파라메트릭 서치
# =============================================================================

def min_max_distance(houses: List[int], k: int) -> int:
    """
    k개의 우체통을 설치하여 가장 먼 집까지의 거리 최소화
    파라메트릭 서치: "최대 거리가 d 이하가 되게 할 수 있는가?"
    시간복잡도: O(n log D)
    """
    houses = sorted(houses)
    n = len(houses)

    def can_cover(max_dist: int) -> bool:
        """최대 거리 max_dist로 모든 집을 커버 가능한지"""
        count = 1
        last_post = houses[0]

        for house in houses:
            if house - last_post > 2 * max_dist:
                count += 1
                last_post = house

        return count <= k

    lo, hi = 0, houses[-1] - houses[0]

    while lo < hi:
        mid = (lo + hi) // 2
        if can_cover(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo


def min_pages_per_student(pages: List[int], students: int) -> int:
    """
    책을 학생들에게 나눠줄 때 최대 페이지 수 최소화
    (연속된 책만 가능)
    """
    def is_feasible(max_pages: int) -> bool:
        count = 1
        current = 0
        for p in pages:
            if p > max_pages:
                return False
            if current + p > max_pages:
                count += 1
                current = p
            else:
                current += p
        return count <= students

    lo, hi = max(pages), sum(pages)

    while lo < hi:
        mid = (lo + hi) // 2
        if is_feasible(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo


# =============================================================================
# 10. 종합: LIS + 이분탐색
# =============================================================================

def longest_increasing_subsequence(nums: List[int]) -> Tuple[int, List[int]]:
    """
    가장 긴 증가하는 부분 수열 (길이 + 실제 수열)
    시간복잡도: O(n log n)
    """
    n = len(nums)
    if n == 0:
        return 0, []

    # tails[i] = 길이 i+1 LIS의 마지막 원소 중 최솟값
    tails = []
    # 각 위치에서의 LIS 길이
    lengths = [0] * n
    # 이전 원소 추적
    prev = [-1] * n
    # tails의 인덱스가 어떤 원래 인덱스에서 왔는지
    tail_idx = []

    for i, num in enumerate(nums):
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
            tail_idx.append(i)
        else:
            tails[pos] = num
            tail_idx[pos] = i

        lengths[i] = pos + 1
        if pos > 0:
            # 이전 원소: lengths[j] == pos인 마지막 j
            for j in range(i - 1, -1, -1):
                if lengths[j] == pos and nums[j] < num:
                    prev[i] = j
                    break

    # LIS 복원
    max_len = max(lengths)
    result = []
    idx = lengths.index(max_len)

    # 최적 시작점 찾기
    for i in range(n - 1, -1, -1):
        if lengths[i] == max_len:
            idx = i
            break

    while idx != -1:
        result.append(nums[idx])
        # 이전 원소 찾기
        target_len = lengths[idx] - 1
        if target_len == 0:
            break
        for j in range(idx - 1, -1, -1):
            if lengths[j] == target_len and nums[j] < nums[idx]:
                idx = j
                break
        else:
            break

    return max_len, result[::-1]


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("실전 문제풀이 (Practice Problems) 예제")
    print("=" * 60)

    # 1. 부분 배열 합
    print("\n[1] 부분 배열 합")
    arr = [1, 2, 3, -3, 1, 2]
    k = 3
    count = subarray_sum_k(arr, k)
    print(f"    배열: {arr}, k={k}")
    print(f"    합이 {k}인 부분 배열 개수: {count}")

    # 2. 작업 스케줄링
    print("\n[2] 작업 스케줄링 (최대 이익)")
    jobs = [(1, 3, 50), (2, 5, 20), (4, 6, 70), (6, 7, 60)]
    profit = max_profit_scheduling(jobs)
    print(f"    작업 (시작, 종료, 이익): {jobs}")
    print(f"    최대 이익: {profit}")

    # 3. 팰린드롬
    print("\n[3] 문자열 → 팰린드롬")
    s = "aebcbda"
    ops = min_operations_to_palindrome(s)
    lps = longest_palindromic_subsequence(s)
    print(f"    문자열: '{s}'")
    print(f"    팰린드롬 만드는 최소 연산: {ops}")
    print(f"    가장 긴 팰린드롬 부분수열 길이: {lps}")

    # 4. MST
    print("\n[4] 최소 신장 트리")
    edges = [(0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2), (2, 3, 4)]
    cost = min_cost_to_connect_all(4, edges)
    print(f"    간선: {edges}")
    print(f"    MST 비용: {cost}")

    # 5. 섬의 개수
    print("\n[5] 섬의 개수")
    grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ]
    islands = count_islands([row[:] for row in grid])
    print(f"    그리드:")
    for row in grid:
        print(f"      {row}")
    print(f"    섬의 개수: {islands}")

    # 6. 뒤에 있는 더 작은 원소
    print("\n[6] 각 원소 뒤의 더 작은 원소 개수")
    nums = [5, 2, 6, 1]
    result = count_smaller_after_self(nums)
    print(f"    배열: {nums}")
    print(f"    결과: {result}")

    # 7. TSP
    print("\n[7] 외판원 문제 (TSP)")
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    min_cost = tsp(dist)
    print(f"    거리 행렬:")
    for row in dist:
        print(f"      {row}")
    print(f"    최소 비용: {min_cost}")

    # 8. 파라메트릭 서치
    print("\n[8] 책 배분 문제")
    pages = [12, 34, 67, 90]
    students = 2
    min_pages = min_pages_per_student(pages, students)
    print(f"    페이지: {pages}, 학생: {students}")
    print(f"    최대 페이지 최소화: {min_pages}")

    # 9. LIS
    print("\n[9] 최장 증가 부분 수열 (LIS)")
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    length, seq = longest_increasing_subsequence(nums)
    print(f"    배열: {nums}")
    print(f"    LIS 길이: {length}")
    print(f"    LIS 예시: {seq}")

    # 10. 알고리즘 선택 가이드
    print("\n[10] 알고리즘 선택 가이드")
    print("    | 문제 유형              | 핵심 알고리즘              |")
    print("    |------------------------|----------------------------|")
    print("    | 부분 배열 합           | 해시맵 + 프리픽스 합       |")
    print("    | 구간 스케줄링          | 정렬 + 그리디/DP           |")
    print("    | 문자열 변환            | DP (LCS, 편집거리)         |")
    print("    | 그래프 연결            | Union-Find, BFS/DFS        |")
    print("    | 구간 쿼리              | 세그먼트 트리, BIT         |")
    print("    | 조합 최적화            | 백트래킹 + 가지치기        |")
    print("    | 완전 탐색 (작은 n)     | 비트마스크 DP              |")
    print("    | 최솟값 최대화          | 파라메트릭 서치            |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
