"""
고급 DP 최적화 (Advanced DP Optimization)
Advanced Dynamic Programming Optimization Techniques

DP의 시간복잡도를 개선하는 최적화 기법들입니다.
"""

from typing import List, Tuple, Callable
from collections import deque
from math import inf


# =============================================================================
# 1. Convex Hull Trick (CHT)
# =============================================================================

class ConvexHullTrick:
    """
    볼록 껍질 트릭
    최솟값 쿼리: min(a[i] * x + b[i]) for all i
    조건: a[i]가 단조 감소 (또는 증가)

    시간복잡도: 삽입 O(1) 평균, 쿼리 O(log n) 또는 O(1)
    """

    def __init__(self):
        self.lines = deque()  # (기울기, y절편)

    def is_bad(self, l1: Tuple[int, int], l2: Tuple[int, int], l3: Tuple[int, int]) -> bool:
        """l2가 불필요한지 확인 (l1과 l3 사이에서)"""
        # 교점 비교: (l1, l2) 교점 >= (l2, l3) 교점이면 l2 불필요
        # (b2 - b1) / (a1 - a2) >= (b3 - b2) / (a2 - a3)
        # (b2 - b1) * (a2 - a3) >= (b3 - b2) * (a1 - a2)
        a1, b1 = l1
        a2, b2 = l2
        a3, b3 = l3
        return (b2 - b1) * (a2 - a3) >= (b3 - b2) * (a1 - a2)

    def add_line(self, a: int, b: int):
        """
        직선 y = ax + b 추가
        a는 단조 감소해야 함
        """
        line = (a, b)

        while len(self.lines) >= 2 and self.is_bad(self.lines[-2], self.lines[-1], line):
            self.lines.pop()

        self.lines.append(line)

    def query_min(self, x: int) -> int:
        """
        x에서의 최솟값 쿼리
        x가 단조 증가할 때 O(1)
        """
        while len(self.lines) >= 2:
            a1, b1 = self.lines[0]
            a2, b2 = self.lines[1]
            if a1 * x + b1 >= a2 * x + b2:
                self.lines.popleft()
            else:
                break

        a, b = self.lines[0]
        return a * x + b


class LiChaoTree:
    """
    Li Chao Tree (세그먼트 트리 기반 CHT)
    임의의 직선 추가와 임의의 x에서 쿼리 지원

    시간복잡도: 삽입 O(log C), 쿼리 O(log C)
    C = 좌표 범위
    """

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi
        self.tree = {}  # 노드별 우세 직선 저장

    def _eval(self, line: Tuple[int, int], x: int) -> int:
        """직선 값 계산"""
        if line is None:
            return inf
        a, b = line
        return a * x + b

    def add_line(self, a: int, b: int):
        """직선 추가"""
        self._add_line_impl((a, b), self.lo, self.hi, 1)

    def _add_line_impl(self, new_line: Tuple[int, int], lo: int, hi: int, node: int):
        if lo > hi:
            return

        mid = (lo + hi) // 2
        cur_line = self.tree.get(node)

        # 중간점에서 비교
        new_better_at_mid = self._eval(new_line, mid) < self._eval(cur_line, mid)

        if cur_line is None or new_better_at_mid:
            self.tree[node], new_line = new_line, cur_line

        if lo == hi or new_line is None:
            return

        # 왼쪽/오른쪽 자식으로 전파
        new_better_at_lo = self._eval(new_line, lo) < self._eval(self.tree.get(node), lo)

        if new_better_at_lo:
            self._add_line_impl(new_line, lo, mid - 1, 2 * node)
        else:
            self._add_line_impl(new_line, mid + 1, hi, 2 * node + 1)

    def query(self, x: int) -> int:
        """x에서의 최솟값"""
        return self._query_impl(x, self.lo, self.hi, 1)

    def _query_impl(self, x: int, lo: int, hi: int, node: int) -> int:
        if lo > hi:
            return inf

        result = self._eval(self.tree.get(node), x)

        if lo == hi:
            return result

        mid = (lo + hi) // 2
        if x <= mid:
            return min(result, self._query_impl(x, lo, mid - 1, 2 * node))
        else:
            return min(result, self._query_impl(x, mid + 1, hi, 2 * node + 1))


# =============================================================================
# 2. Divide and Conquer Optimization
# =============================================================================

def dc_optimization(n: int, m: int, cost: Callable[[int, int], int]) -> List[List[int]]:
    """
    분할 정복 최적화
    조건: opt[k][i] <= opt[k][i+1] (단조성)
    점화식: dp[k][j] = min(dp[k-1][i] + cost(i, j)) for i < j

    시간복잡도: O(k * n log n) (일반 O(k * n^2)에서 개선)

    n: 원소 개수
    m: 분할 그룹 수
    cost(i, j): i+1 ~ j 구간의 비용
    """
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    def compute(k: int, lo: int, hi: int, opt_lo: int, opt_hi: int):
        """dp[k][lo:hi+1] 계산, 최적의 분할점은 opt_lo ~ opt_hi 범위"""
        if lo > hi:
            return

        mid = (lo + hi) // 2
        best_cost = INF
        best_opt = opt_lo

        for i in range(opt_lo, min(opt_hi, mid) + 1):
            curr_cost = dp[k - 1][i] + cost(i, mid)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_opt = i

        dp[k][mid] = best_cost

        # 분할 정복
        compute(k, lo, mid - 1, opt_lo, best_opt)
        compute(k, mid + 1, hi, best_opt, opt_hi)

    for k in range(1, m + 1):
        compute(k, k, n, k - 1, n - 1)

    return dp


def dc_optimization_example():
    """
    예제: 배열을 k개 그룹으로 나누기
    각 그룹의 비용 = 구간 내 원소 차이의 제곱 합
    """
    arr = [1, 5, 2, 8, 3, 7, 4, 6]
    n = len(arr)
    k = 3

    # 전처리: prefix sum for cost calculation
    prefix = [0] * (n + 1)
    prefix_sq = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
        prefix_sq[i + 1] = prefix_sq[i] + arr[i] * arr[i]

    def cost(l: int, r: int) -> int:
        """구간 [l+1, r]의 분산 (제곱합 - 평균*합)"""
        if l >= r:
            return 0
        length = r - l
        s = prefix[r] - prefix[l]
        sq = prefix_sq[r] - prefix_sq[l]
        # 분산 = E[X^2] - E[X]^2, 여기서는 제곱합 - 합^2/n
        return sq * length - s * s

    dp = dc_optimization(n, k, cost)
    return dp[k][n]


# =============================================================================
# 3. Knuth Optimization
# =============================================================================

def knuth_optimization(n: int, cost: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Knuth 최적화
    조건: cost가 사각 부등식 만족 (Quadrangle Inequality)
          cost[a][c] + cost[b][d] <= cost[a][d] + cost[b][c] (a <= b <= c <= d)
    점화식: dp[i][j] = min(dp[i][k] + dp[k][j]) + cost[i][j] for i < k < j

    시간복잡도: O(n^2) (일반 O(n^3)에서 개선)

    예: 최적 이진 탐색 트리, 행렬 체인 곱셈
    """
    INF = float('inf')
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    # 기저: 길이 1
    for i in range(n):
        opt[i][i] = i

    # 길이 2 이상
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            # opt[i][j-1] <= opt[i][j] <= opt[i+1][j]
            lo = opt[i][j - 1] if j > 0 else i
            hi = opt[i + 1][j] if i + 1 < n else j

            for k in range(lo, min(hi, j) + 1):
                curr = dp[i][k] + dp[k + 1][j] + cost[i][j]
                if curr < dp[i][j]:
                    dp[i][j] = curr
                    opt[i][j] = k

    return dp, opt


def optimal_bst(keys: List[int], freq: List[int]) -> int:
    """
    최적 이진 탐색 트리
    freq[i] = keys[i]의 접근 빈도
    """
    n = len(keys)

    # cost[i][j] = freq[i] + ... + freq[j]
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += freq[j]
            cost[i][j] = total

    dp, opt = knuth_optimization(n, cost)
    return dp[0][n - 1]


# =============================================================================
# 4. 1D/1D DP 최적화 (Monotone Queue)
# =============================================================================

def sliding_window_max(arr: List[int], k: int) -> List[int]:
    """
    슬라이딩 윈도우 최댓값
    시간복잡도: O(n)
    """
    result = []
    dq = deque()  # (인덱스, 값)

    for i, val in enumerate(arr):
        # 윈도우 범위 밖 제거
        while dq and dq[0][0] <= i - k:
            dq.popleft()

        # 현재 값보다 작은 원소 제거
        while dq and dq[-1][1] <= val:
            dq.pop()

        dq.append((i, val))

        if i >= k - 1:
            result.append(dq[0][1])

    return result


def dp_with_monotone_queue(arr: List[int], k: int) -> List[int]:
    """
    DP with Monotone Queue
    dp[i] = max(dp[j] + arr[i]) for i-k <= j < i
    시간복잡도: O(n)
    """
    n = len(arr)
    dp = [0] * n
    dq = deque()  # (인덱스, dp값)

    for i in range(n):
        # 범위 밖 제거
        while dq and dq[0][0] < i - k:
            dq.popleft()

        # 최댓값으로 dp 계산
        if dq:
            dp[i] = dq[0][1] + arr[i]
        else:
            dp[i] = arr[i]

        # 현재 dp값 삽입
        while dq and dq[-1][1] <= dp[i]:
            dq.pop()
        dq.append((i, dp[i]))

    return dp


# =============================================================================
# 5. Slope Trick
# =============================================================================

class SlopeTrick:
    """
    Slope Trick
    볼록 함수의 효율적인 관리
    절대값 함수의 합 최적화에 유용
    """

    def __init__(self):
        import heapq
        self.left = []   # 최대 힙 (음수로 저장)
        self.right = []  # 최소 힙
        self.min_f = 0
        self.add_l = 0   # left에 더할 값
        self.add_r = 0   # right에 더할 값

    def add_abs(self, a: int):
        """
        f(x) += |x - a|
        """
        import heapq

        # a 위치에서 기울기 변화: 왼쪽 +1, 오른쪽 -1
        l = -self.left[0] + self.add_l if self.left else -inf
        r = self.right[0] + self.add_r if self.right else inf

        if a <= l:
            # a가 왼쪽에 위치
            self.min_f += l - a
            heapq.heappush(self.left, -(a - self.add_l))
            # 왼쪽 최댓값을 오른쪽으로
            val = -heapq.heappop(self.left) + self.add_l
            heapq.heappush(self.right, val - self.add_r)
        elif a >= r:
            # a가 오른쪽에 위치
            self.min_f += a - r
            heapq.heappush(self.right, a - self.add_r)
            # 오른쪽 최솟값을 왼쪽으로
            val = heapq.heappop(self.right) + self.add_r
            heapq.heappush(self.left, -(val - self.add_l))
        else:
            # a가 평평한 구간에 위치
            heapq.heappush(self.left, -(a - self.add_l))
            heapq.heappush(self.right, a - self.add_r)

    def shift(self, a: int, b: int):
        """
        f(x) → f(x-a) (왼쪽 이동), f(x) → f(x-b) (오른쪽 이동)
        평평한 구간 확장
        """
        self.add_l += a
        self.add_r += b

    def get_min(self) -> int:
        """최솟값 반환"""
        return self.min_f


# =============================================================================
# 6. Alien Trick (WQS Binary Search)
# =============================================================================

def alien_trick_example(arr: List[int], k: int) -> int:
    """
    Alien Trick (WQS Binary Search / Lagrange Relaxation)
    정확히 k개의 원소를 선택하는 문제를 이완

    예: 배열에서 정확히 k개 선택, 인접한 것 불가, 합 최대화
    """

    def check(penalty: float) -> Tuple[float, int]:
        """
        penalty를 사용한 이완 문제 해결
        반환: (최적값, 선택한 개수)
        """
        n = len(arr)
        # dp[i][0]: i까지, arr[i] 선택 안함
        # dp[i][1]: i까지, arr[i] 선택함

        dp = [[-inf, -inf] for _ in range(n)]
        cnt = [[0, 0] for _ in range(n)]

        dp[0][0] = 0
        dp[0][1] = arr[0] - penalty
        cnt[0][1] = 1

        for i in range(1, n):
            # 선택 안함
            if dp[i - 1][0] > dp[i - 1][1]:
                dp[i][0] = dp[i - 1][0]
                cnt[i][0] = cnt[i - 1][0]
            else:
                dp[i][0] = dp[i - 1][1]
                cnt[i][0] = cnt[i - 1][1]

            # 선택함 (이전에 선택 안한 상태에서만)
            dp[i][1] = dp[i - 1][0] + arr[i] - penalty
            cnt[i][1] = cnt[i - 1][0] + 1

        if dp[n - 1][0] > dp[n - 1][1]:
            return dp[n - 1][0], cnt[n - 1][0]
        return dp[n - 1][1], cnt[n - 1][1]

    # 이분 탐색
    lo, hi = -10**9, 10**9

    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        _, count = check(mid)
        if count >= k:
            lo = mid
        else:
            hi = mid

    result, _ = check(lo)
    return int(result + lo * k)


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("고급 DP 최적화 (Advanced DP Optimization) 예제")
    print("=" * 60)

    # 1. Convex Hull Trick
    print("\n[1] Convex Hull Trick (CHT)")
    cht = ConvexHullTrick()
    # 직선: y = -3x + 10, y = -2x + 5, y = -1x + 3
    cht.add_line(-3, 10)
    cht.add_line(-2, 5)
    cht.add_line(-1, 3)
    print("    직선들: y=-3x+10, y=-2x+5, y=-x+3")
    for x in [0, 1, 2, 3, 4, 5]:
        print(f"    min at x={x}: {cht.query_min(x)}")

    # 2. Li Chao Tree
    print("\n[2] Li Chao Tree")
    lct = LiChaoTree(-100, 100)
    lct.add_line(2, 5)   # y = 2x + 5
    lct.add_line(-1, 10) # y = -x + 10
    lct.add_line(1, 0)   # y = x
    print("    직선들: y=2x+5, y=-x+10, y=x")
    for x in [-5, 0, 3, 7]:
        print(f"    min at x={x}: {lct.query(x)}")

    # 3. Divide and Conquer Optimization
    print("\n[3] 분할 정복 최적화")
    result = dc_optimization_example()
    print(f"    배열 [1,5,2,8,3,7,4,6]을 3그룹으로 분할")
    print(f"    최소 비용: {result}")

    # 4. Knuth Optimization
    print("\n[4] Knuth 최적화 (최적 BST)")
    keys = [10, 20, 30, 40]
    freq = [4, 2, 6, 3]
    cost = optimal_bst(keys, freq)
    print(f"    키: {keys}")
    print(f"    빈도: {freq}")
    print(f"    최소 탐색 비용: {cost}")

    # 5. Monotone Queue
    print("\n[5] 모노톤 큐 최적화")
    arr = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = sliding_window_max(arr, k)
    print(f"    배열: {arr}")
    print(f"    윈도우 크기: {k}")
    print(f"    슬라이딩 최댓값: {result}")

    # 6. DP with Monotone Queue
    print("\n[6] 모노톤 큐 DP")
    arr = [2, 1, 5, 1, 3, 2]
    k = 2
    dp = dp_with_monotone_queue(arr, k)
    print(f"    배열: {arr}, k={k}")
    print(f"    dp[i] = max(dp[j] + arr[i]) for i-k <= j < i")
    print(f"    DP: {dp}")

    # 7. Slope Trick
    print("\n[7] Slope Trick")
    st = SlopeTrick()
    points = [1, 5, 2, 8]
    for p in points:
        st.add_abs(p)
    print(f"    점들: {points}")
    print(f"    f(x) = sum(|x - p|) 최솟값: {st.get_min()}")

    # 8. 복잡도 비교
    print("\n[8] 최적화 기법 비교")
    print("    | 기법              | 원래 복잡도 | 최적화 후   | 조건                    |")
    print("    |-------------------|-------------|-------------|-------------------------|")
    print("    | CHT               | O(n²)       | O(n)        | 기울기 단조             |")
    print("    | Li Chao Tree      | O(n²)       | O(n log C)  | 없음                    |")
    print("    | D&C Optimization  | O(kn²)      | O(kn log n) | opt 단조성              |")
    print("    | Knuth Optimization| O(n³)       | O(n²)       | 사각 부등식             |")
    print("    | Monotone Queue    | O(nk)       | O(n)        | 윈도우 최적화           |")
    print("    | Alien Trick       | 제약 문제   | 이완        | 볼록성                  |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
