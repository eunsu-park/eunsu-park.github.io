"""
비트마스크 DP (Bitmask Dynamic Programming)
Bitmask DP

비트 연산을 활용하여 집합 상태를 표현하는 DP 기법입니다.
"""

from typing import List, Tuple
from functools import lru_cache


# =============================================================================
# 1. 비트 연산 기초
# =============================================================================

def bit_operations_demo():
    """비트 연산 기본"""
    n = 5  # 집합 크기

    # 빈 집합
    empty = 0

    # 전체 집합 {0, 1, 2, 3, 4}
    full = (1 << n) - 1  # 11111 (이진수)

    # i번째 원소 추가
    def add(mask: int, i: int) -> int:
        return mask | (1 << i)

    # i번째 원소 제거
    def remove(mask: int, i: int) -> int:
        return mask & ~(1 << i)

    # i번째 원소 토글
    def toggle(mask: int, i: int) -> int:
        return mask ^ (1 << i)

    # i번째 원소 포함 여부
    def contains(mask: int, i: int) -> bool:
        return bool(mask & (1 << i))

    # 원소 개수
    def count(mask: int) -> int:
        return bin(mask).count('1')

    # 최하위 비트 (가장 작은 원소)
    def lowest_bit(mask: int) -> int:
        return mask & (-mask)

    # 부분집합 순회
    def subsets(mask: int):
        """mask의 모든 부분집합을 순회"""
        subset = mask
        while True:
            yield subset
            if subset == 0:
                break
            subset = (subset - 1) & mask

    return {
        'empty': empty,
        'full': full,
        'add': add,
        'remove': remove,
        'toggle': toggle,
        'contains': contains,
        'count': count,
        'lowest_bit': lowest_bit,
        'subsets': subsets
    }


# =============================================================================
# 2. 외판원 문제 (TSP - Traveling Salesman Problem)
# =============================================================================

def tsp(dist: List[List[int]]) -> int:
    """
    외판원 문제 (TSP)
    모든 도시를 방문하고 시작점으로 돌아오는 최소 비용

    시간복잡도: O(n² * 2^n)
    공간복잡도: O(n * 2^n)
    """
    n = len(dist)
    INF = float('inf')

    # dp[mask][i] = mask에 포함된 도시를 방문하고 현재 i에 있을 때 최소 비용
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 시작점 (도시 0)

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue

                new_mask = mask | (1 << next_city)
                dp[new_mask][next_city] = min(
                    dp[new_mask][next_city],
                    dp[mask][last] + dist[last][next_city]
                )

    # 모든 도시 방문 후 시작점으로
    full_mask = (1 << n) - 1
    result = min(dp[full_mask][i] + dist[i][0] for i in range(n))

    return result if result != INF else -1


def tsp_path(dist: List[List[int]]) -> Tuple[int, List[int]]:
    """TSP 최소 비용과 경로 반환"""
    n = len(dist)
    INF = float('inf')

    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    dp[1][0] = 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue

                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + dist[last][next_city]

                if new_cost < dp[new_mask][next_city]:
                    dp[new_mask][next_city] = new_cost
                    parent[new_mask][next_city] = last

    full_mask = (1 << n) - 1
    min_cost = INF
    last_city = -1

    for i in range(n):
        cost = dp[full_mask][i] + dist[i][0]
        if cost < min_cost:
            min_cost = cost
            last_city = i

    # 경로 복원
    path = []
    mask = full_mask
    city = last_city

    while city != -1:
        path.append(city)
        prev_city = parent[mask][city]
        mask ^= (1 << city)
        city = prev_city

    path.reverse()
    path.append(0)  # 시작점으로 복귀

    return min_cost, path


# =============================================================================
# 3. 집합 분할 문제 (Set Partition)
# =============================================================================

def can_partition_k_subsets(nums: List[int], k: int) -> bool:
    """
    배열을 합이 같은 k개의 부분집합으로 분할 가능한지
    시간복잡도: O(n * 2^n)
    """
    total = sum(nums)
    if total % k != 0:
        return False

    target = total // k
    n = len(nums)

    # dp[mask] = mask 집합을 사용했을 때 현재 버킷의 합 (target으로 나눈 나머지)
    dp = [-1] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == -1:
            continue

        for i in range(n):
            if mask & (1 << i):
                continue

            if dp[mask] + nums[i] <= target:
                new_mask = mask | (1 << i)
                dp[new_mask] = (dp[mask] + nums[i]) % target

    return dp[(1 << n) - 1] == 0


# =============================================================================
# 4. 최소 비용 작업 할당 (Assignment Problem)
# =============================================================================

def min_cost_assignment(cost: List[List[int]]) -> int:
    """
    n명의 사람에게 n개의 작업을 1:1 할당하는 최소 비용
    cost[i][j] = 사람 i가 작업 j를 수행하는 비용

    시간복잡도: O(n * 2^n)
    """
    n = len(cost)

    @lru_cache(maxsize=None)
    def dp(mask: int) -> int:
        person = bin(mask).count('1')

        if person == n:
            return 0

        min_cost = float('inf')
        for job in range(n):
            if mask & (1 << job):
                continue

            min_cost = min(min_cost, cost[person][job] + dp(mask | (1 << job)))

        return min_cost

    return dp(0)


# =============================================================================
# 5. 해밀턴 경로 (Hamiltonian Path)
# =============================================================================

def hamiltonian_path_count(adj: List[List[int]]) -> int:
    """
    해밀턴 경로의 개수 (모든 정점을 한 번씩 방문하는 경로)
    adj: 인접 행렬 (adj[i][j] = 1이면 i→j 간선 존재)

    시간복잡도: O(n² * 2^n)
    """
    n = len(adj)

    # dp[mask][i] = mask 정점들을 방문하고 i에서 끝나는 경로 수
    dp = [[0] * n for _ in range(1 << n)]

    # 초기화: 각 정점에서 시작
    for i in range(n):
        dp[1 << i][i] = 1

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == 0:
                continue
            if not (mask & (1 << last)):
                continue

            for next_v in range(n):
                if mask & (1 << next_v):
                    continue
                if not adj[last][next_v]:
                    continue

                new_mask = mask | (1 << next_v)
                dp[new_mask][next_v] += dp[mask][last]

    # 모든 정점 방문한 경로 합
    full_mask = (1 << n) - 1
    return sum(dp[full_mask])


# =============================================================================
# 6. 스티커 최적 배치 (SOS DP 전처리)
# =============================================================================

def sos_dp(arr: List[int]) -> List[int]:
    """
    Sum over Subsets DP
    각 마스크에 대해 부분집합들의 값 합 계산

    result[mask] = sum(arr[subset]) for all subset of mask

    시간복잡도: O(n * 2^n)
    """
    n = len(arr).bit_length()
    dp = arr.copy()

    # 0~(len(arr)-1)까지 확장
    while len(dp) < (1 << n):
        dp.append(0)

    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                dp[mask] += dp[mask ^ (1 << i)]

    return dp


# =============================================================================
# 7. 최대 독립 집합 (Maximum Independent Set on Trees - Bitmask)
# =============================================================================

def max_independent_set(adj: List[List[int]]) -> int:
    """
    그래프에서 최대 독립 집합 크기 (서로 인접하지 않은 정점 집합)
    작은 그래프에서 비트마스크로 brute force

    시간복잡도: O(2^n * n²)
    """
    n = len(adj)
    max_size = 0

    for mask in range(1 << n):
        # mask가 독립 집합인지 확인
        valid = True
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(i + 1, n):
                if not (mask & (1 << j)):
                    continue
                if adj[i][j]:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            max_size = max(max_size, bin(mask).count('1'))

    return max_size


# =============================================================================
# 8. 격자 채우기 (Broken Profile DP)
# =============================================================================

def domino_tiling(m: int, n: int) -> int:
    """
    m×n 격자를 1×2 도미노로 채우는 경우의 수
    비트마스크 DP (profile 방식)

    시간복잡도: O(n * 2^m * 2^m)
    """
    if m > n:
        m, n = n, m

    # dp[col][profile] = 현재 열까지 채우고 프로파일이 profile인 경우의 수
    dp = {0: 1}

    for col in range(n):
        for row in range(m):
            new_dp = {}

            for profile, count in dp.items():
                # 현재 셀이 이미 채워진 경우
                if profile & (1 << row):
                    new_profile = profile ^ (1 << row)
                    new_dp[new_profile] = new_dp.get(new_profile, 0) + count
                else:
                    # 수평 도미노 (다음 열로 확장)
                    new_profile = profile | (1 << row)
                    new_dp[new_profile] = new_dp.get(new_profile, 0) + count

                    # 수직 도미노 (아래 셀과 함께)
                    if row + 1 < m and not (profile & (1 << (row + 1))):
                        new_dp[profile] = new_dp.get(profile, 0) + count

            dp = new_dp

    return dp.get(0, 0)


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("비트마스크 DP (Bitmask DP) 예제")
    print("=" * 60)

    # 1. 비트 연산 기초
    print("\n[1] 비트 연산 기초")
    ops = bit_operations_demo()
    mask = 0b10110  # {1, 2, 4}
    print(f"    mask = {bin(mask)} ({mask})")
    print(f"    원소 개수: {ops['count'](mask)}")
    print(f"    3 포함: {ops['contains'](mask, 3)}")
    print(f"    2 포함: {ops['contains'](mask, 2)}")
    print(f"    3 추가: {bin(ops['add'](mask, 3))}")
    print(f"    부분집합: ", end="")
    for s in ops['subsets'](mask):
        print(f"{bin(s)} ", end="")
    print()

    # 2. TSP
    print("\n[2] 외판원 문제 (TSP)")
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    min_cost, path = tsp_path(dist)
    print(f"    거리 행렬: 4x4")
    print(f"    최소 비용: {min_cost}")
    print(f"    경로: {path}")

    # 3. 집합 분할
    print("\n[3] K개 부분집합 분할")
    nums = [4, 3, 2, 3, 5, 2, 1]
    k = 4
    result = can_partition_k_subsets(nums, k)
    print(f"    배열: {nums}, k={k}")
    print(f"    분할 가능: {result}")

    # 4. 작업 할당
    print("\n[4] 최소 비용 작업 할당")
    cost = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4]
    ]
    min_assign = min_cost_assignment(cost)
    print(f"    비용 행렬: 4x4")
    print(f"    최소 비용: {min_assign}")

    # 5. 해밀턴 경로
    print("\n[5] 해밀턴 경로 개수")
    adj = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    count = hamiltonian_path_count(adj)
    print(f"    인접 행렬: 4x4")
    print(f"    해밀턴 경로 수: {count}")

    # 6. 최대 독립 집합
    print("\n[6] 최대 독립 집합")
    adj2 = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    mis = max_independent_set(adj2)
    print(f"    그래프: 4-cycle")
    print(f"    최대 독립 집합 크기: {mis}")

    # 7. 도미노 타일링
    print("\n[7] 도미노 타일링")
    for m, n in [(2, 3), (2, 4), (3, 4)]:
        count = domino_tiling(m, n)
        print(f"    {m}×{n} 격자: {count}가지")

    # 8. SOS DP
    print("\n[8] SOS DP (부분집합 합)")
    arr = [1, 2, 4, 8]  # 각 원소는 해당 비트의 값
    result = sos_dp(arr)
    print(f"    배열: {arr}")
    print(f"    result[0b0111] = result[7] = {result[7]}")
    print(f"    (부분집합: {{0},{1},{0,1},{2},{0,2},{1,2},{0,1,2}} 합)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
