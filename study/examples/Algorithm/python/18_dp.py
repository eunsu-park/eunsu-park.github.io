"""
동적 프로그래밍 (Dynamic Programming) 기초
Basic Dynamic Programming

복잡한 문제를 작은 하위 문제로 나누어 해결하는 알고리즘 기법입니다.
"""

from typing import List
from functools import lru_cache


# =============================================================================
# 1. 피보나치 수열 (세 가지 방법)
# =============================================================================

# 방법 1: 재귀 (비효율적 - O(2^n))
def fibonacci_recursive(n: int) -> int:
    """재귀: 지수 시간 복잡도"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


# 방법 2: 메모이제이션 (Top-down DP)
def fibonacci_memo(n: int, memo: dict = None) -> int:
    """메모이제이션: O(n)"""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


# 방법 2b: lru_cache 데코레이터 사용
@lru_cache(maxsize=None)
def fibonacci_cached(n: int) -> int:
    """lru_cache 사용: O(n)"""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)


# 방법 3: 테뷸레이션 (Bottom-up DP)
def fibonacci_tabulation(n: int) -> int:
    """테뷸레이션: O(n) 시간, O(n) 공간"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


# 방법 4: 공간 최적화
def fibonacci_optimized(n: int) -> int:
    """공간 최적화: O(n) 시간, O(1) 공간"""
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1


# =============================================================================
# 2. 계단 오르기 (Climbing Stairs)
# =============================================================================
def climb_stairs(n: int) -> int:
    """
    한 번에 1칸 또는 2칸 오를 수 있을 때
    n개의 계단을 오르는 경우의 수
    """
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1


# =============================================================================
# 3. 동전 교환 (Coin Change)
# =============================================================================
def coin_change(coins: List[int], amount: int) -> int:
    """
    주어진 동전으로 금액을 만드는 최소 동전 개수
    불가능하면 -1 반환
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_ways(coins: List[int], amount: int) -> int:
    """
    주어진 동전으로 금액을 만드는 경우의 수
    """
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]


# =============================================================================
# 4. 0/1 배낭 문제 (Knapsack)
# =============================================================================
def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1 배낭 문제: 각 아이템을 넣거나 안 넣거나
    최대 가치 반환
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # 현재 아이템을 넣지 않는 경우
            dp[i][w] = dp[i - 1][w]
            # 현재 아이템을 넣을 수 있으면
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    return dp[n][capacity]


def knapsack_01_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """0/1 배낭 (공간 최적화 - 1D 배열)"""
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # 역순으로 순회 (같은 아이템 중복 사용 방지)
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


# =============================================================================
# 5. 최장 공통 부분 수열 (LCS)
# =============================================================================
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    두 문자열의 최장 공통 부분 수열 길이
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def get_lcs_string(text1: str, text2: str) -> str:
    """LCS 문자열 복원"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 역추적
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(lcs))


# =============================================================================
# 6. 최장 증가 부분 수열 (LIS)
# =============================================================================
def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    최장 증가 부분 수열 길이 (O(n²))
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n  # dp[i] = nums[i]로 끝나는 LIS 길이

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def lis_binary_search(nums: List[int]) -> int:
    """
    최장 증가 부분 수열 길이 (O(n log n))
    이분 탐색 활용
    """
    from bisect import bisect_left

    if not nums:
        return 0

    tails = []  # tails[i] = 길이 i+1인 LIS의 마지막 원소 최솟값

    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)


# =============================================================================
# 7. 최대 부분 배열 합 (Kadane's Algorithm)
# =============================================================================
def max_subarray_sum(nums: List[int]) -> int:
    """
    연속 부분 배열의 최대 합 (카데인 알고리즘)
    O(n) 시간, O(1) 공간
    """
    if not nums:
        return 0

    max_sum = curr_sum = nums[0]

    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)

    return max_sum


# =============================================================================
# 8. House Robber
# =============================================================================
def rob(nums: List[int]) -> int:
    """
    인접한 집을 털 수 없을 때 최대 금액
    """
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)

    prev2, prev1 = nums[0], max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr

    return prev1


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("동적 프로그래밍 (DP) 기초 예제")
    print("=" * 60)

    # 1. 피보나치
    print("\n[1] 피보나치 수열")
    n = 10
    print(f"    fib({n}) = {fibonacci_tabulation(n)}")
    print(f"    처음 10개: {[fibonacci_optimized(i) for i in range(10)]}")

    # 2. 계단 오르기
    print("\n[2] 계단 오르기")
    for n in [2, 3, 5]:
        ways = climb_stairs(n)
        print(f"    {n}개 계단: {ways}가지")

    # 3. 동전 교환
    print("\n[3] 동전 교환")
    coins = [1, 2, 5]
    amount = 11
    min_coins = coin_change(coins, amount)
    ways = coin_change_ways(coins, amount)
    print(f"    동전: {coins}, 금액: {amount}")
    print(f"    최소 동전 수: {min_coins}")
    print(f"    경우의 수: {ways}")

    # 4. 0/1 배낭
    print("\n[4] 0/1 배낭 문제")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    max_value = knapsack_01(weights, values, capacity)
    print(f"    무게: {weights}, 가치: {values}")
    print(f"    용량: {capacity}, 최대 가치: {max_value}")

    # 5. LCS
    print("\n[5] 최장 공통 부분 수열 (LCS)")
    text1, text2 = "ABCDGH", "AEDFHR"
    length = longest_common_subsequence(text1, text2)
    lcs_str = get_lcs_string(text1, text2)
    print(f"    문자열1: {text1}")
    print(f"    문자열2: {text2}")
    print(f"    LCS 길이: {length}, LCS: '{lcs_str}'")

    # 6. LIS
    print("\n[6] 최장 증가 부분 수열 (LIS)")
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    length = longest_increasing_subsequence(nums)
    length_fast = lis_binary_search(nums)
    print(f"    배열: {nums}")
    print(f"    LIS 길이: {length} (O(n²)), {length_fast} (O(n log n))")

    # 7. 최대 부분 배열 합
    print("\n[7] 최대 부분 배열 합 (Kadane)")
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = max_subarray_sum(nums)
    print(f"    배열: {nums}")
    print(f"    최대 합: {max_sum}")

    # 8. House Robber
    print("\n[8] House Robber")
    houses = [2, 7, 9, 3, 1]
    max_money = rob(houses)
    print(f"    집별 금액: {houses}")
    print(f"    최대 금액: {max_money}")

    print("\n" + "=" * 60)
    print("DP 접근 방식 비교")
    print("=" * 60)
    print("""
    | 방식           | 방향      | 구현          | 장점                    |
    |---------------|----------|---------------|------------------------|
    | 메모이제이션   | Top-down | 재귀 + 캐시   | 필요한 부분만 계산       |
    | 테뷸레이션     | Bottom-up| 반복문 + 배열 | 스택 오버플로우 없음     |

    DP 문제 해결 단계:
    1. 상태 정의: dp[i]가 무엇을 의미하는지
    2. 점화식 도출: dp[i]를 이전 상태로 표현
    3. 초기값 설정: base case
    4. 계산 순서 결정: 의존성에 따라
    5. 정답 도출: dp 테이블에서 답 추출
    """)


if __name__ == "__main__":
    main()
