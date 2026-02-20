"""
시간 복잡도 비교 실험
Time Complexity Comparison

다양한 시간 복잡도를 가진 알고리즘들의 실행 시간을 비교합니다.
"""

import time
import random


def measure_time(func, *args):
    """함수 실행 시간 측정"""
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return end - start, result


# =============================================================================
# O(1) - 상수 시간
# =============================================================================
def constant_time(arr):
    """배열의 첫 번째 요소 반환 - O(1)"""
    if arr:
        return arr[0]
    return None


# =============================================================================
# O(log n) - 로그 시간
# =============================================================================
def binary_search(arr, target):
    """이분 탐색 - O(log n)"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# =============================================================================
# O(n) - 선형 시간
# =============================================================================
def linear_search(arr, target):
    """선형 탐색 - O(n)"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


def find_max(arr):
    """최댓값 찾기 - O(n)"""
    if not arr:
        return None
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val


# =============================================================================
# O(n log n) - 선형 로그 시간
# =============================================================================
def merge_sort(arr):
    """병합 정렬 - O(n log n)"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# =============================================================================
# O(n²) - 제곱 시간
# =============================================================================
def bubble_sort(arr):
    """버블 정렬 - O(n²)"""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def has_duplicate_naive(arr):
    """중복 검사 (naive) - O(n²)"""
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] == arr[j]:
                return True
    return False


# =============================================================================
# O(2^n) - 지수 시간
# =============================================================================
def fibonacci_recursive(n):
    """피보나치 (재귀) - O(2^n)"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_dp(n):
    """피보나치 (DP) - O(n)"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


# =============================================================================
# 실험 및 결과 출력
# =============================================================================
def run_experiments():
    """시간 복잡도 실험 실행"""
    print("=" * 60)
    print("시간 복잡도 비교 실험")
    print("=" * 60)

    # 데이터 준비
    sizes = [100, 1000, 10000]

    for size in sizes:
        print(f"\n[ 배열 크기: {size} ]")
        print("-" * 40)

        arr = list(range(size))
        random_arr = random.sample(range(size * 10), size)
        target = arr[size // 2]  # 중간 값

        # O(1) 테스트
        t, _ = measure_time(constant_time, arr)
        print(f"O(1)     상수 시간:      {t * 1000:.6f} ms")

        # O(log n) 테스트
        t, _ = measure_time(binary_search, arr, target)
        print(f"O(log n) 이분 탐색:      {t * 1000:.6f} ms")

        # O(n) 테스트
        t, _ = measure_time(linear_search, arr, target)
        print(f"O(n)     선형 탐색:      {t * 1000:.6f} ms")

        t, _ = measure_time(find_max, random_arr)
        print(f"O(n)     최댓값 찾기:    {t * 1000:.6f} ms")

        # O(n log n) 테스트
        if size <= 10000:
            t, _ = measure_time(merge_sort, random_arr)
            print(f"O(n log n) 병합 정렬:  {t * 1000:.6f} ms")

        # O(n²) 테스트 (작은 크기만)
        if size <= 1000:
            t, _ = measure_time(bubble_sort, random_arr)
            print(f"O(n²)    버블 정렬:      {t * 1000:.6f} ms")

    # O(2^n) vs O(n) 피보나치 비교
    print("\n[ 피보나치 비교: O(2^n) vs O(n) ]")
    print("-" * 40)

    for n in [10, 20, 30]:
        t_recursive, _ = measure_time(fibonacci_recursive, n)
        t_dp, _ = measure_time(fibonacci_dp, n)
        print(f"n={n}: 재귀 O(2^n) = {t_recursive * 1000:.4f} ms, "
              f"DP O(n) = {t_dp * 1000:.6f} ms")

    print("\n" + "=" * 60)
    print("실험 완료!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiments()
