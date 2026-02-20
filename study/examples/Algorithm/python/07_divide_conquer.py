"""
분할 정복 (Divide and Conquer)
Divide and Conquer Algorithms

문제를 작은 부분 문제로 나누어 해결하는 알고리즘입니다.
"""

from typing import List, Tuple, Optional
import random


# =============================================================================
# 1. 병합 정렬 (Merge Sort)
# =============================================================================

def merge_sort(arr: List[int]) -> List[int]:
    """
    병합 정렬
    시간복잡도: O(n log n)
    공간복잡도: O(n)
    안정 정렬
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """두 정렬된 배열 병합"""
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
# 2. 퀵 정렬 (Quick Sort)
# =============================================================================

def quick_sort(arr: List[int]) -> List[int]:
    """
    퀵 정렬 (Lomuto 파티션)
    시간복잡도: 평균 O(n log n), 최악 O(n²)
    공간복잡도: O(log n) - 재귀 스택
    불안정 정렬
    """
    if len(arr) <= 1:
        return arr

    arr = arr.copy()
    _quick_sort(arr, 0, len(arr) - 1)
    return arr


def _quick_sort(arr: List[int], low: int, high: int) -> None:
    if low < high:
        pivot_idx = partition(arr, low, high)
        _quick_sort(arr, low, pivot_idx - 1)
        _quick_sort(arr, pivot_idx + 1, high)


def partition(arr: List[int], low: int, high: int) -> int:
    """Lomuto 파티션"""
    # 랜덤 피벗으로 최악 케이스 방지
    pivot_idx = random.randint(low, high)
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]

    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# =============================================================================
# 3. 거듭제곱 (Power / Exponentiation)
# =============================================================================

def power(base: int, exp: int, mod: int = None) -> int:
    """
    빠른 거듭제곱
    시간복잡도: O(log n)
    """
    if exp == 0:
        return 1

    if exp % 2 == 0:
        half = power(base, exp // 2, mod)
        result = half * half
    else:
        result = base * power(base, exp - 1, mod)

    return result % mod if mod else result


def power_iterative(base: int, exp: int, mod: int = None) -> int:
    """빠른 거듭제곱 (반복)"""
    result = 1

    while exp > 0:
        if exp % 2 == 1:
            result = result * base
            if mod:
                result %= mod
        base = base * base
        if mod:
            base %= mod
        exp //= 2

    return result


# =============================================================================
# 4. 행렬 거듭제곱 (Matrix Exponentiation)
# =============================================================================

def matrix_multiply(A: List[List[int]], B: List[List[int]], mod: int = None) -> List[List[int]]:
    """2x2 행렬 곱셈"""
    n = len(A)
    C = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                if mod:
                    C[i][j] %= mod

    return C


def matrix_power(M: List[List[int]], exp: int, mod: int = None) -> List[List[int]]:
    """
    행렬 거듭제곱
    시간복잡도: O(k³ log n), k = 행렬 크기
    """
    n = len(M)
    # 단위 행렬
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    while exp > 0:
        if exp % 2 == 1:
            result = matrix_multiply(result, M, mod)
        M = matrix_multiply(M, M, mod)
        exp //= 2

    return result


def fibonacci_matrix(n: int, mod: int = None) -> int:
    """
    피보나치 수열 (행렬 거듭제곱)
    시간복잡도: O(log n)
    """
    if n <= 1:
        return n

    # [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n
    M = [[1, 1], [1, 0]]
    result = matrix_power(M, n, mod)
    return result[0][1]


# =============================================================================
# 5. 역순 쌍 개수 (Inversion Count)
# =============================================================================

def count_inversions(arr: List[int]) -> int:
    """
    역순 쌍 개수 (i < j이면서 arr[i] > arr[j])
    병합 정렬 변형
    시간복잡도: O(n log n)
    """

    def merge_count(arr: List[int]) -> Tuple[List[int], int]:
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, left_inv = merge_count(arr[:mid])
        right, right_inv = merge_count(arr[mid:])

        merged = []
        inversions = left_inv + right_inv
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inversions += len(left) - i  # 남은 왼쪽 요소 수만큼 역순
                j += 1

        merged.extend(left[i:])
        merged.extend(right[j:])

        return merged, inversions

    _, count = merge_count(arr)
    return count


# =============================================================================
# 6. 가장 가까운 점 쌍 (Closest Pair of Points)
# =============================================================================

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """두 점 사이 거리"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def closest_pair(points: List[Tuple[float, float]]) -> float:
    """
    가장 가까운 두 점 사이의 거리
    시간복잡도: O(n log n)
    """

    def closest_recursive(px: List, py: List) -> float:
        n = len(px)

        # 기저 케이스: 브루트 포스
        if n <= 3:
            min_dist = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_dist = min(min_dist, distance(px[i], px[j]))
            return min_dist

        mid = n // 2
        mid_point = px[mid]

        # x좌표 기준 분할
        pyl = [p for p in py if p[0] <= mid_point[0]]
        pyr = [p for p in py if p[0] > mid_point[0]]

        dl = closest_recursive(px[:mid], pyl)
        dr = closest_recursive(px[mid:], pyr)

        d = min(dl, dr)

        # 중간 띠에서 확인
        strip = [p for p in py if abs(p[0] - mid_point[0]) < d]

        # 띠 내 점들 비교 (최대 7개만 확인)
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):
                if strip[j][1] - strip[i][1] >= d:
                    break
                d = min(d, distance(strip[i], strip[j]))

        return d

    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])

    return closest_recursive(px, py)


# =============================================================================
# 7. 최대 부분 배열 합 (Maximum Subarray - D&C)
# =============================================================================

def max_subarray_dc(arr: List[int]) -> int:
    """
    최대 부분 배열 합 (분할 정복)
    시간복잡도: O(n log n)
    """

    def max_crossing_sum(arr: List[int], low: int, mid: int, high: int) -> int:
        # 왼쪽 최대
        left_sum = float('-inf')
        total = 0
        for i in range(mid, low - 1, -1):
            total += arr[i]
            left_sum = max(left_sum, total)

        # 오른쪽 최대
        right_sum = float('-inf')
        total = 0
        for i in range(mid + 1, high + 1):
            total += arr[i]
            right_sum = max(right_sum, total)

        return left_sum + right_sum

    def max_subarray(arr: List[int], low: int, high: int) -> int:
        if low == high:
            return arr[low]

        mid = (low + high) // 2

        left_max = max_subarray(arr, low, mid)
        right_max = max_subarray(arr, mid + 1, high)
        cross_max = max_crossing_sum(arr, low, mid, high)

        return max(left_max, right_max, cross_max)

    if not arr:
        return 0
    return max_subarray(arr, 0, len(arr) - 1)


# =============================================================================
# 8. 카라츠바 곱셈 (Karatsuba Multiplication)
# =============================================================================

def karatsuba(x: int, y: int) -> int:
    """
    카라츠바 큰 수 곱셈
    시간복잡도: O(n^1.585)
    """
    # 기저 케이스
    if x < 10 or y < 10:
        return x * y

    # 자릿수 계산
    n = max(len(str(x)), len(str(y)))
    m = n // 2

    # x = a * 10^m + b, y = c * 10^m + d
    divisor = 10 ** m

    a, b = divmod(x, divisor)
    c, d = divmod(y, divisor)

    # 세 번의 곱셈
    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    ad_bc = karatsuba(a + b, c + d) - ac - bd

    return ac * (10 ** (2 * m)) + ad_bc * (10 ** m) + bd


# =============================================================================
# 9. 스트라센 행렬 곱셈 (Strassen's Matrix Multiplication)
# =============================================================================

def strassen(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """
    스트라센 행렬 곱셈
    시간복잡도: O(n^2.807)
    (실제로는 작은 행렬에서 오버헤드가 크므로 기준 크기 이하는 일반 곱셈)
    """
    n = len(A)

    # 기저 케이스
    if n <= 64:  # 임계값
        return naive_matrix_multiply(A, B)

    # 행렬 분할
    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    # 7개의 곱셈 (스트라센 공식)
    M1 = strassen(matrix_add(A11, A22), matrix_add(B11, B22))
    M2 = strassen(matrix_add(A21, A22), B11)
    M3 = strassen(A11, matrix_sub(B12, B22))
    M4 = strassen(A22, matrix_sub(B21, B11))
    M5 = strassen(matrix_add(A11, A12), B22)
    M6 = strassen(matrix_sub(A21, A11), matrix_add(B11, B12))
    M7 = strassen(matrix_sub(A12, A22), matrix_add(B21, B22))

    # 결과 조합
    C11 = matrix_add(matrix_sub(matrix_add(M1, M4), M5), M7)
    C12 = matrix_add(M3, M5)
    C21 = matrix_add(M2, M4)
    C22 = matrix_add(matrix_sub(matrix_add(M1, M3), M2), M6)

    return combine_matrices(C11, C12, C21, C22)


def naive_matrix_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """일반 행렬 곱셈"""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matrix_add(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """행렬 덧셈"""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def matrix_sub(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """행렬 뺄셈"""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def combine_matrices(C11, C12, C21, C22) -> List[List[int]]:
    """4개의 부분 행렬 결합"""
    n = len(C11)
    result = [[0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = C11[i][j]
            result[i][j + n] = C12[i][j]
            result[i + n][j] = C21[i][j]
            result[i + n][j + n] = C22[i][j]
    return result


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("분할 정복 (Divide and Conquer) 예제")
    print("=" * 60)

    # 1. 병합 정렬
    print("\n[1] 병합 정렬")
    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = merge_sort(arr)
    print(f"    원본: {arr}")
    print(f"    정렬: {sorted_arr}")

    # 2. 퀵 정렬
    print("\n[2] 퀵 정렬")
    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = quick_sort(arr)
    print(f"    원본: {arr}")
    print(f"    정렬: {sorted_arr}")

    # 3. 빠른 거듭제곱
    print("\n[3] 빠른 거듭제곱")
    print(f"    2^10 = {power(2, 10)}")
    print(f"    2^10 (반복) = {power_iterative(2, 10)}")
    print(f"    3^7 mod 1000 = {power(3, 7, 1000)}")

    # 4. 피보나치 (행렬 거듭제곱)
    print("\n[4] 피보나치 (행렬 거듭제곱)")
    for n in [10, 20, 50]:
        fib = fibonacci_matrix(n)
        print(f"    F({n}) = {fib}")

    # 5. 역순 쌍 개수
    print("\n[5] 역순 쌍 개수")
    arr = [2, 4, 1, 3, 5]
    inv = count_inversions(arr)
    print(f"    배열: {arr}")
    print(f"    역순 쌍: {inv}개")

    # 6. 가장 가까운 점 쌍
    print("\n[6] 가장 가까운 점 쌍")
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    dist = closest_pair(points)
    print(f"    점들: {points}")
    print(f"    최소 거리: {dist:.4f}")

    # 7. 최대 부분 배열 합 (D&C)
    print("\n[7] 최대 부분 배열 합 (분할 정복)")
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = max_subarray_dc(arr)
    print(f"    배열: {arr}")
    print(f"    최대 합: {max_sum}")

    # 8. 카라츠바 곱셈
    print("\n[8] 카라츠바 곱셈")
    x, y = 1234, 5678
    result = karatsuba(x, y)
    print(f"    {x} × {y} = {result}")
    print(f"    검증: {x * y}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
