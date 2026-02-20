"""
정렬 알고리즘 (Sorting Algorithms)
Sorting Algorithms Comparison

다양한 정렬 알고리즘의 구현과 비교입니다.
"""

import random
import time
from typing import List


# =============================================================================
# 1. 버블 정렬 (Bubble Sort)
# =============================================================================
def bubble_sort(arr: List[int]) -> List[int]:
    """
    버블 정렬
    시간: O(n²), 공간: O(1), 안정 정렬
    """
    arr = arr.copy()
    n = len(arr)

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break

    return arr


# =============================================================================
# 2. 선택 정렬 (Selection Sort)
# =============================================================================
def selection_sort(arr: List[int]) -> List[int]:
    """
    선택 정렬
    시간: O(n²), 공간: O(1), 불안정 정렬
    """
    arr = arr.copy()
    n = len(arr)

    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr


# =============================================================================
# 3. 삽입 정렬 (Insertion Sort)
# =============================================================================
def insertion_sort(arr: List[int]) -> List[int]:
    """
    삽입 정렬
    시간: O(n²), 최선 O(n), 공간: O(1), 안정 정렬
    거의 정렬된 데이터에 효율적
    """
    arr = arr.copy()
    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr


# =============================================================================
# 4. 병합 정렬 (Merge Sort)
# =============================================================================
def merge_sort(arr: List[int]) -> List[int]:
    """
    병합 정렬
    시간: O(n log n), 공간: O(n), 안정 정렬
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
# 5. 퀵 정렬 (Quick Sort)
# =============================================================================
def quick_sort(arr: List[int]) -> List[int]:
    """
    퀵 정렬
    시간: 평균 O(n log n), 최악 O(n²), 공간: O(log n), 불안정 정렬
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def quick_sort_inplace(arr: List[int], low: int = 0, high: int = None) -> List[int]:
    """퀵 정렬 (in-place)"""
    if high is None:
        arr = arr.copy()
        high = len(arr) - 1

    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort_inplace(arr, low, pivot_idx - 1)
        quick_sort_inplace(arr, pivot_idx + 1, high)

    return arr


def partition(arr: List[int], low: int, high: int) -> int:
    """파티션 (Lomuto scheme)"""
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# =============================================================================
# 6. 힙 정렬 (Heap Sort)
# =============================================================================
def heap_sort(arr: List[int]) -> List[int]:
    """
    힙 정렬
    시간: O(n log n), 공간: O(1), 불안정 정렬
    """
    arr = arr.copy()
    n = len(arr)

    # 최대 힙 구성
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 하나씩 추출
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr


def heapify(arr: List[int], n: int, i: int):
    """최대 힙 속성 유지"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


# =============================================================================
# 7. 계수 정렬 (Counting Sort)
# =============================================================================
def counting_sort(arr: List[int]) -> List[int]:
    """
    계수 정렬
    시간: O(n + k), 공간: O(k), 안정 정렬
    k = 최댓값 - 최솟값 + 1
    정수 범위가 작을 때 효율적
    """
    if not arr:
        return []

    min_val, max_val = min(arr), max(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    # 카운트
    for num in arr:
        count[num - min_val] += 1

    # 누적
    for i in range(1, range_val):
        count[i] += count[i - 1]

    # 역순으로 배치 (안정성 유지)
    for i in range(len(arr) - 1, -1, -1):
        num = arr[i]
        count[num - min_val] -= 1
        output[count[num - min_val]] = num

    return output


# =============================================================================
# 8. 기수 정렬 (Radix Sort)
# =============================================================================
def radix_sort(arr: List[int]) -> List[int]:
    """
    기수 정렬 (LSD)
    시간: O(d * (n + k)), 공간: O(n + k)
    d = 자릿수, k = 기수 (보통 10)
    음수 미지원 버전
    """
    if not arr or min(arr) < 0:
        return sorted(arr)  # 음수 포함 시 기본 정렬

    max_val = max(arr)

    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr


def counting_sort_by_digit(arr: List[int], exp: int) -> List[int]:
    """특정 자릿수 기준 계수 정렬"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        count[digit] -= 1
        output[count[digit]] = arr[i]

    return output


# =============================================================================
# 성능 비교
# =============================================================================
def benchmark_sorts(size: int = 1000):
    """정렬 알고리즘 성능 비교"""
    arr = [random.randint(0, 10000) for _ in range(size)]

    algorithms = [
        ("Bubble Sort", bubble_sort, size <= 1000),
        ("Selection Sort", selection_sort, size <= 1000),
        ("Insertion Sort", insertion_sort, size <= 1000),
        ("Merge Sort", merge_sort, True),
        ("Quick Sort", quick_sort, True),
        ("Heap Sort", heap_sort, True),
        ("Counting Sort", counting_sort, True),
        ("Radix Sort", radix_sort, True),
        ("Python sorted()", sorted, True),
    ]

    print(f"\n배열 크기: {size}")
    print("-" * 50)

    for name, func, should_run in algorithms:
        if should_run:
            test_arr = arr.copy()
            start = time.perf_counter()
            result = func(test_arr)
            elapsed = time.perf_counter() - start
            print(f"{name:20s}: {elapsed * 1000:8.3f} ms")
        else:
            print(f"{name:20s}: (크기 초과로 스킵)")


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("정렬 알고리즘 (Sorting Algorithms)")
    print("=" * 60)

    # 테스트 배열
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"\n원본 배열: {arr}")
    print("-" * 40)

    # 각 알고리즘 테스트
    algorithms = [
        ("버블 정렬", bubble_sort),
        ("선택 정렬", selection_sort),
        ("삽입 정렬", insertion_sort),
        ("병합 정렬", merge_sort),
        ("퀵 정렬", quick_sort),
        ("힙 정렬", heap_sort),
        ("계수 정렬", counting_sort),
        ("기수 정렬", radix_sort),
    ]

    for name, func in algorithms:
        result = func(arr.copy())
        print(f"{name}: {result}")

    # 성능 비교
    print("\n" + "=" * 60)
    print("성능 비교 (Performance Benchmark)")
    print("=" * 60)

    for size in [100, 1000, 5000]:
        benchmark_sorts(size)

    # 정렬 알고리즘 비교표
    print("\n" + "=" * 60)
    print("정렬 알고리즘 비교")
    print("=" * 60)
    print("""
    | 알고리즘     | 평균      | 최악      | 공간   | 안정성 | 특징                |
    |-------------|----------|----------|--------|-------|---------------------|
    | 버블 정렬    | O(n²)    | O(n²)    | O(1)   | 안정  | 단순, 교육용         |
    | 선택 정렬    | O(n²)    | O(n²)    | O(1)   | 불안정| 단순, 교환 횟수 적음  |
    | 삽입 정렬    | O(n²)    | O(n²)    | O(1)   | 안정  | 거의 정렬된 데이터에 좋음|
    | 병합 정렬    | O(nlogn) | O(nlogn) | O(n)   | 안정  | 일정한 성능          |
    | 퀵 정렬      | O(nlogn) | O(n²)    | O(logn)| 불안정| 평균적으로 가장 빠름  |
    | 힙 정렬      | O(nlogn) | O(nlogn) | O(1)   | 불안정| in-place, 일정한 성능|
    | 계수 정렬    | O(n+k)   | O(n+k)   | O(k)   | 안정  | 정수, 범위 작을 때    |
    | 기수 정렬    | O(d(n+k))| O(d(n+k))| O(n+k) | 안정  | 자릿수 기반           |

    실무 선택 가이드:
    - 일반적인 경우: 퀵 정렬 또는 언어 내장 정렬
    - 안정성 필요: 병합 정렬
    - 메모리 제한: 힙 정렬
    - 거의 정렬됨: 삽입 정렬
    - 정수 + 범위 작음: 계수/기수 정렬
    """)


if __name__ == "__main__":
    main()
