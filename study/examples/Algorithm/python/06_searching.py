"""
이분 탐색 (Binary Search)
Binary Search Algorithms

정렬된 데이터에서 O(log n) 시간에 검색하는 알고리즘입니다.
"""

from typing import List, Optional
import bisect


# =============================================================================
# 1. 기본 이분 탐색
# =============================================================================
def binary_search(arr: List[int], target: int) -> int:
    """
    정렬된 배열에서 target의 인덱스 찾기
    없으면 -1 반환
    시간복잡도: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # 오버플로우 방지

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def binary_search_recursive(arr: List[int], target: int, left: int, right: int) -> int:
    """이분 탐색 (재귀 버전)"""
    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


# =============================================================================
# 2. Lower Bound / Upper Bound
# =============================================================================
def lower_bound(arr: List[int], target: int) -> int:
    """
    target 이상인 첫 번째 요소의 인덱스
    모든 요소가 target보다 작으면 len(arr) 반환
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left


def upper_bound(arr: List[int], target: int) -> int:
    """
    target 초과인 첫 번째 요소의 인덱스
    모든 요소가 target 이하이면 len(arr) 반환
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left


def count_occurrences(arr: List[int], target: int) -> int:
    """특정 값의 등장 횟수"""
    return upper_bound(arr, target) - lower_bound(arr, target)


# =============================================================================
# 3. 첫 번째/마지막 위치 찾기
# =============================================================================
def find_first_position(arr: List[int], target: int) -> int:
    """target이 처음 등장하는 인덱스 (없으면 -1)"""
    idx = lower_bound(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1


def find_last_position(arr: List[int], target: int) -> int:
    """target이 마지막으로 등장하는 인덱스 (없으면 -1)"""
    idx = upper_bound(arr, target) - 1
    if idx >= 0 and arr[idx] == target:
        return idx
    return -1


# =============================================================================
# 4. 회전 정렬 배열 검색
# =============================================================================
def search_rotated(arr: List[int], target: int) -> int:
    """
    회전된 정렬 배열에서 검색
    예: [4, 5, 6, 7, 0, 1, 2] - 원래 [0,1,2,4,5,6,7]을 회전
    시간복잡도: O(log n)
    """
    if not arr:
        return -1

    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid

        # 왼쪽 절반이 정렬되어 있는 경우
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 오른쪽 절반이 정렬되어 있는 경우
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


# =============================================================================
# 5. 최솟값 찾기 (회전 배열)
# =============================================================================
def find_minimum_rotated(arr: List[int]) -> int:
    """회전 정렬 배열의 최솟값 찾기"""
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid

    return arr[left]


# =============================================================================
# 6. 제곱근 구하기 (정수)
# =============================================================================
def integer_sqrt(n: int) -> int:
    """
    n의 정수 제곱근 (버림)
    예: sqrt(8) = 2
    """
    if n < 0:
        raise ValueError("음수의 제곱근은 정의되지 않습니다")
    if n == 0:
        return 0

    left, right = 1, n

    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid

        if square == n:
            return mid
        elif square < n:
            left = mid + 1
        else:
            right = mid - 1

    return right  # floor(sqrt(n))


# =============================================================================
# 7. 매개변수 탐색 (Parametric Search)
# =============================================================================
def can_split(arr: List[int], m: int, max_sum: int) -> bool:
    """
    배열을 m개 이하의 그룹으로 나눌 때
    각 그룹 합이 max_sum 이하인지 확인
    """
    count = 1
    current_sum = 0

    for num in arr:
        if current_sum + num > max_sum:
            count += 1
            current_sum = num
            if count > m:
                return False
        else:
            current_sum += num

    return True


def split_array_min_largest_sum(arr: List[int], m: int) -> int:
    """
    배열을 m개의 연속 부분 배열로 나눌 때
    각 부분 배열 합의 최댓값을 최소화
    시간복잡도: O(n log(sum))
    """
    left = max(arr)      # 최소 가능한 값: 가장 큰 요소
    right = sum(arr)     # 최대 가능한 값: 전체 합

    while left < right:
        mid = left + (right - left) // 2
        if can_split(arr, m, mid):
            right = mid
        else:
            left = mid + 1

    return left


# =============================================================================
# 8. Peak Element 찾기
# =============================================================================
def find_peak_element(arr: List[int]) -> int:
    """
    배열에서 peak element의 인덱스 찾기
    peak: arr[i] > arr[i-1] and arr[i] > arr[i+1]
    시간복잡도: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return left


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("이분 탐색 (Binary Search) 예제")
    print("=" * 60)

    # 1. 기본 이분 탐색
    print("\n[1] 기본 이분 탐색")
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    result = binary_search(arr, target)
    print(f"    배열: {arr}")
    print(f"    {target} 찾기 -> 인덱스: {result}")

    # 2. Lower/Upper Bound
    print("\n[2] Lower Bound / Upper Bound")
    arr = [1, 2, 2, 2, 3, 4, 5]
    target = 2
    lb = lower_bound(arr, target)
    ub = upper_bound(arr, target)
    count = count_occurrences(arr, target)
    print(f"    배열: {arr}")
    print(f"    target={target}")
    print(f"    lower_bound: {lb}, upper_bound: {ub}")
    print(f"    등장 횟수: {count}")

    # bisect 모듈과 비교
    print(f"    (bisect_left: {bisect.bisect_left(arr, target)}, "
          f"bisect_right: {bisect.bisect_right(arr, target)})")

    # 3. 첫 번째/마지막 위치
    print("\n[3] 첫 번째/마지막 위치")
    arr = [5, 7, 7, 8, 8, 8, 10]
    target = 8
    first = find_first_position(arr, target)
    last = find_last_position(arr, target)
    print(f"    배열: {arr}")
    print(f"    {target}의 첫 위치: {first}, 마지막 위치: {last}")

    # 4. 회전 배열 검색
    print("\n[4] 회전 정렬 배열 검색")
    arr = [4, 5, 6, 7, 0, 1, 2]
    target = 0
    result = search_rotated(arr, target)
    print(f"    회전 배열: {arr}")
    print(f"    {target} 찾기 -> 인덱스: {result}")

    # 5. 회전 배열 최솟값
    print("\n[5] 회전 배열 최솟값")
    arr = [4, 5, 6, 7, 0, 1, 2]
    result = find_minimum_rotated(arr)
    print(f"    회전 배열: {arr}")
    print(f"    최솟값: {result}")

    # 6. 제곱근
    print("\n[6] 정수 제곱근")
    for n in [4, 8, 16, 17, 100]:
        result = integer_sqrt(n)
        print(f"    sqrt({n}) = {result}")

    # 7. 매개변수 탐색
    print("\n[7] 매개변수 탐색 (배열 분할)")
    arr = [7, 2, 5, 10, 8]
    m = 2
    result = split_array_min_largest_sum(arr, m)
    print(f"    배열: {arr}, 분할 수: {m}")
    print(f"    최소 최대합: {result}")

    # 8. Peak Element
    print("\n[8] Peak Element 찾기")
    arr = [1, 2, 1, 3, 5, 6, 4]
    result = find_peak_element(arr)
    print(f"    배열: {arr}")
    print(f"    peak 인덱스: {result} (값: {arr[result]})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
