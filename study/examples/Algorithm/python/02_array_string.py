"""
투 포인터 (Two Pointer) 기법
Two Pointer Technique

두 개의 포인터를 사용하여 배열/리스트 문제를 효율적으로 해결합니다.
"""

from typing import List, Tuple, Optional


# =============================================================================
# 1. 두 수의 합 (정렬된 배열)
# =============================================================================
def two_sum_sorted(arr: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    정렬된 배열에서 합이 target인 두 수의 인덱스 찾기
    시간복잡도: O(n), 공간복잡도: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return (left, right)
        elif current_sum < target:
            left += 1  # 합이 작으면 왼쪽 포인터를 오른쪽으로
        else:
            right -= 1  # 합이 크면 오른쪽 포인터를 왼쪽으로

    return None


# =============================================================================
# 2. 세 수의 합 (3Sum)
# =============================================================================
def three_sum(arr: List[int]) -> List[List[int]]:
    """
    합이 0인 세 수의 조합 모두 찾기 (중복 제거)
    시간복잡도: O(n²), 공간복잡도: O(1)
    """
    arr.sort()
    result = []
    n = len(arr)

    for i in range(n - 2):
        # 중복 건너뛰기
        if i > 0 and arr[i] == arr[i - 1]:
            continue

        left, right = i + 1, n - 1

        while left < right:
            total = arr[i] + arr[left] + arr[right]

            if total == 0:
                result.append([arr[i], arr[left], arr[right]])
                # 중복 건너뛰기
                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result


# =============================================================================
# 3. 물 담기 (Container With Most Water)
# =============================================================================
def max_water(heights: List[int]) -> int:
    """
    두 벽 사이에 담을 수 있는 최대 물의 양
    시간복잡도: O(n), 공간복잡도: O(1)
    """
    left, right = 0, len(heights) - 1
    max_area = 0

    while left < right:
        # 현재 면적 계산
        width = right - left
        height = min(heights[left], heights[right])
        area = width * height
        max_area = max(max_area, area)

        # 더 낮은 쪽의 포인터 이동
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1

    return max_area


# =============================================================================
# 4. 정렬된 두 배열 합병
# =============================================================================
def merge_sorted_arrays(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    정렬된 두 배열을 하나의 정렬된 배열로 합병
    시간복잡도: O(n + m), 공간복잡도: O(n + m)
    """
    result = []
    i, j = 0, 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # 남은 요소 추가
    result.extend(arr1[i:])
    result.extend(arr2[j:])

    return result


# =============================================================================
# 5. 중복 제거 (정렬된 배열)
# =============================================================================
def remove_duplicates(arr: List[int]) -> int:
    """
    정렬된 배열에서 중복 제거 (in-place)
    반환값: 고유한 요소의 개수
    시간복잡도: O(n), 공간복잡도: O(1)
    """
    if not arr:
        return 0

    write_idx = 1  # 다음 고유 값을 쓸 위치

    for read_idx in range(1, len(arr)):
        if arr[read_idx] != arr[write_idx - 1]:
            arr[write_idx] = arr[read_idx]
            write_idx += 1

    return write_idx


# =============================================================================
# 6. 회문 검사 (Palindrome)
# =============================================================================
def is_palindrome(s: str) -> bool:
    """
    문자열이 회문인지 검사 (알파벳/숫자만 비교)
    시간복잡도: O(n), 공간복잡도: O(1)
    """
    left, right = 0, len(s) - 1

    while left < right:
        # 알파벳/숫자가 아니면 건너뛰기
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True


# =============================================================================
# 7. 슬라이딩 윈도우 (최대 합)
# =============================================================================
def max_sum_subarray(arr: List[int], k: int) -> int:
    """
    크기 k인 연속 부분 배열의 최대 합
    시간복잡도: O(n), 공간복잡도: O(1)
    """
    if len(arr) < k:
        return 0

    # 초기 윈도우 합
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # 윈도우 슬라이딩
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("투 포인터 (Two Pointer) 기법 예제")
    print("=" * 60)

    # 1. 두 수의 합
    print("\n[1] 두 수의 합 (정렬된 배열)")
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    target = 10
    result = two_sum_sorted(arr, target)
    print(f"    배열: {arr}")
    print(f"    타겟: {target}")
    print(f"    결과: 인덱스 {result} -> {arr[result[0]]} + {arr[result[1]]} = {target}")

    # 2. 세 수의 합
    print("\n[2] 세 수의 합 (3Sum)")
    arr = [-1, 0, 1, 2, -1, -4]
    result = three_sum(arr)
    print(f"    배열: {arr}")
    print(f"    합이 0인 조합: {result}")

    # 3. 물 담기
    print("\n[3] 물 담기 (Container With Most Water)")
    heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    result = max_water(heights)
    print(f"    높이: {heights}")
    print(f"    최대 물의 양: {result}")

    # 4. 정렬된 배열 합병
    print("\n[4] 정렬된 두 배열 합병")
    arr1 = [1, 3, 5, 7]
    arr2 = [2, 4, 6, 8, 10]
    result = merge_sorted_arrays(arr1, arr2)
    print(f"    배열1: {arr1}")
    print(f"    배열2: {arr2}")
    print(f"    합병 결과: {result}")

    # 5. 중복 제거
    print("\n[5] 중복 제거 (in-place)")
    arr = [1, 1, 2, 2, 2, 3, 4, 4, 5]
    count = remove_duplicates(arr)
    print(f"    원본: [1, 1, 2, 2, 2, 3, 4, 4, 5]")
    print(f"    고유 요소 개수: {count}")
    print(f"    결과 배열 (앞 {count}개): {arr[:count]}")

    # 6. 회문 검사
    print("\n[6] 회문 검사")
    test_strings = ["A man, a plan, a canal: Panama", "race a car", "Was it a car or a cat I saw?"]
    for s in test_strings:
        result = is_palindrome(s)
        print(f"    '{s}' -> {result}")

    # 7. 슬라이딩 윈도우
    print("\n[7] 슬라이딩 윈도우 (크기 k 최대 합)")
    arr = [2, 1, 5, 1, 3, 2]
    k = 3
    result = max_sum_subarray(arr, k)
    print(f"    배열: {arr}, k={k}")
    print(f"    최대 합: {result}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
