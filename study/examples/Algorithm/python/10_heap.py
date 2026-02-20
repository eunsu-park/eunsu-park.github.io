"""
힙과 우선순위 큐 (Heap & Priority Queue)
Heap and Priority Queue

힙 자료구조와 관련 알고리즘을 구현합니다.
"""

from typing import List, Optional, Tuple, Any
import heapq


# =============================================================================
# 1. 최소 힙 (Min Heap) 직접 구현
# =============================================================================

class MinHeap:
    """
    최소 힙 (완전 이진 트리)
    - 부모 노드 ≤ 자식 노드
    - 배열로 구현: parent(i) = (i-1)//2, left(i) = 2i+1, right(i) = 2i+2
    """

    def __init__(self):
        self.heap: List[int] = []

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return len(self.heap) > 0

    def _parent(self, i: int) -> int:
        return (i - 1) // 2

    def _left(self, i: int) -> int:
        return 2 * i + 1

    def _right(self, i: int) -> int:
        return 2 * i + 2

    def _swap(self, i: int, j: int) -> None:
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _sift_up(self, i: int) -> None:
        """삽입 후 위로 재정렬 - O(log n)"""
        while i > 0:
            parent = self._parent(i)
            if self.heap[i] < self.heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        """삭제 후 아래로 재정렬 - O(log n)"""
        n = len(self.heap)

        while True:
            smallest = i
            left = self._left(i)
            right = self._right(i)

            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def push(self, val: int) -> None:
        """삽입 - O(log n)"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self) -> int:
        """최솟값 제거 및 반환 - O(log n)"""
        if not self.heap:
            raise IndexError("pop from empty heap")

        result = self.heap[0]
        last = self.heap.pop()

        if self.heap:
            self.heap[0] = last
            self._sift_down(0)

        return result

    def peek(self) -> int:
        """최솟값 조회 - O(1)"""
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0]

    def heapify(self, arr: List[int]) -> None:
        """배열을 힙으로 변환 - O(n)"""
        self.heap = arr[:]
        # 마지막 비-리프 노드부터 sift_down
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._sift_down(i)


# =============================================================================
# 2. 최대 힙 (Max Heap)
# =============================================================================

class MaxHeap:
    """최대 힙 - 부모 노드 ≥ 자식 노드"""

    def __init__(self):
        self.heap: List[int] = []

    def __len__(self) -> int:
        return len(self.heap)

    def push(self, val: int) -> None:
        """삽입 - O(log n)"""
        # 최소 힙에 음수로 저장
        heapq.heappush(self.heap, -val)

    def pop(self) -> int:
        """최댓값 제거 및 반환 - O(log n)"""
        return -heapq.heappop(self.heap)

    def peek(self) -> int:
        """최댓값 조회 - O(1)"""
        return -self.heap[0]


# =============================================================================
# 3. 힙 정렬 (Heap Sort)
# =============================================================================

def heap_sort(arr: List[int]) -> List[int]:
    """
    힙 정렬
    시간복잡도: O(n log n)
    공간복잡도: O(1) (in-place)
    """
    n = len(arr)
    result = arr[:]

    def sift_down(arr: List[int], n: int, i: int) -> None:
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and arr[left] > arr[largest]:
                largest = left
            if right < n and arr[right] > arr[largest]:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                i = largest
            else:
                break

    # 1. 최대 힙 구성 - O(n)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(result, n, i)

    # 2. 하나씩 추출 - O(n log n)
    for i in range(n - 1, 0, -1):
        result[0], result[i] = result[i], result[0]
        sift_down(result, i, 0)

    return result


# =============================================================================
# 4. K번째 요소 찾기
# =============================================================================

def kth_largest(arr: List[int], k: int) -> int:
    """
    K번째로 큰 요소 - 최소 힙 사용
    시간복잡도: O(n log k)
    공간복잡도: O(k)
    """
    min_heap = []

    for num in arr:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return min_heap[0]


def kth_smallest(arr: List[int], k: int) -> int:
    """
    K번째로 작은 요소 - 최대 힙 사용
    시간복잡도: O(n log k)
    공간복잡도: O(k)
    """
    max_heap = []

    for num in arr:
        heapq.heappush(max_heap, -num)
        if len(max_heap) > k:
            heapq.heappop(max_heap)

    return -max_heap[0]


# =============================================================================
# 5. 중앙값 스트림 (Median Finder)
# =============================================================================

class MedianFinder:
    """
    데이터 스트림의 중앙값
    - 최대 힙 (왼쪽 절반): 작은 값들
    - 최소 힙 (오른쪽 절반): 큰 값들
    """

    def __init__(self):
        self.max_heap: List[int] = []  # 왼쪽 (작은 값들)
        self.min_heap: List[int] = []  # 오른쪽 (큰 값들)

    def add_num(self, num: int) -> None:
        """숫자 추가 - O(log n)"""
        # 왼쪽 힙에 추가
        heapq.heappush(self.max_heap, -num)

        # 왼쪽 최댓값을 오른쪽으로 이동
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # 크기 균형 유지 (왼쪽 ≥ 오른쪽)
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self) -> float:
        """중앙값 반환 - O(1)"""
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2


# =============================================================================
# 6. 우선순위 큐 응용: 작업 스케줄링
# =============================================================================

def schedule_tasks(tasks: List[Tuple[int, int, str]]) -> List[str]:
    """
    우선순위 기반 작업 스케줄링
    tasks: [(우선순위, 도착시간, 작업명), ...]
    낮은 우선순위 번호 = 높은 우선순위
    """
    # (우선순위, 도착시간, 작업명) 형태로 힙에 추가
    task_heap = []
    for priority, arrival, name in tasks:
        heapq.heappush(task_heap, (priority, arrival, name))

    schedule = []
    while task_heap:
        _, _, name = heapq.heappop(task_heap)
        schedule.append(name)

    return schedule


# =============================================================================
# 7. K개 정렬된 리스트 병합
# =============================================================================

def merge_k_sorted_lists(lists: List[List[int]]) -> List[int]:
    """
    K개의 정렬된 리스트 병합
    시간복잡도: O(N log k), N = 전체 원소 수, k = 리스트 수
    """
    result = []
    min_heap = []

    # 각 리스트의 첫 번째 원소를 힙에 추가
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))  # (값, 리스트 인덱스, 원소 인덱스)

    while min_heap:
        val, list_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)

        # 다음 원소가 있으면 힙에 추가
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))

    return result


# =============================================================================
# 8. Top K 빈도 요소
# =============================================================================

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    가장 빈번한 K개 요소
    시간복잡도: O(n log k)
    """
    from collections import Counter

    freq = Counter(nums)

    # 최소 힙으로 K개만 유지
    min_heap = []
    for num, count in freq.items():
        heapq.heappush(min_heap, (count, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return [num for count, num in min_heap]


# =============================================================================
# 9. 가장 가까운 K개 점
# =============================================================================

def k_closest_points(points: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
    """
    원점에서 가장 가까운 K개 점
    시간복잡도: O(n log k)
    """
    # 최대 힙으로 K개만 유지 (거리의 음수 저장)
    max_heap = []

    for x, y in points:
        dist = x * x + y * y
        heapq.heappush(max_heap, (-dist, x, y))
        if len(max_heap) > k:
            heapq.heappop(max_heap)

    return [(x, y) for _, x, y in max_heap]


# =============================================================================
# 10. 회의실 문제 (Meeting Rooms)
# =============================================================================

def min_meeting_rooms(intervals: List[Tuple[int, int]]) -> int:
    """
    필요한 최소 회의실 수
    intervals: [(시작시간, 종료시간), ...]
    시간복잡도: O(n log n)
    """
    if not intervals:
        return 0

    # 시작 시간으로 정렬
    intervals.sort(key=lambda x: x[0])

    # 종료 시간을 저장하는 최소 힙
    end_times = []
    heapq.heappush(end_times, intervals[0][1])

    for start, end in intervals[1:]:
        # 가장 빨리 끝나는 회의가 현재 회의 시작 전에 끝나면 재사용
        if end_times[0] <= start:
            heapq.heappop(end_times)
        heapq.heappush(end_times, end)

    return len(end_times)


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("힙과 우선순위 큐 (Heap & Priority Queue) 예제")
    print("=" * 60)

    # 1. 최소 힙
    print("\n[1] 최소 힙 (직접 구현)")
    min_heap = MinHeap()
    for val in [5, 3, 8, 1, 2, 7]:
        min_heap.push(val)
    print(f"    삽입: [5, 3, 8, 1, 2, 7]")
    print(f"    힙 배열: {min_heap.heap}")
    print(f"    pop 순서: ", end="")
    result = []
    while min_heap:
        result.append(min_heap.pop())
    print(result)

    # 2. 최대 힙
    print("\n[2] 최대 힙")
    max_heap = MaxHeap()
    for val in [5, 3, 8, 1, 2, 7]:
        max_heap.push(val)
    print(f"    삽입: [5, 3, 8, 1, 2, 7]")
    print(f"    pop 순서: ", end="")
    result = []
    while max_heap:
        result.append(max_heap.pop())
    print(result)

    # 3. Heapify
    print("\n[3] Heapify (배열 → 힙)")
    arr = [9, 5, 6, 2, 3]
    heap = MinHeap()
    heap.heapify(arr)
    print(f"    원본: {arr}")
    print(f"    힙: {heap.heap}")

    # 4. 힙 정렬
    print("\n[4] 힙 정렬")
    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = heap_sort(arr)
    print(f"    원본: {arr}")
    print(f"    정렬: {sorted_arr}")

    # 5. K번째 요소
    print("\n[5] K번째 요소")
    arr = [3, 2, 1, 5, 6, 4]
    print(f"    배열: {arr}")
    print(f"    2번째로 큰 수: {kth_largest(arr, 2)}")
    print(f"    2번째로 작은 수: {kth_smallest(arr, 2)}")

    # 6. 중앙값 스트림
    print("\n[6] 중앙값 스트림")
    mf = MedianFinder()
    stream = [2, 3, 4]
    print(f"    스트림: {stream}")
    for num in stream:
        mf.add_num(num)
        print(f"    {num} 추가 후 중앙값: {mf.find_median()}")

    # 7. K개 정렬 리스트 병합
    print("\n[7] K개 정렬 리스트 병합")
    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    merged = merge_k_sorted_lists(lists)
    print(f"    입력: {lists}")
    print(f"    병합: {merged}")

    # 8. Top K 빈도
    print("\n[8] Top K 빈도 요소")
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    result = top_k_frequent(nums, k)
    print(f"    배열: {nums}, k={k}")
    print(f"    결과: {result}")

    # 9. 가장 가까운 K개 점
    print("\n[9] 원점에서 가장 가까운 K개 점")
    points = [(1, 3), (-2, 2), (5, 8), (0, 1)]
    k = 2
    closest = k_closest_points(points, k)
    print(f"    점들: {points}, k={k}")
    print(f"    결과: {closest}")

    # 10. 회의실
    print("\n[10] 최소 회의실 수")
    meetings = [(0, 30), (5, 10), (15, 20)]
    rooms = min_meeting_rooms(meetings)
    print(f"    회의: {meetings}")
    print(f"    필요한 회의실: {rooms}개")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
