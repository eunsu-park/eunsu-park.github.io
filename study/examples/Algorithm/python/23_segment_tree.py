"""
세그먼트 트리 (Segment Tree)
Segment Tree for Range Queries

구간 쿼리와 점 업데이트를 효율적으로 처리하는 자료구조입니다.
"""

from typing import List, Callable, Optional


# =============================================================================
# 1. 기본 세그먼트 트리 (구간 합)
# =============================================================================

class SegmentTree:
    """
    세그먼트 트리 (구간 합)
    - 점 업데이트: O(log n)
    - 구간 쿼리: O(log n)
    - 공간: O(n)
    """

    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: List[int], node: int, start: int, end: int):
        """트리 구성 - O(n)"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx: int, val: int):
        """점 업데이트 - O(log n)"""
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node: int, start: int, end: int, idx: int, val: int):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left: int, right: int) -> int:
        """구간 합 쿼리 - O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return 0  # 범위 벗어남
        if left <= start and end <= right:
            return self.tree[node]  # 완전 포함

        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, left, right)
        right_sum = self._query(2 * node + 1, mid + 1, end, left, right)
        return left_sum + right_sum


# =============================================================================
# 2. 일반 세그먼트 트리 (임의 연산)
# =============================================================================

class GenericSegmentTree:
    """
    일반 세그먼트 트리 (임의의 결합 연산)
    - 결합 법칙을 만족하는 연산이면 사용 가능
    """

    def __init__(self, arr: List[int], func: Callable[[int, int], int], identity: int):
        """
        func: 결합 연산 (예: min, max, gcd, +, *)
        identity: 항등원 (예: inf for min, 0 for +, 1 for *)
        """
        self.n = len(arr)
        self.func = func
        self.identity = identity
        self.tree = [identity] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: List[int], node: int, start: int, end: int):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.func(self.tree[2 * node], self.tree[2 * node + 1])

    def update(self, idx: int, val: int):
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node: int, start: int, end: int, idx: int, val: int):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = self.func(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left: int, right: int) -> int:
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return self.identity
        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        left_val = self._query(2 * node, start, mid, left, right)
        right_val = self._query(2 * node + 1, mid + 1, end, left, right)
        return self.func(left_val, right_val)


# =============================================================================
# 3. Lazy Propagation (구간 업데이트)
# =============================================================================

class LazySegmentTree:
    """
    Lazy Propagation 세그먼트 트리
    - 구간 업데이트: O(log n)
    - 구간 쿼리: O(log n)
    """

    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: List[int], node: int, start: int, end: int):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node: int, start: int, end: int):
        """lazy 값 전파"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2

            # 왼쪽 자식
            self.tree[2 * node] += self.lazy[node] * (mid - start + 1)
            self.lazy[2 * node] += self.lazy[node]

            # 오른쪽 자식
            self.tree[2 * node + 1] += self.lazy[node] * (end - mid)
            self.lazy[2 * node + 1] += self.lazy[node]

            self.lazy[node] = 0

    def update_range(self, left: int, right: int, val: int):
        """구간 [left, right]에 val 더하기"""
        self._update_range(1, 0, self.n - 1, left, right, val)

    def _update_range(self, node: int, start: int, end: int, left: int, right: int, val: int):
        if right < start or end < left:
            return

        if left <= start and end <= right:
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return

        self._push_down(node, start, end)

        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left: int, right: int) -> int:
        """구간 합 쿼리"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return 0

        if left <= start and end <= right:
            return self.tree[node]

        self._push_down(node, start, end)

        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, left, right)
        right_sum = self._query(2 * node + 1, mid + 1, end, left, right)
        return left_sum + right_sum


# =============================================================================
# 4. 반복 세그먼트 트리 (Iterative)
# =============================================================================

class IterativeSegmentTree:
    """
    반복 세그먼트 트리 (비재귀)
    메모리 효율적, 캐시 친화적
    """

    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.tree = [0] * (2 * self.n)

        # 리프 노드 채우기
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]

        # 내부 노드 구성
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx: int, val: int):
        """점 업데이트"""
        idx += self.n
        self.tree[idx] = val

        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def query(self, left: int, right: int) -> int:
        """구간 [left, right] 합"""
        left += self.n
        right += self.n + 1
        result = 0

        while left < right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                result += self.tree[right]
            left //= 2
            right //= 2

        return result


# =============================================================================
# 5. 2D 세그먼트 트리
# =============================================================================

class SegmentTree2D:
    """
    2D 세그먼트 트리 (구간 합)
    - 쿼리/업데이트: O(log n * log m)
    """

    def __init__(self, matrix: List[List[int]]):
        if not matrix or not matrix[0]:
            self.n = self.m = 0
            return

        self.n = len(matrix)
        self.m = len(matrix[0])
        self.tree = [[0] * (4 * self.m) for _ in range(4 * self.n)]
        self._build_x(matrix, 1, 0, self.n - 1)

    def _build_x(self, matrix: List[List[int]], node_x: int, lx: int, rx: int):
        if lx == rx:
            self._build_y(matrix, node_x, lx, rx, 1, 0, self.m - 1, lx)
        else:
            mid = (lx + rx) // 2
            self._build_x(matrix, 2 * node_x, lx, mid)
            self._build_x(matrix, 2 * node_x + 1, mid + 1, rx)
            self._merge_y(node_x, 1, 0, self.m - 1)

    def _build_y(self, matrix, node_x, lx, rx, node_y, ly, ry, row):
        if ly == ry:
            self.tree[node_x][node_y] = matrix[row][ly]
        else:
            mid = (ly + ry) // 2
            self._build_y(matrix, node_x, lx, rx, 2 * node_y, ly, mid, row)
            self._build_y(matrix, node_x, lx, rx, 2 * node_y + 1, mid + 1, ry, row)
            self.tree[node_x][node_y] = self.tree[node_x][2 * node_y] + self.tree[node_x][2 * node_y + 1]

    def _merge_y(self, node_x, node_y, ly, ry):
        if ly == ry:
            self.tree[node_x][node_y] = self.tree[2 * node_x][node_y] + self.tree[2 * node_x + 1][node_y]
        else:
            mid = (ly + ry) // 2
            self._merge_y(node_x, 2 * node_y, ly, mid)
            self._merge_y(node_x, 2 * node_y + 1, mid + 1, ry)
            self.tree[node_x][node_y] = self.tree[node_x][2 * node_y] + self.tree[node_x][2 * node_y + 1]

    def query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """(x1,y1) ~ (x2,y2) 직사각형 구간 합"""
        return self._query_x(1, 0, self.n - 1, x1, x2, y1, y2)

    def _query_x(self, node_x, lx, rx, x1, x2, y1, y2):
        if x2 < lx or rx < x1:
            return 0
        if x1 <= lx and rx <= x2:
            return self._query_y(node_x, 1, 0, self.m - 1, y1, y2)

        mid = (lx + rx) // 2
        left = self._query_x(2 * node_x, lx, mid, x1, x2, y1, y2)
        right = self._query_x(2 * node_x + 1, mid + 1, rx, x1, x2, y1, y2)
        return left + right

    def _query_y(self, node_x, node_y, ly, ry, y1, y2):
        if y2 < ly or ry < y1:
            return 0
        if y1 <= ly and ry <= y2:
            return self.tree[node_x][node_y]

        mid = (ly + ry) // 2
        left = self._query_y(node_x, 2 * node_y, ly, mid, y1, y2)
        right = self._query_y(node_x, 2 * node_y + 1, mid + 1, ry, y1, y2)
        return left + right


# =============================================================================
# 6. 응용: 역순 쌍 개수 (Inversion Count)
# =============================================================================

def count_inversions_segtree(arr: List[int]) -> int:
    """
    역순 쌍 개수 (세그먼트 트리 활용)
    시간복잡도: O(n log n)
    """
    if not arr:
        return 0

    # 좌표 압축
    sorted_arr = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_arr)}
    n = len(sorted_arr)

    # 세그먼트 트리 (빈도 저장)
    tree = [0] * (4 * n)

    def update(node, start, end, idx):
        if start == end:
            tree[node] += 1
        else:
            mid = (start + end) // 2
            if idx <= mid:
                update(2 * node, start, mid, idx)
            else:
                update(2 * node + 1, mid + 1, end, idx)
            tree[node] = tree[2 * node] + tree[2 * node + 1]

    def query(node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return tree[node]
        mid = (start + end) // 2
        return query(2 * node, start, mid, left, right) + \
               query(2 * node + 1, mid + 1, end, left, right)

    inversions = 0
    for val in arr:
        r = rank[val]
        # r보다 큰 값의 개수 (이미 삽입된 것 중)
        inversions += query(1, 0, n - 1, r + 1, n - 1)
        update(1, 0, n - 1, r)

    return inversions


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("세그먼트 트리 (Segment Tree) 예제")
    print("=" * 60)

    # 1. 기본 세그먼트 트리
    print("\n[1] 기본 세그먼트 트리 (구간 합)")
    arr = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(arr)
    print(f"    배열: {arr}")
    print(f"    query(1, 3): {st.query(1, 3)}")  # 3+5+7=15
    st.update(2, 6)  # 5 -> 6
    print(f"    update(2, 6) 후 query(1, 3): {st.query(1, 3)}")  # 3+6+7=16

    # 2. 일반 세그먼트 트리 (최소값)
    print("\n[2] 일반 세그먼트 트리 (구간 최소)")
    arr = [5, 2, 8, 1, 9, 3]
    min_st = GenericSegmentTree(arr, min, float('inf'))
    print(f"    배열: {arr}")
    print(f"    min(1, 4): {min_st.query(1, 4)}")  # min(2,8,1,9)=1
    min_st.update(3, 10)  # 1 -> 10
    print(f"    update(3, 10) 후 min(1, 4): {min_st.query(1, 4)}")  # min(2,8,10,9)=2

    # 3. Lazy Propagation
    print("\n[3] Lazy Propagation (구간 업데이트)")
    arr = [1, 2, 3, 4, 5]
    lazy_st = LazySegmentTree(arr)
    print(f"    배열: {arr}")
    print(f"    query(0, 4): {lazy_st.query(0, 4)}")  # 15
    lazy_st.update_range(1, 3, 10)  # [1,3] 구간에 10 더하기
    print(f"    update_range(1, 3, +10) 후 query(0, 4): {lazy_st.query(0, 4)}")  # 45

    # 4. 반복 세그먼트 트리
    print("\n[4] 반복 세그먼트 트리")
    arr = [1, 3, 5, 7, 9, 11]
    iter_st = IterativeSegmentTree(arr)
    print(f"    배열: {arr}")
    print(f"    query(1, 4): {iter_st.query(1, 4)}")  # 3+5+7+9=24

    # 5. 2D 세그먼트 트리
    print("\n[5] 2D 세그먼트 트리")
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    st2d = SegmentTree2D(matrix)
    print(f"    행렬: {matrix}")
    print(f"    query(0,0,1,1): {st2d.query(0, 0, 1, 1)}")  # 1+2+4+5=12
    print(f"    query(1,1,2,2): {st2d.query(1, 1, 2, 2)}")  # 5+6+8+9=28

    # 6. 역순 쌍 개수
    print("\n[6] 역순 쌍 개수")
    arr = [2, 4, 1, 3, 5]
    inv = count_inversions_segtree(arr)
    print(f"    배열: {arr}")
    print(f"    역순 쌍 개수: {inv}")  # (2,1), (4,1), (4,3) = 3

    # 7. 복잡도 비교
    print("\n[7] 복잡도 비교")
    print("    | 연산       | 배열    | 세그먼트 트리 | Lazy      |")
    print("    |------------|---------|---------------|-----------|")
    print("    | 점 업데이트| O(1)    | O(log n)      | O(log n)  |")
    print("    | 구간 업데이트| O(n)  | O(n)          | O(log n)  |")
    print("    | 구간 쿼리  | O(n)    | O(log n)      | O(log n)  |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
