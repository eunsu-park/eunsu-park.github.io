"""
펜윅 트리 (Fenwick Tree / Binary Indexed Tree)
Fenwick Tree (BIT)

구간 합과 점 업데이트를 효율적으로 처리하는 자료구조입니다.
"""

from typing import List


# =============================================================================
# 1. 기본 펜윅 트리 (구간 합)
# =============================================================================

class FenwickTree:
    """
    펜윅 트리 (Binary Indexed Tree)
    - 점 업데이트: O(log n)
    - 접두사 합: O(log n)
    - 구간 합: O(log n)
    - 공간: O(n)
    """

    def __init__(self, n: int):
        """크기 n의 빈 펜윅 트리 생성"""
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed

    @classmethod
    def from_array(cls, arr: List[int]) -> 'FenwickTree':
        """배열로부터 펜윅 트리 생성 - O(n)"""
        ft = cls(len(arr))

        # 효율적인 구성 (O(n))
        for i, val in enumerate(arr):
            ft.tree[i + 1] += val
            parent = i + 1 + (ft._lowbit(i + 1))
            if parent <= ft.n:
                ft.tree[parent] += ft.tree[i + 1]

        return ft

    def _lowbit(self, x: int) -> int:
        """최하위 비트 (x & -x)"""
        return x & (-x)

    def update(self, idx: int, delta: int):
        """idx 위치에 delta 더하기 (0-indexed) - O(log n)"""
        idx += 1  # 1-indexed로 변환

        while idx <= self.n:
            self.tree[idx] += delta
            idx += self._lowbit(idx)

    def prefix_sum(self, idx: int) -> int:
        """[0, idx] 구간 합 (0-indexed) - O(log n)"""
        idx += 1
        result = 0

        while idx > 0:
            result += self.tree[idx]
            idx -= self._lowbit(idx)

        return result

    def range_sum(self, left: int, right: int) -> int:
        """[left, right] 구간 합 (0-indexed) - O(log n)"""
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

    def get(self, idx: int) -> int:
        """idx 위치의 값 (0-indexed)"""
        return self.range_sum(idx, idx)

    def set(self, idx: int, val: int):
        """idx 위치의 값을 val로 설정"""
        current = self.get(idx)
        self.update(idx, val - current)


# =============================================================================
# 2. 구간 업데이트 + 점 쿼리 펜윅 트리
# =============================================================================

class FenwickTreeRangeUpdate:
    """
    구간 업데이트 + 점 쿼리
    차분 배열 기법 활용
    """

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def _update(self, idx: int, delta: int):
        idx += 1
        while idx <= self.n:
            self.tree[idx] += delta
            idx += self._lowbit(idx)

    def update_range(self, left: int, right: int, delta: int):
        """[left, right] 구간에 delta 더하기"""
        self._update(left, delta)
        if right + 1 < self.n:
            self._update(right + 1, -delta)

    def query(self, idx: int) -> int:
        """idx 위치의 값"""
        idx += 1
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= self._lowbit(idx)
        return result


# =============================================================================
# 3. 구간 업데이트 + 구간 쿼리 펜윅 트리
# =============================================================================

class FenwickTreeRangeUpdateRangeQuery:
    """
    구간 업데이트 + 구간 쿼리
    두 개의 BIT 사용
    """

    def __init__(self, n: int):
        self.n = n
        self.tree1 = [0] * (n + 1)  # B[i]
        self.tree2 = [0] * (n + 1)  # B[i] * i

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def _update(self, tree: List[int], idx: int, delta: int):
        while idx <= self.n:
            tree[idx] += delta
            idx += self._lowbit(idx)

    def _prefix_sum(self, tree: List[int], idx: int) -> int:
        result = 0
        while idx > 0:
            result += tree[idx]
            idx -= self._lowbit(idx)
        return result

    def update_range(self, left: int, right: int, delta: int):
        """[left, right] 구간에 delta 더하기 (1-indexed)"""
        self._update(self.tree1, left, delta)
        self._update(self.tree1, right + 1, -delta)
        self._update(self.tree2, left, delta * (left - 1))
        self._update(self.tree2, right + 1, -delta * right)

    def prefix_sum(self, idx: int) -> int:
        """[1, idx] 구간 합 (1-indexed)"""
        return self._prefix_sum(self.tree1, idx) * idx - self._prefix_sum(self.tree2, idx)

    def range_sum(self, left: int, right: int) -> int:
        """[left, right] 구간 합 (1-indexed)"""
        return self.prefix_sum(right) - self.prefix_sum(left - 1)


# =============================================================================
# 4. 2D 펜윅 트리
# =============================================================================

class FenwickTree2D:
    """
    2D 펜윅 트리
    - 점 업데이트: O(log n * log m)
    - 직사각형 합: O(log n * log m)
    """

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.tree = [[0] * (m + 1) for _ in range(n + 1)]

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def update(self, x: int, y: int, delta: int):
        """(x, y)에 delta 더하기 (0-indexed)"""
        x += 1
        while x <= self.n:
            y_idx = y + 1
            while y_idx <= self.m:
                self.tree[x][y_idx] += delta
                y_idx += self._lowbit(y_idx)
            x += self._lowbit(x)

    def prefix_sum(self, x: int, y: int) -> int:
        """(0,0) ~ (x,y) 합 (0-indexed)"""
        x += 1
        result = 0
        while x > 0:
            y_idx = y + 1
            while y_idx > 0:
                result += self.tree[x][y_idx]
                y_idx -= self._lowbit(y_idx)
            x -= self._lowbit(x)
        return result

    def range_sum(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """(x1,y1) ~ (x2,y2) 직사각형 합 (0-indexed)"""
        result = self.prefix_sum(x2, y2)
        if x1 > 0:
            result -= self.prefix_sum(x1 - 1, y2)
        if y1 > 0:
            result -= self.prefix_sum(x2, y1 - 1)
        if x1 > 0 and y1 > 0:
            result += self.prefix_sum(x1 - 1, y1 - 1)
        return result


# =============================================================================
# 5. 역순 쌍 개수 (Inversion Count)
# =============================================================================

def count_inversions(arr: List[int]) -> int:
    """
    역순 쌍 개수 (펜윅 트리 활용)
    시간복잡도: O(n log n)
    """
    if not arr:
        return 0

    # 좌표 압축
    sorted_arr = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_arr)}
    n = len(sorted_arr)

    ft = FenwickTree(n)
    inversions = 0

    for val in arr:
        r = rank[val]
        # r보다 큰 인덱스의 개수 (이미 삽입된 것 중)
        inversions += ft.prefix_sum(n - 1) - ft.prefix_sum(r)
        ft.update(r, 1)

    return inversions


# =============================================================================
# 6. K번째 원소 찾기
# =============================================================================

class FenwickTreeKth:
    """K번째 원소 찾기를 지원하는 펜윅 트리"""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def update(self, idx: int, delta: int):
        """idx에 delta 더하기 (1-indexed)"""
        while idx <= self.n:
            self.tree[idx] += delta
            idx += self._lowbit(idx)

    def prefix_sum(self, idx: int) -> int:
        """[1, idx] 합"""
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= self._lowbit(idx)
        return result

    def find_kth(self, k: int) -> int:
        """
        k번째 원소의 인덱스 찾기 (1-indexed)
        prefix_sum(idx) >= k인 최소 idx
        시간복잡도: O(log n)
        """
        idx = 0
        bit_mask = 1

        while bit_mask <= self.n:
            bit_mask <<= 1
        bit_mask >>= 1

        while bit_mask > 0:
            next_idx = idx + bit_mask
            if next_idx <= self.n and self.tree[next_idx] < k:
                idx = next_idx
                k -= self.tree[idx]
            bit_mask >>= 1

        return idx + 1


# =============================================================================
# 7. 응용: 구간에서 K보다 작은 원소 개수
# =============================================================================

def count_smaller_in_range(arr: List[int], queries: List[tuple]) -> List[int]:
    """
    쿼리: (left, right, k) - arr[left:right+1]에서 k보다 작은 원소 개수
    오프라인 쿼리 + 펜윅 트리
    시간복잡도: O((n + q) log n)
    """
    n = len(arr)
    q = len(queries)

    # 좌표 압축
    all_vals = sorted(set(arr) | set(k for _, _, k in queries))
    val_to_idx = {v: i + 1 for i, v in enumerate(all_vals)}
    m = len(all_vals)

    # (값, 인덱스, 타입) 이벤트 생성
    events = []
    for i, val in enumerate(arr):
        events.append((val_to_idx[val], i, 'arr', None))

    for qi, (left, right, k) in enumerate(queries):
        k_idx = val_to_idx.get(k, m + 1)
        events.append((k_idx, right, 'query_end', (qi, left, right)))

    events.sort()

    # 결과
    results = [0] * q
    ft = FenwickTree(n)

    for val_idx, pos, event_type, data in events:
        if event_type == 'arr':
            ft.update(pos, 1)
        else:
            qi, left, right = data
            results[qi] = ft.range_sum(left, right)

    return results


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("펜윅 트리 (Fenwick Tree / BIT) 예제")
    print("=" * 60)

    # 1. 기본 펜윅 트리
    print("\n[1] 기본 펜윅 트리")
    arr = [1, 3, 5, 7, 9, 11]
    ft = FenwickTree.from_array(arr)
    print(f"    배열: {arr}")
    print(f"    prefix_sum(3): {ft.prefix_sum(3)}")  # 1+3+5+7=16
    print(f"    range_sum(1, 4): {ft.range_sum(1, 4)}")  # 3+5+7+9=24
    ft.update(2, 5)  # 5 → 10
    print(f"    update(2, +5) 후 range_sum(1, 4): {ft.range_sum(1, 4)}")  # 3+10+7+9=29

    # 2. 구간 업데이트 + 점 쿼리
    print("\n[2] 구간 업데이트 + 점 쿼리")
    ft_ru = FenwickTreeRangeUpdate(6)
    ft_ru.update_range(1, 3, 5)  # [1,3]에 5 더하기
    ft_ru.update_range(2, 4, 3)  # [2,4]에 3 더하기
    print(f"    update_range(1, 3, +5), update_range(2, 4, +3)")
    for i in range(6):
        print(f"    query({i}): {ft_ru.query(i)}")

    # 3. 2D 펜윅 트리
    print("\n[3] 2D 펜윅 트리")
    ft2d = FenwickTree2D(3, 3)
    # 행렬 채우기
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i in range(3):
        for j in range(3):
            ft2d.update(i, j, matrix[i][j])
    print(f"    행렬: {matrix}")
    print(f"    range_sum(0,0,1,1): {ft2d.range_sum(0, 0, 1, 1)}")  # 1+2+4+5=12
    print(f"    range_sum(1,1,2,2): {ft2d.range_sum(1, 1, 2, 2)}")  # 5+6+8+9=28

    # 4. 역순 쌍 개수
    print("\n[4] 역순 쌍 개수")
    arr = [2, 4, 1, 3, 5]
    inv = count_inversions(arr)
    print(f"    배열: {arr}")
    print(f"    역순 쌍: {inv}")  # (2,1), (4,1), (4,3) = 3

    # 5. K번째 원소
    print("\n[5] K번째 원소")
    ft_kth = FenwickTreeKth(10)
    for val in [3, 5, 7, 1, 9]:
        ft_kth.update(val, 1)
    print(f"    삽입된 원소: [3, 5, 7, 1, 9]")
    for k in [1, 2, 3, 4, 5]:
        print(f"    {k}번째 원소: {ft_kth.find_kth(k)}")

    # 6. 펜윅 트리 vs 세그먼트 트리 비교
    print("\n[6] 펜윅 트리 vs 세그먼트 트리")
    print("    | 특성         | 펜윅 트리 | 세그먼트 트리 |")
    print("    |--------------|-----------|---------------|")
    print("    | 공간         | O(n)      | O(4n)         |")
    print("    | 구현 난이도  | 쉬움      | 보통          |")
    print("    | 점 업데이트  | O(log n)  | O(log n)      |")
    print("    | 구간 쿼리    | O(log n)  | O(log n)      |")
    print("    | 구간 업데이트| 2개 BIT   | Lazy          |")
    print("    | 지원 연산    | 가역만    | 임의          |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
