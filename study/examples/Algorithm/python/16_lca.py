"""
최소 공통 조상 (LCA - Lowest Common Ancestor)
LCA and Tree Queries

트리에서 두 노드의 최소 공통 조상을 찾는 알고리즘입니다.
"""

from typing import List, Tuple, Optional
from collections import defaultdict, deque
import math


# =============================================================================
# 1. 기본 LCA (Naive)
# =============================================================================

def lca_naive(n: int, edges: List[Tuple[int, int]], u: int, v: int) -> int:
    """
    기본 LCA (높이 맞추기)
    시간복잡도: O(n) per query
    전처리: O(n)
    """
    # 트리 구성
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # 부모와 깊이 계산
    parent = [-1] * n
    depth = [0] * n

    def dfs(node: int, par: int, d: int):
        parent[node] = par
        depth[node] = d
        for child in adj[node]:
            if child != par:
                dfs(child, node, d + 1)

    dfs(0, -1, 0)

    # 높이 맞추기
    while depth[u] > depth[v]:
        u = parent[u]
    while depth[v] > depth[u]:
        v = parent[v]

    # 동시에 올라가기
    while u != v:
        u = parent[u]
        v = parent[v]

    return u


# =============================================================================
# 2. Binary Lifting (희소 테이블)
# =============================================================================

class LCABinaryLifting:
    """
    Binary Lifting을 이용한 LCA
    전처리: O(n log n)
    쿼리: O(log n)
    """

    def __init__(self, n: int, edges: List[Tuple[int, int]], root: int = 0):
        self.n = n
        self.LOG = max(1, int(math.log2(n)) + 1)

        # 그래프 구성
        self.adj = defaultdict(list)
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # 전처리
        self.parent = [[-1] * n for _ in range(self.LOG)]
        self.depth = [0] * n

        self._preprocess(root)

    def _preprocess(self, root: int):
        """DFS로 부모/깊이 계산 + 희소 테이블 구성"""
        stack = [(root, -1, 0)]

        while stack:
            node, par, d = stack.pop()
            self.parent[0][node] = par
            self.depth[node] = d

            for child in self.adj[node]:
                if child != par:
                    stack.append((child, node, d + 1))

        # 희소 테이블 구성: parent[i][v] = v의 2^i번째 조상
        for i in range(1, self.LOG):
            for v in range(self.n):
                if self.parent[i - 1][v] != -1:
                    self.parent[i][v] = self.parent[i - 1][self.parent[i - 1][v]]

    def query(self, u: int, v: int) -> int:
        """LCA 쿼리 - O(log n)"""
        # u가 더 깊도록 조정
        if self.depth[u] < self.depth[v]:
            u, v = v, u

        # 높이 맞추기
        diff = self.depth[u] - self.depth[v]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u = self.parent[i][u]

        # 같으면 완료
        if u == v:
            return u

        # 동시에 올라가기
        for i in range(self.LOG - 1, -1, -1):
            if self.parent[i][u] != self.parent[i][v]:
                u = self.parent[i][u]
                v = self.parent[i][v]

        return self.parent[0][u]

    def kth_ancestor(self, node: int, k: int) -> int:
        """k번째 조상 찾기 - O(log n)"""
        for i in range(self.LOG):
            if node == -1:
                break
            if (k >> i) & 1:
                node = self.parent[i][node]
        return node

    def distance(self, u: int, v: int) -> int:
        """두 노드 사이 거리 - O(log n)"""
        lca = self.query(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca]


# =============================================================================
# 3. Euler Tour + RMQ (Sparse Table)
# =============================================================================

class LCAEulerTour:
    """
    오일러 경로 + RMQ를 이용한 LCA
    전처리: O(n log n)
    쿼리: O(1)
    """

    def __init__(self, n: int, edges: List[Tuple[int, int]], root: int = 0):
        self.n = n
        self.adj = defaultdict(list)
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # 오일러 경로 및 첫 등장 위치
        self.euler = []  # (깊이, 노드) 쌍
        self.first = [-1] * n  # 각 노드의 첫 등장 인덱스

        self._build_euler_tour(root)
        self._build_sparse_table()

    def _build_euler_tour(self, root: int):
        """오일러 경로 구성 - O(n)"""
        stack = [(root, -1, 0, False)]

        while stack:
            node, parent, depth, visited = stack.pop()

            self.euler.append((depth, node))
            if self.first[node] == -1:
                self.first[node] = len(self.euler) - 1

            if visited:
                continue

            stack.append((node, parent, depth, True))
            for child in self.adj[node]:
                if child != parent:
                    stack.append((child, node, depth + 1, False))

    def _build_sparse_table(self):
        """희소 테이블 구성 - O(n log n)"""
        m = len(self.euler)
        self.LOG = max(1, int(math.log2(m)) + 1)

        # sparse[i][j] = euler[j..j+2^i) 구간의 최솟값 인덱스
        self.sparse = [[0] * m for _ in range(self.LOG)]

        for j in range(m):
            self.sparse[0][j] = j

        for i in range(1, self.LOG):
            length = 1 << i
            for j in range(m - length + 1):
                left = self.sparse[i - 1][j]
                right = self.sparse[i - 1][j + (length >> 1)]
                if self.euler[left][0] <= self.euler[right][0]:
                    self.sparse[i][j] = left
                else:
                    self.sparse[i][j] = right

    def _rmq(self, left: int, right: int) -> int:
        """범위 최소 쿼리 - O(1)"""
        length = right - left + 1
        k = int(math.log2(length))
        left_idx = self.sparse[k][left]
        right_idx = self.sparse[k][right - (1 << k) + 1]
        if self.euler[left_idx][0] <= self.euler[right_idx][0]:
            return left_idx
        return right_idx

    def query(self, u: int, v: int) -> int:
        """LCA 쿼리 - O(1)"""
        left = self.first[u]
        right = self.first[v]
        if left > right:
            left, right = right, left
        idx = self._rmq(left, right)
        return self.euler[idx][1]


# =============================================================================
# 4. 트리에서 경로 합/최대/최소
# =============================================================================

class TreePathQuery:
    """트리 경로 쿼리 (LCA + 가중치)"""

    def __init__(self, n: int, edges: List[Tuple[int, int, int]], root: int = 0):
        """edges: [(u, v, weight), ...]"""
        self.n = n
        self.LOG = max(1, int(math.log2(n)) + 1)

        self.adj = defaultdict(list)
        for a, b, w in edges:
            self.adj[a].append((b, w))
            self.adj[b].append((a, w))

        self.parent = [[-1] * n for _ in range(self.LOG)]
        self.depth = [0] * n
        self.dist_from_root = [0] * n  # 루트로부터의 거리
        self.max_edge = [[0] * n for _ in range(self.LOG)]  # 경로상 최대 간선

        self._preprocess(root)

    def _preprocess(self, root: int):
        stack = [(root, -1, 0, 0)]

        while stack:
            node, par, d, dist = stack.pop()
            self.parent[0][node] = par
            self.depth[node] = d
            self.dist_from_root[node] = dist

            for child, weight in self.adj[node]:
                if child != par:
                    self.max_edge[0][child] = weight
                    stack.append((child, node, d + 1, dist + weight))

        # 희소 테이블
        for i in range(1, self.LOG):
            for v in range(self.n):
                if self.parent[i - 1][v] != -1:
                    self.parent[i][v] = self.parent[i - 1][self.parent[i - 1][v]]
                    self.max_edge[i][v] = max(
                        self.max_edge[i - 1][v],
                        self.max_edge[i - 1][self.parent[i - 1][v]] if self.parent[i - 1][v] != -1 else 0
                    )

    def lca(self, u: int, v: int) -> int:
        """LCA 쿼리"""
        if self.depth[u] < self.depth[v]:
            u, v = v, u

        diff = self.depth[u] - self.depth[v]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u = self.parent[i][u]

        if u == v:
            return u

        for i in range(self.LOG - 1, -1, -1):
            if self.parent[i][u] != self.parent[i][v]:
                u = self.parent[i][u]
                v = self.parent[i][v]

        return self.parent[0][u]

    def path_distance(self, u: int, v: int) -> int:
        """경로 거리 합"""
        ancestor = self.lca(u, v)
        return self.dist_from_root[u] + self.dist_from_root[v] - 2 * self.dist_from_root[ancestor]

    def path_max_edge(self, u: int, v: int) -> int:
        """경로상 최대 간선 가중치"""
        ancestor = self.lca(u, v)
        result = 0

        # u → lca
        curr = u
        diff = self.depth[u] - self.depth[ancestor]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                result = max(result, self.max_edge[i][curr])
                curr = self.parent[i][curr]

        # v → lca
        curr = v
        diff = self.depth[v] - self.depth[ancestor]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                result = max(result, self.max_edge[i][curr])
                curr = self.parent[i][curr]

        return result


# =============================================================================
# 5. 실전 문제: 트리에서 두 노드 사이 경로
# =============================================================================

def find_path(n: int, edges: List[Tuple[int, int]], u: int, v: int) -> List[int]:
    """두 노드 사이의 경로 찾기"""
    lca_solver = LCABinaryLifting(n, edges)
    ancestor = lca_solver.query(u, v)

    # u → lca
    path_u = []
    curr = u
    while curr != ancestor:
        path_u.append(curr)
        curr = lca_solver.parent[0][curr]
    path_u.append(ancestor)

    # v → lca (역순)
    path_v = []
    curr = v
    while curr != ancestor:
        path_v.append(curr)
        curr = lca_solver.parent[0][curr]

    return path_u + path_v[::-1]


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("최소 공통 조상 (LCA) 예제")
    print("=" * 60)

    # 트리 구성
    #        0
    #      / | \
    #     1  2  3
    #    / \    |
    #   4   5   6
    #  /
    # 7

    n = 8
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6), (4, 7)]

    # 1. 기본 LCA
    print("\n[1] 기본 LCA (Naive)")
    lca = lca_naive(n, edges, 7, 5)
    print(f"    LCA(7, 5) = {lca}")
    lca = lca_naive(n, edges, 7, 6)
    print(f"    LCA(7, 6) = {lca}")

    # 2. Binary Lifting
    print("\n[2] Binary Lifting")
    lca_bl = LCABinaryLifting(n, edges)
    print(f"    LCA(7, 5) = {lca_bl.query(7, 5)}")
    print(f"    LCA(7, 6) = {lca_bl.query(7, 6)}")
    print(f"    LCA(4, 6) = {lca_bl.query(4, 6)}")
    print(f"    거리(7, 5) = {lca_bl.distance(7, 5)}")
    print(f"    7의 2번째 조상 = {lca_bl.kth_ancestor(7, 2)}")

    # 3. Euler Tour + RMQ
    print("\n[3] Euler Tour + RMQ (O(1) 쿼리)")
    lca_euler = LCAEulerTour(n, edges)
    print(f"    LCA(7, 5) = {lca_euler.query(7, 5)}")
    print(f"    LCA(7, 6) = {lca_euler.query(7, 6)}")

    # 4. 가중치 트리 경로 쿼리
    print("\n[4] 가중치 트리 경로 쿼리")
    weighted_edges = [
        (0, 1, 3), (0, 2, 5), (0, 3, 4),
        (1, 4, 2), (1, 5, 6), (3, 6, 1), (4, 7, 8)
    ]
    path_query = TreePathQuery(n, weighted_edges)
    print(f"    경로 거리(7, 5) = {path_query.path_distance(7, 5)}")
    print(f"    경로 최대 간선(7, 5) = {path_query.path_max_edge(7, 5)}")
    print(f"    경로 거리(7, 6) = {path_query.path_distance(7, 6)}")

    # 5. 경로 찾기
    print("\n[5] 두 노드 사이 경로")
    path = find_path(n, edges, 7, 6)
    print(f"    경로(7, 6) = {path}")
    path = find_path(n, edges, 5, 2)
    print(f"    경로(5, 2) = {path}")

    # 6. 성능 비교
    print("\n[6] 복잡도 비교")
    print("    | 방법           | 전처리     | 쿼리    |")
    print("    |----------------|------------|---------|")
    print("    | Naive          | O(n)       | O(n)    |")
    print("    | Binary Lifting | O(n log n) | O(log n)|")
    print("    | Euler + RMQ    | O(n log n) | O(1)    |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
