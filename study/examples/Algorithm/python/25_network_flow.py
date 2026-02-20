"""
네트워크 플로우 (Network Flow)
Maximum Flow and Bipartite Matching

그래프에서 최대 유량을 구하는 알고리즘입니다.
"""

from typing import List, Tuple, Dict
from collections import defaultdict, deque


# =============================================================================
# 1. Ford-Fulkerson (DFS 기반)
# =============================================================================

def ford_fulkerson(capacity: List[List[int]], source: int, sink: int) -> int:
    """
    Ford-Fulkerson 알고리즘 (DFS)
    시간복잡도: O(E * max_flow)
    """
    n = len(capacity)
    residual = [row[:] for row in capacity]

    def dfs(node: int, flow: int, visited: List[bool]) -> int:
        if node == sink:
            return flow

        visited[node] = True

        for next_node in range(n):
            if not visited[next_node] and residual[node][next_node] > 0:
                min_flow = min(flow, residual[node][next_node])
                result = dfs(next_node, min_flow, visited)

                if result > 0:
                    residual[node][next_node] -= result
                    residual[next_node][node] += result
                    return result

        return 0

    max_flow = 0
    while True:
        visited = [False] * n
        flow = dfs(source, float('inf'), visited)
        if flow == 0:
            break
        max_flow += flow

    return max_flow


# =============================================================================
# 2. Edmonds-Karp (BFS 기반)
# =============================================================================

def edmonds_karp(capacity: List[List[int]], source: int, sink: int) -> int:
    """
    Edmonds-Karp 알고리즘 (BFS)
    시간복잡도: O(V * E²)
    """
    n = len(capacity)
    residual = [row[:] for row in capacity]

    def bfs() -> List[int]:
        """BFS로 증가 경로 찾기"""
        parent = [-1] * n
        visited = [False] * n
        visited[source] = True
        queue = deque([source])

        while queue:
            node = queue.popleft()

            for next_node in range(n):
                if not visited[next_node] and residual[node][next_node] > 0:
                    visited[next_node] = True
                    parent[next_node] = node
                    queue.append(next_node)

                    if next_node == sink:
                        return parent

        return parent

    max_flow = 0

    while True:
        parent = bfs()

        if parent[sink] == -1:
            break

        # 경로의 최소 용량 찾기
        path_flow = float('inf')
        node = sink
        while node != source:
            prev = parent[node]
            path_flow = min(path_flow, residual[prev][node])
            node = prev

        # 잔여 그래프 업데이트
        node = sink
        while node != source:
            prev = parent[node]
            residual[prev][node] -= path_flow
            residual[node][prev] += path_flow
            node = prev

        max_flow += path_flow

    return max_flow


# =============================================================================
# 3. Dinic 알고리즘
# =============================================================================

class Dinic:
    """
    Dinic 알고리즘
    시간복잡도: O(V² * E)
    이분 그래프에서: O(E * √V)
    """

    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)

    def add_edge(self, u: int, v: int, cap: int):
        """간선 추가 (u → v, 용량 cap)"""
        # (인접 노드, 잔여 용량, 역방향 간선 인덱스)
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def bfs(self, source: int, sink: int) -> bool:
        """레벨 그래프 구성"""
        self.level = [-1] * self.n
        self.level[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for next_node, cap, _ in self.graph[node]:
                if cap > 0 and self.level[next_node] < 0:
                    self.level[next_node] = self.level[node] + 1
                    queue.append(next_node)

        return self.level[sink] >= 0

    def dfs(self, node: int, sink: int, flow: int) -> int:
        """blocking flow 찾기"""
        if node == sink:
            return flow

        for i in range(self.iter[node], len(self.graph[node])):
            self.iter[node] = i
            next_node, cap, rev = self.graph[node][i]

            if cap > 0 and self.level[next_node] == self.level[node] + 1:
                d = self.dfs(next_node, sink, min(flow, cap))

                if d > 0:
                    self.graph[node][i][1] -= d
                    self.graph[next_node][rev][1] += d
                    return d

        return 0

    def max_flow(self, source: int, sink: int) -> int:
        """최대 유량 계산"""
        flow = 0

        while self.bfs(source, sink):
            self.iter = [0] * self.n

            while True:
                f = self.dfs(source, sink, float('inf'))
                if f == 0:
                    break
                flow += f

        return flow


# =============================================================================
# 4. 이분 매칭 (Bipartite Matching)
# =============================================================================

def bipartite_matching(n: int, m: int, edges: List[Tuple[int, int]]) -> int:
    """
    이분 그래프 최대 매칭 (Hopcroft-Karp 간소화)
    n: 왼쪽 정점 수, m: 오른쪽 정점 수
    edges: [(왼쪽, 오른쪽), ...]
    시간복잡도: O(E * √V)
    """
    # 그래프 구성
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    match_left = [-1] * n   # 왼쪽 정점의 매칭
    match_right = [-1] * m  # 오른쪽 정점의 매칭

    def dfs(u: int, visited: List[bool]) -> bool:
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True

            # 매칭 안됨 또는 재매칭 가능
            if match_right[v] == -1 or dfs(match_right[v], visited):
                match_left[u] = v
                match_right[v] = u
                return True

        return False

    matching = 0
    for u in range(n):
        visited = [False] * m
        if dfs(u, visited):
            matching += 1

    return matching


# =============================================================================
# 5. 최소 컷 (Minimum Cut)
# =============================================================================

def min_cut(capacity: List[List[int]], source: int, sink: int) -> Tuple[int, List[int]]:
    """
    최소 컷 = 최대 유량
    반환: (컷 용량, source 측 정점 집합)
    """
    n = len(capacity)
    residual = [row[:] for row in capacity]

    # Edmonds-Karp로 최대 유량 계산
    def bfs_flow() -> bool:
        parent = [-1] * n
        visited = [False] * n
        visited[source] = True
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for next_node in range(n):
                if not visited[next_node] and residual[node][next_node] > 0:
                    visited[next_node] = True
                    parent[next_node] = node
                    queue.append(next_node)

        if parent[sink] == -1:
            return False

        path_flow = float('inf')
        node = sink
        while node != source:
            prev = parent[node]
            path_flow = min(path_flow, residual[prev][node])
            node = prev

        node = sink
        while node != source:
            prev = parent[node]
            residual[prev][node] -= path_flow
            residual[node][prev] += path_flow
            node = prev

        return True

    while bfs_flow():
        pass

    # 잔여 그래프에서 source에서 도달 가능한 정점 찾기
    visited = [False] * n
    queue = deque([source])
    visited[source] = True

    while queue:
        node = queue.popleft()
        for next_node in range(n):
            if not visited[next_node] and residual[node][next_node] > 0:
                visited[next_node] = True
                queue.append(next_node)

    # 컷 용량 계산
    cut_capacity = 0
    source_side = []
    for i in range(n):
        if visited[i]:
            source_side.append(i)
            for j in range(n):
                if not visited[j] and capacity[i][j] > 0:
                    cut_capacity += capacity[i][j]

    return cut_capacity, source_side


# =============================================================================
# 6. 실전 문제: 작업 할당
# =============================================================================

def assign_tasks(workers: int, tasks: int, can_do: List[List[int]]) -> List[int]:
    """
    작업자에게 작업 할당 (이분 매칭)
    can_do[i] = worker i가 수행 가능한 작업 목록
    반환: assignment[worker] = 할당된 작업 (-1이면 미할당)
    """
    edges = []
    for worker, tasks_list in enumerate(can_do):
        for task in tasks_list:
            edges.append((worker, task))

    # 이분 매칭 수행
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    match_worker = [-1] * workers
    match_task = [-1] * tasks

    def dfs(u: int, visited: List[bool]) -> bool:
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True

            if match_task[v] == -1 or dfs(match_task[v], visited):
                match_worker[u] = v
                match_task[v] = u
                return True

        return False

    for u in range(workers):
        visited = [False] * tasks
        dfs(u, visited)

    return match_worker


# =============================================================================
# 7. 실전 문제: 최대 간선 분리 경로
# =============================================================================

def max_edge_disjoint_paths(n: int, edges: List[Tuple[int, int]], source: int, sink: int) -> int:
    """
    간선 분리 경로의 최대 개수
    각 간선 용량 = 1로 설정한 최대 유량
    """
    capacity = [[0] * n for _ in range(n)]

    for u, v in edges:
        capacity[u][v] = 1
        capacity[v][u] = 1  # 무방향 그래프인 경우

    return edmonds_karp(capacity, source, sink)


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("네트워크 플로우 (Network Flow) 예제")
    print("=" * 60)

    # 그래프 예시:
    #     10      10
    #  0 ───→ 1 ───→ 3
    #  │      │      ↑
    #  │10    │2     │10
    #  ↓      ↓      │
    #  2 ───→ 4 ───→ 5
    #     10      10

    # 1. Ford-Fulkerson
    print("\n[1] Ford-Fulkerson")
    capacity = [
        [0, 10, 10, 0, 0, 0],  # 0
        [0, 0, 2, 10, 0, 0],   # 1
        [0, 0, 0, 0, 10, 0],   # 2
        [0, 0, 0, 0, 0, 10],   # 3
        [0, 0, 0, 0, 0, 10],   # 4
        [0, 0, 0, 0, 0, 0]     # 5 (sink)
    ]
    flow = ford_fulkerson(capacity, 0, 5)
    print(f"    source=0, sink=5")
    print(f"    최대 유량: {flow}")

    # 2. Edmonds-Karp
    print("\n[2] Edmonds-Karp")
    flow = edmonds_karp(capacity, 0, 5)
    print(f"    최대 유량: {flow}")

    # 3. Dinic
    print("\n[3] Dinic 알고리즘")
    dinic = Dinic(6)
    dinic.add_edge(0, 1, 10)
    dinic.add_edge(0, 2, 10)
    dinic.add_edge(1, 2, 2)
    dinic.add_edge(1, 3, 10)
    dinic.add_edge(2, 4, 10)
    dinic.add_edge(3, 5, 10)
    dinic.add_edge(4, 5, 10)
    flow = dinic.max_flow(0, 5)
    print(f"    최대 유량: {flow}")

    # 4. 이분 매칭
    print("\n[4] 이분 매칭")
    # 왼쪽: 0, 1, 2 (작업자)
    # 오른쪽: 0, 1, 2 (작업)
    edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)]
    matching = bipartite_matching(3, 3, edges)
    print(f"    간선: {edges}")
    print(f"    최대 매칭: {matching}")

    # 5. 최소 컷
    print("\n[5] 최소 컷")
    cut_cap, source_side = min_cut(capacity, 0, 5)
    print(f"    최소 컷 용량: {cut_cap}")
    print(f"    source 측 정점: {source_side}")

    # 6. 작업 할당
    print("\n[6] 작업 할당")
    can_do = [
        [0, 1],     # 작업자 0: 작업 0, 1 가능
        [1, 2],     # 작업자 1: 작업 1, 2 가능
        [0, 2]      # 작업자 2: 작업 0, 2 가능
    ]
    assignment = assign_tasks(3, 3, can_do)
    print(f"    수행 가능: {can_do}")
    print(f"    할당 결과: {assignment}")

    # 7. 알고리즘 비교
    print("\n[7] 알고리즘 복잡도 비교")
    print("    | 알고리즘      | 시간 복잡도       | 특징               |")
    print("    |---------------|-------------------|-------------------|")
    print("    | Ford-Fulkerson| O(E * max_flow)   | DFS, 정수 용량     |")
    print("    | Edmonds-Karp  | O(V * E²)         | BFS, 안정적        |")
    print("    | Dinic         | O(V² * E)         | 레벨 그래프        |")
    print("    | 이분 매칭     | O(E * √V)         | Dinic 특수 케이스  |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
