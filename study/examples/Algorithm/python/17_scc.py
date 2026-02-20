"""
강한 연결 요소 (SCC - Strongly Connected Components)
Strongly Connected Components

방향 그래프에서 서로 도달 가능한 정점들의 최대 집합을 찾습니다.
"""

from typing import List, Tuple, Set
from collections import defaultdict


# =============================================================================
# 1. Kosaraju 알고리즘
# =============================================================================

def kosaraju_scc(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Kosaraju 알고리즘으로 SCC 찾기
    시간복잡도: O(V + E)
    공간복잡도: O(V + E)

    1. 정방향 DFS로 종료 순서 스택 구성
    2. 역방향 그래프에서 스택 순서대로 DFS
    """
    # 정방향/역방향 그래프
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)

    for u, v in edges:
        graph[u].append(v)
        reverse_graph[v].append(u)

    # 1단계: 정방향 DFS로 종료 순서 기록
    visited = [False] * n
    finish_stack = []

    def dfs1(node: int):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs1(neighbor)
        finish_stack.append(node)

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    # 2단계: 역방향 DFS로 SCC 찾기
    visited = [False] * n
    sccs = []

    def dfs2(node: int, component: List[int]):
        visited[node] = True
        component.append(node)
        for neighbor in reverse_graph[node]:
            if not visited[neighbor]:
                dfs2(neighbor, component)

    while finish_stack:
        node = finish_stack.pop()
        if not visited[node]:
            component = []
            dfs2(node, component)
            sccs.append(component)

    return sccs


# =============================================================================
# 2. Tarjan 알고리즘
# =============================================================================

def tarjan_scc(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Tarjan 알고리즘으로 SCC 찾기
    시간복잡도: O(V + E)
    공간복잡도: O(V)

    low[v] = v에서 도달 가능한 최소 발견 시간
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    disc = [-1] * n  # 발견 시간
    low = [-1] * n   # low-link 값
    on_stack = [False] * n
    stack = []
    sccs = []
    time = [0]  # 전역 시간

    def dfs(node: int):
        disc[node] = low[node] = time[0]
        time[0] += 1
        stack.append(node)
        on_stack[node] = True

        for neighbor in graph[node]:
            if disc[neighbor] == -1:  # 미방문
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])
            elif on_stack[neighbor]:  # 스택에 있음 (back edge)
                low[node] = min(low[node], disc[neighbor])

        # SCC 루트 발견
        if low[node] == disc[node]:
            component = []
            while True:
                v = stack.pop()
                on_stack[v] = False
                component.append(v)
                if v == node:
                    break
            sccs.append(component)

    for i in range(n):
        if disc[i] == -1:
            dfs(i)

    return sccs


# =============================================================================
# 3. SCC 축약 그래프 (DAG)
# =============================================================================

def build_scc_dag(n: int, edges: List[Tuple[int, int]]) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """
    SCC를 찾고 축약 그래프(DAG) 구성
    반환: (sccs, scc_graph, node_to_scc)
    """
    sccs = tarjan_scc(n, edges)

    # 각 노드가 속한 SCC 매핑
    node_to_scc = [-1] * n
    for i, component in enumerate(sccs):
        for node in component:
            node_to_scc[node] = i

    # SCC 간의 간선 (DAG)
    scc_edges = set()
    for u, v in edges:
        scc_u = node_to_scc[u]
        scc_v = node_to_scc[v]
        if scc_u != scc_v:
            scc_edges.add((scc_u, scc_v))

    scc_graph = defaultdict(list)
    for u, v in scc_edges:
        scc_graph[u].append(v)

    return sccs, dict(scc_graph), node_to_scc


# =============================================================================
# 4. 2-SAT 문제
# =============================================================================

class TwoSAT:
    """
    2-SAT 문제 해결기
    n개의 불린 변수와 2-CNF 조건식

    변수 x_i: 노드 2*i (true), 노드 2*i+1 (false)
    조건 (a ∨ b): ¬a → b, ¬b → a
    """

    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)

    def _var(self, x: int, negated: bool) -> int:
        """변수를 그래프 노드로 변환"""
        return 2 * x + (1 if negated else 0)

    def _neg(self, node: int) -> int:
        """노드의 부정"""
        return node ^ 1

    def add_clause(self, x: int, neg_x: bool, y: int, neg_y: bool):
        """
        조건 추가: (x ∨ y)
        neg_x: x가 부정인지
        neg_y: y가 부정인지
        """
        # (x ∨ y) ≡ (¬x → y) ∧ (¬y → x)
        node_x = self._var(x, neg_x)
        node_y = self._var(y, neg_y)

        # ¬x → y
        self.graph[self._neg(node_x)].append(node_y)
        self.reverse_graph[node_y].append(self._neg(node_x))

        # ¬y → x
        self.graph[self._neg(node_y)].append(node_x)
        self.reverse_graph[node_x].append(self._neg(node_y))

    def solve(self) -> Tuple[bool, List[bool]]:
        """
        2-SAT 해결
        반환: (만족 가능 여부, 각 변수의 값)
        """
        total_nodes = 2 * self.n

        # Kosaraju로 SCC 찾기
        visited = [False] * total_nodes
        finish_stack = []

        def dfs1(node: int):
            visited[node] = True
            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    dfs1(neighbor)
            finish_stack.append(node)

        for i in range(total_nodes):
            if not visited[i]:
                dfs1(i)

        visited = [False] * total_nodes
        scc_id = [-1] * total_nodes
        current_scc = 0

        def dfs2(node: int):
            visited[node] = True
            scc_id[node] = current_scc
            for neighbor in self.reverse_graph[node]:
                if not visited[neighbor]:
                    dfs2(neighbor)

        while finish_stack:
            node = finish_stack.pop()
            if not visited[node]:
                dfs2(node)
                current_scc += 1

        # 만족 가능성 검사
        for i in range(self.n):
            if scc_id[2 * i] == scc_id[2 * i + 1]:
                return False, []

        # 해 구성 (나중에 발견된 SCC = 더 작은 SCC ID = 더 높은 위상 순서)
        assignment = [False] * self.n
        for i in range(self.n):
            # scc_id가 작으면 위상 순서에서 뒤에 있음 → 그 값이 true
            assignment[i] = scc_id[2 * i] > scc_id[2 * i + 1]

        return True, assignment


# =============================================================================
# 5. 실전 문제: 학교 도달 가능성
# =============================================================================

def min_roads_to_connect(n: int, roads: List[Tuple[int, int]]) -> int:
    """
    모든 학교가 서로 도달 가능하게 하려면 추가해야 할 최소 도로 수
    = max(진입 차수 0인 SCC 수, 진출 차수 0인 SCC 수)
    (SCC가 1개면 0)
    """
    if not roads:
        return n - 1 if n > 1 else 0

    sccs, scc_graph, node_to_scc = build_scc_dag(n, roads)

    if len(sccs) == 1:
        return 0

    # 각 SCC의 진입/진출 차수
    in_degree = [0] * len(sccs)
    out_degree = [0] * len(sccs)

    for scc_u, neighbors in scc_graph.items():
        out_degree[scc_u] = len(neighbors)
        for scc_v in neighbors:
            in_degree[scc_v] += 1

    # 진입/진출 차수 0인 SCC 개수
    sources = sum(1 for d in in_degree if d == 0)
    sinks = sum(1 for d in out_degree if d == 0)

    return max(sources, sinks)


# =============================================================================
# 6. 실전 문제: 중요 노드 (Articulation Points 유사)
# =============================================================================

def find_critical_nodes(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """
    제거하면 SCC 개수가 증가하는 노드들 찾기
    (간단한 brute force 구현)
    """
    original_scc_count = len(tarjan_scc(n, edges))
    critical = []

    for remove_node in range(n):
        # 해당 노드 제외한 그래프
        new_edges = [(u, v) for u, v in edges if u != remove_node and v != remove_node]

        # 노드 재매핑
        remaining = [i for i in range(n) if i != remove_node]
        if not remaining:
            continue

        node_map = {old: new for new, old in enumerate(remaining)}
        remapped_edges = [(node_map[u], node_map[v]) for u, v in new_edges
                          if u in node_map and v in node_map]

        new_scc_count = len(tarjan_scc(len(remaining), remapped_edges))

        if new_scc_count > original_scc_count - 1:  # -1은 제거된 노드의 SCC
            critical.append(remove_node)

    return critical


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("강한 연결 요소 (SCC) 예제")
    print("=" * 60)

    # 그래프 구성
    #   0 → 1 → 2
    #   ↑   ↓   ↓
    #   4 ← 3 → 5 → 6
    #       ↑       ↓
    #       └───────┘

    n = 7
    edges = [
        (0, 1), (1, 2), (1, 3), (2, 5),
        (3, 4), (4, 0), (3, 5), (5, 6), (6, 3)
    ]

    # 1. Kosaraju 알고리즘
    print("\n[1] Kosaraju 알고리즘")
    sccs = kosaraju_scc(n, edges)
    print(f"    간선: {edges}")
    print(f"    SCC: {sccs}")

    # 2. Tarjan 알고리즘
    print("\n[2] Tarjan 알고리즘")
    sccs = tarjan_scc(n, edges)
    print(f"    SCC: {sccs}")

    # 3. SCC 축약 DAG
    print("\n[3] SCC 축약 그래프 (DAG)")
    sccs, scc_graph, node_to_scc = build_scc_dag(n, edges)
    print(f"    SCC: {sccs}")
    print(f"    노드→SCC: {node_to_scc}")
    print(f"    SCC 간선: {dict(scc_graph)}")

    # 4. 2-SAT 문제
    print("\n[4] 2-SAT 문제")
    # (x0 ∨ x1) ∧ (¬x0 ∨ x2) ∧ (¬x1 ∨ ¬x2)
    sat = TwoSAT(3)
    sat.add_clause(0, False, 1, False)  # x0 ∨ x1
    sat.add_clause(0, True, 2, False)   # ¬x0 ∨ x2
    sat.add_clause(1, True, 2, True)    # ¬x1 ∨ ¬x2

    solvable, assignment = sat.solve()
    print(f"    조건: (x0 ∨ x1) ∧ (¬x0 ∨ x2) ∧ (¬x1 ∨ ¬x2)")
    print(f"    해결 가능: {solvable}")
    if solvable:
        print(f"    해: x0={assignment[0]}, x1={assignment[1]}, x2={assignment[2]}")

    # 불가능한 2-SAT
    print("\n    불가능한 경우:")
    sat2 = TwoSAT(1)
    sat2.add_clause(0, False, 0, False)  # x0 ∨ x0 = x0
    sat2.add_clause(0, True, 0, True)    # ¬x0 ∨ ¬x0 = ¬x0
    solvable2, _ = sat2.solve()
    print(f"    조건: x0 ∧ ¬x0")
    print(f"    해결 가능: {solvable2}")

    # 5. 학교 연결
    print("\n[5] 학교 연결 문제")
    school_roads = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3)]
    min_roads = min_roads_to_connect(5, school_roads)
    print(f"    도로: {school_roads}")
    print(f"    추가 필요 도로: {min_roads}개")

    # 6. 알고리즘 비교
    print("\n[6] Kosaraju vs Tarjan 비교")
    print("    | 특성           | Kosaraju      | Tarjan        |")
    print("    |----------------|---------------|---------------|")
    print("    | 시간복잡도     | O(V + E)      | O(V + E)      |")
    print("    | DFS 횟수       | 2번           | 1번           |")
    print("    | 역그래프 필요  | 예            | 아니오        |")
    print("    | 구현 난이도    | 쉬움          | 보통          |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
