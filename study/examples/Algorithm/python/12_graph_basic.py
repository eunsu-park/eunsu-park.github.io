"""
DFS (깊이 우선 탐색) & BFS (너비 우선 탐색)
Depth-First Search & Breadth-First Search

그래프 탐색의 두 가지 기본 알고리즘입니다.
"""

from collections import deque, defaultdict
from typing import List, Dict, Set, Optional


# =============================================================================
# 그래프 표현
# =============================================================================
def create_adjacency_list(edges: List[List[int]], directed: bool = False) -> Dict[int, List[int]]:
    """간선 리스트로부터 인접 리스트 생성"""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return graph


# =============================================================================
# 1. DFS (재귀)
# =============================================================================
def dfs_recursive(graph: Dict[int, List[int]], start: int, visited: Set[int] = None) -> List[int]:
    """
    DFS 재귀 구현
    시간복잡도: O(V + E), 공간복잡도: O(V)
    """
    if visited is None:
        visited = set()

    result = []
    visited.add(start)
    result.append(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))

    return result


# =============================================================================
# 2. DFS (스택)
# =============================================================================
def dfs_iterative(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    DFS 반복문 구현 (스택 사용)
    시간복잡도: O(V + E), 공간복잡도: O(V)
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            # 역순으로 추가하여 작은 번호 먼저 방문
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result


# =============================================================================
# 3. BFS
# =============================================================================
def bfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    BFS 구현 (큐 사용)
    시간복잡도: O(V + E), 공간복잡도: O(V)
    """
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result


# =============================================================================
# 4. 연결 요소 찾기
# =============================================================================
def count_connected_components(n: int, edges: List[List[int]]) -> int:
    """
    무방향 그래프의 연결 요소 개수
    """
    graph = create_adjacency_list(edges, directed=False)
    visited = set()
    count = 0

    for node in range(n):
        if node not in visited:
            # DFS로 연결된 모든 노드 방문
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(graph[curr])
            count += 1

    return count


# =============================================================================
# 5. 최단 거리 (BFS) - 가중치 없는 그래프
# =============================================================================
def shortest_path_bfs(graph: Dict[int, List[int]], start: int, end: int) -> Optional[List[int]]:
    """
    가중치 없는 그래프에서 최단 경로 찾기
    """
    if start == end:
        return [start]

    visited = set([start])
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()

        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # 경로 없음


# =============================================================================
# 6. 2D 격자 탐색
# =============================================================================
def num_islands(grid: List[List[str]]) -> int:
    """
    섬의 개수 세기 (DFS)
    '1' = 땅, '0' = 물
    """
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'  # 방문 표시
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count


# =============================================================================
# 7. 레벨별 BFS 탐색
# =============================================================================
def bfs_by_level(graph: Dict[int, List[int]], start: int) -> List[List[int]]:
    """
    BFS 레벨(깊이)별로 노드 그룹화
    """
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        result.append(current_level)

    return result


# =============================================================================
# 8. 사이클 검출 (무방향 그래프)
# =============================================================================
def has_cycle_undirected(n: int, edges: List[List[int]]) -> bool:
    """
    무방향 그래프에서 사이클 존재 여부
    """
    graph = create_adjacency_list(edges, directed=False)
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    for node in range(n):
        if node not in visited:
            if dfs(node, -1):
                return True

    return False


# =============================================================================
# 9. 사이클 검출 (방향 그래프)
# =============================================================================
def has_cycle_directed(n: int, edges: List[List[int]]) -> bool:
    """
    방향 그래프에서 사이클 존재 여부
    상태: 0=미방문, 1=방문중(현재 경로), 2=완료
    """
    graph = create_adjacency_list(edges, directed=True)
    state = [0] * n  # 0: 미방문, 1: 방문중, 2: 완료

    def dfs(node):
        if state[node] == 1:  # 방문 중인 노드 재방문 = 사이클
            return True
        if state[node] == 2:  # 이미 완료된 노드
            return False

        state[node] = 1  # 방문 시작

        for neighbor in graph[node]:
            if dfs(neighbor):
                return True

        state[node] = 2  # 방문 완료
        return False

    for node in range(n):
        if state[node] == 0:
            if dfs(node):
                return True

    return False


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("DFS & BFS 예제")
    print("=" * 60)

    # 그래프 생성
    edges = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]
    graph = create_adjacency_list(edges)

    print("\n[그래프 구조]")
    print("       0")
    print("      / \\")
    print("     1   2")
    print("    /|   |\\")
    print("   3 4   5 6")

    # 1. DFS (재귀)
    print("\n[1] DFS (재귀)")
    result = dfs_recursive(graph, 0)
    print(f"    시작: 0, 탐색 순서: {result}")

    # 2. DFS (반복)
    print("\n[2] DFS (반복/스택)")
    result = dfs_iterative(graph, 0)
    print(f"    시작: 0, 탐색 순서: {result}")

    # 3. BFS
    print("\n[3] BFS")
    result = bfs(graph, 0)
    print(f"    시작: 0, 탐색 순서: {result}")

    # 4. 연결 요소
    print("\n[4] 연결 요소 개수")
    edges2 = [[0, 1], [1, 2], [3, 4]]
    count = count_connected_components(5, edges2)
    print(f"    노드 5개, 간선: {edges2}")
    print(f"    연결 요소 개수: {count}")

    # 5. 최단 경로
    print("\n[5] 최단 경로 (BFS)")
    path = shortest_path_bfs(graph, 0, 6)
    print(f"    0 -> 6 최단 경로: {path}")

    # 6. 섬 개수 세기
    print("\n[6] 섬의 개수")
    grid = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1']
    ]
    # 복사본 사용 (원본 변경됨)
    grid_copy = [row[:] for row in grid]
    count = num_islands(grid_copy)
    print(f"    격자:")
    for row in grid:
        print(f"    {row}")
    print(f"    섬의 개수: {count}")

    # 7. 레벨별 BFS
    print("\n[7] 레벨별 BFS 탐색")
    levels = bfs_by_level(graph, 0)
    for i, level in enumerate(levels):
        print(f"    레벨 {i}: {level}")

    # 8. 사이클 검출 (무방향)
    print("\n[8] 사이클 검출 (무방향 그래프)")
    edges_no_cycle = [[0, 1], [1, 2], [2, 3]]
    edges_with_cycle = [[0, 1], [1, 2], [2, 0]]
    print(f"    간선 {edges_no_cycle}: 사이클 = {has_cycle_undirected(4, edges_no_cycle)}")
    print(f"    간선 {edges_with_cycle}: 사이클 = {has_cycle_undirected(3, edges_with_cycle)}")

    # 9. 사이클 검출 (방향)
    print("\n[9] 사이클 검출 (방향 그래프)")
    edges_dag = [[0, 1], [1, 2], [0, 2]]
    edges_cycle = [[0, 1], [1, 2], [2, 0]]
    print(f"    DAG {edges_dag}: 사이클 = {has_cycle_directed(3, edges_dag)}")
    print(f"    간선 {edges_cycle}: 사이클 = {has_cycle_directed(3, edges_cycle)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
