"""
위상 정렬 (Topological Sort)
Topological Sorting

방향 비순환 그래프(DAG)에서 정점을 선형 순서로 나열합니다.
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import deque, defaultdict


# =============================================================================
# 1. Kahn's 알고리즘 (BFS 기반)
# =============================================================================

def topological_sort_kahn(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """
    Kahn's 알고리즘 (진입 차수 기반)
    시간복잡도: O(V + E)
    공간복잡도: O(V + E)

    edges: [(from, to), ...] - from → to 의존 관계
    반환: 위상 정렬된 순서, 사이클 있으면 빈 리스트
    """
    # 그래프 및 진입 차수 구성
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # 진입 차수 0인 노드로 시작
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 모든 노드 방문 여부로 사이클 확인
    return result if len(result) == n else []


# =============================================================================
# 2. DFS 기반 위상 정렬
# =============================================================================

def topological_sort_dfs(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """
    DFS 기반 위상 정렬
    시간복잡도: O(V + E)
    공간복잡도: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    result = []
    has_cycle = False

    def dfs(node: int) -> None:
        nonlocal has_cycle

        if has_cycle:
            return

        color[node] = GRAY  # 방문 중

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:  # 사이클 발견
                has_cycle = True
                return
            if color[neighbor] == WHITE:
                dfs(neighbor)

        color[node] = BLACK  # 방문 완료
        result.append(node)

    for i in range(n):
        if color[i] == WHITE:
            dfs(i)

    return result[::-1] if not has_cycle else []


# =============================================================================
# 3. 사이클 탐지
# =============================================================================

def has_cycle(n: int, edges: List[Tuple[int, int]]) -> bool:
    """
    방향 그래프에서 사이클 존재 여부
    시간복잡도: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node: int) -> bool:
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # 사이클
            if color[neighbor] == WHITE and dfs(neighbor):
                return True

        color[node] = BLACK
        return False

    for i in range(n):
        if color[i] == WHITE and dfs(i):
            return True

    return False


# =============================================================================
# 4. 모든 위상 정렬 순서 찾기
# =============================================================================

def all_topological_sorts(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    모든 가능한 위상 정렬 순서 찾기
    시간복잡도: O(V! * (V + E)) - 최악의 경우
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    result = []
    path = []
    visited = [False] * n

    def backtrack():
        # 모든 노드 방문 완료
        if len(path) == n:
            result.append(path.copy())
            return

        for i in range(n):
            if not visited[i] and in_degree[i] == 0:
                # 선택
                visited[i] = True
                path.append(i)
                for neighbor in graph[i]:
                    in_degree[neighbor] -= 1

                backtrack()

                # 복원
                visited[i] = False
                path.pop()
                for neighbor in graph[i]:
                    in_degree[neighbor] += 1

    backtrack()
    return result


# =============================================================================
# 5. 실전 문제: 수강 과목 순서 (Course Schedule)
# =============================================================================

def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    모든 과목 수강 가능 여부 (사이클 없음 = 가능)
    prerequisites: [course, prereq] - prereq → course
    """
    edges = [(prereq, course) for course, prereq in prerequisites]
    return not has_cycle(num_courses, edges)


def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    과목 수강 순서 반환
    """
    edges = [(prereq, course) for course, prereq in prerequisites]
    return topological_sort_kahn(num_courses, edges)


# =============================================================================
# 6. 실전 문제: 외계인 사전 (Alien Dictionary)
# =============================================================================

def alien_order(words: List[str]) -> str:
    """
    외계인 알파벳 순서 결정
    단어 목록이 사전순으로 정렬되어 있다고 가정
    """
    # 모든 문자 수집
    chars = set()
    for word in words:
        chars.update(word)

    # 그래프 구성
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    for char in chars:
        in_degree[char] = 0

    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]

        # 유효하지 않은 순서 체크
        if len(word1) > len(word2) and word1.startswith(word2):
            return ""

        # 첫 번째 다른 문자로 순서 결정
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break

    # 위상 정렬
    queue = deque([c for c in chars if in_degree[c] == 0])
    result = []

    while queue:
        char = queue.popleft()
        result.append(char)

        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return ''.join(result) if len(result) == len(chars) else ""


# =============================================================================
# 7. 실전 문제: 작업 병렬 실행 (Parallel Courses)
# =============================================================================

def minimum_semesters(n: int, relations: List[List[int]]) -> int:
    """
    모든 과목을 수강하는데 필요한 최소 학기 수
    병렬 수강 가능, relations: [prev, next]
    """
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)

    for prev_course, next_course in relations:
        graph[prev_course].append(next_course)
        in_degree[next_course] += 1

    # 진입 차수 0인 노드로 시작 (1-indexed)
    queue = deque([i for i in range(1, n + 1) if in_degree[i] == 0])
    semesters = 0
    completed = 0

    while queue:
        semesters += 1
        next_queue = deque()

        while queue:
            course = queue.popleft()
            completed += 1

            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    next_queue.append(next_course)

        queue = next_queue

    return semesters if completed == n else -1


# =============================================================================
# 8. 실전 문제: 최장 경로 (DAG)
# =============================================================================

def longest_path_dag(n: int, edges: List[Tuple[int, int, int]]) -> List[int]:
    """
    DAG에서 각 노드까지의 최장 경로 (가중치 포함)
    edges: [(from, to, weight), ...]
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v, w in edges:
        graph[u].append((v, w))
        in_degree[v] += 1

    # 위상 정렬
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor, _ in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 최장 경로 계산
    dist = [0] * n

    for node in topo_order:
        for neighbor, weight in graph[node]:
            dist[neighbor] = max(dist[neighbor], dist[node] + weight)

    return dist


# =============================================================================
# 9. 실전 문제: 빌드 순서 결정
# =============================================================================

def build_order(projects: List[str], dependencies: List[Tuple[str, str]]) -> List[str]:
    """
    빌드 순서 결정
    dependencies: [(proj, depends_on), ...] - proj가 depends_on에 의존
    """
    # 프로젝트 인덱스 매핑
    proj_to_idx = {p: i for i, p in enumerate(projects)}
    n = len(projects)

    # 간선 변환 (depends_on → proj)
    edges = [(proj_to_idx[dep], proj_to_idx[proj]) for proj, dep in dependencies]

    # 위상 정렬
    order = topological_sort_kahn(n, edges)

    if not order:
        return []  # 사이클 존재

    return [projects[i] for i in order]


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("위상 정렬 (Topological Sort) 예제")
    print("=" * 60)

    # 1. Kahn's 알고리즘
    print("\n[1] Kahn's 알고리즘 (BFS)")
    n = 6
    edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
    result = topological_sort_kahn(n, edges)
    print(f"    노드: 0-5, 간선: {edges}")
    print(f"    위상 정렬: {result}")

    # 2. DFS 기반
    print("\n[2] DFS 기반 위상 정렬")
    result = topological_sort_dfs(n, edges)
    print(f"    위상 정렬: {result}")

    # 3. 사이클 탐지
    print("\n[3] 사이클 탐지")
    cyclic_edges = [(0, 1), (1, 2), (2, 0)]
    acyclic_edges = [(0, 1), (1, 2), (0, 2)]
    print(f"    {cyclic_edges}: 사이클 {has_cycle(3, cyclic_edges)}")
    print(f"    {acyclic_edges}: 사이클 {has_cycle(3, acyclic_edges)}")

    # 4. 모든 위상 정렬
    print("\n[4] 모든 위상 정렬 순서")
    n = 4
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    all_orders = all_topological_sorts(n, edges)
    print(f"    노드: 0-3, 간선: {edges}")
    print(f"    모든 순서: {all_orders}")

    # 5. 과목 수강 순서
    print("\n[5] 과목 수강 순서")
    num_courses = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    can = can_finish(num_courses, prerequisites)
    order = find_order(num_courses, prerequisites)
    print(f"    과목 수: {num_courses}, 선수: {prerequisites}")
    print(f"    수강 가능: {can}")
    print(f"    순서: {order}")

    # 6. 외계인 사전
    print("\n[6] 외계인 사전")
    words = ["wrt", "wrf", "er", "ett", "rftt"]
    order = alien_order(words)
    print(f"    단어: {words}")
    print(f"    알파벳 순서: {order}")

    # 7. 최소 학기 수
    print("\n[7] 최소 학기 수")
    n = 3
    relations = [[1, 3], [2, 3]]
    semesters = minimum_semesters(n, relations)
    print(f"    과목: {n}, 관계: {relations}")
    print(f"    최소 학기: {semesters}")

    # 8. DAG 최장 경로
    print("\n[8] DAG 최장 경로")
    n = 4
    edges = [(0, 1, 3), (0, 2, 2), (1, 3, 4), (2, 3, 1)]
    dist = longest_path_dag(n, edges)
    print(f"    간선: {edges}")
    print(f"    각 노드까지 최장 거리: {dist}")

    # 9. 빌드 순서
    print("\n[9] 빌드 순서")
    projects = ['a', 'b', 'c', 'd', 'e', 'f']
    dependencies = [('d', 'a'), ('b', 'f'), ('d', 'b'), ('a', 'f'), ('c', 'd')]
    order = build_order(projects, dependencies)
    print(f"    프로젝트: {projects}")
    print(f"    의존성: {dependencies}")
    print(f"    빌드 순서: {order}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
