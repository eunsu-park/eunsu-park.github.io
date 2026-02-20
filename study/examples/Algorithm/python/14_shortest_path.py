"""
다익스트라 알고리즘 (Dijkstra's Algorithm)
Dijkstra's Shortest Path Algorithm

가중치가 있는 그래프에서 단일 출발점 최단 경로를 찾는 알고리즘입니다.
음의 가중치가 없는 그래프에서 사용합니다.
"""

import heapq
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# =============================================================================
# 가중치 그래프 표현
# =============================================================================
def create_weighted_graph(edges: List[Tuple[int, int, int]], directed: bool = False) -> Dict[int, List[Tuple[int, int]]]:
    """
    간선 리스트 (u, v, weight)로부터 가중치 인접 리스트 생성
    graph[u] = [(v, weight), ...]
    """
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        if not directed:
            graph[v].append((u, w))
    return graph


# =============================================================================
# 1. 다익스트라 기본 구현
# =============================================================================
def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int, n: int) -> List[int]:
    """
    다익스트라 알고리즘 (우선순위 큐 사용)
    시간복잡도: O((V + E) log V)

    Args:
        graph: 인접 리스트 (노드 -> [(이웃, 가중치), ...])
        start: 시작 노드
        n: 총 노드 수

    Returns:
        각 노드까지의 최단 거리 배열 (도달 불가시 무한대)
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0

    # (거리, 노드) 튜플을 저장하는 최소 힙
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        # 이미 처리된 거리보다 크면 스킵
        if d > dist[u]:
            continue

        # 인접 노드 확인
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return dist


# =============================================================================
# 2. 다익스트라 + 경로 추적
# =============================================================================
def dijkstra_with_path(
    graph: Dict[int, List[Tuple[int, int]]],
    start: int,
    end: int,
    n: int
) -> Tuple[int, List[int]]:
    """
    최단 거리와 함께 실제 경로도 반환
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    parent = [-1] * n

    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        if u == end:
            break

        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    # 경로 복원
    if dist[end] == INF:
        return INF, []

    path = []
    node = end
    while node != -1:
        path.append(node)
        node = parent[node]
    path.reverse()

    return dist[end], path


# =============================================================================
# 3. 모든 쌍 최단 경로 (플로이드-워셜)
# =============================================================================
def floyd_warshall(n: int, edges: List[Tuple[int, int, int]]) -> List[List[int]]:
    """
    플로이드-워셜 알고리즘
    모든 정점 쌍 사이의 최단 거리 계산
    시간복잡도: O(V³)
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    # 자기 자신으로의 거리는 0
    for i in range(n):
        dist[i][i] = 0

    # 간선 정보 반영
    for u, v, w in edges:
        dist[u][v] = w

    # k를 경유하는 경로 고려
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


# =============================================================================
# 4. 벨만-포드 알고리즘 (음의 가중치 허용)
# =============================================================================
def bellman_ford(n: int, edges: List[Tuple[int, int, int]], start: int) -> Tuple[List[int], bool]:
    """
    벨만-포드 알고리즘
    음의 가중치를 허용하며, 음의 사이클 검출 가능
    시간복잡도: O(VE)

    Returns:
        (최단 거리 배열, 음의 사이클 존재 여부)
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0

    # V-1번 반복
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # 음의 사이클 검출 (한 번 더 반복해서 갱신되면 음의 사이클)
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, has_negative_cycle


# =============================================================================
# 5. 네트워크 지연 시간 문제
# =============================================================================
def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """
    노드 k에서 모든 노드로 신호가 도달하는 최소 시간
    times[i] = [source, target, time]
    도달 불가능하면 -1 반환
    """
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist = dijkstra(graph, k, n + 1)  # 노드 번호가 1부터 시작

    # 1~n번 노드 중 최대 거리
    max_time = max(dist[1:n + 1])

    return max_time if max_time != float('inf') else -1


# =============================================================================
# 6. K번째 최단 경로
# =============================================================================
def kth_shortest_path(
    graph: Dict[int, List[Tuple[int, int]]],
    start: int,
    end: int,
    k: int,
    n: int
) -> int:
    """
    K번째로 짧은 경로의 길이 반환
    찾을 수 없으면 -1 반환
    """
    INF = float('inf')
    count = [0] * n  # 각 노드에 도착한 횟수
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        count[u] += 1

        if u == end and count[u] == k:
            return d

        # k번 이상 방문한 노드는 더 이상 확장하지 않음
        if count[u] > k:
            continue

        for v, weight in graph[u]:
            heapq.heappush(pq, (d + weight, v))

    return -1


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("다익스트라 & 최단 경로 알고리즘")
    print("=" * 60)

    # 예제 그래프
    #       1
    #    0 ---> 1
    #    |      |
    #  4 |      | 2
    #    v      v
    #    2 ---> 3
    #       3
    edges = [
        (0, 1, 1),
        (0, 2, 4),
        (1, 3, 2),
        (2, 3, 3),
        (1, 2, 2)
    ]

    print("\n[그래프 구조]")
    print("    0 --1--> 1")
    print("    |        |")
    print("    4        2")
    print("    v        v")
    print("    2 --3--> 3")
    print("    (1->2 가중치 2)")

    # 1. 다익스트라 기본
    print("\n[1] 다익스트라 기본")
    graph = create_weighted_graph(edges, directed=True)
    dist = dijkstra(graph, 0, 4)
    print(f"    시작점: 0")
    for i, d in enumerate(dist):
        print(f"    노드 {i}까지 거리: {d}")

    # 2. 다익스트라 + 경로
    print("\n[2] 다익스트라 + 경로 추적")
    distance, path = dijkstra_with_path(graph, 0, 3, 4)
    print(f"    0 -> 3 최단 거리: {distance}")
    print(f"    경로: {' -> '.join(map(str, path))}")

    # 3. 플로이드-워셜
    print("\n[3] 플로이드-워셜 (모든 쌍 최단 거리)")
    all_dist = floyd_warshall(4, edges)
    print("    거리 행렬:")
    for i, row in enumerate(all_dist):
        row_str = [str(d) if d != float('inf') else '∞' for d in row]
        print(f"    {i}: {row_str}")

    # 4. 벨만-포드
    print("\n[4] 벨만-포드 (음의 가중치 허용)")
    edges_negative = [(0, 1, 4), (0, 2, 5), (1, 2, -3), (2, 3, 4)]
    dist, has_neg_cycle = bellman_ford(4, edges_negative, 0)
    print(f"    간선: {edges_negative}")
    print(f"    최단 거리: {dist}")
    print(f"    음의 사이클: {has_neg_cycle}")

    # 음의 사이클 예제
    edges_neg_cycle = [(0, 1, 1), (1, 2, -1), (2, 0, -1)]
    dist, has_neg_cycle = bellman_ford(3, edges_neg_cycle, 0)
    print(f"\n    음의 사이클 그래프: {edges_neg_cycle}")
    print(f"    음의 사이클 존재: {has_neg_cycle}")

    # 5. 네트워크 지연 시간
    print("\n[5] 네트워크 지연 시간")
    times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    n, k = 4, 2
    result = network_delay_time(times, n, k)
    print(f"    times={times}, n={n}, k={k}")
    print(f"    모든 노드 도달 시간: {result}")

    # 6. K번째 최단 경로
    print("\n[6] K번째 최단 경로")
    edges_k = [(0, 1, 1), (0, 2, 3), (1, 2, 1), (1, 3, 2), (2, 3, 1)]
    graph_k = create_weighted_graph(edges_k, directed=True)
    for k in range(1, 4):
        dist = kth_shortest_path(graph_k, 0, 3, k, 4)
        print(f"    0->3 {k}번째 최단 경로: {dist}")

    print("\n" + "=" * 60)
    print("주요 알고리즘 비교")
    print("=" * 60)
    print("""
    | 알고리즘       | 시간복잡도      | 음의 가중치 | 용도                |
    |---------------|----------------|------------|---------------------|
    | 다익스트라     | O((V+E)log V) | 불가        | 단일 출발점 최단거리 |
    | 벨만-포드      | O(VE)         | 가능        | 음의 가중치/사이클   |
    | 플로이드-워셜  | O(V³)         | 가능*       | 모든 쌍 최단거리     |

    * 플로이드-워셜도 음의 사이클이 있으면 올바르지 않음
    """)


if __name__ == "__main__":
    main()
