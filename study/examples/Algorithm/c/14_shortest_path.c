/*
 * 최단 경로 (Shortest Path)
 * Dijkstra, Bellman-Ford, Floyd-Warshall
 *
 * 그래프에서 최단 경로를 찾는 알고리즘입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#define INF INT_MAX

/* =============================================================================
 * 1. 그래프 구조
 * ============================================================================= */

typedef struct Edge {
    int dest;
    int weight;
    struct Edge* next;
} Edge;

typedef struct {
    Edge** adj;
    int vertices;
} Graph;

Graph* graph_create(int vertices) {
    Graph* g = malloc(sizeof(Graph));
    g->vertices = vertices;
    g->adj = calloc(vertices, sizeof(Edge*));
    return g;
}

void graph_add_edge(Graph* g, int src, int dest, int weight) {
    Edge* edge = malloc(sizeof(Edge));
    edge->dest = dest;
    edge->weight = weight;
    edge->next = g->adj[src];
    g->adj[src] = edge;
}

void graph_free(Graph* g) {
    for (int i = 0; i < g->vertices; i++) {
        Edge* e = g->adj[i];
        while (e) {
            Edge* temp = e;
            e = e->next;
            free(temp);
        }
    }
    free(g->adj);
    free(g);
}

/* =============================================================================
 * 2. 다익스트라 알고리즘 (배열 기반)
 * ============================================================================= */

int* dijkstra_array(Graph* g, int src) {
    int* dist = malloc(g->vertices * sizeof(int));
    bool* visited = calloc(g->vertices, sizeof(bool));

    for (int i = 0; i < g->vertices; i++)
        dist[i] = INF;
    dist[src] = 0;

    for (int count = 0; count < g->vertices - 1; count++) {
        /* 최소 거리 정점 찾기 */
        int min_dist = INF, u = -1;
        for (int v = 0; v < g->vertices; v++) {
            if (!visited[v] && dist[v] < min_dist) {
                min_dist = dist[v];
                u = v;
            }
        }

        if (u == -1) break;
        visited[u] = true;

        /* 인접 정점 갱신 */
        Edge* e = g->adj[u];
        while (e) {
            if (!visited[e->dest] && dist[u] != INF &&
                dist[u] + e->weight < dist[e->dest]) {
                dist[e->dest] = dist[u] + e->weight;
            }
            e = e->next;
        }
    }

    free(visited);
    return dist;
}

/* =============================================================================
 * 3. 다익스트라 (우선순위 큐)
 * ============================================================================= */

typedef struct {
    int vertex;
    int dist;
} HeapNode;

typedef struct {
    HeapNode* data;
    int size;
    int capacity;
} MinHeap;

MinHeap* heap_create(int capacity) {
    MinHeap* h = malloc(sizeof(MinHeap));
    h->data = malloc(capacity * sizeof(HeapNode));
    h->size = 0;
    h->capacity = capacity;
    return h;
}

void heap_push(MinHeap* h, int vertex, int dist) {
    int i = h->size++;
    h->data[i].vertex = vertex;
    h->data[i].dist = dist;

    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->data[parent].dist <= h->data[i].dist) break;
        HeapNode temp = h->data[parent];
        h->data[parent] = h->data[i];
        h->data[i] = temp;
        i = parent;
    }
}

HeapNode heap_pop(MinHeap* h) {
    HeapNode min = h->data[0];
    h->data[0] = h->data[--h->size];

    int i = 0;
    while (2 * i + 1 < h->size) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (h->data[left].dist < h->data[smallest].dist)
            smallest = left;
        if (right < h->size && h->data[right].dist < h->data[smallest].dist)
            smallest = right;

        if (smallest == i) break;

        HeapNode temp = h->data[i];
        h->data[i] = h->data[smallest];
        h->data[smallest] = temp;
        i = smallest;
    }

    return min;
}

int* dijkstra_heap(Graph* g, int src) {
    int* dist = malloc(g->vertices * sizeof(int));
    for (int i = 0; i < g->vertices; i++)
        dist[i] = INF;
    dist[src] = 0;

    MinHeap* pq = heap_create(g->vertices * g->vertices);
    heap_push(pq, src, 0);

    while (pq->size > 0) {
        HeapNode node = heap_pop(pq);
        int u = node.vertex;
        int d = node.dist;

        if (d > dist[u]) continue;

        Edge* e = g->adj[u];
        while (e) {
            if (dist[u] + e->weight < dist[e->dest]) {
                dist[e->dest] = dist[u] + e->weight;
                heap_push(pq, e->dest, dist[e->dest]);
            }
            e = e->next;
        }
    }

    free(pq->data);
    free(pq);
    return dist;
}

/* =============================================================================
 * 4. 벨만-포드 알고리즘
 * ============================================================================= */

typedef struct {
    int src;
    int dest;
    int weight;
} EdgeList;

int* bellman_ford(int vertices, EdgeList edges[], int num_edges, int src, bool* has_negative_cycle) {
    int* dist = malloc(vertices * sizeof(int));
    for (int i = 0; i < vertices; i++)
        dist[i] = INF;
    dist[src] = 0;

    /* V-1번 반복 */
    for (int i = 0; i < vertices - 1; i++) {
        for (int j = 0; j < num_edges; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int w = edges[j].weight;

            if (dist[u] != INF && dist[u] + w < dist[v])
                dist[v] = dist[u] + w;
        }
    }

    /* 음수 사이클 검사 */
    *has_negative_cycle = false;
    for (int j = 0; j < num_edges; j++) {
        int u = edges[j].src;
        int v = edges[j].dest;
        int w = edges[j].weight;

        if (dist[u] != INF && dist[u] + w < dist[v]) {
            *has_negative_cycle = true;
            break;
        }
    }

    return dist;
}

/* =============================================================================
 * 5. 플로이드-워셜 알고리즘
 * ============================================================================= */

int** floyd_warshall(int** graph, int vertices) {
    int** dist = malloc(vertices * sizeof(int*));
    for (int i = 0; i < vertices; i++) {
        dist[i] = malloc(vertices * sizeof(int));
        for (int j = 0; j < vertices; j++)
            dist[i][j] = graph[i][j];
    }

    for (int k = 0; k < vertices; k++) {
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    return dist;
}

/* =============================================================================
 * 6. 경로 복원
 * ============================================================================= */

int* dijkstra_with_path(Graph* g, int src, int** parent) {
    int* dist = malloc(g->vertices * sizeof(int));
    *parent = malloc(g->vertices * sizeof(int));
    bool* visited = calloc(g->vertices, sizeof(bool));

    for (int i = 0; i < g->vertices; i++) {
        dist[i] = INF;
        (*parent)[i] = -1;
    }
    dist[src] = 0;

    for (int count = 0; count < g->vertices - 1; count++) {
        int min_dist = INF, u = -1;
        for (int v = 0; v < g->vertices; v++) {
            if (!visited[v] && dist[v] < min_dist) {
                min_dist = dist[v];
                u = v;
            }
        }

        if (u == -1) break;
        visited[u] = true;

        Edge* e = g->adj[u];
        while (e) {
            if (!visited[e->dest] && dist[u] != INF &&
                dist[u] + e->weight < dist[e->dest]) {
                dist[e->dest] = dist[u] + e->weight;
                (*parent)[e->dest] = u;
            }
            e = e->next;
        }
    }

    free(visited);
    return dist;
}

void print_path(int parent[], int dest) {
    if (parent[dest] == -1) {
        printf("%d", dest);
        return;
    }
    print_path(parent, parent[dest]);
    printf(" -> %d", dest);
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("최단 경로 (Shortest Path) 예제\n");
    printf("============================================================\n");

    /* 1. 다익스트라 (배열) */
    printf("\n[1] 다익스트라 알고리즘 (배열)\n");
    Graph* g1 = graph_create(5);
    graph_add_edge(g1, 0, 1, 4);
    graph_add_edge(g1, 0, 2, 1);
    graph_add_edge(g1, 2, 1, 2);
    graph_add_edge(g1, 1, 3, 1);
    graph_add_edge(g1, 2, 3, 5);
    graph_add_edge(g1, 3, 4, 3);

    int* dist1 = dijkstra_array(g1, 0);
    printf("    0에서 각 정점까지 거리:\n");
    for (int i = 0; i < 5; i++)
        printf("      0 -> %d: %d\n", i, dist1[i]);
    free(dist1);

    /* 2. 다익스트라 (힙) */
    printf("\n[2] 다익스트라 알고리즘 (힙)\n");
    int* dist2 = dijkstra_heap(g1, 0);
    printf("    0에서 각 정점까지 거리:\n");
    for (int i = 0; i < 5; i++)
        printf("      0 -> %d: %d\n", i, dist2[i]);
    free(dist2);

    /* 3. 경로 복원 */
    printf("\n[3] 경로 복원\n");
    int* parent;
    int* dist3 = dijkstra_with_path(g1, 0, &parent);
    for (int i = 1; i < 5; i++) {
        printf("    0 -> %d (거리 %d): ", i, dist3[i]);
        print_path(parent, i);
        printf("\n");
    }
    free(dist3);
    free(parent);
    graph_free(g1);

    /* 4. 벨만-포드 */
    printf("\n[4] 벨만-포드 알고리즘\n");
    EdgeList edges[] = {
        {0, 1, 4}, {0, 2, 1}, {2, 1, 2},
        {1, 3, 1}, {2, 3, 5}, {3, 4, 3}
    };
    bool has_neg_cycle;
    int* dist4 = bellman_ford(5, edges, 6, 0, &has_neg_cycle);
    printf("    음수 사이클: %s\n", has_neg_cycle ? "있음" : "없음");
    printf("    거리: ");
    for (int i = 0; i < 5; i++)
        printf("%d ", dist4[i]);
    printf("\n");
    free(dist4);

    /* 음수 사이클 테스트 */
    printf("\n    음수 간선 테스트:\n");
    EdgeList edges_neg[] = {
        {0, 1, 1}, {1, 2, -1}, {2, 0, -1}
    };
    int* dist_neg = bellman_ford(3, edges_neg, 3, 0, &has_neg_cycle);
    printf("    음수 사이클: %s\n", has_neg_cycle ? "있음" : "없음");
    free(dist_neg);

    /* 5. 플로이드-워셜 */
    printf("\n[5] 플로이드-워셜 알고리즘\n");
    int** matrix = malloc(4 * sizeof(int*));
    for (int i = 0; i < 4; i++) {
        matrix[i] = malloc(4 * sizeof(int));
        for (int j = 0; j < 4; j++)
            matrix[i][j] = (i == j) ? 0 : INF;
    }
    matrix[0][1] = 3;
    matrix[0][3] = 7;
    matrix[1][0] = 8;
    matrix[1][2] = 2;
    matrix[2][0] = 5;
    matrix[2][3] = 1;
    matrix[3][0] = 2;

    int** dist5 = floyd_warshall(matrix, 4);
    printf("    모든 쌍 최단 거리:\n");
    printf("       ");
    for (int i = 0; i < 4; i++) printf("%4d ", i);
    printf("\n");
    for (int i = 0; i < 4; i++) {
        printf("    %d: ", i);
        for (int j = 0; j < 4; j++) {
            if (dist5[i][j] == INF)
                printf(" INF ");
            else
                printf("%4d ", dist5[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < 4; i++) {
        free(matrix[i]);
        free(dist5[i]);
    }
    free(matrix);
    free(dist5);

    /* 6. 알고리즘 비교 */
    printf("\n[6] 알고리즘 비교\n");
    printf("    | 알고리즘      | 시간복잡도    | 특징              |\n");
    printf("    |---------------|---------------|-------------------|\n");
    printf("    | 다익스트라(배열)| O(V²)       | 양수 가중치만     |\n");
    printf("    | 다익스트라(힙) | O(E log V)   | 희소 그래프 최적  |\n");
    printf("    | 벨만-포드     | O(V * E)      | 음수 가중치 허용  |\n");
    printf("    | 플로이드-워셜 | O(V³)         | 모든 쌍 최단경로  |\n");

    printf("\n============================================================\n");

    return 0;
}
