/*
 * 그래프 기초 (Graph Basics)
 * DFS, BFS, Graph Representation, Connected Components
 *
 * 그래프 자료구조와 기본 탐색 알고리즘입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX_VERTICES 100

/* =============================================================================
 * 1. 그래프 표현 - 인접 리스트
 * ============================================================================= */

typedef struct AdjNode {
    int vertex;
    int weight;
    struct AdjNode* next;
} AdjNode;

typedef struct {
    AdjNode** adj;
    int vertices;
    bool directed;
} Graph;

Graph* graph_create(int vertices, bool directed) {
    Graph* g = malloc(sizeof(Graph));
    g->vertices = vertices;
    g->directed = directed;
    g->adj = calloc(vertices, sizeof(AdjNode*));
    return g;
}

void graph_add_edge(Graph* g, int src, int dest, int weight) {
    AdjNode* node = malloc(sizeof(AdjNode));
    node->vertex = dest;
    node->weight = weight;
    node->next = g->adj[src];
    g->adj[src] = node;

    if (!g->directed) {
        node = malloc(sizeof(AdjNode));
        node->vertex = src;
        node->weight = weight;
        node->next = g->adj[dest];
        g->adj[dest] = node;
    }
}

void graph_free(Graph* g) {
    for (int i = 0; i < g->vertices; i++) {
        AdjNode* node = g->adj[i];
        while (node) {
            AdjNode* temp = node;
            node = node->next;
            free(temp);
        }
    }
    free(g->adj);
    free(g);
}

void graph_print(Graph* g) {
    for (int i = 0; i < g->vertices; i++) {
        printf("    %d: ", i);
        AdjNode* node = g->adj[i];
        while (node) {
            printf("%d(%d) ", node->vertex, node->weight);
            node = node->next;
        }
        printf("\n");
    }
}

/* =============================================================================
 * 2. 인접 행렬
 * ============================================================================= */

typedef struct {
    int** matrix;
    int vertices;
} AdjMatrix;

AdjMatrix* matrix_create(int vertices) {
    AdjMatrix* m = malloc(sizeof(AdjMatrix));
    m->vertices = vertices;
    m->matrix = malloc(vertices * sizeof(int*));
    for (int i = 0; i < vertices; i++) {
        m->matrix[i] = calloc(vertices, sizeof(int));
    }
    return m;
}

void matrix_add_edge(AdjMatrix* m, int src, int dest, int weight, bool directed) {
    m->matrix[src][dest] = weight;
    if (!directed)
        m->matrix[dest][src] = weight;
}

void matrix_free(AdjMatrix* m) {
    for (int i = 0; i < m->vertices; i++)
        free(m->matrix[i]);
    free(m->matrix);
    free(m);
}

/* =============================================================================
 * 3. DFS (깊이 우선 탐색)
 * ============================================================================= */

void dfs_recursive(Graph* g, int v, bool visited[]) {
    visited[v] = true;
    printf("%d ", v);

    AdjNode* node = g->adj[v];
    while (node) {
        if (!visited[node->vertex])
            dfs_recursive(g, node->vertex, visited);
        node = node->next;
    }
}

void dfs(Graph* g, int start) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    dfs_recursive(g, start, visited);
    free(visited);
}

/* 스택 기반 DFS */
void dfs_iterative(Graph* g, int start) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    int* stack = malloc(g->vertices * sizeof(int));
    int top = -1;

    stack[++top] = start;

    while (top >= 0) {
        int v = stack[top--];

        if (!visited[v]) {
            visited[v] = true;
            printf("%d ", v);

            AdjNode* node = g->adj[v];
            while (node) {
                if (!visited[node->vertex])
                    stack[++top] = node->vertex;
                node = node->next;
            }
        }
    }

    free(visited);
    free(stack);
}

/* =============================================================================
 * 4. BFS (너비 우선 탐색)
 * ============================================================================= */

void bfs(Graph* g, int start) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    int* queue = malloc(g->vertices * sizeof(int));
    int front = 0, rear = 0;

    visited[start] = true;
    queue[rear++] = start;

    while (front < rear) {
        int v = queue[front++];
        printf("%d ", v);

        AdjNode* node = g->adj[v];
        while (node) {
            if (!visited[node->vertex]) {
                visited[node->vertex] = true;
                queue[rear++] = node->vertex;
            }
            node = node->next;
        }
    }

    free(visited);
    free(queue);
}

/* BFS 최단 거리 */
int* bfs_distances(Graph* g, int start) {
    int* dist = malloc(g->vertices * sizeof(int));
    for (int i = 0; i < g->vertices; i++)
        dist[i] = -1;

    int* queue = malloc(g->vertices * sizeof(int));
    int front = 0, rear = 0;

    dist[start] = 0;
    queue[rear++] = start;

    while (front < rear) {
        int v = queue[front++];

        AdjNode* node = g->adj[v];
        while (node) {
            if (dist[node->vertex] == -1) {
                dist[node->vertex] = dist[v] + 1;
                queue[rear++] = node->vertex;
            }
            node = node->next;
        }
    }

    free(queue);
    return dist;
}

/* =============================================================================
 * 5. 연결 요소
 * ============================================================================= */

int count_connected_components(Graph* g) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    int count = 0;

    for (int i = 0; i < g->vertices; i++) {
        if (!visited[i]) {
            dfs_recursive(g, i, visited);
            count++;
        }
    }

    free(visited);
    return count;
}

/* =============================================================================
 * 6. 사이클 탐지
 * ============================================================================= */

/* 무방향 그래프 사이클 탐지 */
bool has_cycle_undirected_util(Graph* g, int v, bool visited[], int parent) {
    visited[v] = true;

    AdjNode* node = g->adj[v];
    while (node) {
        if (!visited[node->vertex]) {
            if (has_cycle_undirected_util(g, node->vertex, visited, v))
                return true;
        } else if (node->vertex != parent) {
            return true;
        }
        node = node->next;
    }

    return false;
}

bool has_cycle_undirected(Graph* g) {
    bool* visited = calloc(g->vertices, sizeof(bool));

    for (int i = 0; i < g->vertices; i++) {
        if (!visited[i]) {
            if (has_cycle_undirected_util(g, i, visited, -1)) {
                free(visited);
                return true;
            }
        }
    }

    free(visited);
    return false;
}

/* 방향 그래프 사이클 탐지 (3색 알고리즘) */
bool has_cycle_directed_util(Graph* g, int v, int color[]) {
    color[v] = 1;  /* 회색: 처리 중 */

    AdjNode* node = g->adj[v];
    while (node) {
        if (color[node->vertex] == 1)
            return true;  /* 회색 노드 발견 = 사이클 */
        if (color[node->vertex] == 0) {
            if (has_cycle_directed_util(g, node->vertex, color))
                return true;
        }
        node = node->next;
    }

    color[v] = 2;  /* 검은색: 완료 */
    return false;
}

bool has_cycle_directed(Graph* g) {
    int* color = calloc(g->vertices, sizeof(int));

    for (int i = 0; i < g->vertices; i++) {
        if (color[i] == 0) {
            if (has_cycle_directed_util(g, i, color)) {
                free(color);
                return true;
            }
        }
    }

    free(color);
    return false;
}

/* =============================================================================
 * 7. 이분 그래프 판별
 * ============================================================================= */

bool is_bipartite(Graph* g) {
    int* color = malloc(g->vertices * sizeof(int));
    for (int i = 0; i < g->vertices; i++)
        color[i] = -1;

    int* queue = malloc(g->vertices * sizeof(int));

    for (int start = 0; start < g->vertices; start++) {
        if (color[start] != -1) continue;

        int front = 0, rear = 0;
        queue[rear++] = start;
        color[start] = 0;

        while (front < rear) {
            int v = queue[front++];

            AdjNode* node = g->adj[v];
            while (node) {
                if (color[node->vertex] == -1) {
                    color[node->vertex] = 1 - color[v];
                    queue[rear++] = node->vertex;
                } else if (color[node->vertex] == color[v]) {
                    free(color);
                    free(queue);
                    return false;
                }
                node = node->next;
            }
        }
    }

    free(color);
    free(queue);
    return true;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("그래프 기초 (Graph Basics) 예제\n");
    printf("============================================================\n");

    /* 1. 그래프 생성 */
    printf("\n[1] 그래프 생성 (인접 리스트)\n");
    Graph* g = graph_create(6, false);
    graph_add_edge(g, 0, 1, 1);
    graph_add_edge(g, 0, 2, 1);
    graph_add_edge(g, 1, 2, 1);
    graph_add_edge(g, 1, 3, 1);
    graph_add_edge(g, 2, 4, 1);
    graph_add_edge(g, 3, 4, 1);
    graph_add_edge(g, 4, 5, 1);

    printf("    그래프 구조:\n");
    graph_print(g);

    /* 2. DFS */
    printf("\n[2] DFS (0에서 시작)\n");
    printf("    재귀: ");
    dfs(g, 0);
    printf("\n");
    printf("    반복: ");
    dfs_iterative(g, 0);
    printf("\n");

    /* 3. BFS */
    printf("\n[3] BFS (0에서 시작)\n");
    printf("    순서: ");
    bfs(g, 0);
    printf("\n");

    int* distances = bfs_distances(g, 0);
    printf("    거리: ");
    for (int i = 0; i < 6; i++)
        printf("%d->%d ", i, distances[i]);
    printf("\n");
    free(distances);

    /* 4. 연결 요소 */
    printf("\n[4] 연결 요소\n");
    printf("    현재 그래프: %d개\n", count_connected_components(g));

    Graph* g2 = graph_create(6, false);
    graph_add_edge(g2, 0, 1, 1);
    graph_add_edge(g2, 2, 3, 1);
    graph_add_edge(g2, 4, 5, 1);
    printf("    분리된 그래프: %d개\n", count_connected_components(g2));
    graph_free(g2);

    /* 5. 사이클 탐지 */
    printf("\n[5] 사이클 탐지\n");
    printf("    무방향 그래프 사이클: %s\n",
           has_cycle_undirected(g) ? "있음" : "없음");

    Graph* dag = graph_create(4, true);
    graph_add_edge(dag, 0, 1, 1);
    graph_add_edge(dag, 1, 2, 1);
    graph_add_edge(dag, 2, 3, 1);
    printf("    방향 그래프 (DAG) 사이클: %s\n",
           has_cycle_directed(dag) ? "있음" : "없음");

    graph_add_edge(dag, 3, 1, 1);  /* 사이클 추가 */
    printf("    방향 그래프 (with cycle) 사이클: %s\n",
           has_cycle_directed(dag) ? "있음" : "없음");
    graph_free(dag);

    /* 6. 이분 그래프 */
    printf("\n[6] 이분 그래프 판별\n");
    Graph* bipartite = graph_create(4, false);
    graph_add_edge(bipartite, 0, 1, 1);
    graph_add_edge(bipartite, 0, 3, 1);
    graph_add_edge(bipartite, 1, 2, 1);
    graph_add_edge(bipartite, 2, 3, 1);
    printf("    4각형 그래프: %s\n",
           is_bipartite(bipartite) ? "이분 그래프" : "이분 그래프 아님");

    Graph* non_bipartite = graph_create(3, false);
    graph_add_edge(non_bipartite, 0, 1, 1);
    graph_add_edge(non_bipartite, 1, 2, 1);
    graph_add_edge(non_bipartite, 2, 0, 1);
    printf("    삼각형 그래프: %s\n",
           is_bipartite(non_bipartite) ? "이분 그래프" : "이분 그래프 아님");
    graph_free(bipartite);
    graph_free(non_bipartite);

    graph_free(g);

    /* 7. 복잡도 요약 */
    printf("\n[7] 그래프 알고리즘 복잡도\n");
    printf("    | 알고리즘     | 시간복잡도  | 공간     |\n");
    printf("    |--------------|-------------|----------|\n");
    printf("    | DFS          | O(V + E)    | O(V)     |\n");
    printf("    | BFS          | O(V + E)    | O(V)     |\n");
    printf("    | 연결 요소    | O(V + E)    | O(V)     |\n");
    printf("    | 사이클 탐지  | O(V + E)    | O(V)     |\n");
    printf("    | 이분 판별    | O(V + E)    | O(V)     |\n");

    printf("\n============================================================\n");

    return 0;
}
