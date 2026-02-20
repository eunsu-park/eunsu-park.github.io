/*
 * 강한 연결 요소 (SCC - Strongly Connected Components)
 * Kosaraju, Tarjan Algorithm
 *
 * 방향 그래프에서 서로 도달 가능한 정점 집합을 찾습니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAXN 1005

/* =============================================================================
 * 1. 그래프 구조
 * ============================================================================= */

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

Node* graph[MAXN];
Node* reverse_graph[MAXN];
int n, m;

void add_edge(Node** g, int u, int v) {
    Node* node = malloc(sizeof(Node));
    node->vertex = v;
    node->next = g[u];
    g[u] = node;
}

/* =============================================================================
 * 2. Kosaraju 알고리즘
 * ============================================================================= */

bool visited[MAXN];
int finish_stack[MAXN];
int stack_top;
int scc_id[MAXN];
int scc_count;

void dfs1(int v) {
    visited[v] = true;
    Node* node = graph[v];
    while (node) {
        if (!visited[node->vertex])
            dfs1(node->vertex);
        node = node->next;
    }
    finish_stack[stack_top++] = v;
}

void dfs2(int v, int scc_num) {
    visited[v] = true;
    scc_id[v] = scc_num;
    Node* node = reverse_graph[v];
    while (node) {
        if (!visited[node->vertex])
            dfs2(node->vertex, scc_num);
        node = node->next;
    }
}

int kosaraju(void) {
    /* 1단계: 정방향 DFS */
    memset(visited, false, sizeof(visited));
    stack_top = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i])
            dfs1(i);
    }

    /* 2단계: 역방향 DFS */
    memset(visited, false, sizeof(visited));
    scc_count = 0;
    for (int i = stack_top - 1; i >= 0; i--) {
        int v = finish_stack[i];
        if (!visited[v]) {
            dfs2(v, scc_count);
            scc_count++;
        }
    }

    return scc_count;
}

/* =============================================================================
 * 3. Tarjan 알고리즘
 * ============================================================================= */

int disc[MAXN];
int low[MAXN];
bool on_stack[MAXN];
int tarjan_stack[MAXN];
int tarjan_top;
int timer;
int tarjan_scc_count;
int tarjan_scc_id[MAXN];

void tarjan_dfs(int v) {
    disc[v] = low[v] = timer++;
    tarjan_stack[tarjan_top++] = v;
    on_stack[v] = true;

    Node* node = graph[v];
    while (node) {
        int u = node->vertex;
        if (disc[u] == -1) {
            tarjan_dfs(u);
            if (low[u] < low[v]) low[v] = low[u];
        } else if (on_stack[u]) {
            if (disc[u] < low[v]) low[v] = disc[u];
        }
        node = node->next;
    }

    /* SCC 발견 */
    if (low[v] == disc[v]) {
        while (true) {
            int u = tarjan_stack[--tarjan_top];
            on_stack[u] = false;
            tarjan_scc_id[u] = tarjan_scc_count;
            if (u == v) break;
        }
        tarjan_scc_count++;
    }
}

int tarjan(void) {
    memset(disc, -1, sizeof(disc));
    memset(low, -1, sizeof(low));
    memset(on_stack, false, sizeof(on_stack));
    timer = 0;
    tarjan_top = 0;
    tarjan_scc_count = 0;

    for (int i = 0; i < n; i++) {
        if (disc[i] == -1)
            tarjan_dfs(i);
    }

    return tarjan_scc_count;
}

/* =============================================================================
 * 4. SCC DAG 구성
 * ============================================================================= */

typedef struct {
    int* edges;
    int size;
    int capacity;
} EdgeSet;

EdgeSet scc_graph[MAXN];

void build_scc_dag(void) {
    for (int i = 0; i < scc_count; i++) {
        scc_graph[i].edges = malloc(n * sizeof(int));
        scc_graph[i].size = 0;
        scc_graph[i].capacity = n;
    }

    bool** added = malloc(scc_count * sizeof(bool*));
    for (int i = 0; i < scc_count; i++) {
        added[i] = calloc(scc_count, sizeof(bool));
    }

    for (int u = 0; u < n; u++) {
        Node* node = graph[u];
        while (node) {
            int v = node->vertex;
            int scc_u = scc_id[u];
            int scc_v = scc_id[v];
            if (scc_u != scc_v && !added[scc_u][scc_v]) {
                scc_graph[scc_u].edges[scc_graph[scc_u].size++] = scc_v;
                added[scc_u][scc_v] = true;
            }
            node = node->next;
        }
    }

    for (int i = 0; i < scc_count; i++) free(added[i]);
    free(added);
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

void cleanup(void) {
    for (int i = 0; i < n; i++) {
        Node* node = graph[i];
        while (node) {
            Node* temp = node;
            node = node->next;
            free(temp);
        }
        graph[i] = NULL;

        node = reverse_graph[i];
        while (node) {
            Node* temp = node;
            node = node->next;
            free(temp);
        }
        reverse_graph[i] = NULL;
    }
}

int main(void) {
    printf("============================================================\n");
    printf("강한 연결 요소 (SCC) 예제\n");
    printf("============================================================\n");

    /*
     * 그래프:
     *   0 → 1 → 2 → 0 (SCC: {0,1,2})
     *   ↓       ↓
     *   3 ← 4 ← 5 (SCC: {3,4,5})
     */
    n = 6;
    add_edge(graph, 0, 1);
    add_edge(graph, 1, 2);
    add_edge(graph, 2, 0);
    add_edge(graph, 0, 3);
    add_edge(graph, 2, 5);
    add_edge(graph, 5, 4);
    add_edge(graph, 4, 3);
    add_edge(graph, 3, 4);  /* 3,4,5 사이클 */

    /* 역그래프 구성 */
    add_edge(reverse_graph, 1, 0);
    add_edge(reverse_graph, 2, 1);
    add_edge(reverse_graph, 0, 2);
    add_edge(reverse_graph, 3, 0);
    add_edge(reverse_graph, 5, 2);
    add_edge(reverse_graph, 4, 5);
    add_edge(reverse_graph, 3, 4);
    add_edge(reverse_graph, 4, 3);

    /* 1. Kosaraju */
    printf("\n[1] Kosaraju 알고리즘\n");
    int num_scc = kosaraju();
    printf("    SCC 개수: %d\n", num_scc);
    printf("    노드별 SCC ID:\n    ");
    for (int i = 0; i < n; i++)
        printf("%d→SCC%d  ", i, scc_id[i]);
    printf("\n");

    /* 2. Tarjan */
    printf("\n[2] Tarjan 알고리즘\n");
    int num_scc2 = tarjan();
    printf("    SCC 개수: %d\n", num_scc2);
    printf("    노드별 SCC ID:\n    ");
    for (int i = 0; i < n; i++)
        printf("%d→SCC%d  ", i, tarjan_scc_id[i]);
    printf("\n");

    /* 3. SCC DAG */
    printf("\n[3] SCC 축약 그래프 (DAG)\n");
    build_scc_dag();
    printf("    SCC 간 간선:\n");
    for (int i = 0; i < scc_count; i++) {
        printf("      SCC%d → ", i);
        for (int j = 0; j < scc_graph[i].size; j++)
            printf("SCC%d ", scc_graph[i].edges[j]);
        printf("\n");
        free(scc_graph[i].edges);
    }

    /* 4. 알고리즘 비교 */
    printf("\n[4] 알고리즘 비교\n");
    printf("    | 알고리즘  | 시간복잡도 | DFS 횟수 | 특징          |\n");
    printf("    |-----------|------------|----------|---------------|\n");
    printf("    | Kosaraju  | O(V + E)   | 2번      | 역그래프 필요 |\n");
    printf("    | Tarjan    | O(V + E)   | 1번      | low-link 사용 |\n");

    cleanup();

    printf("\n============================================================\n");

    return 0;
}
