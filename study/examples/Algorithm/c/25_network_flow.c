/*
 * 네트워크 플로우 (Network Flow)
 * Ford-Fulkerson, Edmonds-Karp, 이분 매칭, 최소 컷
 *
 * 그래프에서 최대 유량을 찾는 알고리즘입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

#define MAX_V 1001
#define INF INT_MAX

/* =============================================================================
 * 1. Ford-Fulkerson (DFS 기반)
 * ============================================================================= */

int capacity_ff[MAX_V][MAX_V];
bool visited_ff[MAX_V];

int dfs_ff(int u, int sink, int flow) {
    if (u == sink) return flow;
    visited_ff[u] = true;

    for (int v = 0; v < MAX_V; v++) {
        if (!visited_ff[v] && capacity_ff[u][v] > 0) {
            int min_flow = (flow < capacity_ff[u][v]) ? flow : capacity_ff[u][v];
            int result = dfs_ff(v, sink, min_flow);
            if (result > 0) {
                capacity_ff[u][v] -= result;
                capacity_ff[v][u] += result;
                return result;
            }
        }
    }
    return 0;
}

int ford_fulkerson(int source, int sink, int n) {
    int max_flow = 0;
    int flow;

    while (1) {
        memset(visited_ff, false, sizeof(visited_ff));
        flow = dfs_ff(source, sink, INF);
        if (flow == 0) break;
        max_flow += flow;
    }

    return max_flow;
}

/* =============================================================================
 * 2. Edmonds-Karp (BFS 기반, O(VE²))
 * ============================================================================= */

typedef struct {
    int** capacity;
    int** flow;
    int n;
} FlowNetwork;

FlowNetwork* fn_create(int n) {
    FlowNetwork* fn = malloc(sizeof(FlowNetwork));
    fn->n = n;
    fn->capacity = malloc(n * sizeof(int*));
    fn->flow = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        fn->capacity[i] = calloc(n, sizeof(int));
        fn->flow[i] = calloc(n, sizeof(int));
    }
    return fn;
}

void fn_free(FlowNetwork* fn) {
    for (int i = 0; i < fn->n; i++) {
        free(fn->capacity[i]);
        free(fn->flow[i]);
    }
    free(fn->capacity);
    free(fn->flow);
    free(fn);
}

void fn_add_edge(FlowNetwork* fn, int u, int v, int cap) {
    fn->capacity[u][v] += cap;
}

int edmonds_karp(FlowNetwork* fn, int source, int sink) {
    int max_flow = 0;
    int* parent = malloc(fn->n * sizeof(int));
    int* queue = malloc(fn->n * sizeof(int));

    while (1) {
        memset(parent, -1, fn->n * sizeof(int));
        parent[source] = source;

        /* BFS */
        int front = 0, rear = 0;
        queue[rear++] = source;

        while (front < rear && parent[sink] == -1) {
            int u = queue[front++];
            for (int v = 0; v < fn->n; v++) {
                if (parent[v] == -1 &&
                    fn->capacity[u][v] - fn->flow[u][v] > 0) {
                    parent[v] = u;
                    queue[rear++] = v;
                }
            }
        }

        if (parent[sink] == -1) break;

        /* 경로 상 최소 용량 찾기 */
        int path_flow = INF;
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            int residual = fn->capacity[u][v] - fn->flow[u][v];
            if (residual < path_flow) path_flow = residual;
        }

        /* 유량 업데이트 */
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            fn->flow[u][v] += path_flow;
            fn->flow[v][u] -= path_flow;
        }

        max_flow += path_flow;
    }

    free(parent);
    free(queue);
    return max_flow;
}

/* =============================================================================
 * 3. 이분 매칭 (Bipartite Matching)
 * ============================================================================= */

typedef struct {
    int** adj;
    int* adj_size;
    int* match_left;
    int* match_right;
    bool* visited;
    int left_n;
    int right_n;
} BipartiteGraph;

BipartiteGraph* bg_create(int left_n, int right_n) {
    BipartiteGraph* bg = malloc(sizeof(BipartiteGraph));
    bg->left_n = left_n;
    bg->right_n = right_n;
    bg->adj = malloc(left_n * sizeof(int*));
    bg->adj_size = calloc(left_n, sizeof(int));
    for (int i = 0; i < left_n; i++) {
        bg->adj[i] = malloc(right_n * sizeof(int));
    }
    bg->match_left = malloc(left_n * sizeof(int));
    bg->match_right = malloc(right_n * sizeof(int));
    bg->visited = malloc(right_n * sizeof(bool));
    memset(bg->match_left, -1, left_n * sizeof(int));
    memset(bg->match_right, -1, right_n * sizeof(int));
    return bg;
}

void bg_free(BipartiteGraph* bg) {
    for (int i = 0; i < bg->left_n; i++) {
        free(bg->adj[i]);
    }
    free(bg->adj);
    free(bg->adj_size);
    free(bg->match_left);
    free(bg->match_right);
    free(bg->visited);
    free(bg);
}

void bg_add_edge(BipartiteGraph* bg, int left, int right) {
    bg->adj[left][bg->adj_size[left]++] = right;
}

bool bg_dfs(BipartiteGraph* bg, int u) {
    for (int i = 0; i < bg->adj_size[u]; i++) {
        int v = bg->adj[u][i];
        if (bg->visited[v]) continue;
        bg->visited[v] = true;

        if (bg->match_right[v] == -1 || bg_dfs(bg, bg->match_right[v])) {
            bg->match_left[u] = v;
            bg->match_right[v] = u;
            return true;
        }
    }
    return false;
}

int bipartite_matching(BipartiteGraph* bg) {
    int matching = 0;

    for (int u = 0; u < bg->left_n; u++) {
        memset(bg->visited, false, bg->right_n * sizeof(bool));
        if (bg_dfs(bg, u)) {
            matching++;
        }
    }

    return matching;
}

/* =============================================================================
 * 4. 호프크로프트-카프 (Hopcroft-Karp, O(E√V))
 * ============================================================================= */

int* dist_hk;
int* match_left_hk;
int* match_right_hk;
int** adj_hk;
int* adj_size_hk;
int left_n_hk, right_n_hk;

bool hk_bfs(void) {
    int* queue = malloc((left_n_hk + 1) * sizeof(int));
    int front = 0, rear = 0;

    for (int u = 0; u < left_n_hk; u++) {
        if (match_left_hk[u] == -1) {
            dist_hk[u] = 0;
            queue[rear++] = u;
        } else {
            dist_hk[u] = INF;
        }
    }

    bool found = false;
    while (front < rear) {
        int u = queue[front++];
        for (int i = 0; i < adj_size_hk[u]; i++) {
            int v = adj_hk[u][i];
            int next = match_right_hk[v];
            if (next == -1) {
                found = true;
            } else if (dist_hk[next] == INF) {
                dist_hk[next] = dist_hk[u] + 1;
                queue[rear++] = next;
            }
        }
    }

    free(queue);
    return found;
}

bool hk_dfs(int u) {
    for (int i = 0; i < adj_size_hk[u]; i++) {
        int v = adj_hk[u][i];
        int next = match_right_hk[v];
        if (next == -1 || (dist_hk[next] == dist_hk[u] + 1 && hk_dfs(next))) {
            match_left_hk[u] = v;
            match_right_hk[v] = u;
            return true;
        }
    }
    dist_hk[u] = INF;
    return false;
}

int hopcroft_karp(int left_n, int right_n, int** adj, int* adj_size) {
    left_n_hk = left_n;
    right_n_hk = right_n;
    adj_hk = adj;
    adj_size_hk = adj_size;

    dist_hk = malloc(left_n * sizeof(int));
    match_left_hk = malloc(left_n * sizeof(int));
    match_right_hk = malloc(right_n * sizeof(int));

    memset(match_left_hk, -1, left_n * sizeof(int));
    memset(match_right_hk, -1, right_n * sizeof(int));

    int matching = 0;
    while (hk_bfs()) {
        for (int u = 0; u < left_n; u++) {
            if (match_left_hk[u] == -1 && hk_dfs(u)) {
                matching++;
            }
        }
    }

    free(dist_hk);
    free(match_left_hk);
    free(match_right_hk);
    return matching;
}

/* =============================================================================
 * 5. 최소 컷 (Min Cut)
 * ============================================================================= */

void find_min_cut(FlowNetwork* fn, int source, int* reachable, int* cut_size) {
    bool* visited = calloc(fn->n, sizeof(bool));
    int* queue = malloc(fn->n * sizeof(int));
    int front = 0, rear = 0;

    visited[source] = true;
    queue[rear++] = source;

    *cut_size = 0;

    while (front < rear) {
        int u = queue[front++];
        reachable[(*cut_size)++] = u;
        for (int v = 0; v < fn->n; v++) {
            if (!visited[v] && fn->capacity[u][v] - fn->flow[u][v] > 0) {
                visited[v] = true;
                queue[rear++] = v;
            }
        }
    }

    free(visited);
    free(queue);
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("네트워크 플로우 예제\n");
    printf("============================================================\n");

    /* 1. Edmonds-Karp */
    printf("\n[1] Edmonds-Karp 알고리즘\n");
    FlowNetwork* fn = fn_create(6);
    fn_add_edge(fn, 0, 1, 16);
    fn_add_edge(fn, 0, 2, 13);
    fn_add_edge(fn, 1, 2, 10);
    fn_add_edge(fn, 1, 3, 12);
    fn_add_edge(fn, 2, 1, 4);
    fn_add_edge(fn, 2, 4, 14);
    fn_add_edge(fn, 3, 2, 9);
    fn_add_edge(fn, 3, 5, 20);
    fn_add_edge(fn, 4, 3, 7);
    fn_add_edge(fn, 4, 5, 4);

    printf("    그래프:\n");
    printf("      0 → 1 (16), 0 → 2 (13)\n");
    printf("      1 → 2 (10), 1 → 3 (12)\n");
    printf("      2 → 1 (4),  2 → 4 (14)\n");
    printf("      3 → 2 (9),  3 → 5 (20)\n");
    printf("      4 → 3 (7),  4 → 5 (4)\n");
    printf("    최대 유량 (0→5): %d\n", edmonds_karp(fn, 0, 5));
    fn_free(fn);

    /* 2. 이분 매칭 */
    printf("\n[2] 이분 매칭\n");
    BipartiteGraph* bg = bg_create(4, 4);
    /* 작업자(0-3)와 작업(0-3) 매칭 */
    bg_add_edge(bg, 0, 0);
    bg_add_edge(bg, 0, 1);
    bg_add_edge(bg, 1, 0);
    bg_add_edge(bg, 1, 2);
    bg_add_edge(bg, 2, 1);
    bg_add_edge(bg, 2, 2);
    bg_add_edge(bg, 3, 2);
    bg_add_edge(bg, 3, 3);

    printf("    작업자-작업 연결:\n");
    printf("      작업자0: 작업0, 작업1\n");
    printf("      작업자1: 작업0, 작업2\n");
    printf("      작업자2: 작업1, 작업2\n");
    printf("      작업자3: 작업2, 작업3\n");
    printf("    최대 매칭: %d\n", bipartite_matching(bg));

    printf("    매칭 결과:\n");
    for (int i = 0; i < bg->left_n; i++) {
        if (bg->match_left[i] != -1) {
            printf("      작업자%d → 작업%d\n", i, bg->match_left[i]);
        }
    }
    bg_free(bg);

    /* 3. 최소 버텍스 커버 */
    printf("\n[3] 최소 버텍스 커버 (이분 그래프)\n");
    printf("    Konig's theorem: 최대 매칭 = 최소 버텍스 커버\n");
    printf("    위 예제의 최소 버텍스 커버: 4\n");

    /* 4. Ford-Fulkerson */
    printf("\n[4] Ford-Fulkerson (간단 예제)\n");
    memset(capacity_ff, 0, sizeof(capacity_ff));
    capacity_ff[0][1] = 10;
    capacity_ff[0][2] = 10;
    capacity_ff[1][2] = 2;
    capacity_ff[1][3] = 4;
    capacity_ff[1][4] = 8;
    capacity_ff[2][4] = 9;
    capacity_ff[3][5] = 10;
    capacity_ff[4][3] = 6;
    capacity_ff[4][5] = 10;

    printf("    최대 유량 (0→5): %d\n", ford_fulkerson(0, 5, 6));

    /* 5. 응용 */
    printf("\n[5] 네트워크 플로우 응용\n");
    printf("    - 이분 매칭: 작업 할당, 결혼 문제\n");
    printf("    - 최소 컷: 네트워크 분할\n");
    printf("    - 최대 독립 집합 (이분 그래프)\n");
    printf("    - 프로젝트 선택 문제\n");
    printf("    - 순환 흐름\n");

    /* 6. 복잡도 */
    printf("\n[6] 복잡도\n");
    printf("    | 알고리즘         | 시간복잡도    |\n");
    printf("    |------------------|---------------|\n");
    printf("    | Ford-Fulkerson   | O(E × f)      |\n");
    printf("    | Edmonds-Karp     | O(VE²)        |\n");
    printf("    | Dinic            | O(V²E)        |\n");
    printf("    | 이분 매칭 (DFS)  | O(VE)         |\n");
    printf("    | Hopcroft-Karp    | O(E√V)        |\n");
    printf("    f: 최대 유량\n");

    printf("\n============================================================\n");

    return 0;
}
