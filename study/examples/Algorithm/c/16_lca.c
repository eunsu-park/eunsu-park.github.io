/*
 * LCA (Lowest Common Ancestor)
 * Binary Lifting, Euler Tour, Tree Path Queries
 *
 * 트리에서 두 노드의 최소 공통 조상을 찾습니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXN 100005
#define LOG 17

/* =============================================================================
 * 1. 트리 구조
 * ============================================================================= */

typedef struct AdjNode {
    int vertex;
    int weight;
    struct AdjNode* next;
} AdjNode;

AdjNode* adj[MAXN];
int depth[MAXN];
int parent[MAXN][LOG];  /* Binary Lifting 테이블 */
int dist[MAXN];         /* 루트로부터의 거리 */
int n;

void add_edge(int u, int v, int w) {
    AdjNode* node = malloc(sizeof(AdjNode));
    node->vertex = v;
    node->weight = w;
    node->next = adj[u];
    adj[u] = node;

    node = malloc(sizeof(AdjNode));
    node->vertex = u;
    node->weight = w;
    node->next = adj[v];
    adj[v] = node;
}

/* =============================================================================
 * 2. Binary Lifting 전처리
 * ============================================================================= */

void dfs_preprocess(int v, int p, int d, int distance) {
    depth[v] = d;
    parent[v][0] = p;
    dist[v] = distance;

    for (int i = 1; i < LOG; i++) {
        if (parent[v][i-1] != -1)
            parent[v][i] = parent[parent[v][i-1]][i-1];
        else
            parent[v][i] = -1;
    }

    AdjNode* node = adj[v];
    while (node) {
        if (node->vertex != p) {
            dfs_preprocess(node->vertex, v, d + 1, distance + node->weight);
        }
        node = node->next;
    }
}

void preprocess(int root) {
    memset(parent, -1, sizeof(parent));
    dfs_preprocess(root, -1, 0, 0);
}

/* =============================================================================
 * 3. LCA 쿼리
 * ============================================================================= */

int lca(int u, int v) {
    /* 깊이 맞추기 */
    if (depth[u] < depth[v]) {
        int temp = u; u = v; v = temp;
    }

    int diff = depth[u] - depth[v];
    for (int i = 0; i < LOG; i++) {
        if ((diff >> i) & 1) {
            u = parent[u][i];
        }
    }

    if (u == v) return u;

    /* 동시에 올라가기 */
    for (int i = LOG - 1; i >= 0; i--) {
        if (parent[u][i] != parent[v][i]) {
            u = parent[u][i];
            v = parent[v][i];
        }
    }

    return parent[u][0];
}

/* =============================================================================
 * 4. K번째 조상
 * ============================================================================= */

int kth_ancestor(int v, int k) {
    for (int i = 0; i < LOG && v != -1; i++) {
        if ((k >> i) & 1) {
            v = parent[v][i];
        }
    }
    return v;
}

/* =============================================================================
 * 5. 두 노드 사이 거리
 * ============================================================================= */

int distance_between(int u, int v) {
    int ancestor = lca(u, v);
    return dist[u] + dist[v] - 2 * dist[ancestor];
}

/* =============================================================================
 * 6. 경로 위 K번째 노드
 * ============================================================================= */

int kth_node_on_path(int u, int v, int k) {
    int ancestor = lca(u, v);
    int dist_u_lca = depth[u] - depth[ancestor];
    int dist_v_lca = depth[v] - depth[ancestor];

    if (k <= dist_u_lca) {
        return kth_ancestor(u, k);
    } else {
        return kth_ancestor(v, dist_u_lca + dist_v_lca - k);
    }
}

/* =============================================================================
 * 7. Euler Tour + RMQ 방식
 * ============================================================================= */

int euler_tour[2 * MAXN];
int first_occurrence[MAXN];
int euler_depth[2 * MAXN];
int euler_idx;

/* Sparse Table for RMQ */
int sparse_table[2 * MAXN][LOG];
int log_table[2 * MAXN];

void euler_dfs(int v, int p, int d) {
    first_occurrence[v] = euler_idx;
    euler_tour[euler_idx] = v;
    euler_depth[euler_idx] = d;
    euler_idx++;

    AdjNode* node = adj[v];
    while (node) {
        if (node->vertex != p) {
            euler_dfs(node->vertex, v, d + 1);
            euler_tour[euler_idx] = v;
            euler_depth[euler_idx] = d;
            euler_idx++;
        }
        node = node->next;
    }
}

void build_sparse_table(int len) {
    /* log 테이블 전처리 */
    log_table[1] = 0;
    for (int i = 2; i <= len; i++) {
        log_table[i] = log_table[i / 2] + 1;
    }

    /* Sparse Table 초기화 */
    for (int i = 0; i < len; i++) {
        sparse_table[i][0] = i;
    }

    for (int j = 1; (1 << j) <= len; j++) {
        for (int i = 0; i + (1 << j) - 1 < len; i++) {
            int left = sparse_table[i][j-1];
            int right = sparse_table[i + (1 << (j-1))][j-1];
            sparse_table[i][j] = (euler_depth[left] < euler_depth[right]) ? left : right;
        }
    }
}

int rmq_query(int l, int r) {
    int k = log_table[r - l + 1];
    int left = sparse_table[l][k];
    int right = sparse_table[r - (1 << k) + 1][k];
    return (euler_depth[left] < euler_depth[right]) ? left : right;
}

int lca_rmq(int u, int v) {
    int l = first_occurrence[u];
    int r = first_occurrence[v];
    if (l > r) {
        int temp = l; l = r; r = temp;
    }
    return euler_tour[rmq_query(l, r)];
}

void preprocess_euler(int root) {
    euler_idx = 0;
    euler_dfs(root, -1, 0);
    build_sparse_table(euler_idx);
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

void cleanup(void) {
    for (int i = 0; i < n; i++) {
        AdjNode* node = adj[i];
        while (node) {
            AdjNode* temp = node;
            node = node->next;
            free(temp);
        }
        adj[i] = NULL;
    }
}

int main(void) {
    printf("============================================================\n");
    printf("LCA (Lowest Common Ancestor) 예제\n");
    printf("============================================================\n");

    /*
     * 트리 구조:
     *        0
     *       /|\
     *      1 2 3
     *     /|   |
     *    4 5   6
     *   /
     *  7
     */
    n = 8;
    add_edge(0, 1, 1);
    add_edge(0, 2, 2);
    add_edge(0, 3, 3);
    add_edge(1, 4, 1);
    add_edge(1, 5, 1);
    add_edge(3, 6, 2);
    add_edge(4, 7, 1);

    /* 1. Binary Lifting 전처리 */
    printf("\n[1] Binary Lifting LCA\n");
    preprocess(0);

    printf("    트리: 0-1-4-7, 0-1-5, 0-2, 0-3-6\n");
    printf("    LCA(4, 5) = %d\n", lca(4, 5));
    printf("    LCA(4, 6) = %d\n", lca(4, 6));
    printf("    LCA(7, 2) = %d\n", lca(7, 2));
    printf("    LCA(5, 7) = %d\n", lca(5, 7));

    /* 2. K번째 조상 */
    printf("\n[2] K번째 조상\n");
    printf("    7의 1번째 조상: %d\n", kth_ancestor(7, 1));
    printf("    7의 2번째 조상: %d\n", kth_ancestor(7, 2));
    printf("    7의 3번째 조상: %d\n", kth_ancestor(7, 3));

    /* 3. 거리 */
    printf("\n[3] 두 노드 사이 거리\n");
    printf("    dist(4, 5) = %d\n", distance_between(4, 5));
    printf("    dist(7, 6) = %d\n", distance_between(7, 6));
    printf("    dist(7, 2) = %d\n", distance_between(7, 2));

    /* 4. 경로 위 K번째 노드 */
    printf("\n[4] 경로 위 K번째 노드\n");
    printf("    7→6 경로의 0번째 노드: %d\n", kth_node_on_path(7, 6, 0));
    printf("    7→6 경로의 2번째 노드: %d\n", kth_node_on_path(7, 6, 2));
    printf("    7→6 경로의 4번째 노드: %d\n", kth_node_on_path(7, 6, 4));

    /* 5. Euler Tour + RMQ */
    printf("\n[5] Euler Tour + RMQ LCA\n");
    preprocess_euler(0);
    printf("    LCA_RMQ(4, 5) = %d\n", lca_rmq(4, 5));
    printf("    LCA_RMQ(7, 6) = %d\n", lca_rmq(7, 6));

    /* 6. 복잡도 비교 */
    printf("\n[6] LCA 알고리즘 비교\n");
    printf("    | 방법           | 전처리    | 쿼리      | 공간      |\n");
    printf("    |----------------|-----------|-----------|------------|\n");
    printf("    | Binary Lifting | O(n log n)| O(log n)  | O(n log n) |\n");
    printf("    | Euler + RMQ    | O(n log n)| O(1)      | O(n log n) |\n");
    printf("    | Tarjan (오프라인)| O(n + q) | O(α(n))   | O(n)       |\n");

    cleanup();

    printf("\n============================================================\n");

    return 0;
}
