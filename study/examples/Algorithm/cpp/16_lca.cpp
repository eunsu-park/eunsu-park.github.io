/*
 * LCA와 트리 쿼리 (LCA and Tree Queries)
 * Binary Lifting, Sparse Table, Euler Tour, HLD 기초
 *
 * 트리에서 최소 공통 조상을 찾는 알고리즘입니다.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// =============================================================================
// 1. Binary Lifting LCA
// =============================================================================

class LCABinaryLifting {
private:
    int n, LOG;
    vector<vector<int>> adj;
    vector<vector<int>> up;  // up[v][j] = 2^j번째 조상
    vector<int> depth;

    void dfs(int v, int p, int d) {
        depth[v] = d;
        up[v][0] = p;

        for (int j = 1; j < LOG; j++) {
            if (up[v][j-1] != -1) {
                up[v][j] = up[up[v][j-1]][j-1];
            }
        }

        for (int u : adj[v]) {
            if (u != p) {
                dfs(u, v, d + 1);
            }
        }
    }

public:
    LCABinaryLifting(int n, const vector<vector<int>>& adj, int root = 0)
        : n(n), adj(adj) {
        LOG = (int)ceil(log2(n + 1)) + 1;
        up.assign(n, vector<int>(LOG, -1));
        depth.assign(n, 0);
        dfs(root, -1, 0);
    }

    int getDepth(int v) const {
        return depth[v];
    }

    int kthAncestor(int v, int k) {
        for (int j = 0; j < LOG && v != -1; j++) {
            if ((k >> j) & 1) {
                v = up[v][j];
            }
        }
        return v;
    }

    int lca(int u, int v) {
        if (depth[u] < depth[v]) swap(u, v);

        // 같은 깊이로 맞추기
        u = kthAncestor(u, depth[u] - depth[v]);

        if (u == v) return u;

        // 함께 올라가기
        for (int j = LOG - 1; j >= 0; j--) {
            if (up[u][j] != up[v][j]) {
                u = up[u][j];
                v = up[v][j];
            }
        }

        return up[u][0];
    }

    int distance(int u, int v) {
        return depth[u] + depth[v] - 2 * depth[lca(u, v)];
    }
};

// =============================================================================
// 2. Euler Tour + RMQ
// =============================================================================

class LCAEulerTour {
private:
    int n;
    vector<vector<int>> adj;
    vector<int> euler;       // Euler tour
    vector<int> first;       // 첫 등장 위치
    vector<int> depth;
    vector<vector<int>> sparse;  // Sparse table

    void dfs(int v, int p, int d) {
        first[v] = euler.size();
        euler.push_back(v);
        depth[v] = d;

        for (int u : adj[v]) {
            if (u != p) {
                dfs(u, v, d + 1);
                euler.push_back(v);
            }
        }
    }

    void buildSparseTable() {
        int m = euler.size();
        int LOG = (int)ceil(log2(m + 1)) + 1;
        sparse.assign(LOG, vector<int>(m));

        for (int i = 0; i < m; i++) {
            sparse[0][i] = euler[i];
        }

        for (int j = 1; j < LOG; j++) {
            for (int i = 0; i + (1 << j) <= m; i++) {
                int left = sparse[j-1][i];
                int right = sparse[j-1][i + (1 << (j-1))];
                sparse[j][i] = (depth[left] < depth[right]) ? left : right;
            }
        }
    }

public:
    LCAEulerTour(int n, const vector<vector<int>>& adj, int root = 0)
        : n(n), adj(adj), first(n), depth(n) {
        dfs(root, -1, 0);
        buildSparseTable();
    }

    int lca(int u, int v) {
        int l = first[u], r = first[v];
        if (l > r) swap(l, r);

        int len = r - l + 1;
        int k = (int)log2(len);

        int left = sparse[k][l];
        int right = sparse[k][r - (1 << k) + 1];

        return (depth[left] < depth[right]) ? left : right;
    }
};

// =============================================================================
// 3. 트리에서 경로 쿼리
// =============================================================================

class TreePathQuery {
private:
    LCABinaryLifting lca;
    vector<long long> prefixSum;  // 루트에서 각 노드까지의 합
    vector<int> value;

public:
    TreePathQuery(int n, const vector<vector<int>>& adj,
                  const vector<int>& values, int root = 0)
        : lca(n, adj, root), prefixSum(n, 0), value(values) {

        // DFS로 prefix sum 계산
        function<void(int, int, long long)> dfs = [&](int v, int p, long long sum) {
            sum += value[v];
            prefixSum[v] = sum;
            for (int u : adj[v]) {
                if (u != p) dfs(u, v, sum);
            }
        };
        dfs(root, -1, 0);
    }

    long long pathSum(int u, int v) {
        int l = lca.lca(u, v);
        return prefixSum[u] + prefixSum[v] - 2 * prefixSum[l] + value[l];
    }

    int pathLength(int u, int v) {
        return lca.distance(u, v);
    }
};

// =============================================================================
// 4. 트리 직경
// =============================================================================

pair<int, pair<int, int>> treeDiameter(int n, const vector<vector<int>>& adj) {
    vector<int> dist(n, -1);

    // 첫 번째 BFS
    auto bfs = [&](int start) -> int {
        fill(dist.begin(), dist.end(), -1);
        queue<int> q;
        q.push(start);
        dist[start] = 0;
        int farthest = start;

        while (!q.empty()) {
            int v = q.front();
            q.pop();

            for (int u : adj[v]) {
                if (dist[u] == -1) {
                    dist[u] = dist[v] + 1;
                    q.push(u);
                    if (dist[u] > dist[farthest]) {
                        farthest = u;
                    }
                }
            }
        }

        return farthest;
    };

    int u = bfs(0);      // 가장 먼 점 찾기
    int v = bfs(u);      // u에서 가장 먼 점 찾기

    return {dist[v], {u, v}};
}

// =============================================================================
// 5. 트리의 중심 (Centroid)
// =============================================================================

int treeCentroid(int n, const vector<vector<int>>& adj) {
    vector<int> subtreeSize(n);

    function<void(int, int)> calcSize = [&](int v, int p) {
        subtreeSize[v] = 1;
        for (int u : adj[v]) {
            if (u != p) {
                calcSize(u, v);
                subtreeSize[v] += subtreeSize[u];
            }
        }
    };

    calcSize(0, -1);

    function<int(int, int)> findCentroid = [&](int v, int p) -> int {
        for (int u : adj[v]) {
            if (u != p && subtreeSize[u] > n / 2) {
                return findCentroid(u, v);
            }
        }
        return v;
    };

    return findCentroid(0, -1);
}

// =============================================================================
// 6. 가중치 LCA
// =============================================================================

class WeightedLCA {
private:
    int n, LOG;
    vector<vector<pair<int, int>>> adj;  // {neighbor, weight}
    vector<vector<int>> up;
    vector<vector<int>> maxWeight;  // 조상 경로의 최대 가중치
    vector<int> depth;

    void dfs(int v, int p, int d, int w) {
        depth[v] = d;
        up[v][0] = p;
        maxWeight[v][0] = w;

        for (int j = 1; j < LOG; j++) {
            if (up[v][j-1] != -1) {
                up[v][j] = up[up[v][j-1]][j-1];
                maxWeight[v][j] = max(maxWeight[v][j-1],
                                      maxWeight[up[v][j-1]][j-1]);
            }
        }

        for (auto [u, weight] : adj[v]) {
            if (u != p) {
                dfs(u, v, d + 1, weight);
            }
        }
    }

public:
    WeightedLCA(int n, const vector<vector<pair<int, int>>>& adj, int root = 0)
        : n(n), adj(adj) {
        LOG = (int)ceil(log2(n + 1)) + 1;
        up.assign(n, vector<int>(LOG, -1));
        maxWeight.assign(n, vector<int>(LOG, 0));
        depth.assign(n, 0);
        dfs(root, -1, 0, 0);
    }

    pair<int, int> lcaWithMaxWeight(int u, int v) {
        int maxW = 0;

        if (depth[u] < depth[v]) swap(u, v);
        int diff = depth[u] - depth[v];

        for (int j = 0; j < LOG; j++) {
            if ((diff >> j) & 1) {
                maxW = max(maxW, maxWeight[u][j]);
                u = up[u][j];
            }
        }

        if (u == v) return {u, maxW};

        for (int j = LOG - 1; j >= 0; j--) {
            if (up[u][j] != up[v][j]) {
                maxW = max(maxW, max(maxWeight[u][j], maxWeight[v][j]));
                u = up[u][j];
                v = up[v][j];
            }
        }

        maxW = max(maxW, max(maxWeight[u][0], maxWeight[v][0]));
        return {up[u][0], maxW};
    }
};

// =============================================================================
// 테스트
// =============================================================================

#include <queue>

int main() {
    cout << "============================================================" << endl;
    cout << "LCA와 트리 쿼리 예제" << endl;
    cout << "============================================================" << endl;

    // 테스트 트리
    //        0
    //       /|\
    //      1 2 3
    //     /|   |
    //    4 5   6
    //    |
    //    7

    int n = 8;
    vector<vector<int>> adj(n);
    adj[0] = {1, 2, 3};
    adj[1] = {0, 4, 5};
    adj[2] = {0};
    adj[3] = {0, 6};
    adj[4] = {1, 7};
    adj[5] = {1};
    adj[6] = {3};
    adj[7] = {4};

    // 1. Binary Lifting LCA
    cout << "\n[1] Binary Lifting LCA" << endl;
    LCABinaryLifting lcaBL(n, adj, 0);
    cout << "    LCA(4, 5) = " << lcaBL.lca(4, 5) << endl;
    cout << "    LCA(7, 6) = " << lcaBL.lca(7, 6) << endl;
    cout << "    LCA(7, 5) = " << lcaBL.lca(7, 5) << endl;
    cout << "    거리(7, 6) = " << lcaBL.distance(7, 6) << endl;

    // 2. K번째 조상
    cout << "\n[2] K번째 조상" << endl;
    cout << "    7의 1번째 조상: " << lcaBL.kthAncestor(7, 1) << endl;
    cout << "    7의 2번째 조상: " << lcaBL.kthAncestor(7, 2) << endl;
    cout << "    7의 3번째 조상: " << lcaBL.kthAncestor(7, 3) << endl;

    // 3. Euler Tour LCA
    cout << "\n[3] Euler Tour LCA" << endl;
    LCAEulerTour lcaET(n, adj, 0);
    cout << "    LCA(4, 5) = " << lcaET.lca(4, 5) << endl;
    cout << "    LCA(7, 6) = " << lcaET.lca(7, 6) << endl;

    // 4. 트리 직경
    cout << "\n[4] 트리 직경" << endl;
    auto [diameter, endpoints] = treeDiameter(n, adj);
    cout << "    직경: " << diameter << endl;
    cout << "    끝점: (" << endpoints.first << ", " << endpoints.second << ")" << endl;

    // 5. 트리 중심
    cout << "\n[5] 트리 중심" << endl;
    int centroid = treeCentroid(n, adj);
    cout << "    중심: " << centroid << endl;

    // 6. 경로 쿼리
    cout << "\n[6] 경로 쿼리" << endl;
    vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8};
    TreePathQuery tpq(n, adj, values, 0);
    cout << "    노드 값: [1, 2, 3, 4, 5, 6, 7, 8]" << endl;
    cout << "    경로 합(7, 6): " << tpq.pathSum(7, 6) << endl;
    cout << "    경로 길이(7, 6): " << tpq.pathLength(7, 6) << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 알고리즘        | 전처리      | 쿼리      |" << endl;
    cout << "    |-----------------|-------------|-----------|" << endl;
    cout << "    | Binary Lifting  | O(N log N)  | O(log N)  |" << endl;
    cout << "    | Euler Tour+RMQ  | O(N log N)  | O(1)      |" << endl;
    cout << "    | Tarjan's Offline| O(N + Q)    | O(1)      |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
