/*
 * 최소 신장 트리 (Minimum Spanning Tree)
 * Kruskal, Prim, Union-Find
 *
 * 그래프의 모든 정점을 최소 비용으로 연결합니다.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>

using namespace std;

// =============================================================================
// 1. Union-Find (Disjoint Set Union)
// =============================================================================

class UnionFind {
private:
    vector<int> parent, rank_;

public:
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // 경로 압축
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        // 랭크 기반 합치기
        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;

        return true;
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};

// =============================================================================
// 2. Kruskal 알고리즘
// =============================================================================

struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

pair<int, vector<Edge>> kruskal(int n, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());

    UnionFind uf(n);
    vector<Edge> mst;
    int totalWeight = 0;

    for (const auto& e : edges) {
        if (uf.unite(e.u, e.v)) {
            mst.push_back(e);
            totalWeight += e.weight;

            if ((int)mst.size() == n - 1) break;
        }
    }

    return {totalWeight, mst};
}

// =============================================================================
// 3. Prim 알고리즘
// =============================================================================

pair<int, vector<pair<int, int>>> prim(int n, const vector<vector<pair<int, int>>>& adj) {
    vector<bool> visited(n, false);
    vector<pair<int, int>> mst;  // {u, v} 간선
    int totalWeight = 0;

    // {weight, vertex, parent}
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<>> pq;
    pq.push({0, 0, -1});

    while (!pq.empty() && (int)mst.size() < n) {
        auto [w, u, parent] = pq.top();
        pq.pop();

        if (visited[u]) continue;
        visited[u] = true;
        totalWeight += w;

        if (parent != -1) {
            mst.push_back({parent, u});
        }

        for (auto [v, weight] : adj[u]) {
            if (!visited[v]) {
                pq.push({weight, v, u});
            }
        }
    }

    return {totalWeight, mst};
}

// =============================================================================
// 4. 2차 최소 신장 트리
// =============================================================================

int secondMST(int n, vector<Edge>& edges) {
    // 먼저 MST 구하기
    auto [mstWeight, mst] = kruskal(n, edges);

    int secondBest = INT_MAX;

    // MST의 각 간선을 제거하고 다시 MST 구하기
    for (int i = 0; i < (int)mst.size(); i++) {
        vector<Edge> filtered;
        for (const auto& e : edges) {
            if (!(e.u == mst[i].u && e.v == mst[i].v && e.weight == mst[i].weight)) {
                filtered.push_back(e);
            }
        }

        auto [newWeight, newMst] = kruskal(n, filtered);

        if ((int)newMst.size() == n - 1) {
            secondBest = min(secondBest, newWeight);
        }
    }

    return secondBest;
}

// =============================================================================
// 5. 최대 신장 트리
// =============================================================================

pair<int, vector<Edge>> maxSpanningTree(int n, vector<Edge>& edges) {
    // 가중치를 음수로 바꿔서 Kruskal 적용
    for (auto& e : edges) {
        e.weight = -e.weight;
    }

    auto [weight, mst] = kruskal(n, edges);

    // 원래 가중치로 복원
    for (auto& e : edges) {
        e.weight = -e.weight;
    }
    for (auto& e : mst) {
        e.weight = -e.weight;
    }

    return {-weight, mst};
}

// =============================================================================
// 6. 크러스컬 응용: 연결 비용 최소화
// =============================================================================

int minCostToConnect(int n, vector<vector<int>>& connections) {
    vector<Edge> edges;
    for (const auto& conn : connections) {
        edges.push_back({conn[0] - 1, conn[1] - 1, conn[2]});
    }

    auto [cost, mst] = kruskal(n, edges);

    if ((int)mst.size() != n - 1) {
        return -1;  // 연결 불가
    }

    return cost;
}

// =============================================================================
// 7. Union-Find 응용: 친구 네트워크
// =============================================================================

class FriendNetwork {
private:
    unordered_map<string, string> parent;
    unordered_map<string, int> size_;

    string find(const string& x) {
        if (parent.find(x) == parent.end()) {
            parent[x] = x;
            size_[x] = 1;
        }
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

public:
    int unite(const string& a, const string& b) {
        string pa = find(a), pb = find(b);

        if (pa == pb) {
            return size_[pa];
        }

        if (size_[pa] < size_[pb]) swap(pa, pb);
        parent[pb] = pa;
        size_[pa] += size_[pb];

        return size_[pa];
    }
};

// =============================================================================
// 8. Union-Find 응용: 중복 연결 찾기
// =============================================================================

vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    UnionFind uf(n + 1);

    for (const auto& e : edges) {
        if (!uf.unite(e[0], e[1])) {
            return e;  // 중복 연결
        }
    }

    return {};
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "최소 신장 트리 예제" << endl;
    cout << "============================================================" << endl;

    // 테스트 그래프
    //     1 --(4)-- 2
    //    /|        /|
    //  (1)|      (3)|
    //  /  |      /  |
    // 0   (2)   /   (5)
    //  \  |  /      |
    //  (3)| /       |
    //    \|/        |
    //     3 --(6)-- 4

    int n = 5;
    vector<Edge> edges = {
        {0, 1, 1}, {0, 3, 3}, {1, 2, 4}, {1, 3, 2},
        {2, 3, 3}, {2, 4, 5}, {3, 4, 6}
    };

    // 1. Kruskal
    cout << "\n[1] Kruskal 알고리즘" << endl;
    auto [kWeight, kMst] = kruskal(n, edges);
    cout << "    MST 가중치: " << kWeight << endl;
    cout << "    MST 간선: ";
    for (const auto& e : kMst) {
        cout << "(" << e.u << "-" << e.v << ":" << e.weight << ") ";
    }
    cout << endl;

    // 2. Prim
    cout << "\n[2] Prim 알고리즘" << endl;
    vector<vector<pair<int, int>>> adj(n);
    for (const auto& e : edges) {
        adj[e.u].push_back({e.v, e.weight});
        adj[e.v].push_back({e.u, e.weight});
    }
    auto [pWeight, pMst] = prim(n, adj);
    cout << "    MST 가중치: " << pWeight << endl;
    cout << "    MST 간선: ";
    for (const auto& [u, v] : pMst) {
        cout << "(" << u << "-" << v << ") ";
    }
    cout << endl;

    // 3. Union-Find
    cout << "\n[3] Union-Find" << endl;
    UnionFind uf(5);
    uf.unite(0, 1);
    uf.unite(2, 3);
    uf.unite(1, 2);
    cout << "    합친 후: 0-1, 2-3, 1-2" << endl;
    cout << "    0과 3 연결됨: " << (uf.connected(0, 3) ? "예" : "아니오") << endl;
    cout << "    0과 4 연결됨: " << (uf.connected(0, 4) ? "예" : "아니오") << endl;

    // 4. 2차 MST
    cout << "\n[4] 2차 최소 신장 트리" << endl;
    vector<Edge> edges2 = edges;  // 복사
    int secondWeight = secondMST(n, edges2);
    cout << "    2차 MST 가중치: " << secondWeight << endl;

    // 5. 최대 신장 트리
    cout << "\n[5] 최대 신장 트리" << endl;
    vector<Edge> edges3 = edges;  // 복사
    auto [maxWeight, maxMst] = maxSpanningTree(n, edges3);
    cout << "    Max ST 가중치: " << maxWeight << endl;

    // 6. 중복 연결
    cout << "\n[6] 중복 연결 찾기" << endl;
    vector<vector<int>> redEdges = {{1, 2}, {1, 3}, {2, 3}};
    auto redundant = findRedundantConnection(redEdges);
    cout << "    간선: (1,2), (1,3), (2,3)" << endl;
    cout << "    중복: (" << redundant[0] << ", " << redundant[1] << ")" << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 알고리즘    | 시간복잡도       | 특징              |" << endl;
    cout << "    |-------------|------------------|-------------------|" << endl;
    cout << "    | Kruskal     | O(E log E)       | 간선 기반, 희소   |" << endl;
    cout << "    | Prim        | O((V+E) log V)   | 정점 기반, 밀집   |" << endl;
    cout << "    | Union-Find  | O(α(n)) ≈ O(1)   | 거의 상수 시간    |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
