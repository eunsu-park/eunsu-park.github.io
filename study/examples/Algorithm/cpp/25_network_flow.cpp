/*
 * 네트워크 플로우 (Network Flow)
 * Ford-Fulkerson, Edmonds-Karp, Dinic, Bipartite Matching
 *
 * 그래프에서 최대 유량을 구하는 알고리즘입니다.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <cstring>

using namespace std;

const int INF = INT_MAX;

// =============================================================================
// 1. Ford-Fulkerson (DFS)
// =============================================================================

class FordFulkerson {
private:
    int n;
    vector<vector<int>> capacity;
    vector<vector<int>> adj;

    int dfs(int u, int t, int pushed, vector<bool>& visited) {
        if (u == t) return pushed;

        visited[u] = true;

        for (int v : adj[u]) {
            if (!visited[v] && capacity[u][v] > 0) {
                int flow = dfs(v, t, min(pushed, capacity[u][v]), visited);
                if (flow > 0) {
                    capacity[u][v] -= flow;
                    capacity[v][u] += flow;
                    return flow;
                }
            }
        }

        return 0;
    }

public:
    FordFulkerson(int n) : n(n), capacity(n, vector<int>(n, 0)), adj(n) {}

    void addEdge(int u, int v, int cap) {
        capacity[u][v] += cap;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int maxFlow(int s, int t) {
        int flow = 0;

        while (true) {
            vector<bool> visited(n, false);
            int pushed = dfs(s, t, INF, visited);
            if (pushed == 0) break;
            flow += pushed;
        }

        return flow;
    }
};

// =============================================================================
// 2. Edmonds-Karp (BFS)
// =============================================================================

class EdmondsKarp {
private:
    int n;
    vector<vector<int>> capacity;
    vector<vector<int>> adj;

public:
    EdmondsKarp(int n) : n(n), capacity(n, vector<int>(n, 0)), adj(n) {}

    void addEdge(int u, int v, int cap) {
        capacity[u][v] += cap;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int maxFlow(int s, int t) {
        int flow = 0;

        while (true) {
            vector<int> parent(n, -1);
            queue<int> q;
            q.push(s);
            parent[s] = s;

            while (!q.empty() && parent[t] == -1) {
                int u = q.front();
                q.pop();

                for (int v : adj[u]) {
                    if (parent[v] == -1 && capacity[u][v] > 0) {
                        parent[v] = u;
                        q.push(v);
                    }
                }
            }

            if (parent[t] == -1) break;

            // 경로상 최소 용량
            int pushed = INF;
            for (int v = t; v != s; v = parent[v]) {
                pushed = min(pushed, capacity[parent[v]][v]);
            }

            // 용량 업데이트
            for (int v = t; v != s; v = parent[v]) {
                capacity[parent[v]][v] -= pushed;
                capacity[v][parent[v]] += pushed;
            }

            flow += pushed;
        }

        return flow;
    }
};

// =============================================================================
// 3. Dinic's Algorithm
// =============================================================================

class Dinic {
private:
    struct Edge {
        int to, cap, rev;
    };

    int n;
    vector<vector<Edge>> graph;
    vector<int> level, iter;

    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        level[s] = 0;
        q.push(s);

        while (!q.empty()) {
            int v = q.front();
            q.pop();

            for (auto& e : graph[v]) {
                if (e.cap > 0 && level[e.to] < 0) {
                    level[e.to] = level[v] + 1;
                    q.push(e.to);
                }
            }
        }

        return level[t] >= 0;
    }

    int dfs(int v, int t, int f) {
        if (v == t) return f;

        for (int& i = iter[v]; i < (int)graph[v].size(); i++) {
            Edge& e = graph[v][i];
            if (e.cap > 0 && level[v] < level[e.to]) {
                int d = dfs(e.to, t, min(f, e.cap));
                if (d > 0) {
                    e.cap -= d;
                    graph[e.to][e.rev].cap += d;
                    return d;
                }
            }
        }

        return 0;
    }

public:
    Dinic(int n) : n(n), graph(n), level(n), iter(n) {}

    void addEdge(int from, int to, int cap) {
        graph[from].push_back({to, cap, (int)graph[to].size()});
        graph[to].push_back({from, 0, (int)graph[from].size() - 1});
    }

    int maxFlow(int s, int t) {
        int flow = 0;

        while (bfs(s, t)) {
            fill(iter.begin(), iter.end(), 0);
            int f;
            while ((f = dfs(s, t, INF)) > 0) {
                flow += f;
            }
        }

        return flow;
    }
};

// =============================================================================
// 4. 이분 매칭 (Hungarian / Hopcroft-Karp)
// =============================================================================

class BipartiteMatching {
private:
    int n, m;
    vector<vector<int>> adj;
    vector<int> matchL, matchR;
    vector<bool> visited;

    bool dfs(int u) {
        for (int v : adj[u]) {
            if (visited[v]) continue;
            visited[v] = true;

            if (matchR[v] == -1 || dfs(matchR[v])) {
                matchL[u] = v;
                matchR[v] = u;
                return true;
            }
        }
        return false;
    }

public:
    BipartiteMatching(int n, int m) : n(n), m(m), adj(n), matchL(n, -1), matchR(m, -1) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }

    int maxMatching() {
        int result = 0;

        for (int u = 0; u < n; u++) {
            visited.assign(m, false);
            if (dfs(u)) result++;
        }

        return result;
    }

    vector<pair<int, int>> getMatching() {
        vector<pair<int, int>> result;
        for (int u = 0; u < n; u++) {
            if (matchL[u] != -1) {
                result.push_back({u, matchL[u]});
            }
        }
        return result;
    }
};

// =============================================================================
// 5. 최소 비용 최대 유량 (MCMF)
// =============================================================================

class MCMF {
private:
    struct Edge {
        int to, cap, cost, rev;
    };

    int n;
    vector<vector<Edge>> graph;

public:
    MCMF(int n) : n(n), graph(n) {}

    void addEdge(int from, int to, int cap, int cost) {
        graph[from].push_back({to, cap, cost, (int)graph[to].size()});
        graph[to].push_back({from, 0, -cost, (int)graph[from].size() - 1});
    }

    pair<int, int> minCostMaxFlow(int s, int t) {
        int totalFlow = 0, totalCost = 0;

        while (true) {
            vector<int> dist(n, INF);
            vector<int> prevV(n, -1), prevE(n, -1);
            vector<bool> inQueue(n, false);
            queue<int> q;

            dist[s] = 0;
            q.push(s);
            inQueue[s] = true;

            while (!q.empty()) {
                int v = q.front();
                q.pop();
                inQueue[v] = false;

                for (int i = 0; i < (int)graph[v].size(); i++) {
                    Edge& e = graph[v][i];
                    if (e.cap > 0 && dist[v] + e.cost < dist[e.to]) {
                        dist[e.to] = dist[v] + e.cost;
                        prevV[e.to] = v;
                        prevE[e.to] = i;
                        if (!inQueue[e.to]) {
                            q.push(e.to);
                            inQueue[e.to] = true;
                        }
                    }
                }
            }

            if (dist[t] == INF) break;

            // 경로상 최소 용량
            int flow = INF;
            for (int v = t; v != s; v = prevV[v]) {
                flow = min(flow, graph[prevV[v]][prevE[v]].cap);
            }

            // 용량 업데이트
            for (int v = t; v != s; v = prevV[v]) {
                Edge& e = graph[prevV[v]][prevE[v]];
                e.cap -= flow;
                graph[v][e.rev].cap += flow;
            }

            totalFlow += flow;
            totalCost += flow * dist[t];
        }

        return {totalFlow, totalCost};
    }
};

// =============================================================================
// 6. 최소 컷
// =============================================================================

vector<pair<int, int>> minCut(int n, vector<vector<int>>& capacity, int s, int t) {
    // Edmonds-Karp 실행 후 잔여 그래프에서 s에서 도달 가능한 정점 찾기
    vector<vector<int>> residual = capacity;
    vector<vector<int>> adj(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (capacity[i][j] > 0) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    // Max flow
    while (true) {
        vector<int> parent(n, -1);
        queue<int> q;
        q.push(s);
        parent[s] = s;

        while (!q.empty() && parent[t] == -1) {
            int u = q.front();
            q.pop();

            for (int v : adj[u]) {
                if (parent[v] == -1 && residual[u][v] > 0) {
                    parent[v] = u;
                    q.push(v);
                }
            }
        }

        if (parent[t] == -1) break;

        int pushed = INF;
        for (int v = t; v != s; v = parent[v]) {
            pushed = min(pushed, residual[parent[v]][v]);
        }

        for (int v = t; v != s; v = parent[v]) {
            residual[parent[v]][v] -= pushed;
            residual[v][parent[v]] += pushed;
        }
    }

    // BFS로 s에서 도달 가능한 정점
    vector<bool> reachable(n, false);
    queue<int> q;
    q.push(s);
    reachable[s] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : adj[u]) {
            if (!reachable[v] && residual[u][v] > 0) {
                reachable[v] = true;
                q.push(v);
            }
        }
    }

    // 최소 컷 간선
    vector<pair<int, int>> cut;
    for (int u = 0; u < n; u++) {
        if (reachable[u]) {
            for (int v = 0; v < n; v++) {
                if (!reachable[v] && capacity[u][v] > 0) {
                    cut.push_back({u, v});
                }
            }
        }
    }

    return cut;
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "네트워크 플로우 예제" << endl;
    cout << "============================================================" << endl;

    // 1. Ford-Fulkerson
    cout << "\n[1] Ford-Fulkerson (DFS)" << endl;
    FordFulkerson ff(6);
    ff.addEdge(0, 1, 16);
    ff.addEdge(0, 2, 13);
    ff.addEdge(1, 2, 10);
    ff.addEdge(1, 3, 12);
    ff.addEdge(2, 1, 4);
    ff.addEdge(2, 4, 14);
    ff.addEdge(3, 2, 9);
    ff.addEdge(3, 5, 20);
    ff.addEdge(4, 3, 7);
    ff.addEdge(4, 5, 4);
    cout << "    최대 유량 (0 → 5): " << ff.maxFlow(0, 5) << endl;

    // 2. Edmonds-Karp
    cout << "\n[2] Edmonds-Karp (BFS)" << endl;
    EdmondsKarp ek(6);
    ek.addEdge(0, 1, 16);
    ek.addEdge(0, 2, 13);
    ek.addEdge(1, 2, 10);
    ek.addEdge(1, 3, 12);
    ek.addEdge(2, 1, 4);
    ek.addEdge(2, 4, 14);
    ek.addEdge(3, 2, 9);
    ek.addEdge(3, 5, 20);
    ek.addEdge(4, 3, 7);
    ek.addEdge(4, 5, 4);
    cout << "    최대 유량 (0 → 5): " << ek.maxFlow(0, 5) << endl;

    // 3. Dinic
    cout << "\n[3] Dinic's Algorithm" << endl;
    Dinic dinic(6);
    dinic.addEdge(0, 1, 16);
    dinic.addEdge(0, 2, 13);
    dinic.addEdge(1, 2, 10);
    dinic.addEdge(1, 3, 12);
    dinic.addEdge(2, 1, 4);
    dinic.addEdge(2, 4, 14);
    dinic.addEdge(3, 2, 9);
    dinic.addEdge(3, 5, 20);
    dinic.addEdge(4, 3, 7);
    dinic.addEdge(4, 5, 4);
    cout << "    최대 유량 (0 → 5): " << dinic.maxFlow(0, 5) << endl;

    // 4. 이분 매칭
    cout << "\n[4] 이분 매칭" << endl;
    BipartiteMatching bm(4, 4);
    bm.addEdge(0, 0);
    bm.addEdge(0, 1);
    bm.addEdge(1, 0);
    bm.addEdge(1, 2);
    bm.addEdge(2, 1);
    bm.addEdge(2, 2);
    bm.addEdge(3, 2);
    bm.addEdge(3, 3);
    cout << "    최대 매칭: " << bm.maxMatching() << endl;
    cout << "    매칭: ";
    for (auto [l, r] : bm.getMatching()) {
        cout << "(" << l << "," << r << ") ";
    }
    cout << endl;

    // 5. MCMF
    cout << "\n[5] 최소 비용 최대 유량" << endl;
    MCMF mcmf(4);
    mcmf.addEdge(0, 1, 2, 1);
    mcmf.addEdge(0, 2, 1, 2);
    mcmf.addEdge(1, 2, 1, 1);
    mcmf.addEdge(1, 3, 1, 3);
    mcmf.addEdge(2, 3, 2, 1);
    auto [flow, cost] = mcmf.minCostMaxFlow(0, 3);
    cout << "    최대 유량: " << flow << ", 최소 비용: " << cost << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 알고리즘       | 시간복잡도      | 특징              |" << endl;
    cout << "    |----------------|-----------------|-------------------|" << endl;
    cout << "    | Ford-Fulkerson | O(Ef)           | DFS, 무한루프 가능|" << endl;
    cout << "    | Edmonds-Karp   | O(VE²)          | BFS, 안정적       |" << endl;
    cout << "    | Dinic          | O(V²E)          | 레벨 그래프       |" << endl;
    cout << "    | 이분 매칭      | O(VE)           | 헝가리안          |" << endl;
    cout << "    | MCMF           | O(VEf) or O(V³E)| SPFA/Bellman-Ford |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
