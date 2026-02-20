/*
 * 최단 경로 (Shortest Path)
 * Dijkstra, Bellman-Ford, Floyd-Warshall, 0-1 BFS
 *
 * 그래프에서 정점 간 최단 거리를 구하는 알고리즘입니다.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>

using namespace std;

const int INF = INT_MAX;

// =============================================================================
// 1. Dijkstra 알고리즘
// =============================================================================

vector<int> dijkstra(int n, const vector<vector<pair<int, int>>>& adj, int start) {
    vector<int> dist(n, INF);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;

        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

// 경로 복원
pair<vector<int>, vector<int>> dijkstraWithPath(int n, const vector<vector<pair<int, int>>>& adj, int start) {
    vector<int> dist(n, INF);
    vector<int> parent(n, -1);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;

        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    return {dist, parent};
}

vector<int> reconstructPath(int start, int end, const vector<int>& parent) {
    vector<int> path;
    for (int at = end; at != -1; at = parent[at]) {
        path.push_back(at);
    }
    reverse(path.begin(), path.end());

    if (path[0] != start) {
        return {};  // 경로 없음
    }

    return path;
}

// =============================================================================
// 2. Bellman-Ford 알고리즘
// =============================================================================

struct Edge {
    int from, to, weight;
};

pair<vector<int>, bool> bellmanFord(int n, const vector<Edge>& edges, int start) {
    vector<int> dist(n, INF);
    dist[start] = 0;

    // n-1번 relaxation
    for (int i = 0; i < n - 1; i++) {
        bool updated = false;
        for (const auto& e : edges) {
            if (dist[e.from] != INF && dist[e.from] + e.weight < dist[e.to]) {
                dist[e.to] = dist[e.from] + e.weight;
                updated = true;
            }
        }
        if (!updated) break;  // 조기 종료
    }

    // 음수 사이클 검사
    for (const auto& e : edges) {
        if (dist[e.from] != INF && dist[e.from] + e.weight < dist[e.to]) {
            return {dist, true};  // 음수 사이클 존재
        }
    }

    return {dist, false};
}

// =============================================================================
// 3. Floyd-Warshall 알고리즘
// =============================================================================

vector<vector<int>> floydWarshall(int n, vector<vector<int>>& dist) {
    // dist는 초기 인접 행렬 (연결 없으면 INF)

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }

    return dist;
}

// 경로 복원 가능한 버전
pair<vector<vector<int>>, vector<vector<int>>> floydWarshallWithPath(int n, vector<vector<int>>& dist) {
    vector<vector<int>> next(n, vector<int>(n, -1));

    // 초기화: 직접 연결된 경우
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && dist[i][j] != INF) {
                next[i][j] = j;
            }
        }
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    next[i][j] = next[i][k];
                }
            }
        }
    }

    return {dist, next};
}

// =============================================================================
// 4. 0-1 BFS
// =============================================================================

vector<int> bfs01(int n, const vector<vector<pair<int, int>>>& adj, int start) {
    vector<int> dist(n, INF);
    deque<int> dq;

    dist[start] = 0;
    dq.push_front(start);

    while (!dq.empty()) {
        int u = dq.front();
        dq.pop_front();

        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (w == 0) {
                    dq.push_front(v);
                } else {
                    dq.push_back(v);
                }
            }
        }
    }

    return dist;
}

// =============================================================================
// 5. SPFA (Shortest Path Faster Algorithm)
// =============================================================================

pair<vector<int>, bool> spfa(int n, const vector<vector<pair<int, int>>>& adj, int start) {
    vector<int> dist(n, INF);
    vector<bool> inQueue(n, false);
    vector<int> cnt(n, 0);  // 큐에 들어간 횟수
    queue<int> q;

    dist[start] = 0;
    q.push(start);
    inQueue[start] = true;
    cnt[start] = 1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inQueue[u] = false;

        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!inQueue[v]) {
                    q.push(v);
                    inQueue[v] = true;
                    cnt[v]++;

                    if (cnt[v] >= n) {
                        return {dist, true};  // 음수 사이클
                    }
                }
            }
        }
    }

    return {dist, false};
}

// =============================================================================
// 6. K번째 최단 경로
// =============================================================================

vector<int> kthShortestPath(int n, const vector<vector<pair<int, int>>>& adj,
                            int start, int end, int k) {
    vector<int> kthDist;
    vector<int> count(n, 0);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    pq.push({0, start});

    while (!pq.empty() && count[end] < k) {
        auto [d, u] = pq.top();
        pq.pop();

        count[u]++;

        if (u == end) {
            kthDist.push_back(d);
        }

        if (count[u] <= k) {
            for (auto [v, w] : adj[u]) {
                pq.push({d + w, v});
            }
        }
    }

    return kthDist;
}

// =============================================================================
// 테스트
// =============================================================================

void printVector(const vector<int>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] == INF) cout << "∞";
        else cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "============================================================" << endl;
    cout << "최단 경로 예제" << endl;
    cout << "============================================================" << endl;

    // 테스트 그래프
    int n = 5;
    vector<vector<pair<int, int>>> adj(n);
    adj[0] = {{1, 4}, {2, 1}};
    adj[1] = {{3, 1}};
    adj[2] = {{1, 2}, {3, 5}};
    adj[3] = {{4, 3}};

    // 1. Dijkstra
    cout << "\n[1] Dijkstra 알고리즘" << endl;
    auto dijkDist = dijkstra(n, adj, 0);
    cout << "    0에서 각 정점까지 거리: ";
    printVector(dijkDist);
    cout << endl;

    // 경로 복원
    auto [dist, parent] = dijkstraWithPath(n, adj, 0);
    auto path = reconstructPath(0, 4, parent);
    cout << "    0 → 4 경로: ";
    printVector(path);
    cout << " (거리: " << dist[4] << ")" << endl;

    // 2. Bellman-Ford
    cout << "\n[2] Bellman-Ford 알고리즘" << endl;
    vector<Edge> edges = {
        {0, 1, 4}, {0, 2, 1}, {1, 3, 1}, {2, 1, 2}, {2, 3, 5}, {3, 4, 3}
    };
    auto [bfDist, hasNegCycle] = bellmanFord(n, edges, 0);
    cout << "    0에서 각 정점까지 거리: ";
    printVector(bfDist);
    cout << endl;
    cout << "    음수 사이클: " << (hasNegCycle ? "있음" : "없음") << endl;

    // 3. Floyd-Warshall
    cout << "\n[3] Floyd-Warshall 알고리즘" << endl;
    vector<vector<int>> distMatrix(4, vector<int>(4, INF));
    for (int i = 0; i < 4; i++) distMatrix[i][i] = 0;
    distMatrix[0][1] = 3;
    distMatrix[0][2] = 8;
    distMatrix[1][2] = 2;
    distMatrix[2][3] = 1;
    distMatrix[0][3] = 7;

    floydWarshall(4, distMatrix);
    cout << "    모든 쌍 최단 거리:" << endl;
    for (int i = 0; i < 4; i++) {
        cout << "      " << i << ": ";
        printVector(distMatrix[i]);
        cout << endl;
    }

    // 4. 0-1 BFS
    cout << "\n[4] 0-1 BFS" << endl;
    vector<vector<pair<int, int>>> adj01(4);
    adj01[0] = {{1, 0}, {2, 1}};
    adj01[1] = {{2, 0}, {3, 1}};
    adj01[2] = {{3, 0}};
    auto dist01 = bfs01(4, adj01, 0);
    cout << "    0에서 각 정점까지 거리 (0-1): ";
    printVector(dist01);
    cout << endl;

    // 5. K번째 최단 경로
    cout << "\n[5] K번째 최단 경로" << endl;
    auto kthDists = kthShortestPath(n, adj, 0, 4, 3);
    cout << "    0 → 4 의 1~3번째 최단 거리: ";
    printVector(kthDists);
    cout << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 알고리즘         | 시간복잡도       | 음수 가중치 |" << endl;
    cout << "    |------------------|------------------|-------------|" << endl;
    cout << "    | Dijkstra         | O((V+E) log V)   | X           |" << endl;
    cout << "    | Bellman-Ford     | O(VE)            | O (사이클X) |" << endl;
    cout << "    | Floyd-Warshall   | O(V³)            | O (사이클X) |" << endl;
    cout << "    | 0-1 BFS          | O(V + E)         | 0,1만       |" << endl;
    cout << "    | SPFA             | O(VE) 평균 O(E)  | O (사이클X) |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
