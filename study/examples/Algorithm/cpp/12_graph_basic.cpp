/*
 * 그래프 기초 (Graph Basics)
 * Graph Representation, DFS, BFS, Connected Components
 *
 * 그래프의 기본 표현과 탐색 알고리즘입니다.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <unordered_set>
#include <algorithm>

using namespace std;

// =============================================================================
// 1. 그래프 표현
// =============================================================================

// 인접 리스트
class AdjacencyList {
public:
    int V;
    vector<vector<int>> adj;
    bool directed;

    AdjacencyList(int v, bool dir = false) : V(v), adj(v), directed(dir) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        if (!directed) {
            adj[v].push_back(u);
        }
    }

    void print() {
        for (int i = 0; i < V; i++) {
            cout << i << ": ";
            for (int neighbor : adj[i]) {
                cout << neighbor << " ";
            }
            cout << endl;
        }
    }
};

// 인접 행렬
class AdjacencyMatrix {
public:
    int V;
    vector<vector<int>> matrix;

    AdjacencyMatrix(int v) : V(v), matrix(v, vector<int>(v, 0)) {}

    void addEdge(int u, int v, int weight = 1) {
        matrix[u][v] = weight;
        matrix[v][u] = weight;  // 무방향
    }

    void print() {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                cout << matrix[i][j] << " ";
            }
            cout << endl;
        }
    }
};

// =============================================================================
// 2. DFS (깊이 우선 탐색)
// =============================================================================

void dfsRecursive(const vector<vector<int>>& adj, int node, vector<bool>& visited, vector<int>& result) {
    visited[node] = true;
    result.push_back(node);

    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            dfsRecursive(adj, neighbor, visited, result);
        }
    }
}

vector<int> dfs(const vector<vector<int>>& adj, int start) {
    int n = adj.size();
    vector<bool> visited(n, false);
    vector<int> result;
    dfsRecursive(adj, start, visited, result);
    return result;
}

// 반복적 DFS
vector<int> dfsIterative(const vector<vector<int>>& adj, int start) {
    int n = adj.size();
    vector<bool> visited(n, false);
    vector<int> result;
    stack<int> st;

    st.push(start);

    while (!st.empty()) {
        int node = st.top();
        st.pop();

        if (visited[node]) continue;
        visited[node] = true;
        result.push_back(node);

        for (auto it = adj[node].rbegin(); it != adj[node].rend(); ++it) {
            if (!visited[*it]) {
                st.push(*it);
            }
        }
    }

    return result;
}

// =============================================================================
// 3. BFS (너비 우선 탐색)
// =============================================================================

vector<int> bfs(const vector<vector<int>>& adj, int start) {
    int n = adj.size();
    vector<bool> visited(n, false);
    vector<int> result;
    queue<int> q;

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }

    return result;
}

// BFS 최단 거리
vector<int> bfsDistance(const vector<vector<int>>& adj, int start) {
    int n = adj.size();
    vector<int> dist(n, -1);
    queue<int> q;

    q.push(start);
    dist[start] = 0;

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        for (int neighbor : adj[node]) {
            if (dist[neighbor] == -1) {
                dist[neighbor] = dist[node] + 1;
                q.push(neighbor);
            }
        }
    }

    return dist;
}

// =============================================================================
// 4. 연결 요소
// =============================================================================

int countConnectedComponents(int n, const vector<vector<int>>& adj) {
    vector<bool> visited(n, false);
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            count++;
            queue<int> q;
            q.push(i);
            visited[i] = true;

            while (!q.empty()) {
                int node = q.front();
                q.pop();
                for (int neighbor : adj[node]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }

    return count;
}

// =============================================================================
// 5. 사이클 탐지
// =============================================================================

// 무방향 그래프 사이클 탐지 (DFS)
bool hasCycleUndirected(const vector<vector<int>>& adj, int node, int parent, vector<bool>& visited) {
    visited[node] = true;

    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            if (hasCycleUndirected(adj, neighbor, node, visited)) {
                return true;
            }
        } else if (neighbor != parent) {
            return true;
        }
    }

    return false;
}

bool detectCycleUndirected(int n, const vector<vector<int>>& adj) {
    vector<bool> visited(n, false);

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            if (hasCycleUndirected(adj, i, -1, visited)) {
                return true;
            }
        }
    }

    return false;
}

// 방향 그래프 사이클 탐지 (DFS)
bool hasCycleDirected(const vector<vector<int>>& adj, int node, vector<int>& color) {
    color[node] = 1;  // 방문 중 (회색)

    for (int neighbor : adj[node]) {
        if (color[neighbor] == 1) {  // 현재 DFS 경로에서 발견
            return true;
        }
        if (color[neighbor] == 0 && hasCycleDirected(adj, neighbor, color)) {
            return true;
        }
    }

    color[node] = 2;  // 완료 (검은색)
    return false;
}

bool detectCycleDirected(int n, const vector<vector<int>>& adj) {
    vector<int> color(n, 0);  // 0: 미방문, 1: 방문 중, 2: 완료

    for (int i = 0; i < n; i++) {
        if (color[i] == 0 && hasCycleDirected(adj, i, color)) {
            return true;
        }
    }

    return false;
}

// =============================================================================
// 6. 이분 그래프 검사
// =============================================================================

bool isBipartite(const vector<vector<int>>& adj) {
    int n = adj.size();
    vector<int> color(n, -1);

    for (int start = 0; start < n; start++) {
        if (color[start] != -1) continue;

        queue<int> q;
        q.push(start);
        color[start] = 0;

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            for (int neighbor : adj[node]) {
                if (color[neighbor] == -1) {
                    color[neighbor] = 1 - color[node];
                    q.push(neighbor);
                } else if (color[neighbor] == color[node]) {
                    return false;
                }
            }
        }
    }

    return true;
}

// =============================================================================
// 테스트
// =============================================================================

void printVector(const vector<int>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "============================================================" << endl;
    cout << "그래프 기초 예제" << endl;
    cout << "============================================================" << endl;

    // 테스트 그래프 생성
    //   0 --- 1
    //   |     |
    //   2 --- 3
    //         |
    //         4
    int n = 5;
    vector<vector<int>> adj(n);
    adj[0] = {1, 2};
    adj[1] = {0, 3};
    adj[2] = {0, 3};
    adj[3] = {1, 2, 4};
    adj[4] = {3};

    // 1. DFS
    cout << "\n[1] DFS (깊이 우선 탐색)" << endl;
    auto dfsResult = dfs(adj, 0);
    cout << "    0에서 시작: ";
    printVector(dfsResult);
    cout << endl;

    // 2. BFS
    cout << "\n[2] BFS (너비 우선 탐색)" << endl;
    auto bfsResult = bfs(adj, 0);
    cout << "    0에서 시작: ";
    printVector(bfsResult);
    cout << endl;

    auto distances = bfsDistance(adj, 0);
    cout << "    0에서 각 노드까지 거리: ";
    printVector(distances);
    cout << endl;

    // 3. 연결 요소
    cout << "\n[3] 연결 요소" << endl;
    vector<vector<int>> disconnected(6);
    disconnected[0] = {1};
    disconnected[1] = {0};
    disconnected[2] = {3};
    disconnected[3] = {2};
    disconnected[4] = {5};
    disconnected[5] = {4};
    cout << "    연결 요소 개수: " << countConnectedComponents(6, disconnected) << endl;

    // 4. 사이클 탐지
    cout << "\n[4] 사이클 탐지" << endl;
    cout << "    무방향 그래프 사이클: " << (detectCycleUndirected(n, adj) ? "있음" : "없음") << endl;

    vector<vector<int>> directedAdj(4);
    directedAdj[0] = {1};
    directedAdj[1] = {2};
    directedAdj[2] = {3};
    directedAdj[3] = {1};  // 사이클
    cout << "    방향 그래프 사이클: " << (detectCycleDirected(4, directedAdj) ? "있음" : "없음") << endl;

    // 5. 이분 그래프
    cout << "\n[5] 이분 그래프" << endl;
    cout << "    현재 그래프: " << (isBipartite(adj) ? "이분 그래프" : "이분 그래프 아님") << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 알고리즘        | 시간복잡도 | 공간복잡도 |" << endl;
    cout << "    |-----------------|------------|------------|" << endl;
    cout << "    | DFS             | O(V + E)   | O(V)       |" << endl;
    cout << "    | BFS             | O(V + E)   | O(V)       |" << endl;
    cout << "    | 연결 요소       | O(V + E)   | O(V)       |" << endl;
    cout << "    | 사이클 탐지     | O(V + E)   | O(V)       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
