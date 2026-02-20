/*
 * 위상 정렬 (Topological Sort)
 * Kahn's Algorithm, DFS-based, Cycle Detection
 *
 * DAG(Directed Acyclic Graph)에서 정점들의 선형 순서를 찾습니다.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>

using namespace std;

// =============================================================================
// 1. Kahn's Algorithm (BFS 기반)
// =============================================================================

vector<int> topologicalSortKahn(int n, const vector<vector<int>>& adj) {
    vector<int> indegree(n, 0);

    // 진입 차수 계산
    for (int u = 0; u < n; u++) {
        for (int v : adj[u]) {
            indegree[v]++;
        }
    }

    // 진입 차수가 0인 정점을 큐에 추가
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }

    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : adj[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    // 사이클이 있으면 모든 정점을 방문할 수 없음
    if ((int)result.size() != n) {
        return {};  // 사이클 존재
    }

    return result;
}

// =============================================================================
// 2. DFS 기반 위상 정렬
// =============================================================================

void topoDFS(int node, const vector<vector<int>>& adj,
             vector<bool>& visited, stack<int>& st) {
    visited[node] = true;

    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            topoDFS(neighbor, adj, visited, st);
        }
    }

    st.push(node);
}

vector<int> topologicalSortDFS(int n, const vector<vector<int>>& adj) {
    vector<bool> visited(n, false);
    stack<int> st;

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            topoDFS(i, adj, visited, st);
        }
    }

    vector<int> result;
    while (!st.empty()) {
        result.push_back(st.top());
        st.pop();
    }

    return result;
}

// =============================================================================
// 3. 사이클 탐지 (DFS)
// =============================================================================

bool hasCycleDFS(int node, const vector<vector<int>>& adj,
                 vector<int>& color) {
    color[node] = 1;  // 방문 중 (회색)

    for (int neighbor : adj[node]) {
        if (color[neighbor] == 1) {
            return true;  // 백 엣지 발견 (사이클)
        }
        if (color[neighbor] == 0 && hasCycleDFS(neighbor, adj, color)) {
            return true;
        }
    }

    color[node] = 2;  // 방문 완료 (검은색)
    return false;
}

bool hasCycle(int n, const vector<vector<int>>& adj) {
    vector<int> color(n, 0);  // 0: 미방문, 1: 방문 중, 2: 완료

    for (int i = 0; i < n; i++) {
        if (color[i] == 0 && hasCycleDFS(i, adj, color)) {
            return true;
        }
    }

    return false;
}

// =============================================================================
// 4. 모든 위상 정렬 순서 찾기
// =============================================================================

void allTopoSorts(vector<vector<int>>& adj, vector<int>& indegree,
                  vector<int>& result, vector<bool>& visited,
                  vector<vector<int>>& allResults, int n) {
    bool found = false;

    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0 && !visited[i]) {
            // 이 정점 선택
            visited[i] = true;
            result.push_back(i);

            for (int neighbor : adj[i]) {
                indegree[neighbor]--;
            }

            allTopoSorts(adj, indegree, result, visited, allResults, n);

            // 백트래킹
            visited[i] = false;
            result.pop_back();
            for (int neighbor : adj[i]) {
                indegree[neighbor]++;
            }

            found = true;
        }
    }

    if (!found && (int)result.size() == n) {
        allResults.push_back(result);
    }
}

vector<vector<int>> findAllTopologicalSorts(int n, const vector<vector<int>>& adj) {
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; u++) {
        for (int v : adj[u]) {
            indegree[v]++;
        }
    }

    vector<int> result;
    vector<bool> visited(n, false);
    vector<vector<int>> allResults;
    vector<vector<int>> adjCopy = adj;

    allTopoSorts(adjCopy, indegree, result, visited, allResults, n);

    return allResults;
}

// =============================================================================
// 5. 과목 수강 순서 (Course Schedule)
// =============================================================================

bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
    vector<vector<int>> adj(numCourses);
    vector<int> indegree(numCourses, 0);

    for (auto& [course, prereq] : prerequisites) {
        adj[prereq].push_back(course);
        indegree[course]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }

    int count = 0;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        count++;

        for (int neighbor : adj[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    return count == numCourses;
}

vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
    vector<vector<int>> adj(numCourses);
    vector<int> indegree(numCourses, 0);

    for (auto& [course, prereq] : prerequisites) {
        adj[prereq].push_back(course);
        indegree[course]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }

    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : adj[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    if ((int)result.size() != numCourses) {
        return {};
    }

    return result;
}

// =============================================================================
// 6. 외계인 사전 (Alien Dictionary)
// =============================================================================

string alienOrder(vector<string>& words) {
    // 그래프 구성
    unordered_map<char, unordered_set<char>> adj;
    unordered_map<char, int> indegree;

    // 모든 문자 초기화
    for (const string& word : words) {
        for (char c : word) {
            if (indegree.find(c) == indegree.end()) {
                indegree[c] = 0;
            }
        }
    }

    // 인접한 단어 비교하여 순서 결정
    for (int i = 0; i < (int)words.size() - 1; i++) {
        string& w1 = words[i];
        string& w2 = words[i + 1];

        // 잘못된 순서 체크 (prefix가 더 뒤에 올 수 없음)
        if (w1.length() > w2.length() &&
            w1.substr(0, w2.length()) == w2) {
            return "";
        }

        for (int j = 0; j < (int)min(w1.length(), w2.length()); j++) {
            if (w1[j] != w2[j]) {
                if (adj[w1[j]].find(w2[j]) == adj[w1[j]].end()) {
                    adj[w1[j]].insert(w2[j]);
                    indegree[w2[j]]++;
                }
                break;
            }
        }
    }

    // 위상 정렬
    queue<char> q;
    for (auto& [c, deg] : indegree) {
        if (deg == 0) {
            q.push(c);
        }
    }

    string result;
    while (!q.empty()) {
        char c = q.front();
        q.pop();
        result += c;

        for (char next : adj[c]) {
            indegree[next]--;
            if (indegree[next] == 0) {
                q.push(next);
            }
        }
    }

    if (result.length() != indegree.size()) {
        return "";  // 사이클 존재
    }

    return result;
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
    cout << "위상 정렬 예제" << endl;
    cout << "============================================================" << endl;

    // 테스트 그래프
    //   5 → 0 ← 4
    //   ↓       ↓
    //   2 → 3 → 1
    int n = 6;
    vector<vector<int>> adj(n);
    adj[5] = {0, 2};
    adj[4] = {0, 1};
    adj[2] = {3};
    adj[3] = {1};

    // 1. Kahn's Algorithm
    cout << "\n[1] Kahn's Algorithm (BFS)" << endl;
    auto result1 = topologicalSortKahn(n, adj);
    cout << "    위상 정렬: ";
    printVector(result1);
    cout << endl;

    // 2. DFS 기반
    cout << "\n[2] DFS 기반 위상 정렬" << endl;
    auto result2 = topologicalSortDFS(n, adj);
    cout << "    위상 정렬: ";
    printVector(result2);
    cout << endl;

    // 3. 사이클 탐지
    cout << "\n[3] 사이클 탐지" << endl;
    cout << "    현재 그래프: " << (hasCycle(n, adj) ? "사이클 있음" : "DAG") << endl;

    vector<vector<int>> cycleAdj(3);
    cycleAdj[0] = {1};
    cycleAdj[1] = {2};
    cycleAdj[2] = {0};
    cout << "    사이클 그래프: " << (hasCycle(3, cycleAdj) ? "사이클 있음" : "DAG") << endl;

    // 4. 모든 위상 정렬
    cout << "\n[4] 모든 위상 정렬 순서" << endl;
    vector<vector<int>> smallAdj(4);
    smallAdj[0] = {1, 2};
    smallAdj[1] = {3};
    smallAdj[2] = {3};
    auto allSorts = findAllTopologicalSorts(4, smallAdj);
    cout << "    그래프: 0→1→3, 0→2→3" << endl;
    cout << "    가능한 순서: " << allSorts.size() << "개" << endl;
    for (auto& order : allSorts) {
        cout << "      ";
        printVector(order);
        cout << endl;
    }

    // 5. 과목 수강 순서
    cout << "\n[5] 과목 수강 순서" << endl;
    vector<pair<int, int>> prereqs = {{1, 0}, {2, 0}, {3, 1}, {3, 2}};
    cout << "    과목 수: 4, 선수과목: (1,0), (2,0), (3,1), (3,2)" << endl;
    cout << "    수강 가능: " << (canFinish(4, prereqs) ? "예" : "아니오") << endl;
    auto order = findOrder(4, prereqs);
    cout << "    수강 순서: ";
    printVector(order);
    cout << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 알고리즘       | 시간복잡도 | 공간복잡도 |" << endl;
    cout << "    |----------------|------------|------------|" << endl;
    cout << "    | Kahn (BFS)     | O(V + E)   | O(V)       |" << endl;
    cout << "    | DFS 기반       | O(V + E)   | O(V)       |" << endl;
    cout << "    | 모든 순서      | O(V! × V)  | O(V)       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
