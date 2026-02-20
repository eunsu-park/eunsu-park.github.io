/*
 * 강한 연결 요소 (Strongly Connected Components)
 * Tarjan, Kosaraju, 2-SAT
 *
 * 방향 그래프에서 서로 도달 가능한 정점들의 집합을 찾습니다.
 */

#include <iostream>
#include <vector>
#include <stack>
#include <algorithm>

using namespace std;

// =============================================================================
// 1. Tarjan's Algorithm
// =============================================================================

class TarjanSCC {
private:
    int n;
    vector<vector<int>> adj;
    vector<int> ids;
    vector<int> low;
    vector<bool> onStack;
    stack<int> st;
    int counter;
    int sccCount;
    vector<vector<int>> sccs;

    void dfs(int v) {
        ids[v] = low[v] = counter++;
        st.push(v);
        onStack[v] = true;

        for (int u : adj[v]) {
            if (ids[u] == -1) {
                dfs(u);
                low[v] = min(low[v], low[u]);
            } else if (onStack[u]) {
                low[v] = min(low[v], ids[u]);
            }
        }

        // SCC의 루트인 경우
        if (ids[v] == low[v]) {
            vector<int> scc;
            while (true) {
                int u = st.top();
                st.pop();
                onStack[u] = false;
                scc.push_back(u);
                if (u == v) break;
            }
            sccs.push_back(scc);
            sccCount++;
        }
    }

public:
    TarjanSCC(int n, const vector<vector<int>>& adj) : n(n), adj(adj) {
        ids.assign(n, -1);
        low.assign(n, 0);
        onStack.assign(n, false);
        counter = 0;
        sccCount = 0;

        for (int i = 0; i < n; i++) {
            if (ids[i] == -1) {
                dfs(i);
            }
        }
    }

    int getSCCCount() const { return sccCount; }
    const vector<vector<int>>& getSCCs() const { return sccs; }

    // SCC 축약 그래프 (DAG)
    vector<vector<int>> getCondensedGraph() {
        vector<int> sccId(n);
        for (int i = 0; i < (int)sccs.size(); i++) {
            for (int v : sccs[i]) {
                sccId[v] = i;
            }
        }

        set<pair<int, int>> edges;
        for (int v = 0; v < n; v++) {
            for (int u : adj[v]) {
                if (sccId[v] != sccId[u]) {
                    edges.insert({sccId[v], sccId[u]});
                }
            }
        }

        vector<vector<int>> dag(sccCount);
        for (auto [u, v] : edges) {
            dag[u].push_back(v);
        }

        return dag;
    }
};

// =============================================================================
// 2. Kosaraju's Algorithm
// =============================================================================

class KosarajuSCC {
private:
    int n;
    vector<vector<int>> adj;
    vector<vector<int>> radj;  // 역방향 그래프
    vector<bool> visited;
    vector<int> order;         // 첫 번째 DFS 순서
    vector<vector<int>> sccs;

    void dfs1(int v) {
        visited[v] = true;
        for (int u : adj[v]) {
            if (!visited[u]) {
                dfs1(u);
            }
        }
        order.push_back(v);
    }

    void dfs2(int v, vector<int>& scc) {
        visited[v] = true;
        scc.push_back(v);
        for (int u : radj[v]) {
            if (!visited[u]) {
                dfs2(u, scc);
            }
        }
    }

public:
    KosarajuSCC(int n, const vector<vector<int>>& adj) : n(n), adj(adj), radj(n) {
        // 역방향 그래프 생성
        for (int v = 0; v < n; v++) {
            for (int u : adj[v]) {
                radj[u].push_back(v);
            }
        }

        // 첫 번째 DFS: 종료 순서
        visited.assign(n, false);
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfs1(i);
            }
        }

        // 두 번째 DFS: SCC 찾기
        visited.assign(n, false);
        for (int i = n - 1; i >= 0; i--) {
            int v = order[i];
            if (!visited[v]) {
                vector<int> scc;
                dfs2(v, scc);
                sccs.push_back(scc);
            }
        }
    }

    int getSCCCount() const { return sccs.size(); }
    const vector<vector<int>>& getSCCs() const { return sccs; }
};

// =============================================================================
// 3. 2-SAT
// =============================================================================

class TwoSAT {
private:
    int n;
    vector<vector<int>> adj;
    vector<int> sccId;
    vector<bool> assignment;

    int var(int x) { return x >= 0 ? 2 * x : 2 * (-x - 1) + 1; }
    int notVar(int x) { return x >= 0 ? 2 * x + 1 : 2 * (-x - 1); }

public:
    TwoSAT(int n) : n(n), adj(2 * n), sccId(2 * n), assignment(n) {}

    // x ∨ y 추가 (둘 중 하나는 참)
    void addClause(int x, int y) {
        // ¬x → y, ¬y → x
        adj[notVar(x)].push_back(var(y));
        adj[notVar(y)].push_back(var(x));
    }

    // x를 참으로 강제
    void setTrue(int x) {
        adj[notVar(x)].push_back(var(x));
    }

    // x를 거짓으로 강제
    void setFalse(int x) {
        adj[var(x)].push_back(notVar(x));
    }

    // x ⊕ y (정확히 하나만 참)
    void addXor(int x, int y) {
        addClause(x, y);
        addClause(-x - 1, -y - 1);
    }

    bool solve() {
        TarjanSCC scc(2 * n, adj);
        auto sccs = scc.getSCCs();

        // SCC ID 할당
        for (int i = 0; i < (int)sccs.size(); i++) {
            for (int v : sccs[i]) {
                sccId[v] = i;
            }
        }

        // 충족 가능성 검사
        for (int i = 0; i < n; i++) {
            if (sccId[2 * i] == sccId[2 * i + 1]) {
                return false;  // x와 ¬x가 같은 SCC
            }
            // SCC 순서가 역방향이므로 비교
            assignment[i] = sccId[2 * i] > sccId[2 * i + 1];
        }

        return true;
    }

    vector<bool> getAssignment() const { return assignment; }
};

// =============================================================================
// 4. 절단점 (Articulation Points)
// =============================================================================

class ArticulationPoints {
private:
    int n;
    vector<vector<int>> adj;
    vector<int> ids, low;
    vector<bool> isAP;
    int counter;

    void dfs(int v, int parent) {
        ids[v] = low[v] = counter++;
        int children = 0;

        for (int u : adj[v]) {
            if (ids[u] == -1) {
                children++;
                dfs(u, v);
                low[v] = min(low[v], low[u]);

                if (parent != -1 && low[u] >= ids[v]) {
                    isAP[v] = true;
                }
            } else if (u != parent) {
                low[v] = min(low[v], ids[u]);
            }
        }

        // 루트가 2개 이상의 자식을 가지면 절단점
        if (parent == -1 && children > 1) {
            isAP[v] = true;
        }
    }

public:
    ArticulationPoints(int n, const vector<vector<int>>& adj) : n(n), adj(adj) {
        ids.assign(n, -1);
        low.assign(n, 0);
        isAP.assign(n, false);
        counter = 0;

        for (int i = 0; i < n; i++) {
            if (ids[i] == -1) {
                dfs(i, -1);
            }
        }
    }

    vector<int> getArticulationPoints() {
        vector<int> result;
        for (int i = 0; i < n; i++) {
            if (isAP[i]) result.push_back(i);
        }
        return result;
    }
};

// =============================================================================
// 5. 브릿지 (Bridges)
// =============================================================================

class Bridges {
private:
    int n;
    vector<vector<int>> adj;
    vector<int> ids, low;
    vector<pair<int, int>> bridges;
    int counter;

    void dfs(int v, int parent) {
        ids[v] = low[v] = counter++;

        for (int u : adj[v]) {
            if (ids[u] == -1) {
                dfs(u, v);
                low[v] = min(low[v], low[u]);

                if (low[u] > ids[v]) {
                    bridges.push_back({v, u});
                }
            } else if (u != parent) {
                low[v] = min(low[v], ids[u]);
            }
        }
    }

public:
    Bridges(int n, const vector<vector<int>>& adj) : n(n), adj(adj) {
        ids.assign(n, -1);
        low.assign(n, 0);
        counter = 0;

        for (int i = 0; i < n; i++) {
            if (ids[i] == -1) {
                dfs(i, -1);
            }
        }
    }

    vector<pair<int, int>> getBridges() { return bridges; }
};

// =============================================================================
// 테스트
// =============================================================================

#include <set>

int main() {
    cout << "============================================================" << endl;
    cout << "강한 연결 요소 예제" << endl;
    cout << "============================================================" << endl;

    // 테스트 그래프
    //  0 → 1 → 2 → 3
    //  ↑   ↓       ↓
    //  4 ← 5   6 ← 7
    //      ↓
    //      6

    int n = 8;
    vector<vector<int>> adj(n);
    adj[0] = {1};
    adj[1] = {2, 5};
    adj[2] = {3};
    adj[3] = {7};
    adj[4] = {0};
    adj[5] = {4, 6};
    adj[6] = {};
    adj[7] = {6};

    // 1. Tarjan's Algorithm
    cout << "\n[1] Tarjan's Algorithm" << endl;
    TarjanSCC tarjan(n, adj);
    cout << "    SCC 개수: " << tarjan.getSCCCount() << endl;
    cout << "    SCCs:" << endl;
    for (const auto& scc : tarjan.getSCCs()) {
        cout << "      { ";
        for (int v : scc) cout << v << " ";
        cout << "}" << endl;
    }

    // 2. Kosaraju's Algorithm
    cout << "\n[2] Kosaraju's Algorithm" << endl;
    KosarajuSCC kosaraju(n, adj);
    cout << "    SCC 개수: " << kosaraju.getSCCCount() << endl;

    // 3. 2-SAT
    cout << "\n[3] 2-SAT" << endl;
    // (x0 ∨ x1) ∧ (¬x0 ∨ x2) ∧ (¬x1 ∨ ¬x2)
    TwoSAT sat(3);
    sat.addClause(0, 1);         // x0 ∨ x1
    sat.addClause(-1, 2);        // ¬x0 ∨ x2
    sat.addClause(-2, -3);       // ¬x1 ∨ ¬x2

    if (sat.solve()) {
        cout << "    충족 가능" << endl;
        auto assignment = sat.getAssignment();
        cout << "    할당: ";
        for (int i = 0; i < 3; i++) {
            cout << "x" << i << "=" << assignment[i] << " ";
        }
        cout << endl;
    } else {
        cout << "    충족 불가능" << endl;
    }

    // 4. 절단점
    cout << "\n[4] 절단점 (무방향 그래프)" << endl;
    vector<vector<int>> undirected(5);
    undirected[0] = {1};
    undirected[1] = {0, 2, 3};
    undirected[2] = {1, 3};
    undirected[3] = {1, 2, 4};
    undirected[4] = {3};

    ArticulationPoints ap(5, undirected);
    auto points = ap.getArticulationPoints();
    cout << "    절단점: ";
    for (int v : points) cout << v << " ";
    cout << endl;

    // 5. 브릿지
    cout << "\n[5] 브릿지 (무방향 그래프)" << endl;
    Bridges br(5, undirected);
    auto bridgeList = br.getBridges();
    cout << "    브릿지: ";
    for (auto [u, v] : bridgeList) {
        cout << "(" << u << "-" << v << ") ";
    }
    cout << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 알고리즘    | 시간복잡도 | 용도               |" << endl;
    cout << "    |-------------|------------|---------------------|" << endl;
    cout << "    | Tarjan      | O(V + E)   | SCC, 단일 DFS       |" << endl;
    cout << "    | Kosaraju    | O(V + E)   | SCC, 2번 DFS        |" << endl;
    cout << "    | 2-SAT       | O(V + E)   | 논리식 충족 가능성  |" << endl;
    cout << "    | 절단점      | O(V + E)   | 그래프 취약점       |" << endl;
    cout << "    | 브릿지      | O(V + E)   | 그래프 취약점       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
