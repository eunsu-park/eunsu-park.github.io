/*
 * 고급 DP 최적화 (Advanced DP Optimization)
 * CHT, D&C Optimization, Knuth Optimization, Monotonic Queue
 *
 * DP의 시간 복잡도를 줄이는 고급 기법들입니다.
 */

#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <climits>
#include <cmath>

using namespace std;

typedef long long ll;
const ll INF = LLONG_MAX / 2;

// =============================================================================
// 1. Convex Hull Trick (CHT)
// =============================================================================

class ConvexHullTrick {
private:
    struct Line {
        ll m, b;  // y = mx + b
        ll eval(ll x) const { return m * x + b; }
    };

    deque<Line> hull;

    bool bad(const Line& l1, const Line& l2, const Line& l3) {
        // l2가 불필요한지 확인
        return (l3.b - l1.b) * (l1.m - l2.m) <= (l2.b - l1.b) * (l1.m - l3.m);
    }

public:
    // 기울기 단조 감소일 때 추가
    void addLine(ll m, ll b) {
        Line line = {m, b};
        while (hull.size() >= 2 && bad(hull[hull.size()-2], hull[hull.size()-1], line)) {
            hull.pop_back();
        }
        hull.push_back(line);
    }

    // x가 단조 증가할 때 최솟값 쿼리
    ll query(ll x) {
        while (hull.size() >= 2 && hull[0].eval(x) >= hull[1].eval(x)) {
            hull.pop_front();
        }
        return hull[0].eval(x);
    }

    // 이분 탐색으로 쿼리 (x가 단조 증가가 아닐 때)
    ll queryBinarySearch(ll x) {
        int lo = 0, hi = hull.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (hull[mid].eval(x) > hull[mid + 1].eval(x)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return hull[lo].eval(x);
    }
};

// =============================================================================
// 2. Li Chao Tree
// =============================================================================

class LiChaoTree {
private:
    struct Line {
        ll m, b;
        ll eval(ll x) const { return m * x + b; }
    };

    struct Node {
        Line line;
        Node *left, *right;
        Node() : line({0, INF}), left(nullptr), right(nullptr) {}
    };

    Node* root;
    ll lo, hi;

    void update(Node*& node, ll l, ll r, Line newLine) {
        if (!node) node = new Node();

        ll mid = (l + r) / 2;
        bool leftBetter = newLine.eval(l) < node->line.eval(l);
        bool midBetter = newLine.eval(mid) < node->line.eval(mid);

        if (midBetter) swap(node->line, newLine);

        if (l == r) return;

        if (leftBetter != midBetter) {
            update(node->left, l, mid, newLine);
        } else {
            update(node->right, mid + 1, r, newLine);
        }
    }

    ll query(Node* node, ll l, ll r, ll x) {
        if (!node) return INF;

        ll mid = (l + r) / 2;
        ll result = node->line.eval(x);

        if (x <= mid) {
            result = min(result, query(node->left, l, mid, x));
        } else {
            result = min(result, query(node->right, mid + 1, r, x));
        }

        return result;
    }

public:
    LiChaoTree(ll lo, ll hi) : root(nullptr), lo(lo), hi(hi) {}

    void addLine(ll m, ll b) {
        update(root, lo, hi, {m, b});
    }

    ll query(ll x) {
        return query(root, lo, hi, x);
    }
};

// =============================================================================
// 3. D&C Optimization
// =============================================================================

// dp[i][j] = min(dp[i-1][k] + cost[k][j]) for k < j
// 조건: opt[i][j] <= opt[i][j+1]

class DivideConquerDP {
private:
    int n, k;
    vector<vector<ll>> dp;
    vector<vector<ll>> cost;

    void compute(int level, int lo, int hi, int optLo, int optHi) {
        if (lo > hi) return;

        int mid = (lo + hi) / 2;
        int opt = optLo;
        dp[level][mid] = INF;

        for (int i = optLo; i <= min(mid - 1, optHi); i++) {
            ll val = dp[level - 1][i] + cost[i + 1][mid];
            if (val < dp[level][mid]) {
                dp[level][mid] = val;
                opt = i;
            }
        }

        compute(level, lo, mid - 1, optLo, opt);
        compute(level, mid + 1, hi, opt, optHi);
    }

public:
    ll solve(int n, int k, const vector<vector<ll>>& cost) {
        this->n = n;
        this->k = k;
        this->cost = cost;

        dp.assign(k + 1, vector<ll>(n + 1, INF));
        dp[0][0] = 0;

        // 첫 번째 그룹
        for (int j = 1; j <= n; j++) {
            dp[1][j] = cost[1][j];
        }

        for (int i = 2; i <= k; i++) {
            compute(i, 1, n, 0, n - 1);
        }

        return dp[k][n];
    }
};

// =============================================================================
// 4. Knuth Optimization
// =============================================================================

// dp[i][j] = min(dp[i][k] + dp[k+1][j]) + cost[i][j] for i <= k < j
// 조건: opt[i][j-1] <= opt[i][j] <= opt[i+1][j]

ll knuthOptimization(int n, const vector<vector<ll>>& cost) {
    vector<vector<ll>> dp(n + 2, vector<ll>(n + 2, 0));
    vector<vector<int>> opt(n + 2, vector<int>(n + 2));

    // 기저 조건
    for (int i = 1; i <= n; i++) {
        opt[i][i] = i;
    }

    // 길이 순으로 계산
    for (int len = 2; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = INF;

            int lo = opt[i][j - 1];
            int hi = opt[i + 1][j];
            if (lo < i) lo = i;
            if (hi > j - 1) hi = j - 1;

            for (int k = lo; k <= hi; k++) {
                ll val = dp[i][k] + dp[k + 1][j] + cost[i][j];
                if (val < dp[i][j]) {
                    dp[i][j] = val;
                    opt[i][j] = k;
                }
            }
        }
    }

    return dp[1][n];
}

// =============================================================================
// 5. Monotonic Queue Optimization
// =============================================================================

// dp[i] = min(dp[j]) + cost[i] for j in [i-k, i-1]

vector<ll> monotonicQueueDP(int n, int k, const vector<ll>& cost) {
    vector<ll> dp(n + 1);
    deque<int> dq;

    dp[0] = 0;
    dq.push_back(0);

    for (int i = 1; i <= n; i++) {
        // 윈도우 벗어난 원소 제거
        while (!dq.empty() && dq.front() < i - k) {
            dq.pop_front();
        }

        // 최솟값 사용
        dp[i] = dp[dq.front()] + cost[i];

        // 새 원소 추가 (모노토닉 유지)
        while (!dq.empty() && dp[dq.back()] >= dp[i]) {
            dq.pop_back();
        }
        dq.push_back(i);
    }

    return dp;
}

// =============================================================================
// 6. SOS DP (Sum over Subsets)
// =============================================================================

void sosDP(vector<ll>& dp, int n) {
    // dp[mask]에 mask의 모든 부분집합의 합 저장
    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            if (mask & (1 << i)) {
                dp[mask] += dp[mask ^ (1 << i)];
            }
        }
    }
}

// =============================================================================
// 7. Aliens Trick (WQS Binary Search)
// =============================================================================

// 정확히 k개 선택 문제를 페널티 λ로 변환
// dp[i] = max(value[i] - λ)의 형태로 변환 후 이분 탐색

pair<ll, int> alienDP(int n, const vector<ll>& values, ll penalty) {
    // 페널티 λ를 적용했을 때의 최대값과 선택 개수 반환
    ll maxVal = 0;
    int count = 0;

    for (int i = 0; i < n; i++) {
        ll val = values[i] - penalty;
        if (val > 0) {
            maxVal += val;
            count++;
        }
    }

    return {maxVal, count};
}

ll aliensOptimization(int n, int k, const vector<ll>& values) {
    ll lo = 0, hi = 1e18;
    ll result = 0;

    while (lo <= hi) {
        ll mid = (lo + hi) / 2;
        auto [val, cnt] = alienDP(n, values, mid);

        if (cnt >= k) {
            result = val + k * mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    return result;
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "고급 DP 최적화 예제" << endl;
    cout << "============================================================" << endl;

    // 1. Convex Hull Trick
    cout << "\n[1] Convex Hull Trick" << endl;
    ConvexHullTrick cht;
    cht.addLine(-2, 4);   // y = -2x + 4
    cht.addLine(-1, 3);   // y = -x + 3
    cht.addLine(0, 2);    // y = 2

    cout << "    직선: y=-2x+4, y=-x+3, y=2" << endl;
    cout << "    x=0 최솟값: " << cht.queryBinarySearch(0) << endl;
    cout << "    x=1 최솟값: " << cht.queryBinarySearch(1) << endl;
    cout << "    x=2 최솟값: " << cht.queryBinarySearch(2) << endl;

    // 2. Li Chao Tree
    cout << "\n[2] Li Chao Tree" << endl;
    LiChaoTree lct(0, 100);
    lct.addLine(-2, 10);
    lct.addLine(1, 0);
    lct.addLine(-1, 8);

    cout << "    직선: y=-2x+10, y=x, y=-x+8" << endl;
    cout << "    x=0 최솟값: " << lct.query(0) << endl;
    cout << "    x=3 최솟값: " << lct.query(3) << endl;
    cout << "    x=5 최솟값: " << lct.query(5) << endl;

    // 3. 모노토닉 큐 DP
    cout << "\n[3] 모노토닉 큐 DP" << endl;
    vector<ll> cost = {0, 1, 3, 2, 4, 1, 5};
    auto dp = monotonicQueueDP(6, 3, cost);
    cout << "    비용: [1, 3, 2, 4, 1, 5], k=3" << endl;
    cout << "    DP: [";
    for (int i = 1; i <= 6; i++) {
        cout << dp[i];
        if (i < 6) cout << ", ";
    }
    cout << "]" << endl;

    // 4. SOS DP
    cout << "\n[4] SOS DP" << endl;
    vector<ll> sos = {1, 2, 3, 4, 5, 6, 7, 8};  // 2^3 = 8
    cout << "    초기: [1, 2, 3, 4, 5, 6, 7, 8]" << endl;
    sosDP(sos, 3);
    cout << "    SOS: [";
    for (int i = 0; i < 8; i++) {
        cout << sos[i];
        if (i < 7) cout << ", ";
    }
    cout << "]" << endl;

    // 5. 복잡도 비교
    cout << "\n[5] 복잡도 비교" << endl;
    cout << "    | 기법              | 원래 복잡도 | 최적화 후    |" << endl;
    cout << "    |-------------------|-------------|--------------|" << endl;
    cout << "    | CHT               | O(n²)       | O(n log n)   |" << endl;
    cout << "    | Li Chao Tree      | O(n²)       | O(n log C)   |" << endl;
    cout << "    | D&C 최적화        | O(kn²)      | O(kn log n)  |" << endl;
    cout << "    | Knuth 최적화      | O(n³)       | O(n²)        |" << endl;
    cout << "    | 모노토닉 큐       | O(nk)       | O(n)         |" << endl;
    cout << "    | SOS DP            | O(3^n)      | O(n × 2^n)   |" << endl;
    cout << "    | Aliens Trick      | O(n²)       | O(n log C)   |" << endl;

    // 6. 적용 조건
    cout << "\n[6] 적용 조건" << endl;
    cout << "    CHT:" << endl;
    cout << "      - dp[i] = min(dp[j] + a[j] × b[i]) 형태" << endl;
    cout << "      - a[j] 또는 b[i]가 단조" << endl;
    cout << "    D&C 최적화:" << endl;
    cout << "      - opt[i][j] <= opt[i][j+1]" << endl;
    cout << "      - Quadrangle Inequality" << endl;
    cout << "    Knuth 최적화:" << endl;
    cout << "      - opt[i][j-1] <= opt[i][j] <= opt[i+1][j]" << endl;
    cout << "      - 구간 DP에서 주로 사용" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
