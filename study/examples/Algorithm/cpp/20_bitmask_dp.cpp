/*
 * 비트마스크 DP (Bitmask DP)
 * TSP, Subset Enumeration, Assignment Problem
 *
 * 집합의 상태를 비트로 표현하여 DP를 수행합니다.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <bitset>

using namespace std;

// =============================================================================
// 1. 비트 연산 기초
// =============================================================================

void bitOperations() {
    int n = 4;  // 집합 크기
    int fullSet = (1 << n) - 1;  // {0, 1, 2, 3}

    cout << "[1] 비트 연산 기초" << endl;

    // 원소 i 포함 여부
    int set = 0b1010;  // {1, 3}
    cout << "    집합 1010 (10진수: " << set << ")" << endl;
    cout << "    원소 1 포함: " << ((set & (1 << 1)) ? "예" : "아니오") << endl;
    cout << "    원소 2 포함: " << ((set & (1 << 2)) ? "예" : "아니오") << endl;

    // 원소 추가/제거
    set |= (1 << 2);   // 2 추가
    cout << "    2 추가 후: " << bitset<4>(set) << endl;
    set &= ~(1 << 1);  // 1 제거
    cout << "    1 제거 후: " << bitset<4>(set) << endl;

    // 원소 토글
    set ^= (1 << 3);  // 3 토글
    cout << "    3 토글 후: " << bitset<4>(set) << endl;

    // 집합 크기 (1의 개수)
    cout << "    집합 크기: " << __builtin_popcount(set) << endl;

    // 최하위 비트
    cout << "    최하위 1비트: " << (set & -set) << endl;
}

// =============================================================================
// 2. 부분집합 순회
// =============================================================================

void enumerateSubsets(int n) {
    cout << "\n[2] 부분집합 순회" << endl;
    cout << "    n = " << n << "인 집합의 모든 부분집합:" << endl;

    // 모든 부분집합
    cout << "    전체: ";
    for (int mask = 0; mask < (1 << n); mask++) {
        cout << bitset<3>(mask) << " ";
    }
    cout << endl;

    // 특정 집합의 부분집합
    int set = 0b101;  // {0, 2}
    cout << "    " << bitset<3>(set) << "의 부분집합: ";
    for (int sub = set; ; sub = (sub - 1) & set) {
        cout << bitset<3>(sub) << " ";
        if (sub == 0) break;
    }
    cout << endl;
}

// =============================================================================
// 3. 외판원 문제 (TSP)
// =============================================================================

const int INF = INT_MAX / 2;

int tsp(int n, const vector<vector<int>>& dist) {
    // dp[mask][i]: mask 집합을 방문하고 현재 i에 있을 때 최소 비용
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));

    dp[1][0] = 0;  // 시작점 0에서 출발

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u))) continue;
            if (dp[mask][u] == INF) continue;

            for (int v = 0; v < n; v++) {
                if (mask & (1 << v)) continue;  // 이미 방문
                if (dist[u][v] == INF) continue;

                int newMask = mask | (1 << v);
                dp[newMask][v] = min(dp[newMask][v], dp[mask][u] + dist[u][v]);
            }
        }
    }

    // 모든 도시 방문 후 시작점으로 복귀
    int fullMask = (1 << n) - 1;
    int result = INF;
    for (int i = 0; i < n; i++) {
        if (dp[fullMask][i] != INF && dist[i][0] != INF) {
            result = min(result, dp[fullMask][i] + dist[i][0]);
        }
    }

    return result;
}

// TSP 경로 복원
pair<int, vector<int>> tspWithPath(int n, const vector<vector<int>>& dist) {
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));
    vector<vector<int>> parent(1 << n, vector<int>(n, -1));

    dp[1][0] = 0;

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u))) continue;
            if (dp[mask][u] == INF) continue;

            for (int v = 0; v < n; v++) {
                if (mask & (1 << v)) continue;
                if (dist[u][v] == INF) continue;

                int newMask = mask | (1 << v);
                if (dp[mask][u] + dist[u][v] < dp[newMask][v]) {
                    dp[newMask][v] = dp[mask][u] + dist[u][v];
                    parent[newMask][v] = u;
                }
            }
        }
    }

    // 최종 결과
    int fullMask = (1 << n) - 1;
    int result = INF;
    int lastCity = -1;

    for (int i = 0; i < n; i++) {
        if (dp[fullMask][i] != INF && dist[i][0] != INF) {
            if (dp[fullMask][i] + dist[i][0] < result) {
                result = dp[fullMask][i] + dist[i][0];
                lastCity = i;
            }
        }
    }

    // 경로 복원
    vector<int> path;
    int mask = fullMask;
    int curr = lastCity;

    while (curr != -1) {
        path.push_back(curr);
        int prev = parent[mask][curr];
        mask ^= (1 << curr);
        curr = prev;
    }

    reverse(path.begin(), path.end());
    path.push_back(0);  // 시작점으로 복귀

    return {result, path};
}

// =============================================================================
// 4. 작업 할당 문제 (Assignment Problem)
// =============================================================================

int minCostAssignment(int n, const vector<vector<int>>& cost) {
    // dp[mask]: mask에 해당하는 작업들이 할당되었을 때 최소 비용
    vector<int> dp(1 << n, INF);
    dp[0] = 0;

    for (int mask = 0; mask < (1 << n); mask++) {
        int person = __builtin_popcount(mask);  // 현재 할당할 사람
        if (person >= n) continue;

        for (int job = 0; job < n; job++) {
            if (mask & (1 << job)) continue;  // 이미 할당된 작업

            int newMask = mask | (1 << job);
            dp[newMask] = min(dp[newMask], dp[mask] + cost[person][job]);
        }
    }

    return dp[(1 << n) - 1];
}

// =============================================================================
// 5. 부분집합 합 (Subset Sum)
// =============================================================================

bool subsetSum(const vector<int>& nums, int target) {
    int n = nums.size();

    for (int mask = 0; mask < (1 << n); mask++) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                sum += nums[i];
            }
        }
        if (sum == target) return true;
    }

    return false;
}

// 최적화: Meet in the Middle
bool subsetSumMITM(const vector<int>& nums, int target) {
    int n = nums.size();
    int half = n / 2;

    // 앞쪽 절반의 모든 부분집합 합
    vector<int> leftSums;
    for (int mask = 0; mask < (1 << half); mask++) {
        int sum = 0;
        for (int i = 0; i < half; i++) {
            if (mask & (1 << i)) sum += nums[i];
        }
        leftSums.push_back(sum);
    }
    sort(leftSums.begin(), leftSums.end());

    // 뒤쪽 절반 확인
    int rightHalf = n - half;
    for (int mask = 0; mask < (1 << rightHalf); mask++) {
        int sum = 0;
        for (int i = 0; i < rightHalf; i++) {
            if (mask & (1 << i)) sum += nums[half + i];
        }

        // target - sum이 leftSums에 있는지 확인
        if (binary_search(leftSums.begin(), leftSums.end(), target - sum)) {
            return true;
        }
    }

    return false;
}

// =============================================================================
// 6. 해밀턴 경로
// =============================================================================

int countHamiltonianPaths(int n, const vector<vector<int>>& adj) {
    // dp[mask][i]: mask 집합을 방문하고 i에서 끝나는 경로 수
    vector<vector<int>> dp(1 << n, vector<int>(n, 0));

    // 시작점 초기화
    for (int i = 0; i < n; i++) {
        dp[1 << i][i] = 1;
    }

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u))) continue;
            if (dp[mask][u] == 0) continue;

            for (int v : adj[u]) {
                if (mask & (1 << v)) continue;

                int newMask = mask | (1 << v);
                dp[newMask][v] += dp[mask][u];
            }
        }
    }

    // 모든 정점 방문한 경로 수
    int fullMask = (1 << n) - 1;
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += dp[fullMask][i];
    }

    return total;
}

// =============================================================================
// 7. SOS DP (Sum over Subsets)
// =============================================================================

void sosDP(vector<int>& dp, int n) {
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
    cout << "비트마스크 DP 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 비트 연산 기초
    bitOperations();

    // 2. 부분집합 순회
    enumerateSubsets(3);

    // 3. TSP
    cout << "\n[3] 외판원 문제 (TSP)" << endl;
    vector<vector<int>> dist = {
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0}
    };
    cout << "    거리 행렬: 4x4" << endl;
    cout << "    최소 비용: " << tsp(4, dist) << endl;

    auto [cost, path] = tspWithPath(4, dist);
    cout << "    경로: ";
    printVector(path);
    cout << endl;

    // 4. 작업 할당
    cout << "\n[4] 작업 할당 문제" << endl;
    vector<vector<int>> costMatrix = {
        {9, 2, 7, 8},
        {6, 4, 3, 7},
        {5, 8, 1, 8},
        {7, 6, 9, 4}
    };
    cout << "    비용 행렬: 4x4" << endl;
    cout << "    최소 비용: " << minCostAssignment(4, costMatrix) << endl;

    // 5. 부분집합 합
    cout << "\n[5] 부분집합 합" << endl;
    vector<int> nums = {3, 34, 4, 12, 5, 2};
    cout << "    배열: [3, 34, 4, 12, 5, 2]" << endl;
    cout << "    합 9 존재: " << (subsetSum(nums, 9) ? "예" : "아니오") << endl;
    cout << "    합 30 존재: " << (subsetSum(nums, 30) ? "예" : "아니오") << endl;

    // 6. SOS DP
    cout << "\n[6] SOS DP" << endl;
    vector<int> sos = {1, 2, 3, 4, 5, 6, 7, 8};  // 2^3 = 8
    cout << "    초기: ";
    printVector(sos);
    cout << endl;
    sosDP(sos, 3);
    cout << "    SOS: ";
    printVector(sos);
    cout << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 문제               | 시간복잡도    | 공간복잡도 |" << endl;
    cout << "    |--------------------|---------------|------------|" << endl;
    cout << "    | TSP                | O(n² × 2^n)   | O(n × 2^n) |" << endl;
    cout << "    | 작업 할당          | O(n × 2^n)    | O(2^n)     |" << endl;
    cout << "    | 부분집합 합        | O(2^n)        | O(1)       |" << endl;
    cout << "    | Meet in the Middle | O(n × 2^(n/2))| O(2^(n/2)) |" << endl;
    cout << "    | SOS DP             | O(n × 2^n)    | O(2^n)     |" << endl;

    // 8. 비트마스크 팁
    cout << "\n[8] 비트마스크 팁" << endl;
    cout << "    - n ≤ 20: 비트마스크 DP 고려" << endl;
    cout << "    - n ≤ 25: Meet in the Middle 고려" << endl;
    cout << "    - __builtin_popcount(x): 1의 개수 (GCC)" << endl;
    cout << "    - x & -x: 최하위 1비트" << endl;
    cout << "    - x & (x-1): 최하위 1비트 제거" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
