/*
 * 동적 프로그래밍 (Dynamic Programming)
 * Fibonacci, Knapsack, LCS, LIS, Edit Distance, Matrix Chain
 *
 * 복잡한 문제를 부분 문제로 나누어 해결합니다.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>

using namespace std;

// =============================================================================
// 1. 피보나치 (Top-down, Bottom-up)
// =============================================================================

// Top-down (Memoization)
vector<long long> memo;

long long fibTopDown(int n) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    return memo[n] = fibTopDown(n - 1) + fibTopDown(n - 2);
}

// Bottom-up (Tabulation)
long long fibBottomUp(int n) {
    if (n <= 1) return n;

    vector<long long> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
}

// 공간 최적화
long long fibOptimized(int n) {
    if (n <= 1) return n;

    long long prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        long long curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

// =============================================================================
// 2. 0/1 배낭 문제
// =============================================================================

int knapsack01(int W, const vector<int>& weights, const vector<int>& values) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            dp[i][w] = dp[i-1][w];  // 안 담음
            if (weights[i-1] <= w) {
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1]);
            }
        }
    }

    return dp[n][W];
}

// 공간 최적화
int knapsack01Optimized(int W, const vector<int>& weights, const vector<int>& values) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int i = 0; i < n; i++) {
        for (int w = W; w >= weights[i]; w--) {
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

// 무한 배낭 (Unbounded)
int unboundedKnapsack(int W, const vector<int>& weights, const vector<int>& values) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int w = 1; w <= W; w++) {
        for (int i = 0; i < n; i++) {
            if (weights[i] <= w) {
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }
    }

    return dp[W];
}

// =============================================================================
// 3. 최장 공통 부분수열 (LCS)
// =============================================================================

int lcs(const string& s1, const string& s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[m][n];
}

// LCS 문자열 복원
string lcsString(const string& s1, const string& s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    // 역추적
    string result;
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (s1[i-1] == s2[j-1]) {
            result = s1[i-1] + result;
            i--; j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            i--;
        } else {
            j--;
        }
    }

    return result;
}

// =============================================================================
// 4. 최장 증가 부분수열 (LIS)
// =============================================================================

// O(n²)
int lisQuadratic(const vector<int>& arr) {
    int n = arr.size();
    vector<int> dp(n, 1);

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    return *max_element(dp.begin(), dp.end());
}

// O(n log n)
int lisNLogN(const vector<int>& arr) {
    vector<int> tails;

    for (int x : arr) {
        auto it = lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) {
            tails.push_back(x);
        } else {
            *it = x;
        }
    }

    return tails.size();
}

// LIS 수열 복원
vector<int> lisWithSequence(const vector<int>& arr) {
    int n = arr.size();
    vector<int> tails;
    vector<int> tailIdx;
    vector<int> prev(n, -1);

    for (int i = 0; i < n; i++) {
        auto it = lower_bound(tails.begin(), tails.end(), arr[i]);
        int pos = it - tails.begin();

        if (it == tails.end()) {
            tails.push_back(arr[i]);
            tailIdx.push_back(i);
        } else {
            *it = arr[i];
            tailIdx[pos] = i;
        }

        if (pos > 0) {
            prev[i] = tailIdx[pos - 1];
        }
    }

    // 역추적
    vector<int> result;
    for (int i = tailIdx.back(); i != -1; i = prev[i]) {
        result.push_back(arr[i]);
    }
    reverse(result.begin(), result.end());

    return result;
}

// =============================================================================
// 5. 편집 거리 (Edit Distance)
// =============================================================================

int editDistance(const string& s1, const string& s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min({dp[i-1][j],      // 삭제
                                    dp[i][j-1],      // 삽입
                                    dp[i-1][j-1]});  // 교체
            }
        }
    }

    return dp[m][n];
}

// =============================================================================
// 6. 행렬 체인 곱셈
// =============================================================================

int matrixChainMultiplication(const vector<int>& dims) {
    int n = dims.size() - 1;
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;

            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] +
                           dims[i] * dims[k+1] * dims[j+1];
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }

    return dp[0][n-1];
}

// =============================================================================
// 7. 동전 교환 (Coin Change)
// =============================================================================

// 최소 동전 개수
int coinChange(const vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i && dp[i - coin] != INT_MAX) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }

    return dp[amount] == INT_MAX ? -1 : dp[amount];
}

// 경우의 수
int coinChangeWays(const vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;

    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }

    return dp[amount];
}

// =============================================================================
// 8. 최대 부분 배열 합 (Kadane's Algorithm)
// =============================================================================

int maxSubarraySum(const vector<int>& arr) {
    int maxSum = arr[0];
    int currSum = arr[0];

    for (int i = 1; i < (int)arr.size(); i++) {
        currSum = max(arr[i], currSum + arr[i]);
        maxSum = max(maxSum, currSum);
    }

    return maxSum;
}

// =============================================================================
// 9. 팰린드롬 부분문자열
// =============================================================================

// 가장 긴 팰린드롬 부분문자열
string longestPalindrome(const string& s) {
    int n = s.length();
    if (n == 0) return "";

    vector<vector<bool>> dp(n, vector<bool>(n, false));
    int start = 0, maxLen = 1;

    // 길이 1
    for (int i = 0; i < n; i++) dp[i][i] = true;

    // 길이 2
    for (int i = 0; i < n - 1; i++) {
        if (s[i] == s[i+1]) {
            dp[i][i+1] = true;
            start = i;
            maxLen = 2;
        }
    }

    // 길이 3 이상
    for (int len = 3; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            if (s[i] == s[j] && dp[i+1][j-1]) {
                dp[i][j] = true;
                start = i;
                maxLen = len;
            }
        }
    }

    return s.substr(start, maxLen);
}

// 팰린드롬 분할 최소 횟수
int minPalindromeCuts(const string& s) {
    int n = s.length();

    // isPalin[i][j]: s[i..j]가 팰린드롬인지
    vector<vector<bool>> isPalin(n, vector<bool>(n, false));
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                isPalin[i][j] = true;
            } else if (s[i] == s[j]) {
                isPalin[i][j] = (j - i == 1) || isPalin[i+1][j-1];
            }
        }
    }

    // dp[i]: s[0..i]의 최소 분할 횟수
    vector<int> dp(n, 0);
    for (int i = 0; i < n; i++) {
        if (isPalin[0][i]) {
            dp[i] = 0;
        } else {
            dp[i] = INT_MAX;
            for (int j = 0; j < i; j++) {
                if (isPalin[j+1][i]) {
                    dp[i] = min(dp[i], dp[j] + 1);
                }
            }
        }
    }

    return dp[n-1];
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
    cout << "동적 프로그래밍 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 피보나치
    cout << "\n[1] 피보나치" << endl;
    memo.assign(50, -1);
    cout << "    fib(10) Top-down: " << fibTopDown(10) << endl;
    cout << "    fib(10) Bottom-up: " << fibBottomUp(10) << endl;
    cout << "    fib(10) Optimized: " << fibOptimized(10) << endl;

    // 2. 0/1 배낭
    cout << "\n[2] 0/1 배낭" << endl;
    vector<int> weights = {2, 3, 4, 5};
    vector<int> values = {3, 4, 5, 6};
    int W = 8;
    cout << "    무게: [2,3,4,5], 가치: [3,4,5,6], 용량: 8" << endl;
    cout << "    최대 가치: " << knapsack01(W, weights, values) << endl;

    // 3. LCS
    cout << "\n[3] 최장 공통 부분수열 (LCS)" << endl;
    string s1 = "ABCDGH", s2 = "AEDFHR";
    cout << "    s1: \"ABCDGH\", s2: \"AEDFHR\"" << endl;
    cout << "    LCS 길이: " << lcs(s1, s2) << endl;
    cout << "    LCS: \"" << lcsString(s1, s2) << "\"" << endl;

    // 4. LIS
    cout << "\n[4] 최장 증가 부분수열 (LIS)" << endl;
    vector<int> arr = {10, 9, 2, 5, 3, 7, 101, 18};
    cout << "    배열: [10,9,2,5,3,7,101,18]" << endl;
    cout << "    LIS 길이 O(n²): " << lisQuadratic(arr) << endl;
    cout << "    LIS 길이 O(n log n): " << lisNLogN(arr) << endl;
    cout << "    LIS: ";
    printVector(lisWithSequence(arr));
    cout << endl;

    // 5. 편집 거리
    cout << "\n[5] 편집 거리" << endl;
    cout << "    \"horse\" → \"ros\": " << editDistance("horse", "ros") << endl;

    // 6. 행렬 체인 곱셈
    cout << "\n[6] 행렬 체인 곱셈" << endl;
    vector<int> dims = {10, 30, 5, 60};
    cout << "    행렬 크기: 10×30, 30×5, 5×60" << endl;
    cout << "    최소 곱셈 횟수: " << matrixChainMultiplication(dims) << endl;

    // 7. 동전 교환
    cout << "\n[7] 동전 교환" << endl;
    vector<int> coins = {1, 2, 5};
    cout << "    동전: [1,2,5], 금액: 11" << endl;
    cout << "    최소 개수: " << coinChange(coins, 11) << endl;
    cout << "    경우의 수: " << coinChangeWays(coins, 11) << endl;

    // 8. 최대 부분 배열 합
    cout << "\n[8] 최대 부분 배열 합" << endl;
    vector<int> subArr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "    배열: [-2,1,-3,4,-1,2,1,-5,4]" << endl;
    cout << "    최대 합: " << maxSubarraySum(subArr) << endl;

    // 9. 팰린드롬
    cout << "\n[9] 팰린드롬" << endl;
    cout << "    \"babad\" 가장 긴 팰린드롬: \"" << longestPalindrome("babad") << "\"" << endl;
    cout << "    \"aab\" 최소 분할: " << minPalindromeCuts("aab") << endl;

    // 10. 복잡도 요약
    cout << "\n[10] 복잡도 요약" << endl;
    cout << "    | 문제              | 시간복잡도    | 공간복잡도 |" << endl;
    cout << "    |-------------------|---------------|------------|" << endl;
    cout << "    | 피보나치          | O(n)          | O(1)       |" << endl;
    cout << "    | 0/1 배낭          | O(nW)         | O(W)       |" << endl;
    cout << "    | LCS               | O(mn)         | O(mn)      |" << endl;
    cout << "    | LIS               | O(n log n)    | O(n)       |" << endl;
    cout << "    | 편집 거리         | O(mn)         | O(n)       |" << endl;
    cout << "    | 행렬 체인         | O(n³)         | O(n²)      |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
