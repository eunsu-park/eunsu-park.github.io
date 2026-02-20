/*
 * 동적 프로그래밍 (Dynamic Programming)
 * Fibonacci, Knapsack, LCS, LIS, Edit Distance
 *
 * 부분 문제의 최적 해를 이용한 문제 해결 기법입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* =============================================================================
 * 1. 피보나치 (Memoization & Tabulation)
 * ============================================================================= */

long long fib_memo[100];
int fib_computed[100];

long long fibonacci_memo(int n) {
    if (n <= 1) return n;
    if (fib_computed[n]) return fib_memo[n];
    fib_computed[n] = 1;
    fib_memo[n] = fibonacci_memo(n - 1) + fibonacci_memo(n - 2);
    return fib_memo[n];
}

long long fibonacci_tab(int n) {
    if (n <= 1) return n;
    long long* dp = malloc((n + 1) * sizeof(long long));
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++)
        dp[i] = dp[i - 1] + dp[i - 2];
    long long result = dp[n];
    free(dp);
    return result;
}

/* =============================================================================
 * 2. 0/1 배낭 문제
 * ============================================================================= */

int knapsack_01(int W, int weights[], int values[], int n) {
    int** dp = malloc((n + 1) * sizeof(int*));
    for (int i = 0; i <= n; i++)
        dp[i] = calloc(W + 1, sizeof(int));

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            dp[i][w] = dp[i - 1][w];
            if (weights[i - 1] <= w) {
                int take = dp[i - 1][w - weights[i - 1]] + values[i - 1];
                dp[i][w] = MAX(dp[i][w], take);
            }
        }
    }

    int result = dp[n][W];
    for (int i = 0; i <= n; i++) free(dp[i]);
    free(dp);
    return result;
}

/* 공간 최적화 */
int knapsack_01_optimized(int W, int weights[], int values[], int n) {
    int* dp = calloc(W + 1, sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int w = W; w >= weights[i]; w--) {
            dp[w] = MAX(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    int result = dp[W];
    free(dp);
    return result;
}

/* =============================================================================
 * 3. 최장 공통 부분 수열 (LCS)
 * ============================================================================= */

int lcs(const char* s1, const char* s2) {
    int m = strlen(s1);
    int n = strlen(s2);

    int** dp = malloc((m + 1) * sizeof(int*));
    for (int i = 0; i <= m; i++)
        dp[i] = calloc(n + 1, sizeof(int));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = MAX(dp[i - 1][j], dp[i][j - 1]);
        }
    }

    int result = dp[m][n];
    for (int i = 0; i <= m; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * 4. 최장 증가 부분 수열 (LIS)
 * ============================================================================= */

/* O(n²) */
int lis_n2(int arr[], int n) {
    int* dp = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) dp[i] = 1;

    int max_len = 1;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i] && dp[j] + 1 > dp[i])
                dp[i] = dp[j] + 1;
        }
        if (dp[i] > max_len) max_len = dp[i];
    }

    free(dp);
    return max_len;
}

/* O(n log n) */
int lower_bound(int arr[], int n, int val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int lis_nlogn(int arr[], int n) {
    int* tails = malloc(n * sizeof(int));
    int len = 0;

    for (int i = 0; i < n; i++) {
        int pos = lower_bound(tails, len, arr[i]);
        tails[pos] = arr[i];
        if (pos == len) len++;
    }

    free(tails);
    return len;
}

/* =============================================================================
 * 5. 편집 거리
 * ============================================================================= */

int edit_distance(const char* s1, const char* s2) {
    int m = strlen(s1);
    int n = strlen(s2);

    int** dp = malloc((m + 1) * sizeof(int*));
    for (int i = 0; i <= m; i++)
        dp[i] = malloc((n + 1) * sizeof(int));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + MIN(dp[i - 1][j - 1],
                                   MIN(dp[i - 1][j], dp[i][j - 1]));
            }
        }
    }

    int result = dp[m][n];
    for (int i = 0; i <= m; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * 6. 동전 교환
 * ============================================================================= */

int coin_change(int coins[], int n, int amount) {
    int* dp = malloc((amount + 1) * sizeof(int));
    for (int i = 0; i <= amount; i++) dp[i] = amount + 1;
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int j = 0; j < n; j++) {
            if (coins[j] <= i && dp[i - coins[j]] + 1 < dp[i]) {
                dp[i] = dp[i - coins[j]] + 1;
            }
        }
    }

    int result = (dp[amount] > amount) ? -1 : dp[amount];
    free(dp);
    return result;
}

/* =============================================================================
 * 7. 행렬 체인 곱셈
 * ============================================================================= */

int matrix_chain(int dims[], int n) {
    int** dp = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        dp[i] = calloc(n, sizeof(int));
    }

    for (int len = 2; len < n; len++) {
        for (int i = 1; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = 2147483647;
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k + 1][j] + dims[i - 1] * dims[k] * dims[j];
                if (cost < dp[i][j]) dp[i][j] = cost;
            }
        }
    }

    int result = dp[1][n - 1];
    for (int i = 0; i < n; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("동적 프로그래밍 (DP) 예제\n");
    printf("============================================================\n");

    /* 1. 피보나치 */
    printf("\n[1] 피보나치\n");
    printf("    fib(10) = %lld\n", fibonacci_tab(10));
    printf("    fib(45) = %lld\n", fibonacci_tab(45));

    /* 2. 0/1 배낭 */
    printf("\n[2] 0/1 배낭 문제\n");
    int weights[] = {1, 2, 3, 4, 5};
    int values[] = {1, 6, 10, 16, 20};
    printf("    무게: [1,2,3,4,5], 가치: [1,6,10,16,20]\n");
    printf("    용량 8의 최대 가치: %d\n", knapsack_01(8, weights, values, 5));

    /* 3. LCS */
    printf("\n[3] 최장 공통 부분 수열 (LCS)\n");
    printf("    'ABCDGH' vs 'AEDFHR': %d\n", lcs("ABCDGH", "AEDFHR"));
    printf("    'AGGTAB' vs 'GXTXAYB': %d\n", lcs("AGGTAB", "GXTXAYB"));

    /* 4. LIS */
    printf("\n[4] 최장 증가 부분 수열 (LIS)\n");
    int arr[] = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    printf("    [10,22,9,33,21,50,41,60,80]\n");
    printf("    O(n²): %d\n", lis_n2(arr, 9));
    printf("    O(n log n): %d\n", lis_nlogn(arr, 9));

    /* 5. 편집 거리 */
    printf("\n[5] 편집 거리\n");
    printf("    'kitten' → 'sitting': %d\n", edit_distance("kitten", "sitting"));
    printf("    'sunday' → 'saturday': %d\n", edit_distance("sunday", "saturday"));

    /* 6. 동전 교환 */
    printf("\n[6] 동전 교환\n");
    int coins[] = {1, 5, 10, 25};
    printf("    동전: [1,5,10,25], 금액 30\n");
    printf("    최소 동전 수: %d\n", coin_change(coins, 4, 30));

    /* 7. 행렬 체인 */
    printf("\n[7] 행렬 체인 곱셈\n");
    int dims[] = {10, 30, 5, 60};
    printf("    차원: 10x30, 30x5, 5x60\n");
    printf("    최소 곱셈 횟수: %d\n", matrix_chain(dims, 4));

    /* 8. DP 문제 분류 */
    printf("\n[8] DP 문제 유형\n");
    printf("    | 유형          | 예시                    | 복잡도      |\n");
    printf("    |---------------|-------------------------|-------------|\n");
    printf("    | 1차원 DP      | 피보나치, 계단 오르기   | O(n)        |\n");
    printf("    | 2차원 DP      | 배낭, LCS, 편집거리     | O(n²) or O(nm)|\n");
    printf("    | 구간 DP       | 행렬 체인, 팰린드롬     | O(n³)       |\n");

    printf("\n============================================================\n");

    return 0;
}
