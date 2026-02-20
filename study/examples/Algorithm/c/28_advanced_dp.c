/*
 * 고급 DP 최적화 (Advanced DP Optimization)
 * CHT, D&C 최적화, Knuth 최적화, 모노토닉 큐
 *
 * DP의 시간 복잡도를 줄이는 고급 기법들입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define INF LLONG_MAX
#define MAX_N 100001

typedef long long ll;

/* =============================================================================
 * 1. Convex Hull Trick (CHT)
 * ============================================================================= */

/* 직선 구조체 */
typedef struct {
    ll m;  /* 기울기 */
    ll b;  /* y절편 */
} Line;

/* CHT 구조체 (기울기 단조 감소 버전) */
typedef struct {
    Line* lines;
    int size;
    int capacity;
} CHT;

CHT* cht_create(int capacity) {
    CHT* cht = malloc(sizeof(CHT));
    cht->lines = malloc(capacity * sizeof(Line));
    cht->size = 0;
    cht->capacity = capacity;
    return cht;
}

void cht_free(CHT* cht) {
    free(cht->lines);
    free(cht);
}

/* 교차점 x좌표 비교 */
bool cht_bad(Line l1, Line l2, Line l3) {
    /* l2가 불필요한지 확인 */
    /* (l3.b - l1.b) / (l1.m - l3.m) <= (l2.b - l1.b) / (l1.m - l2.m) */
    return (l3.b - l1.b) * (l1.m - l2.m) <= (l2.b - l1.b) * (l1.m - l3.m);
}

/* 직선 추가 (기울기 단조 감소) */
void cht_add(CHT* cht, ll m, ll b) {
    Line new_line = {m, b};

    while (cht->size >= 2 &&
           cht_bad(cht->lines[cht->size - 2], cht->lines[cht->size - 1], new_line)) {
        cht->size--;
    }

    cht->lines[cht->size++] = new_line;
}

/* 최솟값 쿼리 */
ll cht_query(CHT* cht, ll x) {
    int lo = 0, hi = cht->size - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        ll y1 = cht->lines[mid].m * x + cht->lines[mid].b;
        ll y2 = cht->lines[mid + 1].m * x + cht->lines[mid + 1].b;
        if (y1 > y2) lo = mid + 1;
        else hi = mid;
    }
    return cht->lines[lo].m * x + cht->lines[lo].b;
}

/* =============================================================================
 * 2. Li Chao Tree
 * ============================================================================= */

typedef struct LCNode {
    ll m, b;
    struct LCNode* left;
    struct LCNode* right;
} LCNode;

LCNode* lc_create(void) {
    LCNode* node = malloc(sizeof(LCNode));
    node->m = 0;
    node->b = INF;
    node->left = NULL;
    node->right = NULL;
    return node;
}

ll lc_eval(ll m, ll b, ll x) {
    return m * x + b;
}

void lc_insert(LCNode* node, ll lo, ll hi, ll m, ll b) {
    if (lo == hi) {
        if (lc_eval(m, b, lo) < lc_eval(node->m, node->b, lo)) {
            node->m = m;
            node->b = b;
        }
        return;
    }

    ll mid = (lo + hi) / 2;
    bool left_better = lc_eval(m, b, lo) < lc_eval(node->m, node->b, lo);
    bool mid_better = lc_eval(m, b, mid) < lc_eval(node->m, node->b, mid);

    if (mid_better) {
        ll tmp_m = node->m, tmp_b = node->b;
        node->m = m; node->b = b;
        m = tmp_m; b = tmp_b;
    }

    if (left_better != mid_better) {
        if (!node->left) node->left = lc_create();
        lc_insert(node->left, lo, mid, m, b);
    } else {
        if (!node->right) node->right = lc_create();
        lc_insert(node->right, mid + 1, hi, m, b);
    }
}

ll lc_query(LCNode* node, ll lo, ll hi, ll x) {
    if (!node) return INF;

    ll result = lc_eval(node->m, node->b, x);
    if (lo == hi) return result;

    ll mid = (lo + hi) / 2;
    if (x <= mid) {
        ll left_val = lc_query(node->left, lo, mid, x);
        if (left_val < result) result = left_val;
    } else {
        ll right_val = lc_query(node->right, mid + 1, hi, x);
        if (right_val < result) result = right_val;
    }
    return result;
}

/* =============================================================================
 * 3. D&C 최적화
 * ============================================================================= */

/* 조건: opt[i][j] <= opt[i][j+1]
 * 시간복잡도: O(kn log n) */

ll** cost;  /* cost[i][j]: i부터 j까지의 비용 */
ll** dp_dc;
int n_dc;

void compute_dp(int k, int lo, int hi, int opt_lo, int opt_hi) {
    if (lo > hi) return;

    int mid = (lo + hi) / 2;
    int opt = opt_lo;
    dp_dc[k][mid] = INF;

    for (int i = opt_lo; i <= opt_hi && i < mid; i++) {
        ll val = dp_dc[k - 1][i] + cost[i + 1][mid];
        if (val < dp_dc[k][mid]) {
            dp_dc[k][mid] = val;
            opt = i;
        }
    }

    compute_dp(k, lo, mid - 1, opt_lo, opt);
    compute_dp(k, mid + 1, hi, opt, opt_hi);
}

ll divide_conquer_opt(int n, int k, ll** cost_matrix) {
    cost = cost_matrix;
    n_dc = n;

    dp_dc = malloc((k + 1) * sizeof(ll*));
    for (int i = 0; i <= k; i++) {
        dp_dc[i] = malloc((n + 1) * sizeof(ll));
        for (int j = 0; j <= n; j++) {
            dp_dc[i][j] = INF;
        }
    }

    dp_dc[0][0] = 0;

    /* 첫 번째 그룹 */
    for (int j = 1; j <= n; j++) {
        dp_dc[1][j] = cost[1][j];
    }

    for (int i = 2; i <= k; i++) {
        compute_dp(i, 1, n, 0, n - 1);
    }

    ll result = dp_dc[k][n];

    for (int i = 0; i <= k; i++) free(dp_dc[i]);
    free(dp_dc);

    return result;
}

/* =============================================================================
 * 4. Knuth 최적화
 * ============================================================================= */

/* 조건: opt[i][j-1] <= opt[i][j] <= opt[i+1][j]
 * 시간복잡도: O(n²) */

ll knuth_opt(int n, ll** cost_matrix) {
    ll** dp = malloc((n + 2) * sizeof(ll*));
    int** opt = malloc((n + 2) * sizeof(int*));

    for (int i = 0; i <= n + 1; i++) {
        dp[i] = calloc(n + 2, sizeof(ll));
        opt[i] = calloc(n + 2, sizeof(int));
    }

    /* 기저 조건 */
    for (int i = 1; i <= n; i++) {
        opt[i][i] = i;
    }

    /* 길이 순으로 계산 */
    for (int len = 2; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = INF;

            int lo = opt[i][j - 1];
            int hi = opt[i + 1][j];
            if (lo < i) lo = i;
            if (hi > j) hi = j;

            for (int k = lo; k <= hi; k++) {
                ll val = dp[i][k - 1] + dp[k + 1][j] + cost_matrix[i][j];
                if (val < dp[i][j]) {
                    dp[i][j] = val;
                    opt[i][j] = k;
                }
            }
        }
    }

    ll result = dp[1][n];

    for (int i = 0; i <= n + 1; i++) {
        free(dp[i]);
        free(opt[i]);
    }
    free(dp);
    free(opt);

    return result;
}

/* =============================================================================
 * 5. 모노토닉 큐 최적화
 * ============================================================================= */

/* dp[i] = min(dp[j] + C[j]) for j in [i-k, i-1]
 * 슬라이딩 윈도우 최솟값 활용 */

ll monotonic_queue_dp(int n, int k, ll arr[]) {
    ll* dp = malloc((n + 1) * sizeof(ll));
    int* deque = malloc((n + 1) * sizeof(int));
    int front = 0, rear = 0;

    dp[0] = 0;
    deque[rear++] = 0;

    for (int i = 1; i <= n; i++) {
        /* 윈도우 벗어난 원소 제거 */
        while (front < rear && deque[front] < i - k) {
            front++;
        }

        /* 최솟값 사용 */
        dp[i] = dp[deque[front]] + arr[i];

        /* 새 원소 추가 (모노토닉 유지) */
        while (front < rear && dp[deque[rear - 1]] >= dp[i]) {
            rear--;
        }
        deque[rear++] = i;
    }

    ll result = dp[n];
    free(dp);
    free(deque);
    return result;
}

/* =============================================================================
 * 6. SOS DP (Sum over Subsets)
 * ============================================================================= */

/* f[mask] = sum of a[submask] for all submask of mask */
void sos_dp(ll a[], int n) {
    int size = 1 << n;

    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < size; mask++) {
            if (mask & (1 << i)) {
                a[mask] += a[mask ^ (1 << i)];
            }
        }
    }
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("고급 DP 최적화 예제\n");
    printf("============================================================\n");

    /* 1. Convex Hull Trick */
    printf("\n[1] Convex Hull Trick\n");
    CHT* cht = cht_create(100);

    /* 직선들: y = -2x + 4, y = -1x + 3, y = -0.5x + 2 */
    cht_add(cht, -2, 4);
    cht_add(cht, -1, 3);
    cht_add(cht, 0, 2);

    printf("    직선: y = -2x + 4, y = -x + 3, y = 2\n");
    printf("    x=0에서 최솟값: %lld\n", cht_query(cht, 0));
    printf("    x=1에서 최솟값: %lld\n", cht_query(cht, 1));
    printf("    x=2에서 최솟값: %lld\n", cht_query(cht, 2));
    printf("    x=3에서 최솟값: %lld\n", cht_query(cht, 3));
    cht_free(cht);

    /* 2. Li Chao Tree */
    printf("\n[2] Li Chao Tree\n");
    LCNode* root = lc_create();
    ll lo = 0, hi = 100;

    lc_insert(root, lo, hi, -2, 10);
    lc_insert(root, lo, hi, 1, 0);
    lc_insert(root, lo, hi, -1, 8);

    printf("    직선: y = -2x + 10, y = x, y = -x + 8\n");
    printf("    x=0에서 최솟값: %lld\n", lc_query(root, lo, hi, 0));
    printf("    x=3에서 최솟값: %lld\n", lc_query(root, lo, hi, 3));
    printf("    x=5에서 최솟값: %lld\n", lc_query(root, lo, hi, 5));

    /* 3. 모노토닉 큐 DP */
    printf("\n[3] 모노토닉 큐 DP\n");
    ll arr[] = {0, 1, 3, 2, 4, 1, 5};
    int n = 6, k = 3;
    printf("    배열: [1, 3, 2, 4, 1, 5]\n");
    printf("    윈도우 크기 k = 3\n");
    printf("    최소 비용: %lld\n", monotonic_queue_dp(n, k, arr));

    /* 4. SOS DP */
    printf("\n[4] SOS DP (Sum over Subsets)\n");
    ll sos_arr[] = {1, 2, 3, 4, 5, 6, 7, 8};  /* 2^3 = 8 */
    printf("    초기 배열: [1, 2, 3, 4, 5, 6, 7, 8]\n");
    sos_dp(sos_arr, 3);
    printf("    SOS 결과:\n");
    for (int mask = 0; mask < 8; mask++) {
        printf("      f[%d%d%d] = %lld\n",
               (mask >> 2) & 1, (mask >> 1) & 1, mask & 1, sos_arr[mask]);
    }

    /* 5. 복잡도 비교 */
    printf("\n[5] 복잡도 비교\n");
    printf("    | 기법              | 원래 복잡도 | 최적화 후    |\n");
    printf("    |-------------------|-------------|-------------|\n");
    printf("    | CHT               | O(n²)       | O(n log n)  |\n");
    printf("    | Li Chao Tree      | O(n²)       | O(n log C)  |\n");
    printf("    | D&C 최적화        | O(kn²)      | O(kn log n) |\n");
    printf("    | Knuth 최적화      | O(n³)       | O(n²)       |\n");
    printf("    | 모노토닉 큐       | O(nk)       | O(n)        |\n");
    printf("    | SOS DP            | O(3^n)      | O(n × 2^n)  |\n");

    /* 6. 적용 조건 */
    printf("\n[6] 적용 조건\n");
    printf("    CHT:\n");
    printf("      - dp[i] = min(dp[j] + a[j] × b[i]) 형태\n");
    printf("      - a[j] 또는 b[i]가 단조\n");
    printf("    D&C 최적화:\n");
    printf("      - opt[i][j] <= opt[i][j+1]\n");
    printf("      - 비용 함수가 Quadrangle Inequality 만족\n");
    printf("    Knuth 최적화:\n");
    printf("      - opt[i][j-1] <= opt[i][j] <= opt[i+1][j]\n");
    printf("      - 구간 DP에서 주로 사용\n");

    printf("\n============================================================\n");

    return 0;
}
