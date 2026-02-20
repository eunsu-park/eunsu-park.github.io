/*
 * 세그먼트 트리 (Segment Tree)
 * 구간 합, 구간 최소/최대, Lazy Propagation
 *
 * 구간 쿼리와 업데이트를 O(log n)에 처리합니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

#define MAX_N 100001
#define INF INT_MAX

/* =============================================================================
 * 1. 기본 세그먼트 트리 (구간 합)
 * ============================================================================= */

typedef struct {
    long long* tree;
    int n;
} SegmentTree;

SegmentTree* st_create(int n) {
    SegmentTree* st = malloc(sizeof(SegmentTree));
    st->n = n;
    st->tree = calloc(4 * n, sizeof(long long));
    return st;
}

void st_free(SegmentTree* st) {
    free(st->tree);
    free(st);
}

void st_build(SegmentTree* st, int arr[], int node, int start, int end) {
    if (start == end) {
        st->tree[node] = arr[start];
        return;
    }
    int mid = (start + end) / 2;
    st_build(st, arr, 2 * node, start, mid);
    st_build(st, arr, 2 * node + 1, mid + 1, end);
    st->tree[node] = st->tree[2 * node] + st->tree[2 * node + 1];
}

/* 점 업데이트 */
void st_update(SegmentTree* st, int node, int start, int end, int idx, long long val) {
    if (start == end) {
        st->tree[node] = val;
        return;
    }
    int mid = (start + end) / 2;
    if (idx <= mid) {
        st_update(st, 2 * node, start, mid, idx, val);
    } else {
        st_update(st, 2 * node + 1, mid + 1, end, idx, val);
    }
    st->tree[node] = st->tree[2 * node] + st->tree[2 * node + 1];
}

/* 구간 합 쿼리 */
long long st_query(SegmentTree* st, int node, int start, int end, int l, int r) {
    if (r < start || end < l) return 0;
    if (l <= start && end <= r) return st->tree[node];

    int mid = (start + end) / 2;
    return st_query(st, 2 * node, start, mid, l, r) +
           st_query(st, 2 * node + 1, mid + 1, end, l, r);
}

/* =============================================================================
 * 2. 구간 최소/최대 세그먼트 트리
 * ============================================================================= */

typedef struct {
    int* tree_min;
    int* tree_max;
    int n;
} MinMaxTree;

MinMaxTree* mmt_create(int n) {
    MinMaxTree* mmt = malloc(sizeof(MinMaxTree));
    mmt->n = n;
    mmt->tree_min = malloc(4 * n * sizeof(int));
    mmt->tree_max = malloc(4 * n * sizeof(int));
    for (int i = 0; i < 4 * n; i++) {
        mmt->tree_min[i] = INF;
        mmt->tree_max[i] = -INF;
    }
    return mmt;
}

void mmt_build(MinMaxTree* mmt, int arr[], int node, int start, int end) {
    if (start == end) {
        mmt->tree_min[node] = arr[start];
        mmt->tree_max[node] = arr[start];
        return;
    }
    int mid = (start + end) / 2;
    mmt_build(mmt, arr, 2 * node, start, mid);
    mmt_build(mmt, arr, 2 * node + 1, mid + 1, end);
    mmt->tree_min[node] = (mmt->tree_min[2 * node] < mmt->tree_min[2 * node + 1])
                          ? mmt->tree_min[2 * node] : mmt->tree_min[2 * node + 1];
    mmt->tree_max[node] = (mmt->tree_max[2 * node] > mmt->tree_max[2 * node + 1])
                          ? mmt->tree_max[2 * node] : mmt->tree_max[2 * node + 1];
}

int mmt_query_min(MinMaxTree* mmt, int node, int start, int end, int l, int r) {
    if (r < start || end < l) return INF;
    if (l <= start && end <= r) return mmt->tree_min[node];

    int mid = (start + end) / 2;
    int left = mmt_query_min(mmt, 2 * node, start, mid, l, r);
    int right = mmt_query_min(mmt, 2 * node + 1, mid + 1, end, l, r);
    return (left < right) ? left : right;
}

int mmt_query_max(MinMaxTree* mmt, int node, int start, int end, int l, int r) {
    if (r < start || end < l) return -INF;
    if (l <= start && end <= r) return mmt->tree_max[node];

    int mid = (start + end) / 2;
    int left = mmt_query_max(mmt, 2 * node, start, mid, l, r);
    int right = mmt_query_max(mmt, 2 * node + 1, mid + 1, end, l, r);
    return (left > right) ? left : right;
}

void mmt_free(MinMaxTree* mmt) {
    free(mmt->tree_min);
    free(mmt->tree_max);
    free(mmt);
}

/* =============================================================================
 * 3. Lazy Propagation (구간 업데이트)
 * ============================================================================= */

typedef struct {
    long long* tree;
    long long* lazy;
    int n;
} LazySegTree;

LazySegTree* lst_create(int n) {
    LazySegTree* lst = malloc(sizeof(LazySegTree));
    lst->n = n;
    lst->tree = calloc(4 * n, sizeof(long long));
    lst->lazy = calloc(4 * n, sizeof(long long));
    return lst;
}

void lst_free(LazySegTree* lst) {
    free(lst->tree);
    free(lst->lazy);
    free(lst);
}

void lst_build(LazySegTree* lst, int arr[], int node, int start, int end) {
    if (start == end) {
        lst->tree[node] = arr[start];
        return;
    }
    int mid = (start + end) / 2;
    lst_build(lst, arr, 2 * node, start, mid);
    lst_build(lst, arr, 2 * node + 1, mid + 1, end);
    lst->tree[node] = lst->tree[2 * node] + lst->tree[2 * node + 1];
}

void lst_propagate(LazySegTree* lst, int node, int start, int end) {
    if (lst->lazy[node] != 0) {
        lst->tree[node] += lst->lazy[node] * (end - start + 1);
        if (start != end) {
            lst->lazy[2 * node] += lst->lazy[node];
            lst->lazy[2 * node + 1] += lst->lazy[node];
        }
        lst->lazy[node] = 0;
    }
}

/* 구간 [l, r]에 val 더하기 */
void lst_update_range(LazySegTree* lst, int node, int start, int end, int l, int r, long long val) {
    lst_propagate(lst, node, start, end);

    if (r < start || end < l) return;

    if (l <= start && end <= r) {
        lst->lazy[node] = val;
        lst_propagate(lst, node, start, end);
        return;
    }

    int mid = (start + end) / 2;
    lst_update_range(lst, 2 * node, start, mid, l, r, val);
    lst_update_range(lst, 2 * node + 1, mid + 1, end, l, r, val);
    lst->tree[node] = lst->tree[2 * node] + lst->tree[2 * node + 1];
}

long long lst_query(LazySegTree* lst, int node, int start, int end, int l, int r) {
    lst_propagate(lst, node, start, end);

    if (r < start || end < l) return 0;
    if (l <= start && end <= r) return lst->tree[node];

    int mid = (start + end) / 2;
    return lst_query(lst, 2 * node, start, mid, l, r) +
           lst_query(lst, 2 * node + 1, mid + 1, end, l, r);
}

/* =============================================================================
 * 4. 동적 세그먼트 트리 (좌표 압축 없이)
 * ============================================================================= */

typedef struct DynamicNode {
    long long sum;
    struct DynamicNode* left;
    struct DynamicNode* right;
} DynamicNode;

DynamicNode* dn_create(void) {
    DynamicNode* node = malloc(sizeof(DynamicNode));
    node->sum = 0;
    node->left = NULL;
    node->right = NULL;
    return node;
}

void dn_update(DynamicNode* node, long long start, long long end, long long idx, long long val) {
    if (start == end) {
        node->sum += val;
        return;
    }
    long long mid = (start + end) / 2;
    if (idx <= mid) {
        if (!node->left) node->left = dn_create();
        dn_update(node->left, start, mid, idx, val);
    } else {
        if (!node->right) node->right = dn_create();
        dn_update(node->right, mid + 1, end, idx, val);
    }
    node->sum = (node->left ? node->left->sum : 0) +
                (node->right ? node->right->sum : 0);
}

long long dn_query(DynamicNode* node, long long start, long long end, long long l, long long r) {
    if (!node || r < start || end < l) return 0;
    if (l <= start && end <= r) return node->sum;

    long long mid = (start + end) / 2;
    return dn_query(node->left, start, mid, l, r) +
           dn_query(node->right, mid + 1, end, l, r);
}

void dn_free(DynamicNode* node) {
    if (!node) return;
    dn_free(node->left);
    dn_free(node->right);
    free(node);
}

/* =============================================================================
 * 5. 머지 소트 트리 (Merge Sort Tree)
 * ============================================================================= */

typedef struct {
    int** tree;
    int* sizes;
    int n;
} MergeSortTree;

void mst_merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (i = left; i <= right; i++) arr[i] = temp[i];
}

MergeSortTree* mst_create(int arr[], int n) {
    MergeSortTree* mst = malloc(sizeof(MergeSortTree));
    mst->n = n;
    mst->tree = malloc(4 * n * sizeof(int*));
    mst->sizes = calloc(4 * n, sizeof(int));

    /* 배열 복사 및 정렬 */
    int* temp = malloc(n * sizeof(int));
    int* copy = malloc(n * sizeof(int));
    memcpy(copy, arr, n * sizeof(int));

    /* 빌드 함수 호출 */
    void build(int node, int start, int end) {
        if (start == end) {
            mst->tree[node] = malloc(sizeof(int));
            mst->tree[node][0] = arr[start];
            mst->sizes[node] = 1;
            return;
        }
        int mid = (start + end) / 2;
        build(2 * node, start, mid);
        build(2 * node + 1, mid + 1, end);

        /* 병합 */
        int left_size = mst->sizes[2 * node];
        int right_size = mst->sizes[2 * node + 1];
        mst->sizes[node] = left_size + right_size;
        mst->tree[node] = malloc(mst->sizes[node] * sizeof(int));

        int i = 0, j = 0, k = 0;
        while (i < left_size && j < right_size) {
            if (mst->tree[2 * node][i] <= mst->tree[2 * node + 1][j])
                mst->tree[node][k++] = mst->tree[2 * node][i++];
            else
                mst->tree[node][k++] = mst->tree[2 * node + 1][j++];
        }
        while (i < left_size) mst->tree[node][k++] = mst->tree[2 * node][i++];
        while (j < right_size) mst->tree[node][k++] = mst->tree[2 * node + 1][j++];
    }

    build(1, 0, n - 1);
    free(temp);
    free(copy);
    return mst;
}

/* 구간 [l, r]에서 k 이하인 원소 개수 */
int count_le(int* arr, int size, int k) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] <= k) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int mst_query(MergeSortTree* mst, int node, int start, int end, int l, int r, int k) {
    if (r < start || end < l) return 0;
    if (l <= start && end <= r) {
        return count_le(mst->tree[node], mst->sizes[node], k);
    }
    int mid = (start + end) / 2;
    return mst_query(mst, 2 * node, start, mid, l, r, k) +
           mst_query(mst, 2 * node + 1, mid + 1, end, l, r, k);
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("세그먼트 트리 예제\n");
    printf("============================================================\n");

    /* 1. 기본 세그먼트 트리 */
    printf("\n[1] 기본 세그먼트 트리 (구간 합)\n");
    int arr1[] = {1, 3, 5, 7, 9, 11};
    int n1 = 6;
    SegmentTree* st = st_create(n1);
    st_build(st, arr1, 1, 0, n1 - 1);

    printf("    배열: [1, 3, 5, 7, 9, 11]\n");
    printf("    구간 합 [1, 3]: %lld\n", st_query(st, 1, 0, n1 - 1, 1, 3));
    printf("    구간 합 [0, 5]: %lld\n", st_query(st, 1, 0, n1 - 1, 0, 5));

    st_update(st, 1, 0, n1 - 1, 2, 10);  /* arr[2] = 10 */
    printf("    arr[2] = 10 후 구간 합 [1, 3]: %lld\n", st_query(st, 1, 0, n1 - 1, 1, 3));
    st_free(st);

    /* 2. 구간 최소/최대 */
    printf("\n[2] 구간 최소/최대 세그먼트 트리\n");
    int arr2[] = {5, 2, 8, 1, 9, 3, 7, 4};
    int n2 = 8;
    MinMaxTree* mmt = mmt_create(n2);
    mmt_build(mmt, arr2, 1, 0, n2 - 1);

    printf("    배열: [5, 2, 8, 1, 9, 3, 7, 4]\n");
    printf("    구간 [0, 3] 최소: %d\n", mmt_query_min(mmt, 1, 0, n2 - 1, 0, 3));
    printf("    구간 [0, 3] 최대: %d\n", mmt_query_max(mmt, 1, 0, n2 - 1, 0, 3));
    printf("    구간 [4, 7] 최소: %d\n", mmt_query_min(mmt, 1, 0, n2 - 1, 4, 7));
    mmt_free(mmt);

    /* 3. Lazy Propagation */
    printf("\n[3] Lazy Propagation\n");
    int arr3[] = {1, 2, 3, 4, 5};
    int n3 = 5;
    LazySegTree* lst = lst_create(n3);
    lst_build(lst, arr3, 1, 0, n3 - 1);

    printf("    배열: [1, 2, 3, 4, 5]\n");
    printf("    초기 구간 합 [0, 4]: %lld\n", lst_query(lst, 1, 0, n3 - 1, 0, 4));

    lst_update_range(lst, 1, 0, n3 - 1, 1, 3, 10);  /* [1, 3]에 10 더하기 */
    printf("    [1, 3]에 10 더한 후 구간 합 [0, 4]: %lld\n", lst_query(lst, 1, 0, n3 - 1, 0, 4));
    printf("    구간 합 [1, 3]: %lld\n", lst_query(lst, 1, 0, n3 - 1, 1, 3));
    lst_free(lst);

    /* 4. 동적 세그먼트 트리 */
    printf("\n[4] 동적 세그먼트 트리\n");
    DynamicNode* root = dn_create();
    long long max_range = 1000000000LL;

    dn_update(root, 0, max_range, 100, 5);
    dn_update(root, 0, max_range, 1000000, 3);
    dn_update(root, 0, max_range, 500, 7);

    printf("    인덱스 100에 5, 1000000에 3, 500에 7 추가\n");
    printf("    구간 [0, 1000] 합: %lld\n", dn_query(root, 0, max_range, 0, 1000));
    printf("    구간 [0, 1000000] 합: %lld\n", dn_query(root, 0, max_range, 0, 1000000));
    dn_free(root);

    /* 5. 복잡도 */
    printf("\n[5] 복잡도 분석\n");
    printf("    | 연산           | 시간복잡도 | 공간복잡도 |\n");
    printf("    |----------------|------------|------------|\n");
    printf("    | 빌드           | O(n)       | O(n)       |\n");
    printf("    | 점 업데이트    | O(log n)   | -          |\n");
    printf("    | 구간 쿼리      | O(log n)   | -          |\n");
    printf("    | 구간 업데이트  | O(log n)   | O(n)       |\n");
    printf("    | 동적 트리      | O(log M)   | O(Q log M) |\n");

    printf("\n[6] 응용\n");
    printf("    - 구간 합/최소/최대 쿼리\n");
    printf("    - 구간 GCD 쿼리\n");
    printf("    - 역순 쌍 개수 (Inversion Count)\n");
    printf("    - K번째 원소 찾기\n");
    printf("    - 2D 세그먼트 트리 (평면 쿼리)\n");

    printf("\n============================================================\n");

    return 0;
}
