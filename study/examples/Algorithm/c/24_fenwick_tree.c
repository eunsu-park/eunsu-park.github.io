/*
 * 펜윅 트리 (Fenwick Tree / Binary Indexed Tree)
 * 구간 합, 역순 쌍, 2D BIT
 *
 * 세그먼트 트리보다 간단하고 메모리 효율적입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_N 100001

/* =============================================================================
 * 1. 기본 펜윅 트리 (1-indexed)
 * ============================================================================= */

typedef struct {
    long long* tree;
    int n;
} BIT;

BIT* bit_create(int n) {
    BIT* bit = malloc(sizeof(BIT));
    bit->n = n;
    bit->tree = calloc(n + 1, sizeof(long long));
    return bit;
}

void bit_free(BIT* bit) {
    free(bit->tree);
    free(bit);
}

/* i번째 원소에 delta 더하기 (1-indexed) */
void bit_update(BIT* bit, int i, long long delta) {
    for (; i <= bit->n; i += i & (-i)) {
        bit->tree[i] += delta;
    }
}

/* [1, i] 구간 합 (1-indexed) */
long long bit_query(BIT* bit, int i) {
    long long sum = 0;
    for (; i > 0; i -= i & (-i)) {
        sum += bit->tree[i];
    }
    return sum;
}

/* [l, r] 구간 합 (1-indexed) */
long long bit_range_query(BIT* bit, int l, int r) {
    return bit_query(bit, r) - bit_query(bit, l - 1);
}

/* 배열로 초기화 */
void bit_build(BIT* bit, int arr[], int n) {
    for (int i = 1; i <= n; i++) {
        bit_update(bit, i, arr[i - 1]);
    }
}

/* =============================================================================
 * 2. 구간 업데이트, 점 쿼리 (Difference Array)
 * ============================================================================= */

typedef struct {
    long long* tree;
    int n;
} BITDiff;

BITDiff* bitd_create(int n) {
    BITDiff* bitd = malloc(sizeof(BITDiff));
    bitd->n = n;
    bitd->tree = calloc(n + 2, sizeof(long long));
    return bitd;
}

void bitd_free(BITDiff* bitd) {
    free(bitd->tree);
    free(bitd);
}

void bitd_update_internal(BITDiff* bitd, int i, long long delta) {
    for (; i <= bitd->n; i += i & (-i)) {
        bitd->tree[i] += delta;
    }
}

/* [l, r] 구간에 delta 더하기 */
void bitd_range_update(BITDiff* bitd, int l, int r, long long delta) {
    bitd_update_internal(bitd, l, delta);
    bitd_update_internal(bitd, r + 1, -delta);
}

/* i번째 원소 값 조회 */
long long bitd_point_query(BITDiff* bitd, int i) {
    long long sum = 0;
    for (; i > 0; i -= i & (-i)) {
        sum += bitd->tree[i];
    }
    return sum;
}

/* =============================================================================
 * 3. 구간 업데이트, 구간 쿼리
 * ============================================================================= */

typedef struct {
    long long* tree1;
    long long* tree2;
    int n;
} BITRange;

BITRange* bitr_create(int n) {
    BITRange* bitr = malloc(sizeof(BITRange));
    bitr->n = n;
    bitr->tree1 = calloc(n + 2, sizeof(long long));
    bitr->tree2 = calloc(n + 2, sizeof(long long));
    return bitr;
}

void bitr_free(BITRange* bitr) {
    free(bitr->tree1);
    free(bitr->tree2);
    free(bitr);
}

void bitr_update_internal(long long* tree, int n, int i, long long delta) {
    for (; i <= n; i += i & (-i)) {
        tree[i] += delta;
    }
}

long long bitr_query_internal(long long* tree, int i) {
    long long sum = 0;
    for (; i > 0; i -= i & (-i)) {
        sum += tree[i];
    }
    return sum;
}

/* [l, r] 구간에 delta 더하기 */
void bitr_range_update(BITRange* bitr, int l, int r, long long delta) {
    bitr_update_internal(bitr->tree1, bitr->n, l, delta);
    bitr_update_internal(bitr->tree1, bitr->n, r + 1, -delta);
    bitr_update_internal(bitr->tree2, bitr->n, l, delta * (l - 1));
    bitr_update_internal(bitr->tree2, bitr->n, r + 1, -delta * r);
}

/* [1, i] 구간 합 */
long long bitr_prefix_sum(BITRange* bitr, int i) {
    return bitr_query_internal(bitr->tree1, i) * i -
           bitr_query_internal(bitr->tree2, i);
}

/* [l, r] 구간 합 */
long long bitr_range_query(BITRange* bitr, int l, int r) {
    return bitr_prefix_sum(bitr, r) - bitr_prefix_sum(bitr, l - 1);
}

/* =============================================================================
 * 4. 2D 펜윅 트리
 * ============================================================================= */

typedef struct {
    long long** tree;
    int rows;
    int cols;
} BIT2D;

BIT2D* bit2d_create(int rows, int cols) {
    BIT2D* bit = malloc(sizeof(BIT2D));
    bit->rows = rows;
    bit->cols = cols;
    bit->tree = malloc((rows + 1) * sizeof(long long*));
    for (int i = 0; i <= rows; i++) {
        bit->tree[i] = calloc(cols + 1, sizeof(long long));
    }
    return bit;
}

void bit2d_free(BIT2D* bit) {
    for (int i = 0; i <= bit->rows; i++) {
        free(bit->tree[i]);
    }
    free(bit->tree);
    free(bit);
}

/* (x, y)에 delta 더하기 */
void bit2d_update(BIT2D* bit, int x, int y, long long delta) {
    for (int i = x; i <= bit->rows; i += i & (-i)) {
        for (int j = y; j <= bit->cols; j += j & (-j)) {
            bit->tree[i][j] += delta;
        }
    }
}

/* [(1,1), (x,y)] 직사각형 합 */
long long bit2d_query(BIT2D* bit, int x, int y) {
    long long sum = 0;
    for (int i = x; i > 0; i -= i & (-i)) {
        for (int j = y; j > 0; j -= j & (-j)) {
            sum += bit->tree[i][j];
        }
    }
    return sum;
}

/* [(x1,y1), (x2,y2)] 직사각형 합 */
long long bit2d_range_query(BIT2D* bit, int x1, int y1, int x2, int y2) {
    return bit2d_query(bit, x2, y2) -
           bit2d_query(bit, x1 - 1, y2) -
           bit2d_query(bit, x2, y1 - 1) +
           bit2d_query(bit, x1 - 1, y1 - 1);
}

/* =============================================================================
 * 5. 역순 쌍 개수 (Inversion Count)
 * ============================================================================= */

long long count_inversions(int arr[], int n) {
    /* 좌표 압축 */
    int* sorted = malloc(n * sizeof(int));
    memcpy(sorted, arr, n * sizeof(int));

    /* 정렬 */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (sorted[i] > sorted[j]) {
                int temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }

    /* 중복 제거 및 압축 */
    int* compressed = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (sorted[mid] < arr[i]) lo = mid + 1;
            else hi = mid;
        }
        compressed[i] = lo + 1;  /* 1-indexed */
    }

    /* 역순 쌍 카운트 */
    BIT* bit = bit_create(n);
    long long inversions = 0;

    for (int i = n - 1; i >= 0; i--) {
        inversions += bit_query(bit, compressed[i] - 1);
        bit_update(bit, compressed[i], 1);
    }

    bit_free(bit);
    free(sorted);
    free(compressed);
    return inversions;
}

/* =============================================================================
 * 6. K번째 원소 찾기
 * ============================================================================= */

/* 누적합이 k 이상이 되는 최소 인덱스 */
int bit_find_kth(BIT* bit, long long k) {
    int pos = 0;
    int log_n = 0;
    while ((1 << (log_n + 1)) <= bit->n) log_n++;

    for (int i = log_n; i >= 0; i--) {
        int next = pos + (1 << i);
        if (next <= bit->n && bit->tree[next] < k) {
            pos = next;
            k -= bit->tree[pos];
        }
    }

    return pos + 1;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("펜윅 트리 (BIT) 예제\n");
    printf("============================================================\n");

    /* 1. 기본 BIT */
    printf("\n[1] 기본 펜윅 트리\n");
    int arr1[] = {1, 3, 5, 7, 9, 11};
    int n1 = 6;
    BIT* bit = bit_create(n1);
    bit_build(bit, arr1, n1);

    printf("    배열: [1, 3, 5, 7, 9, 11]\n");
    printf("    구간 합 [1, 3]: %lld\n", bit_range_query(bit, 1, 3));
    printf("    구간 합 [1, 6]: %lld\n", bit_range_query(bit, 1, 6));
    printf("    구간 합 [3, 5]: %lld\n", bit_range_query(bit, 3, 5));

    bit_update(bit, 3, 5);  /* arr[3]에 5 더하기 */
    printf("    arr[3] += 5 후 구간 합 [1, 6]: %lld\n", bit_range_query(bit, 1, 6));
    bit_free(bit);

    /* 2. 구간 업데이트, 점 쿼리 */
    printf("\n[2] 구간 업데이트, 점 쿼리\n");
    BITDiff* bitd = bitd_create(5);

    bitd_range_update(bitd, 1, 3, 10);  /* [1, 3]에 10 더하기 */
    bitd_range_update(bitd, 2, 4, 5);   /* [2, 4]에 5 더하기 */

    printf("    [1, 3]에 10, [2, 4]에 5 더한 후:\n");
    printf("    ");
    for (int i = 1; i <= 5; i++) {
        printf("arr[%d]=%lld ", i, bitd_point_query(bitd, i));
    }
    printf("\n");
    bitd_free(bitd);

    /* 3. 구간 업데이트, 구간 쿼리 */
    printf("\n[3] 구간 업데이트, 구간 쿼리\n");
    BITRange* bitr = bitr_create(5);

    bitr_range_update(bitr, 1, 3, 10);
    bitr_range_update(bitr, 2, 5, 5);

    printf("    [1, 3]에 10, [2, 5]에 5 더한 후:\n");
    printf("    구간 합 [1, 5]: %lld\n", bitr_range_query(bitr, 1, 5));
    printf("    구간 합 [2, 4]: %lld\n", bitr_range_query(bitr, 2, 4));
    bitr_free(bitr);

    /* 4. 2D BIT */
    printf("\n[4] 2D 펜윅 트리\n");
    BIT2D* bit2d = bit2d_create(4, 4);

    bit2d_update(bit2d, 1, 1, 1);
    bit2d_update(bit2d, 2, 2, 2);
    bit2d_update(bit2d, 3, 3, 3);
    bit2d_update(bit2d, 2, 3, 4);

    printf("    (1,1)=1, (2,2)=2, (3,3)=3, (2,3)=4 설정\n");
    printf("    [(1,1), (3,3)] 합: %lld\n", bit2d_range_query(bit2d, 1, 1, 3, 3));
    printf("    [(2,2), (3,3)] 합: %lld\n", bit2d_range_query(bit2d, 2, 2, 3, 3));
    bit2d_free(bit2d);

    /* 5. 역순 쌍 개수 */
    printf("\n[5] 역순 쌍 개수\n");
    int arr2[] = {8, 4, 2, 1};
    printf("    배열: [8, 4, 2, 1]\n");
    printf("    역순 쌍 개수: %lld\n", count_inversions(arr2, 4));

    int arr3[] = {1, 3, 2, 3, 1};
    printf("    배열: [1, 3, 2, 3, 1]\n");
    printf("    역순 쌍 개수: %lld\n", count_inversions(arr3, 5));

    /* 6. K번째 원소 */
    printf("\n[6] K번째 원소 찾기\n");
    BIT* bit_kth = bit_create(10);
    bit_update(bit_kth, 2, 1);  /* 2 추가 */
    bit_update(bit_kth, 5, 1);  /* 5 추가 */
    bit_update(bit_kth, 3, 1);  /* 3 추가 */
    bit_update(bit_kth, 7, 1);  /* 7 추가 */

    printf("    집합: {2, 3, 5, 7}\n");
    printf("    1번째 원소: %d\n", bit_find_kth(bit_kth, 1));
    printf("    2번째 원소: %d\n", bit_find_kth(bit_kth, 2));
    printf("    3번째 원소: %d\n", bit_find_kth(bit_kth, 3));
    printf("    4번째 원소: %d\n", bit_find_kth(bit_kth, 4));
    bit_free(bit_kth);

    /* 7. 복잡도 */
    printf("\n[7] 복잡도 비교 (BIT vs 세그먼트 트리)\n");
    printf("    | 연산           | BIT        | 세그먼트 트리 |\n");
    printf("    |----------------|------------|---------------|\n");
    printf("    | 점 업데이트    | O(log n)   | O(log n)      |\n");
    printf("    | 구간 합 쿼리   | O(log n)   | O(log n)      |\n");
    printf("    | 구간 업데이트  | O(log n)*  | O(log n)      |\n");
    printf("    | 공간 복잡도    | O(n)       | O(4n)         |\n");
    printf("    | 구현 난이도    | 쉬움       | 중간          |\n");
    printf("    * 구간 업데이트는 추가 배열 필요\n");

    printf("\n============================================================\n");

    return 0;
}
