/*
 * 탐색 알고리즘 (Searching Algorithms)
 * Linear Search, Binary Search, Parametric Search
 *
 * 다양한 탐색 기법과 이분 탐색 응용입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* =============================================================================
 * 1. 선형 탐색 - O(n)
 * ============================================================================= */

int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target)
            return i;
    }
    return -1;
}

/* =============================================================================
 * 2. 이분 탐색 - O(log n)
 * ============================================================================= */

int binary_search(int arr[], int n, int target) {
    int left = 0, right = n - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }

    return -1;
}

/* 재귀 버전 */
int binary_search_recursive(int arr[], int left, int right, int target) {
    if (left > right)
        return -1;

    int mid = left + (right - left) / 2;

    if (arr[mid] == target)
        return mid;
    else if (arr[mid] < target)
        return binary_search_recursive(arr, mid + 1, right, target);
    else
        return binary_search_recursive(arr, left, mid - 1, target);
}

/* =============================================================================
 * 3. Lower Bound / Upper Bound
 * ============================================================================= */

/* target 이상인 첫 번째 위치 */
int lower_bound(int arr[], int n, int target) {
    int left = 0, right = n;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

/* target 초과인 첫 번째 위치 */
int upper_bound(int arr[], int n, int target) {
    int left = 0, right = n;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

/* target의 개수 */
int count_occurrences(int arr[], int n, int target) {
    return upper_bound(arr, n, target) - lower_bound(arr, n, target);
}

/* =============================================================================
 * 4. 삽입 위치 찾기
 * ============================================================================= */

int search_insert(int arr[], int n, int target) {
    int left = 0, right = n;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

/* =============================================================================
 * 5. 회전 정렬 배열 탐색
 * ============================================================================= */

int search_rotated(int arr[], int n, int target) {
    int left = 0, right = n - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;

        /* 왼쪽이 정렬됨 */
        if (arr[left] <= arr[mid]) {
            if (arr[left] <= target && target < arr[mid])
                right = mid - 1;
            else
                left = mid + 1;
        }
        /* 오른쪽이 정렬됨 */
        else {
            if (arr[mid] < target && target <= arr[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }

    return -1;
}

/* 회전 배열에서 최솟값 */
int find_min_rotated(int arr[], int n) {
    int left = 0, right = n - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] > arr[right])
            left = mid + 1;
        else
            right = mid;
    }

    return arr[left];
}

/* =============================================================================
 * 6. 파라메트릭 서치
 * ============================================================================= */

/* 배열을 k개로 나눌 때 최대 합의 최솟값 */
bool can_split(int arr[], int n, int max_sum, int k) {
    int count = 1;
    int current_sum = 0;

    for (int i = 0; i < n; i++) {
        if (arr[i] > max_sum)
            return false;

        if (current_sum + arr[i] > max_sum) {
            count++;
            current_sum = arr[i];
        } else {
            current_sum += arr[i];
        }
    }

    return count <= k;
}

int split_array_min_max(int arr[], int n, int k) {
    int left = 0, right = 0;

    for (int i = 0; i < n; i++) {
        if (arr[i] > left) left = arr[i];
        right += arr[i];
    }

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (can_split(arr, n, mid, k))
            right = mid;
        else
            left = mid + 1;
    }

    return left;
}

/* 나무 자르기: 높이 H로 잘랐을 때 M 이상의 나무를 얻을 수 있는 최대 H */
long long cut_trees(int trees[], int n, long long target) {
    long long left = 0, right = 0;

    for (int i = 0; i < n; i++) {
        if (trees[i] > right)
            right = trees[i];
    }

    while (left < right) {
        long long mid = left + (right - left + 1) / 2;
        long long total = 0;

        for (int i = 0; i < n; i++) {
            if (trees[i] > mid)
                total += trees[i] - mid;
        }

        if (total >= target)
            left = mid;
        else
            right = mid - 1;
    }

    return left;
}

/* =============================================================================
 * 7. 실수 이분 탐색
 * ============================================================================= */

double sqrt_binary_search(double x) {
    if (x < 0) return -1;
    if (x < 1) {
        double lo = x, hi = 1.0;
        while (hi - lo > 1e-9) {
            double mid = (lo + hi) / 2;
            if (mid * mid < x)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }

    double lo = 1.0, hi = x;
    while (hi - lo > 1e-9) {
        double mid = (lo + hi) / 2;
        if (mid * mid < x)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

/* =============================================================================
 * 8. 피크 찾기
 * ============================================================================= */

int find_peak(int arr[], int n) {
    int left = 0, right = n - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] < arr[mid + 1])
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]");
}

int main(void) {
    printf("============================================================\n");
    printf("탐색 알고리즘 (Searching Algorithms) 예제\n");
    printf("============================================================\n");

    /* 1. 이분 탐색 */
    printf("\n[1] 이분 탐색\n");
    int arr1[] = {1, 3, 5, 7, 9, 11, 13, 15};
    printf("    배열: ");
    print_array(arr1, 8);
    printf("\n");
    printf("    7의 위치: %d\n", binary_search(arr1, 8, 7));
    printf("    6의 위치: %d\n", binary_search(arr1, 8, 6));

    /* 2. Lower/Upper Bound */
    printf("\n[2] Lower/Upper Bound\n");
    int arr2[] = {1, 2, 2, 2, 3, 4, 4, 5};
    printf("    배열: ");
    print_array(arr2, 8);
    printf("\n");
    printf("    lower_bound(2): %d\n", lower_bound(arr2, 8, 2));
    printf("    upper_bound(2): %d\n", upper_bound(arr2, 8, 2));
    printf("    2의 개수: %d\n", count_occurrences(arr2, 8, 2));

    /* 3. 삽입 위치 */
    printf("\n[3] 삽입 위치\n");
    int arr3[] = {1, 3, 5, 7};
    printf("    배열: [1,3,5,7]\n");
    printf("    4의 삽입 위치: %d\n", search_insert(arr3, 4, 4));
    printf("    6의 삽입 위치: %d\n", search_insert(arr3, 4, 6));

    /* 4. 회전 배열 */
    printf("\n[4] 회전 배열 탐색\n");
    int arr4[] = {4, 5, 6, 7, 0, 1, 2};
    printf("    배열: ");
    print_array(arr4, 7);
    printf("\n");
    printf("    0의 위치: %d\n", search_rotated(arr4, 7, 0));
    printf("    최솟값: %d\n", find_min_rotated(arr4, 7));

    /* 5. 파라메트릭 서치 - 배열 분할 */
    printf("\n[5] 파라메트릭 서치 - 배열 분할\n");
    int arr5[] = {7, 2, 5, 10, 8};
    printf("    배열: [7,2,5,10,8], k=2\n");
    printf("    최대 합의 최솟값: %d\n", split_array_min_max(arr5, 5, 2));

    /* 6. 나무 자르기 */
    printf("\n[6] 나무 자르기\n");
    int trees[] = {20, 15, 10, 17};
    long long target = 7;
    printf("    나무 높이: [20,15,10,17], 필요량: %lld\n", target);
    printf("    최대 절단 높이: %lld\n", cut_trees(trees, 4, target));

    /* 7. 실수 이분 탐색 */
    printf("\n[7] 제곱근 이분 탐색\n");
    printf("    sqrt(2): %.6f\n", sqrt_binary_search(2));
    printf("    sqrt(10): %.6f\n", sqrt_binary_search(10));

    /* 8. 피크 찾기 */
    printf("\n[8] 피크 찾기\n");
    int arr8[] = {1, 2, 1, 3, 5, 6, 4};
    printf("    배열: [1,2,1,3,5,6,4]\n");
    int peak_idx = find_peak(arr8, 7);
    printf("    피크 인덱스: %d (값: %d)\n", peak_idx, arr8[peak_idx]);

    /* 9. 알고리즘 요약 */
    printf("\n[9] 이분 탐색 응용 정리\n");
    printf("    | 문제 유형        | 핵심 아이디어             |\n");
    printf("    |------------------|---------------------------|\n");
    printf("    | lower_bound      | arr[mid] < target         |\n");
    printf("    | upper_bound      | arr[mid] <= target        |\n");
    printf("    | 회전 배열        | 정렬된 절반 찾기          |\n");
    printf("    | 파라메트릭       | 결정 문제로 변환          |\n");
    printf("    | 최대의 최소      | 가능한 최소값 이분탐색    |\n");

    printf("\n============================================================\n");

    return 0;
}
