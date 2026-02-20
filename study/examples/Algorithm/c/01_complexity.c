/*
 * 시간 복잡도 (Time Complexity)
 * Big O Notation and Complexity Analysis
 *
 * 알고리즘의 효율성을 분석하는 방법입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* =============================================================================
 * 1. O(1) - 상수 시간
 * ============================================================================= */

int constant_time(int arr[], int n) {
    /* 배열 크기와 무관하게 항상 일정한 시간 */
    return arr[0];
}

/* =============================================================================
 * 2. O(log n) - 로그 시간
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

/* =============================================================================
 * 3. O(n) - 선형 시간
 * ============================================================================= */

int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target)
            return i;
    }
    return -1;
}

int sum_array(int arr[], int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

/* =============================================================================
 * 4. O(n log n) - 선형 로그 시간
 * ============================================================================= */

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];

    free(L);
    free(R);
}

void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

/* =============================================================================
 * 5. O(n²) - 이차 시간
 * ============================================================================= */

void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int count_pairs(int arr[], int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            count++;
        }
    }
    return count;  /* n*(n-1)/2 */
}

/* =============================================================================
 * 6. O(2^n) - 지수 시간
 * ============================================================================= */

int fibonacci_recursive(int n) {
    if (n <= 1)
        return n;
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2);
}

/* O(n) 최적화 버전 */
int fibonacci_iterative(int n) {
    if (n <= 1)
        return n;

    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

/* =============================================================================
 * 7. 시간 측정 유틸리티
 * ============================================================================= */

double measure_time(void (*func)(int*, int), int arr[], int n) {
    clock_t start = clock();
    func(arr, n);
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

/* =============================================================================
 * 8. 공간 복잡도 예제
 * ============================================================================= */

/* O(1) 공간 */
void reverse_in_place(int arr[], int n) {
    for (int i = 0; i < n / 2; i++) {
        int temp = arr[i];
        arr[i] = arr[n - 1 - i];
        arr[n - 1 - i] = temp;
    }
}

/* O(n) 공간 */
int* reverse_with_copy(int arr[], int n) {
    int *result = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        result[i] = arr[n - 1 - i];
    }
    return result;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

void print_array(int arr[], int n) {
    printf("    [");
    for (int i = 0; i < n && i < 10; i++) {
        printf("%d", arr[i]);
        if (i < n - 1 && i < 9) printf(", ");
    }
    if (n > 10) printf(", ...");
    printf("]\n");
}

int main(void) {
    printf("============================================================\n");
    printf("시간 복잡도 (Time Complexity) 예제\n");
    printf("============================================================\n");

    /* 1. O(1) */
    printf("\n[1] O(1) - 상수 시간\n");
    int arr1[] = {5, 2, 8, 1, 9};
    printf("    첫 번째 원소: %d\n", constant_time(arr1, 5));

    /* 2. O(log n) */
    printf("\n[2] O(log n) - 이분 탐색\n");
    int arr2[] = {1, 3, 5, 7, 9, 11, 13, 15};
    int idx = binary_search(arr2, 8, 7);
    printf("    배열: [1,3,5,7,9,11,13,15]\n");
    printf("    7의 위치: %d\n", idx);

    /* 3. O(n) */
    printf("\n[3] O(n) - 선형 탐색\n");
    int arr3[] = {4, 2, 7, 1, 9, 3};
    printf("    배열 합: %d\n", sum_array(arr3, 6));

    /* 4. O(n log n) */
    printf("\n[4] O(n log n) - 병합 정렬\n");
    int arr4[] = {64, 34, 25, 12, 22, 11, 90};
    printf("    정렬 전: ");
    print_array(arr4, 7);
    merge_sort(arr4, 0, 6);
    printf("    정렬 후: ");
    print_array(arr4, 7);

    /* 5. O(n²) */
    printf("\n[5] O(n²) - 버블 정렬\n");
    int arr5[] = {64, 34, 25, 12, 22, 11, 90};
    bubble_sort(arr5, 7);
    printf("    정렬 후: ");
    print_array(arr5, 7);
    printf("    5개 원소의 쌍 개수: %d\n", count_pairs(arr5, 5));

    /* 6. O(2^n) vs O(n) */
    printf("\n[6] O(2^n) vs O(n) - 피보나치\n");
    printf("    피보나치(20) 재귀: %d\n", fibonacci_recursive(20));
    printf("    피보나치(20) 반복: %d\n", fibonacci_iterative(20));
    printf("    피보나치(40) 반복: %d\n", fibonacci_iterative(40));

    /* 7. 공간 복잡도 */
    printf("\n[7] 공간 복잡도\n");
    int arr7[] = {1, 2, 3, 4, 5};
    printf("    원본: ");
    print_array(arr7, 5);
    reverse_in_place(arr7, 5);
    printf("    O(1) 공간 뒤집기: ");
    print_array(arr7, 5);

    /* 8. 복잡도 요약 */
    printf("\n[8] 복잡도 요약\n");
    printf("    | 복잡도    | 1000개 연산 | 예시              |\n");
    printf("    |-----------|-------------|-------------------|\n");
    printf("    | O(1)      | 1           | 배열 인덱싱       |\n");
    printf("    | O(log n)  | 10          | 이분 탐색         |\n");
    printf("    | O(n)      | 1000        | 선형 탐색         |\n");
    printf("    | O(n log n)| 10000       | 병합 정렬         |\n");
    printf("    | O(n²)     | 1000000     | 버블 정렬         |\n");
    printf("    | O(2^n)    | 매우 큼     | 모든 부분집합     |\n");

    printf("\n============================================================\n");

    return 0;
}
