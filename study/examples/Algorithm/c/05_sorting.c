/*
 * 정렬 알고리즘 (Sorting Algorithms)
 * Bubble, Selection, Insertion, Merge, Quick, Heap, Counting, Radix
 *
 * 다양한 정렬 알고리즘의 구현과 비교입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =============================================================================
 * 1. 버블 정렬 - O(n²)
 * ============================================================================= */

void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = 1;
            }
        }
        if (!swapped) break;  /* 최적화: 이미 정렬됨 */
    }
}

/* =============================================================================
 * 2. 선택 정렬 - O(n²)
 * ============================================================================= */

void selection_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            int temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
        }
    }
}

/* =============================================================================
 * 3. 삽입 정렬 - O(n²)
 * ============================================================================= */

void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/* =============================================================================
 * 4. 병합 정렬 - O(n log n)
 * ============================================================================= */

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = malloc(n1 * sizeof(int));
    int* R = malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
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
 * 5. 퀵 정렬 - O(n log n) 평균
 * ============================================================================= */

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1;
}

void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

/* 3중 피벗 퀵소트 */
int partition_median(int arr[], int low, int high) {
    int mid = low + (high - low) / 2;

    /* 3개 값 정렬 */
    if (arr[low] > arr[mid]) {
        int t = arr[low]; arr[low] = arr[mid]; arr[mid] = t;
    }
    if (arr[mid] > arr[high]) {
        int t = arr[mid]; arr[mid] = arr[high]; arr[high] = t;
    }
    if (arr[low] > arr[mid]) {
        int t = arr[low]; arr[low] = arr[mid]; arr[mid] = t;
    }

    /* 중앙값을 high-1 위치로 */
    int t = arr[mid]; arr[mid] = arr[high - 1]; arr[high - 1] = t;

    return partition(arr, low, high);
}

/* =============================================================================
 * 6. 힙 정렬 - O(n log n)
 * ============================================================================= */

void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
        heapify(arr, n, largest);
    }
}

void heap_sort(int arr[], int n) {
    /* 최대 힙 구성 */
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    /* 정렬 */
    for (int i = n - 1; i > 0; i--) {
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        heapify(arr, i, 0);
    }
}

/* =============================================================================
 * 7. 계수 정렬 - O(n + k)
 * ============================================================================= */

void counting_sort(int arr[], int n, int max_val) {
    int* count = calloc(max_val + 1, sizeof(int));
    int* output = malloc(n * sizeof(int));

    /* 빈도 계산 */
    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }

    /* 누적 합 */
    for (int i = 1; i <= max_val; i++) {
        count[i] += count[i - 1];
    }

    /* 출력 배열 생성 (안정 정렬) */
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    /* 결과 복사 */
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(count);
    free(output);
}

/* =============================================================================
 * 8. 기수 정렬 - O(d * (n + k))
 * ============================================================================= */

int get_max(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

void counting_sort_exp(int arr[], int n, int exp) {
    int* output = malloc(n * sizeof(int));
    int count[10] = {0};

    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(output);
}

void radix_sort(int arr[], int n) {
    int max = get_max(arr, n);

    for (int exp = 1; max / exp > 0; exp *= 10) {
        counting_sort_exp(arr, n, exp);
    }
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
    printf("]\n");
}

void copy_array(int src[], int dst[], int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

int main(void) {
    printf("============================================================\n");
    printf("정렬 알고리즘 (Sorting Algorithms) 예제\n");
    printf("============================================================\n");

    int original[] = {64, 34, 25, 12, 22, 11, 90, 45};
    int n = 8;
    int arr[8];

    /* 1. 버블 정렬 */
    printf("\n[1] 버블 정렬 - O(n²)\n");
    copy_array(original, arr, n);
    printf("    정렬 전: ");
    print_array(arr, n);
    bubble_sort(arr, n);
    printf("    정렬 후: ");
    print_array(arr, n);

    /* 2. 선택 정렬 */
    printf("\n[2] 선택 정렬 - O(n²)\n");
    copy_array(original, arr, n);
    selection_sort(arr, n);
    printf("    결과: ");
    print_array(arr, n);

    /* 3. 삽입 정렬 */
    printf("\n[3] 삽입 정렬 - O(n²)\n");
    copy_array(original, arr, n);
    insertion_sort(arr, n);
    printf("    결과: ");
    print_array(arr, n);

    /* 4. 병합 정렬 */
    printf("\n[4] 병합 정렬 - O(n log n)\n");
    copy_array(original, arr, n);
    merge_sort(arr, 0, n - 1);
    printf("    결과: ");
    print_array(arr, n);

    /* 5. 퀵 정렬 */
    printf("\n[5] 퀵 정렬 - O(n log n) 평균\n");
    copy_array(original, arr, n);
    quick_sort(arr, 0, n - 1);
    printf("    결과: ");
    print_array(arr, n);

    /* 6. 힙 정렬 */
    printf("\n[6] 힙 정렬 - O(n log n)\n");
    copy_array(original, arr, n);
    heap_sort(arr, n);
    printf("    결과: ");
    print_array(arr, n);

    /* 7. 계수 정렬 */
    printf("\n[7] 계수 정렬 - O(n + k)\n");
    int arr7[] = {4, 2, 2, 8, 3, 3, 1};
    printf("    정렬 전: ");
    print_array(arr7, 7);
    counting_sort(arr7, 7, 8);
    printf("    정렬 후: ");
    print_array(arr7, 7);

    /* 8. 기수 정렬 */
    printf("\n[8] 기수 정렬 - O(d * n)\n");
    int arr8[] = {170, 45, 75, 90, 802, 24, 2, 66};
    printf("    정렬 전: ");
    print_array(arr8, 8);
    radix_sort(arr8, 8);
    printf("    정렬 후: ");
    print_array(arr8, 8);

    /* 9. 알고리즘 비교 */
    printf("\n[9] 정렬 알고리즘 비교\n");
    printf("    | 알고리즘 | 최선     | 평균     | 최악     | 공간  | 안정 |\n");
    printf("    |----------|----------|----------|----------|-------|------|\n");
    printf("    | 버블     | O(n)     | O(n²)    | O(n²)    | O(1)  | Yes  |\n");
    printf("    | 선택     | O(n²)    | O(n²)    | O(n²)    | O(1)  | No   |\n");
    printf("    | 삽입     | O(n)     | O(n²)    | O(n²)    | O(1)  | Yes  |\n");
    printf("    | 병합     | O(nlogn) | O(nlogn) | O(nlogn) | O(n)  | Yes  |\n");
    printf("    | 퀵       | O(nlogn) | O(nlogn) | O(n²)    | O(logn)| No  |\n");
    printf("    | 힙       | O(nlogn) | O(nlogn) | O(nlogn) | O(1)  | No   |\n");
    printf("    | 계수     | O(n+k)   | O(n+k)   | O(n+k)   | O(k)  | Yes  |\n");
    printf("    | 기수     | O(dn)    | O(dn)    | O(dn)    | O(n+k)| Yes  |\n");

    printf("\n============================================================\n");

    return 0;
}
