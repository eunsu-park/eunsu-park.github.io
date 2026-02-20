// parallel_sort.c
// 병렬 병합 정렬 (Parallel Merge Sort)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define THRESHOLD 10000  // 이보다 작으면 단일 스레드

typedef struct {
    int* arr;
    int left;
    int right;
} SortTask;

// 병합
void merge(int* arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = malloc(n1 * sizeof(int));
    int* R = malloc(n2 * sizeof(int));

    memcpy(L, arr + left, n1 * sizeof(int));
    memcpy(R, arr + mid + 1, n2 * sizeof(int));

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// 단일 스레드 병합 정렬
void merge_sort_single(int* arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort_single(arr, left, mid);
        merge_sort_single(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// 멀티스레드 병합 정렬
void* merge_sort_parallel(void* arg) {
    SortTask* task = (SortTask*)arg;
    int* arr = task->arr;
    int left = task->left;
    int right = task->right;

    if (left >= right) return NULL;

    // 작은 배열은 단일 스레드로
    if (right - left < THRESHOLD) {
        merge_sort_single(arr, left, right);
        return NULL;
    }

    int mid = left + (right - left) / 2;

    // 왼쪽 절반: 새 스레드
    SortTask left_task = { arr, left, mid };
    pthread_t left_thread;
    pthread_create(&left_thread, NULL, merge_sort_parallel, &left_task);

    // 오른쪽 절반: 현재 스레드
    SortTask right_task = { arr, mid + 1, right };
    merge_sort_parallel(&right_task);

    // 왼쪽 스레드 대기
    pthread_join(left_thread, NULL);

    // 병합
    merge(arr, left, mid, right);

    return NULL;
}

// 배열 출력
void print_array(int* arr, int n) {
    for (int i = 0; i < n && i < 20; i++) {
        printf("%d ", arr[i]);
    }
    if (n > 20) printf("...");
    printf("\n");
}

// 배열 검증
int is_sorted(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) return 0;
    }
    return 1;
}

int main(void) {
    srand(time(NULL));

    int n = 1000000;  // 백만 개
    int* arr1 = malloc(n * sizeof(int));
    int* arr2 = malloc(n * sizeof(int));

    // 랜덤 배열 생성
    for (int i = 0; i < n; i++) {
        arr1[i] = rand();
        arr2[i] = arr1[i];  // 복사
    }

    printf("배열 크기: %d\n\n", n);

    // 단일 스레드 정렬
    clock_t start = clock();
    merge_sort_single(arr1, 0, n - 1);
    clock_t end = clock();
    double single_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("단일 스레드: %.3f초\n", single_time);
    printf("정렬 검증: %s\n\n", is_sorted(arr1, n) ? "OK" : "FAIL");

    // 멀티스레드 정렬
    start = clock();
    SortTask task = { arr2, 0, n - 1 };
    merge_sort_parallel(&task);
    end = clock();
    double parallel_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("멀티스레드: %.3f초\n", parallel_time);
    printf("정렬 검증: %s\n\n", is_sorted(arr2, n) ? "OK" : "FAIL");

    printf("속도 향상: %.2fx\n", single_time / parallel_time);

    free(arr1);
    free(arr2);

    return 0;
}
