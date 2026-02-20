/*
 * 힙 (Heap)
 * Min Heap, Max Heap, Heap Sort, Priority Queue
 *
 * 완전 이진 트리 기반의 우선순위 자료구조입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* =============================================================================
 * 1. 최소 힙 (Min Heap)
 * ============================================================================= */

typedef struct {
    int* data;
    int size;
    int capacity;
} MinHeap;

MinHeap* minheap_create(int capacity) {
    MinHeap* heap = malloc(sizeof(MinHeap));
    heap->data = malloc(capacity * sizeof(int));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void minheap_free(MinHeap* heap) {
    free(heap->data);
    free(heap);
}

void minheap_swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void minheap_sift_up(MinHeap* heap, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap->data[parent] <= heap->data[idx])
            break;
        minheap_swap(&heap->data[parent], &heap->data[idx]);
        idx = parent;
    }
}

void minheap_sift_down(MinHeap* heap, int idx) {
    while (2 * idx + 1 < heap->size) {
        int smallest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (left < heap->size && heap->data[left] < heap->data[smallest])
            smallest = left;
        if (right < heap->size && heap->data[right] < heap->data[smallest])
            smallest = right;

        if (smallest == idx) break;

        minheap_swap(&heap->data[idx], &heap->data[smallest]);
        idx = smallest;
    }
}

void minheap_push(MinHeap* heap, int val) {
    if (heap->size >= heap->capacity) return;
    heap->data[heap->size] = val;
    minheap_sift_up(heap, heap->size);
    heap->size++;
}

int minheap_pop(MinHeap* heap) {
    if (heap->size == 0) return -1;

    int min = heap->data[0];
    heap->data[0] = heap->data[--heap->size];
    minheap_sift_down(heap, 0);
    return min;
}

int minheap_peek(MinHeap* heap) {
    return heap->size > 0 ? heap->data[0] : -1;
}

/* =============================================================================
 * 2. 최대 힙 (Max Heap)
 * ============================================================================= */

typedef struct {
    int* data;
    int size;
    int capacity;
} MaxHeap;

MaxHeap* maxheap_create(int capacity) {
    MaxHeap* heap = malloc(sizeof(MaxHeap));
    heap->data = malloc(capacity * sizeof(int));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void maxheap_free(MaxHeap* heap) {
    free(heap->data);
    free(heap);
}

void maxheap_sift_up(MaxHeap* heap, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap->data[parent] >= heap->data[idx])
            break;
        minheap_swap(&heap->data[parent], &heap->data[idx]);
        idx = parent;
    }
}

void maxheap_sift_down(MaxHeap* heap, int idx) {
    while (2 * idx + 1 < heap->size) {
        int largest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (left < heap->size && heap->data[left] > heap->data[largest])
            largest = left;
        if (right < heap->size && heap->data[right] > heap->data[largest])
            largest = right;

        if (largest == idx) break;

        minheap_swap(&heap->data[idx], &heap->data[largest]);
        idx = largest;
    }
}

void maxheap_push(MaxHeap* heap, int val) {
    if (heap->size >= heap->capacity) return;
    heap->data[heap->size] = val;
    maxheap_sift_up(heap, heap->size);
    heap->size++;
}

int maxheap_pop(MaxHeap* heap) {
    if (heap->size == 0) return -1;

    int max = heap->data[0];
    heap->data[0] = heap->data[--heap->size];
    maxheap_sift_down(heap, 0);
    return max;
}

/* =============================================================================
 * 3. 힙 정렬
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
        minheap_swap(&arr[i], &arr[largest]);
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
        minheap_swap(&arr[0], &arr[i]);
        heapify(arr, i, 0);
    }
}

/* =============================================================================
 * 4. K번째 최소/최대 원소
 * ============================================================================= */

int kth_smallest(int arr[], int n, int k) {
    MaxHeap* heap = maxheap_create(k);

    for (int i = 0; i < n; i++) {
        if (heap->size < k) {
            maxheap_push(heap, arr[i]);
        } else if (arr[i] < heap->data[0]) {
            maxheap_pop(heap);
            maxheap_push(heap, arr[i]);
        }
    }

    int result = heap->data[0];
    maxheap_free(heap);
    return result;
}

int kth_largest(int arr[], int n, int k) {
    MinHeap* heap = minheap_create(k);

    for (int i = 0; i < n; i++) {
        if (heap->size < k) {
            minheap_push(heap, arr[i]);
        } else if (arr[i] > heap->data[0]) {
            minheap_pop(heap);
            minheap_push(heap, arr[i]);
        }
    }

    int result = heap->data[0];
    minheap_free(heap);
    return result;
}

/* =============================================================================
 * 5. 중앙값 찾기 (두 개의 힙)
 * ============================================================================= */

typedef struct {
    MaxHeap* lower;  /* 하위 절반 (최대 힙) */
    MinHeap* upper;  /* 상위 절반 (최소 힙) */
} MedianFinder;

MedianFinder* median_finder_create(int capacity) {
    MedianFinder* mf = malloc(sizeof(MedianFinder));
    mf->lower = maxheap_create(capacity);
    mf->upper = minheap_create(capacity);
    return mf;
}

void median_finder_free(MedianFinder* mf) {
    maxheap_free(mf->lower);
    minheap_free(mf->upper);
    free(mf);
}

void median_finder_add(MedianFinder* mf, int num) {
    /* lower에 추가 */
    maxheap_push(mf->lower, num);

    /* lower의 최댓값을 upper로 */
    minheap_push(mf->upper, maxheap_pop(mf->lower));

    /* 균형 맞추기 */
    if (mf->upper->size > mf->lower->size) {
        maxheap_push(mf->lower, minheap_pop(mf->upper));
    }
}

double median_finder_get(MedianFinder* mf) {
    if (mf->lower->size > mf->upper->size)
        return mf->lower->data[0];
    return (mf->lower->data[0] + mf->upper->data[0]) / 2.0;
}

/* =============================================================================
 * 6. K개 정렬 리스트 병합
 * ============================================================================= */

typedef struct {
    int val;
    int list_idx;
    int elem_idx;
} HeapNode;

typedef struct {
    HeapNode* data;
    int size;
    int capacity;
} NodeHeap;

NodeHeap* nodeheap_create(int capacity) {
    NodeHeap* heap = malloc(sizeof(NodeHeap));
    heap->data = malloc(capacity * sizeof(HeapNode));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void nodeheap_push(NodeHeap* heap, HeapNode node) {
    int idx = heap->size++;
    heap->data[idx] = node;

    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap->data[parent].val <= heap->data[idx].val)
            break;
        HeapNode temp = heap->data[parent];
        heap->data[parent] = heap->data[idx];
        heap->data[idx] = temp;
        idx = parent;
    }
}

HeapNode nodeheap_pop(NodeHeap* heap) {
    HeapNode min = heap->data[0];
    heap->data[0] = heap->data[--heap->size];

    int idx = 0;
    while (2 * idx + 1 < heap->size) {
        int smallest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (heap->data[left].val < heap->data[smallest].val)
            smallest = left;
        if (right < heap->size && heap->data[right].val < heap->data[smallest].val)
            smallest = right;

        if (smallest == idx) break;

        HeapNode temp = heap->data[idx];
        heap->data[idx] = heap->data[smallest];
        heap->data[smallest] = temp;
        idx = smallest;
    }

    return min;
}

int* merge_k_sorted(int** lists, int* sizes, int k, int* result_size) {
    *result_size = 0;
    for (int i = 0; i < k; i++)
        *result_size += sizes[i];

    int* result = malloc(*result_size * sizeof(int));
    NodeHeap* heap = nodeheap_create(k);

    /* 각 리스트의 첫 원소를 힙에 추가 */
    for (int i = 0; i < k; i++) {
        if (sizes[i] > 0) {
            nodeheap_push(heap, (HeapNode){lists[i][0], i, 0});
        }
    }

    int idx = 0;
    while (heap->size > 0) {
        HeapNode min_node = nodeheap_pop(heap);
        result[idx++] = min_node.val;

        /* 같은 리스트의 다음 원소 추가 */
        if (min_node.elem_idx + 1 < sizes[min_node.list_idx]) {
            int next_val = lists[min_node.list_idx][min_node.elem_idx + 1];
            nodeheap_push(heap, (HeapNode){next_val, min_node.list_idx, min_node.elem_idx + 1});
        }
    }

    free(heap->data);
    free(heap);
    return result;
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
    printf("힙 (Heap) 예제\n");
    printf("============================================================\n");

    /* 1. 최소 힙 */
    printf("\n[1] 최소 힙\n");
    MinHeap* min_heap = minheap_create(10);
    int vals[] = {5, 3, 8, 1, 2, 9, 4};
    printf("    삽입: ");
    print_array(vals, 7);
    printf("\n");

    for (int i = 0; i < 7; i++)
        minheap_push(min_heap, vals[i]);

    printf("    추출: ");
    while (min_heap->size > 0)
        printf("%d ", minheap_pop(min_heap));
    printf("\n");
    minheap_free(min_heap);

    /* 2. 최대 힙 */
    printf("\n[2] 최대 힙\n");
    MaxHeap* max_heap = maxheap_create(10);
    for (int i = 0; i < 7; i++)
        maxheap_push(max_heap, vals[i]);

    printf("    추출: ");
    while (max_heap->size > 0)
        printf("%d ", maxheap_pop(max_heap));
    printf("\n");
    maxheap_free(max_heap);

    /* 3. 힙 정렬 */
    printf("\n[3] 힙 정렬\n");
    int arr3[] = {12, 11, 13, 5, 6, 7};
    printf("    정렬 전: ");
    print_array(arr3, 6);
    printf("\n");
    heap_sort(arr3, 6);
    printf("    정렬 후: ");
    print_array(arr3, 6);
    printf("\n");

    /* 4. K번째 원소 */
    printf("\n[4] K번째 원소\n");
    int arr4[] = {7, 10, 4, 3, 20, 15};
    printf("    배열: ");
    print_array(arr4, 6);
    printf("\n");
    printf("    3번째 최소: %d\n", kth_smallest(arr4, 6, 3));
    printf("    2번째 최대: %d\n", kth_largest(arr4, 6, 2));

    /* 5. 중앙값 찾기 */
    printf("\n[5] 스트림 중앙값\n");
    MedianFinder* mf = median_finder_create(10);
    int stream[] = {2, 3, 4};
    for (int i = 0; i < 3; i++) {
        median_finder_add(mf, stream[i]);
        printf("    삽입 %d 후 중앙값: %.1f\n", stream[i], median_finder_get(mf));
    }
    median_finder_free(mf);

    /* 6. K개 정렬 리스트 병합 */
    printf("\n[6] K개 정렬 리스트 병합\n");
    int list1[] = {1, 4, 5};
    int list2[] = {1, 3, 4};
    int list3[] = {2, 6};
    int* lists[] = {list1, list2, list3};
    int sizes[] = {3, 3, 2};

    int result_size;
    int* merged = merge_k_sorted(lists, sizes, 3, &result_size);
    printf("    병합 결과: ");
    print_array(merged, result_size);
    printf("\n");
    free(merged);

    /* 7. 힙 연산 복잡도 */
    printf("\n[7] 힙 연산 복잡도\n");
    printf("    | 연산      | 시간복잡도 |\n");
    printf("    |-----------|------------|\n");
    printf("    | 삽입      | O(log n)   |\n");
    printf("    | 삭제(최소)| O(log n)   |\n");
    printf("    | 조회(최소)| O(1)       |\n");
    printf("    | 힙 구성   | O(n)       |\n");
    printf("    | 힙 정렬   | O(n log n) |\n");

    printf("\n============================================================\n");

    return 0;
}
