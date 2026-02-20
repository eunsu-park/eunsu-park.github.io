/*
 * 탐욕 알고리즘 (Greedy Algorithm)
 * Activity Selection, Huffman Coding, Fractional Knapsack
 *
 * 매 순간 최적의 선택을 통해 전체 최적해를 구합니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =============================================================================
 * 1. 활동 선택 문제
 * ============================================================================= */

typedef struct {
    int start;
    int end;
} Activity;

int compare_activities(const void* a, const void* b) {
    return ((Activity*)a)->end - ((Activity*)b)->end;
}

int activity_selection(Activity activities[], int n) {
    qsort(activities, n, sizeof(Activity), compare_activities);

    int count = 1;
    int last_end = activities[0].end;

    for (int i = 1; i < n; i++) {
        if (activities[i].start >= last_end) {
            count++;
            last_end = activities[i].end;
        }
    }

    return count;
}

/* =============================================================================
 * 2. 분할 가능 배낭 문제
 * ============================================================================= */

typedef struct {
    double value;
    double weight;
    double ratio;
} Item;

int compare_items(const void* a, const void* b) {
    double r1 = ((Item*)a)->ratio;
    double r2 = ((Item*)b)->ratio;
    return (r2 > r1) - (r2 < r1);  /* 내림차순 */
}

double fractional_knapsack(Item items[], int n, double capacity) {
    qsort(items, n, sizeof(Item), compare_items);

    double total_value = 0;
    double remaining = capacity;

    for (int i = 0; i < n && remaining > 0; i++) {
        if (items[i].weight <= remaining) {
            total_value += items[i].value;
            remaining -= items[i].weight;
        } else {
            total_value += items[i].ratio * remaining;
            remaining = 0;
        }
    }

    return total_value;
}

/* =============================================================================
 * 3. 허프만 코딩
 * ============================================================================= */

typedef struct HuffmanNode {
    char ch;
    int freq;
    struct HuffmanNode* left;
    struct HuffmanNode* right;
} HuffmanNode;

HuffmanNode* create_huffman_node(char ch, int freq) {
    HuffmanNode* node = malloc(sizeof(HuffmanNode));
    node->ch = ch;
    node->freq = freq;
    node->left = node->right = NULL;
    return node;
}

/* 간단한 우선순위 큐 (배열 기반) */
typedef struct {
    HuffmanNode** data;
    int size;
    int capacity;
} MinHeap;

MinHeap* create_heap(int capacity) {
    MinHeap* h = malloc(sizeof(MinHeap));
    h->data = malloc(capacity * sizeof(HuffmanNode*));
    h->size = 0;
    h->capacity = capacity;
    return h;
}

void heap_insert(MinHeap* h, HuffmanNode* node) {
    int i = h->size++;
    h->data[i] = node;
    while (i > 0 && h->data[(i - 1) / 2]->freq > h->data[i]->freq) {
        HuffmanNode* temp = h->data[(i - 1) / 2];
        h->data[(i - 1) / 2] = h->data[i];
        h->data[i] = temp;
        i = (i - 1) / 2;
    }
}

HuffmanNode* heap_extract(MinHeap* h) {
    HuffmanNode* min = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    while (2 * i + 1 < h->size) {
        int smallest = i;
        if (h->data[2 * i + 1]->freq < h->data[smallest]->freq) smallest = 2 * i + 1;
        if (2 * i + 2 < h->size && h->data[2 * i + 2]->freq < h->data[smallest]->freq) smallest = 2 * i + 2;
        if (smallest == i) break;
        HuffmanNode* temp = h->data[i];
        h->data[i] = h->data[smallest];
        h->data[smallest] = temp;
        i = smallest;
    }
    return min;
}

HuffmanNode* build_huffman_tree(char chars[], int freqs[], int n) {
    MinHeap* heap = create_heap(n);
    for (int i = 0; i < n; i++) {
        heap_insert(heap, create_huffman_node(chars[i], freqs[i]));
    }

    while (heap->size > 1) {
        HuffmanNode* left = heap_extract(heap);
        HuffmanNode* right = heap_extract(heap);
        HuffmanNode* merged = create_huffman_node('\0', left->freq + right->freq);
        merged->left = left;
        merged->right = right;
        heap_insert(heap, merged);
    }

    HuffmanNode* root = heap_extract(heap);
    free(heap->data);
    free(heap);
    return root;
}

void print_huffman_codes(HuffmanNode* root, char code[], int depth) {
    if (!root) return;
    if (!root->left && !root->right) {
        code[depth] = '\0';
        printf("      '%c': %s\n", root->ch, code);
        return;
    }
    code[depth] = '0';
    print_huffman_codes(root->left, code, depth + 1);
    code[depth] = '1';
    print_huffman_codes(root->right, code, depth + 1);
}

void free_huffman_tree(HuffmanNode* root) {
    if (!root) return;
    free_huffman_tree(root->left);
    free_huffman_tree(root->right);
    free(root);
}

/* =============================================================================
 * 4. 동전 거스름돈
 * ============================================================================= */

int coin_change_greedy(int coins[], int n, int amount) {
    int count = 0;
    for (int i = n - 1; i >= 0 && amount > 0; i--) {
        count += amount / coins[i];
        amount %= coins[i];
    }
    return (amount == 0) ? count : -1;
}

/* =============================================================================
 * 5. 회의실 배정 (최소 회의실 수)
 * ============================================================================= */

typedef struct {
    int time;
    int type;  /* 1: 시작, -1: 종료 */
} Event;

int compare_events(const void* a, const void* b) {
    Event* e1 = (Event*)a;
    Event* e2 = (Event*)b;
    if (e1->time != e2->time) return e1->time - e2->time;
    return e1->type - e2->type;  /* 종료 먼저 */
}

int min_meeting_rooms(int starts[], int ends[], int n) {
    Event* events = malloc(2 * n * sizeof(Event));
    for (int i = 0; i < n; i++) {
        events[2 * i] = (Event){starts[i], 1};
        events[2 * i + 1] = (Event){ends[i], -1};
    }
    qsort(events, 2 * n, sizeof(Event), compare_events);

    int max_rooms = 0, current = 0;
    for (int i = 0; i < 2 * n; i++) {
        current += events[i].type;
        if (current > max_rooms) max_rooms = current;
    }
    free(events);
    return max_rooms;
}

/* =============================================================================
 * 6. 점프 게임
 * ============================================================================= */

int can_jump(int nums[], int n) {
    int max_reach = 0;
    for (int i = 0; i < n; i++) {
        if (i > max_reach) return 0;
        if (i + nums[i] > max_reach) max_reach = i + nums[i];
    }
    return 1;
}

int min_jumps(int nums[], int n) {
    if (n <= 1) return 0;
    int jumps = 0, current_end = 0, farthest = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i + nums[i] > farthest) farthest = i + nums[i];
        if (i == current_end) {
            jumps++;
            current_end = farthest;
            if (current_end >= n - 1) break;
        }
    }
    return jumps;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("탐욕 알고리즘 (Greedy) 예제\n");
    printf("============================================================\n");

    /* 1. 활동 선택 */
    printf("\n[1] 활동 선택 문제\n");
    Activity activities[] = {{1, 4}, {3, 5}, {0, 6}, {5, 7}, {3, 9}, {5, 9}};
    printf("    최대 활동 수: %d\n", activity_selection(activities, 6));

    /* 2. 분할 배낭 */
    printf("\n[2] 분할 가능 배낭\n");
    Item items[] = {{60, 10, 6}, {100, 20, 5}, {120, 30, 4}};
    printf("    용량 50의 최대 가치: %.2f\n", fractional_knapsack(items, 3, 50));

    /* 3. 허프만 코딩 */
    printf("\n[3] 허프만 코딩\n");
    char chars[] = {'a', 'b', 'c', 'd', 'e'};
    int freqs[] = {5, 9, 12, 13, 16};
    HuffmanNode* root = build_huffman_tree(chars, freqs, 5);
    printf("    코드:\n");
    char code[100];
    print_huffman_codes(root, code, 0);
    free_huffman_tree(root);

    /* 4. 동전 거스름돈 */
    printf("\n[4] 동전 거스름돈\n");
    int coins[] = {1, 5, 10, 50, 100, 500};
    printf("    동전 [1,5,10,50,100,500], 금액 730\n");
    printf("    최소 동전 수: %d\n", coin_change_greedy(coins, 6, 730));

    /* 5. 회의실 배정 */
    printf("\n[5] 최소 회의실 수\n");
    int starts[] = {0, 5, 15};
    int ends[] = {30, 10, 20};
    printf("    회의: (0-30), (5-10), (15-20)\n");
    printf("    필요 회의실: %d개\n", min_meeting_rooms(starts, ends, 3));

    /* 6. 점프 게임 */
    printf("\n[6] 점프 게임\n");
    int nums1[] = {2, 3, 1, 1, 4};
    int nums2[] = {3, 2, 1, 0, 4};
    printf("    [2,3,1,1,4]: 도달=%s, 최소점프=%d\n",
           can_jump(nums1, 5) ? "가능" : "불가", min_jumps(nums1, 5));
    printf("    [3,2,1,0,4]: 도달=%s\n", can_jump(nums2, 5) ? "가능" : "불가");

    printf("\n============================================================\n");

    return 0;
}
