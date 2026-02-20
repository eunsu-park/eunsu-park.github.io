/*
 * 해시 테이블 (Hash Table)
 * Hash Functions, Chaining, Open Addressing
 *
 * 빠른 검색, 삽입, 삭제를 위한 자료구조입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 101
#define DELETED ((void*)-1)

/* =============================================================================
 * 1. 해시 함수들
 * ============================================================================= */

/* 나머지 연산 해시 */
unsigned int hash_mod(int key, int size) {
    return ((key % size) + size) % size;
}

/* 문자열 해시 (djb2) */
unsigned int hash_string(const char* str, int size) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % size;
}

/* 다항식 해시 */
unsigned int hash_polynomial(const char* str, int size) {
    unsigned long hash = 0;
    unsigned long p_pow = 1;
    const unsigned long p = 31;
    while (*str) {
        hash = (hash + (*str - 'a' + 1) * p_pow) % size;
        p_pow = (p_pow * p) % size;
        str++;
    }
    return hash;
}

/* =============================================================================
 * 2. 체이닝 해시 테이블
 * ============================================================================= */

typedef struct ChainNode {
    char* key;
    int value;
    struct ChainNode* next;
} ChainNode;

typedef struct {
    ChainNode** buckets;
    int size;
    int count;
} ChainHashTable;

ChainHashTable* chain_create(int size) {
    ChainHashTable* ht = malloc(sizeof(ChainHashTable));
    ht->buckets = calloc(size, sizeof(ChainNode*));
    ht->size = size;
    ht->count = 0;
    return ht;
}

void chain_insert(ChainHashTable* ht, const char* key, int value) {
    unsigned int idx = hash_string(key, ht->size);

    /* 기존 키 검색 */
    ChainNode* node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            node->value = value;
            return;
        }
        node = node->next;
    }

    /* 새 노드 삽입 */
    ChainNode* new_node = malloc(sizeof(ChainNode));
    new_node->key = strdup(key);
    new_node->value = value;
    new_node->next = ht->buckets[idx];
    ht->buckets[idx] = new_node;
    ht->count++;
}

int chain_get(ChainHashTable* ht, const char* key, int* found) {
    unsigned int idx = hash_string(key, ht->size);
    ChainNode* node = ht->buckets[idx];

    while (node) {
        if (strcmp(node->key, key) == 0) {
            *found = 1;
            return node->value;
        }
        node = node->next;
    }

    *found = 0;
    return 0;
}

void chain_delete(ChainHashTable* ht, const char* key) {
    unsigned int idx = hash_string(key, ht->size);
    ChainNode* node = ht->buckets[idx];
    ChainNode* prev = NULL;

    while (node) {
        if (strcmp(node->key, key) == 0) {
            if (prev) {
                prev->next = node->next;
            } else {
                ht->buckets[idx] = node->next;
            }
            free(node->key);
            free(node);
            ht->count--;
            return;
        }
        prev = node;
        node = node->next;
    }
}

void chain_free(ChainHashTable* ht) {
    for (int i = 0; i < ht->size; i++) {
        ChainNode* node = ht->buckets[i];
        while (node) {
            ChainNode* next = node->next;
            free(node->key);
            free(node);
            node = next;
        }
    }
    free(ht->buckets);
    free(ht);
}

/* =============================================================================
 * 3. 오픈 어드레싱 (선형 탐사)
 * ============================================================================= */

typedef struct {
    int* keys;
    int* values;
    bool* occupied;
    bool* deleted;
    int size;
    int count;
} LinearHashTable;

LinearHashTable* linear_create(int size) {
    LinearHashTable* ht = malloc(sizeof(LinearHashTable));
    ht->keys = malloc(size * sizeof(int));
    ht->values = malloc(size * sizeof(int));
    ht->occupied = calloc(size, sizeof(bool));
    ht->deleted = calloc(size, sizeof(bool));
    ht->size = size;
    ht->count = 0;
    return ht;
}

void linear_insert(LinearHashTable* ht, int key, int value) {
    if (ht->count >= ht->size * 0.7) {
        printf("    Warning: Load factor too high!\n");
        return;
    }

    unsigned int idx = hash_mod(key, ht->size);

    while (ht->occupied[idx] && !ht->deleted[idx] && ht->keys[idx] != key) {
        idx = (idx + 1) % ht->size;
    }

    if (!ht->occupied[idx] || ht->deleted[idx]) {
        ht->count++;
    }

    ht->keys[idx] = key;
    ht->values[idx] = value;
    ht->occupied[idx] = true;
    ht->deleted[idx] = false;
}

int linear_get(LinearHashTable* ht, int key, int* found) {
    unsigned int idx = hash_mod(key, ht->size);
    int start = idx;

    while (ht->occupied[idx]) {
        if (!ht->deleted[idx] && ht->keys[idx] == key) {
            *found = 1;
            return ht->values[idx];
        }
        idx = (idx + 1) % ht->size;
        if (idx == start) break;
    }

    *found = 0;
    return 0;
}

void linear_delete(LinearHashTable* ht, int key) {
    unsigned int idx = hash_mod(key, ht->size);
    int start = idx;

    while (ht->occupied[idx]) {
        if (!ht->deleted[idx] && ht->keys[idx] == key) {
            ht->deleted[idx] = true;
            ht->count--;
            return;
        }
        idx = (idx + 1) % ht->size;
        if (idx == start) break;
    }
}

void linear_free(LinearHashTable* ht) {
    free(ht->keys);
    free(ht->values);
    free(ht->occupied);
    free(ht->deleted);
    free(ht);
}

/* =============================================================================
 * 4. 이중 해싱
 * ============================================================================= */

typedef struct {
    int* keys;
    int* values;
    bool* occupied;
    bool* deleted;
    int size;
    int count;
} DoubleHashTable;

unsigned int hash2(int key, int size) {
    return 7 - (key % 7);  /* 0이 아닌 값 보장 */
}

DoubleHashTable* double_create(int size) {
    DoubleHashTable* ht = malloc(sizeof(DoubleHashTable));
    ht->keys = malloc(size * sizeof(int));
    ht->values = malloc(size * sizeof(int));
    ht->occupied = calloc(size, sizeof(bool));
    ht->deleted = calloc(size, sizeof(bool));
    ht->size = size;
    ht->count = 0;
    return ht;
}

void double_insert(DoubleHashTable* ht, int key, int value) {
    unsigned int idx = hash_mod(key, ht->size);
    unsigned int step = hash2(key, ht->size);

    while (ht->occupied[idx] && !ht->deleted[idx] && ht->keys[idx] != key) {
        idx = (idx + step) % ht->size;
    }

    if (!ht->occupied[idx] || ht->deleted[idx]) {
        ht->count++;
    }

    ht->keys[idx] = key;
    ht->values[idx] = value;
    ht->occupied[idx] = true;
    ht->deleted[idx] = false;
}

int double_get(DoubleHashTable* ht, int key, int* found) {
    unsigned int idx = hash_mod(key, ht->size);
    unsigned int step = hash2(key, ht->size);
    int start = idx;

    while (ht->occupied[idx]) {
        if (!ht->deleted[idx] && ht->keys[idx] == key) {
            *found = 1;
            return ht->values[idx];
        }
        idx = (idx + step) % ht->size;
        if (idx == start) break;
    }

    *found = 0;
    return 0;
}

void double_free(DoubleHashTable* ht) {
    free(ht->keys);
    free(ht->values);
    free(ht->occupied);
    free(ht->deleted);
    free(ht);
}

/* =============================================================================
 * 5. 실전: Two Sum
 * ============================================================================= */

int* two_sum(int nums[], int n, int target, int* result_size) {
    ChainHashTable* ht = chain_create(n * 2);
    int* result = malloc(2 * sizeof(int));
    *result_size = 0;

    for (int i = 0; i < n; i++) {
        int complement = target - nums[i];
        int found;
        int idx = chain_get(ht, (char[]){complement + '0', '\0'}, &found);

        if (found) {
            result[0] = idx;
            result[1] = i;
            *result_size = 2;
            chain_free(ht);
            return result;
        }

        /* 간단한 키 변환 (실제로는 int를 문자열로 변환해야 함) */
        char key[20];
        sprintf(key, "%d", nums[i]);
        chain_insert(ht, key, i);
    }

    chain_free(ht);
    free(result);
    return NULL;
}

/* =============================================================================
 * 6. 실전: 빈도 카운트
 * ============================================================================= */

void count_frequency(int arr[], int n) {
    LinearHashTable* ht = linear_create(n * 2 + 1);

    for (int i = 0; i < n; i++) {
        int found;
        int count = linear_get(ht, arr[i], &found);
        linear_insert(ht, arr[i], found ? count + 1 : 1);
    }

    printf("    빈도:\n");
    for (int i = 0; i < ht->size; i++) {
        if (ht->occupied[i] && !ht->deleted[i]) {
            printf("      %d: %d\n", ht->keys[i], ht->values[i]);
        }
    }

    linear_free(ht);
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("해시 테이블 (Hash Table) 예제\n");
    printf("============================================================\n");

    /* 1. 해시 함수 */
    printf("\n[1] 해시 함수\n");
    printf("    hash_mod(42, 101) = %u\n", hash_mod(42, TABLE_SIZE));
    printf("    hash_string(\"hello\", 101) = %u\n", hash_string("hello", TABLE_SIZE));
    printf("    hash_string(\"world\", 101) = %u\n", hash_string("world", TABLE_SIZE));

    /* 2. 체이닝 해시 테이블 */
    printf("\n[2] 체이닝 해시 테이블\n");
    ChainHashTable* chain_ht = chain_create(TABLE_SIZE);
    chain_insert(chain_ht, "apple", 100);
    chain_insert(chain_ht, "banana", 200);
    chain_insert(chain_ht, "cherry", 300);

    int found;
    printf("    apple: %d\n", chain_get(chain_ht, "apple", &found));
    printf("    banana: %d\n", chain_get(chain_ht, "banana", &found));

    chain_delete(chain_ht, "banana");
    chain_get(chain_ht, "banana", &found);
    printf("    banana 삭제 후: %s\n", found ? "found" : "not found");
    chain_free(chain_ht);

    /* 3. 선형 탐사 */
    printf("\n[3] 선형 탐사 (Linear Probing)\n");
    LinearHashTable* linear_ht = linear_create(TABLE_SIZE);
    linear_insert(linear_ht, 10, 100);
    linear_insert(linear_ht, 111, 200);  /* 충돌: 10 % 101 = 10, 111 % 101 = 10 */
    linear_insert(linear_ht, 212, 300);

    printf("    10: %d\n", linear_get(linear_ht, 10, &found));
    printf("    111: %d\n", linear_get(linear_ht, 111, &found));
    printf("    212: %d\n", linear_get(linear_ht, 212, &found));
    linear_free(linear_ht);

    /* 4. 이중 해싱 */
    printf("\n[4] 이중 해싱 (Double Hashing)\n");
    DoubleHashTable* double_ht = double_create(TABLE_SIZE);
    double_insert(double_ht, 10, 100);
    double_insert(double_ht, 111, 200);
    double_insert(double_ht, 212, 300);

    printf("    10: %d\n", double_get(double_ht, 10, &found));
    printf("    111: %d\n", double_get(double_ht, 111, &found));
    double_free(double_ht);

    /* 5. 빈도 카운트 */
    printf("\n[5] 빈도 카운트\n");
    int arr[] = {1, 2, 3, 1, 2, 1, 4, 2};
    printf("    배열: [1,2,3,1,2,1,4,2]\n");
    count_frequency(arr, 8);

    /* 6. 해시 테이블 비교 */
    printf("\n[6] 충돌 해결 방법 비교\n");
    printf("    | 방법       | 장점              | 단점              |\n");
    printf("    |------------|-------------------|-------------------|\n");
    printf("    | 체이닝     | 삭제 용이         | 메모리 오버헤드   |\n");
    printf("    | 선형 탐사  | 캐시 친화적       | 클러스터링        |\n");
    printf("    | 이중 해싱  | 클러스터링 감소   | 해시 계산 비용    |\n");

    printf("\n============================================================\n");

    return 0;
}
