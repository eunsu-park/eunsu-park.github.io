/*
 * hash_chaining.c
 * 체이닝(Separate Chaining)을 이용한 해시 테이블 구현
 *
 * 체이닝 방식:
 * - 충돌 발생 시 같은 버킷에 연결 리스트로 저장
 * - 장점: 삽입/삭제 간단, 테이블 크기 제한 없음
 * - 단점: 포인터 추가 메모리, 캐시 효율 낮음
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 10
#define KEY_SIZE 50
#define VALUE_SIZE 100

// 노드 구조체 (키-값 쌍을 저장)
typedef struct Node {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
    struct Node *next;  // 다음 노드 (체이닝)
} Node;

// 해시 테이블 구조체
typedef struct {
    Node *buckets[TABLE_SIZE];  // 버킷 배열
    int count;                   // 저장된 항목 개수
    int collisions;              // 충돌 횟수
} HashTable;

// 통계 정보
typedef struct {
    int total_inserts;
    int total_searches;
    int total_deletes;
    int chain_lengths[TABLE_SIZE];
} Statistics;

// djb2 해시 함수
unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

// 해시 테이블 생성
HashTable* ht_create(void) {
    HashTable *ht = malloc(sizeof(HashTable));
    if (!ht) {
        fprintf(stderr, "메모리 할당 실패\n");
        return NULL;
    }

    // 모든 버킷 초기화
    for (int i = 0; i < TABLE_SIZE; i++) {
        ht->buckets[i] = NULL;
    }
    ht->count = 0;
    ht->collisions = 0;

    return ht;
}

// 해시 테이블 해제
void ht_destroy(HashTable *ht) {
    if (!ht) return;

    // 각 버킷의 체인 해제
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = ht->buckets[i];
        while (current) {
            Node *next = current->next;
            free(current);
            current = next;
        }
    }
    free(ht);
}

// 삽입 또는 수정
bool ht_set(HashTable *ht, const char *key, const char *value) {
    if (!ht || !key || !value) return false;

    unsigned int index = hash(key);

    // 기존 키가 있는지 확인
    Node *current = ht->buckets[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            // 기존 키 발견 → 값만 업데이트
            strncpy(current->value, value, VALUE_SIZE - 1);
            current->value[VALUE_SIZE - 1] = '\0';
            return true;
        }
        current = current->next;
    }

    // 새 노드 생성
    Node *node = malloc(sizeof(Node));
    if (!node) {
        fprintf(stderr, "메모리 할당 실패\n");
        return false;
    }

    strncpy(node->key, key, KEY_SIZE - 1);
    node->key[KEY_SIZE - 1] = '\0';
    strncpy(node->value, value, VALUE_SIZE - 1);
    node->value[VALUE_SIZE - 1] = '\0';

    // 버킷 맨 앞에 삽입 (O(1))
    node->next = ht->buckets[index];

    // 충돌 확인 (버킷에 이미 노드가 있으면 충돌)
    if (ht->buckets[index] != NULL) {
        ht->collisions++;
    }

    ht->buckets[index] = node;
    ht->count++;

    return true;
}

// 검색
char* ht_get(HashTable *ht, const char *key) {
    if (!ht || !key) return NULL;

    unsigned int index = hash(key);

    // 체인 탐색
    Node *current = ht->buckets[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            return current->value;  // 찾음!
        }
        current = current->next;
    }

    return NULL;  // 찾지 못함
}

// 삭제
bool ht_delete(HashTable *ht, const char *key) {
    if (!ht || !key) return false;

    unsigned int index = hash(key);

    Node *current = ht->buckets[index];
    Node *prev = NULL;

    // 체인에서 노드 찾기
    while (current) {
        if (strcmp(current->key, key) == 0) {
            // 노드 제거
            if (prev) {
                prev->next = current->next;  // 중간 또는 끝
            } else {
                ht->buckets[index] = current->next;  // 맨 앞
            }
            free(current);
            ht->count--;
            return true;
        }
        prev = current;
        current = current->next;
    }

    return false;  // 찾지 못함
}

// 해시 테이블 출력
void ht_print(HashTable *ht) {
    if (!ht) return;

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║         해시 테이블 상태 (체이닝)         ║\n");
    printf("╠════════════════════════════════════════════╣\n");
    printf("║  항목 개수: %-5d                          ║\n", ht->count);
    printf("║  충돌 횟수: %-5d                          ║\n", ht->collisions);
    printf("║  로드 팩터: %.2f                           ║\n",
           (double)ht->count / TABLE_SIZE);
    printf("╚════════════════════════════════════════════╝\n\n");

    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("[%d]: ", i);

        Node *current = ht->buckets[i];
        if (!current) {
            printf("(비어있음)\n");
            continue;
        }

        // 체인 출력
        int chain_length = 0;
        while (current) {
            printf("[\"%s\":\"%s\"]", current->key, current->value);
            if (current->next) printf(" → ");
            current = current->next;
            chain_length++;
        }
        printf(" (길이: %d)\n", chain_length);
    }
}

// 통계 수집
void ht_get_statistics(HashTable *ht, Statistics *stats) {
    if (!ht || !stats) return;

    memset(stats, 0, sizeof(Statistics));

    stats->total_inserts = ht->count;

    // 각 버킷의 체인 길이 계산
    for (int i = 0; i < TABLE_SIZE; i++) {
        int length = 0;
        Node *current = ht->buckets[i];
        while (current) {
            length++;
            current = current->next;
        }
        stats->chain_lengths[i] = length;
    }
}

// 통계 출력
void print_statistics(HashTable *ht) {
    Statistics stats;
    ht_get_statistics(ht, &stats);

    printf("\n=== 성능 통계 ===\n\n");

    // 최대 체인 길이
    int max_length = 0;
    int empty_buckets = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (stats.chain_lengths[i] > max_length) {
            max_length = stats.chain_lengths[i];
        }
        if (stats.chain_lengths[i] == 0) {
            empty_buckets++;
        }
    }

    double avg_chain_length = (double)ht->count / (TABLE_SIZE - empty_buckets);

    printf("저장된 항목:     %d\n", ht->count);
    printf("충돌 횟수:       %d\n", ht->collisions);
    printf("비어있는 버킷:   %d / %d\n", empty_buckets, TABLE_SIZE);
    printf("최대 체인 길이:  %d\n", max_length);
    printf("평균 체인 길이:  %.2f\n", avg_chain_length);
    printf("로드 팩터:       %.2f\n", (double)ht->count / TABLE_SIZE);

    // 체인 길이 분포
    printf("\n체인 길이 분포:\n");
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (stats.chain_lengths[i] > 0) {
            printf("  버킷 %d: ", i);
            for (int j = 0; j < stats.chain_lengths[i]; j++) {
                printf("█");
            }
            printf(" (%d)\n", stats.chain_lengths[i]);
        }
    }
}

// 키 존재 여부 확인
bool ht_contains(HashTable *ht, const char *key) {
    return ht_get(ht, key) != NULL;
}

// 모든 키 출력
void ht_print_keys(HashTable *ht) {
    if (!ht) return;

    printf("\n=== 저장된 키 목록 ===\n");
    int count = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = ht->buckets[i];
        while (current) {
            printf("  %d. %s\n", ++count, current->key);
            current = current->next;
        }
    }
    printf("총 %d개\n", count);
}

// 테스트 함수
int main(void) {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║      체이닝 해시 테이블 구현 및 테스트    ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    HashTable *ht = ht_create();
    if (!ht) return 1;

    // 1. 삽입 테스트
    printf("\n[ 1단계: 삽입 테스트 ]\n");
    printf("여러 과일 이름과 한글명을 삽입합니다...\n");

    ht_set(ht, "apple", "사과");
    ht_set(ht, "banana", "바나나");
    ht_set(ht, "cherry", "체리");
    ht_set(ht, "date", "대추야자");
    ht_set(ht, "elderberry", "엘더베리");
    ht_set(ht, "fig", "무화과");
    ht_set(ht, "grape", "포도");
    ht_set(ht, "honeydew", "허니듀 멜론");

    ht_print(ht);

    // 2. 검색 테스트
    printf("\n[ 2단계: 검색 테스트 ]\n");
    const char *search_keys[] = {"apple", "grape", "kiwi", "banana"};
    for (int i = 0; i < 4; i++) {
        char *value = ht_get(ht, search_keys[i]);
        if (value) {
            printf("✓ '%s' → '%s'\n", search_keys[i], value);
        } else {
            printf("✗ '%s' → (찾을 수 없음)\n", search_keys[i]);
        }
    }

    // 3. 수정 테스트
    printf("\n[ 3단계: 수정 테스트 ]\n");
    printf("'apple'의 값을 수정합니다...\n");
    ht_set(ht, "apple", "맛있는 사과 🍎");
    printf("수정 후: apple → %s\n", ht_get(ht, "apple"));

    // 4. 삭제 테스트
    printf("\n[ 4단계: 삭제 테스트 ]\n");
    printf("'banana'를 삭제합니다...\n");
    if (ht_delete(ht, "banana")) {
        printf("✓ 삭제 성공\n");
    }
    printf("삭제 확인: banana → %s\n",
           ht_get(ht, "banana") ?: "(찾을 수 없음)");

    ht_print(ht);

    // 5. 충돌 테스트 (같은 해시값을 가지도록)
    printf("\n[ 5단계: 충돌 발생 테스트 ]\n");
    printf("추가 데이터를 삽입하여 충돌을 유발합니다...\n");

    ht_set(ht, "kiwi", "키위");
    ht_set(ht, "lemon", "레몬");
    ht_set(ht, "mango", "망고");

    ht_print(ht);

    // 6. 성능 통계
    print_statistics(ht);

    // 7. 키 목록
    ht_print_keys(ht);

    // 정리
    ht_destroy(ht);

    printf("\n프로그램을 종료합니다.\n");
    return 0;
}
