/*
 * hash_linear_probing.c
 * 선형 탐사(Linear Probing)를 이용한 오픈 어드레싱 해시 테이블
 *
 * 선형 탐사 방식:
 * - 충돌 발생 시 다음 빈 슬롯을 순차적으로 탐색
 * - 장점: 캐시 효율 좋음, 추가 메모리 불필요
 * - 단점: 클러스터링 현상, 삭제 처리 복잡
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 20
#define KEY_SIZE 50
#define VALUE_SIZE 100

// 슬롯 상태
typedef enum {
    EMPTY,      // 비어있음 (한 번도 사용 안 됨)
    OCCUPIED,   // 사용 중
    DELETED     // 삭제됨 (탐색 시 건너뛰지 않음)
} SlotStatus;

// 슬롯 구조체
typedef struct {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
    SlotStatus status;
} Slot;

// 해시 테이블 구조체
typedef struct {
    Slot slots[TABLE_SIZE];
    int count;          // 현재 저장된 항목 수 (OCCUPIED)
    int probes;         // 총 탐사 횟수
    int collisions;     // 충돌 횟수
} HashTable;

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

    // 모든 슬롯 초기화
    for (int i = 0; i < TABLE_SIZE; i++) {
        ht->slots[i].status = EMPTY;
        ht->slots[i].key[0] = '\0';
        ht->slots[i].value[0] = '\0';
    }

    ht->count = 0;
    ht->probes = 0;
    ht->collisions = 0;

    return ht;
}

// 해시 테이블 해제
void ht_destroy(HashTable *ht) {
    free(ht);
}

// 삽입 또는 수정
bool ht_set(HashTable *ht, const char *key, const char *value) {
    if (!ht || !key || !value) return false;

    // 테이블이 가득 찬 경우
    if (ht->count >= TABLE_SIZE) {
        fprintf(stderr, "해시 테이블이 가득 찼습니다!\n");
        return false;
    }

    unsigned int index = hash(key);
    unsigned int original_index = index;
    int probe_count = 0;

    // 선형 탐사
    do {
        probe_count++;

        // 1. 빈 슬롯 또는 삭제된 슬롯 → 새로 삽입
        if (ht->slots[index].status != OCCUPIED) {
            if (probe_count > 1) {
                ht->collisions++;  // 충돌 발생
            }

            strncpy(ht->slots[index].key, key, KEY_SIZE - 1);
            ht->slots[index].key[KEY_SIZE - 1] = '\0';
            strncpy(ht->slots[index].value, value, VALUE_SIZE - 1);
            ht->slots[index].value[VALUE_SIZE - 1] = '\0';
            ht->slots[index].status = OCCUPIED;

            ht->count++;
            ht->probes += probe_count;
            return true;
        }

        // 2. 같은 키 발견 → 값 업데이트
        if (strcmp(ht->slots[index].key, key) == 0) {
            strncpy(ht->slots[index].value, value, VALUE_SIZE - 1);
            ht->slots[index].value[VALUE_SIZE - 1] = '\0';
            ht->probes += probe_count;
            return true;
        }

        // 3. 다음 슬롯으로
        index = (index + 1) % TABLE_SIZE;

    } while (index != original_index);

    // 모든 슬롯 탐색했지만 실패 (이론상 도달 불가)
    return false;
}

// 검색
char* ht_get(HashTable *ht, const char *key) {
    if (!ht || !key) return NULL;

    unsigned int index = hash(key);
    unsigned int original_index = index;

    // 선형 탐사
    do {
        // EMPTY 발견 → 키가 없음 (탐색 종료)
        if (ht->slots[index].status == EMPTY) {
            return NULL;
        }

        // OCCUPIED이고 키가 일치 → 찾음!
        if (ht->slots[index].status == OCCUPIED &&
            strcmp(ht->slots[index].key, key) == 0) {
            return ht->slots[index].value;
        }

        // DELETED 또는 다른 키 → 계속 탐색
        index = (index + 1) % TABLE_SIZE;

    } while (index != original_index);

    return NULL;  // 찾지 못함
}

// 삭제
bool ht_delete(HashTable *ht, const char *key) {
    if (!ht || !key) return false;

    unsigned int index = hash(key);
    unsigned int original_index = index;

    do {
        // EMPTY 발견 → 키가 없음
        if (ht->slots[index].status == EMPTY) {
            return false;
        }

        // OCCUPIED이고 키가 일치 → 삭제
        if (ht->slots[index].status == OCCUPIED &&
            strcmp(ht->slots[index].key, key) == 0) {

            // EMPTY가 아닌 DELETED로 표시 (중요!)
            // 이유: 탐색 체인을 끊지 않기 위해
            ht->slots[index].status = DELETED;
            ht->count--;
            return true;
        }

        index = (index + 1) % TABLE_SIZE;

    } while (index != original_index);

    return false;  // 찾지 못함
}

// 해시 테이블 출력
void ht_print(HashTable *ht) {
    if (!ht) return;

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║      해시 테이블 상태 (선형 탐사)         ║\n");
    printf("╠════════════════════════════════════════════╣\n");
    printf("║  항목 개수: %-5d / %-5d                   ║\n",
           ht->count, TABLE_SIZE);
    printf("║  로드 팩터: %.2f                           ║\n",
           (double)ht->count / TABLE_SIZE);
    printf("║  충돌 횟수: %-5d                          ║\n", ht->collisions);
    printf("║  평균 탐사: %.2f                           ║\n",
           ht->count > 0 ? (double)ht->probes / ht->count : 0.0);
    printf("╚════════════════════════════════════════════╝\n\n");

    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("[%2d] ", i);

        switch (ht->slots[i].status) {
            case EMPTY:
                printf("(비어있음)\n");
                break;

            case DELETED:
                printf("(삭제됨) [이전 키: %s]\n",
                       ht->slots[i].key[0] ? ht->slots[i].key : "?");
                break;

            case OCCUPIED: {
                unsigned int original_hash = hash(ht->slots[i].key);
                if (original_hash == (unsigned int)i) {
                    printf("\"%s\" : \"%s\" ✓\n",
                           ht->slots[i].key, ht->slots[i].value);
                } else {
                    printf("\"%s\" : \"%s\" (원래: [%u], 충돌)\n",
                           ht->slots[i].key, ht->slots[i].value, original_hash);
                }
                break;
            }
        }
    }
}

// 클러스터링 분석
void analyze_clustering(HashTable *ht) {
    printf("\n=== 클러스터링 분석 ===\n\n");

    int max_cluster = 0;
    int current_cluster = 0;
    int num_clusters = 0;

    for (int i = 0; i < TABLE_SIZE; i++) {
        if (ht->slots[i].status == OCCUPIED) {
            current_cluster++;
        } else {
            if (current_cluster > 0) {
                num_clusters++;
                if (current_cluster > max_cluster) {
                    max_cluster = current_cluster;
                }
            }
            current_cluster = 0;
        }
    }

    // 마지막 클러스터 처리
    if (current_cluster > 0) {
        num_clusters++;
        if (current_cluster > max_cluster) {
            max_cluster = current_cluster;
        }
    }

    printf("클러스터 개수:   %d\n", num_clusters);
    printf("최대 클러스터:   %d개 연속\n", max_cluster);

    // 시각화
    printf("\n클러스터 시각화:\n");
    printf("[");
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (ht->slots[i].status == OCCUPIED) {
            printf("█");
        } else if (ht->slots[i].status == DELETED) {
            printf("░");
        } else {
            printf(" ");
        }
    }
    printf("]\n");
    printf("█: 사용중  ░: 삭제됨  (공백): 비어있음\n");
}

// 성능 테스트
void performance_test(void) {
    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║           로드 팩터별 성능 비교            ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    int test_sizes[] = {5, 10, 15, 18};  // 25%, 50%, 75%, 90%

    printf("로드 팩터  | 평균 탐사 | 충돌 횟수\n");
    printf("-----------|-----------|----------\n");

    for (int t = 0; t < 4; t++) {
        HashTable *ht = ht_create();
        int n = test_sizes[t];

        // 테스트 데이터 삽입
        char key[20], value[20];
        for (int i = 0; i < n; i++) {
            sprintf(key, "key%d", i);
            sprintf(value, "value%d", i);
            ht_set(ht, key, value);
        }

        double load_factor = (double)ht->count / TABLE_SIZE;
        double avg_probes = ht->count > 0 ? (double)ht->probes / ht->count : 0.0;

        printf("%6.0f%%    | %9.2f | %9d\n",
               load_factor * 100, avg_probes, ht->collisions);

        ht_destroy(ht);
    }

    printf("\n※ 로드 팩터가 높을수록 성능 저하\n");
    printf("※ 0.7 이하 유지 권장\n");
}

// 메인 테스트
int main(void) {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║   선형 탐사 해시 테이블 구현 및 테스트    ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    HashTable *ht = ht_create();
    if (!ht) return 1;

    // 1. 삽입 테스트
    printf("\n[ 1단계: 삽입 테스트 ]\n");
    printf("여러 데이터를 삽입합니다...\n");

    ht_set(ht, "apple", "사과");
    ht_set(ht, "banana", "바나나");
    ht_set(ht, "cherry", "체리");
    ht_set(ht, "date", "대추야자");
    ht_set(ht, "elderberry", "엘더베리");
    ht_set(ht, "fig", "무화과");
    ht_set(ht, "grape", "포도");

    ht_print(ht);

    // 2. 검색 테스트
    printf("\n[ 2단계: 검색 테스트 ]\n");
    const char *keys[] = {"apple", "grape", "kiwi", "banana"};
    for (int i = 0; i < 4; i++) {
        char *value = ht_get(ht, keys[i]);
        if (value) {
            printf("✓ '%s' → '%s'\n", keys[i], value);
        } else {
            printf("✗ '%s' → (찾을 수 없음)\n", keys[i]);
        }
    }

    // 3. 수정 테스트
    printf("\n[ 3단계: 수정 테스트 ]\n");
    printf("'apple'의 값을 수정합니다...\n");
    ht_set(ht, "apple", "맛있는 사과 🍎");
    printf("수정 후: %s\n", ht_get(ht, "apple"));

    // 4. 삭제 테스트
    printf("\n[ 4단계: 삭제 테스트 ]\n");
    printf("'banana'를 삭제합니다...\n");

    if (ht_delete(ht, "banana")) {
        printf("✓ 삭제 성공\n");
    }

    printf("삭제 확인: %s\n", ht_get(ht, "banana") ?: "(찾을 수 없음)");

    ht_print(ht);

    // 5. 충돌 유발 테스트
    printf("\n[ 5단계: 충돌 및 클러스터링 테스트 ]\n");
    printf("추가 데이터를 삽입합니다...\n");

    ht_set(ht, "honeydew", "허니듀");
    ht_set(ht, "kiwi", "키위");
    ht_set(ht, "lemon", "레몬");
    ht_set(ht, "mango", "망고");

    ht_print(ht);

    // 6. 클러스터링 분석
    analyze_clustering(ht);

    // 7. 삭제 후 재삽입 테스트
    printf("\n[ 6단계: 삭제 후 재삽입 테스트 ]\n");
    printf("여러 항목을 삭제한 후 새로운 항목을 삽입합니다...\n\n");

    ht_delete(ht, "cherry");
    ht_delete(ht, "fig");

    printf("삭제 후:\n");
    ht_print(ht);

    printf("\n새 항목 삽입:\n");
    ht_set(ht, "orange", "오렌지");
    ht_set(ht, "peach", "복숭아");

    ht_print(ht);

    // 정리
    ht_destroy(ht);

    // 8. 성능 테스트
    performance_test();

    printf("\n프로그램을 종료합니다.\n");
    return 0;
}
