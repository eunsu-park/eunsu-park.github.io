/*
 * hash_functions.c
 * 다양한 해시 함수 비교 및 충돌 분석
 *
 * 구현된 해시 함수:
 * 1. hash_simple - 단순 합산 (충돌 많음)
 * 2. hash_djb2 - Daniel J. Bernstein (우수한 분포)
 * 3. hash_sdbm - sdbm 데이터베이스 해시
 * 4. hash_fnv1a - Fowler-Noll-Vo 1a (빠르고 좋은 분포)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define TABLE_SIZE 100
#define MAX_TEST_WORDS 50

// 1. 단순 합산 해시 (나쁜 예)
// 문제점: 같은 문자의 순열이 같은 값 생성 (예: "abc"와 "bca")
unsigned int hash_simple(const char *key) {
    unsigned int hash = 0;
    while (*key) {
        hash += (unsigned char)*key++;
    }
    return hash % TABLE_SIZE;
}

// 2. djb2 해시 (Daniel J. Bernstein) - 추천
// 장점: 단순하지만 우수한 분포 특성
unsigned int hash_djb2(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }
    return hash % TABLE_SIZE;
}

// 3. sdbm 해시
// 장점: 많은 데이터베이스에서 검증된 성능
unsigned int hash_sdbm(const char *key) {
    unsigned int hash = 0;
    int c;
    while ((c = *key++)) {
        hash = c + (hash << 6) + (hash << 16) - hash;
    }
    return hash % TABLE_SIZE;
}

// 4. FNV-1a 해시 (Fowler-Noll-Vo)
// 장점: 빠른 속도와 좋은 분포
unsigned int hash_fnv1a(const char *key) {
    unsigned int hash = 2166136261u;  // FNV offset basis
    while (*key) {
        hash ^= (unsigned char)*key++;
        hash *= 16777619;  // FNV prime
    }
    return hash % TABLE_SIZE;
}

// 충돌 카운터 구조체
typedef struct {
    int simple;
    int djb2;
    int sdbm;
    int fnv1a;
} CollisionStats;

// 충돌 분석 함수
CollisionStats analyze_collisions(const char **keys, int n) {
    CollisionStats stats = {0, 0, 0, 0};

    // 각 해시 함수별로 사용된 버킷 추적
    int buckets_simple[TABLE_SIZE] = {0};
    int buckets_djb2[TABLE_SIZE] = {0};
    int buckets_sdbm[TABLE_SIZE] = {0};
    int buckets_fnv1a[TABLE_SIZE] = {0};

    for (int i = 0; i < n; i++) {
        unsigned int idx;

        // simple
        idx = hash_simple(keys[i]);
        if (buckets_simple[idx] > 0) stats.simple++;
        buckets_simple[idx]++;

        // djb2
        idx = hash_djb2(keys[i]);
        if (buckets_djb2[idx] > 0) stats.djb2++;
        buckets_djb2[idx]++;

        // sdbm
        idx = hash_sdbm(keys[i]);
        if (buckets_sdbm[idx] > 0) stats.sdbm++;
        buckets_sdbm[idx]++;

        // fnv1a
        idx = hash_fnv1a(keys[i]);
        if (buckets_fnv1a[idx] > 0) stats.fnv1a++;
        buckets_fnv1a[idx]++;
    }

    return stats;
}

// 분포 균일성 계산 (표준편차)
double calculate_distribution(unsigned int (*hash_func)(const char*),
                             const char **keys, int n) {
    int buckets[TABLE_SIZE] = {0};

    // 각 버킷에 할당된 키 개수 카운트
    for (int i = 0; i < n; i++) {
        unsigned int idx = hash_func(keys[i]);
        buckets[idx]++;
    }

    // 평균 계산
    double mean = (double)n / TABLE_SIZE;

    // 분산 계산
    double variance = 0.0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        double diff = buckets[i] - mean;
        variance += diff * diff;
    }
    variance /= TABLE_SIZE;

    // 표준편차 반환 (낮을수록 균일한 분포)
    return variance;
}

// 테스트용 단어 목록 생성
const char** generate_test_words(int *count) {
    static const char *words[] = {
        // 과일
        "apple", "banana", "cherry", "date", "elderberry",
        "fig", "grape", "honeydew", "kiwi", "lemon",
        // 색상
        "red", "blue", "green", "yellow", "orange",
        "purple", "pink", "brown", "black", "white",
        // 동물
        "cat", "dog", "elephant", "fox", "giraffe",
        "horse", "iguana", "jaguar", "kangaroo", "lion",
        // 국가
        "korea", "japan", "china", "america", "france",
        "germany", "italy", "spain", "brazil", "india",
        // 프로그래밍
        "python", "java", "javascript", "ruby", "php",
        "swift", "kotlin", "rust", "golang", "typescript"
    };

    *count = sizeof(words) / sizeof(words[0]);
    return words;
}

void print_hash_table(const char *title, const char **keys, int n,
                     unsigned int (*hash_func)(const char*)) {
    printf("\n=== %s ===\n", title);

    // 해시 값 출력
    printf("%-15s | Hash Value\n", "Key");
    printf("----------------+-----------\n");
    for (int i = 0; i < (n < 10 ? n : 10); i++) {  // 처음 10개만
        printf("%-15s | %10u\n", keys[i], hash_func(keys[i]));
    }
    if (n > 10) printf("... (%d more)\n", n - 10);
}

int main(void) {
    int n;
    const char **test_words = generate_test_words(&n);

    printf("╔════════════════════════════════════════════╗\n");
    printf("║     해시 함수 비교 및 충돌 분석 도구      ║\n");
    printf("╚════════════════════════════════════════════╝\n");
    printf("\n테스트 단어 개수: %d\n", n);
    printf("해시 테이블 크기: %d\n\n", TABLE_SIZE);

    // 1. 샘플 단어들의 해시 값 비교
    const char *sample_keys[] = {"apple", "banana", "cherry", "date", "elderberry"};
    int sample_n = sizeof(sample_keys) / sizeof(sample_keys[0]);

    printf("=== 샘플 단어 해시 값 비교 ===\n\n");
    printf("%-12s | Simple | djb2 | sdbm | fnv1a\n", "Key");
    printf("-------------|--------|------|------|------\n");

    for (int i = 0; i < sample_n; i++) {
        printf("%-12s | %6u | %4u | %4u | %5u\n",
               sample_keys[i],
               hash_simple(sample_keys[i]),
               hash_djb2(sample_keys[i]),
               hash_sdbm(sample_keys[i]),
               hash_fnv1a(sample_keys[i]));
    }

    // 2. 충돌 분석
    printf("\n=== 충돌 분석 (총 %d개 단어) ===\n\n", n);
    CollisionStats stats = analyze_collisions(test_words, n);

    printf("해시 함수    | 충돌 횟수 | 충돌률\n");
    printf("-------------|-----------|--------\n");
    printf("Simple       | %9d | %5.1f%%\n", stats.simple,
           100.0 * stats.simple / n);
    printf("djb2         | %9d | %5.1f%%\n", stats.djb2,
           100.0 * stats.djb2 / n);
    printf("sdbm         | %9d | %5.1f%%\n", stats.sdbm,
           100.0 * stats.sdbm / n);
    printf("FNV-1a       | %9d | %5.1f%%\n", stats.fnv1a,
           100.0 * stats.fnv1a / n);

    // 3. 분포 균일성 분석
    printf("\n=== 분포 균일성 분석 (분산) ===\n");
    printf("※ 낮을수록 균일한 분포\n\n");

    double var_simple = calculate_distribution(hash_simple, test_words, n);
    double var_djb2 = calculate_distribution(hash_djb2, test_words, n);
    double var_sdbm = calculate_distribution(hash_sdbm, test_words, n);
    double var_fnv1a = calculate_distribution(hash_fnv1a, test_words, n);

    printf("해시 함수    | 분산 값\n");
    printf("-------------|----------\n");
    printf("Simple       | %8.2f\n", var_simple);
    printf("djb2         | %8.2f ← 추천\n", var_djb2);
    printf("sdbm         | %8.2f\n", var_sdbm);
    printf("FNV-1a       | %8.2f\n", var_fnv1a);

    // 4. 성능 권장사항
    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║              권장 해시 함수                ║\n");
    printf("╠════════════════════════════════════════════╣\n");
    printf("║  1. djb2   - 일반적인 용도 (균형있음)     ║\n");
    printf("║  2. FNV-1a - 빠른 속도 필요시              ║\n");
    printf("║  3. sdbm   - 데이터베이스 용도             ║\n");
    printf("║                                            ║\n");
    printf("║  ⚠️  Simple은 사용하지 마세요!             ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    // 5. 실제 분포 시각화 (djb2)
    printf("\n=== djb2 해시 분포 시각화 ===\n");
    printf("(각 '*'는 하나의 키를 나타냄)\n\n");

    int buckets[TABLE_SIZE] = {0};
    for (int i = 0; i < n; i++) {
        unsigned int idx = hash_djb2(test_words[i]);
        buckets[idx]++;
    }

    // 처음 20개 버킷만 시각화
    for (int i = 0; i < 20; i++) {
        printf("[%2d] ", i);
        for (int j = 0; j < buckets[i]; j++) {
            printf("*");
        }
        printf(" (%d)\n", buckets[i]);
    }
    printf("...\n");

    return 0;
}
