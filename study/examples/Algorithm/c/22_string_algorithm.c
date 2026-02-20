/*
 * 문자열 알고리즘 (String Algorithms)
 * KMP, Rabin-Karp, Z-알고리즘, 매나커
 *
 * 패턴 매칭과 문자열 처리 알고리즘입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 100001
#define MOD 1000000007
#define BASE 31

/* =============================================================================
 * 1. KMP (Knuth-Morris-Pratt)
 * ============================================================================= */

/* 실패 함수 (부분 일치 테이블) 계산 */
int* compute_failure(const char* pattern) {
    int m = strlen(pattern);
    int* fail = calloc(m, sizeof(int));

    int j = 0;
    for (int i = 1; i < m; i++) {
        while (j > 0 && pattern[i] != pattern[j]) {
            j = fail[j - 1];
        }
        if (pattern[i] == pattern[j]) {
            fail[i] = ++j;
        }
    }

    return fail;
}

/* KMP 검색 - 모든 매칭 위치 반환 */
int* kmp_search(const char* text, const char* pattern, int* match_count) {
    int n = strlen(text);
    int m = strlen(pattern);
    int* fail = compute_failure(pattern);
    int* matches = malloc(n * sizeof(int));
    *match_count = 0;

    int j = 0;
    for (int i = 0; i < n; i++) {
        while (j > 0 && text[i] != pattern[j]) {
            j = fail[j - 1];
        }
        if (text[i] == pattern[j]) {
            j++;
            if (j == m) {
                matches[(*match_count)++] = i - m + 1;
                j = fail[j - 1];
            }
        }
    }

    free(fail);
    return matches;
}

/* =============================================================================
 * 2. Rabin-Karp
 * ============================================================================= */

long long compute_hash(const char* s, int len) {
    long long hash = 0;
    long long power = 1;
    for (int i = 0; i < len; i++) {
        hash = (hash + (s[i] - 'a' + 1) * power) % MOD;
        power = (power * BASE) % MOD;
    }
    return hash;
}

int* rabin_karp(const char* text, const char* pattern, int* match_count) {
    int n = strlen(text);
    int m = strlen(pattern);
    int* matches = malloc(n * sizeof(int));
    *match_count = 0;

    if (m > n) return matches;

    long long pattern_hash = compute_hash(pattern, m);
    long long text_hash = compute_hash(text, m);
    long long power = 1;

    for (int i = 0; i < m - 1; i++) {
        power = (power * BASE) % MOD;
    }

    for (int i = 0; i <= n - m; i++) {
        if (text_hash == pattern_hash) {
            /* 해시 충돌 확인 */
            int match = 1;
            for (int j = 0; j < m; j++) {
                if (text[i + j] != pattern[j]) {
                    match = 0;
                    break;
                }
            }
            if (match) matches[(*match_count)++] = i;
        }

        if (i < n - m) {
            /* 롤링 해시 */
            text_hash = (text_hash - (text[i] - 'a' + 1) + MOD) % MOD;
            text_hash = (text_hash * mod_inv(BASE, MOD)) % MOD;
            text_hash = (text_hash + (text[i + m] - 'a' + 1) * power) % MOD;
        }
    }

    return matches;
}

/* 모듈러 역원 (간단 버전) */
long long mod_inv(long long a, long long mod) {
    long long result = 1;
    long long exp = mod - 2;
    while (exp > 0) {
        if (exp & 1) result = (result * a) % mod;
        a = (a * a) % mod;
        exp >>= 1;
    }
    return result;
}

/* 간단한 Rabin-Karp (롤링 해시 개선) */
int* rabin_karp_simple(const char* text, const char* pattern, int* match_count) {
    int n = strlen(text);
    int m = strlen(pattern);
    int* matches = malloc(n * sizeof(int));
    *match_count = 0;

    if (m > n) return matches;

    long long pattern_hash = 0;
    long long text_hash = 0;
    long long power = 1;

    /* 초기 해시 계산 */
    for (int i = 0; i < m; i++) {
        pattern_hash = (pattern_hash * BASE + pattern[i]) % MOD;
        text_hash = (text_hash * BASE + text[i]) % MOD;
        if (i < m - 1) power = (power * BASE) % MOD;
    }

    for (int i = 0; i <= n - m; i++) {
        if (text_hash == pattern_hash) {
            if (strncmp(text + i, pattern, m) == 0) {
                matches[(*match_count)++] = i;
            }
        }
        if (i < n - m) {
            text_hash = ((text_hash - text[i] * power % MOD + MOD) * BASE + text[i + m]) % MOD;
        }
    }

    return matches;
}

/* =============================================================================
 * 3. Z-알고리즘
 * ============================================================================= */

int* z_function(const char* s) {
    int n = strlen(s);
    int* z = calloc(n, sizeof(int));
    int l = 0, r = 0;

    for (int i = 1; i < n; i++) {
        if (i < r) {
            z[i] = (z[i - l] < r - i) ? z[i - l] : r - i;
        }
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        if (i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }

    return z;
}

/* Z-알고리즘으로 패턴 매칭 */
int* z_search(const char* text, const char* pattern, int* match_count) {
    int n = strlen(text);
    int m = strlen(pattern);

    /* pattern + "$" + text 연결 */
    char* combined = malloc(n + m + 2);
    sprintf(combined, "%s$%s", pattern, text);

    int* z = z_function(combined);
    int* matches = malloc(n * sizeof(int));
    *match_count = 0;

    for (int i = m + 1; i < n + m + 1; i++) {
        if (z[i] == m) {
            matches[(*match_count)++] = i - m - 1;
        }
    }

    free(combined);
    free(z);
    return matches;
}

/* =============================================================================
 * 4. 매나커 알고리즘 (Manacher)
 * ============================================================================= */

/* 가장 긴 팰린드롬 부분문자열 */
char* manacher(const char* s) {
    int n = strlen(s);
    if (n == 0) return strdup("");

    /* 문자 사이에 # 삽입 */
    int len = 2 * n + 1;
    char* t = malloc(len + 1);
    for (int i = 0; i < n; i++) {
        t[2 * i] = '#';
        t[2 * i + 1] = s[i];
    }
    t[len - 1] = '#';
    t[len] = '\0';

    int* p = calloc(len, sizeof(int));
    int center = 0, right = 0;
    int max_len = 0, max_center = 0;

    for (int i = 0; i < len; i++) {
        if (i < right) {
            int mirror = 2 * center - i;
            p[i] = (p[mirror] < right - i) ? p[mirror] : right - i;
        }

        while (i - p[i] - 1 >= 0 && i + p[i] + 1 < len &&
               t[i - p[i] - 1] == t[i + p[i] + 1]) {
            p[i]++;
        }

        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }

        if (p[i] > max_len) {
            max_len = p[i];
            max_center = i;
        }
    }

    /* 원본 문자열에서 추출 */
    int start = (max_center - max_len) / 2;
    char* result = malloc(max_len + 1);
    strncpy(result, s + start, max_len);
    result[max_len] = '\0';

    free(t);
    free(p);
    return result;
}

/* =============================================================================
 * 5. 문자열 해싱
 * ============================================================================= */

typedef struct {
    long long* prefix_hash;
    long long* power;
    int len;
} StringHash;

StringHash* create_string_hash(const char* s) {
    int n = strlen(s);
    StringHash* sh = malloc(sizeof(StringHash));
    sh->prefix_hash = malloc((n + 1) * sizeof(long long));
    sh->power = malloc((n + 1) * sizeof(long long));
    sh->len = n;

    sh->prefix_hash[0] = 0;
    sh->power[0] = 1;

    for (int i = 0; i < n; i++) {
        sh->prefix_hash[i + 1] = (sh->prefix_hash[i] * BASE + s[i]) % MOD;
        sh->power[i + 1] = (sh->power[i] * BASE) % MOD;
    }

    return sh;
}

long long get_hash(StringHash* sh, int l, int r) {
    long long hash = (sh->prefix_hash[r + 1] -
                      sh->prefix_hash[l] * sh->power[r - l + 1] % MOD + MOD) % MOD;
    return hash;
}

void free_string_hash(StringHash* sh) {
    free(sh->prefix_hash);
    free(sh->power);
    free(sh);
}

/* =============================================================================
 * 6. 유용한 문자열 함수들
 * ============================================================================= */

/* 접미사 배열 (간단 버전, O(n² log n)) */
int* suffix_array_simple(const char* s) {
    int n = strlen(s);
    int* sa = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) sa[i] = i;

    /* 정렬 */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (strcmp(s + sa[i], s + sa[j]) > 0) {
                int temp = sa[i];
                sa[i] = sa[j];
                sa[j] = temp;
            }
        }
    }

    return sa;
}

/* LCP 배열 (Kasai 알고리즘) */
int* lcp_array(const char* s, int* sa) {
    int n = strlen(s);
    int* rank = malloc(n * sizeof(int));
    int* lcp = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) rank[sa[i]] = i;

    int k = 0;
    for (int i = 0; i < n; i++) {
        if (rank[i] == 0) {
            lcp[0] = 0;
            continue;
        }
        int j = sa[rank[i] - 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) k++;
        lcp[rank[i]] = k;
        if (k > 0) k--;
    }

    free(rank);
    return lcp;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("문자열 알고리즘 예제\n");
    printf("============================================================\n");

    /* 1. KMP */
    printf("\n[1] KMP 알고리즘\n");
    const char* text1 = "ABABDABACDABABCABAB";
    const char* pattern1 = "ABABCABAB";
    int match_count;
    int* matches = kmp_search(text1, pattern1, &match_count);
    printf("    텍스트: %s\n", text1);
    printf("    패턴: %s\n", pattern1);
    printf("    매칭 위치 (%d개): ", match_count);
    for (int i = 0; i < match_count; i++) printf("%d ", matches[i]);
    printf("\n");
    free(matches);

    /* 실패 함수 시각화 */
    const char* pattern2 = "ABAABAB";
    int* fail = compute_failure(pattern2);
    printf("    패턴 \"%s\"의 실패 함수: ", pattern2);
    for (int i = 0; i < (int)strlen(pattern2); i++) printf("%d ", fail[i]);
    printf("\n");
    free(fail);

    /* 2. Z-알고리즘 */
    printf("\n[2] Z-알고리즘\n");
    const char* str = "aabxaabxcaabxaabxay";
    int* z = z_function(str);
    printf("    문자열: %s\n", str);
    printf("    Z-배열: ");
    for (int i = 0; i < (int)strlen(str); i++) printf("%d ", z[i]);
    printf("\n");
    free(z);

    matches = z_search(text1, pattern1, &match_count);
    printf("    Z-알고리즘 매칭 위치: ");
    for (int i = 0; i < match_count; i++) printf("%d ", matches[i]);
    printf("\n");
    free(matches);

    /* 3. Rabin-Karp */
    printf("\n[3] Rabin-Karp\n");
    const char* text2 = "abracadabra";
    const char* pattern3 = "abra";
    matches = rabin_karp_simple(text2, pattern3, &match_count);
    printf("    텍스트: %s\n", text2);
    printf("    패턴: %s\n", pattern3);
    printf("    매칭 위치: ");
    for (int i = 0; i < match_count; i++) printf("%d ", matches[i]);
    printf("\n");
    free(matches);

    /* 4. 매나커 */
    printf("\n[4] 매나커 알고리즘\n");
    const char* str2 = "babad";
    char* palindrome = manacher(str2);
    printf("    문자열: %s\n", str2);
    printf("    가장 긴 팰린드롬: %s\n", palindrome);
    free(palindrome);

    const char* str3 = "forgeeksskeegfor";
    palindrome = manacher(str3);
    printf("    문자열: %s\n", str3);
    printf("    가장 긴 팰린드롬: %s\n", palindrome);
    free(palindrome);

    /* 5. 문자열 해싱 */
    printf("\n[5] 문자열 해싱\n");
    const char* str4 = "abcabc";
    StringHash* sh = create_string_hash(str4);
    printf("    문자열: %s\n", str4);
    printf("    hash[0:2] = %lld\n", get_hash(sh, 0, 2));
    printf("    hash[3:5] = %lld\n", get_hash(sh, 3, 5));
    printf("    동일 여부: %s\n",
           get_hash(sh, 0, 2) == get_hash(sh, 3, 5) ? "예" : "아니오");
    free_string_hash(sh);

    /* 6. 접미사 배열 */
    printf("\n[6] 접미사 배열\n");
    const char* str5 = "banana";
    int* sa = suffix_array_simple(str5);
    printf("    문자열: %s\n", str5);
    printf("    접미사 배열: ");
    for (int i = 0; i < (int)strlen(str5); i++) printf("%d ", sa[i]);
    printf("\n");
    printf("    정렬된 접미사:\n");
    for (int i = 0; i < (int)strlen(str5); i++) {
        printf("      %d: %s\n", sa[i], str5 + sa[i]);
    }
    free(sa);

    /* 7. 복잡도 */
    printf("\n[7] 복잡도\n");
    printf("    | 알고리즘        | 전처리    | 검색      |\n");
    printf("    |-----------------|-----------|----------|\n");
    printf("    | KMP             | O(m)      | O(n)     |\n");
    printf("    | Rabin-Karp      | O(m)      | O(n+m)*  |\n");
    printf("    | Z-알고리즘      | O(n+m)    | -        |\n");
    printf("    | 매나커          | -         | O(n)     |\n");
    printf("    * 평균, 최악 O(nm)\n");

    printf("\n============================================================\n");

    return 0;
}
