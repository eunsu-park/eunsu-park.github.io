/*
 * 트라이 (Trie)
 * Prefix Tree, Autocomplete, XOR Trie
 *
 * 문자열 검색을 위한 트리 자료구조입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define ALPHABET_SIZE 26

/* =============================================================================
 * 1. 기본 트라이
 * ============================================================================= */

typedef struct TrieNode {
    struct TrieNode* children[ALPHABET_SIZE];
    bool is_end;
    int count;  /* 이 접두사로 시작하는 단어 수 */
} TrieNode;

TrieNode* trie_create_node(void) {
    TrieNode* node = malloc(sizeof(TrieNode));
    node->is_end = false;
    node->count = 0;
    for (int i = 0; i < ALPHABET_SIZE; i++)
        node->children[i] = NULL;
    return node;
}

void trie_free(TrieNode* node) {
    if (node == NULL) return;
    for (int i = 0; i < ALPHABET_SIZE; i++)
        trie_free(node->children[i]);
    free(node);
}

void trie_insert(TrieNode* root, const char* word) {
    TrieNode* node = root;
    while (*word) {
        int idx = *word - 'a';
        if (node->children[idx] == NULL)
            node->children[idx] = trie_create_node();
        node = node->children[idx];
        node->count++;
        word++;
    }
    node->is_end = true;
}

bool trie_search(TrieNode* root, const char* word) {
    TrieNode* node = root;
    while (*word) {
        int idx = *word - 'a';
        if (node->children[idx] == NULL)
            return false;
        node = node->children[idx];
        word++;
    }
    return node->is_end;
}

bool trie_starts_with(TrieNode* root, const char* prefix) {
    TrieNode* node = root;
    while (*prefix) {
        int idx = *prefix - 'a';
        if (node->children[idx] == NULL)
            return false;
        node = node->children[idx];
        prefix++;
    }
    return true;
}

int trie_count_prefix(TrieNode* root, const char* prefix) {
    TrieNode* node = root;
    while (*prefix) {
        int idx = *prefix - 'a';
        if (node->children[idx] == NULL)
            return 0;
        node = node->children[idx];
        prefix++;
    }
    return node->count;
}

/* =============================================================================
 * 2. 단어 삭제
 * ============================================================================= */

bool trie_delete_helper(TrieNode* node, const char* word, int depth) {
    if (node == NULL) return false;

    if (*word == '\0') {
        if (!node->is_end) return false;
        node->is_end = false;

        /* 자식이 없으면 삭제 가능 */
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            if (node->children[i]) return false;
        }
        return true;
    }

    int idx = *word - 'a';
    if (trie_delete_helper(node->children[idx], word + 1, depth + 1)) {
        free(node->children[idx]);
        node->children[idx] = NULL;

        /* 현재 노드도 삭제 가능한지 확인 */
        if (!node->is_end) {
            for (int i = 0; i < ALPHABET_SIZE; i++) {
                if (node->children[i]) return false;
            }
            return true;
        }
    }

    return false;
}

void trie_delete(TrieNode* root, const char* word) {
    trie_delete_helper(root, word, 0);
}

/* =============================================================================
 * 3. 자동완성
 * ============================================================================= */

void autocomplete_helper(TrieNode* node, char* prefix, int prefix_len,
                         char** results, int* count, int max_results) {
    if (*count >= max_results) return;

    if (node->is_end) {
        results[*count] = malloc(prefix_len + 1);
        strcpy(results[*count], prefix);
        (*count)++;
    }

    for (int i = 0; i < ALPHABET_SIZE; i++) {
        if (node->children[i]) {
            prefix[prefix_len] = 'a' + i;
            prefix[prefix_len + 1] = '\0';
            autocomplete_helper(node->children[i], prefix, prefix_len + 1,
                               results, count, max_results);
        }
    }
}

char** autocomplete(TrieNode* root, const char* prefix, int* result_count, int max_results) {
    char** results = malloc(max_results * sizeof(char*));
    *result_count = 0;

    /* 접두사 노드 찾기 */
    TrieNode* node = root;
    char* current_prefix = malloc(100);
    strcpy(current_prefix, prefix);
    int prefix_len = strlen(prefix);

    while (*prefix) {
        int idx = *prefix - 'a';
        if (node->children[idx] == NULL) {
            free(current_prefix);
            return results;
        }
        node = node->children[idx];
        prefix++;
    }

    autocomplete_helper(node, current_prefix, prefix_len, results, result_count, max_results);
    free(current_prefix);
    return results;
}

/* =============================================================================
 * 4. 와일드카드 검색
 * ============================================================================= */

bool wildcard_search_helper(TrieNode* node, const char* word) {
    if (*word == '\0')
        return node->is_end;

    if (*word == '.') {
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            if (node->children[i] && wildcard_search_helper(node->children[i], word + 1))
                return true;
        }
        return false;
    }

    int idx = *word - 'a';
    if (node->children[idx] == NULL)
        return false;
    return wildcard_search_helper(node->children[idx], word + 1);
}

bool wildcard_search(TrieNode* root, const char* pattern) {
    return wildcard_search_helper(root, pattern);
}

/* =============================================================================
 * 5. XOR 트라이 (비트 트라이)
 * ============================================================================= */

typedef struct XORTrieNode {
    struct XORTrieNode* children[2];
} XORTrieNode;

XORTrieNode* xor_trie_create_node(void) {
    XORTrieNode* node = malloc(sizeof(XORTrieNode));
    node->children[0] = NULL;
    node->children[1] = NULL;
    return node;
}

void xor_trie_free(XORTrieNode* node) {
    if (node == NULL) return;
    xor_trie_free(node->children[0]);
    xor_trie_free(node->children[1]);
    free(node);
}

void xor_trie_insert(XORTrieNode* root, int num) {
    XORTrieNode* node = root;
    for (int i = 31; i >= 0; i--) {
        int bit = (num >> i) & 1;
        if (node->children[bit] == NULL)
            node->children[bit] = xor_trie_create_node();
        node = node->children[bit];
    }
}

int xor_trie_max_xor(XORTrieNode* root, int num) {
    XORTrieNode* node = root;
    int result = 0;

    for (int i = 31; i >= 0; i--) {
        int bit = (num >> i) & 1;
        int want = 1 - bit;  /* 반대 비트 선호 */

        if (node->children[want]) {
            result |= (1 << i);
            node = node->children[want];
        } else if (node->children[bit]) {
            node = node->children[bit];
        } else {
            break;
        }
    }

    return result;
}

int find_max_xor_pair(int arr[], int n) {
    XORTrieNode* root = xor_trie_create_node();
    int max_xor = 0;

    xor_trie_insert(root, arr[0]);

    for (int i = 1; i < n; i++) {
        int xor_val = xor_trie_max_xor(root, arr[i]);
        if (xor_val > max_xor) max_xor = xor_val;
        xor_trie_insert(root, arr[i]);
    }

    xor_trie_free(root);
    return max_xor;
}

/* =============================================================================
 * 6. 최장 공통 접두사
 * ============================================================================= */

char* longest_common_prefix(char* strs[], int n) {
    if (n == 0) return "";

    TrieNode* root = trie_create_node();

    /* 첫 번째 문자열만 삽입 */
    trie_insert(root, strs[0]);

    char* lcp = malloc(strlen(strs[0]) + 1);
    int lcp_len = 0;

    TrieNode* node = root;
    const char* first = strs[0];

    while (*first) {
        int idx = *first - 'a';

        /* 분기가 있거나 단어 끝이면 중단 */
        int child_count = 0;
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            if (node->children[i]) child_count++;
        }

        if (child_count != 1 || node->is_end)
            break;

        /* 다른 문자열에서 이 접두사 확인 */
        bool all_match = true;
        for (int i = 1; i < n; i++) {
            if (strs[i][lcp_len] != *first) {
                all_match = false;
                break;
            }
        }

        if (!all_match) break;

        lcp[lcp_len++] = *first;
        node = node->children[idx];
        first++;
    }

    lcp[lcp_len] = '\0';
    trie_free(root);
    return lcp;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("트라이 (Trie) 예제\n");
    printf("============================================================\n");

    /* 1. 기본 트라이 */
    printf("\n[1] 기본 트라이 연산\n");
    TrieNode* trie = trie_create_node();

    const char* words[] = {"apple", "app", "apricot", "banana", "band"};
    printf("    삽입: apple, app, apricot, banana, band\n");
    for (int i = 0; i < 5; i++)
        trie_insert(trie, words[i]);

    printf("    search('app'): %s\n", trie_search(trie, "app") ? "true" : "false");
    printf("    search('apt'): %s\n", trie_search(trie, "apt") ? "true" : "false");
    printf("    startsWith('ap'): %s\n", trie_starts_with(trie, "ap") ? "true" : "false");
    printf("    countPrefix('ap'): %d\n", trie_count_prefix(trie, "ap"));

    /* 2. 삭제 */
    printf("\n[2] 단어 삭제\n");
    printf("    delete('app')\n");
    trie_delete(trie, "app");
    printf("    search('app'): %s\n", trie_search(trie, "app") ? "true" : "false");
    printf("    search('apple'): %s\n", trie_search(trie, "apple") ? "true" : "false");

    /* 3. 자동완성 */
    printf("\n[3] 자동완성\n");
    int result_count;
    char** suggestions = autocomplete(trie, "ap", &result_count, 10);
    printf("    'ap'로 시작하는 단어:\n");
    for (int i = 0; i < result_count; i++) {
        printf("      - %s\n", suggestions[i]);
        free(suggestions[i]);
    }
    free(suggestions);

    /* 4. 와일드카드 */
    printf("\n[4] 와일드카드 검색\n");
    printf("    search('b.nd'): %s\n", wildcard_search(trie, "b.nd") ? "true" : "false");
    printf("    search('b..d'): %s\n", wildcard_search(trie, "b..d") ? "true" : "false");
    printf("    search('.pple'): %s\n", wildcard_search(trie, ".pple") ? "true" : "false");

    trie_free(trie);

    /* 5. XOR 트라이 */
    printf("\n[5] XOR 트라이 - 최대 XOR 쌍\n");
    int arr[] = {3, 10, 5, 25, 2, 8};
    printf("    배열: [3, 10, 5, 25, 2, 8]\n");
    printf("    최대 XOR: %d\n", find_max_xor_pair(arr, 6));

    /* 6. 최장 공통 접두사 */
    printf("\n[6] 최장 공통 접두사\n");
    char* strs[] = {"flower", "flow", "flight"};
    char* lcp = longest_common_prefix(strs, 3);
    printf("    [\"flower\", \"flow\", \"flight\"]\n");
    printf("    LCP: '%s'\n", lcp);
    free(lcp);

    /* 7. 복잡도 */
    printf("\n[7] 트라이 복잡도 (m = 문자열 길이)\n");
    printf("    | 연산         | 시간복잡도 |\n");
    printf("    |--------------|------------|\n");
    printf("    | 삽입         | O(m)       |\n");
    printf("    | 검색         | O(m)       |\n");
    printf("    | 접두사 검색  | O(m)       |\n");
    printf("    | 삭제         | O(m)       |\n");
    printf("    | 공간         | O(n * m)   |\n");

    printf("\n============================================================\n");

    return 0;
}
