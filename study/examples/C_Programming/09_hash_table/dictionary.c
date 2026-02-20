/*
 * dictionary.c
 * 해시 테이블을 활용한 실용적인 사전(Dictionary) 프로그램
 *
 * 기능:
 * - 단어 추가/검색/삭제
 * - 전체 목록 출력
 * - 파일 저장/불러오기
 * - 단어 통계 및 검색 제안
 * - 대소문자 구분 없는 검색
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#define TABLE_SIZE 1000
#define KEY_SIZE 100
#define VALUE_SIZE 500
#define FILENAME "dictionary.txt"

// 노드 구조체 (체이닝 방식)
typedef struct Node {
    char word[KEY_SIZE];
    char meaning[VALUE_SIZE];
    int search_count;       // 검색 횟수
    struct Node *next;
} Node;

// 사전 구조체
typedef struct {
    Node *buckets[TABLE_SIZE];
    int count;
    int total_searches;
} Dictionary;

// 통계 구조체
typedef struct {
    char word[KEY_SIZE];
    int count;
} WordStat;

// 대소문자 구분 없는 djb2 해시 함수
unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    while (*key) {
        hash = ((hash << 5) + hash) + tolower((unsigned char)*key++);
    }
    return hash % TABLE_SIZE;
}

// 사전 생성
Dictionary* dict_create(void) {
    Dictionary *dict = calloc(1, sizeof(Dictionary));
    if (!dict) {
        fprintf(stderr, "메모리 할당 실패\n");
    }
    return dict;
}

// 사전 해제
void dict_destroy(Dictionary *dict) {
    if (!dict) return;

    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            Node *next = current->next;
            free(current);
            current = next;
        }
    }
    free(dict);
}

// 단어 추가 또는 수정
void dict_add(Dictionary *dict, const char *word, const char *meaning) {
    if (!dict || !word || !meaning) return;

    unsigned int index = hash(word);

    // 기존 단어 확인
    Node *current = dict->buckets[index];
    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            // 기존 단어 수정
            strncpy(current->meaning, meaning, VALUE_SIZE - 1);
            current->meaning[VALUE_SIZE - 1] = '\0';
            printf("✓ '%s' 업데이트됨\n", word);
            return;
        }
        current = current->next;
    }

    // 새 단어 추가
    Node *node = malloc(sizeof(Node));
    if (!node) {
        fprintf(stderr, "메모리 할당 실패\n");
        return;
    }

    strncpy(node->word, word, KEY_SIZE - 1);
    node->word[KEY_SIZE - 1] = '\0';
    strncpy(node->meaning, meaning, VALUE_SIZE - 1);
    node->meaning[VALUE_SIZE - 1] = '\0';
    node->search_count = 0;

    node->next = dict->buckets[index];
    dict->buckets[index] = node;
    dict->count++;

    printf("✓ '%s' 추가됨\n", word);
}

// 단어 검색
char* dict_search(Dictionary *dict, const char *word) {
    if (!dict || !word) return NULL;

    unsigned int index = hash(word);

    Node *current = dict->buckets[index];
    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            current->search_count++;
            dict->total_searches++;
            return current->meaning;
        }
        current = current->next;
    }

    return NULL;
}

// 단어 삭제
bool dict_delete(Dictionary *dict, const char *word) {
    if (!dict || !word) return false;

    unsigned int index = hash(word);

    Node *current = dict->buckets[index];
    Node *prev = NULL;

    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            if (prev) {
                prev->next = current->next;
            } else {
                dict->buckets[index] = current->next;
            }
            free(current);
            dict->count--;
            printf("✓ '%s' 삭제됨\n", word);
            return true;
        }
        prev = current;
        current = current->next;
    }

    printf("✗ '%s'을(를) 찾을 수 없습니다\n", word);
    return false;
}

// 전체 단어 목록 출력
void dict_list(Dictionary *dict) {
    if (!dict) return;

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║           사전 목록 (총 %d개)            ║\n", dict->count);
    printf("╚════════════════════════════════════════════╝\n\n");

    if (dict->count == 0) {
        printf("  (비어있음)\n");
        return;
    }

    int num = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            printf("  %3d. %-20s : %s\n",
                   ++num, current->word, current->meaning);
            current = current->next;
        }
    }
}

// 파일에 저장
bool dict_save(Dictionary *dict, const char *filename) {
    if (!dict || !filename) return false;

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "파일 열기 실패: %s\n", filename);
        return false;
    }

    // 헤더 작성
    fprintf(fp, "# Dictionary File\n");
    fprintf(fp, "# Count: %d\n\n", dict->count);

    // 모든 단어 저장
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            fprintf(fp, "%s|%s|%d\n",
                   current->word, current->meaning, current->search_count);
            current = current->next;
        }
    }

    fclose(fp);
    printf("✓ %d개 단어를 '%s'에 저장했습니다\n", dict->count, filename);
    return true;
}

// 파일에서 불러오기
bool dict_load(Dictionary *dict, const char *filename) {
    if (!dict || !filename) return false;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "파일 열기 실패: %s\n", filename);
        return false;
    }

    char line[KEY_SIZE + VALUE_SIZE + 50];
    int loaded = 0;

    while (fgets(line, sizeof(line), fp)) {
        // 주석 및 빈 줄 건너뛰기
        if (line[0] == '#' || line[0] == '\n') continue;

        // 줄바꿈 제거
        line[strcspn(line, "\n")] = '\0';

        // 파싱: word|meaning|search_count
        char word[KEY_SIZE], meaning[VALUE_SIZE];
        int search_count = 0;

        char *token = strtok(line, "|");
        if (token) strncpy(word, token, KEY_SIZE - 1);

        token = strtok(NULL, "|");
        if (token) strncpy(meaning, token, VALUE_SIZE - 1);

        token = strtok(NULL, "|");
        if (token) search_count = atoi(token);

        // 사전에 추가 (출력 없이)
        unsigned int index = hash(word);
        Node *node = malloc(sizeof(Node));
        if (!node) continue;

        strncpy(node->word, word, KEY_SIZE - 1);
        node->word[KEY_SIZE - 1] = '\0';
        strncpy(node->meaning, meaning, VALUE_SIZE - 1);
        node->meaning[VALUE_SIZE - 1] = '\0';
        node->search_count = search_count;

        node->next = dict->buckets[index];
        dict->buckets[index] = node;
        dict->count++;
        loaded++;
    }

    fclose(fp);
    printf("✓ %d개 단어를 '%s'에서 불러왔습니다\n", loaded, filename);
    return true;
}

// 검색 제안 (부분 일치)
void dict_suggest(Dictionary *dict, const char *prefix) {
    if (!dict || !prefix) return;

    printf("\n'%s'로 시작하는 단어:\n", prefix);

    int found = 0;
    int len = strlen(prefix);

    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            if (strncasecmp(current->word, prefix, len) == 0) {
                printf("  - %s\n", current->word);
                found++;
            }
            current = current->next;
        }
    }

    if (found == 0) {
        printf("  (없음)\n");
    } else {
        printf("총 %d개 발견\n", found);
    }
}

// 인기 단어 통계
void dict_statistics(Dictionary *dict) {
    if (!dict) return;

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║              사전 통계 정보                ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    printf("총 단어 개수:     %d\n", dict->count);
    printf("총 검색 횟수:     %d\n", dict->total_searches);

    // 검색 횟수 순으로 정렬 (Top 10)
    WordStat *stats = malloc(sizeof(WordStat) * dict->count);
    if (!stats) return;

    int idx = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            strncpy(stats[idx].word, current->word, KEY_SIZE - 1);
            stats[idx].count = current->search_count;
            idx++;
            current = current->next;
        }
    }

    // 버블 정렬 (간단하게)
    for (int i = 0; i < dict->count - 1; i++) {
        for (int j = 0; j < dict->count - i - 1; j++) {
            if (stats[j].count < stats[j + 1].count) {
                WordStat temp = stats[j];
                stats[j] = stats[j + 1];
                stats[j + 1] = temp;
            }
        }
    }

    // Top 10 출력
    printf("\n인기 단어 Top 10:\n");
    int limit = dict->count < 10 ? dict->count : 10;
    for (int i = 0; i < limit; i++) {
        if (stats[i].count > 0) {
            printf("  %2d. %-20s (%d회)\n",
                   i + 1, stats[i].word, stats[i].count);
        }
    }

    free(stats);
}

// 메뉴 출력
void print_menu(void) {
    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║           📖 간단한 사전 프로그램          ║\n");
    printf("╠════════════════════════════════════════════╣\n");
    printf("║  1. 단어 추가                              ║\n");
    printf("║  2. 단어 검색                              ║\n");
    printf("║  3. 단어 삭제                              ║\n");
    printf("║  4. 전체 목록                              ║\n");
    printf("║  5. 검색 제안                              ║\n");
    printf("║  6. 통계 보기                              ║\n");
    printf("║  7. 파일 저장                              ║\n");
    printf("║  8. 파일 불러오기                          ║\n");
    printf("║  0. 종료                                   ║\n");
    printf("╚════════════════════════════════════════════╝\n");
}

// 입력 버퍼 비우기
void clear_input(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// 샘플 데이터 로드
void load_sample_data(Dictionary *dict) {
    dict_add(dict, "apple", "사과; 장미과의 낙엽교목");
    dict_add(dict, "book", "책; 인쇄물을 제본한 것");
    dict_add(dict, "computer", "컴퓨터; 전자 계산기");
    dict_add(dict, "dictionary", "사전; 단어를 모아 일정한 순서로 배열하여 설명한 책");
    dict_add(dict, "education", "교육; 지식과 기술을 가르침");
    dict_add(dict, "friend", "친구; 가까이 사귀어 친하게 지내는 사람");
    dict_add(dict, "galaxy", "은하; 우주 공간에 있는 천체 집단");
    dict_add(dict, "happiness", "행복; 복된 좋은 운수");
    dict_add(dict, "internet", "인터넷; 전 세계의 컴퓨터가 서로 연결된 네트워크");
    dict_add(dict, "javascript", "자바스크립트; 웹 프로그래밍 언어");
}

// 메인 함수
int main(void) {
    Dictionary *dict = dict_create();
    if (!dict) return 1;

    // 샘플 데이터 로드
    printf("샘플 데이터를 불러오는 중...\n");
    load_sample_data(dict);

    // 기존 파일이 있으면 불러오기
    FILE *test = fopen(FILENAME, "r");
    if (test) {
        fclose(test);
        printf("\n기존 사전 파일을 발견했습니다.\n");
        printf("불러오시겠습니까? (y/n): ");
        char choice;
        scanf(" %c", &choice);
        clear_input();

        if (choice == 'y' || choice == 'Y') {
            // 기존 데이터 삭제 후 로드
            dict_destroy(dict);
            dict = dict_create();
            dict_load(dict, FILENAME);
        }
    }

    int choice;
    char word[KEY_SIZE];
    char meaning[VALUE_SIZE];

    while (1) {
        print_menu();
        printf("선택: ");

        if (scanf("%d", &choice) != 1) {
            clear_input();
            printf("✗ 잘못된 입력입니다\n");
            continue;
        }
        clear_input();

        switch (choice) {
            case 1:  // 추가
                printf("\n단어: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                if (strlen(word) == 0) {
                    printf("✗ 단어를 입력하세요\n");
                    break;
                }

                printf("뜻: ");
                fgets(meaning, VALUE_SIZE, stdin);
                meaning[strcspn(meaning, "\n")] = '\0';

                if (strlen(meaning) == 0) {
                    printf("✗ 뜻을 입력하세요\n");
                    break;
                }

                dict_add(dict, word, meaning);
                break;

            case 2:  // 검색
                printf("\n검색할 단어: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                char *result = dict_search(dict, word);
                if (result) {
                    printf("\n┌────────────────────────────────────────┐\n");
                    printf("│ %s\n", word);
                    printf("├────────────────────────────────────────┤\n");
                    printf("│ %s\n", result);
                    printf("└────────────────────────────────────────┘\n");
                } else {
                    printf("\n✗ '%s'을(를) 찾을 수 없습니다\n", word);
                    dict_suggest(dict, word);
                }
                break;

            case 3:  // 삭제
                printf("\n삭제할 단어: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                dict_delete(dict, word);
                break;

            case 4:  // 목록
                dict_list(dict);
                break;

            case 5:  // 제안
                printf("\n검색할 접두사: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                dict_suggest(dict, word);
                break;

            case 6:  // 통계
                dict_statistics(dict);
                break;

            case 7:  // 저장
                dict_save(dict, FILENAME);
                break;

            case 8:  // 불러오기
                printf("\n현재 데이터가 삭제됩니다. 계속하시겠습니까? (y/n): ");
                char confirm;
                scanf(" %c", &confirm);
                clear_input();

                if (confirm == 'y' || confirm == 'Y') {
                    dict_destroy(dict);
                    dict = dict_create();
                    dict_load(dict, FILENAME);
                }
                break;

            case 0:  // 종료
                printf("\n저장하시겠습니까? (y/n): ");
                char save_choice;
                scanf(" %c", &save_choice);
                clear_input();

                if (save_choice == 'y' || save_choice == 'Y') {
                    dict_save(dict, FILENAME);
                }

                printf("사전을 종료합니다.\n");
                dict_destroy(dict);
                return 0;

            default:
                printf("✗ 잘못된 선택입니다\n");
        }
    }

    return 0;
}
