/*
 * addressbook_v1.c
 *
 * 주소록 프로그램 - 완전한 CRUD 기능 구현
 *
 * 기능:
 *   1. 연락처 추가 (Create)
 *   2. 연락처 목록 보기 (Read)
 *   3. 연락처 검색 (Read)
 *   4. 연락처 수정 (Update)
 *   5. 연락처 삭제 (Delete)
 *   6. 파일 저장/불러오기 (Persistence)
 *
 * 컴파일: gcc -Wall -Wextra -std=c11 addressbook_v1.c -o addressbook
 * 실행: ./addressbook
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* 상수 정의 */
#define MAX_CONTACTS 100
#define NAME_LEN 50
#define PHONE_LEN 20
#define EMAIL_LEN 50
#define FILENAME "contacts.dat"

/* 연락처 구조체 */
typedef struct {
    int id;
    char name[NAME_LEN];
    char phone[PHONE_LEN];
    char email[EMAIL_LEN];
} Contact;

/* 주소록 구조체 */
typedef struct {
    Contact contacts[MAX_CONTACTS];
    int count;      // 현재 저장된 연락처 수
    int next_id;    // 다음에 할당할 ID
} AddressBook;

/* 함수 선언 */
void init_addressbook(AddressBook *ab);
void print_menu(void);
void add_contact(AddressBook *ab);
void list_contacts(AddressBook *ab);
void search_contact(AddressBook *ab);
void edit_contact(AddressBook *ab);
void delete_contact(AddressBook *ab);
int save_to_file(AddressBook *ab);
int load_from_file(AddressBook *ab);
void clear_input_buffer(void);
int find_by_id(AddressBook *ab, int id);

/* 메인 함수 */
int main(void) {
    AddressBook ab;
    int choice;

    /* 주소록 초기화 */
    init_addressbook(&ab);

    /* 파일에서 기존 데이터 불러오기 */
    if (load_from_file(&ab) == 0) {
        printf("기존 데이터를 불러왔습니다. (%d명)\n", ab.count);
    }

    /* 프로그램 시작 메시지 */
    printf("\n╔═══════════════════════════════╗\n");
    printf("║      📒 주소록 프로그램       ║\n");
    printf("╚═══════════════════════════════╝\n");

    /* 메인 루프 */
    while (1) {
        print_menu();
        printf("선택: ");

        /* 메뉴 선택 입력 */
        if (scanf("%d", &choice) != 1) {
            clear_input_buffer();
            printf("숫자를 입력해주세요.\n");
            continue;
        }
        clear_input_buffer();

        /* 메뉴 처리 */
        switch (choice) {
            case 1:
                add_contact(&ab);
                break;
            case 2:
                list_contacts(&ab);
                break;
            case 3:
                search_contact(&ab);
                break;
            case 4:
                edit_contact(&ab);
                break;
            case 5:
                delete_contact(&ab);
                break;
            case 6:
                if (save_to_file(&ab) == 0) {
                    printf("✓ 파일에 저장되었습니다.\n");
                }
                break;
            case 0:
                /* 종료 전 저장 확인 */
                printf("변경 사항을 저장하시겠습니까? (y/n): ");
                char save_confirm;
                scanf(" %c", &save_confirm);
                if (save_confirm == 'y' || save_confirm == 'Y') {
                    save_to_file(&ab);
                    printf("저장 완료.\n");
                }
                printf("프로그램을 종료합니다.\n");
                return 0;
            default:
                printf("잘못된 선택입니다.\n");
        }
        printf("\n");
    }

    return 0;
}

/* 주소록 초기화 */
void init_addressbook(AddressBook *ab) {
    ab->count = 0;
    ab->next_id = 1;
    memset(ab->contacts, 0, sizeof(ab->contacts));
}

/* 메뉴 출력 */
void print_menu(void) {
    printf("\n┌─────────────────────────┐\n");
    printf("│  1. 연락처 추가         │\n");
    printf("│  2. 목록 보기           │\n");
    printf("│  3. 검색                │\n");
    printf("│  4. 수정                │\n");
    printf("│  5. 삭제                │\n");
    printf("│  6. 파일 저장           │\n");
    printf("│  0. 종료                │\n");
    printf("└─────────────────────────┘\n");
}

/* 연락처 추가 */
void add_contact(AddressBook *ab) {
    /* 주소록이 가득 찼는지 확인 */
    if (ab->count >= MAX_CONTACTS) {
        printf("주소록이 가득 찼습니다. (최대 %d명)\n", MAX_CONTACTS);
        return;
    }

    /* 새 연락처를 위한 포인터 */
    Contact *c = &ab->contacts[ab->count];
    c->id = ab->next_id++;

    printf("\n═══ 새 연락처 추가 ═══\n\n");

    /* 이름 입력 (필수) */
    printf("이름: ");
    fgets(c->name, NAME_LEN, stdin);
    c->name[strcspn(c->name, "\n")] = '\0';  // 개행 문자 제거

    if (strlen(c->name) == 0) {
        printf("이름은 필수입니다. 추가가 취소되었습니다.\n");
        return;
    }

    /* 전화번호 입력 */
    printf("전화번호: ");
    fgets(c->phone, PHONE_LEN, stdin);
    c->phone[strcspn(c->phone, "\n")] = '\0';

    /* 이메일 입력 */
    printf("이메일: ");
    fgets(c->email, EMAIL_LEN, stdin);
    c->email[strcspn(c->email, "\n")] = '\0';

    /* 연락처 수 증가 */
    ab->count++;
    printf("\n✓ '%s' 연락처가 추가되었습니다. (ID: %d)\n", c->name, c->id);
}

/* 연락처 목록 보기 */
void list_contacts(AddressBook *ab) {
    printf("\n═══ 연락처 목록 ═══ (총 %d명)\n", ab->count);

    if (ab->count == 0) {
        printf("\n저장된 연락처가 없습니다.\n");
        return;
    }

    /* 테이블 헤더 */
    printf("\n%-4s │ %-15s │ %-15s │ %-20s\n", "ID", "이름", "전화번호", "이메일");
    printf("─────┼─────────────────┼─────────────────┼─────────────────────\n");

    /* 모든 연락처 출력 */
    for (int i = 0; i < ab->count; i++) {
        Contact *c = &ab->contacts[i];
        printf("%-4d │ %-15s │ %-15s │ %-20s\n",
               c->id, c->name, c->phone, c->email);
    }
}

/* 연락처 검색 */
void search_contact(AddressBook *ab) {
    char keyword[NAME_LEN];
    int found = 0;

    printf("\n═══ 연락처 검색 ═══\n\n");
    printf("검색어: ");
    fgets(keyword, NAME_LEN, stdin);
    keyword[strcspn(keyword, "\n")] = '\0';

    if (strlen(keyword) == 0) {
        printf("검색어를 입력해주세요.\n");
        return;
    }

    printf("\n검색 결과:\n");
    printf("─────────────────────────────────────────────────────\n");

    /* 모든 연락처에서 검색 */
    for (int i = 0; i < ab->count; i++) {
        Contact *c = &ab->contacts[i];
        /* 이름, 전화번호, 이메일에서 부분 문자열 검색 */
        if (strstr(c->name, keyword) != NULL ||
            strstr(c->phone, keyword) != NULL ||
            strstr(c->email, keyword) != NULL) {

            printf("ID: %d\n", c->id);
            printf("  이름: %s\n", c->name);
            printf("  전화: %s\n", c->phone);
            printf("  이메일: %s\n", c->email);
            printf("─────────────────────────────────────────────────────\n");
            found++;
        }
    }

    if (found == 0) {
        printf("'%s'에 대한 검색 결과가 없습니다.\n", keyword);
    } else {
        printf("총 %d건 검색됨\n", found);
    }
}

/* 연락처 수정 */
void edit_contact(AddressBook *ab) {
    int id;
    char input[EMAIL_LEN];

    printf("\n═══ 연락처 수정 ═══\n\n");
    printf("수정할 연락처 ID: ");
    scanf("%d", &id);
    clear_input_buffer();

    /* ID로 연락처 찾기 */
    int idx = find_by_id(ab, id);
    if (idx == -1) {
        printf("해당 ID의 연락처를 찾을 수 없습니다.\n");
        return;
    }

    Contact *c = &ab->contacts[idx];

    /* 현재 정보 표시 */
    printf("\n현재 정보:\n");
    printf("  이름: %s\n", c->name);
    printf("  전화: %s\n", c->phone);
    printf("  이메일: %s\n", c->email);

    printf("\n새 정보를 입력하세요 (빈 칸: 유지):\n");

    /* 이름 수정 */
    printf("이름 [%s]: ", c->name);
    fgets(input, NAME_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->name, input);
    }

    /* 전화번호 수정 */
    printf("전화번호 [%s]: ", c->phone);
    fgets(input, PHONE_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->phone, input);
    }

    /* 이메일 수정 */
    printf("이메일 [%s]: ", c->email);
    fgets(input, EMAIL_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->email, input);
    }

    printf("\n✓ 연락처가 수정되었습니다.\n");
}

/* 연락처 삭제 */
void delete_contact(AddressBook *ab) {
    int id;

    printf("\n═══ 연락처 삭제 ═══\n\n");
    printf("삭제할 연락처 ID: ");
    scanf("%d", &id);
    clear_input_buffer();

    /* ID로 연락처 찾기 */
    int idx = find_by_id(ab, id);
    if (idx == -1) {
        printf("해당 ID의 연락처를 찾을 수 없습니다.\n");
        return;
    }

    /* 삭제 확인 */
    printf("'%s' 연락처를 삭제하시겠습니까? (y/n): ", ab->contacts[idx].name);
    char confirm;
    scanf(" %c", &confirm);
    clear_input_buffer();

    if (confirm != 'y' && confirm != 'Y') {
        printf("삭제가 취소되었습니다.\n");
        return;
    }

    /* 삭제: 뒤의 요소들을 앞으로 이동 */
    for (int i = idx; i < ab->count - 1; i++) {
        ab->contacts[i] = ab->contacts[i + 1];
    }
    ab->count--;

    printf("✓ 연락처가 삭제되었습니다.\n");
}

/* 파일에 저장 (바이너리 모드) */
int save_to_file(AddressBook *ab) {
    FILE *fp = fopen(FILENAME, "wb");
    if (fp == NULL) {
        printf("파일 저장 실패: 파일을 열 수 없습니다.\n");
        return -1;
    }

    /* 메타데이터 저장 (count, next_id) */
    fwrite(&ab->count, sizeof(int), 1, fp);
    fwrite(&ab->next_id, sizeof(int), 1, fp);

    /* 연락처 배열 저장 */
    fwrite(ab->contacts, sizeof(Contact), ab->count, fp);

    fclose(fp);
    return 0;
}

/* 파일에서 불러오기 (바이너리 모드) */
int load_from_file(AddressBook *ab) {
    FILE *fp = fopen(FILENAME, "rb");
    if (fp == NULL) {
        /* 파일이 없으면 새로 시작 */
        return -1;
    }

    /* 메타데이터 읽기 */
    fread(&ab->count, sizeof(int), 1, fp);
    fread(&ab->next_id, sizeof(int), 1, fp);

    /* 연락처 배열 읽기 */
    fread(ab->contacts, sizeof(Contact), ab->count, fp);

    fclose(fp);
    return 0;
}

/* ID로 연락처 찾기 (인덱스 반환) */
int find_by_id(AddressBook *ab, int id) {
    for (int i = 0; i < ab->count; i++) {
        if (ab->contacts[i].id == id) {
            return i;
        }
    }
    return -1;  /* 찾지 못함 */
}

/* 입력 버퍼 비우기 */
void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
