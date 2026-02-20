// bracket_check.c
// 스택을 이용한 괄호 검사 프로그램
// 여는 괄호와 닫는 괄호의 쌍이 올바른지 확인

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define MAX_SIZE 100

// 문자를 저장하는 스택
typedef struct {
    char data[MAX_SIZE];
    int top;
} CharStack;

// 스택 초기화
void stack_init(CharStack *s) {
    s->top = -1;
}

// 비어있는지 확인
bool stack_isEmpty(CharStack *s) {
    return s->top == -1;
}

// 가득 찼는지 확인
bool stack_isFull(CharStack *s) {
    return s->top == MAX_SIZE - 1;
}

// Push
void stack_push(CharStack *s, char c) {
    if (!stack_isFull(s)) {
        s->data[++s->top] = c;
    }
}

// Pop
char stack_pop(CharStack *s) {
    if (!stack_isEmpty(s)) {
        return s->data[s->top--];
    }
    return '\0';
}

// Peek
char stack_peek(CharStack *s) {
    if (!stack_isEmpty(s)) {
        return s->data[s->top];
    }
    return '\0';
}

// 여는 괄호인지 확인
bool isOpeningBracket(char c) {
    return c == '(' || c == '{' || c == '[';
}

// 닫는 괄호인지 확인
bool isClosingBracket(char c) {
    return c == ')' || c == '}' || c == ']';
}

// 괄호 쌍이 맞는지 확인
bool isMatchingPair(char open, char close) {
    return (open == '(' && close == ')') ||
           (open == '{' && close == '}') ||
           (open == '[' && close == ']');
}

// 괄호 검사 함수
bool checkBrackets(const char *expr) {
    CharStack s;
    stack_init(&s);

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];

        // 여는 괄호: 스택에 push
        if (isOpeningBracket(c)) {
            stack_push(&s, c);
            printf("  [위치 %d] '%c' 여는 괄호 → 스택에 push\n", i, c);
        }
        // 닫는 괄호: 스택에서 pop하여 짝 확인
        else if (isClosingBracket(c)) {
            if (stack_isEmpty(&s)) {
                printf("  [위치 %d] '%c' 오류 - 짝이 없는 닫는 괄호\n", i, c);
                return false;
            }

            char open = stack_pop(&s);
            if (!isMatchingPair(open, c)) {
                printf("  [위치 %d] 오류 - '%c'와 '%c' 불일치\n", i, open, c);
                return false;
            }
            printf("  [위치 %d] '%c' 닫는 괄호 → '%c'와 매칭 OK\n", i, c, open);
        }
    }

    // 스택이 비어있지 않으면 짝이 맞지 않는 여는 괄호가 남아있음
    if (!stack_isEmpty(&s)) {
        printf("  오류 - 닫히지 않은 여는 괄호가 %d개 남아있음\n", s.top + 1);
        return false;
    }

    return true;
}

// 간단한 괄호 검사 (디버그 출력 없음)
bool checkBracketsQuiet(const char *expr) {
    CharStack s;
    stack_init(&s);

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];

        if (isOpeningBracket(c)) {
            stack_push(&s, c);
        } else if (isClosingBracket(c)) {
            if (stack_isEmpty(&s)) {
                return false;
            }
            char open = stack_pop(&s);
            if (!isMatchingPair(open, c)) {
                return false;
            }
        }
    }

    return stack_isEmpty(&s);
}

// 테스트 코드
int main(void) {
    printf("=== 괄호 검사 프로그램 ===\n\n");

    const char *tests[] = {
        "(a + b) * (c - d)",     // 올바른 괄호
        "((a + b) * c",          // 닫히지 않은 괄호
        "{[()]}",                // 올바른 중첩
        "{[(])}",                // 잘못된 중첩
        "((()))",                // 올바른 괄호
        ")(",                    // 잘못된 순서
        "{[a + (b * c)] - d}",   // 올바른 복잡한 표현식
        "((a + b)",              // 하나 부족
        "a + b)",                // 여는 괄호 없음
        "[]{}()"                 // 올바른 연속
    };

    int n = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < n; i++) {
        printf("테스트 %d: \"%s\"\n", i + 1, tests[i]);

        if (checkBrackets(tests[i])) {
            printf("✓ 결과: 올바른 괄호\n");
        } else {
            printf("✗ 결과: 잘못된 괄호\n");
        }
        printf("\n");
    }

    // 추가 테스트: 사용자 입력
    printf("\n=== 직접 테스트해보기 ===\n");
    char input[MAX_SIZE];
    printf("괄호를 포함한 식을 입력하세요 (종료: q): ");

    while (fgets(input, MAX_SIZE, stdin)) {
        // 개행 문자 제거
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "q") == 0) {
            break;
        }

        if (checkBracketsQuiet(input)) {
            printf("✓ 올바른 괄호입니다!\n");
        } else {
            printf("✗ 잘못된 괄호입니다!\n");
        }

        printf("\n다른 식을 입력하세요 (종료: q): ");
    }

    printf("\n프로그램을 종료합니다.\n");
    return 0;
}
