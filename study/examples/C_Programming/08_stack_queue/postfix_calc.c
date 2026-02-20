// postfix_calc.c
// 스택을 이용한 후위 표기법(Postfix) 계산기
// 중위: (3 + 4) * 5  →  후위: 3 4 + 5 *

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>

#define MAX_SIZE 100

// 실수형 스택 (계산 결과가 실수일 수 있음)
typedef struct {
    double data[MAX_SIZE];
    int top;
} Stack;

// 스택 초기화
void stack_init(Stack *s) {
    s->top = -1;
}

// 비어있는지 확인
bool stack_isEmpty(Stack *s) {
    return s->top == -1;
}

// Push
void stack_push(Stack *s, double v) {
    if (s->top < MAX_SIZE - 1) {
        s->data[++s->top] = v;
    }
}

// Pop
double stack_pop(Stack *s) {
    if (!stack_isEmpty(s)) {
        return s->data[s->top--];
    }
    return 0.0;
}

// Peek
double stack_peek(Stack *s) {
    if (!stack_isEmpty(s)) {
        return s->data[s->top];
    }
    return 0.0;
}

// 연산자인지 확인
bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '%';
}

// 연산 수행
double applyOperator(double a, double b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/':
            if (b == 0) {
                printf("오류: 0으로 나눌 수 없습니다!\n");
                return 0;
            }
            return a / b;
        case '%':
            if (b == 0) {
                printf("오류: 0으로 나눌 수 없습니다!\n");
                return 0;
            }
            return (int)a % (int)b;
        default:
            printf("알 수 없는 연산자: %c\n", op);
            return 0;
    }
}

// 후위 표기법 계산
// 공백으로 구분된 토큰 형식: "3 4 + 5 *"
double evaluatePostfix(const char *expr) {
    Stack s;
    stack_init(&s);

    // 입력 문자열 복사 (strtok이 원본을 수정하므로)
    char *str = strdup(expr);
    char *token = strtok(str, " ");

    printf("계산 과정:\n");

    while (token) {
        // 숫자인지 확인 (음수도 처리)
        if (isdigit(token[0]) || (token[0] == '-' && strlen(token) > 1)) {
            double num = atof(token);
            stack_push(&s, num);
            printf("  숫자 %g를 스택에 push\n", num);
        }
        // 연산자
        else if (isOperator(token[0]) && strlen(token) == 1) {
            if (s.top < 1) {
                printf("오류: 연산자 '%c'를 적용할 피연산자가 부족합니다!\n", token[0]);
                free(str);
                return 0;
            }

            double b = stack_pop(&s);
            double a = stack_pop(&s);
            double result = applyOperator(a, b, token[0]);

            printf("  연산: %g %c %g = %g\n", a, token[0], b, result);
            stack_push(&s, result);
        }
        else {
            printf("오류: 잘못된 토큰 '%s'\n", token);
            free(str);
            return 0;
        }

        token = strtok(NULL, " ");
    }

    free(str);

    // 스택에 정확히 하나의 결과만 남아야 함
    if (s.top != 0) {
        printf("오류: 잘못된 표기법 (스택에 %d개의 값이 남음)\n", s.top + 1);
        return 0;
    }

    return stack_pop(&s);
}

// 연산자 우선순위 반환
int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/' || op == '%') return 2;
    return 0;
}

// 중위 표기법을 후위 표기법으로 변환 (심화)
// 간단한 구현: 괄호와 연산자 우선순위 처리
void infixToPostfix(const char *infix, char *postfix) {
    Stack s;
    stack_init(&s);
    int j = 0;

    for (int i = 0; infix[i]; i++) {
        char c = infix[i];

        // 공백 무시
        if (c == ' ') continue;

        // 피연산자 (숫자나 변수)
        if (isalnum(c)) {
            postfix[j++] = c;
            postfix[j++] = ' ';
        }
        // 여는 괄호
        else if (c == '(') {
            stack_push(&s, c);
        }
        // 닫는 괄호
        else if (c == ')') {
            while (!stack_isEmpty(&s) && (char)stack_peek(&s) != '(') {
                postfix[j++] = (char)stack_pop(&s);
                postfix[j++] = ' ';
            }
            stack_pop(&s);  // '(' 제거
        }
        // 연산자
        else if (isOperator(c)) {
            while (!stack_isEmpty(&s) &&
                   precedence((char)stack_peek(&s)) >= precedence(c)) {
                postfix[j++] = (char)stack_pop(&s);
                postfix[j++] = ' ';
            }
            stack_push(&s, c);
        }
    }

    // 스택에 남은 연산자 모두 출력
    while (!stack_isEmpty(&s)) {
        postfix[j++] = (char)stack_pop(&s);
        postfix[j++] = ' ';
    }

    postfix[j] = '\0';
}

// 테스트 코드
int main(void) {
    printf("=== 후위 표기법 계산기 ===\n\n");

    // 후위 표기법 계산 테스트
    const char *postfixExpressions[] = {
        "3 4 +",                  // 3 + 4 = 7
        "3 4 + 5 *",              // (3 + 4) * 5 = 35
        "10 2 / 3 +",             // 10 / 2 + 3 = 8
        "5 1 2 + 4 * + 3 -",      // 5 + ((1 + 2) * 4) - 3 = 14
        "15 7 1 1 + - / 3 * 2 1 1 + + -",  // 복잡한 식
        "8 2 /",                  // 8 / 2 = 4
        "9 3 % 2 *"               // (9 % 3) * 2 = 0
    };

    int n = sizeof(postfixExpressions) / sizeof(postfixExpressions[0]);

    printf("[ 후위 표기법 계산 ]\n");
    for (int i = 0; i < n; i++) {
        printf("\n%d. Expression: %s\n", i + 1, postfixExpressions[i]);
        double result = evaluatePostfix(postfixExpressions[i]);
        printf("   최종 결과: %.2f\n", result);
    }

    // 중위 → 후위 변환 테스트
    printf("\n\n[ 중위 표기법 → 후위 표기법 변환 ]\n");

    const char *infixExpressions[] = {
        "(3 + 4) * 5",
        "a + b * c",
        "(a + b) * (c - d)",
        "a + b - c",
        "a * b + c / d"
    };

    int m = sizeof(infixExpressions) / sizeof(infixExpressions[0]);

    for (int i = 0; i < m; i++) {
        char postfix[MAX_SIZE];
        infixToPostfix(infixExpressions[i], postfix);
        printf("\n중위: %s\n", infixExpressions[i]);
        printf("후위: %s\n", postfix);
    }

    // 대화형 계산기
    printf("\n\n[ 대화형 후위 계산기 ]\n");
    printf("후위 표기법으로 수식을 입력하세요 (예: 3 4 +)\n");
    printf("종료하려면 'q'를 입력하세요.\n\n");

    char input[MAX_SIZE];
    while (1) {
        printf("> ");
        if (!fgets(input, MAX_SIZE, stdin)) break;

        // 개행 문자 제거
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "q") == 0) break;

        double result = evaluatePostfix(input);
        printf("결과: %.2f\n\n", result);
    }

    printf("프로그램을 종료합니다.\n");
    return 0;
}
