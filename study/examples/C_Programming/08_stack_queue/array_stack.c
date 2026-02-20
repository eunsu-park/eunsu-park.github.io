// array_stack.c
// 배열 기반 스택 구현
// LIFO (Last In, First Out) 자료구조

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 100

typedef struct {
    int data[MAX_SIZE];
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

// 가득 찼는지 확인
bool stack_isFull(Stack *s) {
    return s->top == MAX_SIZE - 1;
}

// Push - 맨 위에 요소 추가 (O(1))
bool stack_push(Stack *s, int value) {
    if (stack_isFull(s)) {
        printf("Stack Overflow!\n");
        return false;
    }
    s->data[++s->top] = value;
    return true;
}

// Pop - 맨 위 요소 제거 후 반환 (O(1))
bool stack_pop(Stack *s, int *value) {
    if (stack_isEmpty(s)) {
        printf("Stack Underflow!\n");
        return false;
    }
    *value = s->data[s->top--];
    return true;
}

// Peek - 맨 위 요소 확인 (제거 안함) (O(1))
bool stack_peek(Stack *s, int *value) {
    if (stack_isEmpty(s)) {
        return false;
    }
    *value = s->data[s->top];
    return true;
}

// 스택 출력 (디버깅용)
void stack_print(Stack *s) {
    printf("Stack (top=%d): ", s->top);
    for (int i = 0; i <= s->top; i++) {
        printf("%d ", s->data[i]);
    }
    printf("\n");
}

// 스택 크기 반환
int stack_size(Stack *s) {
    return s->top + 1;
}

// 테스트 코드
int main(void) {
    Stack s;
    stack_init(&s);

    printf("=== 배열 기반 스택 테스트 ===\n\n");

    // Push 테스트
    printf("[ Push 연산 ]\n");
    for (int i = 1; i <= 5; i++) {
        printf("Push %d -> ", i * 10);
        stack_push(&s, i * 10);
        stack_print(&s);
    }

    // Peek 테스트
    printf("\n[ Peek 연산 ]\n");
    int top;
    if (stack_peek(&s, &top)) {
        printf("Top 값: %d (스택은 변경 안됨)\n", top);
        stack_print(&s);
    }

    // Pop 테스트
    printf("\n[ Pop 연산 ]\n");
    int value;
    while (stack_pop(&s, &value)) {
        printf("Popped: %d, ", value);
        stack_print(&s);
    }

    // Underflow 테스트
    printf("\n[ Underflow 테스트 ]\n");
    printf("빈 스택에서 Pop 시도: ");
    stack_pop(&s, &value);

    // Overflow 테스트
    printf("\n[ Overflow 테스트 ]\n");
    Stack s2;
    stack_init(&s2);
    printf("MAX_SIZE를 초과하는 Push 시도...\n");
    for (int i = 0; i <= MAX_SIZE; i++) {
        if (!stack_push(&s2, i)) {
            printf("총 %d개 요소 삽입 후 Overflow 발생\n", i);
            break;
        }
    }

    return 0;
}
