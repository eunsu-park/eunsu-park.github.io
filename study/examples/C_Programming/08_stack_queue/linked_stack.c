// linked_stack.c
// 연결 리스트 기반 스택 구현
// 동적 메모리 할당으로 크기 제한 없음

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// 노드 구조체
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// 스택 구조체
typedef struct {
    Node *top;
    int size;
} LinkedStack;

// 스택 생성
LinkedStack* lstack_create(void) {
    LinkedStack *s = malloc(sizeof(LinkedStack));
    if (s) {
        s->top = NULL;
        s->size = 0;
    }
    return s;
}

// 스택 해제 (모든 메모리 해제)
void lstack_destroy(LinkedStack *s) {
    Node *current = s->top;
    while (current) {
        Node *next = current->next;
        free(current);
        current = next;
    }
    free(s);
}

// 비어있는지 확인
bool lstack_isEmpty(LinkedStack *s) {
    return s->top == NULL;
}

// Push - 맨 위에 요소 추가 (O(1))
bool lstack_push(LinkedStack *s, int value) {
    Node *node = malloc(sizeof(Node));
    if (!node) {
        printf("메모리 할당 실패!\n");
        return false;
    }

    node->data = value;
    node->next = s->top;  // 새 노드가 현재 top을 가리킴
    s->top = node;        // top을 새 노드로 갱신
    s->size++;
    return true;
}

// Pop - 맨 위 요소 제거 후 반환 (O(1))
bool lstack_pop(LinkedStack *s, int *value) {
    if (lstack_isEmpty(s)) {
        printf("Stack Underflow!\n");
        return false;
    }

    Node *node = s->top;
    *value = node->data;
    s->top = node->next;  // top을 다음 노드로 이동
    free(node);           // 제거된 노드 메모리 해제
    s->size--;
    return true;
}

// Peek - 맨 위 요소 확인 (제거 안함) (O(1))
bool lstack_peek(LinkedStack *s, int *value) {
    if (lstack_isEmpty(s)) {
        return false;
    }
    *value = s->top->data;
    return true;
}

// 스택 출력 (top부터 bottom까지)
void lstack_print(LinkedStack *s) {
    printf("Stack (size=%d): ", s->size);
    Node *current = s->top;
    while (current) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("(top → bottom)\n");
}

// 스택 크기 반환
int lstack_size(LinkedStack *s) {
    return s->size;
}

// 테스트 코드
int main(void) {
    LinkedStack *s = lstack_create();
    if (!s) {
        printf("스택 생성 실패!\n");
        return 1;
    }

    printf("=== 연결 리스트 기반 스택 테스트 ===\n\n");

    // Push 테스트
    printf("[ Push 연산 ]\n");
    for (int i = 1; i <= 5; i++) {
        printf("Push %d -> ", i * 10);
        lstack_push(s, i * 10);
        lstack_print(s);
    }

    // Peek 테스트
    printf("\n[ Peek 연산 ]\n");
    int top;
    if (lstack_peek(s, &top)) {
        printf("Top 값: %d (스택은 변경 안됨)\n", top);
        lstack_print(s);
    }

    // Pop 테스트
    printf("\n[ Pop 연산 ]\n");
    int value;
    while (lstack_pop(s, &value)) {
        printf("Popped: %d, ", value);
        lstack_print(s);
    }

    // Underflow 테스트
    printf("\n[ Underflow 테스트 ]\n");
    printf("빈 스택에서 Pop 시도: ");
    lstack_pop(s, &value);

    // 대량 삽입 테스트 (동적 할당의 장점)
    printf("\n[ 대량 삽입 테스트 ]\n");
    printf("10000개 요소 삽입 중...\n");
    for (int i = 0; i < 10000; i++) {
        lstack_push(s, i);
    }
    printf("삽입 완료! 현재 크기: %d\n", lstack_size(s));

    // 메모리 정리
    printf("\n메모리 해제 중...\n");
    lstack_destroy(s);
    printf("스택 해제 완료!\n");

    return 0;
}
