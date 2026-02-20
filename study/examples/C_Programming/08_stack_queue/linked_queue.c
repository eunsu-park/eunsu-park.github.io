// linked_queue.c
// 연결 리스트 기반 큐 구현
// 동적 메모리 할당으로 크기 제한 없음

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// 노드 구조체
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// 큐 구조체
typedef struct {
    Node *front;  // 첫 번째 노드 (제거 위치)
    Node *rear;   // 마지막 노드 (삽입 위치)
    int size;
} LinkedQueue;

// 큐 생성
LinkedQueue* lqueue_create(void) {
    LinkedQueue *q = malloc(sizeof(LinkedQueue));
    if (q) {
        q->front = NULL;
        q->rear = NULL;
        q->size = 0;
    }
    return q;
}

// 큐 해제 (모든 메모리 해제)
void lqueue_destroy(LinkedQueue *q) {
    Node *current = q->front;
    while (current) {
        Node *next = current->next;
        free(current);
        current = next;
    }
    free(q);
}

// 비어있는지 확인
bool lqueue_isEmpty(LinkedQueue *q) {
    return q->front == NULL;
}

// Enqueue - 뒤에 요소 추가 (O(1))
bool lqueue_enqueue(LinkedQueue *q, int value) {
    Node *node = malloc(sizeof(Node));
    if (!node) {
        printf("메모리 할당 실패!\n");
        return false;
    }

    node->data = value;
    node->next = NULL;

    // 큐가 비어있으면 front와 rear 모두 새 노드를 가리킴
    if (q->rear == NULL) {
        q->front = q->rear = node;
    } else {
        // 기존 rear 뒤에 새 노드 추가
        q->rear->next = node;
        q->rear = node;
    }
    q->size++;
    return true;
}

// Dequeue - 앞에서 요소 제거 (O(1))
bool lqueue_dequeue(LinkedQueue *q, int *value) {
    if (lqueue_isEmpty(q)) {
        printf("Queue is empty!\n");
        return false;
    }

    Node *node = q->front;
    *value = node->data;
    q->front = node->next;

    // 마지막 노드를 제거한 경우 rear도 NULL로 설정
    if (q->front == NULL) {
        q->rear = NULL;
    }

    free(node);
    q->size--;
    return true;
}

// Front - 앞의 값 확인 (제거 안함)
bool lqueue_front(LinkedQueue *q, int *value) {
    if (lqueue_isEmpty(q)) {
        return false;
    }
    *value = q->front->data;
    return true;
}

// Rear - 뒤의 값 확인
bool lqueue_rear(LinkedQueue *q, int *value) {
    if (lqueue_isEmpty(q)) {
        return false;
    }
    *value = q->rear->data;
    return true;
}

// 큐 출력 (front부터 rear까지)
void lqueue_print(LinkedQueue *q) {
    printf("Queue (size=%d): front -> ", q->size);
    Node *current = q->front;
    while (current) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("<- rear\n");
}

// 큐 크기 반환
int lqueue_size(LinkedQueue *q) {
    return q->size;
}

// 모든 요소 출력 후 큐 초기화
void lqueue_clear(LinkedQueue *q) {
    int value;
    while (lqueue_dequeue(q, &value)) {
        // 단순히 모든 요소 제거
    }
}

// 테스트 코드
int main(void) {
    LinkedQueue *q = lqueue_create();
    if (!q) {
        printf("큐 생성 실패!\n");
        return 1;
    }

    printf("=== 연결 리스트 기반 큐 테스트 ===\n\n");

    // Enqueue 테스트
    printf("[ 1단계: Enqueue 5개 ]\n");
    for (int i = 1; i <= 5; i++) {
        printf("Enqueue %d -> ", i * 10);
        lqueue_enqueue(q, i * 10);
        lqueue_print(q);
    }

    // Front/Rear 확인
    printf("\n[ 2단계: Front/Rear 확인 ]\n");
    int front_val, rear_val;
    if (lqueue_front(q, &front_val) && lqueue_rear(q, &rear_val)) {
        printf("Front 값: %d (먼저 들어온 값)\n", front_val);
        printf("Rear 값: %d (나중에 들어온 값)\n", rear_val);
        lqueue_print(q);
    }

    // Dequeue 테스트
    printf("\n[ 3단계: Dequeue 2개 ]\n");
    int value;
    for (int i = 0; i < 2; i++) {
        if (lqueue_dequeue(q, &value)) {
            printf("Dequeued: %d -> ", value);
            lqueue_print(q);
        }
    }

    // 추가 Enqueue
    printf("\n[ 4단계: Enqueue 2개 더 ]\n");
    for (int i = 6; i <= 7; i++) {
        printf("Enqueue %d -> ", i * 10);
        lqueue_enqueue(q, i * 10);
        lqueue_print(q);
    }

    // 모든 요소 Dequeue
    printf("\n[ 5단계: 모든 요소 Dequeue ]\n");
    while (lqueue_dequeue(q, &value)) {
        printf("Dequeued: %d -> ", value);
        lqueue_print(q);
    }

    // Empty 상태에서 제거 시도
    printf("\n[ 6단계: Empty 상태에서 Dequeue 시도 ]\n");
    printf("Dequeue -> ");
    lqueue_dequeue(q, &value);

    // 대량 삽입 테스트
    printf("\n[ 7단계: 대량 삽입 테스트 ]\n");
    printf("10000개 요소 삽입 중...\n");
    for (int i = 0; i < 10000; i++) {
        lqueue_enqueue(q, i);
    }
    printf("삽입 완료! 현재 크기: %d\n", lqueue_size(q));

    // 대량 제거 테스트
    printf("\n처음 5개 요소 확인:\n");
    for (int i = 0; i < 5; i++) {
        lqueue_dequeue(q, &value);
        printf("  Dequeued: %d\n", value);
    }

    // 메모리 정리
    printf("\n메모리 해제 중...\n");
    lqueue_destroy(q);
    printf("큐 해제 완료!\n");

    return 0;
}
