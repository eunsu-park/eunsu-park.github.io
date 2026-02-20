// circular_queue.c
// 원형 큐(Circular Queue) 구현
// 배열의 앞부분 재사용으로 공간 효율성 향상

#include <stdio.h>
#include <stdbool.h>

#define MAX_SIZE 5

// 원형 큐 구조체
typedef struct {
    int data[MAX_SIZE];
    int front;  // 첫 번째 요소 위치
    int rear;   // 마지막 요소 위치
    int count;  // 현재 요소 개수
} CircularQueue;

// 큐 초기화
void queue_init(CircularQueue *q) {
    q->front = 0;
    q->rear = -1;
    q->count = 0;
}

// 비어있는지 확인
bool queue_isEmpty(CircularQueue *q) {
    return q->count == 0;
}

// 가득 찼는지 확인
bool queue_isFull(CircularQueue *q) {
    return q->count == MAX_SIZE;
}

// Enqueue - 뒤에 요소 추가 (O(1))
bool queue_enqueue(CircularQueue *q, int value) {
    if (queue_isFull(q)) {
        printf("Queue is full!\n");
        return false;
    }

    q->rear = (q->rear + 1) % MAX_SIZE;  // 원형으로 순환
    q->data[q->rear] = value;
    q->count++;
    return true;
}

// Dequeue - 앞에서 요소 제거 (O(1))
bool queue_dequeue(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) {
        printf("Queue is empty!\n");
        return false;
    }

    *value = q->data[q->front];
    q->front = (q->front + 1) % MAX_SIZE;  // 원형으로 순환
    q->count--;
    return true;
}

// Front - 앞의 값 확인 (제거 안함)
bool queue_front(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) {
        return false;
    }
    *value = q->data[q->front];
    return true;
}

// Rear - 뒤의 값 확인
bool queue_rear(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) {
        return false;
    }
    *value = q->data[q->rear];
    return true;
}

// 큐 출력 (front부터 rear까지)
void queue_print(CircularQueue *q) {
    printf("Queue (count=%d): [", q->count);
    if (!queue_isEmpty(q)) {
        int i = q->front;
        for (int c = 0; c < q->count; c++) {
            printf("%d", q->data[i]);
            if (c < q->count - 1) printf(", ");
            i = (i + 1) % MAX_SIZE;
        }
    }
    printf("] (front=%d, rear=%d)\n", q->front, q->rear);
}

// 배열 상태 시각화 (디버깅용)
void queue_visualize(CircularQueue *q) {
    printf("배열 상태: [");
    for (int i = 0; i < MAX_SIZE; i++) {
        if (q->count > 0) {
            int start = q->front;
            int end = q->rear;
            bool inRange = false;

            // 원형 큐에서 i가 유효 범위인지 확인
            if (start <= end) {
                inRange = (i >= start && i <= end);
            } else {
                inRange = (i >= start || i <= end);
            }

            if (inRange) {
                printf("%d", q->data[i]);
            } else {
                printf("-");
            }
        } else {
            printf("-");
        }
        if (i < MAX_SIZE - 1) printf(" ");
    }
    printf("]\n");

    // front와 rear 위치 표시
    printf("           ");
    for (int i = 0; i < MAX_SIZE; i++) {
        if (i == q->front && i == q->rear && q->count > 0) {
            printf("FR");
        } else if (i == q->front && q->count > 0) {
            printf("F ");
        } else if (i == q->rear && q->count > 0) {
            printf("R ");
        } else {
            printf("  ");
        }
    }
    printf("\n");
}

// 큐 크기 반환
int queue_size(CircularQueue *q) {
    return q->count;
}

// 테스트 코드
int main(void) {
    CircularQueue q;
    queue_init(&q);

    printf("=== 원형 큐 테스트 ===\n\n");

    // Enqueue 테스트 (큐를 가득 채움)
    printf("[ 1단계: Enqueue 5개 (큐 가득 채우기) ]\n");
    for (int i = 1; i <= 5; i++) {
        printf("Enqueue %d -> ", i * 10);
        queue_enqueue(&q, i * 10);
        queue_print(&q);
        queue_visualize(&q);
        printf("\n");
    }

    // Full 상태에서 추가 시도
    printf("[ 2단계: Full 상태에서 Enqueue 시도 ]\n");
    printf("Enqueue 60 -> ");
    queue_enqueue(&q, 60);
    printf("\n");

    // Dequeue 2개
    int value;
    printf("[ 3단계: Dequeue 2개 ]\n");
    for (int i = 0; i < 2; i++) {
        queue_dequeue(&q, &value);
        printf("Dequeued: %d -> ", value);
        queue_print(&q);
        queue_visualize(&q);
        printf("\n");
    }

    // 다시 Enqueue (원형 특성 확인)
    printf("[ 4단계: 다시 Enqueue 2개 (원형 순환 확인) ]\n");
    for (int i = 6; i <= 7; i++) {
        printf("Enqueue %d -> ", i * 10);
        queue_enqueue(&q, i * 10);
        queue_print(&q);
        queue_visualize(&q);
        printf("\n");
    }

    // Front와 Rear 확인
    printf("[ 5단계: Front/Rear 확인 ]\n");
    int front_val, rear_val;
    if (queue_front(&q, &front_val) && queue_rear(&q, &rear_val)) {
        printf("Front 값: %d, Rear 값: %d\n", front_val, rear_val);
    }
    printf("\n");

    // 모든 요소 Dequeue
    printf("[ 6단계: 모든 요소 Dequeue ]\n");
    while (queue_dequeue(&q, &value)) {
        printf("Dequeued: %d -> ", value);
        queue_print(&q);
    }

    // Empty 상태에서 제거 시도
    printf("\n[ 7단계: Empty 상태에서 Dequeue 시도 ]\n");
    printf("Dequeue -> ");
    queue_dequeue(&q, &value);

    return 0;
}
