/*
 * 스택과 큐 (Stack and Queue)
 * Stack, Queue, Deque, Monotonic Stack
 *
 * 기본 자료구조와 응용 알고리즘입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* =============================================================================
 * 1. 스택 구현
 * ============================================================================= */

typedef struct {
    int* data;
    int top;
    int capacity;
} Stack;

Stack* stack_create(int capacity) {
    Stack* s = malloc(sizeof(Stack));
    s->data = malloc(capacity * sizeof(int));
    s->top = -1;
    s->capacity = capacity;
    return s;
}

void stack_free(Stack* s) {
    free(s->data);
    free(s);
}

bool stack_is_empty(Stack* s) {
    return s->top == -1;
}

bool stack_is_full(Stack* s) {
    return s->top == s->capacity - 1;
}

void stack_push(Stack* s, int value) {
    if (!stack_is_full(s)) {
        s->data[++s->top] = value;
    }
}

int stack_pop(Stack* s) {
    if (!stack_is_empty(s)) {
        return s->data[s->top--];
    }
    return -1;
}

int stack_peek(Stack* s) {
    if (!stack_is_empty(s)) {
        return s->data[s->top];
    }
    return -1;
}

/* =============================================================================
 * 2. 큐 구현 (원형 큐)
 * ============================================================================= */

typedef struct {
    int* data;
    int front;
    int rear;
    int size;
    int capacity;
} Queue;

Queue* queue_create(int capacity) {
    Queue* q = malloc(sizeof(Queue));
    q->data = malloc(capacity * sizeof(int));
    q->front = 0;
    q->rear = -1;
    q->size = 0;
    q->capacity = capacity;
    return q;
}

void queue_free(Queue* q) {
    free(q->data);
    free(q);
}

bool queue_is_empty(Queue* q) {
    return q->size == 0;
}

bool queue_is_full(Queue* q) {
    return q->size == q->capacity;
}

void queue_enqueue(Queue* q, int value) {
    if (!queue_is_full(q)) {
        q->rear = (q->rear + 1) % q->capacity;
        q->data[q->rear] = value;
        q->size++;
    }
}

int queue_dequeue(Queue* q) {
    if (!queue_is_empty(q)) {
        int value = q->data[q->front];
        q->front = (q->front + 1) % q->capacity;
        q->size--;
        return value;
    }
    return -1;
}

int queue_front(Queue* q) {
    if (!queue_is_empty(q)) {
        return q->data[q->front];
    }
    return -1;
}

/* =============================================================================
 * 3. 덱 구현 (Double-ended Queue)
 * ============================================================================= */

typedef struct {
    int* data;
    int front;
    int rear;
    int size;
    int capacity;
} Deque;

Deque* deque_create(int capacity) {
    Deque* d = malloc(sizeof(Deque));
    d->data = malloc(capacity * sizeof(int));
    d->front = 0;
    d->rear = 0;
    d->size = 0;
    d->capacity = capacity;
    return d;
}

void deque_free(Deque* d) {
    free(d->data);
    free(d);
}

bool deque_is_empty(Deque* d) {
    return d->size == 0;
}

void deque_push_front(Deque* d, int value) {
    if (d->size < d->capacity) {
        d->front = (d->front - 1 + d->capacity) % d->capacity;
        d->data[d->front] = value;
        d->size++;
    }
}

void deque_push_back(Deque* d, int value) {
    if (d->size < d->capacity) {
        d->data[d->rear] = value;
        d->rear = (d->rear + 1) % d->capacity;
        d->size++;
    }
}

int deque_pop_front(Deque* d) {
    if (!deque_is_empty(d)) {
        int value = d->data[d->front];
        d->front = (d->front + 1) % d->capacity;
        d->size--;
        return value;
    }
    return -1;
}

int deque_pop_back(Deque* d) {
    if (!deque_is_empty(d)) {
        d->rear = (d->rear - 1 + d->capacity) % d->capacity;
        d->size--;
        return d->data[d->rear];
    }
    return -1;
}

int deque_front(Deque* d) {
    if (!deque_is_empty(d)) {
        return d->data[d->front];
    }
    return -1;
}

int deque_back(Deque* d) {
    if (!deque_is_empty(d)) {
        return d->data[(d->rear - 1 + d->capacity) % d->capacity];
    }
    return -1;
}

/* =============================================================================
 * 4. 괄호 검사
 * ============================================================================= */

bool is_valid_parentheses(const char* s) {
    Stack* stack = stack_create(strlen(s));

    for (int i = 0; s[i]; i++) {
        char c = s[i];
        if (c == '(' || c == '{' || c == '[') {
            stack_push(stack, c);
        } else {
            if (stack_is_empty(stack)) {
                stack_free(stack);
                return false;
            }
            char top = stack_pop(stack);
            if ((c == ')' && top != '(') ||
                (c == '}' && top != '{') ||
                (c == ']' && top != '[')) {
                stack_free(stack);
                return false;
            }
        }
    }

    bool valid = stack_is_empty(stack);
    stack_free(stack);
    return valid;
}

/* =============================================================================
 * 5. 후위 표기식 계산
 * ============================================================================= */

int evaluate_postfix(const char* expr) {
    Stack* stack = stack_create(strlen(expr));

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];

        if (c >= '0' && c <= '9') {
            stack_push(stack, c - '0');
        } else if (c == ' ') {
            continue;
        } else {
            int b = stack_pop(stack);
            int a = stack_pop(stack);
            int result;

            switch (c) {
                case '+': result = a + b; break;
                case '-': result = a - b; break;
                case '*': result = a * b; break;
                case '/': result = a / b; break;
                default: result = 0;
            }
            stack_push(stack, result);
        }
    }

    int result = stack_pop(stack);
    stack_free(stack);
    return result;
}

/* =============================================================================
 * 6. 모노토닉 스택 - 다음으로 큰 원소
 * ============================================================================= */

int* next_greater_element(int arr[], int n) {
    int* result = malloc(n * sizeof(int));
    Stack* stack = stack_create(n);

    for (int i = n - 1; i >= 0; i--) {
        while (!stack_is_empty(stack) && stack_peek(stack) <= arr[i]) {
            stack_pop(stack);
        }
        result[i] = stack_is_empty(stack) ? -1 : stack_peek(stack);
        stack_push(stack, arr[i]);
    }

    stack_free(stack);
    return result;
}

/* =============================================================================
 * 7. 슬라이딩 윈도우 최댓값 (덱 활용)
 * ============================================================================= */

int* sliding_window_max(int arr[], int n, int k, int* result_size) {
    *result_size = n - k + 1;
    int* result = malloc((*result_size) * sizeof(int));
    Deque* dq = deque_create(n);

    for (int i = 0; i < n; i++) {
        /* 윈도우 범위 밖 제거 */
        while (!deque_is_empty(dq) && deque_front(dq) <= i - k) {
            deque_pop_front(dq);
        }

        /* 현재 값보다 작은 원소 제거 */
        while (!deque_is_empty(dq) && arr[deque_back(dq)] < arr[i]) {
            deque_pop_back(dq);
        }

        deque_push_back(dq, i);

        if (i >= k - 1) {
            result[i - k + 1] = arr[deque_front(dq)];
        }
    }

    deque_free(dq);
    return result;
}

/* =============================================================================
 * 8. 히스토그램에서 가장 큰 직사각형
 * ============================================================================= */

int largest_rectangle_histogram(int heights[], int n) {
    Stack* stack = stack_create(n + 1);
    int max_area = 0;

    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];

        while (!stack_is_empty(stack) && heights[stack_peek(stack)] > h) {
            int height = heights[stack_pop(stack)];
            int width = stack_is_empty(stack) ? i : i - stack_peek(stack) - 1;
            int area = height * width;
            if (area > max_area) max_area = area;
        }

        stack_push(stack, i);
    }

    stack_free(stack);
    return max_area;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]");
}

int main(void) {
    printf("============================================================\n");
    printf("스택과 큐 (Stack and Queue) 예제\n");
    printf("============================================================\n");

    /* 1. 스택 */
    printf("\n[1] 스택 기본 연산\n");
    Stack* s = stack_create(10);
    stack_push(s, 1);
    stack_push(s, 2);
    stack_push(s, 3);
    printf("    Push: 1, 2, 3\n");
    printf("    Top: %d\n", stack_peek(s));
    printf("    Pop: %d\n", stack_pop(s));
    printf("    Pop: %d\n", stack_pop(s));
    stack_free(s);

    /* 2. 큐 */
    printf("\n[2] 큐 기본 연산\n");
    Queue* q = queue_create(10);
    queue_enqueue(q, 1);
    queue_enqueue(q, 2);
    queue_enqueue(q, 3);
    printf("    Enqueue: 1, 2, 3\n");
    printf("    Front: %d\n", queue_front(q));
    printf("    Dequeue: %d\n", queue_dequeue(q));
    printf("    Dequeue: %d\n", queue_dequeue(q));
    queue_free(q);

    /* 3. 괄호 검사 */
    printf("\n[3] 괄호 검사\n");
    printf("    '()[]{}': %s\n", is_valid_parentheses("()[]{}") ? "valid" : "invalid");
    printf("    '([)]': %s\n", is_valid_parentheses("([)]") ? "valid" : "invalid");
    printf("    '{[()]}': %s\n", is_valid_parentheses("{[()]}") ? "valid" : "invalid");

    /* 4. 후위 표기식 */
    printf("\n[4] 후위 표기식 계산\n");
    printf("    '2 3 + 4 *' = %d\n", evaluate_postfix("2 3 + 4 *"));
    printf("    '5 1 2 + 4 * + 3 -' = %d\n", evaluate_postfix("5 1 2 + 4 * + 3 -"));

    /* 5. 다음으로 큰 원소 */
    printf("\n[5] 다음으로 큰 원소 (모노토닉 스택)\n");
    int arr5[] = {4, 5, 2, 25};
    int* nge = next_greater_element(arr5, 4);
    printf("    배열: [4, 5, 2, 25]\n");
    printf("    NGE:  ");
    print_array(nge, 4);
    printf("\n");
    free(nge);

    /* 6. 슬라이딩 윈도우 최댓값 */
    printf("\n[6] 슬라이딩 윈도우 최댓값\n");
    int arr6[] = {1, 3, -1, -3, 5, 3, 6, 7};
    int result_size;
    int* max_vals = sliding_window_max(arr6, 8, 3, &result_size);
    printf("    배열: [1,3,-1,-3,5,3,6,7], k=3\n");
    printf("    최댓값: ");
    print_array(max_vals, result_size);
    printf("\n");
    free(max_vals);

    /* 7. 히스토그램 */
    printf("\n[7] 히스토그램 최대 직사각형\n");
    int heights[] = {2, 1, 5, 6, 2, 3};
    printf("    높이: [2,1,5,6,2,3]\n");
    printf("    최대 넓이: %d\n", largest_rectangle_histogram(heights, 6));

    /* 8. 자료구조 비교 */
    printf("\n[8] 자료구조 비교\n");
    printf("    | 자료구조 | 삽입    | 삭제    | 특징            |\n");
    printf("    |----------|---------|---------|------------------|\n");
    printf("    | 스택     | O(1)    | O(1)    | LIFO            |\n");
    printf("    | 큐       | O(1)    | O(1)    | FIFO            |\n");
    printf("    | 덱       | O(1)    | O(1)    | 양쪽 삽입/삭제  |\n");

    printf("\n============================================================\n");

    return 0;
}
