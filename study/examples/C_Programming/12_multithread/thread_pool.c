// thread_pool.c
// 스레드 풀 (Thread Pool) 구현
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>

#define POOL_SIZE 4
#define QUEUE_SIZE 100

// 작업 정의
typedef struct Task {
    void (*function)(void* arg);
    void* arg;
} Task;

// 작업 큐
typedef struct {
    Task tasks[QUEUE_SIZE];
    int front;
    int rear;
    int count;

    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;

    bool shutdown;
} TaskQueue;

// 스레드 풀
typedef struct {
    pthread_t threads[POOL_SIZE];
    TaskQueue queue;
    int thread_count;
} ThreadPool;

// 작업 큐 초기화
void queue_init(TaskQueue* q) {
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    q->shutdown = false;

    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

// 작업 큐 정리
void queue_destroy(TaskQueue* q) {
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

// 작업 추가
bool queue_push(TaskQueue* q, Task task) {
    pthread_mutex_lock(&q->mutex);

    while (q->count == QUEUE_SIZE && !q->shutdown) {
        pthread_cond_wait(&q->not_full, &q->mutex);
    }

    if (q->shutdown) {
        pthread_mutex_unlock(&q->mutex);
        return false;
    }

    q->tasks[q->rear] = task;
    q->rear = (q->rear + 1) % QUEUE_SIZE;
    q->count++;

    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);

    return true;
}

// 작업 가져오기
bool queue_pop(TaskQueue* q, Task* task) {
    pthread_mutex_lock(&q->mutex);

    while (q->count == 0 && !q->shutdown) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }

    if (q->count == 0 && q->shutdown) {
        pthread_mutex_unlock(&q->mutex);
        return false;
    }

    *task = q->tasks[q->front];
    q->front = (q->front + 1) % QUEUE_SIZE;
    q->count--;

    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);

    return true;
}

// 워커 스레드 함수
void* worker_thread(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    Task task;

    printf("[워커] 스레드 시작 (TID: %lu)\n", pthread_self());

    while (queue_pop(&pool->queue, &task)) {
        printf("[워커 %lu] 작업 실행\n", pthread_self());
        task.function(task.arg);
    }

    printf("[워커 %lu] 스레드 종료\n", pthread_self());
    return NULL;
}

// 스레드 풀 생성
ThreadPool* pool_create(int size) {
    ThreadPool* pool = malloc(sizeof(ThreadPool));
    pool->thread_count = size;

    queue_init(&pool->queue);

    for (int i = 0; i < size; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }

    return pool;
}

// 작업 제출
bool pool_submit(ThreadPool* pool, void (*function)(void*), void* arg) {
    Task task = { .function = function, .arg = arg };
    return queue_push(&pool->queue, task);
}

// 스레드 풀 종료
void pool_shutdown(ThreadPool* pool) {
    pthread_mutex_lock(&pool->queue.mutex);
    pool->queue.shutdown = true;
    pthread_cond_broadcast(&pool->queue.not_empty);
    pthread_mutex_unlock(&pool->queue.mutex);

    for (int i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    queue_destroy(&pool->queue);
    free(pool);
}

// ============ 테스트 ============

typedef struct {
    int id;
    int value;
} WorkItem;

void process_work(void* arg) {
    WorkItem* item = (WorkItem*)arg;

    printf("작업 %d 처리 중 (값: %d)...\n", item->id, item->value);
    usleep((rand() % 500 + 100) * 1000);  // 100~600ms 처리
    printf("작업 %d 완료!\n", item->id);

    free(item);
}

int main(void) {
    srand(time(NULL));

    printf("스레드 풀 생성 (크기: %d)\n\n", POOL_SIZE);
    ThreadPool* pool = pool_create(POOL_SIZE);

    // 작업 제출
    for (int i = 0; i < 10; i++) {
        WorkItem* item = malloc(sizeof(WorkItem));
        item->id = i;
        item->value = rand() % 100;

        printf("작업 %d 제출 (값: %d)\n", i, item->value);
        pool_submit(pool, process_work, item);

        usleep(100000);  // 100ms 간격
    }

    printf("\n모든 작업 제출 완료. 풀 종료 대기...\n\n");
    sleep(2);  // 작업 처리 대기

    pool_shutdown(pool);
    printf("\n프로그램 종료\n");

    return 0;
}
