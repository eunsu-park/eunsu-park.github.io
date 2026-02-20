// producer_consumer.c
// 생산자-소비자 패턴 (경계 버퍼)
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>

#define BUFFER_SIZE 5
#define NUM_ITEMS 20

// 경계 버퍼
typedef struct {
    int buffer[BUFFER_SIZE];
    int count;      // 현재 아이템 수
    int in;         // 다음 삽입 위치
    int out;        // 다음 추출 위치

    pthread_mutex_t mutex;
    pthread_cond_t not_full;   // 버퍼가 가득 차지 않음
    pthread_cond_t not_empty;  // 버퍼가 비어있지 않음

    bool done;      // 생산 완료 플래그
} BoundedBuffer;

BoundedBuffer* buffer_create(void) {
    BoundedBuffer* bb = malloc(sizeof(BoundedBuffer));
    bb->count = 0;
    bb->in = 0;
    bb->out = 0;
    bb->done = false;

    pthread_mutex_init(&bb->mutex, NULL);
    pthread_cond_init(&bb->not_full, NULL);
    pthread_cond_init(&bb->not_empty, NULL);

    return bb;
}

void buffer_destroy(BoundedBuffer* bb) {
    pthread_mutex_destroy(&bb->mutex);
    pthread_cond_destroy(&bb->not_full);
    pthread_cond_destroy(&bb->not_empty);
    free(bb);
}

void buffer_put(BoundedBuffer* bb, int item) {
    pthread_mutex_lock(&bb->mutex);

    // 버퍼가 가득 찼으면 대기
    while (bb->count == BUFFER_SIZE) {
        printf("[생산자] 버퍼 가득 참. 대기...\n");
        pthread_cond_wait(&bb->not_full, &bb->mutex);
    }

    // 아이템 삽입
    bb->buffer[bb->in] = item;
    bb->in = (bb->in + 1) % BUFFER_SIZE;
    bb->count++;

    printf("[생산자] 아이템 %d 생산 (버퍼: %d/%d)\n",
           item, bb->count, BUFFER_SIZE);

    // 소비자에게 알림
    pthread_cond_signal(&bb->not_empty);

    pthread_mutex_unlock(&bb->mutex);
}

int buffer_get(BoundedBuffer* bb, int* item) {
    pthread_mutex_lock(&bb->mutex);

    // 버퍼가 비어있고 생산 완료 아니면 대기
    while (bb->count == 0 && !bb->done) {
        printf("[소비자] 버퍼 비어있음. 대기...\n");
        pthread_cond_wait(&bb->not_empty, &bb->mutex);
    }

    // 버퍼가 비어있고 생산 완료면 종료
    if (bb->count == 0 && bb->done) {
        pthread_mutex_unlock(&bb->mutex);
        return 0;  // 더 이상 아이템 없음
    }

    // 아이템 추출
    *item = bb->buffer[bb->out];
    bb->out = (bb->out + 1) % BUFFER_SIZE;
    bb->count--;

    printf("[소비자] 아이템 %d 소비 (버퍼: %d/%d)\n",
           *item, bb->count, BUFFER_SIZE);

    // 생산자에게 알림
    pthread_cond_signal(&bb->not_full);

    pthread_mutex_unlock(&bb->mutex);
    return 1;  // 성공
}

void buffer_set_done(BoundedBuffer* bb) {
    pthread_mutex_lock(&bb->mutex);
    bb->done = true;
    pthread_cond_broadcast(&bb->not_empty);  // 모든 소비자 깨움
    pthread_mutex_unlock(&bb->mutex);
}

// 생산자 스레드
void* producer(void* arg) {
    BoundedBuffer* bb = (BoundedBuffer*)arg;

    for (int i = 1; i <= NUM_ITEMS; i++) {
        usleep((rand() % 500) * 1000);  // 0~500ms 대기
        buffer_put(bb, i);
    }

    printf("[생산자] 생산 완료\n");
    buffer_set_done(bb);

    return NULL;
}

// 소비자 스레드
void* consumer(void* arg) {
    BoundedBuffer* bb = (BoundedBuffer*)arg;
    int item;

    while (buffer_get(bb, &item)) {
        usleep((rand() % 800) * 1000);  // 0~800ms 처리 시간
    }

    printf("[소비자] 소비 완료\n");
    return NULL;
}

int main(void) {
    srand(time(NULL));

    BoundedBuffer* bb = buffer_create();

    pthread_t prod;
    pthread_t cons[2];

    // 생산자 1명
    pthread_create(&prod, NULL, producer, bb);

    // 소비자 2명
    pthread_create(&cons[0], NULL, consumer, bb);
    pthread_create(&cons[1], NULL, consumer, bb);

    // 대기
    pthread_join(prod, NULL);
    pthread_join(cons[0], NULL);
    pthread_join(cons[1], NULL);

    buffer_destroy(bb);
    printf("\n프로그램 종료\n");

    return 0;
}
