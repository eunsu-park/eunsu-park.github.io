// multi_threads.c
// 여러 스레드 생성 예제
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

// 스레드에 전달할 데이터
typedef struct {
    int id;
    char name[32];
} ThreadData;

void* thread_func(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    printf("스레드 %d (%s) 시작\n", data->id, data->name);

    // 작업 시뮬레이션
    int sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += i;
    }

    printf("스레드 %d 완료: sum = %d\n", data->id, sum);

    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    ThreadData data[NUM_THREADS];

    // 스레드 생성
    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].id = i;
        snprintf(data[i].name, sizeof(data[i].name), "Worker-%d", i);

        int result = pthread_create(&threads[i], NULL, thread_func, &data[i]);
        if (result != 0) {
            fprintf(stderr, "스레드 %d 생성 실패\n", i);
            exit(1);
        }
    }

    printf("모든 스레드 생성 완료. 대기 중...\n");

    // 모든 스레드 대기
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("프로그램 종료\n");
    return 0;
}
