// race_condition.c
// 경쟁 조건 (Race Condition) 시연
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define ITERATIONS 100000

// 공유 변수
int counter = 0;

void* increment(void* arg) {
    (void)arg;

    for (int i = 0; i < ITERATIONS; i++) {
        counter++;  // 원자적이지 않음!
        // 실제로는: temp = counter; temp = temp + 1; counter = temp;
    }

    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];

    // 스레드 생성
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, increment, NULL);
    }

    // 대기
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // 예상: NUM_THREADS * ITERATIONS = 1,000,000
    // 실제: 그보다 적은 값 (경쟁 조건으로 인한 손실)
    printf("예상값: %d\n", NUM_THREADS * ITERATIONS);
    printf("실제값: %d\n", counter);
    printf("손실: %d\n", NUM_THREADS * ITERATIONS - counter);

    return 0;
}
