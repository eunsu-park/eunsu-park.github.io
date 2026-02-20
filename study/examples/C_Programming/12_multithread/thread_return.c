// thread_return.c
// 스레드 반환값 받기
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void* calculate_sum(void* arg) {
    int n = *(int*)arg;

    // 동적 할당하여 결과 반환
    long* result = malloc(sizeof(long));
    *result = 0;

    for (int i = 1; i <= n; i++) {
        *result += i;
    }

    printf("스레드: 1부터 %d까지 합 계산 완료\n", n);
    return result;
}

int main(void) {
    pthread_t thread;
    int n = 100;

    pthread_create(&thread, NULL, calculate_sum, &n);

    // 반환값 받기
    void* ret_val;
    pthread_join(thread, &ret_val);

    long* result = (long*)ret_val;
    printf("결과: %ld\n", *result);

    free(result);  // 동적 할당된 메모리 해제
    return 0;
}
