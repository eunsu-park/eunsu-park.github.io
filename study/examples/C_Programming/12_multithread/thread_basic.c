// thread_basic.c
// 기본 스레드 프로그램

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// 스레드 함수
void* print_message(void* arg) {
    char* message = (char*)arg;

    for (int i = 0; i < 5; i++) {
        printf("[스레드] %s - %d\n", message, i);
        sleep(1);
    }

    return NULL;
}

int main(void) {
    pthread_t thread;
    const char* msg = "Hello from thread";

    printf("=== 기본 스레드 예제 ===\n\n");

    // 스레드 생성
    int result = pthread_create(&thread, NULL, print_message, (void*)msg);
    if (result != 0) {
        fprintf(stderr, "스레드 생성 실패: %d\n", result);
        return 1;
    }

    // 메인 스레드도 작업 수행
    for (int i = 0; i < 5; i++) {
        printf("[메인] Main thread - %d\n", i);
        sleep(1);
    }

    // 스레드 종료 대기
    pthread_join(thread, NULL);

    printf("\n모든 작업 완료\n");
    return 0;
}

/*
 * 컴파일 방법:
 * gcc thread_basic.c -o thread_basic -pthread
 *
 * 실행:
 * ./thread_basic
 *
 * 메인 스레드와 생성된 스레드가 동시에 실행됩니다.
 */
