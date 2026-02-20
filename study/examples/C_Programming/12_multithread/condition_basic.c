// condition_basic.c
// 조건 변수 (Condition Variable) 기본 사용법
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool ready = false;

void* waiter(void* arg) {
    int id = *(int*)arg;

    pthread_mutex_lock(&mutex);

    while (!ready) {  // 조건이 false인 동안 대기
        printf("[대기자 %d] 조건 대기 중...\n", id);
        pthread_cond_wait(&cond, &mutex);  // 대기 (뮤텍스 해제됨)
    }
    // pthread_cond_wait에서 깨어나면 뮤텍스 다시 획득됨

    printf("[대기자 %d] 조건 만족! 작업 시작\n", id);

    pthread_mutex_unlock(&mutex);
    return NULL;
}

void* signaler(void* arg) {
    (void)arg;

    sleep(2);  // 2초 대기

    pthread_mutex_lock(&mutex);
    ready = true;
    printf("[신호자] 조건 설정 완료. 신호 전송!\n");
    pthread_cond_broadcast(&cond);  // 모든 대기자에게 신호
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(void) {
    pthread_t waiters[3];
    pthread_t sig;
    int ids[] = {1, 2, 3};

    // 대기 스레드 생성
    for (int i = 0; i < 3; i++) {
        pthread_create(&waiters[i], NULL, waiter, &ids[i]);
    }

    // 신호 스레드 생성
    pthread_create(&sig, NULL, signaler, NULL);

    // 대기
    for (int i = 0; i < 3; i++) {
        pthread_join(waiters[i], NULL);
    }
    pthread_join(sig, NULL);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
}
