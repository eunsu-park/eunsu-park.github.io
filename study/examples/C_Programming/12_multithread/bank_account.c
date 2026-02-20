// bank_account.c
// 뮤텍스를 이용한 스레드 안전 은행 계좌
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

typedef struct {
    int balance;
    pthread_mutex_t lock;
} Account;

Account* account_create(int initial_balance) {
    Account* acc = malloc(sizeof(Account));
    acc->balance = initial_balance;
    pthread_mutex_init(&acc->lock, NULL);
    return acc;
}

void account_destroy(Account* acc) {
    pthread_mutex_destroy(&acc->lock);
    free(acc);
}

int account_deposit(Account* acc, int amount) {
    pthread_mutex_lock(&acc->lock);

    acc->balance += amount;
    int new_balance = acc->balance;

    pthread_mutex_unlock(&acc->lock);
    return new_balance;
}

int account_withdraw(Account* acc, int amount) {
    pthread_mutex_lock(&acc->lock);

    if (acc->balance >= amount) {
        acc->balance -= amount;
        int new_balance = acc->balance;
        pthread_mutex_unlock(&acc->lock);
        return new_balance;
    }

    pthread_mutex_unlock(&acc->lock);
    return -1;  // 잔액 부족
}

int account_get_balance(Account* acc) {
    pthread_mutex_lock(&acc->lock);
    int balance = acc->balance;
    pthread_mutex_unlock(&acc->lock);
    return balance;
}

// 이체 (두 계좌 간)
int account_transfer(Account* from, Account* to, int amount) {
    // 데드락 방지: 항상 같은 순서로 잠금
    // 주소값이 작은 계좌 먼저 잠금
    Account* first = (from < to) ? from : to;
    Account* second = (from < to) ? to : from;

    pthread_mutex_lock(&first->lock);
    pthread_mutex_lock(&second->lock);

    int result = -1;
    if (from->balance >= amount) {
        from->balance -= amount;
        to->balance += amount;
        result = from->balance;
    }

    pthread_mutex_unlock(&second->lock);
    pthread_mutex_unlock(&first->lock);

    return result;
}

// 테스트용 스레드 데이터
typedef struct {
    Account* acc;
    int thread_id;
} ThreadArg;

void* depositor(void* arg) {
    ThreadArg* ta = (ThreadArg*)arg;

    for (int i = 0; i < 100; i++) {
        int new_balance = account_deposit(ta->acc, 100);
        printf("[입금자 %d] 입금 100원 -> 잔액: %d\n", ta->thread_id, new_balance);
        usleep(rand() % 10000);
    }

    return NULL;
}

void* withdrawer(void* arg) {
    ThreadArg* ta = (ThreadArg*)arg;

    for (int i = 0; i < 100; i++) {
        int result = account_withdraw(ta->acc, 100);
        if (result >= 0) {
            printf("[출금자 %d] 출금 100원 -> 잔액: %d\n", ta->thread_id, result);
        } else {
            printf("[출금자 %d] 잔액 부족\n", ta->thread_id);
        }
        usleep(rand() % 10000);
    }

    return NULL;
}

int main(void) {
    srand(time(NULL));

    Account* acc = account_create(10000);
    printf("초기 잔액: %d\n\n", account_get_balance(acc));

    pthread_t depositors[3];
    pthread_t withdrawers[3];
    ThreadArg args[6];

    // 입금자 3명
    for (int i = 0; i < 3; i++) {
        args[i].acc = acc;
        args[i].thread_id = i;
        pthread_create(&depositors[i], NULL, depositor, &args[i]);
    }

    // 출금자 3명
    for (int i = 0; i < 3; i++) {
        args[i + 3].acc = acc;
        args[i + 3].thread_id = i;
        pthread_create(&withdrawers[i], NULL, withdrawer, &args[i + 3]);
    }

    // 대기
    for (int i = 0; i < 3; i++) {
        pthread_join(depositors[i], NULL);
        pthread_join(withdrawers[i], NULL);
    }

    printf("\n최종 잔액: %d\n", account_get_balance(acc));
    printf("예상 잔액: %d (초기 10000 + 입금 30000 - 출금 최대 30000)\n", 10000);

    account_destroy(acc);
    return 0;
}
