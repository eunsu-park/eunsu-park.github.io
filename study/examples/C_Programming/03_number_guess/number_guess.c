// number_guess.c
// 숫자 맞추기 게임

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int secret, guess, attempts;
    int min = 1, max = 100;

    // 난수 시드 초기화
    srand(time(NULL));

    printf("=== 숫자 맞추기 게임 ===\n");
    printf("1부터 100 사이의 숫자를 맞춰보세요!\n\n");

    // 1-100 사이 랜덤 숫자 생성
    secret = rand() % 100 + 1;
    attempts = 0;

    while (1) {
        printf("숫자를 입력하세요 (%d ~ %d): ", min, max);

        if (scanf("%d", &guess) != 1) {
            printf("올바른 숫자를 입력하세요.\n");
            while (getchar() != '\n');  // 입력 버퍼 비우기
            continue;
        }

        attempts++;

        if (guess < min || guess > max) {
            printf("범위 내의 숫자를 입력하세요!\n");
            continue;
        }

        if (guess == secret) {
            printf("\n정답입니다! 🎉\n");
            printf("%d번 만에 맞추셨습니다.\n", attempts);
            break;
        } else if (guess < secret) {
            printf("더 큰 숫자입니다.\n");
            if (guess > min) min = guess + 1;
        } else {
            printf("더 작은 숫자입니다.\n");
            if (guess < max) max = guess - 1;
        }
    }

    return 0;
}
