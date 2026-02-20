// calculator.c
// 기본 계산기 프로그램

#include <stdio.h>
#include <stdlib.h>

// 덧셈
double add(double a, double b) {
    return a + b;
}

// 뺄셈
double subtract(double a, double b) {
    return a - b;
}

// 곱셈
double multiply(double a, double b) {
    return a * b;
}

// 나눗셈
double divide(double a, double b) {
    if (b == 0) {
        printf("오류: 0으로 나눌 수 없습니다.\n");
        return 0;
    }
    return a / b;
}

// 메뉴 출력
void print_menu(void) {
    printf("\n=== 계산기 ===\n");
    printf("1. 덧셈 (+)\n");
    printf("2. 뺄셈 (-)\n");
    printf("3. 곱셈 (*)\n");
    printf("4. 나눗셈 (/)\n");
    printf("5. 종료\n");
    printf("선택: ");
}

int main(void) {
    int choice;
    double num1, num2, result;

    printf("간단한 계산기 프로그램\n");

    while (1) {
        print_menu();

        if (scanf("%d", &choice) != 1) {
            printf("잘못된 입력입니다.\n");
            // 입력 버퍼 비우기
            while (getchar() != '\n');
            continue;
        }

        if (choice == 5) {
            printf("프로그램을 종료합니다.\n");
            break;
        }

        if (choice < 1 || choice > 5) {
            printf("1-5 사이의 숫자를 입력하세요.\n");
            continue;
        }

        // 두 숫자 입력
        printf("첫 번째 숫자: ");
        if (scanf("%lf", &num1) != 1) {
            printf("잘못된 입력입니다.\n");
            while (getchar() != '\n');
            continue;
        }

        printf("두 번째 숫자: ");
        if (scanf("%lf", &num2) != 1) {
            printf("잘못된 입력입니다.\n");
            while (getchar() != '\n');
            continue;
        }

        // 계산 수행
        switch (choice) {
            case 1:
                result = add(num1, num2);
                printf("%.2lf + %.2lf = %.2lf\n", num1, num2, result);
                break;
            case 2:
                result = subtract(num1, num2);
                printf("%.2lf - %.2lf = %.2lf\n", num1, num2, result);
                break;
            case 3:
                result = multiply(num1, num2);
                printf("%.2lf * %.2lf = %.2lf\n", num1, num2, result);
                break;
            case 4:
                result = divide(num1, num2);
                if (num2 != 0) {
                    printf("%.2lf / %.2lf = %.2lf\n", num1, num2, result);
                }
                break;
        }
    }

    return 0;
}
