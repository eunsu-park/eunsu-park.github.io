// ansi_demo.c
// ANSI Escape Codes 시연 프로그램
// 터미널에서 커서 이동과 색상을 제어하는 방법을 보여줍니다.

#include <stdio.h>
#include <unistd.h>

// ANSI Escape Codes 정의
#define CLEAR_SCREEN "\033[2J"
#define CURSOR_HOME "\033[H"
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"

// 커서 이동: \033[row;colH
#define MOVE_CURSOR(row, col) printf("\033[%d;%dH", row, col)

// 색상 코드
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"

// 배경색 코드
#define BG_RED "\033[41m"
#define BG_GREEN "\033[42m"
#define BG_YELLOW "\033[43m"
#define BG_BLUE "\033[44m"

// 텍스트 스타일
#define BOLD "\033[1m"
#define UNDERLINE "\033[4m"

int main(void) {
    // 화면 지우기
    printf(CLEAR_SCREEN);
    printf(CURSOR_HOME);

    // 커서 숨기기
    printf(HIDE_CURSOR);

    // 제목 표시
    MOVE_CURSOR(2, 10);
    printf(BOLD COLOR_CYAN "=== ANSI Escape Codes 시연 ===" COLOR_RESET);

    // 여러 위치에 색상 텍스트 출력
    MOVE_CURSOR(5, 10);
    printf(COLOR_RED "빨간색 텍스트" COLOR_RESET);

    MOVE_CURSOR(7, 10);
    printf(COLOR_GREEN "초록색 텍스트" COLOR_RESET);

    MOVE_CURSOR(9, 10);
    printf(COLOR_BLUE "파란색 텍스트" COLOR_RESET);

    MOVE_CURSOR(11, 10);
    printf(BOLD COLOR_YELLOW "굵은 노란색 텍스트" COLOR_RESET);

    MOVE_CURSOR(13, 10);
    printf(UNDERLINE COLOR_MAGENTA "밑줄 마젠타 텍스트" COLOR_RESET);

    // 배경색 예제
    MOVE_CURSOR(15, 10);
    printf(BG_RED "    배경색 빨강    " COLOR_RESET);

    MOVE_CURSOR(16, 10);
    printf(BG_GREEN "    배경색 초록    " COLOR_RESET);

    // 박스 그리기 (UTF-8 박스 문자 사용)
    MOVE_CURSOR(19, 5);
    printf(COLOR_CYAN "┌────────────────────┐" COLOR_RESET);
    for (int i = 20; i < 25; i++) {
        MOVE_CURSOR(i, 5);
        printf(COLOR_CYAN "│                    │" COLOR_RESET);
    }
    MOVE_CURSOR(25, 5);
    printf(COLOR_CYAN "└────────────────────┘" COLOR_RESET);

    MOVE_CURSOR(22, 10);
    printf(COLOR_YELLOW "박스 안의 텍스트" COLOR_RESET);

    // 그리드 패턴 그리기
    MOVE_CURSOR(19, 40);
    printf(BOLD "그리드 패턴:" COLOR_RESET);
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 10; col++) {
            MOVE_CURSOR(20 + row, 40 + col * 2);
            if ((row + col) % 2 == 0) {
                printf(COLOR_GREEN "■" COLOR_RESET);
            } else {
                printf(COLOR_RED "□" COLOR_RESET);
            }
        }
    }

    // 애니메이션 효과 (진행 바)
    MOVE_CURSOR(27, 5);
    printf("로딩 중: [");
    for (int i = 0; i < 20; i++) {
        printf(COLOR_GREEN "█" COLOR_RESET);
        fflush(stdout);
        usleep(100000); // 100ms 대기
    }
    printf("]");

    // 안내 메시지
    MOVE_CURSOR(29, 5);
    printf(COLOR_YELLOW "3초 후 자동으로 종료됩니다..." COLOR_RESET);

    fflush(stdout);
    sleep(3);

    // 화면 지우기 및 커서 보이기
    printf(CLEAR_SCREEN);
    printf(SHOW_CURSOR);
    MOVE_CURSOR(1, 1);

    printf("ANSI Escape Codes 시연을 마칩니다.\n");

    return 0;
}
