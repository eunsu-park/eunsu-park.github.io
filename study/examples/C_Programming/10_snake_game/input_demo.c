// input_demo.c
// 비동기 키보드 입력 시연 프로그램
// termios를 사용하여 non-blocking 입력을 구현합니다.

#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// 원래 터미널 설정 저장
static struct termios original_termios;

// 터미널을 raw 모드로 설정
void enable_raw_mode(void) {
    // 현재 터미널 설정 저장
    tcgetattr(STDIN_FILENO, &original_termios);

    struct termios raw = original_termios;

    // 입력 플래그 수정
    // ECHO: 입력한 문자 화면에 표시 안함
    // ICANON: 라인 버퍼링 끄기 (엔터 없이 즉시 읽기)
    raw.c_lflag &= ~(ECHO | ICANON);

    // 최소 입력 문자 수: 0 (non-blocking 가능)
    raw.c_cc[VMIN] = 0;
    // 타임아웃: 0 (즉시 반환)
    raw.c_cc[VTIME] = 0;

    // 수정된 설정 적용
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

// 터미널 설정 복원
void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_termios);
}

// 키 입력 확인 (non-blocking)
// 반환: 1 = 입력 있음, 0 = 입력 없음
int kbhit(void) {
    int ch = getchar();
    if (ch != EOF) {
        ungetc(ch, stdin);  // 읽은 문자를 다시 버퍼에 넣음
        return 1;
    }
    return 0;
}

// 키 읽기 (non-blocking)
int getch(void) {
    return getchar();
}

// 방향키 및 특수키 코드
typedef enum {
    KEY_NONE = 0,
    KEY_UP,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_QUIT,
    KEY_SPACE,
    KEY_ENTER,
    KEY_OTHER
} KeyCode;

// 키 읽기 및 해석 (escape sequence 처리)
KeyCode read_key(void) {
    int ch = getchar();

    // 입력 없음
    if (ch == EOF) return KEY_NONE;

    // 종료 키
    if (ch == 'q' || ch == 'Q') return KEY_QUIT;

    // 스페이스바
    if (ch == ' ') return KEY_SPACE;

    // 엔터
    if (ch == '\n' || ch == '\r') return KEY_ENTER;

    // Escape sequence (방향키 등)
    // 방향키는 3바이트: ESC(27) + '[' + 문자
    if (ch == '\033') {
        int ch2 = getchar();
        if (ch2 == '[') {
            int ch3 = getchar();
            switch (ch3) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
        // ESC만 눌렀을 때
        return KEY_QUIT;
    }

    // WASD 키 지원
    switch (ch) {
        case 'w': case 'W': return KEY_UP;
        case 's': case 'S': return KEY_DOWN;
        case 'a': case 'A': return KEY_LEFT;
        case 'd': case 'D': return KEY_RIGHT;
    }

    return KEY_OTHER;
}

// 키 코드를 문자열로 변환
const char* keycode_to_string(KeyCode key) {
    switch (key) {
        case KEY_UP: return "위쪽";
        case KEY_DOWN: return "아래쪽";
        case KEY_LEFT: return "왼쪽";
        case KEY_RIGHT: return "오른쪽";
        case KEY_SPACE: return "스페이스";
        case KEY_ENTER: return "엔터";
        case KEY_QUIT: return "종료";
        default: return "기타";
    }
}

int main(void) {
    // raw 모드 활성화
    enable_raw_mode();
    // 프로그램 종료 시 자동으로 터미널 설정 복원
    atexit(disable_raw_mode);

    // 화면 초기화
    printf("\033[2J\033[H");  // 화면 지우기 + 커서 홈
    printf("\033[?25l");       // 커서 숨기기

    printf("=== 비동기 키보드 입력 시연 ===\n\n");
    printf("조작법:\n");
    printf("  - 방향키 또는 WASD: 이동\n");
    printf("  - 스페이스: 점프\n");
    printf("  - Q 또는 ESC: 종료\n\n");
    printf("게임 화면:\n");
    printf("┌────────────────────────────────────────┐\n");
    for (int i = 0; i < 15; i++) {
        printf("│                                        │\n");
    }
    printf("└────────────────────────────────────────┘\n");

    // 플레이어 초기 위치
    int x = 20, y = 12;
    int jump_height = 0;
    int key_count = 0;

    // 상태 표시 위치
    int status_row = 23;

    // 메인 루프
    while (1) {
        // 키 입력 읽기 (non-blocking)
        KeyCode key = read_key();

        if (key == KEY_QUIT) break;

        // 이전 위치 지우기
        printf("\033[%d;%dH ", y - jump_height, x);

        // 입력 처리
        switch (key) {
            case KEY_UP:
                if (y > 7) y--;
                key_count++;
                break;
            case KEY_DOWN:
                if (y < 20) y++;
                key_count++;
                break;
            case KEY_LEFT:
                if (x > 3) x--;
                key_count++;
                break;
            case KEY_RIGHT:
                if (x < 41) x--;
                key_count++;
                break;
            case KEY_SPACE:
                jump_height = (jump_height == 0) ? 2 : 0;
                key_count++;
                break;
            default:
                break;
        }

        // 새 위치에 플레이어 그리기
        printf("\033[%d;%dH\033[32m@\033[0m", y - jump_height, x);

        // 상태 정보 업데이트
        printf("\033[%d;1H", status_row);
        printf("위치: (%d, %d)  ", x, y);
        printf("점프: %s  ", jump_height > 0 ? "ON " : "OFF");
        printf("입력 횟수: %d  ", key_count);
        if (key != KEY_NONE) {
            printf("마지막 입력: %s    ", keycode_to_string(key));
        }

        // 화면 갱신
        fflush(stdout);

        // 프레임 레이트 제어 (50ms = 20 FPS)
        usleep(50000);
    }

    // 종료 처리
    printf("\033[2J\033[H");  // 화면 지우기
    printf("\033[?25h");       // 커서 보이기

    printf("프로그램을 종료합니다.\n");
    printf("총 입력 횟수: %d\n", key_count);

    return 0;
}
