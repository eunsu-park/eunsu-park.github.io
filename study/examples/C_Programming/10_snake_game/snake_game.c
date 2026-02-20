// snake_game.c
// 완성된 뱀 게임 구현
// ANSI escape codes를 사용한 터미널 기반 Snake 게임입니다.
//
// 컴파일: gcc -o snake_game snake_game.c
// 실행: ./snake_game

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

// ============ 게임 설정 ============
#define WIDTH 40
#define HEIGHT 20
#define INITIAL_SPEED 150000  // 마이크로초 (150ms)
#define MIN_SPEED 50000       // 최소 50ms

// ============ ANSI 제어 코드 ============
#define CLEAR "\033[2J"
#define HOME "\033[H"
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"
#define MOVE(r,c) printf("\033[%d;%dH", r, c)

// ============ ANSI 색상 코드 ============
#define RESET "\033[0m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define CYAN "\033[36m"
#define MAGENTA "\033[35m"
#define BOLD "\033[1m"

// ============ 방향 열거형 ============
typedef enum { UP, DOWN, LEFT, RIGHT } Direction;

// ============ 좌표 구조체 ============
typedef struct {
    int x, y;
} Point;

// ============ 뱀 노드 (연결 리스트) ============
typedef struct Node {
    Point pos;
    struct Node* next;
} Node;

// ============ 게임 상태 구조체 ============
typedef struct {
    Node* head;        // 뱀 머리
    Node* tail;        // 뱀 꼬리
    Direction dir;     // 현재 방향
    Point food;        // 음식 위치
    int score;         // 점수
    int length;        // 뱀 길이
    bool game_over;    // 게임 오버 플래그
    bool paused;       // 일시정지 플래그
    int speed;         // 게임 속도
    int high_score;    // 최고 점수
} Game;

// ============ 터미널 설정 ============
static struct termios orig_termios;

// 터미널 설정 복원
void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    printf(SHOW_CURSOR);
}

// Raw 모드 활성화 (non-blocking 입력)
void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disable_raw_mode);

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);  // 에코 끄기, 라인 버퍼링 끄기
    raw.c_cc[VMIN] = 0;   // 최소 입력 문자 수 0
    raw.c_cc[VTIME] = 0;  // 타임아웃 0 (즉시 반환)

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    printf(HIDE_CURSOR);
}

// ============ 입력 처리 ============

/**
 * 키보드 입력 읽기 및 방향 반환
 * ESC sequence 처리 (방향키)
 */
Direction read_direction(Direction current) {
    int ch = getchar();
    if (ch == EOF) return current;

    // ESC sequence (방향키)
    if (ch == '\033') {
        if (getchar() == '[') {
            switch (getchar()) {
                case 'A': return (current != DOWN) ? UP : current;
                case 'B': return (current != UP) ? DOWN : current;
                case 'C': return (current != LEFT) ? RIGHT : current;
                case 'D': return (current != RIGHT) ? LEFT : current;
            }
        }
    }

    // WASD 키
    switch (ch) {
        case 'w': case 'W': return (current != DOWN) ? UP : current;
        case 's': case 'S': return (current != UP) ? DOWN : current;
        case 'a': case 'A': return (current != RIGHT) ? LEFT : current;
        case 'd': case 'D': return (current != LEFT) ? RIGHT : current;
        case 'q': case 'Q': return -1;  // 종료 신호
    }

    return current;
}

// 일시정지 키 확인
int check_pause_key(void) {
    int ch = getchar();
    if (ch == 'p' || ch == 'P') return 1;
    if (ch == 'q' || ch == 'Q') return -1;
    return 0;
}

// ============ 뱀 관련 함수 ============

/**
 * 특정 위치에 뱀이 있는지 확인
 */
bool snake_at(Node* head, int x, int y) {
    for (Node* n = head; n; n = n->next) {
        if (n->pos.x == x && n->pos.y == y) return true;
    }
    return false;
}

/**
 * 음식 생성 (뱀과 겹치지 않는 위치)
 */
void spawn_food(Game* g) {
    do {
        g->food.x = 1 + rand() % (WIDTH - 2);
        g->food.y = 1 + rand() % (HEIGHT - 2);
    } while (snake_at(g->head, g->food.x, g->food.y));
}

// ============ 게임 초기화 ============

/**
 * 게임 상태 초기화
 */
Game* game_init(int high_score) {
    Game* g = malloc(sizeof(Game));
    if (!g) return NULL;

    // 뱀 초기화 (길이 3)
    g->head = NULL;
    g->tail = NULL;
    g->length = 0;

    for (int i = 0; i < 3; i++) {
        Node* n = malloc(sizeof(Node));
        if (!n) {
            // 메모리 할당 실패 시 정리
            while (g->head) {
                Node* temp = g->head;
                g->head = g->head->next;
                free(temp);
            }
            free(g);
            return NULL;
        }

        n->pos.x = WIDTH / 2 - i;
        n->pos.y = HEIGHT / 2;
        n->next = g->head;
        g->head = n;
        g->length++;
    }

    // 꼬리 찾기
    Node* curr = g->head;
    while (curr->next) curr = curr->next;
    g->tail = curr;

    // 게임 상태 초기화
    g->dir = RIGHT;
    g->score = 0;
    g->game_over = false;
    g->paused = false;
    g->speed = INITIAL_SPEED;
    g->high_score = high_score;

    spawn_food(g);
    return g;
}

/**
 * 게임 메모리 해제
 */
void game_free(Game* g) {
    if (!g) return;

    Node* n = g->head;
    while (n) {
        Node* next = n->next;
        free(n);
        n = next;
    }
    free(g);
}

// ============ 게임 업데이트 ============

/**
 * 게임 상태 업데이트
 * 반환: true = 음식을 먹음, false = 먹지 않음
 */
bool game_update(Game* g) {
    if (g->paused || g->game_over) return false;

    // 다음 머리 위치 계산
    Point next = g->head->pos;
    switch (g->dir) {
        case UP:    next.y--; break;
        case DOWN:  next.y++; break;
        case LEFT:  next.x--; break;
        case RIGHT: next.x++; break;
    }

    // 벽 충돌 검사
    if (next.x <= 0 || next.x >= WIDTH - 1 ||
        next.y <= 0 || next.y >= HEIGHT - 1) {
        g->game_over = true;
        return false;
    }

    // 자기 몸 충돌 검사
    if (snake_at(g->head, next.x, next.y)) {
        g->game_over = true;
        return false;
    }

    // 새 머리 추가
    Node* new_head = malloc(sizeof(Node));
    if (!new_head) {
        g->game_over = true;
        return false;
    }

    new_head->pos = next;
    new_head->next = g->head;
    g->head = new_head;
    g->length++;

    // 음식 확인
    if (next.x == g->food.x && next.y == g->food.y) {
        g->score += 10;
        spawn_food(g);

        // 속도 증가 (점점 빨라짐)
        if (g->speed > MIN_SPEED) {
            g->speed -= 5000;
            if (g->speed < MIN_SPEED) g->speed = MIN_SPEED;
        }

        return true;
    }

    // 음식을 먹지 않았으면 꼬리 제거
    Node* curr = g->head;
    while (curr->next && curr->next->next) {
        curr = curr->next;
    }
    if (curr->next) {
        free(curr->next);
        curr->next = NULL;
        g->tail = curr;
        g->length--;
    }

    return false;
}

// ============ 화면 그리기 ============

/**
 * 게임 테두리 그리기
 */
void draw_border(void) {
    // 상단
    MOVE(1, 1);
    printf(CYAN "╔");
    for (int i = 1; i < WIDTH - 1; i++) printf("═");
    printf("╗" RESET);

    // 좌우 측면
    for (int i = 2; i < HEIGHT; i++) {
        MOVE(i, 1);
        printf(CYAN "║" RESET);
        MOVE(i, WIDTH);
        printf(CYAN "║" RESET);
    }

    // 하단
    MOVE(HEIGHT, 1);
    printf(CYAN "╚");
    for (int i = 1; i < WIDTH - 1; i++) printf("═");
    printf("╝" RESET);
}

/**
 * 게임 화면 그리기
 */
void draw_game(Game* g) {
    printf(CLEAR HOME);

    draw_border();

    // 음식 그리기
    MOVE(g->food.y + 1, g->food.x + 1);
    printf(RED "●" RESET);

    // 뱀 그리기
    bool is_head = true;
    for (Node* n = g->head; n; n = n->next) {
        MOVE(n->pos.y + 1, n->pos.x + 1);
        if (is_head) {
            printf(BOLD GREEN "◆" RESET);  // 머리
            is_head = false;
        } else {
            printf(GREEN "■" RESET);       // 몸통
        }
    }

    // 점수 및 정보 표시
    MOVE(HEIGHT + 1, 1);
    printf(YELLOW "점수: %d  |  길이: %d  |  최고: %d" RESET,
           g->score, g->length, g->high_score);

    MOVE(HEIGHT + 2, 1);
    printf("조작: ↑↓←→ 또는 WASD  |  P: 일시정지  |  Q: 종료");

    if (g->paused) {
        MOVE(HEIGHT / 2, WIDTH / 2 - 5);
        printf(BOLD YELLOW "일시정지" RESET);
    }

    fflush(stdout);
}

/**
 * 게임 오버 화면
 */
void draw_game_over(Game* g) {
    MOVE(HEIGHT / 2 - 1, WIDTH / 2 - 5);
    printf(BOLD RED "GAME OVER!" RESET);

    MOVE(HEIGHT / 2, WIDTH / 2 - 7);
    printf("최종 점수: " YELLOW "%d" RESET, g->score);

    if (g->score > g->high_score) {
        MOVE(HEIGHT / 2 + 1, WIDTH / 2 - 6);
        printf(BOLD MAGENTA "★ 신기록! ★" RESET);
    }

    MOVE(HEIGHT / 2 + 3, WIDTH / 2 - 8);
    printf("R: 재시작  |  Q: 종료");

    fflush(stdout);
}

// ============ 최고 점수 관리 ============

#define SCORE_FILE ".snake_highscore"

int load_high_score(void) {
    FILE* f = fopen(SCORE_FILE, "r");
    if (!f) return 0;

    int score = 0;
    fscanf(f, "%d", &score);
    fclose(f);
    return score;
}

void save_high_score(int score) {
    FILE* f = fopen(SCORE_FILE, "w");
    if (f) {
        fprintf(f, "%d", score);
        fclose(f);
    }
}

// ============ 메인 함수 ============

int main(void) {
    srand(time(NULL));
    enable_raw_mode();

    int high_score = load_high_score();
    Game* game = game_init(high_score);

    if (!game) {
        fprintf(stderr, "게임 초기화 실패\n");
        return 1;
    }

    draw_game(game);

    // 메인 게임 루프
    while (1) {
        if (!game->game_over) {
            // 입력 처리
            Direction new_dir = read_direction(game->dir);

            if (new_dir == (Direction)-1) {
                // Q 키로 종료
                break;
            }

            game->dir = new_dir;

            // 일시정지 처리
            int pause_key = check_pause_key();
            if (pause_key == 1) {
                game->paused = !game->paused;
            } else if (pause_key == -1) {
                break;
            }

            // 게임 업데이트 및 그리기
            game_update(game);
            draw_game(game);

            if (game->game_over) {
                // 최고 점수 저장
                if (game->score > game->high_score) {
                    save_high_score(game->score);
                }
                draw_game_over(game);
            }
        } else {
            // 게임 오버 상태에서 키 입력 처리
            int ch = getchar();
            if (ch == 'r' || ch == 'R') {
                // 재시작
                int final_high = (game->score > game->high_score) ?
                                 game->score : game->high_score;
                game_free(game);
                game = game_init(final_high);
                if (!game) break;
                draw_game(game);
            } else if (ch == 'q' || ch == 'Q') {
                // 종료
                break;
            }
        }

        usleep(game->speed);
    }

    game_free(game);

    // 화면 정리
    printf(CLEAR HOME SHOW_CURSOR);
    MOVE(1, 1);
    printf("게임을 종료합니다. 플레이해주셔서 감사합니다!\n");

    return 0;
}
