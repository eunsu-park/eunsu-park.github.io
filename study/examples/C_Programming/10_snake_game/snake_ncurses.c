// snake_ncurses.c
// NCurses 라이브러리를 사용한 뱀 게임
//
// *** 이 파일은 ncurses 라이브러리가 필요합니다 ***
//
// 설치 방법:
//   macOS:   brew install ncurses
//   Ubuntu:  sudo apt install libncurses5-dev
//   Fedora:  sudo dnf install ncurses-devel
//
// 컴파일:
//   macOS:   gcc -o snake_ncurses snake_ncurses.c -lncurses
//   Linux:   gcc -o snake_ncurses snake_ncurses.c -lncurses
//
// 실행:
//   ./snake_ncurses

#include <ncurses.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

// ============ 게임 설정 ============
#define WIDTH 40
#define HEIGHT 20
#define INITIAL_SPEED 150000  // 150ms
#define MIN_SPEED 50000       // 50ms

// ============ 색상 정의 ============
enum {
    COLOR_SNAKE = 1,
    COLOR_FOOD,
    COLOR_BORDER,
    COLOR_TEXT
};

// ============ 방향 열거형 ============
typedef enum { UP, DOWN, LEFT, RIGHT } Direction;

// ============ 좌표 구조체 ============
typedef struct {
    int x, y;
} Point;

// ============ 뱀 노드 ============
typedef struct Node {
    Point pos;
    struct Node* next;
} Node;

// ============ 게임 상태 ============
typedef struct {
    Node* head;
    Node* tail;
    Direction dir;
    Point food;
    int score;
    int length;
    bool game_over;
    bool paused;
    int speed;
    int high_score;
} Game;

// ============ 전역 변수 ============
WINDOW* game_win;
WINDOW* info_win;

// ============ 유틸리티 함수 ============

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
 * 음식 생성
 */
void spawn_food(Game* g) {
    do {
        g->food.x = 1 + rand() % (WIDTH - 2);
        g->food.y = 1 + rand() % (HEIGHT - 2);
    } while (snake_at(g->head, g->food.x, g->food.y));
}

// ============ 게임 초기화 ============

/**
 * NCurses 초기화
 */
void init_ncurses(void) {
    initscr();              // NCurses 시작
    cbreak();               // 라인 버퍼링 끄기
    noecho();               // 입력 문자 표시 안함
    nodelay(stdscr, TRUE);  // Non-blocking 입력
    keypad(stdscr, TRUE);   // 방향키 활성화
    curs_set(0);            // 커서 숨기기

    // 색상 초기화
    if (has_colors()) {
        start_color();
        init_pair(COLOR_SNAKE, COLOR_GREEN, COLOR_BLACK);
        init_pair(COLOR_FOOD, COLOR_RED, COLOR_BLACK);
        init_pair(COLOR_BORDER, COLOR_CYAN, COLOR_BLACK);
        init_pair(COLOR_TEXT, COLOR_YELLOW, COLOR_BLACK);
    }

    // 게임 윈도우 생성
    game_win = newwin(HEIGHT, WIDTH, 1, 2);
    info_win = newwin(3, WIDTH, HEIGHT + 2, 2);
}

/**
 * NCurses 종료
 */
void cleanup_ncurses(void) {
    delwin(game_win);
    delwin(info_win);
    endwin();
}

/**
 * 게임 초기화
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

    // 상태 초기화
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

// ============ 입력 처리 ============

/**
 * 키보드 입력 처리
 * 반환: 0 = 계속, -1 = 종료
 */
int handle_input(Game* g) {
    int ch = getch();

    if (ch == ERR) return 0;  // 입력 없음

    if (ch == 'q' || ch == 'Q') return -1;  // 종료

    if (ch == 'p' || ch == 'P') {
        g->paused = !g->paused;
        return 0;
    }

    if (g->paused || g->game_over) return 0;

    // 방향 변경 (반대 방향 불가)
    switch (ch) {
        case KEY_UP:
        case 'w':
        case 'W':
            if (g->dir != DOWN) g->dir = UP;
            break;
        case KEY_DOWN:
        case 's':
        case 'S':
            if (g->dir != UP) g->dir = DOWN;
            break;
        case KEY_LEFT:
        case 'a':
        case 'A':
            if (g->dir != RIGHT) g->dir = LEFT;
            break;
        case KEY_RIGHT:
        case 'd':
        case 'D':
            if (g->dir != LEFT) g->dir = RIGHT;
            break;
    }

    return 0;
}

// ============ 게임 업데이트 ============

/**
 * 게임 상태 업데이트
 */
bool game_update(Game* g) {
    if (g->paused || g->game_over) return false;

    // 다음 머리 위치
    Point next = g->head->pos;
    switch (g->dir) {
        case UP:    next.y--; break;
        case DOWN:  next.y++; break;
        case LEFT:  next.x--; break;
        case RIGHT: next.x++; break;
    }

    // 벽 충돌
    if (next.x <= 0 || next.x >= WIDTH - 1 ||
        next.y <= 0 || next.y >= HEIGHT - 1) {
        g->game_over = true;
        return false;
    }

    // 자기 몸 충돌
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

        // 속도 증가
        if (g->speed > MIN_SPEED) {
            g->speed -= 5000;
            if (g->speed < MIN_SPEED) g->speed = MIN_SPEED;
        }

        return true;
    }

    // 꼬리 제거
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
 * 게임 화면 그리기
 */
void draw_game(Game* g) {
    // 게임 윈도우 지우기
    werase(game_win);

    // 테두리 그리기
    wattron(game_win, COLOR_PAIR(COLOR_BORDER));
    box(game_win, 0, 0);
    wattroff(game_win, COLOR_PAIR(COLOR_BORDER));

    // 음식 그리기
    wattron(game_win, COLOR_PAIR(COLOR_FOOD) | A_BOLD);
    mvwaddch(game_win, g->food.y, g->food.x, 'O');
    wattroff(game_win, COLOR_PAIR(COLOR_FOOD) | A_BOLD);

    // 뱀 그리기
    wattron(game_win, COLOR_PAIR(COLOR_SNAKE));
    bool is_head = true;
    for (Node* n = g->head; n; n = n->next) {
        if (is_head) {
            wattron(game_win, A_BOLD);
            mvwaddch(game_win, n->pos.y, n->pos.x, '@');
            wattroff(game_win, A_BOLD);
            is_head = false;
        } else {
            mvwaddch(game_win, n->pos.y, n->pos.x, '#');
        }
    }
    wattroff(game_win, COLOR_PAIR(COLOR_SNAKE));

    // 일시정지 메시지
    if (g->paused) {
        wattron(game_win, COLOR_PAIR(COLOR_TEXT) | A_BOLD);
        mvwprintw(game_win, HEIGHT / 2, WIDTH / 2 - 5, "일시정지");
        wattroff(game_win, COLOR_PAIR(COLOR_TEXT) | A_BOLD);
    }

    // 게임 윈도우 갱신
    wrefresh(game_win);

    // 정보 윈도우 그리기
    werase(info_win);
    wattron(info_win, COLOR_PAIR(COLOR_TEXT));
    mvwprintw(info_win, 0, 1, "점수: %d  |  길이: %d  |  최고: %d",
              g->score, g->length, g->high_score);
    mvwprintw(info_win, 1, 1, "조작: ↑↓←→ / WASD  |  P: 일시정지  |  Q: 종료");
    wattroff(info_win, COLOR_PAIR(COLOR_TEXT));
    wrefresh(info_win);
}

/**
 * 게임 오버 화면
 */
void draw_game_over(Game* g) {
    wattron(game_win, COLOR_PAIR(COLOR_TEXT) | A_BOLD);

    mvwprintw(game_win, HEIGHT / 2 - 1, WIDTH / 2 - 5, "GAME OVER!");
    mvwprintw(game_win, HEIGHT / 2, WIDTH / 2 - 7, "최종 점수: %d", g->score);

    if (g->score > g->high_score) {
        mvwprintw(game_win, HEIGHT / 2 + 1, WIDTH / 2 - 6, "★ 신기록! ★");
    }

    mvwprintw(game_win, HEIGHT / 2 + 3, WIDTH / 2 - 8, "R: 재시작  |  Q: 종료");

    wattroff(game_win, COLOR_PAIR(COLOR_TEXT) | A_BOLD);
    wrefresh(game_win);
}

// ============ 최고 점수 관리 ============

#define SCORE_FILE ".snake_ncurses_highscore"

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

    init_ncurses();
    atexit(cleanup_ncurses);

    int high_score = load_high_score();
    Game* game = game_init(high_score);

    if (!game) {
        cleanup_ncurses();
        fprintf(stderr, "게임 초기화 실패\n");
        return 1;
    }

    draw_game(game);

    // 메인 게임 루프
    while (1) {
        // 입력 처리
        if (handle_input(game) == -1) {
            break;  // 종료
        }

        if (!game->game_over) {
            // 게임 업데이트
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
            // 게임 오버 상태에서 재시작 처리
            int ch = getch();
            if (ch == 'r' || ch == 'R') {
                int final_high = (game->score > game->high_score) ?
                                 game->score : game->high_score;
                game_free(game);
                game = game_init(final_high);
                if (!game) break;
                draw_game(game);
            } else if (ch == 'q' || ch == 'Q') {
                break;
            }
        }

        usleep(game->speed);
    }

    game_free(game);

    // 종료 메시지
    clear();
    mvprintw(0, 0, "게임을 종료합니다. 플레이해주셔서 감사합니다!");
    refresh();
    nodelay(stdscr, FALSE);
    getch();

    return 0;
}
