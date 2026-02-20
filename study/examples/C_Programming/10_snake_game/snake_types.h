// snake_types.h
// 뱀 게임 데이터 구조 정의
// 게임에 필요한 모든 타입과 상수를 정의합니다.

#ifndef SNAKE_TYPES_H
#define SNAKE_TYPES_H

#include <stdbool.h>

// ============ 게임 설정 상수 ============

// 화면 크기 (테두리 포함)
#define SCREEN_WIDTH 40
#define SCREEN_HEIGHT 20

// 게임 영역 크기 (테두리 제외)
#define GAME_WIDTH (SCREEN_WIDTH - 2)
#define GAME_HEIGHT (SCREEN_HEIGHT - 2)

// 게임 속도 (마이크로초)
#define INITIAL_GAME_SPEED 150000  // 150ms
#define MIN_GAME_SPEED 50000       // 50ms (최대 속도)
#define SPEED_INCREMENT 5000       // 음식 먹을 때마다 5ms 빨라짐

// 점수
#define POINTS_PER_FOOD 10

// 뱀 초기 설정
#define INITIAL_SNAKE_LENGTH 3
#define INITIAL_SNAKE_X (SCREEN_WIDTH / 2)
#define INITIAL_SNAKE_Y (SCREEN_HEIGHT / 2)

// ============ 방향 열거형 ============

// 뱀의 이동 방향
typedef enum {
    DIR_UP,     // 위쪽
    DIR_DOWN,   // 아래쪽
    DIR_LEFT,   // 왼쪽
    DIR_RIGHT   // 오른쪽
} Direction;

// ============ 좌표 구조체 ============

// 2D 좌표를 나타내는 구조체
typedef struct {
    int x;  // X 좌표 (가로)
    int y;  // Y 좌표 (세로)
} Point;

// ============ 뱀 관련 구조체 ============

// 뱀의 몸통 노드 (연결 리스트)
// 각 노드는 뱀의 한 칸을 나타냅니다.
typedef struct SnakeNode {
    Point pos;              // 이 노드의 위치
    struct SnakeNode* next; // 다음 노드 (꼬리 방향)
} SnakeNode;

// 뱀 전체를 나타내는 구조체
typedef struct {
    SnakeNode* head;  // 뱀의 머리 (첫 번째 노드)
    SnakeNode* tail;  // 뱀의 꼬리 (마지막 노드)
    Direction dir;    // 현재 이동 방향
    int length;       // 뱀의 길이
} Snake;

// ============ 음식 구조체 ============

// 음식 (단순히 좌표만 필요)
typedef Point Food;

// ============ 게임 상태 구조체 ============

// 게임의 전체 상태를 나타내는 구조체
typedef struct {
    Snake snake;        // 뱀 객체
    Food food;          // 음식 위치
    int score;          // 현재 점수
    int speed;          // 현재 게임 속도 (마이크로초)
    bool game_over;     // 게임 오버 플래그
    bool paused;        // 일시정지 플래그
} GameState;

// ============ 유틸리티 매크로 ============

// 두 점이 같은지 확인
#define POINT_EQUALS(p1, p2) ((p1).x == (p2).x && (p1).y == (p2).y)

// 점이 게임 영역 내에 있는지 확인 (테두리 제외)
#define POINT_IN_BOUNDS(p) \
    ((p).x > 0 && (p).x < SCREEN_WIDTH - 1 && \
     (p).y > 0 && (p).y < SCREEN_HEIGHT - 1)

// 반대 방향인지 확인
#define IS_OPPOSITE_DIR(d1, d2) \
    ((d1) == DIR_UP && (d2) == DIR_DOWN) || \
    ((d1) == DIR_DOWN && (d2) == DIR_UP) || \
    ((d1) == DIR_LEFT && (d2) == DIR_RIGHT) || \
    ((d1) == DIR_RIGHT && (d2) == DIR_LEFT)

// ============ ANSI 색상 코드 ============

// 화면 제어
#define ANSI_CLEAR "\033[2J"
#define ANSI_HOME "\033[H"
#define ANSI_HIDE_CURSOR "\033[?25l"
#define ANSI_SHOW_CURSOR "\033[?25h"

// 색상
#define ANSI_RESET "\033[0m"
#define ANSI_BOLD "\033[1m"
#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_BLUE "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_CYAN "\033[36m"

// ============ 함수 선언 (snake.c에서 구현) ============

// 뱀 생성 및 해제
Snake* snake_create(int start_x, int start_y, Direction initial_dir);
void snake_destroy(Snake* snake);

// 뱀 제어
void snake_change_direction(Snake* snake, Direction new_dir);
Point snake_next_head_position(const Snake* snake);
bool snake_move(Snake* snake, Point food_pos);

// 충돌 검사
bool snake_hits_wall(const Snake* snake);
bool snake_hits_self(const Snake* snake);
bool snake_occupies_position(const Snake* snake, int x, int y);

// 유틸리티
int snake_get_length(const Snake* snake);
Point snake_get_head_position(const Snake* snake);

#endif // SNAKE_TYPES_H
