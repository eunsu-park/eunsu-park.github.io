// snake.c
// 뱀 게임 핵심 로직 구현
// 뱀의 생성, 이동, 충돌 검사 등의 기능을 제공합니다.

#include <stdio.h>
#include <stdlib.h>
#include "snake_types.h"

// ============ 뱀 생성 및 해제 ============

/**
 * 뱀 생성
 *
 * @param start_x 시작 X 좌표 (머리 위치)
 * @param start_y 시작 Y 좌표 (머리 위치)
 * @param initial_dir 초기 이동 방향
 * @return 생성된 뱀 포인터 (실패 시 NULL)
 */
Snake* snake_create(int start_x, int start_y, Direction initial_dir) {
    Snake* snake = malloc(sizeof(Snake));
    if (!snake) {
        return NULL;
    }

    // 뱀 초기화
    snake->head = NULL;
    snake->tail = NULL;
    snake->length = 0;
    snake->dir = initial_dir;

    // 초기 몸통 생성 (기본 3칸)
    // 머리부터 꼬리 순서로 추가
    for (int i = 0; i < INITIAL_SNAKE_LENGTH; i++) {
        SnakeNode* node = malloc(sizeof(SnakeNode));
        if (!node) {
            // 메모리 할당 실패 시 이미 생성된 노드들 해제
            snake_destroy(snake);
            return NULL;
        }

        // 방향에 따라 초기 위치 설정
        node->pos.x = start_x;
        node->pos.y = start_y;

        switch (initial_dir) {
            case DIR_RIGHT:
                node->pos.x -= i;
                break;
            case DIR_LEFT:
                node->pos.x += i;
                break;
            case DIR_DOWN:
                node->pos.y -= i;
                break;
            case DIR_UP:
                node->pos.y += i;
                break;
        }

        node->next = NULL;

        // 연결 리스트에 추가
        if (snake->head == NULL) {
            // 첫 번째 노드
            snake->head = node;
            snake->tail = node;
        } else {
            // 꼬리에 추가
            snake->tail->next = node;
            snake->tail = node;
        }
        snake->length++;
    }

    return snake;
}

/**
 * 뱀 메모리 해제
 *
 * @param snake 해제할 뱀 포인터
 */
void snake_destroy(Snake* snake) {
    if (!snake) return;

    SnakeNode* current = snake->head;
    while (current) {
        SnakeNode* next = current->next;
        free(current);
        current = next;
    }
    free(snake);
}

// ============ 뱀 제어 ============

/**
 * 뱀의 방향 변경
 * 반대 방향으로는 변경할 수 없음 (자기 몸과 충돌 방지)
 *
 * @param snake 뱀 포인터
 * @param new_dir 새로운 방향
 */
void snake_change_direction(Snake* snake, Direction new_dir) {
    if (!snake) return;

    // 현재 진행방향의 반대로는 못 감
    if (IS_OPPOSITE_DIR(snake->dir, new_dir)) {
        return;
    }

    snake->dir = new_dir;
}

/**
 * 다음 머리 위치 계산 (실제 이동은 하지 않음)
 *
 * @param snake 뱀 포인터
 * @return 다음 머리가 위치할 좌표
 */
Point snake_next_head_position(const Snake* snake) {
    Point next = snake->head->pos;

    switch (snake->dir) {
        case DIR_UP:
            next.y--;
            break;
        case DIR_DOWN:
            next.y++;
            break;
        case DIR_LEFT:
            next.x--;
            break;
        case DIR_RIGHT:
            next.x++;
            break;
    }

    return next;
}

/**
 * 뱀 이동
 * 머리를 한 칸 앞으로 이동하고, 음식을 먹지 않았으면 꼬리 제거
 *
 * @param snake 뱀 포인터
 * @param food_pos 음식 위치
 * @return true = 음식을 먹음, false = 음식을 먹지 않음
 */
bool snake_move(Snake* snake, Point food_pos) {
    if (!snake || !snake->head) return false;

    // 다음 머리 위치 계산
    Point next = snake_next_head_position(snake);

    // 새 머리 노드 생성
    SnakeNode* new_head = malloc(sizeof(SnakeNode));
    if (!new_head) {
        return false;  // 메모리 할당 실패
    }

    new_head->pos = next;
    new_head->next = snake->head;
    snake->head = new_head;
    snake->length++;

    // 음식을 먹었는지 확인
    if (POINT_EQUALS(next, food_pos)) {
        // 음식을 먹었으면 꼬리를 유지 (길이 증가)
        return true;
    }

    // 음식을 먹지 않았으면 꼬리 제거
    if (snake->length > 1) {
        // 꼬리 이전 노드 찾기
        SnakeNode* current = snake->head;
        while (current->next != snake->tail) {
            current = current->next;
        }

        // 꼬리 제거
        free(snake->tail);
        snake->tail = current;
        snake->tail->next = NULL;
        snake->length--;
    }

    return false;
}

// ============ 충돌 검사 ============

/**
 * 뱀이 벽에 부딪혔는지 확인
 *
 * @param snake 뱀 포인터
 * @return true = 벽 충돌, false = 충돌 없음
 */
bool snake_hits_wall(const Snake* snake) {
    if (!snake || !snake->head) return false;

    Point head = snake->head->pos;

    // 테두리에 닿으면 충돌
    return (head.x <= 0 || head.x >= SCREEN_WIDTH - 1 ||
            head.y <= 0 || head.y >= SCREEN_HEIGHT - 1);
}

/**
 * 뱀이 자기 몸에 부딪혔는지 확인
 *
 * @param snake 뱀 포인터
 * @return true = 자기 몸 충돌, false = 충돌 없음
 */
bool snake_hits_self(const Snake* snake) {
    if (!snake || !snake->head) return false;

    Point head = snake->head->pos;
    SnakeNode* current = snake->head->next;  // 머리 다음 노드부터 확인

    while (current) {
        if (POINT_EQUALS(head, current->pos)) {
            return true;
        }
        current = current->next;
    }

    return false;
}

/**
 * 특정 위치에 뱀이 있는지 확인
 * 음식 생성 시 뱀과 겹치지 않도록 할 때 사용
 *
 * @param snake 뱀 포인터
 * @param x X 좌표
 * @param y Y 좌표
 * @return true = 뱀이 해당 위치에 있음, false = 없음
 */
bool snake_occupies_position(const Snake* snake, int x, int y) {
    if (!snake) return false;

    SnakeNode* current = snake->head;
    while (current) {
        if (current->pos.x == x && current->pos.y == y) {
            return true;
        }
        current = current->next;
    }

    return false;
}

// ============ 유틸리티 함수 ============

/**
 * 뱀의 길이 반환
 *
 * @param snake 뱀 포인터
 * @return 뱀의 길이
 */
int snake_get_length(const Snake* snake) {
    if (!snake) return 0;
    return snake->length;
}

/**
 * 뱀 머리의 위치 반환
 *
 * @param snake 뱀 포인터
 * @return 머리 위치 좌표
 */
Point snake_get_head_position(const Snake* snake) {
    Point invalid = {-1, -1};
    if (!snake || !snake->head) {
        return invalid;
    }
    return snake->head->pos;
}
