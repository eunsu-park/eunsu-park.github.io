/*
 * 게임 이론 (Game Theory)
 * Nim 게임, Sprague-Grundy, Minimax
 *
 * 2인 게임의 최적 전략을 찾는 알고리즘입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

#define MAX_N 1001

/* =============================================================================
 * 1. Nim 게임
 * ============================================================================= */

/* Nim 게임: 여러 더미에서 돌을 가져가는 게임
 * 마지막 돌을 가져가는 사람이 승리
 * XOR이 0이면 현재 플레이어 패배, 아니면 승리 */
bool nim_game(int piles[], int n) {
    int xor_sum = 0;
    for (int i = 0; i < n; i++) {
        xor_sum ^= piles[i];
    }
    return xor_sum != 0;
}

/* 승리 전략: XOR을 0으로 만드는 수 찾기 */
void nim_winning_move(int piles[], int n, int* pile_idx, int* take) {
    int xor_sum = 0;
    for (int i = 0; i < n; i++) {
        xor_sum ^= piles[i];
    }

    if (xor_sum == 0) {
        *pile_idx = -1;
        *take = -1;
        return;
    }

    for (int i = 0; i < n; i++) {
        int target = piles[i] ^ xor_sum;
        if (target < piles[i]) {
            *pile_idx = i;
            *take = piles[i] - target;
            return;
        }
    }
}

/* =============================================================================
 * 2. Sprague-Grundy 정리
 * ============================================================================= */

/* Grundy 수 (Nimber) 계산
 * mex: minimum excludant (집합에 없는 최소 비음수 정수) */
int mex(int set[], int n) {
    bool* exists = calloc(n + 1, sizeof(bool));
    for (int i = 0; i < n; i++) {
        if (set[i] <= n) exists[set[i]] = true;
    }
    int result = 0;
    while (exists[result]) result++;
    free(exists);
    return result;
}

/* 간단한 게임의 Grundy 수 (1~3개 가져갈 수 있는 게임) */
int* grundy_simple(int max_n) {
    int* grundy = calloc(max_n + 1, sizeof(int));
    grundy[0] = 0;

    for (int i = 1; i <= max_n; i++) {
        int moves[3];
        int move_count = 0;

        for (int take = 1; take <= 3 && take <= i; take++) {
            moves[move_count++] = grundy[i - take];
        }

        grundy[i] = mex(moves, move_count);
    }

    return grundy;
}

/* 일반적인 Grundy 수 계산 (가져갈 수 있는 양 배열 주어짐) */
int* grundy_general(int max_n, int allowed[], int allowed_count) {
    int* grundy = calloc(max_n + 1, sizeof(int));

    for (int i = 1; i <= max_n; i++) {
        int* moves = malloc(allowed_count * sizeof(int));
        int move_count = 0;

        for (int j = 0; j < allowed_count; j++) {
            if (allowed[j] <= i) {
                moves[move_count++] = grundy[i - allowed[j]];
            }
        }

        grundy[i] = mex(moves, move_count);
        free(moves);
    }

    return grundy;
}

/* 복합 게임의 Grundy 수 (XOR 연산) */
int combined_grundy(int grundy_values[], int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        result ^= grundy_values[i];
    }
    return result;
}

/* =============================================================================
 * 3. Minimax 알고리즘
 * ============================================================================= */

#define BOARD_SIZE 3

typedef struct {
    int board[BOARD_SIZE][BOARD_SIZE];
    int current_player;  /* 1: X, -1: O */
} TicTacToe;

void ttt_init(TicTacToe* game) {
    memset(game->board, 0, sizeof(game->board));
    game->current_player = 1;
}

int ttt_check_winner(TicTacToe* game) {
    /* 행 검사 */
    for (int i = 0; i < 3; i++) {
        int sum = game->board[i][0] + game->board[i][1] + game->board[i][2];
        if (sum == 3) return 1;
        if (sum == -3) return -1;
    }

    /* 열 검사 */
    for (int j = 0; j < 3; j++) {
        int sum = game->board[0][j] + game->board[1][j] + game->board[2][j];
        if (sum == 3) return 1;
        if (sum == -3) return -1;
    }

    /* 대각선 검사 */
    int diag1 = game->board[0][0] + game->board[1][1] + game->board[2][2];
    int diag2 = game->board[0][2] + game->board[1][1] + game->board[2][0];
    if (diag1 == 3 || diag2 == 3) return 1;
    if (diag1 == -3 || diag2 == -3) return -1;

    return 0;
}

bool ttt_is_full(TicTacToe* game) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (game->board[i][j] == 0) return false;
        }
    }
    return true;
}

/* Minimax with Alpha-Beta Pruning */
int minimax(TicTacToe* game, int depth, int alpha, int beta, bool is_maximizing) {
    int winner = ttt_check_winner(game);
    if (winner != 0) return winner * 10;
    if (ttt_is_full(game)) return 0;

    if (is_maximizing) {
        int max_eval = INT_MIN;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (game->board[i][j] == 0) {
                    game->board[i][j] = 1;
                    int eval = minimax(game, depth + 1, alpha, beta, false);
                    game->board[i][j] = 0;
                    if (eval > max_eval) max_eval = eval;
                    if (eval > alpha) alpha = eval;
                    if (beta <= alpha) return max_eval;
                }
            }
        }
        return max_eval;
    } else {
        int min_eval = INT_MAX;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (game->board[i][j] == 0) {
                    game->board[i][j] = -1;
                    int eval = minimax(game, depth + 1, alpha, beta, true);
                    game->board[i][j] = 0;
                    if (eval < min_eval) min_eval = eval;
                    if (eval < beta) beta = eval;
                    if (beta <= alpha) return min_eval;
                }
            }
        }
        return min_eval;
    }
}

/* 최적의 수 찾기 */
void find_best_move(TicTacToe* game, int* best_row, int* best_col) {
    int best_val = INT_MIN;
    *best_row = -1;
    *best_col = -1;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (game->board[i][j] == 0) {
                game->board[i][j] = 1;
                int move_val = minimax(game, 0, INT_MIN, INT_MAX, false);
                game->board[i][j] = 0;

                if (move_val > best_val) {
                    *best_row = i;
                    *best_col = j;
                    best_val = move_val;
                }
            }
        }
    }
}

/* =============================================================================
 * 4. 기타 유명한 게임들
 * ============================================================================= */

/* Bash 게임: n개 돌에서 1~k개 가져갈 수 있음 */
bool bash_game(int n, int k) {
    return (n % (k + 1)) != 0;
}

/* Wythoff 게임: 두 더미에서 같은 수 또는 한 더미에서만 가져갈 수 있음 */
bool wythoff_game(int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    double phi = (1.0 + sqrt(5.0)) / 2.0;
    int k = b - a;
    int cold_a = (int)(k * phi);
    return !(a == cold_a);
}

/* Euclid 게임: gcd 게임 */
bool euclid_game(int a, int b) {
    if (a < b) { int t = a; a = b; b = t; }
    if (b == 0) return false;

    int moves = 0;
    while (b > 0) {
        if (a >= 2 * b) return (moves % 2 == 0);
        a = a - b;
        if (a < b) { int t = a; a = b; b = t; }
        moves++;
    }
    return (moves % 2 == 1);
}

/* 돌 게임 (Stone Game) - DP 기반 */
int stone_game(int piles[], int n) {
    int** dp = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        dp[i] = calloc(n, sizeof(int));
        dp[i][i] = piles[i];
    }

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            int take_left = piles[i] - dp[i + 1][j];
            int take_right = piles[j] - dp[i][j - 1];
            dp[i][j] = (take_left > take_right) ? take_left : take_right;
        }
    }

    int result = dp[0][n - 1];
    for (int i = 0; i < n; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

void print_board(TicTacToe* game) {
    char symbols[] = {'O', '.', 'X'};
    for (int i = 0; i < 3; i++) {
        printf("      ");
        for (int j = 0; j < 3; j++) {
            printf("%c ", symbols[game->board[i][j] + 1]);
        }
        printf("\n");
    }
}

int main(void) {
    printf("============================================================\n");
    printf("게임 이론 예제\n");
    printf("============================================================\n");

    /* 1. Nim 게임 */
    printf("\n[1] Nim 게임\n");
    int piles1[] = {3, 4, 5};
    printf("    더미: [3, 4, 5]\n");
    printf("    XOR: %d ^ %d ^ %d = %d\n", 3, 4, 5, 3 ^ 4 ^ 5);
    printf("    선공 승리: %s\n", nim_game(piles1, 3) ? "예" : "아니오");

    int pile_idx, take;
    nim_winning_move(piles1, 3, &pile_idx, &take);
    if (pile_idx >= 0) {
        printf("    승리 전략: 더미 %d에서 %d개 가져가기\n", pile_idx, take);
    }

    int piles2[] = {1, 2, 3};
    printf("    더미: [1, 2, 3]\n");
    printf("    XOR: %d ^ %d ^ %d = %d\n", 1, 2, 3, 1 ^ 2 ^ 3);
    printf("    선공 승리: %s\n", nim_game(piles2, 3) ? "예" : "아니오");

    /* 2. Sprague-Grundy */
    printf("\n[2] Sprague-Grundy 정리\n");
    int* grundy = grundy_simple(15);
    printf("    게임: 1~3개 가져갈 수 있음\n");
    printf("    Grundy 수: ");
    for (int i = 0; i <= 10; i++) {
        printf("G(%d)=%d ", i, grundy[i]);
    }
    printf("\n");
    printf("    패턴: 0, 1, 2, 3, 0, 1, 2, 3, ... (주기 4)\n");
    free(grundy);

    /* 커스텀 게임 */
    int allowed[] = {1, 3, 4};
    grundy = grundy_general(15, allowed, 3);
    printf("    게임: 1, 3, 4개 가져갈 수 있음\n");
    printf("    Grundy 수: ");
    for (int i = 0; i <= 10; i++) {
        printf("G(%d)=%d ", i, grundy[i]);
    }
    printf("\n");
    free(grundy);

    /* 3. Minimax (틱택토) */
    printf("\n[3] Minimax 알고리즘 (틱택토)\n");
    TicTacToe game;
    ttt_init(&game);

    int row, col;
    find_best_move(&game, &row, &col);
    printf("    빈 보드에서 최적의 첫 수: (%d, %d)\n", row, col);

    /* 상황 설정 */
    game.board[0][0] = 1;   /* X */
    game.board[1][1] = -1;  /* O */
    game.board[2][2] = 1;   /* X */

    printf("    현재 상태:\n");
    print_board(&game);

    find_best_move(&game, &row, &col);
    printf("    X의 최적 수: (%d, %d)\n", row, col);

    /* 4. 기타 게임 */
    printf("\n[4] 기타 유명한 게임\n");

    printf("    Bash 게임 (n=10, k=3): %s\n",
           bash_game(10, 3) ? "선공 승리" : "후공 승리");
    printf("    Bash 게임 (n=12, k=3): %s\n",
           bash_game(12, 3) ? "선공 승리" : "후공 승리");

    printf("    Wythoff 게임 (3, 5): %s\n",
           wythoff_game(3, 5) ? "선공 승리" : "후공 승리");

    printf("    Euclid 게임 (10, 6): %s\n",
           euclid_game(10, 6) ? "선공 승리" : "후공 승리");

    /* 5. Stone Game */
    printf("\n[5] Stone Game (DP)\n");
    int stones[] = {5, 3, 4, 5};
    printf("    돌 배열: [5, 3, 4, 5]\n");
    int diff = stone_game(stones, 4);
    printf("    선공-후공 점수 차: %d\n", diff);
    printf("    선공 %s\n", diff > 0 ? "승리" : (diff < 0 ? "패배" : "무승부"));

    /* 6. 복잡도 */
    printf("\n[6] 복잡도\n");
    printf("    | 알고리즘          | 시간복잡도    |\n");
    printf("    |-------------------|---------------|\n");
    printf("    | Nim XOR          | O(n)          |\n");
    printf("    | Grundy 계산      | O(n × k)      |\n");
    printf("    | Minimax          | O(b^d)        |\n");
    printf("    | Alpha-Beta       | O(b^(d/2))    |\n");
    printf("    | Stone Game DP    | O(n²)         |\n");
    printf("    b: 분기 계수, d: 깊이, k: 가능한 이동 수\n");

    printf("\n============================================================\n");

    return 0;
}
