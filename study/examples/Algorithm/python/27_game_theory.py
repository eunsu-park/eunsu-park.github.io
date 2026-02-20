"""
게임 이론 (Game Theory)
Game Theory Algorithms

조합 게임 이론과 최적 전략을 다루는 알고리즘입니다.
"""

from typing import List, Dict, Tuple, Set, Optional
from functools import lru_cache


# =============================================================================
# 1. 님 게임 (Nim Game)
# =============================================================================

def nim_xor(piles: List[int]) -> int:
    """
    님 게임의 XOR 값 (Nim-sum)
    XOR != 0이면 선공 승리, XOR == 0이면 후공 승리
    """
    result = 0
    for pile in piles:
        result ^= pile
    return result


def nim_winning_move(piles: List[int]) -> Optional[Tuple[int, int]]:
    """
    님 게임에서 승리하는 수 찾기
    반환: (더미 인덱스, 남길 개수) 또는 None
    """
    xor = nim_xor(piles)

    if xor == 0:
        return None  # 패배 상태, 승리 수 없음

    for i, pile in enumerate(piles):
        target = pile ^ xor
        if target < pile:
            return (i, target)  # pile에서 target개만 남기기

    return None


def nim_game_simulation(piles: List[int], verbose: bool = False) -> int:
    """
    님 게임 시뮬레이션
    반환: 승자 (0: 선공, 1: 후공)
    """
    current = 0  # 현재 플레이어

    while max(piles) > 0:
        move = nim_winning_move(piles)

        if move is None:
            # 패배 상태: 아무 수나
            for i, pile in enumerate(piles):
                if pile > 0:
                    move = (i, pile - 1)
                    break

        pile_idx, new_count = move
        if verbose:
            print(f"    Player {current}: 더미 {pile_idx}에서 {piles[pile_idx]}→{new_count}")

        piles[pile_idx] = new_count
        current = 1 - current

    return 1 - current  # 마지막에 가져간 사람이 승리


# =============================================================================
# 2. 스프라그-그런디 정리 (Sprague-Grundy Theorem)
# =============================================================================

def mex(s: Set[int]) -> int:
    """
    Minimum Excludant
    집합에 없는 가장 작은 음이 아닌 정수
    """
    i = 0
    while i in s:
        i += 1
    return i


def calculate_grundy(position: int, moves: List[int], memo: Dict[int, int] = None) -> int:
    """
    스프라그-그런디 수 계산
    position: 현재 상태 (예: 돌의 개수)
    moves: 가능한 이동량 리스트
    """
    if memo is None:
        memo = {}

    if position in memo:
        return memo[position]

    if position == 0:
        memo[position] = 0
        return 0

    reachable = set()
    for move in moves:
        if position >= move:
            reachable.add(calculate_grundy(position - move, moves, memo))

    result = mex(reachable)
    memo[position] = result
    return result


def multi_pile_grundy(piles: List[int], moves: List[int]) -> int:
    """
    여러 더미 게임의 전체 그런디 수
    각 더미의 그런디 수를 XOR
    """
    memo = {}
    total_grundy = 0

    for pile in piles:
        grundy = calculate_grundy(pile, moves, memo)
        total_grundy ^= grundy

    return total_grundy


# =============================================================================
# 3. 변형 님 게임
# =============================================================================

def staircase_nim(stairs: List[int]) -> int:
    """
    계단 님 (Staircase Nim)
    홀수 번째 계단의 XOR = 그런디 수
    """
    xor = 0
    for i in range(0, len(stairs), 2):  # 홀수 인덱스 (0-indexed의 짝수)
        xor ^= stairs[i]
    return xor


def misere_nim(piles: List[int]) -> bool:
    """
    미제르 님 (마지막 가져가는 사람이 패배)
    반환: True면 선공 승리
    """
    xor = nim_xor(piles)
    all_one_or_zero = all(p <= 1 for p in piles)

    if all_one_or_zero:
        # 1인 더미의 개수가 홀수면 후공 승리
        ones = sum(1 for p in piles if p == 1)
        return ones % 2 == 0
    else:
        return xor != 0


def poker_nim(piles: List[int], k: int) -> bool:
    """
    포커 님: 더미에 돌을 추가할 수도 있음 (최대 k개)
    반환: True면 선공 승리
    규칙: 일반 님과 동일 (XOR != 0이면 선공 승리)
    """
    return nim_xor(piles) != 0


# =============================================================================
# 4. 미니맥스 알고리즘 (Minimax)
# =============================================================================

def minimax(position, depth: int, is_maximizing: bool,
            evaluate, get_moves, is_terminal) -> int:
    """
    미니맥스 알고리즘
    position: 현재 게임 상태
    depth: 탐색 깊이
    is_maximizing: 최대화 플레이어의 턴인지
    evaluate: 상태 평가 함수
    get_moves: 가능한 수 반환 함수
    is_terminal: 종료 상태 확인 함수
    """
    if depth == 0 or is_terminal(position):
        return evaluate(position)

    moves = get_moves(position)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            new_position = apply_move(position, move)
            eval_score = minimax(new_position, depth - 1, False,
                                evaluate, get_moves, is_terminal)
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_position = apply_move(position, move)
            eval_score = minimax(new_position, depth - 1, True,
                                evaluate, get_moves, is_terminal)
            min_eval = min(min_eval, eval_score)
        return min_eval


def apply_move(position, move):
    """수를 적용한 새 상태 반환 (추상 함수)"""
    # 구현은 게임에 따라 다름
    pass


# =============================================================================
# 5. 알파-베타 가지치기 (Alpha-Beta Pruning)
# =============================================================================

def alpha_beta(position, depth: int, alpha: float, beta: float,
               is_maximizing: bool, evaluate, get_moves, is_terminal) -> int:
    """
    알파-베타 가지치기
    alpha: 최대화 플레이어의 최선의 보장값
    beta: 최소화 플레이어의 최선의 보장값
    """
    if depth == 0 or is_terminal(position):
        return evaluate(position)

    moves = get_moves(position)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            new_position = apply_move(position, move)
            eval_score = alpha_beta(new_position, depth - 1, alpha, beta,
                                   False, evaluate, get_moves, is_terminal)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # 가지치기
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_position = apply_move(position, move)
            eval_score = alpha_beta(new_position, depth - 1, alpha, beta,
                                   True, evaluate, get_moves, is_terminal)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # 가지치기
        return min_eval


# =============================================================================
# 6. 틱택토 (Tic-Tac-Toe) 구현
# =============================================================================

class TicTacToe:
    """틱택토 게임"""

    def __init__(self):
        self.board = [[' '] * 3 for _ in range(3)]
        self.current_player = 'X'

    def get_moves(self) -> List[Tuple[int, int]]:
        """가능한 수 반환"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def make_move(self, row: int, col: int) -> bool:
        """수 두기"""
        if self.board[row][col] != ' ':
            return False
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def undo_move(self, row: int, col: int):
        """수 되돌리기"""
        self.board[row][col] = ' '
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self) -> Optional[str]:
        """승자 확인"""
        # 행
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        # 열
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        # 대각선
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        return None

    def is_terminal(self) -> bool:
        """게임 종료 확인"""
        return self.check_winner() is not None or len(self.get_moves()) == 0

    def evaluate(self) -> int:
        """상태 평가 (X 관점)"""
        winner = self.check_winner()
        if winner == 'X':
            return 10
        elif winner == 'O':
            return -10
        return 0

    def minimax(self, is_maximizing: bool) -> int:
        """미니맥스"""
        if self.is_terminal():
            return self.evaluate()

        if is_maximizing:
            max_eval = float('-inf')
            for row, col in self.get_moves():
                self.make_move(row, col)
                eval_score = self.minimax(False)
                self.undo_move(row, col)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for row, col in self.get_moves():
                self.make_move(row, col)
                eval_score = self.minimax(True)
                self.undo_move(row, col)
                min_eval = min(min_eval, eval_score)
            return min_eval

    def best_move(self) -> Tuple[int, int]:
        """최선의 수 찾기"""
        best_score = float('-inf') if self.current_player == 'X' else float('inf')
        best_move = None

        for row, col in self.get_moves():
            self.make_move(row, col)
            score = self.minimax(self.current_player == 'X')
            self.undo_move(row, col)

            if self.current_player == 'X':
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
            else:
                if score < best_score:
                    best_score = score
                    best_move = (row, col)

        return best_move

    def display(self):
        """보드 출력"""
        for i, row in enumerate(self.board):
            print("    " + " | ".join(row))
            if i < 2:
                print("    " + "-" * 9)


# =============================================================================
# 7. 돌 게임 (Stone Game)
# =============================================================================

@lru_cache(maxsize=None)
def stone_game_dp(piles: Tuple[int, ...], left: int, right: int) -> int:
    """
    돌 게임: 양 끝에서만 가져갈 수 있음
    선공이 얻을 수 있는 최대 점수 차이 반환
    """
    if left > right:
        return 0

    # 선공이 왼쪽 선택
    pick_left = piles[left] - stone_game_dp(piles, left + 1, right)
    # 선공이 오른쪽 선택
    pick_right = piles[right] - stone_game_dp(piles, left, right - 1)

    return max(pick_left, pick_right)


def stone_game(piles: List[int]) -> bool:
    """
    돌 게임: 선공이 이기면 True
    """
    n = len(piles)
    diff = stone_game_dp(tuple(piles), 0, n - 1)
    return diff > 0


# =============================================================================
# 8. 바시 게임 (Bash Game)
# =============================================================================

def bash_game(n: int, k: int) -> bool:
    """
    바시 게임: n개 돌에서 최대 k개씩 가져감
    마지막 돌을 가져가는 사람이 승리
    반환: 선공이 이기면 True
    """
    return n % (k + 1) != 0


def bash_game_optimal_move(n: int, k: int) -> int:
    """바시 게임에서 최적의 수 (가져갈 돌의 개수)"""
    if n % (k + 1) == 0:
        return 1  # 패배 상태, 아무 수나
    return n % (k + 1)


# =============================================================================
# 9. 위더프 게임 (Wythoff's Game)
# =============================================================================

def wythoff_game(a: int, b: int) -> bool:
    """
    위더프 게임: 두 더미에서 같은 개수 또는 한 더미에서 임의 개수
    마지막 돌을 가져가는 사람이 승리
    반환: 선공이 이기면 True
    """
    phi = (1 + 5 ** 0.5) / 2  # 황금비

    if a > b:
        a, b = b, a

    k = b - a
    ak = int(k * phi)

    return a != ak


# =============================================================================
# 10. 유클리드 게임 (Euclid's Game)
# =============================================================================

def euclid_game(a: int, b: int) -> bool:
    """
    유클리드 게임: 큰 수에서 작은 수의 배수를 뺌
    0을 만드는 사람이 승리
    반환: 선공이 이기면 True
    """
    if a < b:
        a, b = b, a

    if b == 0:
        return False  # 이미 끝남

    # 재귀적 분석
    turn = True  # True: 선공의 턴
    while b > 0:
        if a >= 2 * b or a == b:
            return turn
        a, b = b, a - b
        turn = not turn

    return not turn


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("게임 이론 (Game Theory) 예제")
    print("=" * 60)

    # 1. 님 게임
    print("\n[1] 님 게임 (Nim Game)")
    piles = [3, 4, 5]
    xor = nim_xor(piles)
    move = nim_winning_move(piles)
    print(f"    더미: {piles}")
    print(f"    XOR: {xor} ({'선공 승리' if xor != 0 else '후공 승리'})")
    if move:
        print(f"    승리 수: 더미 {move[0]}에서 {move[1]}개로")

    # 시뮬레이션
    print("\n    게임 시뮬레이션:")
    piles_copy = [3, 4, 5]
    winner = nim_game_simulation(piles_copy, verbose=True)
    print(f"    승자: Player {winner}")

    # 2. 스프라그-그런디
    print("\n[2] 스프라그-그런디 정리")
    moves = [1, 3, 4]  # 한 번에 1, 3, 4개 가져갈 수 있음
    memo = {}
    for n in range(10):
        g = calculate_grundy(n, moves, memo)
        print(f"    G({n}) = {g}", end="  ")
    print()

    # 여러 더미
    piles = [7, 5]
    total_g = multi_pile_grundy(piles, moves)
    print(f"    더미 {piles}, 이동 {moves}")
    print(f"    전체 그런디: {total_g} ({'선공 승리' if total_g != 0 else '후공 승리'})")

    # 3. 변형 님
    print("\n[3] 변형 님 게임")
    # 미제르 님
    piles_misere = [1, 2, 3]
    print(f"    미제르 님 {piles_misere}: 선공 {'승리' if misere_nim(piles_misere) else '패배'}")

    # 계단 님
    stairs = [3, 1, 2, 4]  # 계단 1, 2, 3, 4
    print(f"    계단 님 {stairs}: 그런디 = {staircase_nim(stairs)}")

    # 4. 틱택토
    print("\n[4] 틱택토 미니맥스")
    game = TicTacToe()
    game.board = [['X', 'O', 'X'],
                  [' ', 'O', ' '],
                  [' ', ' ', ' ']]
    game.current_player = 'X'
    print("    현재 상태:")
    game.display()
    best = game.best_move()
    print(f"    X의 최선의 수: {best}")

    # 5. 돌 게임
    print("\n[5] 돌 게임")
    piles = [5, 3, 4, 5]
    print(f"    더미: {piles}")
    result = stone_game(piles)
    diff = stone_game_dp(tuple(piles), 0, len(piles) - 1)
    print(f"    선공 {'승리' if result else '패배'} (점수 차이: {diff})")

    # 6. 바시 게임
    print("\n[6] 바시 게임")
    n, k = 10, 3
    print(f"    n={n}, k={k}")
    print(f"    선공 {'승리' if bash_game(n, k) else '패배'}")
    if bash_game(n, k):
        print(f"    최적의 수: {bash_game_optimal_move(n, k)}개 가져가기")

    # 7. 위더프 게임
    print("\n[7] 위더프 게임")
    test_cases = [(1, 2), (3, 5), (4, 7), (5, 8)]
    for a, b in test_cases:
        result = wythoff_game(a, b)
        print(f"    ({a}, {b}): 선공 {'승리' if result else '패배'}")

    # 8. 유클리드 게임
    print("\n[8] 유클리드 게임")
    test_cases = [(25, 7), (24, 10), (100, 45)]
    for a, b in test_cases:
        result = euclid_game(a, b)
        print(f"    ({a}, {b}): 선공 {'승리' if result else '패배'}")

    # 9. 알고리즘 요약
    print("\n[9] 게임 이론 알고리즘 요약")
    print("    | 게임           | 승리 조건                    |")
    print("    |----------------|------------------------------|")
    print("    | 님 게임        | XOR != 0                     |")
    print("    | 미제르 님      | 복잡한 조건                  |")
    print("    | 바시 게임      | n % (k+1) != 0               |")
    print("    | 위더프 게임    | 황금비 기반 패배 위치        |")
    print("    | 일반 게임      | 스프라그-그런디 정리         |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
