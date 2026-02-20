"""
백트래킹 (Backtracking)
Backtracking Algorithms

해를 찾는 도중 막히면 되돌아가서 다시 해를 찾는 기법입니다.
완전 탐색을 효율적으로 수행할 수 있습니다.
"""

from typing import List


# =============================================================================
# 1. 순열 (Permutations)
# =============================================================================
def permutations(nums: List[int]) -> List[List[int]]:
    """
    배열의 모든 순열 생성
    시간복잡도: O(n! * n)
    """
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i + 1:])
            path.pop()

    backtrack([], nums)
    return result


def permutations_inplace(nums: List[int]) -> List[List[int]]:
    """순열 (swap 방식)"""
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return

        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result


# =============================================================================
# 2. 조합 (Combinations)
# =============================================================================
def combinations(n: int, k: int) -> List[List[int]]:
    """
    1~n에서 k개를 선택하는 모든 조합
    시간복잡도: O(C(n,k) * k)
    """
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        # 남은 숫자가 부족하면 가지치기
        need = k - len(path)
        available = n - start + 1
        if available < need:
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    합이 target인 조합 찾기 (같은 숫자 여러 번 사용 가능)
    """
    result = []
    candidates.sort()

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i부터 (중복 허용)
            path.pop()

    backtrack(0, [], target)
    return result


# =============================================================================
# 3. 부분 집합 (Subsets)
# =============================================================================
def subsets(nums: List[int]) -> List[List[int]]:
    """
    배열의 모든 부분 집합
    시간복잡도: O(2^n * n)
    """
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    """중복 요소가 있는 배열의 모든 부분 집합"""
    result = []
    nums.sort()

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            # 같은 레벨에서 중복 건너뛰기
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


# =============================================================================
# 4. N-Queens 문제
# =============================================================================
def solve_n_queens(n: int) -> List[List[str]]:
    """
    N-Queens: n x n 체스판에 n개의 퀸을 서로 공격하지 않게 배치
    """
    result = []
    board = [['.'] * n for _ in range(n)]

    # 열, 대각선 체크용 집합
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # 퀸 배치
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            # 복원
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result


def count_n_queens(n: int) -> int:
    """N-Queens 해의 개수만 카운트"""
    count = [0]
    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(row):
        if row == n:
            count[0] += 1
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count[0]


# =============================================================================
# 5. 스도쿠 풀기
# =============================================================================
def solve_sudoku(board: List[List[str]]) -> bool:
    """
    9x9 스도쿠 풀기 (in-place)
    성공하면 True 반환
    """
    def is_valid(row, col, num):
        # 행 체크
        if num in board[row]:
            return False

        # 열 체크
        for r in range(9):
            if board[r][col] == num:
                return False

        # 3x3 박스 체크
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False

        return True

    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num
                            if backtrack():
                                return True
                            board[row][col] = '.'
                    return False
        return True

    return backtrack()


# =============================================================================
# 6. 단어 검색 (Word Search)
# =============================================================================
def word_search(board: List[List[str]], word: str) -> bool:
    """
    2D 그리드에서 단어 찾기 (인접 칸으로만 이동)
    """
    if not board or not board[0]:
        return False

    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if board[r][c] != word[idx]:
            return False

        # 방문 표시
        temp = board[r][c]
        board[r][c] = '#'

        # 4방향 탐색
        found = (backtrack(r + 1, c, idx + 1) or
                 backtrack(r - 1, c, idx + 1) or
                 backtrack(r, c + 1, idx + 1) or
                 backtrack(r, c - 1, idx + 1))

        # 복원
        board[r][c] = temp

        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False


# =============================================================================
# 7. 괄호 생성
# =============================================================================
def generate_parentheses(n: int) -> List[str]:
    """
    n쌍의 유효한 괄호 조합 생성
    """
    result = []

    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return

        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()

        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()

    backtrack([], 0, 0)
    return result


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("백트래킹 (Backtracking) 예제")
    print("=" * 60)

    # 1. 순열
    print("\n[1] 순열 (Permutations)")
    nums = [1, 2, 3]
    perms = permutations(nums)
    print(f"    {nums}의 순열 ({len(perms)}개):")
    for p in perms[:6]:  # 처음 6개만
        print(f"    {p}")

    # 2. 조합
    print("\n[2] 조합 (Combinations)")
    n, k = 4, 2
    combs = combinations(n, k)
    print(f"    C({n}, {k}) = {len(combs)}개:")
    print(f"    {combs}")

    print("\n    조합 합 (Combination Sum)")
    candidates = [2, 3, 6, 7]
    target = 7
    result = combination_sum(candidates, target)
    print(f"    {candidates}에서 합이 {target}인 조합: {result}")

    # 3. 부분 집합
    print("\n[3] 부분 집합 (Subsets)")
    nums = [1, 2, 3]
    subs = subsets(nums)
    print(f"    {nums}의 부분집합 ({len(subs)}개): {subs}")

    print("\n    중복 포함 부분 집합")
    nums_dup = [1, 2, 2]
    subs_dup = subsets_with_dup(nums_dup)
    print(f"    {nums_dup}의 부분집합: {subs_dup}")

    # 4. N-Queens
    print("\n[4] N-Queens 문제")
    for n in [4, 8]:
        count = count_n_queens(n)
        print(f"    {n}-Queens 해의 개수: {count}")

    print("\n    4-Queens 해 예시:")
    solutions = solve_n_queens(4)
    for row in solutions[0]:
        print(f"    {row}")

    # 5. 괄호 생성
    print("\n[5] 괄호 생성")
    n = 3
    parens = generate_parentheses(n)
    print(f"    {n}쌍의 괄호 ({len(parens)}개):")
    print(f"    {parens}")

    # 6. 단어 검색
    print("\n[6] 단어 검색 (Word Search)")
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    print("    보드:")
    for row in board:
        print(f"    {row}")
    for word in ["ABCCED", "SEE", "ABCB"]:
        result = word_search([row[:] for row in board], word)
        print(f"    '{word}' 존재? {result}")

    print("\n" + "=" * 60)
    print("백트래킹 패턴 정리")
    print("=" * 60)
    print("""
    백트래킹 템플릿:

    def backtrack(상태):
        if 종료 조건:
            결과에 추가
            return

        for 선택 in 선택지:
            if 유효하지 않은 선택:
                continue  # 가지치기 (Pruning)

            선택 적용
            backtrack(다음 상태)
            선택 취소  # 백트래킹

    핵심 포인트:
    1. 상태 표현 방법 결정
    2. 종료 조건 정의
    3. 가지치기로 불필요한 탐색 줄이기
    4. 상태 복원 (원래대로 되돌리기)
    """)


if __name__ == "__main__":
    main()
