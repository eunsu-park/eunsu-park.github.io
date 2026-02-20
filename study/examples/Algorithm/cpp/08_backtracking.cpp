/*
 * 백트래킹 (Backtracking)
 * N-Queens, Permutations, Combinations, Sudoku
 *
 * 모든 가능성을 탐색하되 불필요한 경로를 가지치기합니다.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

// =============================================================================
// 1. N-Queens
// =============================================================================

class NQueens {
private:
    int n;
    vector<vector<string>> solutions;

    bool isSafe(vector<string>& board, int row, int col) {
        // 같은 열 검사
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') return false;
        }

        // 왼쪽 위 대각선
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') return false;
        }

        // 오른쪽 위 대각선
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') return false;
        }

        return true;
    }

    void solve(vector<string>& board, int row) {
        if (row == n) {
            solutions.push_back(board);
            return;
        }

        for (int col = 0; col < n; col++) {
            if (isSafe(board, row, col)) {
                board[row][col] = 'Q';
                solve(board, row + 1);
                board[row][col] = '.';
            }
        }
    }

public:
    vector<vector<string>> solveNQueens(int n) {
        this->n = n;
        solutions.clear();
        vector<string> board(n, string(n, '.'));
        solve(board, 0);
        return solutions;
    }
};

// =============================================================================
// 2. 순열 (Permutations)
// =============================================================================

class Permutations {
private:
    vector<vector<int>> result;

    void backtrack(vector<int>& nums, int start) {
        if (start == (int)nums.size()) {
            result.push_back(nums);
            return;
        }

        for (int i = start; i < (int)nums.size(); i++) {
            swap(nums[start], nums[i]);
            backtrack(nums, start + 1);
            swap(nums[start], nums[i]);
        }
    }

public:
    vector<vector<int>> permute(vector<int>& nums) {
        result.clear();
        backtrack(nums, 0);
        return result;
    }
};

// 중복 있는 순열
class PermutationsUnique {
private:
    vector<vector<int>> result;

    void backtrack(vector<int>& nums, vector<bool>& used, vector<int>& current) {
        if (current.size() == nums.size()) {
            result.push_back(current);
            return;
        }

        for (int i = 0; i < (int)nums.size(); i++) {
            if (used[i]) continue;
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;

            used[i] = true;
            current.push_back(nums[i]);
            backtrack(nums, used, current);
            current.pop_back();
            used[i] = false;
        }
    }

public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        result.clear();
        sort(nums.begin(), nums.end());
        vector<bool> used(nums.size(), false);
        vector<int> current;
        backtrack(nums, used, current);
        return result;
    }
};

// =============================================================================
// 3. 조합 (Combinations)
// =============================================================================

class Combinations {
private:
    vector<vector<int>> result;

    void backtrack(int n, int k, int start, vector<int>& current) {
        if ((int)current.size() == k) {
            result.push_back(current);
            return;
        }

        // 가지치기: 남은 원소가 부족하면 중단
        for (int i = start; i <= n - (k - (int)current.size()) + 1; i++) {
            current.push_back(i);
            backtrack(n, k, i + 1, current);
            current.pop_back();
        }
    }

public:
    vector<vector<int>> combine(int n, int k) {
        result.clear();
        vector<int> current;
        backtrack(n, k, 1, current);
        return result;
    }
};

// 합이 target인 조합
class CombinationSum {
private:
    vector<vector<int>> result;

    void backtrack(vector<int>& candidates, int target, int start, vector<int>& current) {
        if (target == 0) {
            result.push_back(current);
            return;
        }

        for (int i = start; i < (int)candidates.size() && candidates[i] <= target; i++) {
            current.push_back(candidates[i]);
            backtrack(candidates, target - candidates[i], i, current);  // 같은 원소 재사용 가능
            current.pop_back();
        }
    }

public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        result.clear();
        sort(candidates.begin(), candidates.end());
        vector<int> current;
        backtrack(candidates, target, 0, current);
        return result;
    }
};

// =============================================================================
// 4. 부분집합 (Subsets)
// =============================================================================

class Subsets {
private:
    vector<vector<int>> result;

    void backtrack(vector<int>& nums, int start, vector<int>& current) {
        result.push_back(current);

        for (int i = start; i < (int)nums.size(); i++) {
            current.push_back(nums[i]);
            backtrack(nums, i + 1, current);
            current.pop_back();
        }
    }

public:
    vector<vector<int>> subsets(vector<int>& nums) {
        result.clear();
        vector<int> current;
        backtrack(nums, 0, current);
        return result;
    }
};

// =============================================================================
// 5. 스도쿠
// =============================================================================

class Sudoku {
private:
    bool isValid(vector<vector<char>>& board, int row, int col, char c) {
        for (int i = 0; i < 9; i++) {
            if (board[row][i] == c) return false;
            if (board[i][col] == c) return false;
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) return false;
        }
        return true;
    }

    bool solve(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {
                        if (isValid(board, i, j, c)) {
                            board[i][j] = c;
                            if (solve(board)) return true;
                            board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

public:
    void solveSudoku(vector<vector<char>>& board) {
        solve(board);
    }
};

// =============================================================================
// 6. 단어 검색 (Word Search)
// =============================================================================

class WordSearch {
private:
    int rows, cols;
    int dx[4] = {0, 0, 1, -1};
    int dy[4] = {1, -1, 0, 0};

    bool dfs(vector<vector<char>>& board, const string& word, int idx, int r, int c) {
        if (idx == (int)word.size()) return true;
        if (r < 0 || r >= rows || c < 0 || c >= cols) return false;
        if (board[r][c] != word[idx]) return false;

        char temp = board[r][c];
        board[r][c] = '#';  // 방문 표시

        for (int d = 0; d < 4; d++) {
            if (dfs(board, word, idx + 1, r + dx[d], c + dy[d])) {
                board[r][c] = temp;
                return true;
            }
        }

        board[r][c] = temp;
        return false;
    }

public:
    bool exist(vector<vector<char>>& board, const string& word) {
        rows = board.size();
        cols = board[0].size();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (dfs(board, word, 0, i, j)) return true;
            }
        }
        return false;
    }
};

// =============================================================================
// 테스트
// =============================================================================

void printVector(const vector<int>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i < v.size() - 1) cout << ",";
    }
    cout << "]";
}

int main() {
    cout << "============================================================" << endl;
    cout << "백트래킹 예제" << endl;
    cout << "============================================================" << endl;

    // 1. N-Queens
    cout << "\n[1] N-Queens" << endl;
    NQueens nq;
    auto queens = nq.solveNQueens(4);
    cout << "    4-Queens 해의 개수: " << queens.size() << endl;
    if (!queens.empty()) {
        cout << "    첫 번째 해:" << endl;
        for (const auto& row : queens[0]) {
            cout << "      " << row << endl;
        }
    }

    // 2. 순열
    cout << "\n[2] 순열" << endl;
    vector<int> nums = {1, 2, 3};
    Permutations perm;
    auto perms = perm.permute(nums);
    cout << "    [1,2,3]의 순열 (" << perms.size() << "개):" << endl;
    cout << "    ";
    for (const auto& p : perms) {
        printVector(p);
        cout << " ";
    }
    cout << endl;

    // 3. 조합
    cout << "\n[3] 조합" << endl;
    Combinations comb;
    auto combs = comb.combine(4, 2);
    cout << "    C(4,2) (" << combs.size() << "개):" << endl;
    cout << "    ";
    for (const auto& c : combs) {
        printVector(c);
        cout << " ";
    }
    cout << endl;

    // 4. 조합 합
    cout << "\n[4] 조합 합" << endl;
    vector<int> candidates = {2, 3, 6, 7};
    CombinationSum cs;
    auto combSums = cs.combinationSum(candidates, 7);
    cout << "    [2,3,6,7], target=7:" << endl;
    cout << "    ";
    for (const auto& c : combSums) {
        printVector(c);
        cout << " ";
    }
    cout << endl;

    // 5. 부분집합
    cout << "\n[5] 부분집합" << endl;
    vector<int> nums2 = {1, 2, 3};
    Subsets sub;
    auto subs = sub.subsets(nums2);
    cout << "    [1,2,3]의 부분집합 (" << subs.size() << "개):" << endl;
    cout << "    ";
    for (const auto& s : subs) {
        printVector(s);
        cout << " ";
    }
    cout << endl;

    // 6. 단어 검색
    cout << "\n[6] 단어 검색" << endl;
    vector<vector<char>> board = {
        {'A', 'B', 'C', 'E'},
        {'S', 'F', 'C', 'S'},
        {'A', 'D', 'E', 'E'}
    };
    WordSearch ws;
    cout << "    \"ABCCED\": " << (ws.exist(board, "ABCCED") ? "있음" : "없음") << endl;
    cout << "    \"SEE\": " << (ws.exist(board, "SEE") ? "있음" : "없음") << endl;
    cout << "    \"ABCB\": " << (ws.exist(board, "ABCB") ? "있음" : "없음") << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 문제           | 시간복잡도    |" << endl;
    cout << "    |----------------|---------------|" << endl;
    cout << "    | N-Queens       | O(N!)         |" << endl;
    cout << "    | 순열           | O(N!)         |" << endl;
    cout << "    | 조합 C(n,k)    | O(C(n,k))     |" << endl;
    cout << "    | 부분집합       | O(2^N)        |" << endl;
    cout << "    | 스도쿠         | O(9^(빈칸수)) |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
