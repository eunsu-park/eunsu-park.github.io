/*
 * 실전 문제 풀이 (Practice Problems)
 * 종합 문제 (다양한 알고리즘 조합)
 *
 * 코딩 테스트에서 자주 나오는 유형들입니다.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <climits>

using namespace std;

// =============================================================================
// 1. 부분 배열 합 (투 포인터)
// =============================================================================

// 합이 target 이상인 최소 길이 부분 배열
int minSubarrayLen(int target, const vector<int>& nums) {
    int n = nums.size();
    int left = 0, sum = 0;
    int minLen = INT_MAX;

    for (int right = 0; right < n; right++) {
        sum += nums[right];

        while (sum >= target) {
            minLen = min(minLen, right - left + 1);
            sum -= nums[left++];
        }
    }

    return minLen == INT_MAX ? 0 : minLen;
}

// =============================================================================
// 2. 작업 스케줄링 (Greedy)
// =============================================================================

struct Job {
    int deadline, profit;
};

int jobScheduling(vector<Job>& jobs) {
    sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
        return a.profit > b.profit;
    });

    int maxDeadline = 0;
    for (const auto& job : jobs) {
        maxDeadline = max(maxDeadline, job.deadline);
    }

    vector<bool> slots(maxDeadline + 1, false);
    int totalProfit = 0;

    for (const auto& job : jobs) {
        for (int t = job.deadline; t >= 1; t--) {
            if (!slots[t]) {
                slots[t] = true;
                totalProfit += job.profit;
                break;
            }
        }
    }

    return totalProfit;
}

// =============================================================================
// 3. 최소 회의실 수 (이벤트 정렬)
// =============================================================================

int minMeetingRooms(vector<pair<int, int>>& intervals) {
    vector<pair<int, int>> events;

    for (const auto& [start, end] : intervals) {
        events.push_back({start, 1});
        events.push_back({end, -1});
    }

    sort(events.begin(), events.end());

    int rooms = 0, maxRooms = 0;
    for (const auto& [time, type] : events) {
        rooms += type;
        maxRooms = max(maxRooms, rooms);
    }

    return maxRooms;
}

// =============================================================================
// 4. 팰린드롬 변환 (DP)
// =============================================================================

int minPalindromeInsertions(const string& s) {
    int n = s.length();
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            if (s[i] == s[j]) {
                dp[i][j] = dp[i + 1][j - 1];
            } else {
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[0][n - 1];
}

// =============================================================================
// 5. 섬의 개수 (DFS/BFS)
// =============================================================================

void dfsIsland(vector<vector<int>>& grid, int i, int j,
               vector<vector<bool>>& visited) {
    int rows = grid.size(), cols = grid[0].size();
    if (i < 0 || i >= rows || j < 0 || j >= cols) return;
    if (visited[i][j] || grid[i][j] == 0) return;

    visited[i][j] = true;
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int d = 0; d < 4; d++) {
        dfsIsland(grid, i + dx[d], j + dy[d], visited);
    }
}

int numIslands(vector<vector<int>>& grid) {
    if (grid.empty()) return 0;

    int rows = grid.size(), cols = grid[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    int count = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == 1 && !visited[i][j]) {
                dfsIsland(grid, i, j, visited);
                count++;
            }
        }
    }

    return count;
}

// =============================================================================
// 6. Union-Find 응용
// =============================================================================

class UnionFind {
private:
    vector<int> parent, rank_;

public:
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;

        return true;
    }
};

// 중복 연결 찾기
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    UnionFind uf(n + 1);

    for (const auto& edge : edges) {
        if (!uf.unite(edge[0], edge[1])) {
            return edge;
        }
    }

    return {};
}

// =============================================================================
// 7. LIS (Longest Increasing Subsequence)
// =============================================================================

int lengthOfLIS(const vector<int>& nums) {
    vector<int> tails;

    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }

    return tails.size();
}

// =============================================================================
// 8. 이분 탐색 응용 (파라메트릭 서치)
// =============================================================================

bool canShip(const vector<int>& weights, int capacity, int days) {
    int current = 0;
    int dayCount = 1;

    for (int w : weights) {
        if (w > capacity) return false;
        if (current + w > capacity) {
            dayCount++;
            current = w;
        } else {
            current += w;
        }
    }

    return dayCount <= days;
}

int shipWithinDays(const vector<int>& weights, int days) {
    int lo = *max_element(weights.begin(), weights.end());
    int hi = accumulate(weights.begin(), weights.end(), 0);

    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (canShip(weights, mid, days)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    return lo;
}

// =============================================================================
// 9. 단어 사다리 (BFS)
// =============================================================================

int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (wordSet.find(endWord) == wordSet.end()) return 0;

    queue<pair<string, int>> q;
    q.push({beginWord, 1});

    while (!q.empty()) {
        auto [word, dist] = q.front();
        q.pop();

        if (word == endWord) return dist;

        for (int i = 0; i < (int)word.length(); i++) {
            char original = word[i];
            for (char c = 'a'; c <= 'z'; c++) {
                word[i] = c;
                if (wordSet.find(word) != wordSet.end()) {
                    q.push({word, dist + 1});
                    wordSet.erase(word);
                }
            }
            word[i] = original;
        }
    }

    return 0;
}

// =============================================================================
// 10. 문제 풀이 전략
// =============================================================================

void printStrategy() {
    cout << "\n[10] 문제 풀이 전략" << endl;
    cout << "    1. 문제 이해: 입력/출력, 제약 조건 확인" << endl;
    cout << "    2. 예제 분석: 손으로 풀어보기" << endl;
    cout << "    3. 알고리즘 선택:" << endl;
    cout << "       - N <= 20: 완전 탐색, 비트마스크" << endl;
    cout << "       - N <= 10^3: O(N^2) DP, 브루트포스" << endl;
    cout << "       - N <= 10^5: O(N log N) 정렬, 이분탐색" << endl;
    cout << "       - N <= 10^7: O(N) 투 포인터, 해시" << endl;
    cout << "    4. 구현 및 테스트" << endl;
    cout << "    5. 엣지 케이스 확인" << endl;
}

// =============================================================================
// 테스트
// =============================================================================

void printVector(const vector<int>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "============================================================" << endl;
    cout << "실전 문제 풀이 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 부분 배열 합
    cout << "\n[1] 부분 배열 합 (투 포인터)" << endl;
    vector<int> arr1 = {2, 3, 1, 2, 4, 3};
    cout << "    배열: [2, 3, 1, 2, 4, 3], target = 7" << endl;
    cout << "    최소 길이: " << minSubarrayLen(7, arr1) << endl;

    // 2. 작업 스케줄링
    cout << "\n[2] 작업 스케줄링 (Greedy)" << endl;
    vector<Job> jobs = {{4, 20}, {1, 10}, {1, 40}, {1, 30}};
    cout << "    작업: {마감:4,이익:20}, {1,10}, {1,40}, {1,30}" << endl;
    cout << "    최대 이익: " << jobScheduling(jobs) << endl;

    // 3. 최소 회의실
    cout << "\n[3] 최소 회의실 수" << endl;
    vector<pair<int, int>> meetings = {{0, 30}, {5, 10}, {15, 20}};
    cout << "    회의: [0-30], [5-10], [15-20]" << endl;
    cout << "    최소 회의실: " << minMeetingRooms(meetings) << endl;

    // 4. 팰린드롬 변환
    cout << "\n[4] 팰린드롬 변환" << endl;
    cout << "    문자열: \"abcde\"" << endl;
    cout << "    최소 삽입: " << minPalindromeInsertions("abcde") << endl;

    // 5. 섬의 개수
    cout << "\n[5] 섬의 개수" << endl;
    vector<vector<int>> grid = {
        {1, 1, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    cout << "    그리드: 4x4" << endl;
    cout << "    섬의 개수: " << numIslands(grid) << endl;

    // 6. 중복 연결
    cout << "\n[6] 중복 연결 찾기 (Union-Find)" << endl;
    vector<vector<int>> edges = {{1, 2}, {1, 3}, {2, 3}};
    auto redundant = findRedundantConnection(edges);
    cout << "    간선: (1,2), (1,3), (2,3)" << endl;
    cout << "    중복: (" << redundant[0] << ", " << redundant[1] << ")" << endl;

    // 7. LIS
    cout << "\n[7] 최장 증가 부분수열 (LIS)" << endl;
    vector<int> arr2 = {10, 9, 2, 5, 3, 7, 101, 18};
    cout << "    배열: [10, 9, 2, 5, 3, 7, 101, 18]" << endl;
    cout << "    LIS 길이: " << lengthOfLIS(arr2) << endl;

    // 8. 이분 탐색 응용
    cout << "\n[8] 이분 탐색 응용 (배송)" << endl;
    vector<int> weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cout << "    물건 무게: [1-10]" << endl;
    cout << "    5일 내 배송 최소 용량: " << shipWithinDays(weights, 5) << endl;

    // 9. 단어 사다리
    cout << "\n[9] 단어 사다리 (BFS)" << endl;
    vector<string> wordList = {"hot", "dot", "dog", "lot", "log", "cog"};
    cout << "    hit -> cog" << endl;
    cout << "    최소 변환: " << ladderLength("hit", "cog", wordList) << endl;

    // 10. 문제 풀이 전략
    printStrategy();

    cout << "\n============================================================" << endl;

    return 0;
}
