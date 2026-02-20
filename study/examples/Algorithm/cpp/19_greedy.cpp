/*
 * 탐욕 알고리즘 (Greedy Algorithm)
 * Activity Selection, Huffman, Interval Scheduling, Fractional Knapsack
 *
 * 매 순간 최선의 선택을 하여 최적해를 구합니다.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <unordered_map>

using namespace std;

// =============================================================================
// 1. 활동 선택 문제
// =============================================================================

struct Activity {
    int start, end, idx;
};

vector<int> activitySelection(vector<Activity>& activities) {
    // 종료 시간 기준 정렬
    sort(activities.begin(), activities.end(),
         [](const Activity& a, const Activity& b) {
             return a.end < b.end;
         });

    vector<int> selected;
    int lastEnd = 0;

    for (const auto& act : activities) {
        if (act.start >= lastEnd) {
            selected.push_back(act.idx);
            lastEnd = act.end;
        }
    }

    return selected;
}

// =============================================================================
// 2. 회의실 배정 (최소 회의실 수)
// =============================================================================

int minMeetingRooms(vector<pair<int, int>>& intervals) {
    vector<pair<int, int>> events;

    for (const auto& [start, end] : intervals) {
        events.push_back({start, 1});   // 시작
        events.push_back({end, -1});    // 종료
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
// 3. 분할 가능 배낭 (Fractional Knapsack)
// =============================================================================

struct Item {
    int weight, value;
    double ratio() const { return (double)value / weight; }
};

double fractionalKnapsack(int W, vector<Item>& items) {
    // 가치/무게 비율로 정렬
    sort(items.begin(), items.end(),
         [](const Item& a, const Item& b) {
             return a.ratio() > b.ratio();
         });

    double totalValue = 0;
    int remaining = W;

    for (const auto& item : items) {
        if (remaining >= item.weight) {
            totalValue += item.value;
            remaining -= item.weight;
        } else {
            totalValue += item.ratio() * remaining;
            break;
        }
    }

    return totalValue;
}

// =============================================================================
// 4. 허프만 코딩
// =============================================================================

struct HuffmanNode {
    char ch;
    int freq;
    HuffmanNode *left, *right;

    HuffmanNode(char c, int f) : ch(c), freq(f), left(nullptr), right(nullptr) {}
};

struct Compare {
    bool operator()(HuffmanNode* a, HuffmanNode* b) {
        return a->freq > b->freq;
    }
};

void generateCodes(HuffmanNode* root, string code,
                   unordered_map<char, string>& codes) {
    if (!root) return;

    if (!root->left && !root->right) {
        codes[root->ch] = code.empty() ? "0" : code;
        return;
    }

    generateCodes(root->left, code + "0", codes);
    generateCodes(root->right, code + "1", codes);
}

unordered_map<char, string> huffmanCoding(const string& text) {
    // 빈도 계산
    unordered_map<char, int> freq;
    for (char c : text) freq[c]++;

    // 우선순위 큐 (최소 힙)
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, Compare> pq;
    for (auto& [c, f] : freq) {
        pq.push(new HuffmanNode(c, f));
    }

    // 트리 구축
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();

        HuffmanNode* parent = new HuffmanNode('\0', left->freq + right->freq);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }

    // 코드 생성
    unordered_map<char, string> codes;
    if (!pq.empty()) {
        generateCodes(pq.top(), "", codes);
    }

    return codes;
}

// =============================================================================
// 5. 작업 스케줄링 (Job Scheduling with Deadlines)
// =============================================================================

struct Job {
    int id, deadline, profit;
};

int jobScheduling(vector<Job>& jobs) {
    // 이익 기준 내림차순
    sort(jobs.begin(), jobs.end(),
         [](const Job& a, const Job& b) {
             return a.profit > b.profit;
         });

    int maxDeadline = 0;
    for (const auto& job : jobs) {
        maxDeadline = max(maxDeadline, job.deadline);
    }

    vector<int> slots(maxDeadline + 1, -1);  // 각 시간대 사용 여부
    int totalProfit = 0;

    for (const auto& job : jobs) {
        // 마감 직전부터 빈 슬롯 찾기
        for (int t = job.deadline; t >= 1; t--) {
            if (slots[t] == -1) {
                slots[t] = job.id;
                totalProfit += job.profit;
                break;
            }
        }
    }

    return totalProfit;
}

// =============================================================================
// 6. 최소 동전 거스름돈
// =============================================================================

int minCoins(vector<int>& coins, int amount) {
    // 큰 동전부터 사용 (탐욕적 접근 - 특정 화폐에서만 최적)
    sort(coins.rbegin(), coins.rend());

    int count = 0;
    for (int coin : coins) {
        count += amount / coin;
        amount %= coin;
    }

    return amount == 0 ? count : -1;
}

// =============================================================================
// 7. 구간 커버 문제
// =============================================================================

int intervalCover(vector<pair<int, int>>& intervals, int target) {
    // 시작점 기준 정렬
    sort(intervals.begin(), intervals.end());

    int count = 0;
    int current = 0;
    int i = 0;
    int n = intervals.size();

    while (current < target) {
        int maxEnd = current;

        // 현재 위치를 포함하는 구간 중 가장 멀리 가는 것 선택
        while (i < n && intervals[i].first <= current) {
            maxEnd = max(maxEnd, intervals[i].second);
            i++;
        }

        if (maxEnd == current) {
            return -1;  // 커버 불가
        }

        current = maxEnd;
        count++;
    }

    return count;
}

// =============================================================================
// 8. 점프 게임
// =============================================================================

bool canJump(const vector<int>& nums) {
    int maxReach = 0;

    for (int i = 0; i < (int)nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
    }

    return true;
}

int minJumps(const vector<int>& nums) {
    int n = nums.size();
    if (n <= 1) return 0;

    int jumps = 0;
    int currentEnd = 0;
    int farthest = 0;

    for (int i = 0; i < n - 1; i++) {
        farthest = max(farthest, i + nums[i]);

        if (i == currentEnd) {
            jumps++;
            currentEnd = farthest;

            if (currentEnd >= n - 1) break;
        }
    }

    return jumps;
}

// =============================================================================
// 9. 주유소 (Gas Station)
// =============================================================================

int canCompleteCircuit(const vector<int>& gas, const vector<int>& cost) {
    int n = gas.size();
    int totalTank = 0;
    int currTank = 0;
    int startStation = 0;

    for (int i = 0; i < n; i++) {
        int diff = gas[i] - cost[i];
        totalTank += diff;
        currTank += diff;

        if (currTank < 0) {
            startStation = i + 1;
            currTank = 0;
        }
    }

    return totalTank >= 0 ? startStation : -1;
}

// =============================================================================
// 10. 문자열 분할
// =============================================================================

vector<int> partitionLabels(const string& s) {
    // 각 문자의 마지막 등장 위치
    vector<int> last(26, 0);
    for (int i = 0; i < (int)s.length(); i++) {
        last[s[i] - 'a'] = i;
    }

    vector<int> result;
    int start = 0, end = 0;

    for (int i = 0; i < (int)s.length(); i++) {
        end = max(end, last[s[i] - 'a']);

        if (i == end) {
            result.push_back(end - start + 1);
            start = i + 1;
        }
    }

    return result;
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
    cout << "탐욕 알고리즘 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 활동 선택
    cout << "\n[1] 활동 선택" << endl;
    vector<Activity> activities = {
        {1, 4, 0}, {3, 5, 1}, {0, 6, 2}, {5, 7, 3},
        {3, 9, 4}, {5, 9, 5}, {6, 10, 6}, {8, 11, 7}
    };
    auto selected = activitySelection(activities);
    cout << "    선택된 활동: ";
    printVector(selected);
    cout << endl;

    // 2. 최소 회의실
    cout << "\n[2] 최소 회의실" << endl;
    vector<pair<int, int>> meetings = {{0, 30}, {5, 10}, {15, 20}};
    cout << "    회의: [(0,30), (5,10), (15,20)]" << endl;
    cout << "    최소 회의실: " << minMeetingRooms(meetings) << endl;

    // 3. 분할 가능 배낭
    cout << "\n[3] 분할 가능 배낭" << endl;
    vector<Item> items = {{10, 60}, {20, 100}, {30, 120}};
    cout << "    물건: (무게, 가치) = (10,60), (20,100), (30,120)" << endl;
    cout << "    용량 50, 최대 가치: " << fractionalKnapsack(50, items) << endl;

    // 4. 허프만 코딩
    cout << "\n[4] 허프만 코딩" << endl;
    string text = "aabbbcccc";
    auto codes = huffmanCoding(text);
    cout << "    텍스트: \"" << text << "\"" << endl;
    cout << "    코드:" << endl;
    for (auto& [ch, code] : codes) {
        cout << "      '" << ch << "': " << code << endl;
    }

    // 5. 작업 스케줄링
    cout << "\n[5] 작업 스케줄링" << endl;
    vector<Job> jobs = {{1, 4, 20}, {2, 1, 10}, {3, 1, 40}, {4, 1, 30}};
    cout << "    작업: (id, 마감, 이익)" << endl;
    cout << "    최대 이익: " << jobScheduling(jobs) << endl;

    // 6. 점프 게임
    cout << "\n[6] 점프 게임" << endl;
    vector<int> nums1 = {2, 3, 1, 1, 4};
    vector<int> nums2 = {3, 2, 1, 0, 4};
    cout << "    [2,3,1,1,4] 도달 가능: " << (canJump(nums1) ? "예" : "아니오") << endl;
    cout << "    [2,3,1,1,4] 최소 점프: " << minJumps(nums1) << endl;
    cout << "    [3,2,1,0,4] 도달 가능: " << (canJump(nums2) ? "예" : "아니오") << endl;

    // 7. 주유소
    cout << "\n[7] 주유소" << endl;
    vector<int> gas = {1, 2, 3, 4, 5};
    vector<int> cost = {3, 4, 5, 1, 2};
    cout << "    gas: [1,2,3,4,5], cost: [3,4,5,1,2]" << endl;
    cout << "    시작 위치: " << canCompleteCircuit(gas, cost) << endl;

    // 8. 문자열 분할
    cout << "\n[8] 문자열 분할" << endl;
    string s = "ababcbacadefegdehijhklij";
    auto parts = partitionLabels(s);
    cout << "    문자열: \"" << s << "\"" << endl;
    cout << "    분할 크기: ";
    printVector(parts);
    cout << endl;

    // 9. 탐욕 vs DP
    cout << "\n[9] 탐욕 vs DP" << endl;
    cout << "    | 기준           | 탐욕          | DP              |" << endl;
    cout << "    |----------------|---------------|-----------------|" << endl;
    cout << "    | 접근 방식      | 지역 최적     | 전역 최적       |" << endl;
    cout << "    | 결정 변경      | X             | O               |" << endl;
    cout << "    | 시간 복잡도    | 보통 낮음     | 보통 높음       |" << endl;
    cout << "    | 최적 보장      | 특정 문제만   | 항상 보장       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
