/*
 * 스택과 큐 (Stack and Queue)
 * Stack, Queue, Deque, Monotonic Stack/Queue
 *
 * 선형 자료구조의 활용입니다.
 */

#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <deque>
#include <string>
#include <unordered_map>

using namespace std;

// =============================================================================
// 1. 스택 기본 활용
// =============================================================================

// 괄호 유효성 검사
bool isValidParentheses(const string& s) {
    stack<char> st;
    unordered_map<char, char> pairs = {{')', '('}, {']', '['}, {'}', '{'}};

    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        } else {
            if (st.empty() || st.top() != pairs[c])
                return false;
            st.pop();
        }
    }

    return st.empty();
}

// 후위 표기법 계산
int evalRPN(const vector<string>& tokens) {
    stack<int> st;

    for (const string& token : tokens) {
        if (token == "+" || token == "-" || token == "*" || token == "/") {
            int b = st.top(); st.pop();
            int a = st.top(); st.pop();

            if (token == "+") st.push(a + b);
            else if (token == "-") st.push(a - b);
            else if (token == "*") st.push(a * b);
            else st.push(a / b);
        } else {
            st.push(stoi(token));
        }
    }

    return st.top();
}

// 중위 → 후위 변환
string infixToPostfix(const string& infix) {
    stack<char> st;
    string postfix;
    unordered_map<char, int> precedence = {{'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}};

    for (char c : infix) {
        if (isalnum(c)) {
            postfix += c;
        } else if (c == '(') {
            st.push(c);
        } else if (c == ')') {
            while (!st.empty() && st.top() != '(') {
                postfix += st.top();
                st.pop();
            }
            st.pop();  // '(' 제거
        } else {
            while (!st.empty() && st.top() != '(' &&
                   precedence[st.top()] >= precedence[c]) {
                postfix += st.top();
                st.pop();
            }
            st.push(c);
        }
    }

    while (!st.empty()) {
        postfix += st.top();
        st.pop();
    }

    return postfix;
}

// =============================================================================
// 2. 모노토닉 스택 (Monotonic Stack)
// =============================================================================

// 다음 큰 원소 (Next Greater Element)
vector<int> nextGreaterElement(const vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st;  // 인덱스 저장

    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[st.top()] < nums[i]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }

    return result;
}

// 히스토그램에서 가장 큰 직사각형
int largestRectangleArea(const vector<int>& heights) {
    stack<int> st;
    int maxArea = 0;
    int n = heights.size();

    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];

        while (!st.empty() && heights[st.top()] > h) {
            int height = heights[st.top()];
            st.pop();
            int width = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, height * width);
        }
        st.push(i);
    }

    return maxArea;
}

// 일일 온도 (Daily Temperatures)
vector<int> dailyTemperatures(const vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> result(n, 0);
    stack<int> st;

    for (int i = 0; i < n; i++) {
        while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
            result[st.top()] = i - st.top();
            st.pop();
        }
        st.push(i);
    }

    return result;
}

// =============================================================================
// 3. 큐 활용
// =============================================================================

// BFS용 큐 (간단 예시)
vector<int> bfsOrder(const vector<vector<int>>& graph, int start) {
    vector<int> result;
    vector<bool> visited(graph.size(), false);
    queue<int> q;

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }

    return result;
}

// =============================================================================
// 4. 덱 (Deque) 활용
// =============================================================================

// 슬라이딩 윈도우 최댓값
vector<int> maxSlidingWindow(const vector<int>& nums, int k) {
    deque<int> dq;  // 인덱스 저장
    vector<int> result;

    for (int i = 0; i < (int)nums.size(); i++) {
        // 윈도우 벗어난 원소 제거
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }

        // 현재 원소보다 작은 원소 제거
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }

        dq.push_back(i);

        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }

    return result;
}

// =============================================================================
// 5. 두 스택으로 큐 구현
// =============================================================================

class MyQueue {
private:
    stack<int> input, output;

    void transfer() {
        if (output.empty()) {
            while (!input.empty()) {
                output.push(input.top());
                input.pop();
            }
        }
    }

public:
    void push(int x) {
        input.push(x);
    }

    int pop() {
        transfer();
        int val = output.top();
        output.pop();
        return val;
    }

    int peek() {
        transfer();
        return output.top();
    }

    bool empty() {
        return input.empty() && output.empty();
    }
};

// =============================================================================
// 6. 최소 스택 (Min Stack)
// =============================================================================

class MinStack {
private:
    stack<int> st;
    stack<int> minSt;

public:
    void push(int val) {
        st.push(val);
        if (minSt.empty() || val <= minSt.top()) {
            minSt.push(val);
        }
    }

    void pop() {
        if (st.top() == minSt.top()) {
            minSt.pop();
        }
        st.pop();
    }

    int top() {
        return st.top();
    }

    int getMin() {
        return minSt.top();
    }
};

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
    cout << "스택과 큐 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 괄호 검사
    cout << "\n[1] 괄호 유효성 검사" << endl;
    cout << "    \"()[]{}\" : " << (isValidParentheses("()[]{}") ? "유효" : "무효") << endl;
    cout << "    \"([)]\"   : " << (isValidParentheses("([)]") ? "유효" : "무효") << endl;

    // 2. 후위 표기법
    cout << "\n[2] 후위 표기법" << endl;
    vector<string> rpn = {"2", "1", "+", "3", "*"};
    cout << "    [\"2\",\"1\",\"+\",\"3\",\"*\"] = " << evalRPN(rpn) << endl;

    cout << "    \"a+b*c\" → \"" << infixToPostfix("a+b*c") << "\"" << endl;

    // 3. 모노토닉 스택
    cout << "\n[3] 모노토닉 스택" << endl;
    vector<int> nums = {2, 1, 2, 4, 3};
    cout << "    배열: [2,1,2,4,3]" << endl;
    cout << "    다음 큰 원소: ";
    printVector(nextGreaterElement(nums));
    cout << endl;

    vector<int> heights = {2, 1, 5, 6, 2, 3};
    cout << "    히스토그램 [2,1,5,6,2,3] 최대 직사각형: " << largestRectangleArea(heights) << endl;

    vector<int> temps = {73, 74, 75, 71, 69, 72, 76, 73};
    cout << "    일일 온도 [73,74,75,71,69,72,76,73]: ";
    printVector(dailyTemperatures(temps));
    cout << endl;

    // 4. 슬라이딩 윈도우 최댓값
    cout << "\n[4] 슬라이딩 윈도우 최댓값" << endl;
    vector<int> window = {1, 3, -1, -3, 5, 3, 6, 7};
    cout << "    배열: [1,3,-1,-3,5,3,6,7], k=3" << endl;
    cout << "    최댓값: ";
    printVector(maxSlidingWindow(window, 3));
    cout << endl;

    // 5. 두 스택으로 큐
    cout << "\n[5] 두 스택으로 큐 구현" << endl;
    MyQueue q;
    q.push(1);
    q.push(2);
    cout << "    push(1), push(2)" << endl;
    cout << "    peek(): " << q.peek() << endl;
    cout << "    pop(): " << q.pop() << endl;
    cout << "    empty(): " << (q.empty() ? "true" : "false") << endl;

    // 6. 최소 스택
    cout << "\n[6] 최소 스택" << endl;
    MinStack ms;
    ms.push(-2);
    ms.push(0);
    ms.push(-3);
    cout << "    push(-2), push(0), push(-3)" << endl;
    cout << "    getMin(): " << ms.getMin() << endl;
    ms.pop();
    cout << "    pop() 후 getMin(): " << ms.getMin() << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 연산         | 스택  | 큐    | 덱    |" << endl;
    cout << "    |--------------|-------|-------|-------|" << endl;
    cout << "    | push/enqueue | O(1)  | O(1)  | O(1)  |" << endl;
    cout << "    | pop/dequeue  | O(1)  | O(1)  | O(1)  |" << endl;
    cout << "    | peek/front   | O(1)  | O(1)  | O(1)  |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
