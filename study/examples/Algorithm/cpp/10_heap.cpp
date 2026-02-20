/*
 * 힙 (Heap)
 * Min/Max Heap, Priority Queue, Heap Sort
 *
 * 우선순위 기반의 효율적인 자료구조입니다.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <functional>

using namespace std;

// =============================================================================
// 1. 최소 힙 구현
// =============================================================================

class MinHeap {
private:
    vector<int> heap;

    void heapifyUp(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (heap[parent] <= heap[idx]) break;
            swap(heap[parent], heap[idx]);
            idx = parent;
        }
    }

    void heapifyDown(int idx) {
        int size = heap.size();
        while (true) {
            int smallest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;

            if (left < size && heap[left] < heap[smallest])
                smallest = left;
            if (right < size && heap[right] < heap[smallest])
                smallest = right;

            if (smallest == idx) break;
            swap(heap[idx], heap[smallest]);
            idx = smallest;
        }
    }

public:
    void push(int val) {
        heap.push_back(val);
        heapifyUp(heap.size() - 1);
    }

    int pop() {
        if (heap.empty()) throw runtime_error("Heap is empty");
        int minVal = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        if (!heap.empty()) heapifyDown(0);
        return minVal;
    }

    int top() const {
        if (heap.empty()) throw runtime_error("Heap is empty");
        return heap[0];
    }

    bool empty() const { return heap.empty(); }
    size_t size() const { return heap.size(); }
};

// =============================================================================
// 2. 힙 정렬
// =============================================================================

void heapify(vector<int>& arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;
    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(vector<int>& arr) {
    int n = arr.size();

    // 힙 구성
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // 추출
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

// =============================================================================
// 3. K번째 요소
// =============================================================================

// 배열에서 K번째로 큰 원소
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;

    for (int num : nums) {
        minHeap.push(num);
        if ((int)minHeap.size() > k) {
            minHeap.pop();
        }
    }

    return minHeap.top();
}

// 배열에서 K번째로 작은 원소
int findKthSmallest(vector<int>& nums, int k) {
    priority_queue<int> maxHeap;

    for (int num : nums) {
        maxHeap.push(num);
        if ((int)maxHeap.size() > k) {
            maxHeap.pop();
        }
    }

    return maxHeap.top();
}

// =============================================================================
// 4. 중앙값 스트림
// =============================================================================

class MedianFinder {
private:
    priority_queue<int> maxHeap;  // 작은 절반
    priority_queue<int, vector<int>, greater<int>> minHeap;  // 큰 절반

public:
    void addNum(int num) {
        maxHeap.push(num);
        minHeap.push(maxHeap.top());
        maxHeap.pop();

        if (minHeap.size() > maxHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }

    double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        }
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};

// =============================================================================
// 5. K개 정렬 리스트 병합
// =============================================================================

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);

    for (ListNode* list : lists) {
        if (list) pq.push(list);
    }

    ListNode dummy(0);
    ListNode* curr = &dummy;

    while (!pq.empty()) {
        ListNode* node = pq.top();
        pq.pop();
        curr->next = node;
        curr = curr->next;
        if (node->next) pq.push(node->next);
    }

    return dummy.next;
}

// =============================================================================
// 6. 가장 가까운 K개 점
// =============================================================================

vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
    auto dist = [](const vector<int>& p) {
        return p[0] * p[0] + p[1] * p[1];
    };

    auto cmp = [&dist](const vector<int>& a, const vector<int>& b) {
        return dist(a) < dist(b);
    };

    priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> maxHeap(cmp);

    for (auto& point : points) {
        maxHeap.push(point);
        if ((int)maxHeap.size() > k) {
            maxHeap.pop();
        }
    }

    vector<vector<int>> result;
    while (!maxHeap.empty()) {
        result.push_back(maxHeap.top());
        maxHeap.pop();
    }

    return result;
}

// =============================================================================
// 7. Top K 빈도 원소
// =============================================================================

vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int num : nums) {
        freq[num]++;
    }

    auto cmp = [&freq](int a, int b) { return freq[a] > freq[b]; };
    priority_queue<int, vector<int>, decltype(cmp)> minHeap(cmp);

    for (auto& [num, count] : freq) {
        minHeap.push(num);
        if ((int)minHeap.size() > k) {
            minHeap.pop();
        }
    }

    vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top());
        minHeap.pop();
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
    cout << "힙 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 최소 힙
    cout << "\n[1] 최소 힙" << endl;
    MinHeap minHeap;
    minHeap.push(5);
    minHeap.push(3);
    minHeap.push(7);
    minHeap.push(1);
    cout << "    삽입: 5, 3, 7, 1" << endl;
    cout << "    최솟값: " << minHeap.top() << endl;
    cout << "    pop 순서: ";
    while (!minHeap.empty()) {
        cout << minHeap.pop() << " ";
    }
    cout << endl;

    // 2. STL priority_queue
    cout << "\n[2] STL priority_queue" << endl;
    priority_queue<int> maxPQ;  // 최대 힙
    priority_queue<int, vector<int>, greater<int>> minPQ;  // 최소 힙

    for (int x : {3, 1, 4, 1, 5, 9}) {
        maxPQ.push(x);
        minPQ.push(x);
    }

    cout << "    최대 힙 top: " << maxPQ.top() << endl;
    cout << "    최소 힙 top: " << minPQ.top() << endl;

    // 3. 힙 정렬
    cout << "\n[3] 힙 정렬" << endl;
    vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    cout << "    정렬 전: ";
    printVector(arr);
    cout << endl;
    heapSort(arr);
    cout << "    정렬 후: ";
    printVector(arr);
    cout << endl;

    // 4. K번째 원소
    cout << "\n[4] K번째 원소" << endl;
    vector<int> nums = {3, 2, 1, 5, 6, 4};
    cout << "    배열: [3,2,1,5,6,4]" << endl;
    cout << "    2번째로 큰 원소: " << findKthLargest(nums, 2) << endl;
    cout << "    3번째로 작은 원소: " << findKthSmallest(nums, 3) << endl;

    // 5. 중앙값 스트림
    cout << "\n[5] 중앙값 스트림" << endl;
    MedianFinder mf;
    mf.addNum(1);
    mf.addNum(2);
    cout << "    [1, 2] 중앙값: " << mf.findMedian() << endl;
    mf.addNum(3);
    cout << "    [1, 2, 3] 중앙값: " << mf.findMedian() << endl;

    // 6. Top K 빈도
    cout << "\n[6] Top K 빈도 원소" << endl;
    vector<int> freqNums = {1, 1, 1, 2, 2, 3};
    auto topK = topKFrequent(freqNums, 2);
    cout << "    [1,1,1,2,2,3], k=2: ";
    printVector(topK);
    cout << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 연산         | 시간복잡도 |" << endl;
    cout << "    |--------------|------------|" << endl;
    cout << "    | 삽입         | O(log n)   |" << endl;
    cout << "    | 삭제 (top)   | O(log n)   |" << endl;
    cout << "    | 최솟값/최댓값| O(1)       |" << endl;
    cout << "    | 힙 구성      | O(n)       |" << endl;
    cout << "    | 힙 정렬      | O(n log n) |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
