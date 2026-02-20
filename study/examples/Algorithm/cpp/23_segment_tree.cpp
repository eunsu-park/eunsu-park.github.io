/*
 * 세그먼트 트리 (Segment Tree)
 * Range Sum/Min/Max Query, Lazy Propagation
 *
 * 구간 쿼리와 업데이트를 효율적으로 처리합니다.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <functional>

using namespace std;

// =============================================================================
// 1. 기본 세그먼트 트리 (구간 합)
// =============================================================================

class SegmentTree {
private:
    vector<long long> tree;
    int n;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
            return;
        }

        int mid = (start + end) / 2;
        build(arr, 2 * node, start, mid);
        build(arr, 2 * node + 1, mid + 1, end);
        tree[node] = tree[2 * node] + tree[2 * node + 1];
    }

    void update(int node, int start, int end, int idx, long long val) {
        if (start == end) {
            tree[node] = val;
            return;
        }

        int mid = (start + end) / 2;
        if (idx <= mid) {
            update(2 * node, start, mid, idx, val);
        } else {
            update(2 * node + 1, mid + 1, end, idx, val);
        }
        tree[node] = tree[2 * node] + tree[2 * node + 1];
    }

    long long query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) {
            return 0;
        }
        if (l <= start && end <= r) {
            return tree[node];
        }

        int mid = (start + end) / 2;
        return query(2 * node, start, mid, l, r) +
               query(2 * node + 1, mid + 1, end, l, r);
    }

public:
    SegmentTree(const vector<int>& arr) : n(arr.size()) {
        tree.resize(4 * n);
        build(arr, 1, 0, n - 1);
    }

    void update(int idx, long long val) {
        update(1, 0, n - 1, idx, val);
    }

    long long query(int l, int r) {
        return query(1, 0, n - 1, l, r);
    }
};

// =============================================================================
// 2. 구간 최솟값 세그먼트 트리
// =============================================================================

class MinSegmentTree {
private:
    vector<int> tree;
    int n;
    const int INF = INT_MAX;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
            return;
        }

        int mid = (start + end) / 2;
        build(arr, 2 * node, start, mid);
        build(arr, 2 * node + 1, mid + 1, end);
        tree[node] = min(tree[2 * node], tree[2 * node + 1]);
    }

    void update(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
            return;
        }

        int mid = (start + end) / 2;
        if (idx <= mid) {
            update(2 * node, start, mid, idx, val);
        } else {
            update(2 * node + 1, mid + 1, end, idx, val);
        }
        tree[node] = min(tree[2 * node], tree[2 * node + 1]);
    }

    int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) {
            return INF;
        }
        if (l <= start && end <= r) {
            return tree[node];
        }

        int mid = (start + end) / 2;
        return min(query(2 * node, start, mid, l, r),
                   query(2 * node + 1, mid + 1, end, l, r));
    }

public:
    MinSegmentTree(const vector<int>& arr) : n(arr.size()) {
        tree.resize(4 * n, INF);
        build(arr, 1, 0, n - 1);
    }

    void update(int idx, int val) {
        update(1, 0, n - 1, idx, val);
    }

    int query(int l, int r) {
        return query(1, 0, n - 1, l, r);
    }
};

// =============================================================================
// 3. Lazy Propagation (구간 업데이트)
// =============================================================================

class LazySegmentTree {
private:
    vector<long long> tree, lazy;
    int n;

    void propagate(int node, int start, int end) {
        if (lazy[node] != 0) {
            tree[node] += (end - start + 1) * lazy[node];
            if (start != end) {
                lazy[2 * node] += lazy[node];
                lazy[2 * node + 1] += lazy[node];
            }
            lazy[node] = 0;
        }
    }

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
            return;
        }

        int mid = (start + end) / 2;
        build(arr, 2 * node, start, mid);
        build(arr, 2 * node + 1, mid + 1, end);
        tree[node] = tree[2 * node] + tree[2 * node + 1];
    }

    void updateRange(int node, int start, int end, int l, int r, long long val) {
        propagate(node, start, end);

        if (r < start || end < l) return;

        if (l <= start && end <= r) {
            lazy[node] = val;
            propagate(node, start, end);
            return;
        }

        int mid = (start + end) / 2;
        updateRange(2 * node, start, mid, l, r, val);
        updateRange(2 * node + 1, mid + 1, end, l, r, val);
        tree[node] = tree[2 * node] + tree[2 * node + 1];
    }

    long long query(int node, int start, int end, int l, int r) {
        propagate(node, start, end);

        if (r < start || end < l) return 0;

        if (l <= start && end <= r) {
            return tree[node];
        }

        int mid = (start + end) / 2;
        return query(2 * node, start, mid, l, r) +
               query(2 * node + 1, mid + 1, end, l, r);
    }

public:
    LazySegmentTree(const vector<int>& arr) : n(arr.size()) {
        tree.resize(4 * n, 0);
        lazy.resize(4 * n, 0);
        build(arr, 1, 0, n - 1);
    }

    void updateRange(int l, int r, long long val) {
        updateRange(1, 0, n - 1, l, r, val);
    }

    long long query(int l, int r) {
        return query(1, 0, n - 1, l, r);
    }
};

// =============================================================================
// 4. 동적 세그먼트 트리
// =============================================================================

class DynamicSegmentTree {
private:
    struct Node {
        long long sum = 0;
        Node *left = nullptr, *right = nullptr;
    };

    Node* root;
    long long lo, hi;

    void update(Node*& node, long long start, long long end, long long idx, long long val) {
        if (!node) node = new Node();

        if (start == end) {
            node->sum += val;
            return;
        }

        long long mid = (start + end) / 2;
        if (idx <= mid) {
            update(node->left, start, mid, idx, val);
        } else {
            update(node->right, mid + 1, end, idx, val);
        }

        node->sum = (node->left ? node->left->sum : 0) +
                    (node->right ? node->right->sum : 0);
    }

    long long query(Node* node, long long start, long long end, long long l, long long r) {
        if (!node || r < start || end < l) return 0;
        if (l <= start && end <= r) return node->sum;

        long long mid = (start + end) / 2;
        return query(node->left, start, mid, l, r) +
               query(node->right, mid + 1, end, l, r);
    }

public:
    DynamicSegmentTree(long long lo, long long hi) : root(nullptr), lo(lo), hi(hi) {}

    void update(long long idx, long long val) {
        update(root, lo, hi, idx, val);
    }

    long long query(long long l, long long r) {
        return query(root, lo, hi, l, r);
    }
};

// =============================================================================
// 5. 머지 소트 트리 (구간 K번째 원소)
// =============================================================================

class MergeSortTree {
private:
    vector<vector<int>> tree;
    int n;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = {arr[start]};
            return;
        }

        int mid = (start + end) / 2;
        build(arr, 2 * node, start, mid);
        build(arr, 2 * node + 1, mid + 1, end);

        merge(tree[2 * node].begin(), tree[2 * node].end(),
              tree[2 * node + 1].begin(), tree[2 * node + 1].end(),
              back_inserter(tree[node]));
    }

    // [l, r] 구간에서 x 이하인 원소 개수
    int countLessEqual(int node, int start, int end, int l, int r, int x) {
        if (r < start || end < l) return 0;
        if (l <= start && end <= r) {
            return upper_bound(tree[node].begin(), tree[node].end(), x) -
                   tree[node].begin();
        }

        int mid = (start + end) / 2;
        return countLessEqual(2 * node, start, mid, l, r, x) +
               countLessEqual(2 * node + 1, mid + 1, end, l, r, x);
    }

public:
    MergeSortTree(const vector<int>& arr) : n(arr.size()) {
        tree.resize(4 * n);
        build(arr, 1, 0, n - 1);
    }

    // [l, r] 구간에서 k번째 작은 원소
    int kthSmallest(int l, int r, int k) {
        int lo = INT_MIN, hi = INT_MAX;

        while (lo < hi) {
            int mid = lo / 2 + hi / 2;
            if (countLessEqual(1, 0, n - 1, l, r, mid) < k) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        return lo;
    }
};

// =============================================================================
// 6. 2D 세그먼트 트리
// =============================================================================

class SegmentTree2D {
private:
    vector<vector<long long>> tree;
    int n, m;

    void buildY(const vector<vector<int>>& arr, int nx, int lx, int rx, int ny, int ly, int ry) {
        if (ly == ry) {
            if (lx == rx) {
                tree[nx][ny] = arr[lx][ly];
            } else {
                tree[nx][ny] = tree[2 * nx][ny] + tree[2 * nx + 1][ny];
            }
            return;
        }

        int my = (ly + ry) / 2;
        buildY(arr, nx, lx, rx, 2 * ny, ly, my);
        buildY(arr, nx, lx, rx, 2 * ny + 1, my + 1, ry);
        tree[nx][ny] = tree[nx][2 * ny] + tree[nx][2 * ny + 1];
    }

    void buildX(const vector<vector<int>>& arr, int nx, int lx, int rx) {
        if (lx != rx) {
            int mx = (lx + rx) / 2;
            buildX(arr, 2 * nx, lx, mx);
            buildX(arr, 2 * nx + 1, mx + 1, rx);
        }
        buildY(arr, nx, lx, rx, 1, 0, m - 1);
    }

    long long queryY(int nx, int ny, int ly, int ry, int y1, int y2) {
        if (y2 < ly || ry < y1) return 0;
        if (y1 <= ly && ry <= y2) return tree[nx][ny];

        int my = (ly + ry) / 2;
        return queryY(nx, 2 * ny, ly, my, y1, y2) +
               queryY(nx, 2 * ny + 1, my + 1, ry, y1, y2);
    }

    long long queryX(int nx, int lx, int rx, int x1, int x2, int y1, int y2) {
        if (x2 < lx || rx < x1) return 0;
        if (x1 <= lx && rx <= x2) return queryY(nx, 1, 0, m - 1, y1, y2);

        int mx = (lx + rx) / 2;
        return queryX(2 * nx, lx, mx, x1, x2, y1, y2) +
               queryX(2 * nx + 1, mx + 1, rx, x1, x2, y1, y2);
    }

public:
    SegmentTree2D(const vector<vector<int>>& arr) {
        n = arr.size();
        m = arr[0].size();
        tree.assign(4 * n, vector<long long>(4 * m, 0));
        buildX(arr, 1, 0, n - 1);
    }

    long long query(int x1, int y1, int x2, int y2) {
        return queryX(1, 0, n - 1, x1, x2, y1, y2);
    }
};

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "세그먼트 트리 예제" << endl;
    cout << "============================================================" << endl;

    vector<int> arr = {1, 3, 5, 7, 9, 11};

    // 1. 기본 세그먼트 트리
    cout << "\n[1] 구간 합 세그먼트 트리" << endl;
    cout << "    배열: [1, 3, 5, 7, 9, 11]" << endl;
    SegmentTree st(arr);
    cout << "    sum[1, 3] = " << st.query(1, 3) << endl;
    st.update(2, 10);
    cout << "    arr[2] = 10 업데이트 후" << endl;
    cout << "    sum[1, 3] = " << st.query(1, 3) << endl;

    // 2. 구간 최솟값
    cout << "\n[2] 구간 최솟값 세그먼트 트리" << endl;
    MinSegmentTree minSt(arr);
    cout << "    min[0, 5] = " << minSt.query(0, 5) << endl;
    cout << "    min[2, 4] = " << minSt.query(2, 4) << endl;

    // 3. Lazy Propagation
    cout << "\n[3] Lazy Propagation" << endl;
    vector<int> arr2 = {1, 2, 3, 4, 5};
    LazySegmentTree lazySt(arr2);
    cout << "    배열: [1, 2, 3, 4, 5]" << endl;
    cout << "    sum[0, 4] = " << lazySt.query(0, 4) << endl;
    lazySt.updateRange(1, 3, 10);  // [1, 3] 구간에 10 더하기
    cout << "    [1, 3] += 10 후" << endl;
    cout << "    sum[0, 4] = " << lazySt.query(0, 4) << endl;

    // 4. 동적 세그먼트 트리
    cout << "\n[4] 동적 세그먼트 트리" << endl;
    DynamicSegmentTree dynSt(0, 1000000000);
    dynSt.update(100, 5);
    dynSt.update(500000000, 10);
    cout << "    범위: [0, 10^9]" << endl;
    cout << "    update(100, 5), update(5×10^8, 10)" << endl;
    cout << "    sum[0, 10^9] = " << dynSt.query(0, 1000000000) << endl;

    // 5. 2D 세그먼트 트리
    cout << "\n[5] 2D 세그먼트 트리" << endl;
    vector<vector<int>> arr2d = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    SegmentTree2D st2d(arr2d);
    cout << "    3x3 행렬" << endl;
    cout << "    sum[(0,0) to (2,2)] = " << st2d.query(0, 0, 2, 2) << endl;
    cout << "    sum[(0,0) to (1,1)] = " << st2d.query(0, 0, 1, 1) << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 연산            | 시간복잡도 | 공간복잡도 |" << endl;
    cout << "    |-----------------|------------|------------|" << endl;
    cout << "    | 빌드            | O(n)       | O(n)       |" << endl;
    cout << "    | 점 업데이트     | O(log n)   | O(1)       |" << endl;
    cout << "    | 구간 쿼리       | O(log n)   | O(1)       |" << endl;
    cout << "    | 구간 업데이트   | O(log n)   | O(n) lazy  |" << endl;
    cout << "    | 2D 쿼리         | O(log² n)  | O(n²)      |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
