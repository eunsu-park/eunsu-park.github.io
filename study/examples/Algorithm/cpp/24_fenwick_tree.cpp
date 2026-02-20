/*
 * 펜윅 트리 (Fenwick Tree / Binary Indexed Tree)
 * Point Update, Range Sum Query, 2D BIT
 *
 * 세그먼트 트리보다 간단하고 빠른 구간 합 자료구조입니다.
 */

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// =============================================================================
// 1. 기본 펜윅 트리 (1-indexed)
// =============================================================================

class FenwickTree {
private:
    vector<long long> tree;
    int n;

public:
    FenwickTree(int n) : n(n), tree(n + 1, 0) {}

    FenwickTree(const vector<int>& arr) : n(arr.size()), tree(arr.size() + 1, 0) {
        for (int i = 0; i < n; i++) {
            update(i + 1, arr[i]);
        }
    }

    // arr[idx] += delta
    void update(int idx, long long delta) {
        while (idx <= n) {
            tree[idx] += delta;
            idx += idx & (-idx);  // 다음 노드
        }
    }

    // sum(arr[1..idx])
    long long prefixSum(int idx) {
        long long sum = 0;
        while (idx > 0) {
            sum += tree[idx];
            idx -= idx & (-idx);  // 부모 노드
        }
        return sum;
    }

    // sum(arr[l..r])
    long long rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};

// =============================================================================
// 2. 0-indexed 펜윅 트리
// =============================================================================

class FenwickTree0 {
private:
    vector<long long> tree;
    int n;

public:
    FenwickTree0(int n) : n(n), tree(n, 0) {}

    void update(int idx, long long delta) {
        while (idx < n) {
            tree[idx] += delta;
            idx |= (idx + 1);
        }
    }

    long long prefixSum(int idx) {
        long long sum = 0;
        while (idx >= 0) {
            sum += tree[idx];
            idx = (idx & (idx + 1)) - 1;
        }
        return sum;
    }

    long long rangeSum(int l, int r) {
        return prefixSum(r) - (l > 0 ? prefixSum(l - 1) : 0);
    }
};

// =============================================================================
// 3. 구간 업데이트, 점 쿼리
// =============================================================================

class FenwickTreeRangeUpdate {
private:
    vector<long long> tree;
    int n;

    void update(int idx, long long delta) {
        while (idx <= n) {
            tree[idx] += delta;
            idx += idx & (-idx);
        }
    }

    long long query(int idx) {
        long long sum = 0;
        while (idx > 0) {
            sum += tree[idx];
            idx -= idx & (-idx);
        }
        return sum;
    }

public:
    FenwickTreeRangeUpdate(int n) : n(n), tree(n + 1, 0) {}

    // arr[l..r] += delta
    void updateRange(int l, int r, long long delta) {
        update(l, delta);
        update(r + 1, -delta);
    }

    // arr[idx]의 현재 값
    long long get(int idx) {
        return query(idx);
    }
};

// =============================================================================
// 4. 구간 업데이트, 구간 쿼리
// =============================================================================

class FenwickTreeRangeRange {
private:
    vector<long long> tree1, tree2;
    int n;

    void update(vector<long long>& tree, int idx, long long delta) {
        while (idx <= n) {
            tree[idx] += delta;
            idx += idx & (-idx);
        }
    }

    long long query(const vector<long long>& tree, int idx) {
        long long sum = 0;
        while (idx > 0) {
            sum += tree[idx];
            idx -= idx & (-idx);
        }
        return sum;
    }

public:
    FenwickTreeRangeRange(int n) : n(n), tree1(n + 1, 0), tree2(n + 1, 0) {}

    // arr[l..r] += delta
    void updateRange(int l, int r, long long delta) {
        update(tree1, l, delta);
        update(tree1, r + 1, -delta);
        update(tree2, l, delta * (l - 1));
        update(tree2, r + 1, -delta * r);
    }

    // sum(arr[1..idx])
    long long prefixSum(int idx) {
        return query(tree1, idx) * idx - query(tree2, idx);
    }

    // sum(arr[l..r])
    long long rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};

// =============================================================================
// 5. 2D 펜윅 트리
// =============================================================================

class FenwickTree2D {
private:
    vector<vector<long long>> tree;
    int n, m;

public:
    FenwickTree2D(int n, int m) : n(n), m(m), tree(n + 1, vector<long long>(m + 1, 0)) {}

    void update(int x, int y, long long delta) {
        for (int i = x; i <= n; i += i & (-i)) {
            for (int j = y; j <= m; j += j & (-j)) {
                tree[i][j] += delta;
            }
        }
    }

    // sum from (1,1) to (x,y)
    long long prefixSum(int x, int y) {
        long long sum = 0;
        for (int i = x; i > 0; i -= i & (-i)) {
            for (int j = y; j > 0; j -= j & (-j)) {
                sum += tree[i][j];
            }
        }
        return sum;
    }

    // sum from (x1,y1) to (x2,y2)
    long long rangeSum(int x1, int y1, int x2, int y2) {
        return prefixSum(x2, y2)
             - prefixSum(x1 - 1, y2)
             - prefixSum(x2, y1 - 1)
             + prefixSum(x1 - 1, y1 - 1);
    }
};

// =============================================================================
// 6. 역순 쌍 개수 (Inversion Count)
// =============================================================================

long long countInversions(vector<int>& arr) {
    int n = arr.size();

    // 좌표 압축
    vector<int> sorted = arr;
    sort(sorted.begin(), sorted.end());
    sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());

    for (int& x : arr) {
        x = lower_bound(sorted.begin(), sorted.end(), x) - sorted.begin() + 1;
    }

    // BIT로 역순 쌍 개수
    FenwickTree bit(sorted.size());
    long long inversions = 0;

    for (int i = n - 1; i >= 0; i--) {
        inversions += bit.prefixSum(arr[i] - 1);
        bit.update(arr[i], 1);
    }

    return inversions;
}

// =============================================================================
// 7. K번째 원소 찾기
// =============================================================================

class FenwickTreeKth {
private:
    vector<long long> tree;
    int n;

public:
    FenwickTreeKth(int n) : n(n), tree(n + 1, 0) {}

    void update(int idx, long long delta) {
        while (idx <= n) {
            tree[idx] += delta;
            idx += idx & (-idx);
        }
    }

    // k번째 원소의 인덱스 (1-indexed)
    int kthElement(long long k) {
        int idx = 0;
        int bitMask = 1;
        while (bitMask <= n) bitMask <<= 1;

        for (; bitMask > 0; bitMask >>= 1) {
            int next = idx + bitMask;
            if (next <= n && tree[next] < k) {
                k -= tree[next];
                idx = next;
            }
        }

        return idx + 1;
    }
};

// =============================================================================
// 8. 최솟값 펜윅 트리
// =============================================================================

class FenwickTreeMin {
private:
    vector<int> tree;
    vector<int> arr;
    int n;
    const int INF = INT_MAX;

public:
    FenwickTreeMin(int n) : n(n), tree(n + 1, INF), arr(n + 1, INF) {}

    // arr[idx] = val (val < 기존 값일 때만 유효)
    void update(int idx, int val) {
        arr[idx] = val;
        while (idx <= n) {
            tree[idx] = min(tree[idx], val);
            idx += idx & (-idx);
        }
    }

    // min(arr[1..idx])
    int prefixMin(int idx) {
        int result = INF;
        while (idx > 0) {
            result = min(result, tree[idx]);
            idx -= idx & (-idx);
        }
        return result;
    }
};

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "펜윅 트리 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 기본 펜윅 트리
    cout << "\n[1] 기본 펜윅 트리" << endl;
    vector<int> arr = {1, 3, 5, 7, 9, 11};
    FenwickTree ft(arr);
    cout << "    배열: [1, 3, 5, 7, 9, 11] (1-indexed)" << endl;
    cout << "    sum[1, 3] = " << ft.rangeSum(1, 3) << endl;
    cout << "    sum[2, 5] = " << ft.rangeSum(2, 5) << endl;
    ft.update(3, 5);  // arr[3] += 5
    cout << "    arr[3] += 5 후 sum[1, 3] = " << ft.rangeSum(1, 3) << endl;

    // 2. 구간 업데이트
    cout << "\n[2] 구간 업데이트, 점 쿼리" << endl;
    FenwickTreeRangeUpdate ftru(6);
    ftru.updateRange(2, 4, 10);  // arr[2..4] += 10
    cout << "    arr[2..4] += 10" << endl;
    cout << "    arr[1] = " << ftru.get(1) << endl;
    cout << "    arr[3] = " << ftru.get(3) << endl;
    cout << "    arr[5] = " << ftru.get(5) << endl;

    // 3. 구간 업데이트, 구간 쿼리
    cout << "\n[3] 구간 업데이트, 구간 쿼리" << endl;
    FenwickTreeRangeRange ftrr(6);
    ftrr.updateRange(1, 3, 5);   // arr[1..3] += 5
    ftrr.updateRange(2, 5, 10);  // arr[2..5] += 10
    cout << "    arr[1..3] += 5, arr[2..5] += 10" << endl;
    cout << "    sum[1, 6] = " << ftrr.rangeSum(1, 6) << endl;

    // 4. 2D 펜윅 트리
    cout << "\n[4] 2D 펜윅 트리" << endl;
    FenwickTree2D ft2d(3, 3);
    ft2d.update(1, 1, 1);
    ft2d.update(1, 2, 2);
    ft2d.update(2, 1, 3);
    ft2d.update(2, 2, 4);
    cout << "    3x3 행렬, (1,1)=1, (1,2)=2, (2,1)=3, (2,2)=4" << endl;
    cout << "    sum[(1,1) to (2,2)] = " << ft2d.rangeSum(1, 1, 2, 2) << endl;

    // 5. 역순 쌍 개수
    cout << "\n[5] 역순 쌍 개수" << endl;
    vector<int> invArr = {8, 4, 2, 1};
    cout << "    배열: [8, 4, 2, 1]" << endl;
    cout << "    역순 쌍: " << countInversions(invArr) << endl;

    // 6. K번째 원소
    cout << "\n[6] K번째 원소" << endl;
    FenwickTreeKth ftkth(10);
    ftkth.update(2, 1);  // 2 추가
    ftkth.update(5, 1);  // 5 추가
    ftkth.update(7, 1);  // 7 추가
    cout << "    집합: {2, 5, 7}" << endl;
    cout << "    1번째 원소: " << ftkth.kthElement(1) << endl;
    cout << "    2번째 원소: " << ftkth.kthElement(2) << endl;
    cout << "    3번째 원소: " << ftkth.kthElement(3) << endl;

    // 7. 세그먼트 트리 vs 펜윅 트리
    cout << "\n[7] 세그먼트 트리 vs 펜윅 트리" << endl;
    cout << "    | 기준           | 세그먼트 트리    | 펜윅 트리       |" << endl;
    cout << "    |----------------|------------------|-----------------|" << endl;
    cout << "    | 구현 복잡도    | 중간             | 낮음            |" << endl;
    cout << "    | 메모리         | 4N               | N               |" << endl;
    cout << "    | 상수 계수      | 크다             | 작다            |" << endl;
    cout << "    | 지원 연산      | 다양             | 제한적          |" << endl;
    cout << "    | Lazy 지원      | O                | 복잡            |" << endl;

    // 8. 복잡도 요약
    cout << "\n[8] 복잡도 요약" << endl;
    cout << "    | 연산           | 시간복잡도 | 공간복잡도 |" << endl;
    cout << "    |----------------|------------|------------|" << endl;
    cout << "    | 점 업데이트    | O(log n)   | O(1)       |" << endl;
    cout << "    | 구간 합        | O(log n)   | O(1)       |" << endl;
    cout << "    | 구간 업데이트  | O(log n)   | O(n)       |" << endl;
    cout << "    | 2D 쿼리        | O(log² n)  | O(nm)      |" << endl;
    cout << "    | K번째 원소     | O(log n)   | O(1)       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
