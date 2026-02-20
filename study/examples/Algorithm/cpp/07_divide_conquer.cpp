/*
 * 분할 정복 (Divide and Conquer)
 * Merge Sort, Quick Sort, Binary Search, Power
 *
 * 문제를 작은 부분으로 나누어 해결합니다.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>

using namespace std;

// =============================================================================
// 1. 빠른 거듭제곱
// =============================================================================

long long power(long long base, long long exp, long long mod = LLONG_MAX) {
    long long result = 1;
    base %= mod;

    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp >>= 1;
    }

    return result;
}

// =============================================================================
// 2. 행렬 거듭제곱
// =============================================================================

using Matrix = vector<vector<long long>>;

Matrix matrixMultiply(const Matrix& A, const Matrix& B, long long mod) {
    int n = A.size();
    Matrix C(n, vector<long long>(n, 0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod;
            }
        }
    }

    return C;
}

Matrix matrixPower(Matrix A, long long exp, long long mod) {
    int n = A.size();
    Matrix result(n, vector<long long>(n, 0));

    // 단위 행렬
    for (int i = 0; i < n; i++) result[i][i] = 1;

    while (exp > 0) {
        if (exp & 1) {
            result = matrixMultiply(result, A, mod);
        }
        A = matrixMultiply(A, A, mod);
        exp >>= 1;
    }

    return result;
}

// 피보나치 (행렬 거듭제곱)
long long fibonacciMatrix(long long n, long long mod = 1e9 + 7) {
    if (n <= 1) return n;

    Matrix M = {{1, 1}, {1, 0}};
    Matrix result = matrixPower(M, n - 1, mod);
    return result[0][0];
}

// =============================================================================
// 3. 가장 가까운 점 쌍 (Closest Pair)
// =============================================================================

struct Point {
    double x, y;
};

double dist(const Point& p1, const Point& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

double bruteForce(vector<Point>& points, int left, int right) {
    double minDist = DBL_MAX;
    for (int i = left; i < right; i++) {
        for (int j = i + 1; j <= right; j++) {
            minDist = min(minDist, dist(points[i], points[j]));
        }
    }
    return minDist;
}

double stripClosest(vector<Point>& strip, double d) {
    double minDist = d;
    sort(strip.begin(), strip.end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
    });

    for (size_t i = 0; i < strip.size(); i++) {
        for (size_t j = i + 1; j < strip.size() && (strip[j].y - strip[i].y) < minDist; j++) {
            minDist = min(minDist, dist(strip[i], strip[j]));
        }
    }

    return minDist;
}

double closestPairHelper(vector<Point>& points, int left, int right) {
    if (right - left <= 3) {
        return bruteForce(points, left, right);
    }

    int mid = (left + right) / 2;
    Point midPoint = points[mid];

    double dl = closestPairHelper(points, left, mid);
    double dr = closestPairHelper(points, mid + 1, right);
    double d = min(dl, dr);

    vector<Point> strip;
    for (int i = left; i <= right; i++) {
        if (abs(points[i].x - midPoint.x) < d) {
            strip.push_back(points[i]);
        }
    }

    return min(d, stripClosest(strip, d));
}

double closestPair(vector<Point>& points) {
    sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
    });
    return closestPairHelper(points, 0, points.size() - 1);
}

// =============================================================================
// 4. 역순 쌍 개수 (Inversion Count)
// =============================================================================

long long mergeAndCount(vector<int>& arr, int left, int mid, int right) {
    vector<int> L(arr.begin() + left, arr.begin() + mid + 1);
    vector<int> R(arr.begin() + mid + 1, arr.begin() + right + 1);

    size_t i = 0, j = 0;
    int k = left;
    long long inversions = 0;

    while (i < L.size() && j < R.size()) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
            inversions += (L.size() - i);  // 역순 쌍
        }
    }

    while (i < L.size()) arr[k++] = L[i++];
    while (j < R.size()) arr[k++] = R[j++];

    return inversions;
}

long long countInversions(vector<int>& arr, int left, int right) {
    long long inversions = 0;
    if (left < right) {
        int mid = (left + right) / 2;
        inversions += countInversions(arr, left, mid);
        inversions += countInversions(arr, mid + 1, right);
        inversions += mergeAndCount(arr, left, mid, right);
    }
    return inversions;
}

// =============================================================================
// 5. 카라츠바 곱셈 (Karatsuba Multiplication)
// =============================================================================

string addStrings(const string& a, const string& b) {
    string result;
    int carry = 0;
    int i = a.size() - 1, j = b.size() - 1;

    while (i >= 0 || j >= 0 || carry) {
        int sum = carry;
        if (i >= 0) sum += a[i--] - '0';
        if (j >= 0) sum += b[j--] - '0';
        result = char(sum % 10 + '0') + result;
        carry = sum / 10;
    }

    return result.empty() ? "0" : result;
}

// =============================================================================
// 6. 최대 부분 배열 합 (분할정복)
// =============================================================================

int maxCrossingSum(const vector<int>& arr, int left, int mid, int right) {
    int leftSum = INT_MIN;
    int sum = 0;
    for (int i = mid; i >= left; i--) {
        sum += arr[i];
        leftSum = max(leftSum, sum);
    }

    int rightSum = INT_MIN;
    sum = 0;
    for (int i = mid + 1; i <= right; i++) {
        sum += arr[i];
        rightSum = max(rightSum, sum);
    }

    return leftSum + rightSum;
}

int maxSubArrayDC(const vector<int>& arr, int left, int right) {
    if (left == right) return arr[left];

    int mid = (left + right) / 2;
    return max({
        maxSubArrayDC(arr, left, mid),
        maxSubArrayDC(arr, mid + 1, right),
        maxCrossingSum(arr, left, mid, right)
    });
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "분할 정복 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 빠른 거듭제곱
    cout << "\n[1] 빠른 거듭제곱" << endl;
    cout << "    2^10 = " << power(2, 10) << endl;
    cout << "    3^20 mod 1000 = " << power(3, 20, 1000) << endl;

    // 2. 피보나치 (행렬 거듭제곱)
    cout << "\n[2] 피보나치 (행렬 거듭제곱)" << endl;
    cout << "    F(10) = " << fibonacciMatrix(10) << endl;
    cout << "    F(50) mod 10^9+7 = " << fibonacciMatrix(50) << endl;

    // 3. 가장 가까운 점 쌍
    cout << "\n[3] 가장 가까운 점 쌍" << endl;
    vector<Point> points = {{2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}};
    cout << "    점 6개" << endl;
    cout << "    최소 거리: " << closestPair(points) << endl;

    // 4. 역순 쌍 개수
    cout << "\n[4] 역순 쌍 개수" << endl;
    vector<int> inv = {8, 4, 2, 1};
    cout << "    배열: [8, 4, 2, 1]" << endl;
    cout << "    역순 쌍: " << countInversions(inv, 0, inv.size() - 1) << endl;

    // 5. 최대 부분 배열 합
    cout << "\n[5] 최대 부분 배열 합 (분할정복)" << endl;
    vector<int> arr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "    배열: [-2,1,-3,4,-1,2,1,-5,4]" << endl;
    cout << "    최대 합: " << maxSubArrayDC(arr, 0, arr.size() - 1) << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 알고리즘           | 시간복잡도     |" << endl;
    cout << "    |--------------------|----------------|" << endl;
    cout << "    | 빠른 거듭제곱      | O(log n)       |" << endl;
    cout << "    | 행렬 거듭제곱      | O(k³ log n)    |" << endl;
    cout << "    | 가장 가까운 점 쌍  | O(n log n)     |" << endl;
    cout << "    | 역순 쌍 개수       | O(n log n)     |" << endl;
    cout << "    | 병합/퀵 정렬       | O(n log n)     |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
