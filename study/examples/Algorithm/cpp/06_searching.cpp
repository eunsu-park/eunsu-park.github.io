/*
 * 탐색 알고리즘 (Searching Algorithms)
 * Binary Search, Lower/Upper Bound, Parametric Search
 *
 * 효율적인 탐색 기법들입니다.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// =============================================================================
// 1. 이분 탐색 기본
// =============================================================================

int binarySearch(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }

    return -1;
}

// 재귀 버전
int binarySearchRecursive(const vector<int>& arr, int target, int left, int right) {
    if (left > right) return -1;

    int mid = left + (right - left) / 2;

    if (arr[mid] == target)
        return mid;
    else if (arr[mid] < target)
        return binarySearchRecursive(arr, target, mid + 1, right);
    else
        return binarySearchRecursive(arr, target, left, mid - 1);
}

// =============================================================================
// 2. Lower Bound / Upper Bound
// =============================================================================

// target 이상인 첫 번째 위치
int lowerBound(const vector<int>& arr, int target) {
    int left = 0, right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

// target 초과인 첫 번째 위치
int upperBound(const vector<int>& arr, int target) {
    int left = 0, right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

// target의 개수
int countOccurrences(const vector<int>& arr, int target) {
    return upperBound(arr, target) - lowerBound(arr, target);
}

// =============================================================================
// 3. 회전 정렬 배열 탐색
// =============================================================================

int searchRotated(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;

        // 왼쪽이 정렬됨
        if (arr[left] <= arr[mid]) {
            if (arr[left] <= target && target < arr[mid])
                right = mid - 1;
            else
                left = mid + 1;
        }
        // 오른쪽이 정렬됨
        else {
            if (arr[mid] < target && target <= arr[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }

    return -1;
}

// 회전 배열의 최솟값
int findMin(const vector<int>& arr) {
    int left = 0, right = arr.size() - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] > arr[right])
            left = mid + 1;
        else
            right = mid;
    }

    return arr[left];
}

// =============================================================================
// 4. 파라메트릭 서치
// =============================================================================

// 나무 자르기 (최대 높이)
long long cutWood(const vector<int>& trees, int h) {
    long long total = 0;
    for (int tree : trees) {
        if (tree > h) {
            total += tree - h;
        }
    }
    return total;
}

int maxCuttingHeight(const vector<int>& trees, long long need) {
    int left = 0;
    int right = *max_element(trees.begin(), trees.end());
    int result = 0;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (cutWood(trees, mid) >= need) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

// 배송 최소 용량
bool canShip(const vector<int>& weights, int capacity, int days) {
    int currentWeight = 0;
    int dayCount = 1;

    for (int w : weights) {
        if (w > capacity) return false;
        if (currentWeight + w > capacity) {
            dayCount++;
            currentWeight = w;
        } else {
            currentWeight += w;
        }
    }

    return dayCount <= days;
}

int shipWithinDays(const vector<int>& weights, int days) {
    int left = *max_element(weights.begin(), weights.end());
    int right = 0;
    for (int w : weights) right += w;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canShip(weights, mid, days))
            right = mid;
        else
            left = mid + 1;
    }

    return left;
}

// =============================================================================
// 5. 실수 이분 탐색
// =============================================================================

// 제곱근 (소수점까지)
double sqrtBinary(double x, double precision = 1e-10) {
    if (x < 0) return -1;
    if (x < 1) {
        double lo = x, hi = 1;
        while (hi - lo > precision) {
            double mid = (lo + hi) / 2;
            if (mid * mid < x)
                lo = mid;
            else
                hi = mid;
        }
        return (lo + hi) / 2;
    }

    double lo = 1, hi = x;
    while (hi - lo > precision) {
        double mid = (lo + hi) / 2;
        if (mid * mid < x)
            lo = mid;
        else
            hi = mid;
    }
    return (lo + hi) / 2;
}

// =============================================================================
// 6. 2D 배열 탐색
// =============================================================================

// 행과 열이 정렬된 2D 배열 탐색
bool searchMatrix(const vector<vector<int>>& matrix, int target) {
    if (matrix.empty()) return false;

    int rows = matrix.size();
    int cols = matrix[0].size();
    int row = 0, col = cols - 1;

    while (row < rows && col >= 0) {
        if (matrix[row][col] == target)
            return true;
        else if (matrix[row][col] > target)
            col--;
        else
            row++;
    }

    return false;
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "탐색 알고리즘 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 이분 탐색
    cout << "\n[1] 이분 탐색" << endl;
    vector<int> arr = {1, 3, 5, 7, 9, 11, 13, 15};
    cout << "    배열: [1,3,5,7,9,11,13,15]" << endl;
    cout << "    7의 위치: " << binarySearch(arr, 7) << endl;
    cout << "    6의 위치: " << binarySearch(arr, 6) << endl;

    // 2. Lower/Upper Bound
    cout << "\n[2] Lower/Upper Bound" << endl;
    vector<int> arr2 = {1, 2, 2, 2, 3, 4, 5};
    cout << "    배열: [1,2,2,2,3,4,5]" << endl;
    cout << "    lower_bound(2): " << lowerBound(arr2, 2) << endl;
    cout << "    upper_bound(2): " << upperBound(arr2, 2) << endl;
    cout << "    2의 개수: " << countOccurrences(arr2, 2) << endl;

    // 3. 회전 정렬 배열
    cout << "\n[3] 회전 정렬 배열" << endl;
    vector<int> rotated = {4, 5, 6, 7, 0, 1, 2};
    cout << "    배열: [4,5,6,7,0,1,2]" << endl;
    cout << "    0의 위치: " << searchRotated(rotated, 0) << endl;
    cout << "    최솟값: " << findMin(rotated) << endl;

    // 4. 파라메트릭 서치
    cout << "\n[4] 파라메트릭 서치" << endl;
    vector<int> trees = {20, 15, 10, 17};
    cout << "    나무: [20,15,10,17], 필요량: 7" << endl;
    cout << "    최대 절단 높이: " << maxCuttingHeight(trees, 7) << endl;

    vector<int> weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cout << "    물건: [1-10], 5일 배송" << endl;
    cout << "    최소 용량: " << shipWithinDays(weights, 5) << endl;

    // 5. 실수 이분 탐색
    cout << "\n[5] 실수 이분 탐색" << endl;
    cout << "    sqrt(2) = " << sqrtBinary(2) << endl;
    cout << "    sqrt(10) = " << sqrtBinary(10) << endl;

    // 6. 2D 배열 탐색
    cout << "\n[6] 2D 배열 탐색" << endl;
    vector<vector<int>> matrix = {
        {1, 4, 7, 11},
        {2, 5, 8, 12},
        {3, 6, 9, 16},
        {10, 13, 14, 17}
    };
    cout << "    5 찾기: " << (searchMatrix(matrix, 5) ? "있음" : "없음") << endl;
    cout << "    15 찾기: " << (searchMatrix(matrix, 15) ? "있음" : "없음") << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 알고리즘          | 시간복잡도 |" << endl;
    cout << "    |-------------------|------------|" << endl;
    cout << "    | 선형 탐색         | O(n)       |" << endl;
    cout << "    | 이분 탐색         | O(log n)   |" << endl;
    cout << "    | 파라메트릭 서치   | O(log M × f(n)) |" << endl;
    cout << "    | 2D 행렬 탐색      | O(m + n)   |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
