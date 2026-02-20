/*
 * 시간 복잡도 (Time Complexity)
 * Big O Notation and Complexity Analysis
 *
 * 알고리즘의 효율성을 분석하는 방법입니다.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <functional>

using namespace std;

// =============================================================================
// 1. O(1) - 상수 시간
// =============================================================================

int constantTime(const vector<int>& arr) {
    // 배열 크기와 무관하게 항상 일정한 시간
    return arr[0];
}

// =============================================================================
// 2. O(log n) - 로그 시간
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

// =============================================================================
// 3. O(n) - 선형 시간
// =============================================================================

int linearSearch(const vector<int>& arr, int target) {
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == target)
            return i;
    }
    return -1;
}

int sumArray(const vector<int>& arr) {
    int sum = 0;
    for (int x : arr) {
        sum += x;
    }
    return sum;
}

// =============================================================================
// 4. O(n log n) - 선형 로그 시간
// =============================================================================

void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> L(arr.begin() + left, arr.begin() + mid + 1);
    vector<int> R(arr.begin() + mid + 1, arr.begin() + right + 1);

    size_t i = 0, j = 0;
    int k = left;

    while (i < L.size() && j < R.size()) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < L.size()) arr[k++] = L[i++];
    while (j < R.size()) arr[k++] = R[j++];
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// =============================================================================
// 5. O(n²) - 이차 시간
// =============================================================================

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int countPairs(const vector<int>& arr) {
    int count = 0;
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            count++;
        }
    }
    return count;  // n*(n-1)/2
}

// =============================================================================
// 6. O(2^n) - 지수 시간
// =============================================================================

int fibonacciRecursive(int n) {
    if (n <= 1) return n;
    return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
}

// O(n) 최적화 버전
int fibonacciIterative(int n) {
    if (n <= 1) return n;

    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// =============================================================================
// 7. 시간 측정 유틸리티
// =============================================================================

template<typename Func, typename... Args>
double measureTime(Func func, Args&&... args) {
    auto start = chrono::high_resolution_clock::now();
    func(forward<Args>(args)...);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    return diff.count();
}

// =============================================================================
// 8. 공간 복잡도 예제
// =============================================================================

// O(1) 공간
void reverseInPlace(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n / 2; i++) {
        swap(arr[i], arr[n - 1 - i]);
    }
}

// O(n) 공간
vector<int> reverseWithCopy(const vector<int>& arr) {
    vector<int> result(arr.rbegin(), arr.rend());
    return result;
}

// =============================================================================
// 테스트
// =============================================================================

void printArray(const vector<int>& arr, int limit = 10) {
    cout << "    [";
    for (size_t i = 0; i < arr.size() && i < (size_t)limit; i++) {
        cout << arr[i];
        if (i < arr.size() - 1 && i < (size_t)limit - 1) cout << ", ";
    }
    if (arr.size() > (size_t)limit) cout << ", ...";
    cout << "]" << endl;
}

int main() {
    cout << "============================================================" << endl;
    cout << "시간 복잡도 (Time Complexity) 예제" << endl;
    cout << "============================================================" << endl;

    // 1. O(1)
    cout << "\n[1] O(1) - 상수 시간" << endl;
    vector<int> arr1 = {5, 2, 8, 1, 9};
    cout << "    첫 번째 원소: " << constantTime(arr1) << endl;

    // 2. O(log n)
    cout << "\n[2] O(log n) - 이분 탐색" << endl;
    vector<int> arr2 = {1, 3, 5, 7, 9, 11, 13, 15};
    int idx = binarySearch(arr2, 7);
    cout << "    배열: [1,3,5,7,9,11,13,15]" << endl;
    cout << "    7의 위치: " << idx << endl;

    // 3. O(n)
    cout << "\n[3] O(n) - 선형 탐색" << endl;
    vector<int> arr3 = {4, 2, 7, 1, 9, 3};
    cout << "    배열 합: " << sumArray(arr3) << endl;

    // 4. O(n log n)
    cout << "\n[4] O(n log n) - 병합 정렬" << endl;
    vector<int> arr4 = {64, 34, 25, 12, 22, 11, 90};
    cout << "    정렬 전: ";
    printArray(arr4);
    mergeSort(arr4, 0, arr4.size() - 1);
    cout << "    정렬 후: ";
    printArray(arr4);

    // 5. O(n²)
    cout << "\n[5] O(n²) - 버블 정렬" << endl;
    vector<int> arr5 = {64, 34, 25, 12, 22, 11, 90};
    bubbleSort(arr5);
    cout << "    정렬 후: ";
    printArray(arr5);
    cout << "    5개 원소의 쌍 개수: " << countPairs(vector<int>(5)) << endl;

    // 6. O(2^n) vs O(n)
    cout << "\n[6] O(2^n) vs O(n) - 피보나치" << endl;
    cout << "    피보나치(20) 재귀: " << fibonacciRecursive(20) << endl;
    cout << "    피보나치(20) 반복: " << fibonacciIterative(20) << endl;
    cout << "    피보나치(40) 반복: " << fibonacciIterative(40) << endl;

    // 7. 공간 복잡도
    cout << "\n[7] 공간 복잡도" << endl;
    vector<int> arr7 = {1, 2, 3, 4, 5};
    cout << "    원본: ";
    printArray(arr7);
    reverseInPlace(arr7);
    cout << "    O(1) 공간 뒤집기: ";
    printArray(arr7);

    // 8. 복잡도 요약
    cout << "\n[8] 복잡도 요약" << endl;
    cout << "    | 복잡도    | 1000개 연산 | 예시              |" << endl;
    cout << "    |-----------|-------------|-------------------|" << endl;
    cout << "    | O(1)      | 1           | 배열 인덱싱       |" << endl;
    cout << "    | O(log n)  | 10          | 이분 탐색         |" << endl;
    cout << "    | O(n)      | 1000        | 선형 탐색         |" << endl;
    cout << "    | O(n log n)| 10000       | 병합 정렬         |" << endl;
    cout << "    | O(n²)     | 1000000     | 버블 정렬         |" << endl;
    cout << "    | O(2^n)    | 매우 큼     | 모든 부분집합     |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
