/*
 * 정렬 알고리즘 (Sorting Algorithms)
 * Bubble, Selection, Insertion, Merge, Quick, Heap, Counting, Radix
 *
 * 다양한 정렬 알고리즘의 구현과 비교입니다.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>

using namespace std;

// =============================================================================
// 1. 버블 정렬 - O(n²)
// =============================================================================

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;  // 최적화
    }
}

// =============================================================================
// 2. 선택 정렬 - O(n²)
// =============================================================================

void selectionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }
        swap(arr[i], arr[minIdx]);
    }
}

// =============================================================================
// 3. 삽입 정렬 - O(n²)
// =============================================================================

void insertionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// =============================================================================
// 4. 병합 정렬 - O(n log n)
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
// 5. 퀵 정렬 - O(n log n) 평균
// =============================================================================

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// 3-way 퀵소트 (중복 원소에 효율적)
void quickSort3Way(vector<int>& arr, int low, int high) {
    if (low >= high) return;

    int lt = low, gt = high;
    int pivot = arr[low];
    int i = low + 1;

    while (i <= gt) {
        if (arr[i] < pivot) {
            swap(arr[lt++], arr[i++]);
        } else if (arr[i] > pivot) {
            swap(arr[i], arr[gt--]);
        } else {
            i++;
        }
    }

    quickSort3Way(arr, low, lt - 1);
    quickSort3Way(arr, gt + 1, high);
}

// =============================================================================
// 6. 힙 정렬 - O(n log n)
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

    // 하나씩 추출
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

// =============================================================================
// 7. 계수 정렬 - O(n + k)
// =============================================================================

void countingSort(vector<int>& arr) {
    if (arr.empty()) return;

    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;

    vector<int> count(range, 0);
    vector<int> output(arr.size());

    for (int x : arr) {
        count[x - minVal]++;
    }

    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }

    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i] - minVal] - 1] = arr[i];
        count[arr[i] - minVal]--;
    }

    arr = output;
}

// =============================================================================
// 8. 기수 정렬 - O(d × (n + k))
// =============================================================================

void countingSortForRadix(vector<int>& arr, int exp) {
    int n = arr.size();
    vector<int> output(n);
    vector<int> count(10, 0);

    for (int x : arr) {
        count[(x / exp) % 10]++;
    }

    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    arr = output;
}

void radixSort(vector<int>& arr) {
    if (arr.empty()) return;

    int maxVal = *max_element(arr.begin(), arr.end());

    for (int exp = 1; maxVal / exp > 0; exp *= 10) {
        countingSortForRadix(arr, exp);
    }
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
    cout << "정렬 알고리즘 예제" << endl;
    cout << "============================================================" << endl;

    vector<int> original = {64, 34, 25, 12, 22, 11, 90};

    // 1. 버블 정렬
    cout << "\n[1] 버블 정렬" << endl;
    vector<int> arr1 = original;
    bubbleSort(arr1);
    cout << "    결과: ";
    printVector(arr1);
    cout << endl;

    // 2. 선택 정렬
    cout << "\n[2] 선택 정렬" << endl;
    vector<int> arr2 = original;
    selectionSort(arr2);
    cout << "    결과: ";
    printVector(arr2);
    cout << endl;

    // 3. 삽입 정렬
    cout << "\n[3] 삽입 정렬" << endl;
    vector<int> arr3 = original;
    insertionSort(arr3);
    cout << "    결과: ";
    printVector(arr3);
    cout << endl;

    // 4. 병합 정렬
    cout << "\n[4] 병합 정렬" << endl;
    vector<int> arr4 = original;
    mergeSort(arr4, 0, arr4.size() - 1);
    cout << "    결과: ";
    printVector(arr4);
    cout << endl;

    // 5. 퀵 정렬
    cout << "\n[5] 퀵 정렬" << endl;
    vector<int> arr5 = original;
    quickSort(arr5, 0, arr5.size() - 1);
    cout << "    결과: ";
    printVector(arr5);
    cout << endl;

    // 6. 힙 정렬
    cout << "\n[6] 힙 정렬" << endl;
    vector<int> arr6 = original;
    heapSort(arr6);
    cout << "    결과: ";
    printVector(arr6);
    cout << endl;

    // 7. 계수 정렬
    cout << "\n[7] 계수 정렬" << endl;
    vector<int> arr7 = original;
    countingSort(arr7);
    cout << "    결과: ";
    printVector(arr7);
    cout << endl;

    // 8. 기수 정렬
    cout << "\n[8] 기수 정렬" << endl;
    vector<int> arr8 = {170, 45, 75, 90, 802, 24, 2, 66};
    cout << "    원본: ";
    printVector(arr8);
    cout << endl;
    radixSort(arr8);
    cout << "    결과: ";
    printVector(arr8);
    cout << endl;

    // 9. 복잡도 비교
    cout << "\n[9] 정렬 알고리즘 복잡도 비교" << endl;
    cout << "    | 알고리즘   | 최선      | 평균      | 최악      | 공간  | 안정 |" << endl;
    cout << "    |------------|-----------|-----------|-----------|-------|------|" << endl;
    cout << "    | 버블       | O(n)      | O(n²)     | O(n²)     | O(1)  | O    |" << endl;
    cout << "    | 선택       | O(n²)     | O(n²)     | O(n²)     | O(1)  | X    |" << endl;
    cout << "    | 삽입       | O(n)      | O(n²)     | O(n²)     | O(1)  | O    |" << endl;
    cout << "    | 병합       | O(n log n)| O(n log n)| O(n log n)| O(n)  | O    |" << endl;
    cout << "    | 퀵         | O(n log n)| O(n log n)| O(n²)     | O(log)| X    |" << endl;
    cout << "    | 힙         | O(n log n)| O(n log n)| O(n log n)| O(1)  | X    |" << endl;
    cout << "    | 계수       | O(n+k)    | O(n+k)    | O(n+k)    | O(k)  | O    |" << endl;
    cout << "    | 기수       | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k)| O    |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
