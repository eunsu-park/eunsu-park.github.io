/*
 * 분할 정복 (Divide and Conquer)
 * Merge Sort, Quick Sort, Fast Exponentiation, Closest Pair
 *
 * 문제를 작은 부분으로 나누어 해결하는 기법입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* =============================================================================
 * 1. 병합 정렬
 * ============================================================================= */

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = malloc(n1 * sizeof(int));
    int* R = malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

/* =============================================================================
 * 2. 퀵 정렬
 * ============================================================================= */

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

/* =============================================================================
 * 3. 빠른 거듭제곱
 * ============================================================================= */

long long power(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;

    while (exp > 0) {
        if (exp & 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }

    return result;
}

/* 행렬 거듭제곱 */
typedef struct {
    long long a[2][2];
} Matrix;

Matrix matrix_multiply(Matrix A, Matrix B, long long mod) {
    Matrix C = {{{0}}};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                C.a[i][j] = (C.a[i][j] + A.a[i][k] * B.a[k][j]) % mod;
            }
        }
    }

    return C;
}

Matrix matrix_power(Matrix M, long long n, long long mod) {
    Matrix result = {{{1, 0}, {0, 1}}};  /* 단위 행렬 */

    while (n > 0) {
        if (n & 1)
            result = matrix_multiply(result, M, mod);
        M = matrix_multiply(M, M, mod);
        n >>= 1;
    }

    return result;
}

/* 피보나치 O(log n) */
long long fibonacci_matrix(long long n, long long mod) {
    if (n <= 1) return n;

    Matrix M = {{{1, 1}, {1, 0}}};
    Matrix result = matrix_power(M, n - 1, mod);

    return result.a[0][0];
}

/* =============================================================================
 * 4. 가장 가까운 점 쌍
 * ============================================================================= */

typedef struct {
    double x, y;
} Point;

int compare_x(const void* a, const void* b) {
    Point* p1 = (Point*)a;
    Point* p2 = (Point*)b;
    return (p1->x > p2->x) - (p1->x < p2->x);
}

int compare_y(const void* a, const void* b) {
    Point* p1 = (Point*)a;
    Point* p2 = (Point*)b;
    return (p1->y > p2->y) - (p1->y < p2->y);
}

double dist(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                (p1.y - p2.y) * (p1.y - p2.y));
}

double min_double(double a, double b) {
    return (a < b) ? a : b;
}

double brute_force(Point points[], int n) {
    double min_dist = DBL_MAX;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double d = dist(points[i], points[j]);
            if (d < min_dist) min_dist = d;
        }
    }
    return min_dist;
}

double strip_closest(Point strip[], int size, double d) {
    double min_dist = d;
    qsort(strip, size, sizeof(Point), compare_y);

    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size && (strip[j].y - strip[i].y) < min_dist; j++) {
            double dist_ij = dist(strip[i], strip[j]);
            if (dist_ij < min_dist)
                min_dist = dist_ij;
        }
    }

    return min_dist;
}

double closest_pair_impl(Point points[], int n) {
    if (n <= 3)
        return brute_force(points, n);

    int mid = n / 2;
    Point mid_point = points[mid];

    double dl = closest_pair_impl(points, mid);
    double dr = closest_pair_impl(points + mid, n - mid);
    double d = min_double(dl, dr);

    Point* strip = malloc(n * sizeof(Point));
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(points[i].x - mid_point.x) < d)
            strip[j++] = points[i];
    }

    double strip_d = strip_closest(strip, j, d);
    free(strip);

    return min_double(d, strip_d);
}

double closest_pair(Point points[], int n) {
    qsort(points, n, sizeof(Point), compare_x);
    return closest_pair_impl(points, n);
}

/* =============================================================================
 * 5. 역순 쌍 개수 (Inversion Count)
 * ============================================================================= */

long long merge_count(int arr[], int temp[], int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    long long inv_count = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            inv_count += (mid - i + 1);
        }
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (i = left; i <= right; i++)
        arr[i] = temp[i];

    return inv_count;
}

long long merge_sort_count(int arr[], int temp[], int left, int right) {
    long long inv_count = 0;

    if (left < right) {
        int mid = left + (right - left) / 2;
        inv_count += merge_sort_count(arr, temp, left, mid);
        inv_count += merge_sort_count(arr, temp, mid + 1, right);
        inv_count += merge_count(arr, temp, left, mid, right);
    }

    return inv_count;
}

long long count_inversions(int arr[], int n) {
    int* temp = malloc(n * sizeof(int));
    long long count = merge_sort_count(arr, temp, 0, n - 1);
    free(temp);
    return count;
}

/* =============================================================================
 * 6. 카라츠바 곱셈
 * ============================================================================= */

long long karatsuba(long long x, long long y) {
    if (x < 10 || y < 10)
        return x * y;

    int n = 0;
    long long temp = (x > y) ? x : y;
    while (temp > 0) {
        n++;
        temp /= 10;
    }

    long long half = 1;
    for (int i = 0; i < n / 2; i++)
        half *= 10;

    long long a = x / half;
    long long b = x % half;
    long long c = y / half;
    long long d = y % half;

    long long ac = karatsuba(a, c);
    long long bd = karatsuba(b, d);
    long long ad_bc = karatsuba(a + b, c + d) - ac - bd;

    return ac * half * half + ad_bc * half + bd;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

int main(void) {
    printf("============================================================\n");
    printf("분할 정복 (Divide and Conquer) 예제\n");
    printf("============================================================\n");

    /* 1. 병합 정렬 */
    printf("\n[1] 병합 정렬\n");
    int arr1[] = {38, 27, 43, 3, 9, 82, 10};
    printf("    정렬 전: ");
    print_array(arr1, 7);
    merge_sort(arr1, 0, 6);
    printf("    정렬 후: ");
    print_array(arr1, 7);

    /* 2. 퀵 정렬 */
    printf("\n[2] 퀵 정렬\n");
    int arr2[] = {10, 7, 8, 9, 1, 5};
    printf("    정렬 전: ");
    print_array(arr2, 6);
    quick_sort(arr2, 0, 5);
    printf("    정렬 후: ");
    print_array(arr2, 6);

    /* 3. 빠른 거듭제곱 */
    printf("\n[3] 빠른 거듭제곱\n");
    printf("    2^10 mod 1000 = %lld\n", power(2, 10, 1000));
    printf("    3^20 mod 1000000007 = %lld\n", power(3, 20, 1000000007));

    /* 4. 행렬 거듭제곱 - 피보나치 */
    printf("\n[4] 행렬 거듭제곱 - 피보나치\n");
    printf("    fib(10) = %lld\n", fibonacci_matrix(10, 1000000007));
    printf("    fib(50) = %lld\n", fibonacci_matrix(50, 1000000007));

    /* 5. 가장 가까운 점 쌍 */
    printf("\n[5] 가장 가까운 점 쌍\n");
    Point points[] = {{2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}};
    printf("    점들: (2,3), (12,30), (40,50), (5,1), (12,10), (3,4)\n");
    printf("    최소 거리: %.4f\n", closest_pair(points, 6));

    /* 6. 역순 쌍 개수 */
    printf("\n[6] 역순 쌍 개수\n");
    int arr6[] = {8, 4, 2, 1};
    printf("    배열: [8, 4, 2, 1]\n");
    printf("    역순 쌍: %lld\n", count_inversions(arr6, 4));

    /* 7. 카라츠바 곱셈 */
    printf("\n[7] 카라츠바 곱셈\n");
    printf("    1234 × 5678 = %lld\n", karatsuba(1234, 5678));
    printf("    검증: %lld\n", (long long)1234 * 5678);

    /* 8. 알고리즘 요약 */
    printf("\n[8] 분할 정복 요약\n");
    printf("    | 알고리즘      | 시간 복잡도   | 분할 방식        |\n");
    printf("    |---------------|---------------|------------------|\n");
    printf("    | 병합 정렬     | O(n log n)    | 반씩 분할        |\n");
    printf("    | 퀵 정렬       | O(n log n)    | 피벗 기준 분할   |\n");
    printf("    | 거듭제곱      | O(log n)      | 지수 반감        |\n");
    printf("    | 가장 가까운 쌍| O(n log n)    | 좌표 기준 분할   |\n");
    printf("    | 역순 쌍 개수  | O(n log n)    | 병합 시 계산     |\n");

    printf("\n============================================================\n");

    return 0;
}
