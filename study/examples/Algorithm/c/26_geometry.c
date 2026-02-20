/*
 * 기하 알고리즘 (Computational Geometry)
 * CCW, 볼록 껍질, 선분 교차, 다각형
 *
 * 2D 평면에서의 기하학적 연산입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define EPS 1e-9
#define PI 3.14159265358979323846

/* =============================================================================
 * 1. 점과 벡터
 * ============================================================================= */

typedef struct {
    double x, y;
} Point;

Point point_create(double x, double y) {
    Point p = {x, y};
    return p;
}

Point point_add(Point a, Point b) {
    return point_create(a.x + b.x, a.y + b.y);
}

Point point_sub(Point a, Point b) {
    return point_create(a.x - b.x, a.y - b.y);
}

Point point_scale(Point p, double k) {
    return point_create(p.x * k, p.y * k);
}

double point_dot(Point a, Point b) {
    return a.x * b.x + a.y * b.y;
}

double point_cross(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

double point_norm(Point p) {
    return sqrt(p.x * p.x + p.y * p.y);
}

double point_dist(Point a, Point b) {
    return point_norm(point_sub(a, b));
}

/* =============================================================================
 * 2. CCW (Counter-Clockwise)
 * ============================================================================= */

/* 반시계 방향: 1, 시계 방향: -1, 일직선: 0 */
int ccw(Point a, Point b, Point c) {
    double cross = point_cross(point_sub(b, a), point_sub(c, a));
    if (cross > EPS) return 1;
    if (cross < -EPS) return -1;
    return 0;
}

/* =============================================================================
 * 3. 선분 교차 판정
 * ============================================================================= */

bool on_segment(Point p, Point a, Point b) {
    double min_x = (a.x < b.x) ? a.x : b.x;
    double max_x = (a.x > b.x) ? a.x : b.x;
    double min_y = (a.y < b.y) ? a.y : b.y;
    double max_y = (a.y > b.y) ? a.y : b.y;

    return p.x >= min_x - EPS && p.x <= max_x + EPS &&
           p.y >= min_y - EPS && p.y <= max_y + EPS;
}

bool segments_intersect(Point a, Point b, Point c, Point d) {
    int d1 = ccw(a, b, c);
    int d2 = ccw(a, b, d);
    int d3 = ccw(c, d, a);
    int d4 = ccw(c, d, b);

    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true;
    }

    /* 일직선 상의 교차 */
    if (d1 == 0 && on_segment(c, a, b)) return true;
    if (d2 == 0 && on_segment(d, a, b)) return true;
    if (d3 == 0 && on_segment(a, c, d)) return true;
    if (d4 == 0 && on_segment(b, c, d)) return true;

    return false;
}

/* 선분 교차점 계산 */
bool line_intersection(Point a, Point b, Point c, Point d, Point* result) {
    double a1 = b.y - a.y;
    double b1 = a.x - b.x;
    double c1 = a1 * a.x + b1 * a.y;

    double a2 = d.y - c.y;
    double b2 = c.x - d.x;
    double c2 = a2 * c.x + b2 * c.y;

    double det = a1 * b2 - a2 * b1;
    if (fabs(det) < EPS) return false;  /* 평행 */

    result->x = (b2 * c1 - b1 * c2) / det;
    result->y = (a1 * c2 - a2 * c1) / det;
    return true;
}

/* =============================================================================
 * 4. 볼록 껍질 (Convex Hull)
 * ============================================================================= */

int compare_points(const void* a, const void* b) {
    Point* p1 = (Point*)a;
    Point* p2 = (Point*)b;
    if (fabs(p1->x - p2->x) > EPS)
        return (p1->x < p2->x) ? -1 : 1;
    return (p1->y < p2->y) ? -1 : 1;
}

/* Graham Scan */
int convex_hull_graham(Point points[], int n, Point hull[]) {
    if (n < 3) return 0;

    /* 가장 아래, 왼쪽 점 찾기 */
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].y < points[min_idx].y ||
            (fabs(points[i].y - points[min_idx].y) < EPS &&
             points[i].x < points[min_idx].x)) {
            min_idx = i;
        }
    }

    Point pivot = points[min_idx];
    points[min_idx] = points[0];
    points[0] = pivot;

    /* 각도 기준 정렬 */
    for (int i = 1; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            int c = ccw(pivot, points[i], points[j]);
            if (c < 0 || (c == 0 &&
                point_dist(pivot, points[i]) > point_dist(pivot, points[j]))) {
                Point temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
        }
    }

    /* 스택 사용 */
    int hull_size = 0;
    for (int i = 0; i < n; i++) {
        while (hull_size >= 2 &&
               ccw(hull[hull_size - 2], hull[hull_size - 1], points[i]) <= 0) {
            hull_size--;
        }
        hull[hull_size++] = points[i];
    }

    return hull_size;
}

/* Monotone Chain */
int convex_hull_monotone(Point points[], int n, Point hull[]) {
    if (n < 3) return 0;

    qsort(points, n, sizeof(Point), compare_points);

    int hull_size = 0;

    /* 아래 껍질 */
    for (int i = 0; i < n; i++) {
        while (hull_size >= 2 &&
               ccw(hull[hull_size - 2], hull[hull_size - 1], points[i]) <= 0) {
            hull_size--;
        }
        hull[hull_size++] = points[i];
    }

    /* 위 껍질 */
    int lower_size = hull_size;
    for (int i = n - 2; i >= 0; i--) {
        while (hull_size > lower_size &&
               ccw(hull[hull_size - 2], hull[hull_size - 1], points[i]) <= 0) {
            hull_size--;
        }
        hull[hull_size++] = points[i];
    }

    return hull_size - 1;  /* 마지막 점은 첫 점과 동일 */
}

/* =============================================================================
 * 5. 다각형 연산
 * ============================================================================= */

/* 다각형 넓이 (신발끈 공식) */
double polygon_area(Point poly[], int n) {
    double area = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += point_cross(poly[i], poly[j]);
    }
    return fabs(area) / 2.0;
}

/* 점이 다각형 내부에 있는지 (Ray Casting) */
bool point_in_polygon(Point p, Point poly[], int n) {
    int crossings = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        if ((poly[i].y <= p.y && p.y < poly[j].y) ||
            (poly[j].y <= p.y && p.y < poly[i].y)) {
            double x = (poly[j].x - poly[i].x) * (p.y - poly[i].y) /
                       (poly[j].y - poly[i].y) + poly[i].x;
            if (p.x < x) crossings++;
        }
    }
    return (crossings % 2) == 1;
}

/* 다각형 무게중심 */
Point polygon_centroid(Point poly[], int n) {
    double cx = 0, cy = 0, area = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        double cross = point_cross(poly[i], poly[j]);
        cx += (poly[i].x + poly[j].x) * cross;
        cy += (poly[i].y + poly[j].y) * cross;
        area += cross;
    }
    area /= 2.0;
    return point_create(cx / (6.0 * area), cy / (6.0 * area));
}

/* =============================================================================
 * 6. 가장 가까운 점 쌍 (Closest Pair)
 * ============================================================================= */

double min_double(double a, double b) {
    return (a < b) ? a : b;
}

int compare_by_y(const void* a, const void* b) {
    Point* p1 = (Point*)a;
    Point* p2 = (Point*)b;
    return (p1->y < p2->y) ? -1 : 1;
}

double closest_pair_recursive(Point px[], Point py[], int n) {
    if (n <= 3) {
        double min_dist = 1e18;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = point_dist(px[i], px[j]);
                if (d < min_dist) min_dist = d;
            }
        }
        return min_dist;
    }

    int mid = n / 2;
    Point mid_point = px[mid];

    Point* pyl = malloc(mid * sizeof(Point));
    Point* pyr = malloc((n - mid) * sizeof(Point));
    int li = 0, ri = 0;

    for (int i = 0; i < n; i++) {
        if (py[i].x <= mid_point.x && li < mid)
            pyl[li++] = py[i];
        else
            pyr[ri++] = py[i];
    }

    double dl = closest_pair_recursive(px, pyl, mid);
    double dr = closest_pair_recursive(px + mid, pyr, n - mid);
    double d = min_double(dl, dr);

    free(pyl);
    free(pyr);

    /* 스트립 내 점들 */
    Point* strip = malloc(n * sizeof(Point));
    int strip_n = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(py[i].x - mid_point.x) < d) {
            strip[strip_n++] = py[i];
        }
    }

    for (int i = 0; i < strip_n; i++) {
        for (int j = i + 1; j < strip_n && (strip[j].y - strip[i].y) < d; j++) {
            double dist = point_dist(strip[i], strip[j]);
            if (dist < d) d = dist;
        }
    }

    free(strip);
    return d;
}

double closest_pair(Point points[], int n) {
    Point* px = malloc(n * sizeof(Point));
    Point* py = malloc(n * sizeof(Point));
    for (int i = 0; i < n; i++) {
        px[i] = py[i] = points[i];
    }

    qsort(px, n, sizeof(Point), compare_points);
    qsort(py, n, sizeof(Point), compare_by_y);

    double result = closest_pair_recursive(px, py, n);

    free(px);
    free(py);
    return result;
}

/* =============================================================================
 * 7. 회전 캘리퍼스 (Rotating Calipers)
 * ============================================================================= */

/* 볼록 다각형의 지름 (가장 먼 두 점 사이 거리) */
double rotating_calipers(Point hull[], int n) {
    if (n < 2) return 0;
    if (n == 2) return point_dist(hull[0], hull[1]);

    int j = 1;
    double max_dist = 0;

    for (int i = 0; i < n; i++) {
        Point edge = point_sub(hull[(i + 1) % n], hull[i]);

        while (1) {
            Point next_edge = point_sub(hull[(j + 1) % n], hull[j]);
            double cross = point_cross(edge, point_sub(hull[(j + 1) % n], hull[j]));
            if (cross <= 0) break;
            j = (j + 1) % n;
        }

        double d1 = point_dist(hull[i], hull[j]);
        double d2 = point_dist(hull[(i + 1) % n], hull[j]);
        if (d1 > max_dist) max_dist = d1;
        if (d2 > max_dist) max_dist = d2;
    }

    return max_dist;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("기하 알고리즘 예제\n");
    printf("============================================================\n");

    /* 1. CCW */
    printf("\n[1] CCW (반시계 방향 판정)\n");
    Point a = {0, 0}, b = {4, 0}, c = {2, 2};
    printf("    A(0,0), B(4,0), C(2,2)\n");
    int result = ccw(a, b, c);
    printf("    CCW: %d (%s)\n", result,
           result > 0 ? "반시계" : (result < 0 ? "시계" : "일직선"));

    /* 2. 선분 교차 */
    printf("\n[2] 선분 교차 판정\n");
    Point p1 = {0, 0}, p2 = {4, 4}, p3 = {0, 4}, p4 = {4, 0};
    printf("    선분1: (0,0)-(4,4), 선분2: (0,4)-(4,0)\n");
    printf("    교차: %s\n", segments_intersect(p1, p2, p3, p4) ? "예" : "아니오");

    Point intersection;
    if (line_intersection(p1, p2, p3, p4, &intersection)) {
        printf("    교차점: (%.1f, %.1f)\n", intersection.x, intersection.y);
    }

    /* 3. 볼록 껍질 */
    printf("\n[3] 볼록 껍질\n");
    Point points[] = {{0, 0}, {1, 1}, {2, 2}, {4, 4}, {0, 4}, {4, 0}, {2, 1}, {1, 2}};
    int n = 8;
    Point hull[10];

    printf("    점들: ");
    for (int i = 0; i < n; i++) {
        printf("(%.0f,%.0f) ", points[i].x, points[i].y);
    }
    printf("\n");

    int hull_size = convex_hull_monotone(points, n, hull);
    printf("    볼록 껍질 (%d개): ", hull_size);
    for (int i = 0; i < hull_size; i++) {
        printf("(%.0f,%.0f) ", hull[i].x, hull[i].y);
    }
    printf("\n");

    /* 4. 다각형 넓이 */
    printf("\n[4] 다각형 넓이\n");
    Point triangle[] = {{0, 0}, {4, 0}, {2, 3}};
    printf("    삼각형: (0,0), (4,0), (2,3)\n");
    printf("    넓이: %.1f\n", polygon_area(triangle, 3));

    Point square[] = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    printf("    정사각형: (0,0), (4,0), (4,4), (0,4)\n");
    printf("    넓이: %.1f\n", polygon_area(square, 4));

    /* 5. 점의 다각형 내부 판정 */
    printf("\n[5] 점의 다각형 내부 판정\n");
    Point test1 = {2, 2}, test2 = {5, 5};
    printf("    정사각형: (0,0), (4,0), (4,4), (0,4)\n");
    printf("    점 (2,2): %s\n", point_in_polygon(test1, square, 4) ? "내부" : "외부");
    printf("    점 (5,5): %s\n", point_in_polygon(test2, square, 4) ? "내부" : "외부");

    /* 6. 가장 가까운 점 쌍 */
    printf("\n[6] 가장 가까운 점 쌍\n");
    Point closest_points[] = {{2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}};
    printf("    점들: (2,3), (12,30), (40,50), (5,1), (12,10), (3,4)\n");
    printf("    최소 거리: %.4f\n", closest_pair(closest_points, 6));

    /* 7. 복잡도 */
    printf("\n[7] 복잡도\n");
    printf("    | 알고리즘        | 시간복잡도    |\n");
    printf("    |-----------------|---------------|\n");
    printf("    | CCW             | O(1)          |\n");
    printf("    | 선분 교차       | O(1)          |\n");
    printf("    | 볼록 껍질       | O(n log n)    |\n");
    printf("    | 다각형 넓이     | O(n)          |\n");
    printf("    | 점 내부 판정    | O(n)          |\n");
    printf("    | 가장 가까운 쌍  | O(n log n)    |\n");
    printf("    | 회전 캘리퍼스   | O(n)          |\n");

    printf("\n============================================================\n");

    return 0;
}
