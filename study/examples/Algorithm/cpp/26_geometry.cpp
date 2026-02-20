/*
 * 기하 알고리즘 (Computational Geometry)
 * CCW, Convex Hull, Line Intersection, Point in Polygon
 *
 * 점, 선, 다각형 등 기하학적 문제를 해결합니다.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stack>

using namespace std;

const double EPS = 1e-9;
const double PI = acos(-1.0);

// =============================================================================
// 1. 점과 벡터
// =============================================================================

struct Point {
    double x, y;

    Point(double x = 0, double y = 0) : x(x), y(y) {}

    Point operator+(const Point& p) const { return Point(x + p.x, y + p.y); }
    Point operator-(const Point& p) const { return Point(x - p.x, y - p.y); }
    Point operator*(double t) const { return Point(x * t, y * t); }
    Point operator/(double t) const { return Point(x / t, y / t); }

    bool operator<(const Point& p) const {
        if (abs(x - p.x) > EPS) return x < p.x;
        return y < p.y;
    }

    bool operator==(const Point& p) const {
        return abs(x - p.x) < EPS && abs(y - p.y) < EPS;
    }

    double norm() const { return sqrt(x * x + y * y); }
    double norm2() const { return x * x + y * y; }
};

// 내적
double dot(const Point& a, const Point& b) {
    return a.x * b.x + a.y * b.y;
}

// 외적
double cross(const Point& a, const Point& b) {
    return a.x * b.y - a.y * b.x;
}

// 두 점 사이 거리
double dist(const Point& a, const Point& b) {
    return (a - b).norm();
}

// =============================================================================
// 2. CCW (Counter-Clockwise)
// =============================================================================

// 양수: 반시계, 음수: 시계, 0: 일직선
double ccw(const Point& a, const Point& b, const Point& c) {
    return cross(b - a, c - a);
}

int ccwSign(const Point& a, const Point& b, const Point& c) {
    double v = ccw(a, b, c);
    if (v > EPS) return 1;   // 반시계
    if (v < -EPS) return -1; // 시계
    return 0;                 // 일직선
}

// =============================================================================
// 3. 볼록 껍질 (Convex Hull)
// =============================================================================

vector<Point> convexHull(vector<Point> points) {
    int n = points.size();
    if (n < 3) return points;

    sort(points.begin(), points.end());

    vector<Point> hull;

    // 하단 껍질
    for (int i = 0; i < n; i++) {
        while (hull.size() >= 2 &&
               ccw(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    // 상단 껍질
    int lower = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while (hull.size() > lower &&
               ccw(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    hull.pop_back();  // 마지막 점(시작점과 동일) 제거
    return hull;
}

// =============================================================================
// 4. 선분 교차 판정
// =============================================================================

bool onSegment(const Point& p, const Point& q, const Point& r) {
    return min(p.x, r.x) <= q.x && q.x <= max(p.x, r.x) &&
           min(p.y, r.y) <= q.y && q.y <= max(p.y, r.y);
}

bool segmentIntersect(const Point& a, const Point& b,
                      const Point& c, const Point& d) {
    int d1 = ccwSign(c, d, a);
    int d2 = ccwSign(c, d, b);
    int d3 = ccwSign(a, b, c);
    int d4 = ccwSign(a, b, d);

    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true;
    }

    if (d1 == 0 && onSegment(c, a, d)) return true;
    if (d2 == 0 && onSegment(c, b, d)) return true;
    if (d3 == 0 && onSegment(a, c, b)) return true;
    if (d4 == 0 && onSegment(a, d, b)) return true;

    return false;
}

// =============================================================================
// 5. 직선 교차점
// =============================================================================

// ax + by + c = 0 형태의 직선
struct Line {
    double a, b, c;

    Line(const Point& p1, const Point& p2) {
        a = p2.y - p1.y;
        b = p1.x - p2.x;
        c = -a * p1.x - b * p1.y;
    }
};

bool lineIntersection(const Line& l1, const Line& l2, Point& intersection) {
    double det = l1.a * l2.b - l2.a * l1.b;
    if (abs(det) < EPS) return false;  // 평행

    intersection.x = (l1.b * l2.c - l2.b * l1.c) / det;
    intersection.y = (l2.a * l1.c - l1.a * l2.c) / det;
    return true;
}

// =============================================================================
// 6. 점과 다각형
// =============================================================================

// 점이 다각형 내부에 있는지 (Ray Casting)
bool pointInPolygon(const Point& p, const vector<Point>& polygon) {
    int n = polygon.size();
    int count = 0;

    for (int i = 0; i < n; i++) {
        Point a = polygon[i];
        Point b = polygon[(i + 1) % n];

        if ((a.y <= p.y && p.y < b.y) || (b.y <= p.y && p.y < a.y)) {
            double x = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (p.x < x) count++;
        }
    }

    return count % 2 == 1;
}

// =============================================================================
// 7. 다각형 넓이
// =============================================================================

double polygonArea(const vector<Point>& polygon) {
    int n = polygon.size();
    double area = 0;

    for (int i = 0; i < n; i++) {
        area += cross(polygon[i], polygon[(i + 1) % n]);
    }

    return abs(area) / 2.0;
}

// =============================================================================
// 8. 최근접 점 쌍
// =============================================================================

double closestPair(vector<Point>& points, int lo, int hi) {
    if (hi - lo <= 3) {
        double minDist = 1e18;
        for (int i = lo; i < hi; i++) {
            for (int j = i + 1; j < hi; j++) {
                minDist = min(minDist, dist(points[i], points[j]));
            }
        }
        sort(points.begin() + lo, points.begin() + hi,
             [](const Point& a, const Point& b) { return a.y < b.y; });
        return minDist;
    }

    int mid = (lo + hi) / 2;
    double midX = points[mid].x;

    double d = min(closestPair(points, lo, mid),
                   closestPair(points, mid, hi));

    // 병합 (y 좌표 기준)
    vector<Point> merged(hi - lo);
    merge(points.begin() + lo, points.begin() + mid,
          points.begin() + mid, points.begin() + hi,
          merged.begin(),
          [](const Point& a, const Point& b) { return a.y < b.y; });
    copy(merged.begin(), merged.end(), points.begin() + lo);

    // 스트립 내 점들 확인
    vector<Point> strip;
    for (int i = lo; i < hi; i++) {
        if (abs(points[i].x - midX) < d) {
            strip.push_back(points[i]);
        }
    }

    for (int i = 0; i < (int)strip.size(); i++) {
        for (int j = i + 1; j < (int)strip.size() &&
             strip[j].y - strip[i].y < d; j++) {
            d = min(d, dist(strip[i], strip[j]));
        }
    }

    return d;
}

double closestPairDistance(vector<Point> points) {
    sort(points.begin(), points.end());
    return closestPair(points, 0, points.size());
}

// =============================================================================
// 9. 회전 캘리퍼스
// =============================================================================

double convexDiameter(vector<Point>& hull) {
    int n = hull.size();
    if (n == 1) return 0;
    if (n == 2) return dist(hull[0], hull[1]);

    int j = 1;
    double maxDist = 0;

    for (int i = 0; i < n; i++) {
        while (true) {
            int next = (j + 1) % n;
            Point v1 = hull[(i + 1) % n] - hull[i];
            Point v2 = hull[next] - hull[j];

            if (cross(v1, v2) > 0) {
                j = next;
            } else {
                break;
            }
        }
        maxDist = max(maxDist, dist(hull[i], hull[j]));
    }

    return maxDist;
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "기하 알고리즘 예제" << endl;
    cout << "============================================================" << endl;

    // 1. CCW
    cout << "\n[1] CCW (Counter-Clockwise)" << endl;
    Point a(0, 0), b(1, 1), c(2, 0);
    cout << "    A(0,0), B(1,1), C(2,0)" << endl;
    int ccwResult = ccwSign(a, b, c);
    cout << "    CCW: " << (ccwResult > 0 ? "반시계" : ccwResult < 0 ? "시계" : "일직선") << endl;

    // 2. 볼록 껍질
    cout << "\n[2] 볼록 껍질" << endl;
    vector<Point> points = {{0, 0}, {1, 1}, {2, 2}, {0, 2}, {2, 0}, {1, 3}};
    auto hull = convexHull(points);
    cout << "    입력 점: (0,0), (1,1), (2,2), (0,2), (2,0), (1,3)" << endl;
    cout << "    볼록 껍질: ";
    for (auto& p : hull) {
        cout << "(" << p.x << "," << p.y << ") ";
    }
    cout << endl;

    // 3. 선분 교차
    cout << "\n[3] 선분 교차" << endl;
    Point p1(0, 0), p2(2, 2), p3(0, 2), p4(2, 0);
    cout << "    선분1: (0,0)-(2,2), 선분2: (0,2)-(2,0)" << endl;
    cout << "    교차: " << (segmentIntersect(p1, p2, p3, p4) ? "예" : "아니오") << endl;

    // 4. 다각형 넓이
    cout << "\n[4] 다각형 넓이" << endl;
    vector<Point> polygon = {{0, 0}, {4, 0}, {4, 3}, {0, 3}};
    cout << "    사각형 (0,0)-(4,0)-(4,3)-(0,3)" << endl;
    cout << "    넓이: " << polygonArea(polygon) << endl;

    // 5. 점의 포함 판정
    cout << "\n[5] 점의 다각형 포함 판정" << endl;
    Point inside(2, 1), outside(5, 5);
    cout << "    사각형: (0,0)-(4,0)-(4,3)-(0,3)" << endl;
    cout << "    (2,1) 포함: " << (pointInPolygon(inside, polygon) ? "예" : "아니오") << endl;
    cout << "    (5,5) 포함: " << (pointInPolygon(outside, polygon) ? "예" : "아니오") << endl;

    // 6. 최근접 점 쌍
    cout << "\n[6] 최근접 점 쌍" << endl;
    vector<Point> pts = {{0, 0}, {3, 4}, {1, 1}, {5, 5}, {2, 1}};
    cout << "    점: (0,0), (3,4), (1,1), (5,5), (2,1)" << endl;
    cout << "    최근접 거리: " << closestPairDistance(pts) << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 알고리즘       | 시간복잡도    |" << endl;
    cout << "    |----------------|---------------|" << endl;
    cout << "    | CCW            | O(1)          |" << endl;
    cout << "    | 볼록 껍질      | O(n log n)    |" << endl;
    cout << "    | 선분 교차      | O(1)          |" << endl;
    cout << "    | 다각형 넓이    | O(n)          |" << endl;
    cout << "    | 최근접 점 쌍   | O(n log n)    |" << endl;
    cout << "    | 회전 캘리퍼스  | O(n)          |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
