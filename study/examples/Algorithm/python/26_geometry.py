"""
기하 알고리즘 (Computational Geometry)
Computational Geometry Algorithms

점, 선, 다각형 등의 기하학적 객체를 다루는 알고리즘입니다.
"""

from typing import List, Tuple, Optional
from math import sqrt, atan2, pi, inf
from functools import cmp_to_key


# =============================================================================
# 1. 기본 기하 연산
# =============================================================================

class Point:
    """2D 점"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)

    def __eq__(self, other: 'Point') -> bool:
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def dot(self, other: 'Point') -> float:
        """내적"""
        return self.x * other.x + self.y * other.y

    def cross(self, other: 'Point') -> float:
        """외적 (z 성분)"""
        return self.x * other.y - self.y * other.x

    def norm(self) -> float:
        """벡터 크기"""
        return sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> 'Point':
        """단위 벡터"""
        n = self.norm()
        return Point(self.x / n, self.y / n) if n > 0 else Point(0, 0)


def distance(p1: Point, p2: Point) -> float:
    """두 점 사이 거리"""
    return (p2 - p1).norm()


# =============================================================================
# 2. CCW (Counter-Clockwise)
# =============================================================================

def ccw(a: Point, b: Point, c: Point) -> int:
    """
    세 점의 방향 판별
    반환: 1 (반시계), -1 (시계), 0 (일직선)
    """
    cross = (b - a).cross(c - a)
    if cross > 1e-9:
        return 1   # 반시계
    elif cross < -1e-9:
        return -1  # 시계
    return 0       # 일직선


def ccw_tuple(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
    """튜플 버전 CCW"""
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross > 1e-9:
        return 1
    elif cross < -1e-9:
        return -1
    return 0


# =============================================================================
# 3. 선분 교차 판정
# =============================================================================

def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    선분 p1-p2와 p3-p4의 교차 여부
    """
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    # 일반적인 교차
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    # 경계 케이스 (점이 선분 위에 있는 경우)
    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (min(p.x, r.x) <= q.x <= max(p.x, r.x) and
                min(p.y, r.y) <= q.y <= max(p.y, r.y))

    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True

    return False


def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """
    두 직선의 교점 계산
    직선 p1-p2와 직선 p3-p4
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1.cross(d2)

    if abs(cross) < 1e-9:
        return None  # 평행

    t = (p3 - p1).cross(d2) / cross
    return p1 + d1 * t


# =============================================================================
# 4. 볼록 껍질 (Convex Hull)
# =============================================================================

def convex_hull_graham(points: List[Point]) -> List[Point]:
    """
    Graham Scan 알고리즘
    시간복잡도: O(n log n)
    """
    if len(points) < 3:
        return points[:]

    # 가장 아래, 왼쪽 점 찾기
    start = min(points, key=lambda p: (p.y, p.x))

    # 각도로 정렬
    def polar_angle(p: Point) -> float:
        return atan2(p.y - start.y, p.x - start.x)

    def dist_sq(p: Point) -> float:
        return (p.x - start.x) ** 2 + (p.y - start.y) ** 2

    sorted_points = sorted(points, key=lambda p: (polar_angle(p), dist_sq(p)))

    # 스택으로 볼록 껍질 구성
    hull = []
    for p in sorted_points:
        while len(hull) >= 2 and ccw(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


def convex_hull_monotone_chain(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Monotone Chain 알고리즘
    시간복잡도: O(n log n)
    """
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    # 하부 껍질
    lower = []
    for p in points:
        while len(lower) >= 2 and ccw_tuple(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # 상부 껍질
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and ccw_tuple(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# =============================================================================
# 5. 다각형 연산
# =============================================================================

def polygon_area(vertices: List[Point]) -> float:
    """
    다각형 넓이 (신발끈 공식)
    정점이 반시계 방향으로 정렬되어 있어야 함
    """
    n = len(vertices)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i].cross(vertices[j])

    return abs(area) / 2


def polygon_perimeter(vertices: List[Point]) -> float:
    """다각형 둘레"""
    n = len(vertices)
    perimeter = 0
    for i in range(n):
        j = (i + 1) % n
        perimeter += distance(vertices[i], vertices[j])
    return perimeter


def point_in_polygon(point: Point, polygon: List[Point]) -> int:
    """
    점이 다각형 내부에 있는지 판정
    반환: 1 (내부), 0 (경계), -1 (외부)
    """
    n = len(polygon)
    winding = 0

    for i in range(n):
        j = (i + 1) % n
        p1, p2 = polygon[i], polygon[j]

        # 경계 위의 점 검사
        if ccw(p1, p2, point) == 0:
            if (min(p1.x, p2.x) <= point.x <= max(p1.x, p2.x) and
                min(p1.y, p2.y) <= point.y <= max(p1.y, p2.y)):
                return 0

        # Winding number 계산
        if p1.y <= point.y:
            if p2.y > point.y and ccw(p1, p2, point) > 0:
                winding += 1
        else:
            if p2.y <= point.y and ccw(p1, p2, point) < 0:
                winding -= 1

    return 1 if winding != 0 else -1


def is_convex(polygon: List[Point]) -> bool:
    """다각형이 볼록인지 검사"""
    n = len(polygon)
    if n < 3:
        return False

    sign = 0
    for i in range(n):
        d = ccw(polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n])
        if d != 0:
            if sign == 0:
                sign = d
            elif sign != d:
                return False

    return True


# =============================================================================
# 6. 점과 직선/선분 거리
# =============================================================================

def point_to_line_distance(point: Point, line_p1: Point, line_p2: Point) -> float:
    """점에서 직선까지의 거리"""
    v = line_p2 - line_p1
    w = point - line_p1
    return abs(v.cross(w)) / v.norm()


def point_to_segment_distance(point: Point, seg_p1: Point, seg_p2: Point) -> float:
    """점에서 선분까지의 거리"""
    v = seg_p2 - seg_p1
    w = point - seg_p1

    c1 = w.dot(v)
    if c1 <= 0:
        return distance(point, seg_p1)

    c2 = v.dot(v)
    if c2 <= c1:
        return distance(point, seg_p2)

    t = c1 / c2
    proj = seg_p1 + v * t
    return distance(point, proj)


# =============================================================================
# 7. 가장 가까운 점 쌍 (Closest Pair)
# =============================================================================

def closest_pair(points: List[Point]) -> Tuple[Point, Point, float]:
    """
    가장 가까운 점 쌍 찾기
    분할 정복: O(n log n)
    """
    def brute_force(pts: List[Point]) -> Tuple[Point, Point, float]:
        min_dist = inf
        p1, p2 = None, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = distance(pts[i], pts[j])
                if d < min_dist:
                    min_dist = d
                    p1, p2 = pts[i], pts[j]
        return p1, p2, min_dist

    def closest_split(pts_y: List[Point], mid_x: float, delta: float) -> Tuple[Point, Point, float]:
        # 중간선 근처의 점들만
        strip = [p for p in pts_y if abs(p.x - mid_x) < delta]

        min_dist = delta
        p1, p2 = None, None

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j].y - strip[i].y < min_dist:
                d = distance(strip[i], strip[j])
                if d < min_dist:
                    min_dist = d
                    p1, p2 = strip[i], strip[j]
                j += 1

        return p1, p2, min_dist

    def divide_conquer(pts_x: List[Point], pts_y: List[Point]) -> Tuple[Point, Point, float]:
        if len(pts_x) <= 3:
            return brute_force(pts_x)

        mid = len(pts_x) // 2
        mid_point = pts_x[mid]

        # 왼쪽/오른쪽 분할
        left_x = pts_x[:mid]
        right_x = pts_x[mid:]

        left_set = set(id(p) for p in left_x)
        left_y = [p for p in pts_y if id(p) in left_set]
        right_y = [p for p in pts_y if id(p) not in left_set]

        # 재귀 호출
        l1, l2, left_dist = divide_conquer(left_x, left_y)
        r1, r2, right_dist = divide_conquer(right_x, right_y)

        if left_dist < right_dist:
            best = (l1, l2, left_dist)
        else:
            best = (r1, r2, right_dist)

        # 분할선 근처 검사
        s1, s2, split_dist = closest_split(pts_y, mid_point.x, best[2])
        if split_dist < best[2]:
            return s1, s2, split_dist

        return best

    if len(points) < 2:
        return None, None, inf

    pts_x = sorted(points, key=lambda p: p.x)
    pts_y = sorted(points, key=lambda p: p.y)

    return divide_conquer(pts_x, pts_y)


# =============================================================================
# 8. 회전하는 캘리퍼스 (Rotating Calipers)
# =============================================================================

def rotating_calipers_diameter(hull: List[Point]) -> Tuple[Point, Point, float]:
    """
    볼록 껍질의 지름 (가장 먼 점 쌍)
    시간복잡도: O(n)
    """
    n = len(hull)
    if n < 2:
        return None, None, 0
    if n == 2:
        return hull[0], hull[1], distance(hull[0], hull[1])

    # 가장 먼 점 쌍 찾기
    max_dist = 0
    p1, p2 = None, None

    j = 1
    for i in range(n):
        # hull[i]-hull[(i+1)%n] 에지에 대해 가장 먼 점 찾기
        while True:
            next_j = (j + 1) % n
            # 삼각형 넓이 비교
            area1 = abs((hull[(i + 1) % n] - hull[i]).cross(hull[j] - hull[i]))
            area2 = abs((hull[(i + 1) % n] - hull[i]).cross(hull[next_j] - hull[i]))
            if area2 > area1:
                j = next_j
            else:
                break

        d1 = distance(hull[i], hull[j])
        d2 = distance(hull[(i + 1) % n], hull[j])

        if d1 > max_dist:
            max_dist = d1
            p1, p2 = hull[i], hull[j]
        if d2 > max_dist:
            max_dist = d2
            p1, p2 = hull[(i + 1) % n], hull[j]

    return p1, p2, max_dist


# =============================================================================
# 9. 반평면 교집합 (Half-Plane Intersection)
# =============================================================================

class HalfPlane:
    """반평면: ax + by + c >= 0"""
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c
        self.angle = atan2(b, a)

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> 'HalfPlane':
        """p1 → p2 방향의 왼쪽 반평면"""
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = -a * p1.x - b * p1.y
        return cls(a, b, c)

    def side(self, p: Point) -> float:
        """점이 어느 쪽에 있는지 (양수: 내부, 음수: 외부)"""
        return self.a * p.x + self.b * p.y + self.c


def half_plane_intersection_point(h1: HalfPlane, h2: HalfPlane) -> Optional[Point]:
    """두 반평면 경계선의 교점"""
    det = h1.a * h2.b - h2.a * h1.b
    if abs(det) < 1e-9:
        return None
    x = (h1.b * h2.c - h2.b * h1.c) / det
    y = (h2.a * h1.c - h1.a * h2.c) / det
    return Point(x, y)


# =============================================================================
# 10. 실전 문제: 삼각형 넓이
# =============================================================================

def triangle_area(p1: Point, p2: Point, p3: Point) -> float:
    """삼각형 넓이"""
    return abs((p2 - p1).cross(p3 - p1)) / 2


def triangle_circumcircle(p1: Point, p2: Point, p3: Point) -> Tuple[Point, float]:
    """삼각형의 외접원 (중심, 반지름)"""
    ax, ay = p1.x, p1.y
    bx, by = p2.x, p2.y
    cx, cy = p3.x, p3.y

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None, 0

    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

    center = Point(ux, uy)
    radius = distance(center, p1)

    return center, radius


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("기하 알고리즘 (Computational Geometry) 예제")
    print("=" * 60)

    # 1. CCW
    print("\n[1] CCW (방향 판별)")
    a, b, c = Point(0, 0), Point(4, 0), Point(2, 2)
    d = Point(2, -2)
    print(f"    A={a}, B={b}, C={c}")
    print(f"    CCW(A,B,C) = {ccw(a, b, c)} (반시계)")
    print(f"    CCW(A,B,D) = {ccw(a, b, d)} (시계, D={d})")

    # 2. 선분 교차
    print("\n[2] 선분 교차 판정")
    p1, p2 = Point(0, 0), Point(4, 4)
    p3, p4 = Point(0, 4), Point(4, 0)
    print(f"    선분1: {p1}-{p2}")
    print(f"    선분2: {p3}-{p4}")
    print(f"    교차: {segments_intersect(p1, p2, p3, p4)}")

    intersection = line_intersection(p1, p2, p3, p4)
    print(f"    교점: {intersection}")

    # 3. 볼록 껍질
    print("\n[3] 볼록 껍질 (Convex Hull)")
    points = [Point(0, 0), Point(1, 1), Point(2, 2), Point(0, 2),
              Point(2, 0), Point(1, 0), Point(0, 1), Point(2, 1)]
    hull = convex_hull_graham(points)
    print(f"    점들: {points}")
    print(f"    볼록 껍질: {hull}")

    # 튜플 버전
    pts_tuple = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0), (0, 1), (2, 1)]
    hull_tuple = convex_hull_monotone_chain(pts_tuple)
    print(f"    Monotone Chain: {hull_tuple}")

    # 4. 다각형 연산
    print("\n[4] 다각형 연산")
    polygon = [Point(0, 0), Point(4, 0), Point(4, 3), Point(0, 3)]
    print(f"    다각형: {polygon}")
    print(f"    넓이: {polygon_area(polygon)}")
    print(f"    둘레: {polygon_perimeter(polygon)}")
    print(f"    볼록: {is_convex(polygon)}")

    test_point = Point(2, 1)
    print(f"    점 {test_point} 위치: {point_in_polygon(test_point, polygon)} (1=내부)")

    # 5. 점-선분 거리
    print("\n[5] 점에서 선분까지 거리")
    point = Point(2, 3)
    seg_start = Point(0, 0)
    seg_end = Point(4, 0)
    dist = point_to_segment_distance(point, seg_start, seg_end)
    print(f"    점: {point}")
    print(f"    선분: {seg_start}-{seg_end}")
    print(f"    거리: {dist}")

    # 6. 가장 가까운 점 쌍
    print("\n[6] 가장 가까운 점 쌍")
    rand_points = [Point(2, 3), Point(12, 30), Point(40, 50),
                   Point(5, 1), Point(12, 10), Point(3, 4)]
    p1, p2, dist = closest_pair(rand_points)
    print(f"    점들: {rand_points}")
    print(f"    가장 가까운 쌍: {p1}, {p2}")
    print(f"    거리: {dist:.4f}")

    # 7. 회전 캘리퍼스
    print("\n[7] 볼록 껍질 지름 (회전 캘리퍼스)")
    hull_for_diameter = [Point(0, 0), Point(4, 0), Point(5, 2), Point(3, 4), Point(1, 3)]
    p1, p2, diam = rotating_calipers_diameter(hull_for_diameter)
    print(f"    볼록 껍질: {hull_for_diameter}")
    print(f"    가장 먼 쌍: {p1}, {p2}")
    print(f"    지름: {diam:.4f}")

    # 8. 삼각형
    print("\n[8] 삼각형 연산")
    t1, t2, t3 = Point(0, 0), Point(4, 0), Point(2, 3)
    print(f"    삼각형: {t1}, {t2}, {t3}")
    print(f"    넓이: {triangle_area(t1, t2, t3)}")
    center, radius = triangle_circumcircle(t1, t2, t3)
    print(f"    외접원 중심: {center}")
    print(f"    외접원 반지름: {radius:.4f}")

    # 9. 알고리즘 복잡도
    print("\n[9] 알고리즘 복잡도")
    print("    | 알고리즘          | 시간 복잡도    |")
    print("    |-------------------|----------------|")
    print("    | CCW               | O(1)           |")
    print("    | 선분 교차         | O(1)           |")
    print("    | 볼록 껍질         | O(n log n)     |")
    print("    | 점 in 다각형     | O(n)           |")
    print("    | 가장 가까운 쌍   | O(n log n)     |")
    print("    | 회전 캘리퍼스     | O(n)           |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
