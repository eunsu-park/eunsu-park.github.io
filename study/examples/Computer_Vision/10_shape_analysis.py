"""
10. 도형 분석
- moments (모멘트)
- boundingRect, minAreaRect, minEnclosingCircle
- convexHull (볼록 껍질)
- matchShapes (도형 매칭)
"""

import cv2
import numpy as np


def create_shapes():
    """다양한 도형 이미지 생성"""
    img = np.zeros((400, 500), dtype=np.uint8)

    # 직사각형
    cv2.rectangle(img, (30, 30), (130, 100), 255, -1)

    # 회전된 사각형
    pts = np.array([[200, 30], [280, 60], [250, 140], [170, 110]], np.int32)
    cv2.fillPoly(img, [pts], 255)

    # 원
    cv2.circle(img, (400, 80), 50, 255, -1)

    # 불규칙한 도형
    pts2 = np.array([[50, 200], [100, 180], [150, 220], [130, 280],
                     [80, 300], [30, 260]], np.int32)
    cv2.fillPoly(img, [pts2], 255)

    # L자 모양
    pts3 = np.array([[200, 180], [280, 180], [280, 220], [240, 220],
                     [240, 320], [200, 320]], np.int32)
    cv2.fillPoly(img, [pts3], 255)

    # 별 모양
    pts_star = []
    for i in range(5):
        outer = np.radians(i * 72 - 90)
        inner = np.radians(i * 72 + 36 - 90)
        pts_star.append([int(400 + 50 * np.cos(outer)), int(250 + 50 * np.sin(outer))])
        pts_star.append([int(400 + 25 * np.cos(inner)), int(250 + 25 * np.sin(inner))])
    cv2.fillPoly(img, [np.array(pts_star, np.int32)], 255)

    return img


def moments_demo():
    """모멘트 데모"""
    print("=" * 50)
    print("모멘트 (moments)")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        # 모멘트 계산
        M = cv2.moments(cnt)

        # 무게중심 (centroid)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # 중심점 표시
            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_img, f'{i}', (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            print(f"\n도형 {i}:")
            print(f"  면적 (m00): {M['m00']:.1f}")
            print(f"  무게중심: ({cx}, {cy})")

            # Hu 모멘트 (불변 모멘트)
            hu = cv2.HuMoments(M)
            print(f"  Hu[0]: {hu[0][0]:.6f}")

    print("\n모멘트 종류:")
    print("  m00: 면적 (0차 모멘트)")
    print("  m10, m01: 1차 모멘트 (무게중심 계산)")
    print("  m20, m02, m11: 2차 모멘트")
    print("  Hu 모멘트: 회전, 크기 불변")

    cv2.imwrite('moments_centroids.jpg', color_img)


def bounding_shapes_demo():
    """경계 도형 데모"""
    print("\n" + "=" * 50)
    print("경계 도형")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 각 경계 도형 종류별 이미지
    bound_rect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    min_rect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    min_circle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    fit_ellipse = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # 1. 바운딩 사각형 (축 정렬)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(bound_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 2. 최소 면적 회전 사각형
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(min_rect, [box], 0, (0, 255, 0), 2)

        # 3. 최소 외접원
        (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(min_circle, (int(x_c), int(y_c)), int(radius), (0, 255, 0), 2)

        # 4. 타원 피팅 (점이 5개 이상 필요)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(fit_ellipse, ellipse, (0, 255, 0), 2)

    print("경계 도형 종류:")
    print("  boundingRect: 축 정렬 사각형")
    print("  minAreaRect: 최소 면적 회전 사각형")
    print("  minEnclosingCircle: 최소 외접원")
    print("  fitEllipse: 타원 피팅")

    cv2.imwrite('bound_rect.jpg', bound_rect)
    cv2.imwrite('min_rect.jpg', min_rect)
    cv2.imwrite('min_circle.jpg', min_circle)
    cv2.imwrite('fit_ellipse.jpg', fit_ellipse)


def convex_hull_demo():
    """볼록 껍질 데모"""
    print("\n" + "=" * 50)
    print("볼록 껍질 (Convex Hull)")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    defects_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # 볼록 껍질
        hull = cv2.convexHull(cnt)
        cv2.drawContours(hull_img, [hull], 0, (0, 255, 0), 2)

        # 볼록성 검사
        is_convex = cv2.isContourConvex(cnt)

        # 볼록 결함 (Convexity Defects)
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(cnt, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    far = tuple(cnt[f][0])
                    # 깊이가 충분히 큰 결함만 표시
                    if d > 1000:
                        cv2.circle(defects_img, far, 5, (0, 0, 255), -1)

        # 면적 비교
        contour_area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print(f"도형 at ({cx}, {cy}): 볼록={is_convex}, Solidity={solidity:.2f}")

    print("\n볼록 껍질 용도:")
    print("  - 도형 단순화")
    print("  - Solidity = 면적/볼록껍질면적 (채움 정도)")
    print("  - 손 제스처 인식 (결함점이 손가락 사이)")

    cv2.imwrite('convex_hull.jpg', hull_img)
    cv2.imwrite('convex_defects.jpg', defects_img)


def match_shapes_demo():
    """도형 매칭 데모"""
    print("\n" + "=" * 50)
    print("도형 매칭 (matchShapes)")
    print("=" * 50)

    # 기준 도형
    template = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(template, (100, 100), 50, 255, -1)

    # 비교 도형들
    shapes = []

    # 원 (비슷함)
    shape1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(shape1, (100, 100), 60, 255, -1)
    shapes.append(('Circle (larger)', shape1))

    # 타원 (다름)
    shape2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(shape2, (100, 100), (60, 40), 0, 0, 360, 255, -1)
    shapes.append(('Ellipse', shape2))

    # 사각형 (많이 다름)
    shape3 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(shape3, (40, 40), (160, 160), 255, -1)
    shapes.append(('Square', shape3))

    # 삼각형 (많이 다름)
    shape4 = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array([[100, 30], [30, 170], [170, 170]], np.int32)
    cv2.fillPoly(shape4, [pts], 255)
    shapes.append(('Triangle', shape4))

    # 템플릿 윤곽선
    cnt_template, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("템플릿: 원")
    print("매칭 결과 (낮을수록 유사):\n")

    for name, shape in shapes:
        cnt_shape, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hu 모멘트 기반 매칭
        match1 = cv2.matchShapes(cnt_template[0], cnt_shape[0], cv2.CONTOURS_MATCH_I1, 0)
        match2 = cv2.matchShapes(cnt_template[0], cnt_shape[0], cv2.CONTOURS_MATCH_I2, 0)
        match3 = cv2.matchShapes(cnt_template[0], cnt_shape[0], cv2.CONTOURS_MATCH_I3, 0)

        print(f"  {name:15}: I1={match1:.4f}, I2={match2:.4f}, I3={match3:.4f}")

    print("\n매칭 방법:")
    print("  CONTOURS_MATCH_I1: ∑|1/huA - 1/huB|")
    print("  CONTOURS_MATCH_I2: ∑|huA - huB|")
    print("  CONTOURS_MATCH_I3: max(|huA - huB|/|huA|)")

    cv2.imwrite('match_template.jpg', template)


def extreme_points_demo():
    """극단점 데모"""
    print("\n" + "=" * 50)
    print("극단점 (Extreme Points)")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # 극단점 찾기
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        # 표시
        cv2.circle(color_img, leftmost, 5, (255, 0, 0), -1)    # 파랑: 왼쪽
        cv2.circle(color_img, rightmost, 5, (0, 255, 0), -1)   # 초록: 오른쪽
        cv2.circle(color_img, topmost, 5, (0, 0, 255), -1)     # 빨강: 위
        cv2.circle(color_img, bottommost, 5, (255, 255, 0), -1) # 청록: 아래

    print("극단점:")
    print("  - 가장 왼쪽, 오른쪽, 위, 아래 점")
    print("  - 손 검출에서 손가락 끝 찾기에 활용")

    cv2.imwrite('extreme_points.jpg', color_img)


def shape_descriptors_demo():
    """도형 기술자 데모"""
    print("\n" + "=" * 50)
    print("도형 기술자 (Shape Descriptors)")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        # 기술자 계산
        aspect_ratio = float(w) / h
        extent = area / (w * h)  # 경계 사각형 대비 면적
        solidity = area / hull_area if hull_area > 0 else 0  # 볼록 껍질 대비 면적
        equiv_diameter = np.sqrt(4 * area / np.pi)  # 등가 직경
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        print(f"\n도형 {i}:")
        print(f"  Aspect Ratio (가로/세로): {aspect_ratio:.2f}")
        print(f"  Extent (면적/경계면적): {extent:.2f}")
        print(f"  Solidity (면적/볼록면적): {solidity:.2f}")
        print(f"  Equivalent Diameter: {equiv_diameter:.1f}")
        print(f"  Circularity (원형도): {circularity:.2f}")


def main():
    """메인 함수"""
    # 모멘트
    moments_demo()

    # 경계 도형
    bounding_shapes_demo()

    # 볼록 껍질
    convex_hull_demo()

    # 도형 매칭
    match_shapes_demo()

    # 극단점
    extreme_points_demo()

    # 도형 기술자
    shape_descriptors_demo()

    print("\n도형 분석 데모 완료!")


if __name__ == '__main__':
    main()
