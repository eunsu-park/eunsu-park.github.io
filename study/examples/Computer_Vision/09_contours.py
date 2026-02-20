"""
09. 윤곽선 검출
- findContours, drawContours
- 윤곽선 계층 구조
- 윤곽선 근사 (approxPolyDP)
"""

import cv2
import numpy as np


def create_shapes_image():
    """도형이 있는 이진 이미지"""
    img = np.zeros((400, 500), dtype=np.uint8)

    # 사각형
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

    # 원
    cv2.circle(img, (300, 100), 50, 255, -1)

    # 삼각형
    pts = np.array([[400, 50], [350, 150], [450, 150]], np.int32)
    cv2.fillPoly(img, [pts], 255)

    # 중첩된 사각형 (계층 구조)
    cv2.rectangle(img, (50, 200), (200, 350), 255, -1)
    cv2.rectangle(img, (80, 230), (170, 320), 0, -1)  # 내부 구멍
    cv2.rectangle(img, (100, 250), (150, 290), 255, -1)  # 구멍 안의 객체

    # 별 모양
    pts_star = np.array([
        [350, 200], [365, 250], [420, 250], [375, 280],
        [390, 330], [350, 300], [310, 330], [325, 280],
        [280, 250], [335, 250]
    ], np.int32)
    cv2.fillPoly(img, [pts_star], 255)

    return img


def find_contours_demo():
    """윤곽선 찾기 데모"""
    print("=" * 50)
    print("윤곽선 찾기 (findContours)")
    print("=" * 50)

    img = create_shapes_image()

    # 윤곽선 찾기
    # RETR_EXTERNAL: 외부 윤곽선만
    # RETR_LIST: 모든 윤곽선 (계층 무시)
    # RETR_TREE: 모든 윤곽선 (계층 포함)
    # RETR_CCOMP: 2단계 계층

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"검출된 윤곽선 수: {len(contours)}")
    print(f"계층 구조 shape: {hierarchy.shape if hierarchy is not None else None}")

    # 검색 모드 설명
    print("\n검색 모드 (Retrieval Mode):")
    print("  RETR_EXTERNAL: 가장 바깥 윤곽선만")
    print("  RETR_LIST: 모든 윤곽선 (평평하게)")
    print("  RETR_TREE: 전체 계층 구조")
    print("  RETR_CCOMP: 2단계 계층")

    return img, contours, hierarchy


def draw_contours_demo():
    """윤곽선 그리기 데모"""
    print("\n" + "=" * 50)
    print("윤곽선 그리기 (drawContours)")
    print("=" * 50)

    img = create_shapes_image()
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컬러 이미지로 변환 (그리기용)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 모든 윤곽선 그리기
    all_contours = color_img.copy()
    cv2.drawContours(all_contours, contours, -1, (0, 255, 0), 2)

    # 특정 윤곽선만 그리기
    specific = color_img.copy()
    cv2.drawContours(specific, contours, 0, (255, 0, 0), 2)  # 첫 번째 윤곽선
    cv2.drawContours(specific, contours, 1, (0, 255, 0), 2)  # 두 번째 윤곽선

    # 채우기
    filled = color_img.copy()
    cv2.drawContours(filled, contours, 0, (0, 0, 255), -1)  # thickness=-1 → 채우기

    print("drawContours 파라미터:")
    print("  contourIdx=-1: 모든 윤곽선")
    print("  contourIdx=n: n번째 윤곽선만")
    print("  thickness=-1: 내부 채우기")

    cv2.imwrite('contours_original.jpg', img)
    cv2.imwrite('contours_all.jpg', all_contours)
    cv2.imwrite('contours_specific.jpg', specific)
    cv2.imwrite('contours_filled.jpg', filled)


def contour_hierarchy_demo():
    """윤곽선 계층 구조 데모"""
    print("\n" + "=" * 50)
    print("윤곽선 계층 구조")
    print("=" * 50)

    img = create_shapes_image()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # hierarchy[i] = [Next, Previous, First_Child, Parent]
    print("계층 구조 형식: [Next, Previous, First_Child, Parent]")
    print("(-1은 해당 관계 없음)")

    for i, h in enumerate(hierarchy[0]):
        print(f"윤곽선 {i}: Next={h[0]}, Prev={h[1]}, Child={h[2]}, Parent={h[3]}")

    # 외부 윤곽선만 (부모가 없는 것)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    external_only = color_img.copy()

    for i, h in enumerate(hierarchy[0]):
        if h[3] == -1:  # 부모가 없음 = 외부 윤곽선
            cv2.drawContours(external_only, contours, i, (0, 255, 0), 2)

    cv2.imwrite('contours_external.jpg', external_only)


def contour_approximation_demo():
    """윤곽선 근사 데모"""
    print("\n" + "=" * 50)
    print("윤곽선 근사 (approxPolyDP)")
    print("=" * 50)

    # 복잡한 곡선 이미지
    img = np.zeros((300, 400), dtype=np.uint8)
    pts = []
    for angle in range(0, 360, 5):
        r = 80 + 20 * np.sin(5 * np.radians(angle))
        x = int(200 + r * np.cos(np.radians(angle)))
        y = int(150 + r * np.sin(np.radians(angle)))
        pts.append([x, y])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], 255)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 원본 윤곽선
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 다양한 epsilon 값으로 근사
    epsilons = [0.01, 0.02, 0.05, 0.1]

    for eps in epsilons:
        approx_img = color_img.copy()
        for cnt in contours:
            # epsilon = 둘레의 비율
            epsilon = eps * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            print(f"epsilon={eps}: {len(cnt)} 점 → {len(approx)} 점")

            cv2.drawContours(approx_img, [approx], -1, (0, 255, 0), 2)
            # 꼭짓점 표시
            for pt in approx:
                cv2.circle(approx_img, tuple(pt[0]), 3, (0, 0, 255), -1)

        cv2.imwrite(f'approx_eps_{eps}.jpg', approx_img)


def contour_properties_demo():
    """윤곽선 속성 데모"""
    print("\n" + "=" * 50)
    print("윤곽선 속성")
    print("=" * 50)

    img = create_shapes_image()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        # 면적
        area = cv2.contourArea(cnt)

        # 둘레 (호 길이)
        perimeter = cv2.arcLength(cnt, True)

        # 경계 사각형
        x, y, w, h = cv2.boundingRect(cnt)

        # 중심점 (모멘트)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        print(f"\n윤곽선 {i}:")
        print(f"  면적: {area:.1f}")
        print(f"  둘레: {perimeter:.1f}")
        print(f"  경계 사각형: ({x}, {y}, {w}, {h})")
        print(f"  중심: ({cx}, {cy})")


def detect_shapes():
    """도형 인식"""
    print("\n" + "=" * 50)
    print("도형 인식")
    print("=" * 50)

    img = create_shapes_image()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # 윤곽선 근사
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 중심점 계산
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            continue

        # 꼭짓점 수로 도형 판별
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            # 정사각형 vs 직사각형
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices > 5:
            # 원 판별 (면적/둘레 비율)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            shape = "Circle" if circularity > 0.8 else f"Polygon({vertices})"
        else:
            shape = f"Polygon({vertices})"

        # 결과 표시
        cv2.drawContours(color_img, [approx], -1, (0, 255, 0), 2)
        cv2.putText(color_img, shape, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"  {shape}: 꼭짓점 {vertices}개")

    cv2.imwrite('shapes_detected.jpg', color_img)


def main():
    """메인 함수"""
    # 윤곽선 찾기
    find_contours_demo()

    # 윤곽선 그리기
    draw_contours_demo()

    # 계층 구조
    contour_hierarchy_demo()

    # 윤곽선 근사
    contour_approximation_demo()

    # 윤곽선 속성
    contour_properties_demo()

    # 도형 인식
    detect_shapes()

    print("\n윤곽선 검출 데모 완료!")


if __name__ == '__main__':
    main()
