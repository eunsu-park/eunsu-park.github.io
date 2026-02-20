"""
11. 허프 변환
- HoughLines (표준 허프 직선)
- HoughLinesP (확률적 허프 직선)
- HoughCircles (허프 원)
"""

import cv2
import numpy as np


def create_line_image():
    """직선이 있는 테스트 이미지"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # 다양한 직선
    cv2.line(img, (50, 50), (450, 50), (0, 0, 0), 2)      # 수평선
    cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 2)      # 수직선
    cv2.line(img, (100, 100), (400, 300), (0, 0, 0), 2)   # 대각선
    cv2.line(img, (100, 300), (400, 100), (0, 0, 0), 2)   # 대각선
    cv2.line(img, (250, 150), (250, 350), (0, 0, 0), 2)   # 수직선

    return img


def create_circle_image():
    """원이 있는 테스트 이미지"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # 다양한 원
    cv2.circle(img, (100, 100), 40, (0, 0, 0), 2)
    cv2.circle(img, (300, 100), 50, (0, 0, 0), 2)
    cv2.circle(img, (150, 250), 60, (0, 0, 0), 2)
    cv2.circle(img, (350, 280), 45, (0, 0, 0), 2)

    # 채워진 원
    cv2.circle(img, (450, 350), 30, (0, 0, 0), -1)

    return img


def hough_lines_demo():
    """표준 허프 직선 변환 데모"""
    print("=" * 50)
    print("표준 허프 직선 변환 (HoughLines)")
    print("=" * 50)

    img = create_line_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150)

    # 허프 직선 변환
    # rho: 거리 해상도 (픽셀)
    # theta: 각도 해상도 (라디안)
    # threshold: 최소 투표 수
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    result = img.copy()

    if lines is not None:
        print(f"검출된 직선 수: {len(lines)}")

        for line in lines:
            rho, theta = line[0]

            # 극좌표 → 직교좌표 변환
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # 직선 그리기 (충분히 긴 선분)
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print("\nHoughLines 파라미터:")
    print("  rho: 거리 해상도 (보통 1 픽셀)")
    print("  theta: 각도 해상도 (보통 π/180)")
    print("  threshold: 직선으로 판단할 최소 투표 수")

    cv2.imwrite('hough_lines_input.jpg', img)
    cv2.imwrite('hough_lines_edges.jpg', edges)
    cv2.imwrite('hough_lines_result.jpg', result)


def hough_lines_p_demo():
    """확률적 허프 직선 변환 데모"""
    print("\n" + "=" * 50)
    print("확률적 허프 직선 변환 (HoughLinesP)")
    print("=" * 50)

    img = create_line_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 확률적 허프 변환
    # minLineLength: 최소 직선 길이
    # maxLineGap: 직선으로 간주할 최대 간격
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                            minLineLength=50, maxLineGap=10)

    result = img.copy()

    if lines is not None:
        print(f"검출된 선분 수: {len(lines)}")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(result, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(result, (x2, y2), 5, (0, 0, 255), -1)

    print("\nHoughLinesP 장점:")
    print("  - 선분의 시작점, 끝점 반환")
    print("  - 표준 방법보다 빠름")
    print("  - 파라미터 조정 용이")

    cv2.imwrite('hough_linesp_result.jpg', result)


def hough_lines_params_demo():
    """허프 직선 파라미터 영향"""
    print("\n" + "=" * 50)
    print("허프 직선 파라미터 영향")
    print("=" * 50)

    img = create_line_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 다양한 threshold 값
    thresholds = [30, 50, 100, 150]

    for thresh in thresholds:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, thresh,
                                minLineLength=30, maxLineGap=10)
        result = img.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            print(f"threshold={thresh}: {len(lines)} 선분 검출")

        cv2.imwrite(f'hough_thresh_{thresh}.jpg', result)


def hough_circles_demo():
    """허프 원 변환 데모"""
    print("\n" + "=" * 50)
    print("허프 원 변환 (HoughCircles)")
    print("=" * 50)

    img = create_circle_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 블러 적용 (노이즈 감소)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 허프 원 변환
    # cv2.HOUGH_GRADIENT: 허프 그래디언트 방법
    # dp: 누적 배열 해상도 비율 (1 = 입력과 동일)
    # minDist: 원 중심 간 최소 거리
    # param1: Canny 엣지 검출 상위 임계값
    # param2: 원 검출 임계값 (낮을수록 더 많은 원 검출)
    # minRadius, maxRadius: 원 반지름 범위

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"검출된 원 수: {len(circles[0])}")

        for circle in circles[0, :]:
            cx, cy, r = circle

            # 원 그리기
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
            # 중심점 표시
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

            print(f"  원: 중심=({cx}, {cy}), 반지름={r}")

    print("\nHoughCircles 파라미터:")
    print("  dp: 누적 배열 해상도 (1 권장)")
    print("  minDist: 원 중심 간 최소 거리")
    print("  param1: Canny 상위 임계값")
    print("  param2: 원 검출 임계값 (낮으면 많이 검출)")
    print("  minRadius/maxRadius: 반지름 범위")

    cv2.imwrite('hough_circles_input.jpg', img)
    cv2.imwrite('hough_circles_result.jpg', result)


def hough_circles_params_demo():
    """허프 원 파라미터 영향"""
    print("\n" + "=" * 50)
    print("허프 원 파라미터 영향")
    print("=" * 50)

    img = create_circle_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # param2 값 변화
    param2_values = [20, 30, 40, 50]

    for p2 in param2_values:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 50,
            param1=100, param2=p2, minRadius=20, maxRadius=100
        )

        result = img.copy()
        count = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))
            count = len(circles[0])

            for circle in circles[0, :]:
                cx, cy, r = circle
                cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

        print(f"param2={p2}: {count} 원 검출")
        cv2.imwrite(f'hough_circles_p2_{p2}.jpg', result)


def practical_lane_detection():
    """실용 예제: 차선 검출 시뮬레이션"""
    print("\n" + "=" * 50)
    print("실용 예제: 차선 검출")
    print("=" * 50)

    # 도로 이미지 시뮬레이션
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = [100, 100, 100]  # 회색 도로

    # 차선 그리기
    cv2.line(img, (100, 400), (250, 200), (255, 255, 255), 5)  # 왼쪽 차선
    cv2.line(img, (500, 400), (350, 200), (255, 255, 255), 5)  # 오른쪽 차선
    cv2.line(img, (300, 400), (300, 250), (255, 255, 0), 3)    # 중앙선 (점선 역할)

    # 그레이스케일 및 엣지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # ROI (Region of Interest) 마스크
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(50, 400), (550, 400), (350, 180), (250, 180)]], np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 허프 직선 검출
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30,
                            minLineLength=50, maxLineGap=100)

    result = img.copy()

    if lines is not None:
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 기울기 계산
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)

                # 기울기로 좌/우 차선 분류
                if slope < -0.3:  # 왼쪽 차선
                    left_lines.append(line[0])
                elif slope > 0.3:  # 오른쪽 차선
                    right_lines.append(line[0])

        # 차선 그리기
        for line in left_lines:
            cv2.line(result, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)
        for line in right_lines:
            cv2.line(result, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)

        print(f"왼쪽 차선: {len(left_lines)}, 오른쪽 차선: {len(right_lines)}")

    cv2.imwrite('lane_input.jpg', img)
    cv2.imwrite('lane_edges.jpg', masked_edges)
    cv2.imwrite('lane_result.jpg', result)
    print("차선 검출 이미지 저장 완료")


def main():
    """메인 함수"""
    # 표준 허프 직선
    hough_lines_demo()

    # 확률적 허프 직선
    hough_lines_p_demo()

    # 직선 파라미터
    hough_lines_params_demo()

    # 허프 원
    hough_circles_demo()

    # 원 파라미터
    hough_circles_params_demo()

    # 실용 예제
    practical_lane_detection()

    print("\n허프 변환 데모 완료!")


if __name__ == '__main__':
    main()
