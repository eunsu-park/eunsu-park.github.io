"""
08. 엣지 검출
- Sobel, Scharr 필터
- Laplacian
- Canny 엣지 검출
"""

import cv2
import numpy as np


def create_test_image():
    """테스트 이미지 생성"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 200

    # 사각형
    cv2.rectangle(img, (50, 50), (150, 150), 50, -1)

    # 원
    cv2.circle(img, (300, 150), 60, 80, -1)

    # 삼각형
    pts = np.array([[200, 250], [150, 290], [250, 290]], np.int32)
    cv2.fillPoly(img, [pts], 100)

    # 텍스트
    cv2.putText(img, 'EDGE', (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 30, 2)

    return img


def sobel_demo():
    """Sobel 필터 데모"""
    print("=" * 50)
    print("Sobel 필터")
    print("=" * 50)

    img = create_test_image()

    # Sobel 필터 (x 방향)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    # Sobel 필터 (y 방향)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # 절대값 및 8비트 변환
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # x, y 합성
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

    # 크기 계산 (정확한 방법)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

    print("Sobel 필터:")
    print("  - 1차 미분 기반 엣지 검출")
    print("  - x 방향: 수직 엣지 검출")
    print("  - y 방향: 수평 엣지 검출")
    print("  - ksize: 커널 크기 (3, 5, 7)")

    cv2.imwrite('edge_original.jpg', img)
    cv2.imwrite('sobel_x.jpg', sobel_x_abs)
    cv2.imwrite('sobel_y.jpg', sobel_y_abs)
    cv2.imwrite('sobel_combined.jpg', sobel_combined)
    cv2.imwrite('sobel_magnitude.jpg', sobel_magnitude)


def scharr_demo():
    """Scharr 필터 데모"""
    print("\n" + "=" * 50)
    print("Scharr 필터")
    print("=" * 50)

    img = create_test_image()

    # Scharr 필터 (Sobel보다 정확)
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    scharr_x_abs = cv2.convertScaleAbs(scharr_x)
    scharr_y_abs = cv2.convertScaleAbs(scharr_y)

    scharr_combined = cv2.addWeighted(scharr_x_abs, 0.5, scharr_y_abs, 0.5, 0)

    print("Scharr 필터:")
    print("  - Sobel의 개선 버전")
    print("  - 3x3 커널만 지원")
    print("  - 더 정확한 그래디언트 계산")
    print("  - Sobel(ksize=-1)과 동일")

    cv2.imwrite('scharr_x.jpg', scharr_x_abs)
    cv2.imwrite('scharr_y.jpg', scharr_y_abs)
    cv2.imwrite('scharr_combined.jpg', scharr_combined)


def laplacian_demo():
    """Laplacian 필터 데모"""
    print("\n" + "=" * 50)
    print("Laplacian 필터")
    print("=" * 50)

    img = create_test_image()

    # 노이즈에 민감하므로 블러 적용
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Laplacian 필터
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # 커널 크기 변화
    lap_k1 = cv2.Laplacian(blurred, cv2.CV_64F, ksize=1)
    lap_k3 = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    lap_k5 = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)

    print("Laplacian 필터:")
    print("  - 2차 미분 기반 엣지 검출")
    print("  - 모든 방향의 엣지 한 번에 검출")
    print("  - 노이즈에 민감 → 블러 필요")

    cv2.imwrite('laplacian.jpg', laplacian_abs)
    cv2.imwrite('laplacian_k1.jpg', cv2.convertScaleAbs(lap_k1))
    cv2.imwrite('laplacian_k3.jpg', cv2.convertScaleAbs(lap_k3))
    cv2.imwrite('laplacian_k5.jpg', cv2.convertScaleAbs(lap_k5))


def canny_demo():
    """Canny 엣지 검출 데모"""
    print("\n" + "=" * 50)
    print("Canny 엣지 검출")
    print("=" * 50)

    img = create_test_image()

    # Canny 엣지 검출
    # threshold1: 낮은 임계값
    # threshold2: 높은 임계값
    canny_50_150 = cv2.Canny(img, 50, 150)
    canny_100_200 = cv2.Canny(img, 100, 200)
    canny_30_100 = cv2.Canny(img, 30, 100)

    print("Canny 엣지 검출 단계:")
    print("  1. 가우시안 필터로 노이즈 제거")
    print("  2. Sobel로 그래디언트 계산")
    print("  3. 비최대 억제 (Non-Maximum Suppression)")
    print("  4. 이중 임계값으로 엣지 결정")
    print("     - 강한 엣지: > threshold2")
    print("     - 약한 엣지: threshold1 ~ threshold2")
    print("     - 비엣지: < threshold1")
    print("  5. 히스테리시스 엣지 추적")

    cv2.imwrite('canny_50_150.jpg', canny_50_150)
    cv2.imwrite('canny_100_200.jpg', canny_100_200)
    cv2.imwrite('canny_30_100.jpg', canny_30_100)


def canny_with_blur():
    """블러와 함께 Canny 사용"""
    print("\n" + "=" * 50)
    print("블러 + Canny")
    print("=" * 50)

    img = create_test_image()

    # 노이즈 추가
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 블러 없이
    canny_no_blur = cv2.Canny(noisy, 50, 150)

    # 가우시안 블러 후
    blurred = cv2.GaussianBlur(noisy, (5, 5), 0)
    canny_with_blur = cv2.Canny(blurred, 50, 150)

    # Canny 내부에서 apertureSize 조정
    canny_aperture3 = cv2.Canny(noisy, 50, 150, apertureSize=3)
    canny_aperture5 = cv2.Canny(noisy, 50, 150, apertureSize=5)

    print("노이즈 제거:")
    print("  - 블러 전처리 권장")
    print("  - apertureSize 조정 가능 (3, 5, 7)")

    cv2.imwrite('canny_noisy.jpg', noisy)
    cv2.imwrite('canny_no_blur.jpg', canny_no_blur)
    cv2.imwrite('canny_with_blur.jpg', canny_with_blur)


def auto_canny_threshold():
    """자동 Canny 임계값"""
    print("\n" + "=" * 50)
    print("자동 Canny 임계값")
    print("=" * 50)

    img = create_test_image()

    # 중앙값 기반 자동 임계값
    median = np.median(img)
    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    auto_canny = cv2.Canny(img, lower, upper)

    print(f"이미지 중앙값: {median}")
    print(f"자동 임계값: lower={lower}, upper={upper}")
    print(f"공식: lower = (1-sigma)*median, upper = (1+sigma)*median")

    cv2.imwrite('canny_auto.jpg', auto_canny)

    return lower, upper


def log_edge_detection():
    """LoG (Laplacian of Gaussian) 엣지 검출"""
    print("\n" + "=" * 50)
    print("LoG 엣지 검출")
    print("=" * 50)

    img = create_test_image()

    # LoG = Gaussian 블러 + Laplacian
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    # Zero-crossing 찾기 (간단한 방법)
    log_abs = cv2.convertScaleAbs(log)

    print("LoG (Laplacian of Gaussian):")
    print("  - Gaussian으로 노이즈 제거")
    print("  - Laplacian으로 2차 미분")
    print("  - Zero-crossing이 엣지")

    cv2.imwrite('log_edge.jpg', log_abs)


def compare_edge_methods():
    """엣지 검출 방법 비교"""
    print("\n" + "=" * 50)
    print("엣지 검출 방법 비교")
    print("=" * 50)

    img = create_test_image()

    # 각 방법 적용
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))

    canny = cv2.Canny(img, 50, 150)

    print("""
    | 방법 | 특징 | 사용 상황 |
    |------|------|----------|
    | Sobel | 1차 미분, 방향별 | 그래디언트 방향 필요시 |
    | Scharr | Sobel 개선 | 더 정확한 그래디언트 |
    | Laplacian | 2차 미분 | 모든 방향 한번에 |
    | Canny | 다단계 처리 | 가장 많이 사용 |
    """)

    # 비교 이미지 생성
    compare = np.hstack([
        sobel,
        laplacian,
        canny
    ])
    cv2.imwrite('edge_compare.jpg', compare)


def practical_example():
    """실용 예제: 윤곽선 추출"""
    print("\n" + "=" * 50)
    print("실용 예제: 윤곽선 추출")
    print("=" * 50)

    # 컬러 이미지 생성
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 200), -1)
    cv2.circle(img, (300, 150), 60, (200, 0, 0), -1)

    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny 엣지
    edges = cv2.Canny(gray, 50, 150)

    # 윤곽선을 원본에 표시
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    cv2.imwrite('practical_input.jpg', img)
    cv2.imwrite('practical_edges.jpg', edges)
    cv2.imwrite('practical_contours.jpg', result)
    print("윤곽선 추출 이미지 저장 완료")


def main():
    """메인 함수"""
    # Sobel
    sobel_demo()

    # Scharr
    scharr_demo()

    # Laplacian
    laplacian_demo()

    # Canny
    canny_demo()

    # 블러 + Canny
    canny_with_blur()

    # 자동 임계값
    auto_canny_threshold()

    # LoG
    log_edge_detection()

    # 방법 비교
    compare_edge_methods()

    # 실용 예제
    practical_example()

    print("\n엣지 검출 데모 완료!")


if __name__ == '__main__':
    main()
