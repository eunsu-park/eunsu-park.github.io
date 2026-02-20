"""
07. 이진화 및 임계처리
- 기본 임계처리 (threshold)
- Otsu's method
- 적응형 임계처리 (adaptiveThreshold)
- 다중 임계처리
"""

import cv2
import numpy as np


def create_gradient_image():
    """그라데이션 테스트 이미지 생성"""
    img = np.zeros((200, 400), dtype=np.uint8)

    # 수평 그라데이션
    for j in range(400):
        img[:, j] = int(j * 255 / 400)

    return img


def create_text_image():
    """텍스트가 있는 테스트 이미지"""
    img = np.zeros((200, 400), dtype=np.uint8)
    img[:] = 200  # 밝은 배경

    cv2.putText(img, 'Threshold', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 50, 3)
    cv2.putText(img, 'OpenCV', (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 80, 2)

    return img


def create_uneven_lighting():
    """불균일 조명 이미지"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 불균일 조명 배경
    for i in range(300):
        for j in range(400):
            img[i, j] = int(150 + 80 * np.sin(i / 50) * np.cos(j / 50))

    # 텍스트 추가
    cv2.putText(img, 'UNEVEN', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 30, 3)
    cv2.putText(img, 'LIGHTING', (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 2)

    return img


def basic_threshold_demo():
    """기본 임계처리 데모"""
    print("=" * 50)
    print("기본 임계처리 (threshold)")
    print("=" * 50)

    img = create_gradient_image()

    # 임계값 127로 이진화
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    print("THRESH_BINARY:")
    print("  - 픽셀 > 임계값 → 최대값 (255)")
    print("  - 픽셀 <= 임계값 → 0")

    print("\nTHRESH_BINARY_INV:")
    print("  - 픽셀 > 임계값 → 0")
    print("  - 픽셀 <= 임계값 → 최대값 (255)")

    cv2.imwrite('thresh_original.jpg', img)
    cv2.imwrite('thresh_binary.jpg', binary)
    cv2.imwrite('thresh_binary_inv.jpg', binary_inv)


def threshold_types_demo():
    """다양한 임계처리 타입"""
    print("\n" + "=" * 50)
    print("임계처리 타입")
    print("=" * 50)

    img = create_gradient_image()
    thresh_val = 127

    # 다양한 임계처리 타입
    _, binary = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    _, binary_inv = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    _, trunc = cv2.threshold(img, thresh_val, 255, cv2.THRESH_TRUNC)
    _, tozero = cv2.threshold(img, thresh_val, 255, cv2.THRESH_TOZERO)
    _, tozero_inv = cv2.threshold(img, thresh_val, 255, cv2.THRESH_TOZERO_INV)

    print("임계처리 타입:")
    print("  BINARY:     픽셀 > T → 255, 아니면 0")
    print("  BINARY_INV: 픽셀 > T → 0, 아니면 255")
    print("  TRUNC:      픽셀 > T → T, 아니면 원본")
    print("  TOZERO:     픽셀 > T → 원본, 아니면 0")
    print("  TOZERO_INV: 픽셀 > T → 0, 아니면 원본")

    cv2.imwrite('thresh_trunc.jpg', trunc)
    cv2.imwrite('thresh_tozero.jpg', tozero)
    cv2.imwrite('thresh_tozero_inv.jpg', tozero_inv)


def otsu_demo():
    """Otsu's method 데모"""
    print("\n" + "=" * 50)
    print("Otsu's Method")
    print("=" * 50)

    img = create_text_image()

    # 일반 임계처리
    _, binary_100 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    _, binary_150 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # Otsu's method (자동 임계값)
    otsu_val, binary_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"수동 임계값: 100, 150")
    print(f"Otsu 자동 임계값: {otsu_val}")

    print("\nOtsu's method 특성:")
    print("  - 히스토그램 기반 자동 임계값 결정")
    print("  - 이중 피크(bimodal) 분포에 효과적")
    print("  - 클래스 간 분산 최대화")

    cv2.imwrite('otsu_input.jpg', img)
    cv2.imwrite('otsu_100.jpg', binary_100)
    cv2.imwrite('otsu_150.jpg', binary_150)
    cv2.imwrite('otsu_auto.jpg', binary_otsu)


def adaptive_threshold_demo():
    """적응형 임계처리 데모"""
    print("\n" + "=" * 50)
    print("적응형 임계처리 (Adaptive Threshold)")
    print("=" * 50)

    img = create_uneven_lighting()

    # 일반 임계처리 (불균일 조명에서 실패)
    _, binary_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu도 불균일 조명에서 제한적
    _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 적응형 임계처리 (Mean)
    binary_mean = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,  # 블록 크기 (홀수)
        2    # C (상수)
    )

    # 적응형 임계처리 (Gaussian)
    binary_gaussian = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    print("적응형 임계처리:")
    print("  - 각 픽셀 주변 영역의 평균으로 임계값 결정")
    print("  - MEAN: 단순 평균")
    print("  - GAUSSIAN: 가우시안 가중 평균")
    print("  - 불균일 조명에 효과적")

    cv2.imwrite('adaptive_input.jpg', img)
    cv2.imwrite('adaptive_global.jpg', binary_global)
    cv2.imwrite('adaptive_otsu.jpg', binary_otsu)
    cv2.imwrite('adaptive_mean.jpg', binary_mean)
    cv2.imwrite('adaptive_gaussian.jpg', binary_gaussian)


def adaptive_params_demo():
    """적응형 임계처리 파라미터"""
    print("\n" + "=" * 50)
    print("적응형 임계처리 파라미터")
    print("=" * 50)

    img = create_uneven_lighting()

    # 블록 크기 변화
    sizes = [5, 11, 31, 51]
    for size in sizes:
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            size, 2
        )
        cv2.imwrite(f'adaptive_size_{size}.jpg', binary)

    # C 값 변화
    c_values = [0, 2, 5, 10]
    for c in c_values:
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, c
        )
        cv2.imwrite(f'adaptive_c_{c}.jpg', binary)

    print("파라미터 영향:")
    print("  - 블록 크기: 클수록 넓은 영역 고려")
    print("  - C 값: 임계값에서 뺄 상수")
    print("    C가 크면 → 더 많은 영역이 전경")


def triangle_threshold_demo():
    """삼각형 임계처리 데모"""
    print("\n" + "=" * 50)
    print("삼각형 임계처리 (Triangle)")
    print("=" * 50)

    # 한쪽으로 치우친 히스토그램 이미지
    img = np.zeros((200, 400), dtype=np.uint8)
    img[:] = 200
    cv2.rectangle(img, (50, 50), (150, 150), 30, -1)
    cv2.circle(img, (300, 100), 40, 50, -1)

    # Triangle method
    tri_val, binary_tri = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )

    # Otsu와 비교
    otsu_val, binary_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"Triangle 자동 임계값: {tri_val}")
    print(f"Otsu 자동 임계값: {otsu_val}")

    print("\nTriangle method:")
    print("  - 단봉(unimodal) 분포에 효과적")
    print("  - 히스토그램 피크에서 가장 먼 점 찾기")

    cv2.imwrite('triangle_input.jpg', img)
    cv2.imwrite('triangle_result.jpg', binary_tri)


def multi_threshold_demo():
    """다중 임계처리"""
    print("\n" + "=" * 50)
    print("다중 임계처리")
    print("=" * 50)

    img = create_gradient_image()

    # 다중 레벨 양자화
    result = np.zeros_like(img)
    thresholds = [50, 100, 150, 200]
    values = [0, 64, 128, 192, 255]

    for i, (low, high, val) in enumerate(
        zip([0] + thresholds, thresholds + [256], values)
    ):
        mask = (img >= low) & (img < high)
        result[mask] = val

    print("다중 임계처리:")
    print(f"  임계값: {thresholds}")
    print(f"  결과값: {values}")

    cv2.imwrite('multi_thresh.jpg', result)


def practical_document_scan():
    """실용 예제: 문서 스캔 이진화"""
    print("\n" + "=" * 50)
    print("실용 예제: 문서 스캔")
    print("=" * 50)

    # 문서 이미지 시뮬레이션
    img = np.zeros((300, 400), dtype=np.uint8)

    # 불균일 배경
    for i in range(300):
        for j in range(400):
            img[i, j] = int(200 + 30 * np.sin(i / 100) + 20 * np.cos(j / 100))

    # 텍스트
    cv2.putText(img, 'Document', (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 30, 2)
    cv2.putText(img, 'Scanning', (90, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 50, 2)
    cv2.putText(img, 'Example', (110, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, 40, 2)

    # 가우시안 블러로 노이즈 감소
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 적응형 임계처리
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )

    # 모폴로지로 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('document_original.jpg', img)
    cv2.imwrite('document_binary.jpg', cleaned)
    print("문서 스캔 이미지 저장 완료")


def main():
    """메인 함수"""
    # 기본 임계처리
    basic_threshold_demo()

    # 임계처리 타입
    threshold_types_demo()

    # Otsu's method
    otsu_demo()

    # 적응형 임계처리
    adaptive_threshold_demo()

    # 적응형 파라미터
    adaptive_params_demo()

    # Triangle method
    triangle_threshold_demo()

    # 다중 임계처리
    multi_threshold_demo()

    # 실용 예제
    practical_document_scan()

    print("\n이진화 및 임계처리 데모 완료!")


if __name__ == '__main__':
    main()
