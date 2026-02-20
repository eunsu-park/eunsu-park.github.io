"""
12. 히스토그램 분석
- calcHist (히스토그램 계산)
- equalizeHist (히스토그램 평활화)
- CLAHE (적응형 히스토그램 평활화)
- 히스토그램 역투영
"""

import cv2
import numpy as np


def create_low_contrast_image():
    """저대비 이미지 생성"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 좁은 범위의 밝기값만 사용 (100-150)
    img[:] = 120

    # 도형 그리기
    cv2.rectangle(img, (50, 50), (150, 150), 140, -1)
    cv2.circle(img, (300, 150), 60, 130, -1)
    cv2.putText(img, 'LOW', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 150, 2)

    return img


def create_color_image():
    """컬러 테스트 이미지"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 다양한 색상 영역
    img[0:150, 0:200] = [200, 50, 50]      # 파랑
    img[0:150, 200:400] = [50, 200, 50]    # 초록
    img[150:300, 0:200] = [50, 50, 200]    # 빨강
    img[150:300, 200:400] = [200, 200, 50] # 청록

    return img


def calc_histogram_demo():
    """히스토그램 계산 데모"""
    print("=" * 50)
    print("히스토그램 계산 (calcHist)")
    print("=" * 50)

    img = create_low_contrast_image()

    # 히스토그램 계산
    # images: 입력 이미지 리스트
    # channels: 채널 인덱스 (그레이: [0], BGR: [0], [1], [2])
    # mask: 마스크 (None = 전체)
    # histSize: 빈 개수 (보통 256)
    # ranges: 값 범위

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    print(f"히스토그램 shape: {hist.shape}")
    print(f"총 픽셀 수: {hist.sum()}")
    print(f"최대 빈도 값: {hist.max():.0f}")
    print(f"최대 빈도 위치: {hist.argmax()}")

    # 히스토그램 시각화 (텍스트)
    print("\n히스토그램 분포 (간략):")
    for i in range(0, 256, 32):
        count = hist[i:i+32].sum()
        bar = '#' * int(count / 1000)
        print(f"  {i:3d}-{i+31:3d}: {bar} ({count:.0f})")

    cv2.imwrite('histogram_input.jpg', img)

    return hist


def histogram_color_demo():
    """컬러 히스토그램 데모"""
    print("\n" + "=" * 50)
    print("컬러 히스토그램")
    print("=" * 50)

    img = create_color_image()

    # BGR 각 채널 히스토그램
    colors = ('b', 'g', 'r')

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        peak = hist.argmax()
        print(f"{color.upper()} 채널: 최대 빈도 위치={peak}, 값={hist[peak][0]:.0f}")

    cv2.imwrite('histogram_color.jpg', img)


def equalize_histogram_demo():
    """히스토그램 평활화 데모"""
    print("\n" + "=" * 50)
    print("히스토그램 평활화 (equalizeHist)")
    print("=" * 50)

    img = create_low_contrast_image()

    # 히스토그램 평활화
    equalized = cv2.equalizeHist(img)

    # 전후 통계 비교
    print("평활화 전:")
    print(f"  Min={img.min()}, Max={img.max()}")
    print(f"  Mean={img.mean():.1f}, Std={img.std():.1f}")

    print("\n평활화 후:")
    print(f"  Min={equalized.min()}, Max={equalized.max()}")
    print(f"  Mean={equalized.mean():.1f}, Std={equalized.std():.1f}")

    print("\n평활화 효과:")
    print("  - 명암 대비 향상")
    print("  - 히스토그램이 균일하게 분포")
    print("  - 전체 이미지에 동일하게 적용")

    cv2.imwrite('equalize_before.jpg', img)
    cv2.imwrite('equalize_after.jpg', equalized)


def clahe_demo():
    """CLAHE 데모"""
    print("\n" + "=" * 50)
    print("CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    print("=" * 50)

    img = create_low_contrast_image()

    # 일반 평활화
    equalized = cv2.equalizeHist(img)

    # CLAHE
    # clipLimit: 대비 제한 (높을수록 대비 강함)
    # tileGridSize: 타일 크기

    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    result1 = clahe1.apply(img)
    result2 = clahe2.apply(img)
    result3 = clahe3.apply(img)

    print("CLAHE vs 일반 평활화:")
    print("  - CLAHE: 지역적으로 적용 (타일 단위)")
    print("  - 노이즈 증폭 방지 (clipLimit)")
    print("  - 불균일 조명에 효과적")

    print("\nCLAHE 파라미터:")
    print("  clipLimit: 대비 제한 (2.0~4.0 권장)")
    print("  tileGridSize: 타일 크기 (8x8 권장)")

    cv2.imwrite('clahe_equalized.jpg', equalized)
    cv2.imwrite('clahe_2_8.jpg', result1)
    cv2.imwrite('clahe_4_8.jpg', result2)
    cv2.imwrite('clahe_2_16.jpg', result3)


def clahe_color_demo():
    """컬러 이미지 CLAHE 데모"""
    print("\n" + "=" * 50)
    print("컬러 이미지 CLAHE")
    print("=" * 50)

    img = create_color_image()

    # LAB 색상 공간으로 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # L 채널에만 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # 다시 병합
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    print("컬러 이미지 CLAHE 방법:")
    print("  1. BGR → LAB 변환")
    print("  2. L 채널에 CLAHE 적용")
    print("  3. LAB → BGR 변환")
    print("  (색상(a,b)은 유지, 밝기(L)만 조정)")

    cv2.imwrite('clahe_color_before.jpg', img)
    cv2.imwrite('clahe_color_after.jpg', result)


def histogram_comparison_demo():
    """히스토그램 비교 데모"""
    print("\n" + "=" * 50)
    print("히스토그램 비교 (compareHist)")
    print("=" * 50)

    # 비교할 이미지들 생성
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img1[:] = 100
    cv2.rectangle(img1, (20, 20), (80, 80), 150, -1)

    img2 = img1.copy()  # 동일

    img3 = np.zeros((100, 100), dtype=np.uint8)
    img3[:] = 50
    cv2.rectangle(img3, (20, 20), (80, 80), 200, -1)

    # 히스토그램 계산
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])

    # 정규화
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist3, hist3, 0, 1, cv2.NORM_MINMAX)

    # 비교 방법
    methods = [
        (cv2.HISTCMP_CORREL, 'Correlation'),
        (cv2.HISTCMP_CHISQR, 'Chi-Square'),
        (cv2.HISTCMP_INTERSECT, 'Intersection'),
        (cv2.HISTCMP_BHATTACHARYYA, 'Bhattacharyya'),
    ]

    print("img1 vs img2 (동일), img1 vs img3 (다름):\n")

    for method, name in methods:
        score12 = cv2.compareHist(hist1, hist2, method)
        score13 = cv2.compareHist(hist1, hist3, method)
        print(f"  {name:15}: 동일={score12:.4f}, 다름={score13:.4f}")

    print("\n비교 방법 해석:")
    print("  Correlation: 1에 가까울수록 유사")
    print("  Chi-Square: 0에 가까울수록 유사")
    print("  Intersection: 클수록 유사")
    print("  Bhattacharyya: 0에 가까울수록 유사")


def back_projection_demo():
    """히스토그램 역투영 데모"""
    print("\n" + "=" * 50)
    print("히스토그램 역투영 (Back Projection)")
    print("=" * 50)

    # 대상 이미지 (다양한 색상)
    target = np.zeros((300, 400, 3), dtype=np.uint8)
    target[:] = [100, 100, 100]  # 회색 배경

    # 빨간 물체들
    cv2.circle(target, (100, 100), 40, (50, 50, 200), -1)
    cv2.circle(target, (300, 200), 50, (30, 30, 180), -1)
    cv2.rectangle(target, (150, 200), (220, 280), (40, 40, 210), -1)

    # 파란 물체
    cv2.circle(target, (350, 100), 30, (200, 50, 50), -1)

    # ROI (빨간색 샘플)
    roi = target[60:140, 60:140]  # 빨간 원 영역

    # HSV 변환
    hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ROI 히스토그램 계산 (Hue, Saturation만)
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # 역투영
    back_proj = cv2.calcBackProject([hsv_target], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # 결과 개선 (형태학적 연산)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(back_proj, -1, kernel, back_proj)
    _, thresh = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)

    print("역투영 과정:")
    print("  1. ROI의 히스토그램 계산")
    print("  2. 대상 이미지의 각 픽셀이 ROI와 얼마나 유사한지 계산")
    print("  3. 유사한 색상 영역이 밝게 표시됨")
    print("  4. 객체 추적에 활용 (MeanShift, CamShift)")

    cv2.imwrite('backproj_target.jpg', target)
    cv2.imwrite('backproj_roi.jpg', roi)
    cv2.imwrite('backproj_result.jpg', back_proj)
    cv2.imwrite('backproj_thresh.jpg', thresh)


def main():
    """메인 함수"""
    # 히스토그램 계산
    calc_histogram_demo()

    # 컬러 히스토그램
    histogram_color_demo()

    # 히스토그램 평활화
    equalize_histogram_demo()

    # CLAHE
    clahe_demo()

    # 컬러 CLAHE
    clahe_color_demo()

    # 히스토그램 비교
    histogram_comparison_demo()

    # 역투영
    back_projection_demo()

    print("\n히스토그램 분석 데모 완료!")


if __name__ == '__main__':
    main()
