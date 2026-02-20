"""
05. 이미지 필터링
- blur, GaussianBlur, medianBlur
- bilateralFilter
- 커스텀 필터 (filter2D)
- 샤프닝
"""

import cv2
import numpy as np


def create_noisy_image():
    """노이즈가 있는 테스트 이미지 생성"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # 도형 그리기
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.circle(img, (300, 150), 50, (255, 0, 0), -1)
    cv2.putText(img, 'Filter', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 가우시안 노이즈 추가
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, noisy


def create_salt_pepper_noise(img):
    """소금-후추 노이즈 추가"""
    noisy = img.copy()
    # 소금 (흰색)
    salt = np.random.random(img.shape[:2]) < 0.02
    noisy[salt] = 255
    # 후추 (검정)
    pepper = np.random.random(img.shape[:2]) < 0.02
    noisy[pepper] = 0
    return noisy


def blur_demo():
    """블러 필터 데모"""
    print("=" * 50)
    print("블러 필터")
    print("=" * 50)

    original, noisy = create_noisy_image()

    # 평균 블러 (Box Filter)
    blur_3x3 = cv2.blur(noisy, (3, 3))
    blur_5x5 = cv2.blur(noisy, (5, 5))
    blur_7x7 = cv2.blur(noisy, (7, 7))

    print("평균 블러 (Box Filter):")
    print("  - 모든 픽셀에 동일한 가중치")
    print("  - 커널 크기가 클수록 더 흐려짐")

    cv2.imwrite('original.jpg', original)
    cv2.imwrite('noisy.jpg', noisy)
    cv2.imwrite('blur_3x3.jpg', blur_3x3)
    cv2.imwrite('blur_5x5.jpg', blur_5x5)
    cv2.imwrite('blur_7x7.jpg', blur_7x7)


def gaussian_blur_demo():
    """가우시안 블러 데모"""
    print("\n" + "=" * 50)
    print("가우시안 블러")
    print("=" * 50)

    _, noisy = create_noisy_image()

    # 가우시안 블러
    # GaussianBlur(src, ksize, sigmaX)
    gauss_3x3 = cv2.GaussianBlur(noisy, (3, 3), 0)
    gauss_5x5 = cv2.GaussianBlur(noisy, (5, 5), 0)
    gauss_7x7 = cv2.GaussianBlur(noisy, (7, 7), 0)

    # sigma 값에 따른 차이
    gauss_s1 = cv2.GaussianBlur(noisy, (5, 5), 1)
    gauss_s3 = cv2.GaussianBlur(noisy, (5, 5), 3)
    gauss_s5 = cv2.GaussianBlur(noisy, (5, 5), 5)

    print("가우시안 블러:")
    print("  - 중앙에 높은 가중치, 가장자리에 낮은 가중치")
    print("  - 자연스러운 블러 효과")
    print("  - sigma가 클수록 더 흐려짐")

    cv2.imwrite('gauss_3x3.jpg', gauss_3x3)
    cv2.imwrite('gauss_5x5.jpg', gauss_5x5)
    cv2.imwrite('gauss_sigma1.jpg', gauss_s1)
    cv2.imwrite('gauss_sigma5.jpg', gauss_s5)


def median_blur_demo():
    """미디언 블러 데모"""
    print("\n" + "=" * 50)
    print("미디언 블러")
    print("=" * 50)

    original, _ = create_noisy_image()
    sp_noisy = create_salt_pepper_noise(original)

    # 미디언 블러
    median_3 = cv2.medianBlur(sp_noisy, 3)
    median_5 = cv2.medianBlur(sp_noisy, 5)
    median_7 = cv2.medianBlur(sp_noisy, 7)

    print("미디언 블러:")
    print("  - 커널 내 중앙값 사용")
    print("  - 소금-후추 노이즈 제거에 효과적")
    print("  - 엣지 보존 효과")

    cv2.imwrite('salt_pepper_noisy.jpg', sp_noisy)
    cv2.imwrite('median_3.jpg', median_3)
    cv2.imwrite('median_5.jpg', median_5)


def bilateral_filter_demo():
    """양방향 필터 데모"""
    print("\n" + "=" * 50)
    print("양방향 필터 (Bilateral Filter)")
    print("=" * 50)

    _, noisy = create_noisy_image()

    # 양방향 필터
    # bilateralFilter(src, d, sigmaColor, sigmaSpace)
    # d: 필터 크기 (-1이면 sigmaSpace에서 자동 계산)
    # sigmaColor: 색상 공간에서의 시그마
    # sigmaSpace: 좌표 공간에서의 시그마

    bilateral_1 = cv2.bilateralFilter(noisy, 9, 75, 75)
    bilateral_2 = cv2.bilateralFilter(noisy, 9, 150, 150)
    bilateral_3 = cv2.bilateralFilter(noisy, -1, 75, 75)

    print("양방향 필터:")
    print("  - 엣지를 보존하면서 노이즈 제거")
    print("  - sigmaColor: 색상 차이 허용 범위")
    print("  - sigmaSpace: 공간적 영향 범위")
    print("  - 다른 필터보다 느림")

    cv2.imwrite('bilateral_75.jpg', bilateral_1)
    cv2.imwrite('bilateral_150.jpg', bilateral_2)


def custom_filter_demo():
    """커스텀 필터 데모"""
    print("\n" + "=" * 50)
    print("커스텀 필터 (filter2D)")
    print("=" * 50)

    original, _ = create_noisy_image()

    # 평균 필터 커널
    kernel_avg = np.ones((3, 3), dtype=np.float32) / 9
    avg_filtered = cv2.filter2D(original, -1, kernel_avg)

    # 샤프닝 커널
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(original, -1, kernel_sharpen)

    # 강한 샤프닝
    kernel_sharpen_strong = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    sharpened_strong = cv2.filter2D(original, -1, kernel_sharpen_strong)

    # 엠보싱 커널
    kernel_emboss = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)
    embossed = cv2.filter2D(original, -1, kernel_emboss)

    print("커스텀 커널 예시:")
    print(f"  평균 필터:\n{kernel_avg}")
    print(f"\n  샤프닝:\n{kernel_sharpen}")
    print(f"\n  엠보싱:\n{kernel_emboss}")

    cv2.imwrite('custom_avg.jpg', avg_filtered)
    cv2.imwrite('custom_sharpen.jpg', sharpened)
    cv2.imwrite('custom_sharpen_strong.jpg', sharpened_strong)
    cv2.imwrite('custom_emboss.jpg', embossed)


def unsharp_masking_demo():
    """언샤프 마스킹 데모"""
    print("\n" + "=" * 50)
    print("언샤프 마스킹 (Unsharp Masking)")
    print("=" * 50)

    original, _ = create_noisy_image()

    # 언샤프 마스킹: 원본 + (원본 - 블러) * 강도
    blurred = cv2.GaussianBlur(original, (5, 5), 0)

    # 방법 1: 직접 계산
    unsharp = cv2.addWeighted(original, 1.5, blurred, -0.5, 0)

    # 방법 2: 공식 적용
    alpha = 1.5  # 샤프닝 강도
    unsharp2 = cv2.addWeighted(original, 1 + alpha, blurred, -alpha, 0)

    print("언샤프 마스킹:")
    print("  결과 = 원본 + alpha * (원본 - 블러)")
    print("  alpha가 클수록 샤프닝 효과 강함")

    cv2.imwrite('unsharp_mask.jpg', unsharp)
    cv2.imwrite('unsharp_mask2.jpg', unsharp2)


def filter_comparison():
    """필터 비교"""
    print("\n" + "=" * 50)
    print("필터 비교 정리")
    print("=" * 50)

    print("""
    | 필터 | 특징 | 사용 상황 |
    |------|------|----------|
    | blur (Box) | 균일한 가중치 | 단순 평균화 |
    | GaussianBlur | 중앙 가중치 높음 | 자연스러운 블러 |
    | medianBlur | 중앙값 사용 | 소금-후추 노이즈 |
    | bilateralFilter | 엣지 보존 | 피부 보정 등 |
    """)


def main():
    """메인 함수"""
    # 블러 필터
    blur_demo()

    # 가우시안 블러
    gaussian_blur_demo()

    # 미디언 블러
    median_blur_demo()

    # 양방향 필터
    bilateral_filter_demo()

    # 커스텀 필터
    custom_filter_demo()

    # 언샤프 마스킹
    unsharp_masking_demo()

    # 비교
    filter_comparison()

    print("\n이미지 필터링 데모 완료!")


if __name__ == '__main__':
    main()
