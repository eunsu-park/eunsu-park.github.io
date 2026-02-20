"""
03. 색상 공간
- BGR, RGB, HSV, LAB, YCrCb
- cvtColor 변환
- 채널 분리/병합
- 색상 기반 객체 추출
"""

import cv2
import numpy as np


def create_color_image():
    """다양한 색상의 테스트 이미지 생성"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 무지개 색상
    colors = [
        [0, 0, 255],      # 빨강
        [0, 128, 255],    # 주황
        [0, 255, 255],    # 노랑
        [0, 255, 0],      # 초록
        [255, 255, 0],    # 청록
        [255, 0, 0],      # 파랑
        [255, 0, 128],    # 보라
    ]

    width = 400 // len(colors)
    for i, color in enumerate(colors):
        img[:, i*width:(i+1)*width] = color

    return img


def bgr_rgb_demo():
    """BGR vs RGB 데모"""
    print("=" * 50)
    print("BGR vs RGB")
    print("=" * 50)

    img = create_color_image()

    # OpenCV는 BGR 순서
    print("OpenCV는 BGR 순서 사용")
    print(f"빨간색 픽셀 BGR: {img[150, 25]}")  # [0, 0, 255]

    # RGB로 변환
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"변환 후 RGB: {rgb[150, 25]}")  # [255, 0, 0]

    # matplotlib은 RGB 사용
    # plt.imshow(rgb)  # 올바른 색상
    # plt.imshow(img)  # 색상 뒤바뀜

    cv2.imwrite('color_bgr.jpg', img)
    cv2.imwrite('color_rgb.jpg', rgb)


def hsv_demo():
    """HSV 색상 공간 데모"""
    print("\n" + "=" * 50)
    print("HSV 색상 공간")
    print("=" * 50)

    img = create_color_image()

    # BGR -> HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV 채널 분리
    h, s, v = cv2.split(hsv)

    print("HSV 범위:")
    print("  H (Hue): 0-179")
    print("  S (Saturation): 0-255")
    print("  V (Value): 0-255")

    # 빨간색 영역의 HSV
    print(f"\n빨간색 HSV: {hsv[150, 25]}")
    print(f"  H={hsv[150, 25, 0]}, S={hsv[150, 25, 1]}, V={hsv[150, 25, 2]}")

    # 채널별 저장
    cv2.imwrite('hsv_h.jpg', h)
    cv2.imwrite('hsv_s.jpg', s)
    cv2.imwrite('hsv_v.jpg', v)
    print("\nHSV 채널 이미지 저장 완료")

    return hsv


def color_extraction_demo():
    """색상 기반 객체 추출 데모"""
    print("\n" + "=" * 50)
    print("색상 기반 객체 추출")
    print("=" * 50)

    # 다양한 색상의 이미지 생성
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(img, (100, 150), 50, (0, 0, 255), -1)   # 빨간 원
    cv2.circle(img, (200, 150), 50, (0, 255, 0), -1)   # 초록 원
    cv2.circle(img, (300, 150), 50, (255, 0, 0), -1)   # 파란 원

    # HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 (HSV)
    # 빨간색은 H=0 또는 H=180 근처
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 마스크 생성
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 빨간색 추출
    red_only = cv2.bitwise_and(img, img, mask=red_mask)

    # 초록색 범위
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_only = cv2.bitwise_and(img, img, mask=green_mask)

    # 파란색 범위
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_only = cv2.bitwise_and(img, img, mask=blue_mask)

    cv2.imwrite('original_circles.jpg', img)
    cv2.imwrite('red_extracted.jpg', red_only)
    cv2.imwrite('green_extracted.jpg', green_only)
    cv2.imwrite('blue_extracted.jpg', blue_only)
    print("색상 추출 이미지 저장 완료")


def lab_demo():
    """LAB 색상 공간 데모"""
    print("\n" + "=" * 50)
    print("LAB 색상 공간")
    print("=" * 50)

    img = create_color_image()

    # BGR -> LAB 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # LAB 채널 분리
    l, a, b = cv2.split(lab)

    print("LAB 범위:")
    print("  L (Lightness): 0-255")
    print("  a (Green-Red): 0-255 (128이 중립)")
    print("  b (Blue-Yellow): 0-255 (128이 중립)")

    cv2.imwrite('lab_l.jpg', l)
    cv2.imwrite('lab_a.jpg', a)
    cv2.imwrite('lab_b.jpg', b)
    print("LAB 채널 이미지 저장 완료")


def ycrcb_demo():
    """YCrCb 색상 공간 데모"""
    print("\n" + "=" * 50)
    print("YCrCb 색상 공간")
    print("=" * 50)

    img = create_color_image()

    # BGR -> YCrCb 변환
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # YCrCb 채널 분리
    y, cr, cb = cv2.split(ycrcb)

    print("YCrCb 범위:")
    print("  Y (Luminance): 0-255")
    print("  Cr (Red-difference): 0-255")
    print("  Cb (Blue-difference): 0-255")

    cv2.imwrite('ycrcb_y.jpg', y)
    cv2.imwrite('ycrcb_cr.jpg', cr)
    cv2.imwrite('ycrcb_cb.jpg', cb)
    print("YCrCb 채널 이미지 저장 완료")


def grayscale_methods():
    """그레이스케일 변환 방법"""
    print("\n" + "=" * 50)
    print("그레이스케일 변환")
    print("=" * 50)

    img = create_color_image()

    # 방법 1: cvtColor
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 방법 2: 가중치 평균 (직접 계산)
    # Gray = 0.299*R + 0.587*G + 0.114*B
    b, g, r = cv2.split(img)
    gray2 = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    # 방법 3: 단순 평균
    gray3 = np.mean(img, axis=2).astype(np.uint8)

    cv2.imwrite('gray_cvtcolor.jpg', gray1)
    cv2.imwrite('gray_weighted.jpg', gray2)
    cv2.imwrite('gray_average.jpg', gray3)
    print("그레이스케일 변환 이미지 저장 완료")

    print(f"\n변환 결과 비교 (특정 픽셀):")
    print(f"  cvtColor: {gray1[150, 25]}")
    print(f"  가중치: {gray2[150, 25]}")
    print(f"  평균: {gray3[150, 25]}")


def color_conversion_table():
    """색상 변환 코드 표"""
    print("\n" + "=" * 50)
    print("주요 색상 변환 코드")
    print("=" * 50)

    conversions = [
        ("BGR -> Gray", "cv2.COLOR_BGR2GRAY"),
        ("BGR -> RGB", "cv2.COLOR_BGR2RGB"),
        ("BGR -> HSV", "cv2.COLOR_BGR2HSV"),
        ("BGR -> LAB", "cv2.COLOR_BGR2LAB"),
        ("BGR -> YCrCb", "cv2.COLOR_BGR2YCrCb"),
        ("HSV -> BGR", "cv2.COLOR_HSV2BGR"),
        ("Gray -> BGR", "cv2.COLOR_GRAY2BGR"),
    ]

    for desc, code in conversions:
        print(f"  {desc:15} -> {code}")


def main():
    """메인 함수"""
    # BGR vs RGB
    bgr_rgb_demo()

    # HSV
    hsv_demo()

    # 색상 추출
    color_extraction_demo()

    # LAB
    lab_demo()

    # YCrCb
    ycrcb_demo()

    # 그레이스케일
    grayscale_methods()

    # 변환 코드 표
    color_conversion_table()

    print("\n색상 공간 데모 완료!")


if __name__ == '__main__':
    main()
