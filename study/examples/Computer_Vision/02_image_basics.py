"""
02. 이미지 기초 연산
- imread, imshow, imwrite
- 픽셀 접근 및 수정
- ROI (Region of Interest)
- 이미지 복사 및 채널 조작
"""

import cv2
import numpy as np


def create_sample_image():
    """샘플 이미지 생성"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 컬러 영역
    img[0:100, 0:200] = [255, 0, 0]      # 파랑
    img[0:100, 200:400] = [0, 255, 0]    # 초록
    img[100:200, 0:200] = [0, 0, 255]    # 빨강
    img[100:200, 200:400] = [255, 255, 0] # 청록
    img[200:300, :] = [128, 128, 128]    # 회색

    return img


def image_read_write_demo():
    """이미지 읽기/쓰기 데모"""
    print("=" * 50)
    print("이미지 읽기/쓰기")
    print("=" * 50)

    # 이미지 생성 및 저장
    img = create_sample_image()
    cv2.imwrite('test_image.jpg', img)
    print("test_image.jpg 저장 완료")

    # 이미지 읽기
    # cv2.IMREAD_COLOR: 컬러 (기본값)
    # cv2.IMREAD_GRAYSCALE: 그레이스케일
    # cv2.IMREAD_UNCHANGED: 원본 그대로 (알파 채널 포함)

    img_color = cv2.imread('test_image.jpg', cv2.IMREAD_COLOR)
    img_gray = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

    print(f"컬러 이미지 shape: {img_color.shape}")
    print(f"그레이 이미지 shape: {img_gray.shape}")

    return img_color


def pixel_access_demo(img):
    """픽셀 접근 데모"""
    print("\n" + "=" * 50)
    print("픽셀 접근")
    print("=" * 50)

    # 단일 픽셀 접근 (y, x 순서 주의!)
    pixel = img[50, 100]  # (y=50, x=100) 위치
    print(f"픽셀 (50, 100) BGR 값: {pixel}")

    # 개별 채널 접근
    b = img[50, 100, 0]
    g = img[50, 100, 1]
    r = img[50, 100, 2]
    print(f"B={b}, G={g}, R={r}")

    # 픽셀 수정
    img_copy = img.copy()
    img_copy[50, 100] = [0, 0, 0]  # 검정으로 변경

    # 영역 수정 (더 효율적)
    img_copy[0:50, 0:50] = [255, 255, 255]  # 흰색 사각형

    return img_copy


def roi_demo(img):
    """ROI (Region of Interest) 데모"""
    print("\n" + "=" * 50)
    print("ROI (Region of Interest)")
    print("=" * 50)

    # ROI 추출 (슬라이싱)
    roi = img[50:150, 100:250]  # y: 50~150, x: 100~250
    print(f"원본 shape: {img.shape}")
    print(f"ROI shape: {roi.shape}")

    # ROI 복사 (원본 영향 없음)
    roi_copy = img[50:150, 100:250].copy()

    # ROI 붙여넣기
    img_with_roi = img.copy()
    img_with_roi[150:250, 200:350] = roi  # 다른 위치에 붙여넣기

    cv2.imwrite('roi_demo.jpg', img_with_roi)
    print("roi_demo.jpg 저장 완료")

    return roi


def channel_operations_demo(img):
    """채널 연산 데모"""
    print("\n" + "=" * 50)
    print("채널 연산")
    print("=" * 50)

    # 채널 분리
    b, g, r = cv2.split(img)
    print(f"B 채널 shape: {b.shape}")
    print(f"G 채널 shape: {g.shape}")
    print(f"R 채널 shape: {r.shape}")

    # 채널 병합
    merged = cv2.merge([b, g, r])
    print(f"병합 후 shape: {merged.shape}")

    # 채널 순서 변경 (BGR -> RGB)
    rgb = cv2.merge([r, g, b])

    # 단일 채널만 사용한 이미지 생성
    zeros = np.zeros_like(b)
    only_blue = cv2.merge([b, zeros, zeros])
    only_green = cv2.merge([zeros, g, zeros])
    only_red = cv2.merge([zeros, zeros, r])

    cv2.imwrite('only_blue.jpg', only_blue)
    cv2.imwrite('only_green.jpg', only_green)
    cv2.imwrite('only_red.jpg', only_red)
    print("채널 분리 이미지 저장 완료")


def image_properties_demo(img):
    """이미지 속성 데모"""
    print("\n" + "=" * 50)
    print("이미지 속성")
    print("=" * 50)

    # 기본 속성
    print(f"Shape (H, W, C): {img.shape}")
    print(f"Height: {img.shape[0]}")
    print(f"Width: {img.shape[1]}")
    print(f"Channels: {img.shape[2] if len(img.shape) > 2 else 1}")

    # 데이터 타입
    print(f"Data type: {img.dtype}")

    # 전체 픽셀 수
    print(f"Total pixels: {img.size}")

    # 메모리 사용량
    print(f"Memory (bytes): {img.nbytes}")

    # 픽셀 값 범위
    print(f"Min value: {img.min()}")
    print(f"Max value: {img.max()}")
    print(f"Mean value: {img.mean():.2f}")


def image_arithmetic_demo():
    """이미지 산술 연산 데모"""
    print("\n" + "=" * 50)
    print("이미지 산술 연산")
    print("=" * 50)

    # 두 이미지 생성
    img1 = np.full((200, 200, 3), 100, dtype=np.uint8)
    img2 = np.full((200, 200, 3), 200, dtype=np.uint8)

    # NumPy 덧셈 (오버플로우 발생)
    result_np = img1 + img2
    print(f"NumPy 덧셈 결과 (100+200): {result_np[0, 0]}")  # 44 (오버플로우)

    # OpenCV 덧셈 (포화 연산, saturate)
    result_cv = cv2.add(img1, img2)
    print(f"OpenCV 덧셈 결과 (100+200): {result_cv[0, 0]}")  # 255 (포화)

    # 가중치 합 (블렌딩)
    alpha = 0.7
    beta = 0.3
    blended = cv2.addWeighted(img1, alpha, img2, beta, 0)
    print(f"블렌딩 결과 (0.7*100 + 0.3*200): {blended[0, 0]}")

    # 뺄셈
    diff = cv2.subtract(img2, img1)
    print(f"뺄셈 결과 (200-100): {diff[0, 0]}")

    # 비트 연산
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    masked = cv2.bitwise_and(img1, img1, mask=mask)
    cv2.imwrite('masked_result.jpg', masked)
    print("masked_result.jpg 저장 완료")


def main():
    """메인 함수"""
    # 이미지 읽기/쓰기
    img = image_read_write_demo()

    # 픽셀 접근
    modified = pixel_access_demo(img)

    # ROI
    roi = roi_demo(img)

    # 채널 연산
    channel_operations_demo(img)

    # 이미지 속성
    image_properties_demo(img)

    # 산술 연산
    image_arithmetic_demo()

    print("\n이미지 기초 연산 완료!")


if __name__ == '__main__':
    main()
