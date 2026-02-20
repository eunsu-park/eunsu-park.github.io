"""
06. 모폴로지 연산
- 침식 (erode), 팽창 (dilate)
- 열기 (opening), 닫기 (closing)
- 그래디언트, 탑햇, 블랙햇
- 구조 요소 (structuring element)
"""

import cv2
import numpy as np


def create_binary_image():
    """이진 이미지 생성"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 사각형
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

    # 원
    cv2.circle(img, (300, 150), 50, 255, -1)

    # 텍스트
    cv2.putText(img, 'MORPH', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)

    return img


def create_noisy_binary():
    """노이즈가 있는 이진 이미지"""
    img = create_binary_image()

    # 작은 노이즈 점 추가 (소금-후추)
    noise_salt = np.random.random(img.shape) < 0.01
    noise_pepper = np.random.random(img.shape) < 0.01
    img[noise_salt] = 255
    img[noise_pepper] = 0

    return img


def structuring_element_demo():
    """구조 요소 데모"""
    print("=" * 50)
    print("구조 요소 (Structuring Element)")
    print("=" * 50)

    # 사각형 구조 요소
    rect_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    rect_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 십자형 구조 요소
    cross_3x3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cross_5x5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # 타원형 구조 요소
    ellipse_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("사각형 3x3:")
    print(rect_3x3)
    print("\n십자형 3x3:")
    print(cross_3x3)
    print("\n타원형 5x5:")
    print(ellipse_5x5)

    return rect_5x5


def erosion_demo():
    """침식 (Erosion) 데모"""
    print("\n" + "=" * 50)
    print("침식 (Erosion)")
    print("=" * 50)

    img = create_binary_image()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 침식 적용
    eroded_1 = cv2.erode(img, kernel, iterations=1)
    eroded_2 = cv2.erode(img, kernel, iterations=2)
    eroded_3 = cv2.erode(img, kernel, iterations=3)

    print("침식 특성:")
    print("  - 전경(흰색) 영역 축소")
    print("  - 작은 노이즈 제거")
    print("  - iterations 증가 → 더 많이 축소")

    cv2.imwrite('morph_original.jpg', img)
    cv2.imwrite('erode_1.jpg', eroded_1)
    cv2.imwrite('erode_2.jpg', eroded_2)
    cv2.imwrite('erode_3.jpg', eroded_3)


def dilation_demo():
    """팽창 (Dilation) 데모"""
    print("\n" + "=" * 50)
    print("팽창 (Dilation)")
    print("=" * 50)

    img = create_binary_image()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 팽창 적용
    dilated_1 = cv2.dilate(img, kernel, iterations=1)
    dilated_2 = cv2.dilate(img, kernel, iterations=2)
    dilated_3 = cv2.dilate(img, kernel, iterations=3)

    print("팽창 특성:")
    print("  - 전경(흰색) 영역 확대")
    print("  - 구멍 메우기")
    print("  - 객체 연결")

    cv2.imwrite('dilate_1.jpg', dilated_1)
    cv2.imwrite('dilate_2.jpg', dilated_2)
    cv2.imwrite('dilate_3.jpg', dilated_3)


def opening_demo():
    """열기 (Opening) 데모"""
    print("\n" + "=" * 50)
    print("열기 (Opening) = 침식 + 팽창")
    print("=" * 50)

    img = create_noisy_binary()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 열기 = 침식 후 팽창
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 수동으로도 가능
    opened_manual = cv2.dilate(cv2.erode(img, kernel), kernel)

    print("열기 특성:")
    print("  - 침식 후 팽창")
    print("  - 작은 노이즈(흰색 점) 제거")
    print("  - 객체 크기는 거의 유지")

    cv2.imwrite('noisy_binary.jpg', img)
    cv2.imwrite('opened.jpg', opened)


def closing_demo():
    """닫기 (Closing) 데모"""
    print("\n" + "=" * 50)
    print("닫기 (Closing) = 팽창 + 침식")
    print("=" * 50)

    img = create_noisy_binary()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 닫기 = 팽창 후 침식
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    print("닫기 특성:")
    print("  - 팽창 후 침식")
    print("  - 작은 구멍(검정 점) 채우기")
    print("  - 객체 크기는 거의 유지")

    cv2.imwrite('closed.jpg', closed)


def gradient_demo():
    """모폴로지 그래디언트 데모"""
    print("\n" + "=" * 50)
    print("모폴로지 그래디언트")
    print("=" * 50)

    img = create_binary_image()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 그래디언트 = 팽창 - 침식
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # 수동 계산
    dilated = cv2.dilate(img, kernel)
    eroded = cv2.erode(img, kernel)
    gradient_manual = cv2.subtract(dilated, eroded)

    print("그래디언트 특성:")
    print("  - 팽창 - 침식")
    print("  - 객체의 외곽선 추출")

    cv2.imwrite('gradient.jpg', gradient)


def tophat_blackhat_demo():
    """탑햇, 블랙햇 데모"""
    print("\n" + "=" * 50)
    print("탑햇 & 블랙햇")
    print("=" * 50)

    # 불균일한 조명의 이미지 시뮬레이션
    img = np.zeros((300, 400), dtype=np.uint8)

    # 그라데이션 배경 (불균일 조명)
    for i in range(300):
        for j in range(400):
            img[i, j] = int(50 + 100 * (i / 300) + 50 * (j / 400))

    # 밝은 객체
    cv2.rectangle(img, (100, 100), (150, 150), 255, -1)
    cv2.circle(img, (300, 150), 30, 255, -1)

    # 어두운 객체
    cv2.rectangle(img, (50, 200), (100, 250), 30, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    # 탑햇 = 원본 - 열기 (밝은 영역 강조)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # 블랙햇 = 닫기 - 원본 (어두운 영역 강조)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    print("탑햇:")
    print("  - 원본 - 열기")
    print("  - 배경보다 밝은 영역 강조")
    print("  - 불균일 조명 보정에 활용")

    print("\n블랙햇:")
    print("  - 닫기 - 원본")
    print("  - 배경보다 어두운 영역 강조")

    cv2.imwrite('uneven_lighting.jpg', img)
    cv2.imwrite('tophat.jpg', tophat)
    cv2.imwrite('blackhat.jpg', blackhat)


def hit_miss_demo():
    """히트미스 변환 데모"""
    print("\n" + "=" * 50)
    print("히트미스 변환 (Hit-Miss)")
    print("=" * 50)

    # 특정 패턴 찾기
    img = np.zeros((200, 200), dtype=np.uint8)
    # L자 모양 패턴 생성
    cv2.rectangle(img, (50, 50), (70, 100), 255, -1)
    cv2.rectangle(img, (50, 80), (100, 100), 255, -1)

    # 다른 위치에도
    cv2.rectangle(img, (120, 120), (140, 170), 255, -1)
    cv2.rectangle(img, (120, 150), (170, 170), 255, -1)

    # 히트미스 커널 (L자 모서리 찾기)
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ], dtype=np.int8)

    # 히트미스 변환
    hitmiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

    print("히트미스 변환:")
    print("  - 특정 패턴 찾기")
    print("  - 1: 전경, 0: 배경, -1: 무관")

    cv2.imwrite('hitmiss_input.jpg', img)
    cv2.imwrite('hitmiss_result.jpg', hitmiss)


def practical_example():
    """실용 예제: 텍스트 정리"""
    print("\n" + "=" * 50)
    print("실용 예제: 텍스트 정리")
    print("=" * 50)

    # 노이즈 있는 텍스트 이미지 생성
    img = np.zeros((100, 400), dtype=np.uint8)
    cv2.putText(img, 'OpenCV', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)

    # 노이즈 추가
    noise = np.random.random(img.shape) < 0.05
    img[noise] = 255

    # 열기로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 닫기로 글자 보완
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    enhanced = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel2)

    cv2.imwrite('text_noisy.jpg', img)
    cv2.imwrite('text_cleaned.jpg', cleaned)
    cv2.imwrite('text_enhanced.jpg', enhanced)
    print("텍스트 정리 이미지 저장 완료")


def main():
    """메인 함수"""
    # 구조 요소
    structuring_element_demo()

    # 침식
    erosion_demo()

    # 팽창
    dilation_demo()

    # 열기
    opening_demo()

    # 닫기
    closing_demo()

    # 그래디언트
    gradient_demo()

    # 탑햇/블랙햇
    tophat_blackhat_demo()

    # 히트미스
    hit_miss_demo()

    # 실용 예제
    practical_example()

    print("\n모폴로지 연산 데모 완료!")


if __name__ == '__main__':
    main()
