"""
04. 기하학적 변환
- resize, rotate, flip
- 어파인 변환 (warpAffine)
- 원근 변환 (warpPerspective)
- 이동, 회전, 스케일링
"""

import cv2
import numpy as np


def create_test_image():
    """테스트 이미지 생성"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]  # 밝은 회색 배경

    # 사각형
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)

    # 텍스트
    cv2.putText(img, 'OpenCV', (180, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 원
    cv2.circle(img, (300, 200), 50, (255, 0, 0), -1)

    return img


def resize_demo():
    """크기 조정 데모"""
    print("=" * 50)
    print("크기 조정 (resize)")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]
    print(f"원본 크기: {w}x{h}")

    # 절대 크기로 조정
    resized1 = cv2.resize(img, (200, 150))  # (width, height)
    print(f"절대 크기 조정: {resized1.shape[1]}x{resized1.shape[0]}")

    # 비율로 조정
    resized2 = cv2.resize(img, None, fx=0.5, fy=0.5)
    print(f"50% 축소: {resized2.shape[1]}x{resized2.shape[0]}")

    resized3 = cv2.resize(img, None, fx=2, fy=2)
    print(f"200% 확대: {resized3.shape[1]}x{resized3.shape[0]}")

    # 보간법 비교
    print("\n보간법 종류:")
    print("  cv2.INTER_NEAREST: 최근접 이웃 (빠름, 품질 낮음)")
    print("  cv2.INTER_LINEAR: 양선형 (기본값, 균형)")
    print("  cv2.INTER_CUBIC: 바이큐빅 (품질 좋음, 느림)")
    print("  cv2.INTER_AREA: 영역 (축소에 적합)")

    # 각 보간법 적용
    enlarged_nearest = cv2.resize(img, (800, 600), interpolation=cv2.INTER_NEAREST)
    enlarged_linear = cv2.resize(img, (800, 600), interpolation=cv2.INTER_LINEAR)
    enlarged_cubic = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite('resize_half.jpg', resized2)
    cv2.imwrite('resize_double.jpg', resized3)
    cv2.imwrite('resize_nearest.jpg', enlarged_nearest)
    cv2.imwrite('resize_cubic.jpg', enlarged_cubic)
    print("\n크기 조정 이미지 저장 완료")


def rotate_demo():
    """회전 데모"""
    print("\n" + "=" * 50)
    print("회전 (rotate)")
    print("=" * 50)

    img = create_test_image()

    # 간단한 90도 회전 (내장 함수)
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    print("간단한 회전:")
    print(f"  90도 시계방향: {rotated_90.shape[1]}x{rotated_90.shape[0]}")
    print(f"  180도: {rotated_180.shape[1]}x{rotated_180.shape[0]}")
    print(f"  270도(반시계): {rotated_270.shape[1]}x{rotated_270.shape[0]}")

    cv2.imwrite('rotate_90.jpg', rotated_90)
    cv2.imwrite('rotate_180.jpg', rotated_180)
    cv2.imwrite('rotate_270.jpg', rotated_270)

    # 임의 각도 회전 (getRotationMatrix2D 사용)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 45도 회전, 스케일 1.0
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_45 = cv2.warpAffine(img, M, (w, h))

    # 30도 회전, 스케일 0.8
    M = cv2.getRotationMatrix2D(center, 30, 0.8)
    rotated_30_scaled = cv2.warpAffine(img, M, (w, h))

    cv2.imwrite('rotate_45.jpg', rotated_45)
    cv2.imwrite('rotate_30_scaled.jpg', rotated_30_scaled)
    print("\n회전 이미지 저장 완료")


def flip_demo():
    """뒤집기 데모"""
    print("\n" + "=" * 50)
    print("뒤집기 (flip)")
    print("=" * 50)

    img = create_test_image()

    # 수평 뒤집기 (좌우)
    flipped_h = cv2.flip(img, 1)

    # 수직 뒤집기 (상하)
    flipped_v = cv2.flip(img, 0)

    # 둘 다 (상하좌우)
    flipped_both = cv2.flip(img, -1)

    print("뒤집기 코드:")
    print("  flip(img, 1): 수평 (좌우)")
    print("  flip(img, 0): 수직 (상하)")
    print("  flip(img, -1): 상하좌우")

    cv2.imwrite('flip_horizontal.jpg', flipped_h)
    cv2.imwrite('flip_vertical.jpg', flipped_v)
    cv2.imwrite('flip_both.jpg', flipped_both)
    print("\n뒤집기 이미지 저장 완료")


def translation_demo():
    """이동 데모"""
    print("\n" + "=" * 50)
    print("이동 (Translation)")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]

    # 이동 행렬: [[1, 0, tx], [0, 1, ty]]
    # tx: x축 이동 (양수: 오른쪽)
    # ty: y축 이동 (양수: 아래)

    tx, ty = 50, 30
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, M, (w, h))

    print(f"이동: x={tx}, y={ty}")
    print(f"이동 행렬:\n{M}")

    cv2.imwrite('translated.jpg', translated)
    print("\n이동 이미지 저장 완료")


def affine_transform_demo():
    """어파인 변환 데모"""
    print("\n" + "=" * 50)
    print("어파인 변환 (Affine Transform)")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]

    # 어파인 변환: 3개의 점 대응
    # 원본 이미지의 3점
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    # 변환 후 3점
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    # 변환 행렬 계산
    M = cv2.getAffineTransform(pts1, pts2)

    # 변환 적용
    affine = cv2.warpAffine(img, M, (w, h))

    print("어파인 변환 특성:")
    print("  - 평행선은 평행 유지")
    print("  - 3점 대응으로 정의")
    print("  - 이동, 회전, 스케일, 전단(shear) 조합")

    cv2.imwrite('affine.jpg', affine)
    print("\n어파인 변환 이미지 저장 완료")


def perspective_transform_demo():
    """원근 변환 데모"""
    print("\n" + "=" * 50)
    print("원근 변환 (Perspective Transform)")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]

    # 원근 변환: 4개의 점 대응
    # 원본 이미지의 4점 (사각형 모서리)
    pts1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])

    # 변환 후 4점 (원근감 적용)
    pts2 = np.float32([[50, 50], [w-50, 20], [30, h-30], [w-30, h-50]])

    # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # 변환 적용
    perspective = cv2.warpPerspective(img, M, (w, h))

    print("원근 변환 특성:")
    print("  - 평행선이 한 점에서 만남 (원근감)")
    print("  - 4점 대응으로 정의")
    print("  - 문서 스캔 보정에 활용")

    cv2.imwrite('perspective.jpg', perspective)

    # 역변환 (문서 보정 시뮬레이션)
    pts_doc = np.float32([[50, 50], [350, 30], [60, 280], [340, 270]])
    pts_rect = np.float32([[0, 0], [300, 0], [0, 200], [300, 200]])

    M_rect = cv2.getPerspectiveTransform(pts_doc, pts_rect)
    rectified = cv2.warpPerspective(img, M_rect, (300, 200))

    cv2.imwrite('perspective_rectified.jpg', rectified)
    print("원근 변환 이미지 저장 완료")


def combined_transforms_demo():
    """복합 변환 데모"""
    print("\n" + "=" * 50)
    print("복합 변환")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]

    # 이동 -> 회전 -> 스케일을 한 번에
    center = (w // 2, h // 2)
    angle = 30
    scale = 0.8

    # 회전 + 스케일 행렬
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 추가 이동 적용
    M[0, 2] += 50  # x 이동
    M[1, 2] += 20  # y 이동

    result = cv2.warpAffine(img, M, (w, h))

    print(f"복합 변환: 회전 {angle}도, 스케일 {scale}, 이동 (50, 20)")

    cv2.imwrite('combined_transform.jpg', result)
    print("복합 변환 이미지 저장 완료")


def main():
    """메인 함수"""
    # 크기 조정
    resize_demo()

    # 회전
    rotate_demo()

    # 뒤집기
    flip_demo()

    # 이동
    translation_demo()

    # 어파인 변환
    affine_transform_demo()

    # 원근 변환
    perspective_transform_demo()

    # 복합 변환
    combined_transforms_demo()

    print("\n기하학적 변환 데모 완료!")


if __name__ == '__main__':
    main()
