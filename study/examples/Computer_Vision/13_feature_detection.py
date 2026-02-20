"""
13. 특징점 검출
- Harris 코너 검출
- FAST 특징점
- SIFT, ORB
- 키포인트와 디스크립터
"""

import cv2
import numpy as np


def create_test_image():
    """코너가 있는 테스트 이미지"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # 사각형 (명확한 코너)
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
    cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)

    # 다른 사각형
    cv2.rectangle(img, (200, 80), (350, 180), (50, 50, 50), -1)

    # 체커보드 패턴 (많은 코너)
    for i in range(4):
        for j in range(4):
            x = 50 + i * 40
            y = 220 + j * 40
            if (i + j) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + 40, y + 40), (0, 0, 0), -1)

    # 원 (코너 없음)
    cv2.circle(img, (400, 100), 50, (80, 80, 80), -1)

    # 텍스트
    cv2.putText(img, 'FEATURES', (280, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img


def harris_corner_demo():
    """Harris 코너 검출 데모"""
    print("=" * 50)
    print("Harris 코너 검출")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris 코너 검출
    # blockSize: 코너 검출 윈도우 크기
    # ksize: Sobel 커널 크기
    # k: Harris 파라미터 (0.04~0.06)
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # 결과 확장 (시각화용)
    harris_dilated = cv2.dilate(harris, None)

    # 임계값 적용
    threshold = 0.01 * harris_dilated.max()
    result = img.copy()
    result[harris_dilated > threshold] = [0, 0, 255]  # 빨간색으로 표시

    # 정밀 코너 위치 (SubPixel)
    _, harris_binary = cv2.threshold(harris_dilated, threshold, 255, cv2.THRESH_BINARY)
    harris_binary = np.uint8(harris_binary)

    # 연결된 컴포넌트로 코너 개수 세기
    num_corners = cv2.connectedComponents(harris_binary)[0] - 1

    print(f"검출된 코너 수: {num_corners}")
    print("\nHarris 코너 특성:")
    print("  - 회전 불변")
    print("  - 크기 변화에는 민감")
    print("  - 코너 응답 함수 R = det(M) - k*trace(M)^2")

    cv2.imwrite('harris_input.jpg', img)
    cv2.imwrite('harris_result.jpg', result)


def shi_tomasi_demo():
    """Shi-Tomasi 코너 검출 데모"""
    print("\n" + "=" * 50)
    print("Shi-Tomasi 코너 검출 (goodFeaturesToTrack)")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi 코너 검출
    # maxCorners: 최대 코너 수
    # qualityLevel: 품질 수준 (0~1)
    # minDistance: 코너 간 최소 거리
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10
    )

    result = img.copy()

    if corners is not None:
        corners = np.int32(corners)
        print(f"검출된 코너 수: {len(corners)}")

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)

    print("\nShi-Tomasi vs Harris:")
    print("  - R = min(λ1, λ2) 사용")
    print("  - Harris보다 안정적")
    print("  - 추적에 적합한 코너 선택")

    cv2.imwrite('shi_tomasi_result.jpg', result)


def fast_demo():
    """FAST 특징점 검출 데모"""
    print("\n" + "=" * 50)
    print("FAST 특징점 검출")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FAST 검출기 생성
    # threshold: 밝기 차이 임계값
    # nonmaxSuppression: 비최대 억제
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

    # 키포인트 검출
    keypoints = fast.detect(gray, None)

    # 결과 그리기
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    print(f"검출된 키포인트 수: {len(keypoints)}")
    print(f"Threshold: {fast.getThreshold()}")
    print(f"NonMax Suppression: {fast.getNonmaxSuppression()}")

    print("\nFAST 특성:")
    print("  - 매우 빠른 속도")
    print("  - 원형 패턴으로 코너 검출")
    print("  - 디스크립터 없음 (검출만)")

    cv2.imwrite('fast_result.jpg', result)


def orb_demo():
    """ORB 특징점 검출 데모"""
    print("\n" + "=" * 50)
    print("ORB (Oriented FAST and Rotated BRIEF)")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB 검출기 생성
    orb = cv2.ORB_create(nfeatures=500)

    # 키포인트와 디스크립터 계산
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 결과 그리기
    result = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"검출된 키포인트 수: {len(keypoints)}")
    if descriptors is not None:
        print(f"디스크립터 shape: {descriptors.shape}")
        print(f"디스크립터 타입: {descriptors.dtype}")

    print("\nORB 특성:")
    print("  - FAST 기반 키포인트 검출")
    print("  - BRIEF 기반 디스크립터")
    print("  - 회전 불변성 추가")
    print("  - 특허 없음, 빠름")

    cv2.imwrite('orb_result.jpg', result)

    return keypoints, descriptors


def sift_demo():
    """SIFT 특징점 검출 데모"""
    print("\n" + "=" * 50)
    print("SIFT (Scale-Invariant Feature Transform)")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        # SIFT 검출기 생성
        sift = cv2.SIFT_create()

        # 키포인트와 디스크립터 계산
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # 결과 그리기
        result = cv2.drawKeypoints(
            img, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        print(f"검출된 키포인트 수: {len(keypoints)}")
        if descriptors is not None:
            print(f"디스크립터 shape: {descriptors.shape}")
            print(f"디스크립터 타입: {descriptors.dtype}")

        print("\nSIFT 특성:")
        print("  - 크기 불변 (DoG 피라미드)")
        print("  - 회전 불변 (방향 할당)")
        print("  - 128차원 디스크립터")
        print("  - opencv-contrib-python 필요")

        cv2.imwrite('sift_result.jpg', result)

    except AttributeError:
        print("SIFT를 사용하려면 opencv-contrib-python이 필요합니다.")
        print("pip install opencv-contrib-python")


def keypoint_info_demo():
    """키포인트 정보 데모"""
    print("\n" + "=" * 50)
    print("키포인트 정보")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=10)
    keypoints, _ = orb.detectAndCompute(gray, None)

    print("키포인트 속성:")
    for i, kp in enumerate(keypoints[:5]):
        print(f"\n  키포인트 {i}:")
        print(f"    위치 (pt): ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
        print(f"    크기 (size): {kp.size:.1f}")
        print(f"    각도 (angle): {kp.angle:.1f}")
        print(f"    응답 (response): {kp.response:.4f}")
        print(f"    옥타브 (octave): {kp.octave}")


def compare_detectors():
    """특징점 검출기 비교"""
    print("\n" + "=" * 50)
    print("특징점 검출기 비교")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detectors = []

    # FAST
    fast = cv2.FastFeatureDetector_create()
    kp_fast = fast.detect(gray, None)
    detectors.append(('FAST', len(kp_fast)))

    # ORB
    orb = cv2.ORB_create()
    kp_orb, _ = orb.detectAndCompute(gray, None)
    detectors.append(('ORB', len(kp_orb)))

    # SIFT (가능한 경우)
    try:
        sift = cv2.SIFT_create()
        kp_sift, _ = sift.detectAndCompute(gray, None)
        detectors.append(('SIFT', len(kp_sift)))
    except AttributeError:
        detectors.append(('SIFT', 'N/A'))

    # Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
    detectors.append(('Shi-Tomasi', len(corners) if corners is not None else 0))

    print("\n| 검출기 | 키포인트 수 | 특징 |")
    print("|--------|-----------|------|")
    print(f"| FAST | {detectors[0][1]} | 빠름, 디스크립터 없음 |")
    print(f"| ORB | {detectors[1][1]} | 빠름, 특허 무료 |")
    print(f"| SIFT | {detectors[2][1]} | 정확, 느림 |")
    print(f"| Shi-Tomasi | {detectors[3][1]} | 추적용 |")


def main():
    """메인 함수"""
    # Harris 코너
    harris_corner_demo()

    # Shi-Tomasi
    shi_tomasi_demo()

    # FAST
    fast_demo()

    # ORB
    orb_demo()

    # SIFT
    sift_demo()

    # 키포인트 정보
    keypoint_info_demo()

    # 비교
    compare_detectors()

    print("\n특징점 검출 데모 완료!")


if __name__ == '__main__':
    main()
