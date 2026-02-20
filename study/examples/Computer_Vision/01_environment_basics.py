"""
01. 환경 설정 및 기초
- OpenCV 설치 확인
- 버전 정보
- 기본 테스트 이미지 생성
"""

import cv2
import numpy as np
import sys


def check_opencv_installation():
    """OpenCV 설치 및 버전 확인"""
    print("=" * 50)
    print("OpenCV 환경 확인")
    print("=" * 50)

    # 버전 확인
    print(f"OpenCV 버전: {cv2.__version__}")
    print(f"Python 버전: {sys.version}")
    print(f"NumPy 버전: {np.__version__}")

    # 빌드 정보
    print("\n[빌드 정보]")
    build_info = cv2.getBuildInformation()
    # 주요 정보만 출력
    for line in build_info.split('\n')[:20]:
        if line.strip():
            print(line)


def check_available_modules():
    """사용 가능한 모듈 확인"""
    print("\n" + "=" * 50)
    print("사용 가능한 주요 기능")
    print("=" * 50)

    # SIFT 확인 (contrib 필요)
    try:
        sift = cv2.SIFT_create()
        print("SIFT: 사용 가능")
    except AttributeError:
        print("SIFT: 사용 불가 (opencv-contrib-python 필요)")

    # ORB 확인
    try:
        orb = cv2.ORB_create()
        print("ORB: 사용 가능")
    except AttributeError:
        print("ORB: 사용 불가")

    # DNN 확인
    try:
        net = cv2.dnn.readNet
        print("DNN 모듈: 사용 가능")
    except AttributeError:
        print("DNN 모듈: 사용 불가")

    # Haar Cascade 확인
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Haar Cascade 경로: {cv2.data.haarcascades}")


def create_test_image():
    """테스트용 이미지 생성"""
    print("\n" + "=" * 50)
    print("테스트 이미지 생성")
    print("=" * 50)

    # 컬러 이미지 생성 (400x400, BGR)
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # 배경 그라데이션
    for i in range(400):
        img[i, :] = [i * 255 // 400, 100, 255 - i * 255 // 400]

    # 도형 그리기
    # 사각형
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), 2)
    cv2.rectangle(img, (60, 60), (140, 140), (0, 255, 0), -1)  # 채움

    # 원
    cv2.circle(img, (300, 100), 50, (255, 0, 0), 2)
    cv2.circle(img, (300, 100), 30, (255, 0, 0), -1)

    # 선
    cv2.line(img, (50, 250), (350, 250), (0, 0, 255), 3)
    cv2.line(img, (50, 280), (350, 320), (255, 255, 0), 2)

    # 타원
    cv2.ellipse(img, (200, 350), (100, 30), 0, 0, 360, (255, 0, 255), 2)

    # 텍스트
    cv2.putText(img, 'OpenCV Test', (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 저장
    cv2.imwrite('sample.jpg', img)
    print("sample.jpg 생성 완료")

    # 그레이스케일 이미지도 생성
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('sample_gray.jpg', gray)
    print("sample_gray.jpg 생성 완료")

    return img


def basic_operations_demo(img):
    """기본 연산 데모"""
    print("\n" + "=" * 50)
    print("기본 연산 데모")
    print("=" * 50)

    # 이미지 속성
    print(f"이미지 shape: {img.shape}")
    print(f"이미지 dtype: {img.dtype}")
    print(f"이미지 size: {img.size}")

    # 픽셀 접근
    pixel = img[100, 100]
    print(f"픽셀 (100, 100) BGR 값: {pixel}")

    # ROI (Region of Interest)
    roi = img[50:150, 50:150]
    print(f"ROI shape: {roi.shape}")


def main():
    """메인 함수"""
    # 환경 확인
    check_opencv_installation()
    check_available_modules()

    # 테스트 이미지 생성
    img = create_test_image()

    # 기본 연산 데모
    basic_operations_demo(img)

    # 이미지 표시 (GUI 환경에서)
    print("\n[이미지 표시]")
    print("이미지 창을 닫으려면 아무 키나 누르세요...")

    try:
        cv2.imshow('Test Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"GUI 표시 불가: {e}")
        print("Headless 환경에서는 cv2.imshow() 사용 불가")

    print("\n환경 설정 확인 완료!")


if __name__ == '__main__':
    main()
