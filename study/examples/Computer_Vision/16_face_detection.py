"""
16. 얼굴 검출 및 인식
- Haar Cascade 얼굴 검출
- 눈, 미소 검출
- LBP (Local Binary Patterns)
- 얼굴 인식 기초
"""

import cv2
import numpy as np


def create_face_image():
    """얼굴 형태 시뮬레이션 이미지"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = [220, 220, 220]

    # 얼굴 1 (왼쪽)
    cv2.ellipse(img, (150, 180), (60, 75), 0, 0, 360, (180, 160, 140), -1)
    cv2.circle(img, (130, 160), 10, (50, 50, 50), -1)  # 왼쪽 눈
    cv2.circle(img, (170, 160), 10, (50, 50, 50), -1)  # 오른쪽 눈
    cv2.ellipse(img, (150, 210), (20, 10), 0, 0, 180, (100, 80, 80), 2)  # 입

    # 얼굴 2 (오른쪽)
    cv2.ellipse(img, (350, 200), (55, 70), 0, 0, 360, (175, 155, 135), -1)
    cv2.circle(img, (332, 180), 9, (45, 45, 45), -1)  # 왼쪽 눈
    cv2.circle(img, (368, 180), 9, (45, 45, 45), -1)  # 오른쪽 눈
    cv2.ellipse(img, (350, 225), (18, 8), 0, 0, 180, (90, 70, 70), 2)  # 입

    return img


def haar_cascade_face_detection():
    """Haar Cascade 얼굴 검출 데모"""
    print("=" * 50)
    print("Haar Cascade 얼굴 검출")
    print("=" * 50)

    img = create_face_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Haar Cascade 분류기 로드
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print("얼굴 검출기를 로드할 수 없습니다.")
        return

    # 얼굴 검출
    # scaleFactor: 각 이미지 스케일에서 이미지 크기를 줄이는 비율
    # minNeighbors: 각 후보 사각형이 유지되기 위해 필요한 이웃 수
    # minSize: 최소 객체 크기
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    result = img.copy()
    print(f"검출된 얼굴 수: {len(faces)}")

    for i, (x, y, w, h) in enumerate(faces):
        # 얼굴 영역 표시
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result, f'Face {i+1}', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  Face {i+1}: x={x}, y={y}, w={w}, h={h}")

    print("\nHaar Cascade 파라미터:")
    print("  scaleFactor: 1.1~1.3 (작을수록 정밀, 느림)")
    print("  minNeighbors: 3~6 (클수록 엄격)")
    print("  minSize: 최소 검출 크기")

    cv2.imwrite('face_haar_input.jpg', img)
    cv2.imwrite('face_haar_result.jpg', result)


def cascade_eye_detection():
    """눈 검출 데모"""
    print("\n" + "=" * 50)
    print("눈 검출 (Haar Cascade)")
    print("=" * 50)

    img = create_face_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 분류기 로드
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

    if face_cascade.empty() or eye_cascade.empty():
        print("분류기를 로드할 수 없습니다.")
        return

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    result = img.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 얼굴 영역 내에서 눈 검출
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = result[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        print(f"얼굴 ({x}, {y})에서 검출된 눈: {len(eyes)}")

    print("\n눈 검출 팁:")
    print("  - 얼굴 영역 내에서만 검출 (ROI)")
    print("  - minNeighbors를 높게 설정")
    print("  - 상반부에서만 검출하면 정확도 향상")

    cv2.imwrite('face_eye_result.jpg', result)


def available_cascades():
    """사용 가능한 Haar Cascade 목록"""
    print("\n" + "=" * 50)
    print("사용 가능한 Haar Cascade 분류기")
    print("=" * 50)

    cascades = [
        ('haarcascade_frontalface_default.xml', '정면 얼굴'),
        ('haarcascade_frontalface_alt.xml', '정면 얼굴 (대체)'),
        ('haarcascade_frontalface_alt2.xml', '정면 얼굴 (대체 2)'),
        ('haarcascade_profileface.xml', '측면 얼굴'),
        ('haarcascade_eye.xml', '눈'),
        ('haarcascade_eye_tree_eyeglasses.xml', '눈 (안경 포함)'),
        ('haarcascade_smile.xml', '미소'),
        ('haarcascade_upperbody.xml', '상체'),
        ('haarcascade_lowerbody.xml', '하체'),
        ('haarcascade_fullbody.xml', '전신'),
    ]

    print(f"\n경로: {cv2.data.haarcascades}\n")

    for filename, description in cascades:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
        status = "OK" if not cascade.empty() else "N/A"
        print(f"  [{status}] {filename}")
        print(f"       - {description}")


def lbp_face_detection():
    """LBP 기반 얼굴 검출 데모"""
    print("\n" + "=" * 50)
    print("LBP 얼굴 검출")
    print("=" * 50)

    img = create_face_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LBP Cascade 로드 (있는 경우)
    lbp_cascade_path = cv2.data.haarcascades + '../lbpcascades/lbpcascade_frontalface_improved.xml'

    try:
        lbp_cascade = cv2.CascadeClassifier(lbp_cascade_path)

        if lbp_cascade.empty():
            raise FileNotFoundError

        faces = lbp_cascade.detectMultiScale(gray, 1.1, 5)

        result = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

        print(f"LBP로 검출된 얼굴: {len(faces)}")
        cv2.imwrite('face_lbp_result.jpg', result)

    except (FileNotFoundError, cv2.error):
        print("LBP Cascade를 찾을 수 없습니다.")
        print("대부분의 경우 Haar Cascade 사용을 권장합니다.")

    print("\nHaar vs LBP 비교:")
    print("  Haar: 더 정확, 느림, 조명 변화에 민감")
    print("  LBP: 더 빠름, 조명 변화에 강건, 정확도 낮음")


def face_recognition_concept():
    """얼굴 인식 개념 설명"""
    print("\n" + "=" * 50)
    print("얼굴 인식 개념")
    print("=" * 50)

    # 테스트 이미지 생성
    img1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.ellipse(img1, (50, 50), (30, 40), 0, 0, 360, 150, -1)
    cv2.circle(img1, (40, 45), 5, 50, -1)
    cv2.circle(img1, (60, 45), 5, 50, -1)

    img2 = img1.copy()  # 동일인
    img3 = np.zeros((100, 100), dtype=np.uint8)  # 다른 사람
    cv2.ellipse(img3, (50, 50), (35, 35), 0, 0, 360, 160, -1)
    cv2.circle(img3, (35, 45), 6, 60, -1)
    cv2.circle(img3, (65, 45), 6, 60, -1)

    print("얼굴 인식 파이프라인:")
    print("  1. 얼굴 검출 (Detection)")
    print("  2. 얼굴 정렬 (Alignment)")
    print("  3. 특징 추출 (Feature Extraction)")
    print("  4. 특징 비교 (Matching)")

    print("\nOpenCV 얼굴 인식기 (opencv-contrib 필요):")
    print("  - EigenFaces: PCA 기반")
    print("  - FisherFaces: LDA 기반")
    print("  - LBPH: Local Binary Pattern Histogram")

    # LBPH 얼굴 인식기 예시 (opencv-contrib 필요)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # 학습 데이터
        faces = [img1, img2, img3]
        labels = np.array([0, 0, 1])  # 0: 첫 번째 사람, 1: 두 번째 사람

        recognizer.train(faces, labels)

        # 예측
        label, confidence = recognizer.predict(img1)
        print(f"\n테스트: label={label}, confidence={confidence:.2f}")
        print("  confidence가 낮을수록 유사")

    except AttributeError:
        print("\n참고: LBPH 인식기를 사용하려면")
        print("  pip install opencv-contrib-python")

    cv2.imwrite('face_sample1.jpg', img1)
    cv2.imwrite('face_sample2.jpg', img3)


def face_detection_comparison():
    """얼굴 검출 방법 비교"""
    print("\n" + "=" * 50)
    print("얼굴 검출/인식 방법 비교")
    print("=" * 50)

    print("""
    | 방법 | 장점 | 단점 | 용도 |
    |------|------|------|------|
    | Haar Cascade | 빠름, 간단 | 측면/기울기 약함 | 실시간 검출 |
    | LBP | 매우 빠름 | 정확도 낮음 | 임베디드 |
    | HOG + SVM | 정확 | 느림 | 검출 |
    | DNN (SSD) | 매우 정확 | GPU 권장 | 고정밀 검출 |
    | DNN (Face) | 특징 추출 | 모델 필요 | 인식 |
    """)

    print("최신 트렌드:")
    print("  - MTCNN: 다단계 CNN (검출+정렬)")
    print("  - RetinaFace: 고정밀 검출")
    print("  - ArcFace, FaceNet: 임베딩 기반 인식")
    print("  - InsightFace: 종합 프레임워크")


def real_time_detection_template():
    """실시간 검출 템플릿"""
    print("\n" + "=" * 50)
    print("실시간 얼굴 검출 템플릿")
    print("=" * 50)

    code = '''
# 실시간 얼굴 검출 코드
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

    print(code)

    print("성능 최적화 팁:")
    print("  1. 프레임 스킵 (매 프레임 검출 불필요)")
    print("  2. 이미지 축소 후 검출")
    print("  3. 이전 검출 영역 주변만 탐색")
    print("  4. 멀티스레딩 활용")


def main():
    """메인 함수"""
    # Haar Cascade 얼굴 검출
    haar_cascade_face_detection()

    # 눈 검출
    cascade_eye_detection()

    # 사용 가능한 Cascade 목록
    available_cascades()

    # LBP 검출
    lbp_face_detection()

    # 얼굴 인식 개념
    face_recognition_concept()

    # 방법 비교
    face_detection_comparison()

    # 실시간 템플릿
    real_time_detection_template()

    print("\n얼굴 검출 및 인식 데모 완료!")


if __name__ == '__main__':
    main()
