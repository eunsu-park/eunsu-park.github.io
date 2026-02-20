"""
20. 실전 프로젝트
- 문서 스캐너
- 차량 번호판 인식
- 실시간 객체 추적
- 이미지 파노라마
"""

import cv2
import numpy as np


# ============================================================
# 프로젝트 1: 문서 스캐너
# ============================================================

def document_scanner():
    """문서 스캐너 프로젝트"""
    print("=" * 60)
    print("프로젝트 1: 문서 스캐너")
    print("=" * 60)

    # 시뮬레이션용 문서 이미지 생성
    # 실제로는 카메라로 촬영한 이미지 사용
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = [150, 150, 150]  # 회색 배경

    # 기울어진 문서 (사다리꼴)
    doc_pts = np.array([[150, 100], [650, 80], [700, 520], [100, 550]], np.int32)
    cv2.fillPoly(img, [doc_pts], (255, 255, 255))

    # 문서 내용 시뮬레이션
    cv2.putText(img, 'DOCUMENT TITLE', (220, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.line(img, (200, 250), (600, 240), (100, 100, 100), 2)
    cv2.line(img, (200, 300), (600, 290), (100, 100, 100), 2)
    cv2.line(img, (200, 350), (550, 340), (100, 100, 100), 2)

    cv2.imwrite('scanner_input.jpg', img)

    # 1. 그레이스케일 및 블러
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    # 3. 컨투어 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 가장 큰 사각형 컨투어 찾기
    doc_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:  # 최소 크기
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > max_area:
                doc_contour = approx
                max_area = area

    if doc_contour is not None:
        # 코너 정렬
        pts = doc_contour.reshape(4, 2)
        rect = order_points(pts)

        # 결과 이미지에 표시
        result_contour = img.copy()
        cv2.drawContours(result_contour, [doc_contour], -1, (0, 255, 0), 3)
        for pt in rect:
            cv2.circle(result_contour, tuple(pt.astype(int)), 10, (0, 0, 255), -1)

        cv2.imwrite('scanner_contour.jpg', result_contour)

        # 5. 원근 변환
        width, height = 500, 700  # 출력 크기
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (width, height))

        # 6. 이진화 (스캔 효과)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        scanned = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        cv2.imwrite('scanner_warped.jpg', warped)
        cv2.imwrite('scanner_result.jpg', scanned)

        print("문서 스캐너 완료!")
        print("  - scanner_input.jpg: 원본")
        print("  - scanner_contour.jpg: 문서 검출")
        print("  - scanner_warped.jpg: 원근 보정")
        print("  - scanner_result.jpg: 최종 스캔")
    else:
        print("문서를 찾을 수 없습니다.")

    print("\n처리 파이프라인:")
    print("  1. 그레이스케일 + 블러")
    print("  2. Canny 엣지 검출")
    print("  3. 컨투어 검출 및 근사화")
    print("  4. 4각형 문서 선택")
    print("  5. 원근 변환")
    print("  6. 이진화 (선택)")


def order_points(pts):
    """코너 점을 [좌상, 우상, 우하, 좌하] 순서로 정렬"""
    rect = np.zeros((4, 2), dtype=np.float32)

    # 합이 가장 작은 것: 좌상
    # 합이 가장 큰 것: 우하
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 차이가 가장 작은 것: 우상
    # 차이가 가장 큰 것: 좌하
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# ============================================================
# 프로젝트 2: 차량 번호판 인식 (개념)
# ============================================================

def license_plate_recognition():
    """차량 번호판 인식 프로젝트 (개념)"""
    print("\n" + "=" * 60)
    print("프로젝트 2: 차량 번호판 인식")
    print("=" * 60)

    # 시뮬레이션용 번호판 이미지
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # 차량 형태
    cv2.rectangle(img, (100, 100), (500, 350), (80, 80, 80), -1)
    cv2.rectangle(img, (120, 120), (480, 250), (60, 60, 60), -1)

    # 번호판
    plate_x, plate_y = 200, 280
    plate_w, plate_h = 200, 50
    cv2.rectangle(img, (plate_x, plate_y), (plate_x+plate_w, plate_y+plate_h),
                 (255, 255, 255), -1)
    cv2.rectangle(img, (plate_x, plate_y), (plate_x+plate_w, plate_y+plate_h),
                 (0, 0, 0), 2)
    cv2.putText(img, '12AB3456', (plate_x+20, plate_y+35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imwrite('lpr_input.jpg', img)

    print("\n번호판 인식 파이프라인:")
    print("""
1. 번호판 검출 (Plate Detection)
   - Haar Cascade (학습된 분류기)
   - DNN (YOLO, SSD)
   - 엣지 기반 검출

2. 번호판 영역 추출
   - 컨투어 검출
   - 원근 보정

3. 문자 분할 (Character Segmentation)
   - 이진화
   - 컨투어로 각 문자 분리
   - 연결 요소 분석

4. 문자 인식 (OCR)
   - Tesseract OCR
   - DNN 기반 인식
   - 템플릿 매칭

5. 후처리
   - 형식 검증
   - 노이즈 제거
""")

    code = '''
# 번호판 인식 코드 예시
import cv2
import pytesseract

# 1. 번호판 검출
plate_cascade = cv2.CascadeClassifier('haarcascade_plate.xml')
plates = plate_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in plates:
    # 2. 번호판 영역 추출
    plate_img = gray[y:y+h, x:x+w]

    # 3. 전처리
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2)
    _, thresh = cv2.threshold(plate_img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. OCR
    text = pytesseract.image_to_string(thresh, config='--psm 7')
    print(f"번호판: {text.strip()}")
'''
    print(code)

    print("\n필요 라이브러리:")
    print("  - pytesseract: pip install pytesseract")
    print("  - Tesseract-OCR: 시스템 설치 필요")


# ============================================================
# 프로젝트 3: 실시간 객체 추적
# ============================================================

def object_tracking_project():
    """실시간 객체 추적 프로젝트"""
    print("\n" + "=" * 60)
    print("프로젝트 3: 실시간 객체 추적")
    print("=" * 60)

    # 시뮬레이션 프레임 시퀀스 생성
    frames = []
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = [50, 50, 50]

        # 움직이는 객체
        x = 100 + i * 15
        y = 240 + int(50 * np.sin(i * 0.3))
        cv2.circle(frame, (x, y), 40, (0, 200, 0), -1)

        # 고정 객체
        cv2.rectangle(frame, (400, 100), (500, 200), (200, 0, 0), -1)

        frames.append(frame)

    # 첫 프레임에서 추적 대상 선택
    first_frame = frames[0].copy()
    bbox = (60, 200, 80, 80)  # x, y, w, h
    cv2.rectangle(first_frame, (bbox[0], bbox[1]),
                 (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    cv2.imwrite('tracking_init.jpg', first_frame)

    # 추적 시뮬레이션
    print("\n추적 시뮬레이션 (KCF Tracker 개념)")

    # 추적 결과 시각화
    result_frame = frames[15].copy()
    new_x = 100 + 15 * 15
    new_y = 240 + int(50 * np.sin(15 * 0.3))
    cv2.rectangle(result_frame, (new_x-40, new_y-40),
                 (new_x+40, new_y+40), (0, 255, 0), 2)
    cv2.putText(result_frame, 'Tracking', (new_x-30, new_y-50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite('tracking_result.jpg', result_frame)

    print("\n완성된 추적 코드:")
    code = '''
import cv2

# 비디오 캡처
cap = cv2.VideoCapture(0)  # 또는 'video.mp4'

# 첫 프레임 읽기
ret, frame = cap.read()

# ROI 선택 (마우스로 드래그)
bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
cv2.destroyAllWindows()

# 추적기 생성 (여러 옵션)
# tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMIL_create()
tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerCSRT_create()  # 더 정확

# 초기화
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 추적 업데이트
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
    print(code)

    print("\n추적기 비교:")
    trackers = [
        ('KCF', '빠름, 일반적 성능'),
        ('CSRT', '정확, 다소 느림'),
        ('MOSSE', '매우 빠름, 낮은 정확도'),
        ('MedianFlow', '예측 가능한 움직임'),
    ]
    for name, desc in trackers:
        print(f"  {name}: {desc}")


# ============================================================
# 프로젝트 4: 이미지 파노라마
# ============================================================

def panorama_stitching():
    """이미지 파노라마 프로젝트"""
    print("\n" + "=" * 60)
    print("프로젝트 4: 이미지 파노라마")
    print("=" * 60)

    # 시뮬레이션용 겹치는 이미지 생성
    # 배경
    full_scene = np.zeros((300, 800, 3), dtype=np.uint8)
    full_scene[:] = [200, 200, 200]

    # 장면에 객체 배치
    cv2.circle(full_scene, (100, 150), 50, (0, 0, 150), -1)
    cv2.rectangle(full_scene, (250, 100), (350, 200), (0, 150, 0), -1)
    cv2.circle(full_scene, (500, 150), 60, (150, 0, 0), -1)
    cv2.rectangle(full_scene, (650, 80), (750, 220), (150, 150, 0), -1)

    # 겹치는 부분이 있는 두 이미지
    img1 = full_scene[:, :450].copy()
    img2 = full_scene[:, 300:].copy()

    cv2.imwrite('panorama_img1.jpg', img1)
    cv2.imwrite('panorama_img2.jpg', img2)

    print("스티칭할 이미지 생성 완료")

    # 특징점 검출 및 매칭
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB 특징점
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is not None and des2 is not None:
        # 매칭
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        print(f"좋은 매칭: {len(good)}")

        if len(good) >= 4:
            # 호모그래피 계산
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # 파노라마 생성
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]

                # 결과 이미지 크기 계산
                corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)

                all_corners = np.concatenate([
                    np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
                    transformed
                ])

                x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
                x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

                translation = np.array([
                    [1, 0, -x_min],
                    [0, 1, -y_min],
                    [0, 0, 1]
                ])

                # 워핑 및 합성
                result_width = x_max - x_min
                result_height = y_max - y_min

                result = cv2.warpPerspective(img2, translation @ H,
                                            (result_width, result_height))
                result[-y_min:-y_min+h1, -x_min:-x_min+w1] = img1

                cv2.imwrite('panorama_result.jpg', result)
                print("파노라마 생성 완료: panorama_result.jpg")

    print("\n파노라마 생성 파이프라인:")
    print("  1. 특징점 검출 (SIFT/ORB)")
    print("  2. 특징점 매칭")
    print("  3. 호모그래피 계산")
    print("  4. 이미지 워핑")
    print("  5. 블렌딩 (경계 부드럽게)")

    # OpenCV Stitcher 클래스 사용
    print("\nOpenCV Stitcher 사용:")
    code = '''
# 간단한 방법: cv2.Stitcher
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, panorama = stitcher.stitch([img1, img2, img3])

if status == cv2.Stitcher_OK:
    cv2.imwrite('panorama.jpg', panorama)
else:
    print(f"스티칭 실패: {status}")
'''
    print(code)


# ============================================================
# 프로젝트 5: 증강 현실 마커
# ============================================================

def ar_marker_project():
    """증강 현실 마커 프로젝트 (개념)"""
    print("\n" + "=" * 60)
    print("프로젝트 5: AR 마커 기반 증강 현실")
    print("=" * 60)

    # ArUco 마커 생성 (시뮬레이션)
    marker_size = 200

    # 시뮬레이션용 마커 이미지
    marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker[:] = 255

    # 간단한 패턴 (실제 ArUco 마커는 더 복잡)
    cv2.rectangle(marker, (10, 10), (190, 190), 0, 10)
    cv2.rectangle(marker, (40, 40), (80, 80), 0, -1)
    cv2.rectangle(marker, (120, 40), (160, 80), 0, -1)
    cv2.rectangle(marker, (40, 120), (80, 160), 0, -1)
    cv2.rectangle(marker, (120, 120), (160, 160), 0, -1)
    cv2.rectangle(marker, (80, 80), (120, 120), 0, -1)

    cv2.imwrite('ar_marker.jpg', marker)

    print("\nArUco 마커:")
    print("  - OpenCV에 내장된 마커 시스템")
    print("  - 자동 검출 및 ID 인식")
    print("  - 4개 코너로 포즈 추정")

    code = '''
# ArUco 마커 생성
import cv2

# 딕셔너리 선택
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 마커 생성 (ID=42, 크기=200x200)
marker = cv2.aruco.generateImageMarker(aruco_dict, 42, 200)
cv2.imwrite('marker_42.png', marker)

# 마커 검출
detector = cv2.aruco.ArucoDetector(aruco_dict)
corners, ids, rejected = detector.detectMarkers(gray)

# 마커 그리기
cv2.aruco.drawDetectedMarkers(image, corners, ids)

# 포즈 추정 (카메라 캘리브레이션 필요)
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    corners, marker_length, camera_matrix, dist_coeffs
)

# 좌표축 그리기
for rvec, tvec in zip(rvecs, tvecs):
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
'''
    print(code)

    print("\nAR 응용:")
    print("  - 3D 객체 오버레이")
    print("  - 가상 가구 배치")
    print("  - 게임/교육")


# ============================================================
# 프로젝트 구조 가이드
# ============================================================

def project_structure_guide():
    """프로젝트 구조 가이드"""
    print("\n" + "=" * 60)
    print("컴퓨터 비전 프로젝트 구조 가이드")
    print("=" * 60)

    print("""
권장 프로젝트 구조:

project/
├── main.py           # 메인 실행 파일
├── config.py         # 설정 (경로, 파라미터)
├── requirements.txt  # 의존성
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── detection.py      # 객체 검출
│   ├── preprocessing.py  # 전처리
│   ├── tracking.py       # 추적
│   └── utils.py          # 유틸리티
│
├── models/           # 학습된 모델 파일
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── ...
│
├── data/             # 입력 데이터
│   ├── images/
│   └── videos/
│
├── output/           # 결과 저장
│   ├── results/
│   └── logs/
│
└── tests/            # 테스트 코드
    └── test_detection.py
""")

    print("\n개발 팁:")
    print("  1. 모듈화: 기능별로 분리")
    print("  2. 설정 파일: 하드코딩 피하기")
    print("  3. 로깅: 디버깅 용이")
    print("  4. 테스트: 단위 테스트 작성")
    print("  5. 문서화: 함수/클래스 docstring")


def main():
    """메인 함수"""
    # 프로젝트 1: 문서 스캐너
    document_scanner()

    # 프로젝트 2: 번호판 인식
    license_plate_recognition()

    # 프로젝트 3: 객체 추적
    object_tracking_project()

    # 프로젝트 4: 파노라마
    panorama_stitching()

    # 프로젝트 5: AR 마커
    ar_marker_project()

    # 프로젝트 구조 가이드
    project_structure_guide()

    print("\n" + "=" * 60)
    print("실전 프로젝트 데모 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
