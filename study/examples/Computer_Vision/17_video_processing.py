"""
17. 비디오 처리
- VideoCapture (카메라, 파일)
- VideoWriter (비디오 저장)
- 프레임 처리
- 백그라운드 차감
- 광학 흐름 (Optical Flow)
"""

import cv2
import numpy as np


def video_capture_basics():
    """비디오 캡처 기초"""
    print("=" * 50)
    print("VideoCapture 기초")
    print("=" * 50)

    print("\n카메라 캡처:")
    print("  cap = cv2.VideoCapture(0)  # 기본 카메라")
    print("  cap = cv2.VideoCapture(1)  # 두 번째 카메라")

    print("\n파일 캡처:")
    print("  cap = cv2.VideoCapture('video.mp4')")
    print("  cap = cv2.VideoCapture('rtsp://...')  # 스트리밍")

    print("\n주요 속성:")
    properties = [
        ('CAP_PROP_FRAME_WIDTH', '프레임 너비'),
        ('CAP_PROP_FRAME_HEIGHT', '프레임 높이'),
        ('CAP_PROP_FPS', '초당 프레임 수'),
        ('CAP_PROP_FRAME_COUNT', '총 프레임 수'),
        ('CAP_PROP_POS_FRAMES', '현재 프레임 위치'),
        ('CAP_PROP_POS_MSEC', '현재 시간 (ms)'),
    ]

    for prop, desc in properties:
        print(f"  cv2.{prop}: {desc}")

    # 시뮬레이션 비디오 생성
    print("\n시뮬레이션 비디오 생성 중...")
    frames = []
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = [100, 100, 100]
        cv2.circle(frame, (100 + i * 15, 240), 30, (0, 255, 0), -1)
        cv2.putText(frame, f'Frame {i}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frames.append(frame)

    cv2.imwrite('video_frame_sample.jpg', frames[15])
    print("샘플 프레임 저장: video_frame_sample.jpg")


def video_writer_demo():
    """비디오 저장 데모"""
    print("\n" + "=" * 50)
    print("VideoWriter (비디오 저장)")
    print("=" * 50)

    # 프레임 생성
    width, height = 640, 480
    fps = 30.0

    # 코덱 설정
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4

    print(f"비디오 설정: {width}x{height}, {fps}fps")

    # VideoWriter 생성
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    if not out.isOpened():
        print("VideoWriter를 열 수 없습니다.")
        return

    # 프레임 작성
    for i in range(90):  # 3초
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [50, 50, 50]

        # 움직이는 원
        x = int(100 + 5 * i)
        y = int(240 + 100 * np.sin(i * 0.1))
        cv2.circle(frame, (x, y), 40, (0, 200, 0), -1)

        # 프레임 번호
        cv2.putText(frame, f'Frame: {i}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print("비디오 저장 완료: output_video.mp4")

    print("\n지원 코덱:")
    codecs = [
        ("'XVID'", '.avi', 'MPEG-4'),
        ("'mp4v'", '.mp4', 'MPEG-4'),
        ("'avc1'", '.mp4', 'H.264 (macOS)'),
        ("'MJPG'", '.avi', 'Motion JPEG'),
    ]
    for code, ext, desc in codecs:
        print(f"  cv2.VideoWriter_fourcc(*{code}) → {ext} ({desc})")


def frame_processing_demo():
    """프레임 처리 데모"""
    print("\n" + "=" * 50)
    print("프레임 처리")
    print("=" * 50)

    # 시뮬레이션 프레임
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [150, 150, 150]
    cv2.rectangle(frame, (200, 150), (440, 330), (0, 100, 200), -1)
    cv2.circle(frame, (320, 240), 50, (200, 100, 0), -1)

    # 다양한 처리
    # 1. 그레이스케일
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 블러
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # 3. 엣지
    edges = cv2.Canny(gray, 50, 150)

    # 4. 컬러 조정
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 50)  # 채도 증가
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    print("프레임 처리 예제:")
    print("  1. 그레이스케일 변환")
    print("  2. 가우시안 블러")
    print("  3. Canny 엣지 검출")
    print("  4. 색상 보정 (HSV)")

    cv2.imwrite('frame_original.jpg', frame)
    cv2.imwrite('frame_gray.jpg', gray)
    cv2.imwrite('frame_blurred.jpg', blurred)
    cv2.imwrite('frame_edges.jpg', edges)
    cv2.imwrite('frame_enhanced.jpg', enhanced)

    print("\n실시간 처리 템플릿:")
    code = '''
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 처리
    processed = your_processing_function(frame)

    cv2.imshow('Video', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
    print(code)


def background_subtraction_demo():
    """배경 차감 데모"""
    print("\n" + "=" * 50)
    print("배경 차감 (Background Subtraction)")
    print("=" * 50)

    # 시뮬레이션 프레임들 (배경 + 움직이는 객체)
    background = np.zeros((480, 640, 3), dtype=np.uint8)
    background[:] = [120, 120, 120]
    cv2.rectangle(background, (100, 300), (250, 400), (80, 80, 80), -1)  # 고정 객체

    # 배경 차감기 생성
    # MOG2: Mixture of Gaussians
    bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )

    # KNN
    bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
        history=500,
        dist2Threshold=400,
        detectShadows=True
    )

    print("배경 차감기:")
    print("  1. MOG2 (Mixture of Gaussians)")
    print("     - 복잡한 배경에 효과적")
    print("     - 그림자 검출 가능")
    print("")
    print("  2. KNN (K-Nearest Neighbors)")
    print("     - 비정형 분포에 효과적")
    print("     - 조명 변화에 강건")

    # 시뮬레이션 (움직이는 원)
    for i in range(30):
        frame = background.copy()
        x = 100 + i * 15
        cv2.circle(frame, (x, 200), 40, (0, 200, 0), -1)

        # 배경 차감 적용
        fg_mask_mog2 = bg_subtractor_mog2.apply(frame)
        fg_mask_knn = bg_subtractor_knn.apply(frame)

        if i == 15:  # 중간 프레임 저장
            cv2.imwrite('bg_frame.jpg', frame)
            cv2.imwrite('bg_mask_mog2.jpg', fg_mask_mog2)
            cv2.imwrite('bg_mask_knn.jpg', fg_mask_knn)

    print("\n파라미터:")
    print("  history: 학습에 사용할 과거 프레임 수")
    print("  varThreshold (MOG2): 픽셀-모델 거리 임계값")
    print("  dist2Threshold (KNN): 거리 임계값")
    print("  detectShadows: 그림자 검출 여부")

    print("\n후처리:")
    print("  - 모폴로지 연산 (노이즈 제거)")
    print("  - 컨투어 검출 (객체 추출)")
    print("  - 바운딩 박스 그리기")


def optical_flow_demo():
    """광학 흐름 데모"""
    print("\n" + "=" * 50)
    print("광학 흐름 (Optical Flow)")
    print("=" * 50)

    # 두 프레임 생성 (움직임 시뮬레이션)
    frame1 = np.zeros((300, 400), dtype=np.uint8)
    frame1[:] = 100
    cv2.circle(frame1, (100, 150), 30, 200, -1)
    cv2.rectangle(frame1, (250, 100), (320, 200), 180, -1)

    frame2 = np.zeros((300, 400), dtype=np.uint8)
    frame2[:] = 100
    cv2.circle(frame2, (130, 150), 30, 200, -1)  # 오른쪽으로 이동
    cv2.rectangle(frame2, (250, 130), (320, 230), 180, -1)  # 아래로 이동

    # Lucas-Kanade (sparse)
    print("1. Lucas-Kanade (Sparse Optical Flow)")
    print("   - 특정 점들의 움직임 추적")
    print("   - 빠름, 특징점 기반")

    # 특징점 검출
    p0 = cv2.goodFeaturesToTrack(frame1, maxCorners=100, qualityLevel=0.3,
                                  minDistance=7, blockSize=7)

    if p0 is not None:
        # 광학 흐름 계산
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            frame1, frame2, p0, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # 좋은 점만 선택
        good_old = p0[status == 1]
        good_new = p1[status == 1]

        # 시각화
        result_lk = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
        for old, new in zip(good_old, good_new):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.line(result_lk, (a, b), (c, d), (0, 255, 0), 2)
            cv2.circle(result_lk, (a, b), 5, (0, 0, 255), -1)

        cv2.imwrite('optflow_lk.jpg', result_lk)
        print(f"   추적된 점: {len(good_new)}")

    # Farneback (dense)
    print("\n2. Farneback (Dense Optical Flow)")
    print("   - 모든 픽셀의 움직임 계산")
    print("   - 느림, 전체 움직임 파악")

    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # 흐름을 색상으로 시각화
    hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    result_fb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite('optflow_frame1.jpg', frame1)
    cv2.imwrite('optflow_frame2.jpg', frame2)
    cv2.imwrite('optflow_farneback.jpg', result_fb)

    print(f"   흐름 벡터 shape: {flow.shape}")

    print("\n광학 흐름 활용:")
    print("  - 움직임 검출")
    print("  - 객체 추적")
    print("  - 비디오 압축")
    print("  - 동작 인식")


def video_tracking_demo():
    """객체 추적 데모"""
    print("\n" + "=" * 50)
    print("객체 추적 (Object Tracking)")
    print("=" * 50)

    print("OpenCV 추적기 종류:")
    trackers = [
        ('BOOSTING', '오래된 방식, 느림'),
        ('MIL', 'Multiple Instance Learning'),
        ('KCF', 'Kernelized Correlation Filters, 빠름'),
        ('TLD', 'Tracking-Learning-Detection'),
        ('MEDIANFLOW', '예측 가능한 움직임에 좋음'),
        ('GOTURN', 'Deep Learning 기반'),
        ('MOSSE', '매우 빠름'),
        ('CSRT', '정확, 다소 느림'),
    ]

    for name, desc in trackers:
        print(f"  cv2.Tracker{name}_create(): {desc}")

    print("\n추적기 사용 템플릿:")
    code = '''
# 추적기 생성
tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerCSRT_create()  # 더 정확

# 초기 바운딩 박스 설정
bbox = (x, y, w, h)  # 또는 cv2.selectROI()
tracker.init(first_frame, bbox)

# 추적 루프
while True:
    ret, frame = cap.read()
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
'''
    print(code)


def performance_tips():
    """성능 최적화 팁"""
    print("\n" + "=" * 50)
    print("비디오 처리 성능 최적화")
    print("=" * 50)

    print("""
1. 프레임 크기 줄이기
   - 처리 전: frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
   - 검출 후 원본에 결과 매핑

2. 프레임 스킵
   - 매 프레임 처리 불필요
   - frame_count % skip_frames == 0 일 때만 처리

3. ROI (Region of Interest) 활용
   - 전체 프레임 대신 관심 영역만 처리
   - roi = frame[y:y+h, x:x+w]

4. 멀티스레딩/멀티프로세싱
   - 캡처와 처리 분리
   - Queue로 프레임 전달

5. GPU 가속 (CUDA)
   - cv2.cuda.GpuMat()
   - cv2.cuda 모듈 함수 사용

6. 캡처 버퍼 설정
   - cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
   - 지연 최소화
""")


def main():
    """메인 함수"""
    # VideoCapture 기초
    video_capture_basics()

    # VideoWriter
    video_writer_demo()

    # 프레임 처리
    frame_processing_demo()

    # 배경 차감
    background_subtraction_demo()

    # 광학 흐름
    optical_flow_demo()

    # 객체 추적
    video_tracking_demo()

    # 성능 팁
    performance_tips()

    print("\n비디오 처리 데모 완료!")


if __name__ == '__main__':
    main()
