# Computer_Vision 예제

Computer_Vision 폴더의 20개 레슨에 해당하는 실행 가능한 예제 코드입니다.

## 폴더 구조

```
examples/
├── 01_environment_basics.py     # 환경 설정 및 기초
├── 02_image_basics.py           # 이미지 기초 연산
├── 03_color_spaces.py           # 색상 공간
├── 04_geometric_transforms.py   # 기하학적 변환
├── 05_filtering.py              # 이미지 필터링
├── 06_morphology.py             # 모폴로지 연산
├── 07_thresholding.py           # 이진화 및 임계처리
├── 08_edge_detection.py         # 엣지 검출
├── 09_contours.py               # 윤곽선 검출
├── 10_shape_analysis.py         # 도형 분석
├── 11_hough_transform.py        # 허프 변환
├── 12_histogram.py              # 히스토그램 분석
├── 13_feature_detection.py      # 특징점 검출
├── 14_feature_matching.py       # 특징점 매칭
├── 15_object_detection.py       # 객체 검출 기초
├── 16_face_detection.py         # 얼굴 검출 및 인식
├── 17_video_processing.py       # 비디오 처리
├── 18_camera_calibration.py     # 카메라 캘리브레이션
├── 19_dnn_module.py             # 딥러닝 DNN 모듈
└── 20_practical_project.py      # 실전 프로젝트
```

## 환경 설정

```bash
# 가상환경 생성
python -m venv cv-env
source cv-env/bin/activate  # Windows: cv-env\Scripts\activate

# 필수 패키지 설치
pip install opencv-python numpy matplotlib

# 확장 패키지 (SIFT, SURF 등)
pip install opencv-contrib-python

# 얼굴 인식용 (선택)
pip install dlib face_recognition
```

## 실행 방법

```bash
# 개별 예제 실행
cd Computer_Vision/examples
python 01_environment_basics.py

# 테스트 이미지 준비 (필요 시)
# 예제 실행 전에 sample.jpg 등의 테스트 이미지가 필요합니다
# 웹캠 예제는 카메라가 연결되어 있어야 합니다
```

## 레슨별 예제 개요

| 레슨 | 주제 | 핵심 함수/개념 |
|------|------|---------------|
| 01 | 환경 설정 | `cv2.__version__`, 설치 확인 |
| 02 | 이미지 기초 | `imread`, `imshow`, `imwrite`, ROI |
| 03 | 색상 공간 | `cvtColor`, BGR/HSV/LAB, `split`/`merge` |
| 04 | 기하 변환 | `resize`, `rotate`, `warpAffine`, `warpPerspective` |
| 05 | 필터링 | `GaussianBlur`, `medianBlur`, `bilateralFilter` |
| 06 | 모폴로지 | `erode`, `dilate`, `morphologyEx` |
| 07 | 이진화 | `threshold`, OTSU, `adaptiveThreshold` |
| 08 | 엣지 검출 | `Sobel`, `Laplacian`, `Canny` |
| 09 | 윤곽선 | `findContours`, `drawContours`, `approxPolyDP` |
| 10 | 도형 분석 | `moments`, `boundingRect`, `convexHull` |
| 11 | 허프 변환 | `HoughLines`, `HoughLinesP`, `HoughCircles` |
| 12 | 히스토그램 | `calcHist`, `equalizeHist`, CLAHE |
| 13 | 특징점 검출 | Harris, SIFT, ORB |
| 14 | 특징점 매칭 | BFMatcher, FLANN, homography |
| 15 | 객체 검출 | template matching, Haar cascade |
| 16 | 얼굴 검출 | Haar face, dlib landmarks |
| 17 | 비디오 처리 | `VideoCapture`, 배경 차분, 옵티컬플로우 |
| 18 | 캘리브레이션 | 체스보드, 왜곡 보정 |
| 19 | DNN 모듈 | `readNet`, `blobFromImage` |
| 20 | 실전 프로젝트 | 문서 스캐너, 차선 검출 |

## 테스트 이미지 준비

예제 실행을 위해 다음 테스트 이미지가 필요합니다:

```bash
# 간단한 테스트 이미지 생성 (예제 01에 포함)
python 01_environment_basics.py  # 테스트 이미지 자동 생성

# 또는 직접 이미지 준비
# - sample.jpg: 일반 컬러 이미지
# - face.jpg: 얼굴이 포함된 이미지 (16번 예제용)
# - checkerboard.jpg: 체스보드 이미지 (18번 예제용)
```

## 학습 순서

### 1단계: 기초 (01-04)
```
01 → 02 → 03 → 04
```

### 2단계: 이미지 처리 (05-08)
```
05 → 06 → 07 → 08
```

### 3단계: 객체 분석 (09-12)
```
09 → 10 → 11 → 12
```

### 4단계: 특징/검출 (13-16)
```
13 → 14 → 15 → 16
```

### 5단계: 고급 (17-20)
```
17 → 18 → 19 → 20
```

## 참고 자료

- [OpenCV 공식 문서](https://docs.opencv.org/)
- [OpenCV-Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyImageSearch](https://pyimagesearch.com/)
