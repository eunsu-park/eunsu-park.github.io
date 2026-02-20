"""
19. OpenCV DNN 모듈
- 딥러닝 모델 로드
- 이미지 분류
- 객체 검출 (YOLO, SSD)
- 시맨틱 세그멘테이션
"""

import cv2
import numpy as np


def dnn_module_overview():
    """DNN 모듈 개요"""
    print("=" * 50)
    print("OpenCV DNN 모듈 개요")
    print("=" * 50)

    print("\n1. 지원 프레임워크:")
    frameworks = [
        ('Caffe', '.caffemodel, .prototxt'),
        ('TensorFlow', '.pb, .pbtxt'),
        ('Darknet', '.weights, .cfg'),
        ('ONNX', '.onnx'),
        ('Torch', '.t7, .net'),
    ]

    for name, files in frameworks:
        print(f"   {name}: {files}")

    print("\n2. 모델 로드 함수:")
    print("   cv2.dnn.readNet(model, config)")
    print("   cv2.dnn.readNetFromCaffe(prototxt, caffemodel)")
    print("   cv2.dnn.readNetFromTensorflow(model, config)")
    print("   cv2.dnn.readNetFromDarknet(cfg, weights)")
    print("   cv2.dnn.readNetFromONNX(onnx)")

    print("\n3. 백엔드 및 타겟:")
    print("   백엔드: DNN_BACKEND_OPENCV, DNN_BACKEND_CUDA")
    print("   타겟: DNN_TARGET_CPU, DNN_TARGET_CUDA")


def blob_creation_demo():
    """Blob 생성 데모"""
    print("\n" + "=" * 50)
    print("Blob 생성")
    print("=" * 50)

    # 테스트 이미지
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = [150, 150, 150]
    cv2.circle(img, (320, 240), 100, (0, 200, 0), -1)

    # Blob 생성
    # scalefactor: 픽셀 값 스케일링 (보통 1/255)
    # size: 네트워크 입력 크기
    # mean: 평균 값 빼기 (BGR 순서)
    # swapRB: BGR -> RGB 변환
    # crop: 크기 조정 시 크롭 여부

    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1/255.0,
        size=(224, 224),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    print(f"원본 이미지: {img.shape}")
    print(f"Blob shape: {blob.shape}")
    print(f"Blob dtype: {blob.dtype}")

    print("\nblobFromImage 파라미터:")
    print("  scalefactor: 보통 1/255.0 (0-1 정규화)")
    print("  size: 네트워크 입력 크기 (224x224, 416x416 등)")
    print("  mean: ImageNet 평균 (104.0, 117.0, 123.0)")
    print("  swapRB: OpenCV BGR -> 모델 RGB")
    print("  crop: True면 크롭, False면 리사이즈만")

    # 여러 이미지 처리
    images = [img, img.copy()]
    blob_batch = cv2.dnn.blobFromImages(
        images,
        scalefactor=1/255.0,
        size=(224, 224),
        mean=(0, 0, 0),
        swapRB=True
    )
    print(f"\nBatch blob shape: {blob_batch.shape}")

    cv2.imwrite('dnn_input.jpg', img)


def image_classification_demo():
    """이미지 분류 데모 (개념)"""
    print("\n" + "=" * 50)
    print("이미지 분류 (Image Classification)")
    print("=" * 50)

    print("\n모델 예시:")
    models = [
        ('ResNet', 'Residual Networks, 깊은 네트워크'),
        ('VGG', 'Visual Geometry Group, 단순 구조'),
        ('MobileNet', '경량화, 모바일용'),
        ('EfficientNet', '효율적 스케일링'),
        ('GoogLeNet', 'Inception 모듈'),
    ]

    for name, desc in models:
        print(f"   {name}: {desc}")

    code = '''
# 이미지 분류 코드 템플릿
import cv2

# 모델 로드 (예: MobileNet)
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'mobilenet.caffemodel'
)

# 이미지 전처리
img = cv2.imread('image.jpg')
blob = cv2.dnn.blobFromImage(
    img, 1/255.0, (224, 224), (104, 117, 123), swapRB=True
)

# 추론
net.setInput(blob)
output = net.forward()

# 결과 해석
class_id = np.argmax(output)
confidence = output[0][class_id]
print(f"Class: {class_id}, Confidence: {confidence:.2f}")
'''
    print(code)

    print("\n참고: 실제 실행에는 모델 파일이 필요합니다.")
    print("  MobileNet: https://github.com/shicai/MobileNet-Caffe")
    print("  ONNX Models: https://github.com/onnx/models")


def object_detection_yolo_demo():
    """YOLO 객체 검출 데모 (개념)"""
    print("\n" + "=" * 50)
    print("객체 검출 - YOLO")
    print("=" * 50)

    print("\nYOLO (You Only Look Once):")
    print("  - 실시간 객체 검출")
    print("  - 단일 네트워크로 검출 + 분류")
    print("  - 버전: YOLOv3, YOLOv4, YOLOv5, YOLOv8")

    code = '''
# YOLO 객체 검출 코드
import cv2
import numpy as np

# 모델 로드 (Darknet)
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# 출력 레이어 이름
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 이미지 전처리
img = cv2.imread('image.jpg')
blob = cv2.dnn.blobFromImage(
    img, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False
)

# 추론
net.setInput(blob)
outputs = net.forward(output_layers)

# 결과 처리
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # 바운딩 박스 좌표
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])
            h = int(detection[3] * img.shape[0])

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# NMS (Non-Maximum Suppression)
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 결과 시각화
for i in indices.flatten():
    x, y, w, h = boxes[i]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
'''
    print(code)

    print("\n모델 다운로드:")
    print("  YOLOv3: https://pjreddie.com/darknet/yolo/")
    print("  YOLOv4: https://github.com/AlexeyAB/darknet")


def object_detection_ssd_demo():
    """SSD 객체 검출 데모 (개념)"""
    print("\n" + "=" * 50)
    print("객체 검출 - SSD")
    print("=" * 50)

    print("\nSSD (Single Shot Detector):")
    print("  - 다중 스케일 특징 맵 사용")
    print("  - 빠른 속도")
    print("  - MobileNet + SSD 조합 인기")

    code = '''
# SSD 객체 검출 코드
import cv2

# 모델 로드 (TensorFlow)
net = cv2.dnn.readNetFromTensorflow(
    'frozen_inference_graph.pb',
    'ssd_mobilenet_v2_coco.pbtxt'
)

# 이미지 전처리
img = cv2.imread('image.jpg')
blob = cv2.dnn.blobFromImage(
    img, size=(300, 300), mean=(127.5, 127.5, 127.5),
    scalefactor=1/127.5, swapRB=True
)

# 추론
net.setInput(blob)
detections = net.forward()

# 결과 처리
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        class_id = int(detections[0, 0, i, 1])
        x1 = int(detections[0, 0, i, 3] * img.shape[1])
        y1 = int(detections[0, 0, i, 4] * img.shape[0])
        x2 = int(detections[0, 0, i, 5] * img.shape[1])
        y2 = int(detections[0, 0, i, 6] * img.shape[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
'''
    print(code)

    print("\n모델 다운로드:")
    print("  TensorFlow Model Zoo:")
    print("  https://github.com/tensorflow/models/blob/master/research/object_detection/")


def face_detection_dnn_demo():
    """DNN 얼굴 검출 데모"""
    print("\n" + "=" * 50)
    print("DNN 얼굴 검출")
    print("=" * 50)

    print("\nOpenCV DNN 얼굴 검출기:")
    print("  - Caffe 기반 SSD")
    print("  - 300x300 입력")
    print("  - Haar Cascade보다 정확")

    code = '''
# DNN 얼굴 검출
import cv2

# 모델 로드
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# 이미지 전처리
img = cv2.imread('image.jpg')
h, w = img.shape[:2]
blob = cv2.dnn.blobFromImage(
    img, 1.0, (300, 300), (104.0, 177.0, 123.0)
)

# 추론
net.setInput(blob)
detections = net.forward()

# 결과 처리
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{confidence:.2f}"
        cv2.putText(img, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
'''
    print(code)

    print("\n모델 다운로드:")
    print("  https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector")


def semantic_segmentation_demo():
    """시맨틱 세그멘테이션 데모 (개념)"""
    print("\n" + "=" * 50)
    print("시맨틱 세그멘테이션")
    print("=" * 50)

    print("\n세그멘테이션 유형:")
    print("  - Semantic: 픽셀 단위 클래스 분류")
    print("  - Instance: 개별 객체 구분")
    print("  - Panoptic: Semantic + Instance")

    print("\n주요 모델:")
    models = [
        ('FCN', 'Fully Convolutional Network'),
        ('U-Net', '의료 이미지용'),
        ('DeepLab', 'Atrous convolution'),
        ('SegNet', '인코더-디코더 구조'),
        ('PSPNet', 'Pyramid Pooling'),
    ]

    for name, desc in models:
        print(f"   {name}: {desc}")

    code = '''
# 시맨틱 세그멘테이션 코드
import cv2
import numpy as np

# 모델 로드 (예: ENet)
net = cv2.dnn.readNet('enet-model.net')

# 이미지 전처리
img = cv2.imread('image.jpg')
blob = cv2.dnn.blobFromImage(
    img, 1/255.0, (1024, 512), (0, 0, 0), swapRB=True
)

# 추론
net.setInput(blob)
output = net.forward()

# 결과 처리 (클래스 맵)
class_map = np.argmax(output[0], axis=0)

# 컬러 맵 적용
colors = np.random.randint(0, 255, (num_classes, 3))
segmentation = colors[class_map]
'''
    print(code)


def pose_estimation_dnn_demo():
    """포즈 추정 DNN 데모 (개념)"""
    print("\n" + "=" * 50)
    print("포즈 추정 (Pose Estimation)")
    print("=" * 50)

    print("\n포즈 추정 유형:")
    print("  - 2D: 이미지상의 관절 위치")
    print("  - 3D: 3차원 공간의 관절 위치")

    print("\n주요 모델:")
    models = [
        ('OpenPose', 'Bottom-up 방식, 다중 인원'),
        ('PoseNet', '경량화, 실시간'),
        ('HRNet', '고해상도, 정확'),
        ('MediaPipe', 'Google, 모바일 최적화'),
    ]

    for name, desc in models:
        print(f"   {name}: {desc}")

    print("\n관절 포인트 (COCO 데이터셋):")
    keypoints = [
        "0: nose", "1: neck",
        "2: right_shoulder", "3: right_elbow", "4: right_wrist",
        "5: left_shoulder", "6: left_elbow", "7: left_wrist",
        "8: right_hip", "9: right_knee", "10: right_ankle",
        "11: left_hip", "12: left_knee", "13: left_ankle",
        "14: right_eye", "15: left_eye",
        "16: right_ear", "17: left_ear"
    ]
    for kp in keypoints:
        print(f"   {kp}")


def dnn_performance_tips():
    """DNN 성능 최적화"""
    print("\n" + "=" * 50)
    print("DNN 성능 최적화")
    print("=" * 50)

    print("""
1. GPU 가속 사용
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

2. 입력 크기 조정
   - 작은 입력 = 빠른 추론
   - 정확도와 속도 트레이드오프

3. 모델 최적화
   - INT8 양자화
   - 모델 프루닝
   - 지식 증류

4. 배치 처리
   - 여러 이미지 동시 처리
   - blobFromImages() 사용

5. 비동기 추론
   - net.forwardAsync()
   - 추론 중 다른 작업 수행

6. 모델 선택
   - 속도 중시: MobileNet, EfficientNet-Lite
   - 정확도 중시: ResNet, EfficientNet

7. 추론 시간 측정
""")

    # 시간 측정 예시
    print("추론 시간 측정:")
    code = '''
import time

# 워밍업
for _ in range(10):
    net.forward()

# 측정
times = []
for _ in range(100):
    start = time.time()
    net.forward()
    times.append(time.time() - start)

print(f"평균: {np.mean(times)*1000:.2f}ms")
print(f"FPS: {1/np.mean(times):.2f}")
'''
    print(code)


def model_download_guide():
    """모델 다운로드 가이드"""
    print("\n" + "=" * 50)
    print("모델 다운로드 가이드")
    print("=" * 50)

    print("""
1. YOLO
   - 공식: https://pjreddie.com/darknet/yolo/
   - v4: https://github.com/AlexeyAB/darknet
   - v5+: https://github.com/ultralytics/yolov5

2. SSD MobileNet
   - TensorFlow Model Zoo
   - https://github.com/tensorflow/models/

3. 얼굴 검출
   - OpenCV DNN Face Detector
   - https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

4. 포즈 추정
   - OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
   - 경량 버전: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch

5. 세그멘테이션
   - ENet: https://github.com/e-lab/ENet-training
   - DeepLab: https://github.com/tensorflow/models/tree/master/research/deeplab

6. ONNX Model Zoo
   - https://github.com/onnx/models
   - 다양한 사전 학습 모델

7. OpenVINO Model Zoo
   - https://github.com/openvinotoolkit/open_model_zoo
   - Intel 최적화 모델
""")


def main():
    """메인 함수"""
    # DNN 모듈 개요
    dnn_module_overview()

    # Blob 생성
    blob_creation_demo()

    # 이미지 분류
    image_classification_demo()

    # YOLO 객체 검출
    object_detection_yolo_demo()

    # SSD 객체 검출
    object_detection_ssd_demo()

    # DNN 얼굴 검출
    face_detection_dnn_demo()

    # 시맨틱 세그멘테이션
    semantic_segmentation_demo()

    # 포즈 추정
    pose_estimation_dnn_demo()

    # 성능 최적화
    dnn_performance_tips()

    # 모델 다운로드 가이드
    model_download_guide()

    print("\nDNN 모듈 데모 완료!")


if __name__ == '__main__':
    main()
