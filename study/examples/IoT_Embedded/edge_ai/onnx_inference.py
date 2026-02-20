#!/usr/bin/env python3
"""
ONNX Runtime 기반 Edge AI 추론
이미지 분류 및 객체 검출 예제

참고: content/ko/IoT_Embedded/09_Edge_AI_ONNX.md
"""

import numpy as np
import time
from typing import Optional, Tuple, List, Dict
import os

# ONNX Runtime 설치 확인
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("경고: onnxruntime이 설치되지 않았습니다.")
    print("설치: pip install onnxruntime")

# OpenCV 설치 확인
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("경고: opencv-python이 설치되지 않았습니다.")
    print("설치: pip install opencv-python")

# PIL 설치 확인
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("경고: Pillow가 설치되지 않았습니다.")
    print("설치: pip install Pillow")


# === ONNX 모델 래퍼 ===

class ONNXModel:
    """ONNX 모델 기본 래퍼"""

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        if not HAS_ONNX:
            raise ImportError("onnxruntime이 필요합니다")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        if providers is None:
            # 사용 가능한 프로바이더 자동 선택
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        # 세션 옵션 설정
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # CPU 스레드 수

        # 세션 생성
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # 입출력 정보
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        self.output_name = self.session.get_outputs()[0].name

        print(f"모델 로드 완료: {model_path}")
        print(f"  프로바이더: {self.session.get_providers()}")
        print(f"  입력: {self.input_name} {self.input_shape}")
        print(f"  출력: {self.output_name}")

    def get_input_shape(self) -> list:
        """입력 형태 반환"""
        return self.input_shape

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """추론 수행"""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]

    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """성능 벤치마크"""
        # 더미 입력 생성
        dummy_shape = [1 if x == 'batch' or x == 'N' or x is None else x
                      for x in self.input_shape]
        dummy_input = np.random.randn(*dummy_shape).astype(np.float32)

        # 워밍업
        for _ in range(10):
            self.predict(dummy_input)

        # 측정
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.predict(dummy_input)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        times = np.array(times)

        results = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": 1000.0 / np.mean(times)
        }

        return results


# === 이미지 분류 모델 ===

class ImageClassifier(ONNXModel):
    """ONNX 이미지 분류 모델"""

    # ImageNet 클래스 (상위 10개만 예시)
    IMAGENET_CLASSES = [
        'tench', 'goldfish', 'great_white_shark', 'tiger_shark',
        'hammerhead', 'electric_ray', 'stingray', 'cock', 'hen', 'ostrich'
        # ... 실제로는 1000개 클래스
    ]

    def __init__(self, model_path: str, labels_path: Optional[str] = None):
        super().__init__(model_path)

        # 레이블 로드 (있는 경우)
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f]
        else:
            self.labels = self.IMAGENET_CLASSES

        # 입력 크기 추출
        self.input_height = self.input_shape[2] if len(self.input_shape) > 2 else 224
        self.input_width = self.input_shape[3] if len(self.input_shape) > 3 else 224

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """이미지 전처리 (PIL 사용)"""
        if not HAS_PIL:
            raise ImportError("Pillow가 필요합니다")

        # 이미지 로드
        image = Image.open(image_path).convert('RGB')

        # 리사이즈
        image = image.resize((self.input_width, self.input_height))

        # NumPy 배열로 변환
        img_array = np.array(image).astype(np.float32)

        # 정규화 (ImageNet 표준)
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        img_array = (img_array - mean) / std

        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        # 배치 차원 추가
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def classify(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """이미지 분류"""
        # 전처리
        input_data = self.preprocess_image(image_path)

        # 추론
        start = time.perf_counter()
        output = self.predict(input_data)
        inference_time = (time.perf_counter() - start) * 1000

        # Softmax
        probs = self._softmax(output[0])

        # Top-K 결과
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append((label, float(probs[idx])))

        print(f"추론 시간: {inference_time:.2f}ms")

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax 함수"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


# === 객체 검출 모델 (YOLO) ===

class YOLODetector:
    """YOLO ONNX 객체 검출기"""

    # COCO 데이터셋 80개 클래스
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        if not HAS_ONNX:
            raise ImportError("onnxruntime이 필요합니다")

        if not os.path.exists(model_path):
            # 모델이 없는 경우 시뮬레이션 모드
            print(f"경고: 모델 파일을 찾을 수 없습니다: {model_path}")
            print("시뮬레이션 모드로 실행합니다.")
            self.simulation_mode = True
            self.input_height = 640
            self.input_width = 640
            return

        self.simulation_mode = False

        # ONNX 세션 생성
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 입력 정보
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        print(f"YOLO 모델 로드 완료")
        print(f"  입력 크기: {self.input_width}x{self.input_height}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """이미지 전처리"""
        orig_height, orig_width = image.shape[:2]

        # 리사이즈
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # BGR to RGB, HWC to CHW
        input_data = resized[:, :, ::-1].transpose(2, 0, 1)

        # 정규화 (0-1)
        input_data = input_data.astype(np.float32) / 255.0

        # 배치 차원 추가
        input_data = np.expand_dims(input_data, axis=0)

        # 스케일 비율 저장
        scale = (orig_width / self.input_width, orig_height / self.input_height)

        return input_data, scale

    def detect(self, image: np.ndarray) -> List[Dict]:
        """객체 검출"""
        if self.simulation_mode:
            # 시뮬레이션: 랜덤 검출 결과 반환
            print("시뮬레이션 모드: 랜덤 검출 결과를 생성합니다.")
            return self._simulate_detection(image)

        if not HAS_CV2:
            raise ImportError("opencv-python이 필요합니다")

        # 전처리
        input_data, scale = self.preprocess(image)

        # 추론
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_data})
        inference_time = (time.perf_counter() - start) * 1000

        # 후처리
        detections = self.postprocess(outputs[0], scale)

        print(f"추론 시간: {inference_time:.2f}ms")
        print(f"검출된 객체: {len(detections)}개")

        return detections

    def postprocess(self, output: np.ndarray, scale: Tuple[float, float]) -> List[Dict]:
        """출력 후처리"""
        if not HAS_CV2:
            return []

        predictions = output[0]

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            confidence = pred[4]

            if confidence > self.conf_threshold:
                class_probs = pred[5:]
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]

                if class_score > self.conf_threshold:
                    # 박스 좌표 (center_x, center_y, width, height)
                    cx, cy, w, h = pred[:4]

                    # 원본 스케일로 변환
                    x1 = int((cx - w / 2) * scale[0])
                    y1 = int((cy - h / 2) * scale[1])
                    x2 = int((cx + w / 2) * scale[0])
                    y2 = int((cy + h / 2) * scale[1])

                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence * class_score))
                    class_ids.append(int(class_id))

        # NMS (Non-Maximum Suppression)
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.iou_threshold
            )

            results = []
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                results.append({
                    'box': boxes[idx],
                    'score': scores[idx],
                    'class_id': class_ids[idx],
                    'class_name': self.COCO_CLASSES[class_ids[idx]]
                })

            return results

        return []

    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """시뮬레이션: 랜덤 검출 결과"""
        height, width = image.shape[:2]

        num_detections = np.random.randint(1, 5)
        detections = []

        for _ in range(num_detections):
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = np.random.randint(x1 + 50, width)
            y2 = np.random.randint(y1 + 50, height)

            class_id = np.random.randint(0, len(self.COCO_CLASSES))

            detections.append({
                'box': [x1, y1, x2, y2],
                'score': np.random.uniform(0.5, 0.95),
                'class_id': class_id,
                'class_name': self.COCO_CLASSES[class_id]
            })

        return detections

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """검출 결과 시각화"""
        if not HAS_CV2:
            print("경고: opencv-python이 없어 시각화를 건너뜁니다.")
            return image

        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']}: {det['score']:.2f}"

            # 박스 그리기
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 라벨 배경
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

            # 라벨 텍스트
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result


# === 사용 예제 ===

def example_basic_inference():
    """기본 ONNX 추론 예제"""
    print("\n=== 기본 ONNX 추론 예제 ===")

    if not HAS_ONNX:
        print("onnxruntime이 설치되지 않아 예제를 실행할 수 없습니다.")
        return

    # 시뮬레이션: 더미 모델 생성
    print("시뮬레이션 모드: 더미 입력으로 테스트합니다.")

    # 더미 데이터
    batch_size = 1
    channels = 3
    height = 224
    width = 224

    dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    print(f"입력 형태: {dummy_input.shape}")
    print(f"입력 데이터 범위: [{dummy_input.min():.2f}, {dummy_input.max():.2f}]")


def example_image_classification():
    """이미지 분류 예제"""
    print("\n=== 이미지 분류 예제 ===")

    if not HAS_ONNX:
        print("onnxruntime이 설치되지 않아 예제를 실행할 수 없습니다.")
        return

    # 모델 경로 (예시)
    model_path = "resnet18.onnx"

    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("PyTorch에서 변환 예시:")
        print("  import torch")
        print("  model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)")
        print("  dummy_input = torch.randn(1, 3, 224, 224)")
        print("  torch.onnx.export(model, dummy_input, 'resnet18.onnx')")
        return

    # 분류기 생성
    classifier = ImageClassifier(model_path)

    # 벤치마크
    print("\n성능 벤치마크:")
    results = classifier.benchmark(num_iterations=50)
    print(f"  평균: {results['mean_ms']:.2f}ms")
    print(f"  FPS: {results['fps']:.1f}")


def example_object_detection():
    """객체 검출 예제"""
    print("\n=== 객체 검출 예제 (시뮬레이션) ===")

    # 시뮬레이션 모드로 실행
    detector = YOLODetector("yolov5s.onnx")  # 파일이 없어도 시뮬레이션 가능

    # 더미 이미지 생성
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 검출
    detections = detector.detect(dummy_image)

    # 결과 출력
    print("\n검출 결과:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']}: {det['score']:.2f}")
        print(f"     박스: {det['box']}")

    # 시각화 (OpenCV가 있는 경우)
    if HAS_CV2:
        result_image = detector.draw_detections(dummy_image, detections)
        print("\n결과 이미지 생성 완료")


def example_performance_comparison():
    """성능 비교 예제"""
    print("\n=== 성능 비교 예제 ===")

    if not HAS_ONNX:
        print("onnxruntime이 설치되지 않아 예제를 실행할 수 없습니다.")
        return

    print("배치 크기별 성능 비교 (시뮬레이션)")

    input_shape = (1, 3, 224, 224)

    for batch_size in [1, 4, 8, 16]:
        data = np.random.randn(batch_size, *input_shape[1:]).astype(np.float32)

        start = time.perf_counter()
        # 시뮬레이션: 간단한 연산
        _ = np.mean(data, axis=(2, 3))
        elapsed = time.perf_counter() - start

        throughput = batch_size / elapsed
        print(f"배치 크기 {batch_size:2d}: {throughput:.1f} samples/sec")


# === 메인 실행 ===

if __name__ == "__main__":
    print("=" * 60)
    print("ONNX Runtime Edge AI 추론 예제")
    print("=" * 60)

    # ONNX Runtime 설치 확인
    if HAS_ONNX:
        print(f"\nONNX Runtime 버전: {ort.__version__}")
        print(f"사용 가능한 프로바이더: {ort.get_available_providers()}")
    else:
        print("\n경고: ONNX Runtime이 설치되지 않았습니다.")
        print("설치: pip install onnxruntime")

    # 예제 실행
    example_basic_inference()
    example_image_classification()
    example_object_detection()
    example_performance_comparison()

    print("\n" + "=" * 60)
    print("모든 예제 완료")
    print("=" * 60)
