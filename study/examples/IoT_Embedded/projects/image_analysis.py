#!/usr/bin/env python3
"""
영상 분석 프로젝트 - 시뮬레이션 모드
Pi Camera를 사용한 영상 캡처, 객체 검출, 모션 감지

시뮬레이션 모드로 실제 하드웨어 없이 실행 가능
"""

import time
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import io
import random

# numpy는 시뮬레이션 모드에서 선택적
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[경고] numpy가 설치되지 않음. 시뮬레이션 모드로 실행합니다.")
    print("       실제 사용 시 'pip install numpy' 설치 필요\n")


# ==============================================================================
# 데이터 클래스
# ==============================================================================

class DetectionClass(Enum):
    """검출 가능한 객체 클래스"""
    PERSON = "person"
    CAR = "car"
    DOG = "dog"
    CAT = "cat"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """바운딩 박스"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Detection:
    """객체 검출 결과"""
    class_name: str
    confidence: float
    bbox: BoundingBox
    timestamp: datetime


@dataclass
class MotionEvent:
    """모션 감지 이벤트"""
    regions: List[BoundingBox]
    area: int
    timestamp: datetime
    frame_id: int


# ==============================================================================
# 시뮬레이션 카메라
# ==============================================================================

class SimulatedCamera:
    """시뮬레이션 카메라 (실제 Pi Camera 대체)"""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.resolution = resolution
        self.is_running = False
        self.frame_count = 0
        print(f"[시뮬레이션] 카메라 초기화: {resolution[0]}x{resolution[1]}")

    def start(self):
        """카메라 시작"""
        self.is_running = True
        print("[시뮬레이션] 카메라 시작")

    def stop(self):
        """카메라 중지"""
        self.is_running = False
        print("[시뮬레이션] 카메라 중지")

    def capture_frame(self):
        """프레임 캡처 (시뮬레이션)"""
        if not self.is_running:
            raise RuntimeError("카메라가 시작되지 않음")

        # 랜덤 노이즈로 프레임 생성
        width, height = self.resolution

        if HAS_NUMPY:
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            # 간단한 패턴 추가 (움직이는 사각형)
            x_offset = (self.frame_count * 5) % width
            y_offset = 100
            frame[y_offset:y_offset+50, x_offset:x_offset+50] = [255, 0, 0]
        else:
            # numpy 없을 때는 간단한 리스트로 시뮬레이션
            frame = [[[random.randint(0, 255) for _ in range(3)]
                      for _ in range(width)]
                     for _ in range(height)]

        self.frame_count += 1
        return frame

    def capture_image(self, filename: str):
        """이미지 저장 (시뮬레이션)"""
        frame = self.capture_frame()
        print(f"[시뮬레이션] 이미지 저장: {filename} ({frame.shape})")
        # 실제로는 PIL.Image.fromarray(frame).save(filename)


# ==============================================================================
# TFLite 객체 검출 (시뮬레이션)
# ==============================================================================

class TFLiteObjectDetector:
    """TFLite 객체 검출기 (시뮬레이션)"""

    COCO_LABELS = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe'
    ]

    def __init__(self, model_path: str = None, threshold: float = 0.5):
        self.model_path = model_path or "simulated_model.tflite"
        self.threshold = threshold
        self.input_size = (300, 300)
        print(f"[시뮬레이션] TFLite 모델 로드: {self.model_path}")
        print(f"  - 입력 크기: {self.input_size}")
        print(f"  - 임계값: {self.threshold}")

    def preprocess(self, frame):
        """전처리"""
        # 리사이즈 시뮬레이션
        target_h, target_w = self.input_size

        if HAS_NUMPY:
            # 실제로는 cv2.resize 사용
            resized = np.random.randint(0, 256, (target_h, target_w, 3), dtype=np.uint8)
            # 정규화
            normalized = resized.astype(np.float32) / 255.0
            return normalized
        else:
            # numpy 없을 때는 간단히 처리
            return frame

    def detect(self, frame) -> List[Detection]:
        """객체 검출 (시뮬레이션)"""
        # 전처리
        input_data = self.preprocess(frame)

        # 추론 시뮬레이션 (일부 확률로 객체 검출)
        detections = []

        # 랜덤하게 0-3개 객체 검출
        num_objects = random.choice([0, 0, 0, 1, 1, 2])  # 대부분 0, 가끔 1-2개

        if HAS_NUMPY:
            h, w = frame.shape[:2]
        else:
            h, w = len(frame), len(frame[0]) if frame else 0

        for _ in range(num_objects):
            # 랜덤 클래스 선택
            class_id = random.randint(0, min(len(self.COCO_LABELS), 10) - 1)
            class_name = self.COCO_LABELS[class_id]

            # 랜덤 신뢰도
            confidence = random.uniform(self.threshold, 1.0)

            # 랜덤 바운딩 박스
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = x1 + random.randint(50, w // 3)
            y2 = y1 + random.randint(50, h // 3)

            bbox = BoundingBox(
                x1=min(x1, w-1),
                y1=min(y1, h-1),
                x2=min(x2, w-1),
                y2=min(y2, h-1)
            )

            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                timestamp=datetime.now()
            )
            detections.append(detection)

        return detections

    def draw_detections(self, frame, detections: List[Detection]):
        """검출 결과 시각화 (시뮬레이션)"""
        if HAS_NUMPY:
            result = frame.copy()
        else:
            result = frame  # 시뮬레이션 모드에서는 복사하지 않음

        for det in detections:
            # 실제로는 cv2.rectangle, cv2.putText 사용
            label = f"{det.class_name}: {det.confidence:.2f}"
            print(f"  [검출] {label} at ({det.bbox.x1}, {det.bbox.y1})")

        return result


# ==============================================================================
# 모션 감지
# ==============================================================================

class MotionDetector:
    """모션 감지기"""

    def __init__(self, threshold: int = 30, min_area: int = 500):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
        self.motion_count = 0
        print(f"[시뮬레이션] 모션 감지기 초기화")
        print(f"  - 임계값: {threshold}")
        print(f"  - 최소 영역: {min_area} pixels")

    def detect_motion(self, frame) -> Tuple[bool, List[BoundingBox]]:
        """모션 감지"""
        if HAS_NUMPY:
            # 그레이스케일 변환 시뮬레이션
            gray = np.mean(frame, axis=2).astype(np.uint8)

            # 가우시안 블러 시뮬레이션
            # 실제로는 cv2.GaussianBlur 사용

            if self.prev_frame is None:
                self.prev_frame = gray
                return False, []

            # 프레임 차이 계산
            frame_delta = np.abs(gray.astype(np.int16) - self.prev_frame.astype(np.int16))

            # 임계값 적용
            thresh = (frame_delta > self.threshold).astype(np.uint8) * 255

            # 변화 영역 비율 계산
            motion_pixels = np.sum(thresh > 0)
            total_pixels = thresh.size
            motion_ratio = motion_pixels / total_pixels

            self.prev_frame = gray

            # 일정 비율 이상 변화 시 모션 감지
            motion_detected = motion_ratio > 0.05  # 5% 이상 변화

            h, w = frame.shape[:2]
        else:
            # numpy 없을 때 간단한 시뮬레이션
            if self.prev_frame is None:
                self.prev_frame = frame
                return False, []

            # 랜덤하게 모션 감지 (10% 확률)
            motion_detected = random.random() < 0.1
            h, w = len(frame), len(frame[0]) if frame else 0

        regions = []
        if motion_detected:
            # 시뮬레이션: 랜덤 모션 영역 생성
            num_regions = random.randint(1, 3)

            for _ in range(num_regions):
                x1 = random.randint(0, w // 2)
                y1 = random.randint(0, h // 2)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)

                bbox = BoundingBox(
                    x1=min(x1, w-1),
                    y1=min(y1, h-1),
                    x2=min(x2, w-1),
                    y2=min(y2, h-1)
                )

                if bbox.area >= self.min_area:
                    regions.append(bbox)

            self.motion_count += 1

        return motion_detected, regions


# ==============================================================================
# 영상 스트리밍 (개념)
# ==============================================================================

class VideoStreamer:
    """비디오 스트리머 (MJPEG over HTTP 개념)"""

    def __init__(self, camera: SimulatedCamera, port: int = 8080):
        self.camera = camera
        self.port = port
        self.is_streaming = False
        self.frame_rate = 30
        self.lock = threading.Lock()
        self.current_frame = None
        print(f"[시뮬레이션] 비디오 스트리머 초기화 (포트 {port})")

    def _capture_loop(self):
        """캡처 루프"""
        frame_interval = 1.0 / self.frame_rate

        while self.is_streaming:
            start_time = time.time()

            frame = self.camera.capture_frame()

            # JPEG 인코딩 시뮬레이션
            # 실제로는 PIL.Image.fromarray(frame).save(buffer, 'JPEG')

            with self.lock:
                self.current_frame = frame

            # 프레임 레이트 유지
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def start_streaming(self):
        """스트리밍 시작"""
        if self.is_streaming:
            return

        self.is_streaming = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[시뮬레이션] 스트리밍 시작: http://0.0.0.0:{self.port}/video_feed")

    def stop_streaming(self):
        """스트리밍 중지"""
        self.is_streaming = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print("[시뮬레이션] 스트리밍 중지")

    def get_frame(self):
        """현재 프레임 반환"""
        with self.lock:
            return self.current_frame


# ==============================================================================
# 결과 로깅
# ==============================================================================

class ResultLogger:
    """검출 및 모션 결과 로깅"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        print(f"[시뮬레이션] 로거 초기화: {self.log_file}")

    def log_detection(self, detections: List[Detection], frame_id: int):
        """객체 검출 로깅"""
        log_entry = {
            "type": "detection",
            "timestamp": datetime.now().isoformat(),
            "frame_id": frame_id,
            "count": len(detections),
            "objects": [
                {
                    "class": det.class_name,
                    "confidence": det.confidence,
                    "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                }
                for det in detections
            ]
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_motion(self, event: MotionEvent):
        """모션 이벤트 로깅"""
        log_entry = {
            "type": "motion",
            "timestamp": event.timestamp.isoformat(),
            "frame_id": event.frame_id,
            "region_count": len(event.regions),
            "total_area": event.area,
            "regions": [
                [r.x1, r.y1, r.x2, r.y2] for r in event.regions
            ]
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def get_statistics(self) -> Dict:
        """로그 통계"""
        if not os.path.exists(self.log_file):
            return {}

        detection_count = 0
        motion_count = 0
        object_classes = {}

        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['type'] == 'detection':
                    detection_count += 1
                    for obj in entry['objects']:
                        cls = obj['class']
                        object_classes[cls] = object_classes.get(cls, 0) + 1
                elif entry['type'] == 'motion':
                    motion_count += 1

        return {
            "detection_events": detection_count,
            "motion_events": motion_count,
            "object_classes": object_classes
        }


# ==============================================================================
# 성능 모니터링
# ==============================================================================

class PerformanceMonitor:
    """성능 모니터"""

    def __init__(self):
        self.metrics = {
            "fps": [],
            "detection_time": [],
            "frame_count": 0,
            "start_time": None
        }

    def start(self):
        """모니터링 시작"""
        self.metrics["start_time"] = time.time()

    def record_frame(self, processing_time: float):
        """프레임 처리 시간 기록"""
        self.metrics["frame_count"] += 1
        self.metrics["detection_time"].append(processing_time)

        if processing_time > 0:
            fps = 1.0 / processing_time
            self.metrics["fps"].append(fps)

    def get_report(self) -> Dict:
        """성능 보고서"""
        if self.metrics["start_time"] is None:
            return {}

        elapsed = time.time() - self.metrics["start_time"]

        if HAS_NUMPY:
            avg_fps = np.mean(self.metrics["fps"]) if self.metrics["fps"] else 0
            avg_detection_time = np.mean(self.metrics["detection_time"]) if self.metrics["detection_time"] else 0
        else:
            avg_fps = sum(self.metrics["fps"]) / len(self.metrics["fps"]) if self.metrics["fps"] else 0
            avg_detection_time = sum(self.metrics["detection_time"]) / len(self.metrics["detection_time"]) if self.metrics["detection_time"] else 0

        return {
            "total_frames": self.metrics["frame_count"],
            "elapsed_time": elapsed,
            "average_fps": avg_fps,
            "average_detection_time_ms": avg_detection_time * 1000,
            "frames_per_second_actual": self.metrics["frame_count"] / elapsed if elapsed > 0 else 0
        }


# ==============================================================================
# 통합 영상 분석 시스템
# ==============================================================================

class ImageAnalysisSystem:
    """통합 영상 분석 시스템"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}

        # 카메라
        resolution = config.get('resolution', (640, 480))
        self.camera = SimulatedCamera(resolution)

        # 객체 검출
        self.detector = TFLiteObjectDetector(
            threshold=config.get('detection_threshold', 0.5)
        )

        # 모션 감지
        self.motion_detector = MotionDetector(
            threshold=config.get('motion_threshold', 30),
            min_area=config.get('min_motion_area', 500)
        )

        # 로거
        self.logger = ResultLogger(log_dir=config.get('log_dir', 'logs'))

        # 성능 모니터
        self.perf_monitor = PerformanceMonitor()

        # 스트리밍 (옵션)
        self.enable_streaming = config.get('enable_streaming', False)
        if self.enable_streaming:
            self.streamer = VideoStreamer(self.camera, port=config.get('stream_port', 8080))

        self.is_running = False

        print("\n" + "="*60)
        print("영상 분석 시스템 초기화 완료")
        print("="*60)

    def run(self, duration: float = 60, detect_objects: bool = True, detect_motion: bool = True):
        """시스템 실행"""
        print(f"\n시스템 시작 (실행 시간: {duration}초)")
        print(f"  - 객체 검출: {'ON' if detect_objects else 'OFF'}")
        print(f"  - 모션 감지: {'ON' if detect_motion else 'OFF'}")
        print(f"  - 스트리밍: {'ON' if self.enable_streaming else 'OFF'}")
        print()

        self.camera.start()
        self.perf_monitor.start()

        if self.enable_streaming:
            self.streamer.start_streaming()

        self.is_running = True
        start_time = time.time()
        frame_id = 0

        try:
            while time.time() - start_time < duration and self.is_running:
                frame_start = time.time()

                # 프레임 캡처
                frame = self.camera.capture_frame()
                frame_id += 1

                # 객체 검출
                if detect_objects:
                    detections = self.detector.detect(frame)
                    if detections:
                        print(f"[프레임 {frame_id}] 객체 검출: {len(detections)}개")
                        for det in detections:
                            print(f"  - {det.class_name} (신뢰도: {det.confidence:.2f})")
                        self.logger.log_detection(detections, frame_id)

                # 모션 감지
                if detect_motion:
                    motion_detected, regions = self.motion_detector.detect_motion(frame)
                    if motion_detected:
                        total_area = sum(r.area for r in regions)
                        event = MotionEvent(
                            regions=regions,
                            area=total_area,
                            timestamp=datetime.now(),
                            frame_id=frame_id
                        )
                        print(f"[프레임 {frame_id}] 모션 감지: {len(regions)}개 영역 (면적: {total_area})")
                        self.logger.log_motion(event)

                # 성능 기록
                processing_time = time.time() - frame_start
                self.perf_monitor.record_frame(processing_time)

                # 프레임 레이트 조절
                sleep_time = max(0, 0.1 - processing_time)  # ~10 FPS
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n사용자 중단")

        finally:
            self._cleanup()

    def _cleanup(self):
        """정리"""
        self.is_running = False
        self.camera.stop()

        if self.enable_streaming:
            self.streamer.stop_streaming()

        # 통계 출력
        print("\n" + "="*60)
        print("실행 완료 - 통계")
        print("="*60)

        # 성능 통계
        perf_report = self.perf_monitor.get_report()
        print("\n[성능]")
        print(f"  총 프레임: {perf_report.get('total_frames', 0)}")
        print(f"  실행 시간: {perf_report.get('elapsed_time', 0):.1f}초")
        print(f"  평균 FPS: {perf_report.get('average_fps', 0):.1f}")
        print(f"  평균 검출 시간: {perf_report.get('average_detection_time_ms', 0):.1f}ms")

        # 로그 통계
        log_stats = self.logger.get_statistics()
        print("\n[검출 통계]")
        print(f"  검출 이벤트: {log_stats.get('detection_events', 0)}회")
        print(f"  모션 이벤트: {log_stats.get('motion_events', 0)}회")

        object_classes = log_stats.get('object_classes', {})
        if object_classes:
            print("\n[검출된 객체]")
            for cls, count in sorted(object_classes.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {count}회")

        print("\n" + "="*60)


# ==============================================================================
# 메인 실행
# ==============================================================================

def main():
    """메인 함수"""
    print("영상 분석 프로젝트 - 시뮬레이션 모드")
    print("="*60)
    print("이 프로그램은 실제 Pi Camera 없이 시뮬레이션으로 동작합니다.")
    print()

    # 설정
    config = {
        'resolution': (640, 480),
        'detection_threshold': 0.6,
        'motion_threshold': 30,
        'min_motion_area': 500,
        'log_dir': 'logs',
        'enable_streaming': False,  # 실제 Flask 서버는 실행하지 않음
        'stream_port': 8080
    }

    # 시스템 생성
    system = ImageAnalysisSystem(config)

    # 실행 (테스트용으로 짧게 설정, 실제 사용시 duration 증가)
    import sys
    test_mode = '--test' in sys.argv
    duration = 5 if test_mode else 30

    system.run(
        duration=duration,  # 테스트: 5초, 일반: 30초
        detect_objects=True,
        detect_motion=True
    )


if __name__ == "__main__":
    main()
