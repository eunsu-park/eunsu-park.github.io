"""
TorchServe Custom Handler Example
=================================

TorchServe에서 사용할 커스텀 핸들러 예제입니다.

사용 방법:
    1. 모델 아카이브 생성:
       torch-model-archiver --model-name mymodel \\
           --version 1.0 \\
           --serialized-file model.pt \\
           --handler torchserve_handler.py \\
           --export-path model_store

    2. TorchServe 시작:
       torchserve --start --model-store model_store --models mymodel=mymodel.mar

    3. 예측 요청:
       curl -X POST http://localhost:8080/predictions/mymodel \\
           -H "Content-Type: application/json" \\
           -d '{"data": [1.0, 2.0, 3.0, 4.0]}'
"""

import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class ChurnPredictionHandler(BaseHandler):
    """
    고객 이탈 예측 모델 핸들러

    이 핸들러는 다음을 수행합니다:
    1. 모델 초기화 및 로드
    2. 입력 데이터 전처리
    3. 추론 수행
    4. 결과 후처리
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.device = None
        self.class_names = None
        self.feature_names = None

    def initialize(self, context):
        """
        모델 초기화

        Args:
            context: TorchServe 컨텍스트 객체
        """
        logger.info("Initializing model...")

        # 컨텍스트에서 정보 추출
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # 디바이스 설정
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.device = torch.device(f"cuda:{properties.get('gpu_id')}")
            logger.info(f"Using GPU: {properties.get('gpu_id')}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        # 모델 로드
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # 추가 설정 파일 로드
        self._load_config(model_dir)

        self.initialized = True
        logger.info("Model initialization complete")

    def _load_config(self, model_dir):
        """설정 파일 로드"""
        # 클래스 이름
        class_file = os.path.join(model_dir, "index_to_name.json")
        if os.path.exists(class_file):
            with open(class_file) as f:
                self.class_names = json.load(f)
            logger.info(f"Loaded class names: {self.class_names}")
        else:
            self.class_names = {"0": "not_churned", "1": "churned"}

        # 피처 이름
        feature_file = os.path.join(model_dir, "feature_names.json")
        if os.path.exists(feature_file):
            with open(feature_file) as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded feature names: {self.feature_names}")

    def preprocess(self, data):
        """
        입력 데이터 전처리

        Args:
            data: 요청 데이터 리스트

        Returns:
            torch.Tensor: 전처리된 입력 텐서
        """
        logger.info(f"Preprocessing {len(data)} samples")
        inputs = []

        for row in data:
            # 요청 데이터 파싱
            if isinstance(row, dict):
                features = row.get("data") or row.get("body")
            else:
                features = row.get("body")

            # 바이트 데이터 처리
            if isinstance(features, (bytes, bytearray)):
                features = json.loads(features.decode("utf-8"))

            # JSON 문자열 처리
            if isinstance(features, str):
                features = json.loads(features)

            # dict인 경우 값만 추출
            if isinstance(features, dict):
                if "data" in features:
                    features = features["data"]
                else:
                    features = list(features.values())

            # 텐서로 변환
            tensor = torch.tensor(features, dtype=torch.float32)
            inputs.append(tensor)

        # 배치로 묶기
        batch = torch.stack(inputs).to(self.device)
        logger.info(f"Input batch shape: {batch.shape}")

        return batch

    def inference(self, data):
        """
        모델 추론

        Args:
            data: 전처리된 입력 텐서

        Returns:
            torch.Tensor: 모델 출력
        """
        logger.info("Running inference...")
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(data)

            # 확률로 변환 (분류 모델인 경우)
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                probabilities = F.softmax(outputs, dim=1)
            else:
                probabilities = torch.sigmoid(outputs)

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.4f}s")

        return probabilities

    def postprocess(self, data):
        """
        출력 후처리

        Args:
            data: 모델 출력 텐서

        Returns:
            list: JSON 직렬화 가능한 결과 리스트
        """
        logger.info("Postprocessing results...")
        results = []

        for prob in data:
            prob_list = prob.cpu().numpy().tolist()

            # 이진 분류
            if len(prob_list) == 1:
                prediction = 1 if prob_list[0] > 0.5 else 0
                probabilities = [1 - prob_list[0], prob_list[0]]
            # 다중 클래스
            else:
                prediction = int(torch.argmax(prob).item())
                probabilities = prob_list

            result = {
                "prediction": prediction,
                "probabilities": probabilities,
                "confidence": max(probabilities)
            }

            # 클래스 이름 추가
            if self.class_names:
                result["class_name"] = self.class_names.get(
                    str(prediction),
                    f"class_{prediction}"
                )

            results.append(result)

        logger.info(f"Processed {len(results)} results")
        return results

    def handle(self, data, context):
        """
        전체 요청 처리 (preprocess -> inference -> postprocess)

        TorchServe가 호출하는 메인 메서드
        """
        if not self.initialized:
            self.initialize(context)

        if data is None:
            return None

        # 전처리
        model_input = self.preprocess(data)

        # 추론
        model_output = self.inference(model_input)

        # 후처리
        return self.postprocess(model_output)


# 핸들러 인스턴스 (TorchServe가 로드)
_service = ChurnPredictionHandler()


def handle(data, context):
    """TorchServe 엔트리 포인트"""
    return _service.handle(data, context)


# ============================================================
# 로컬 테스트용 코드
# ============================================================

if __name__ == "__main__":
    import torch.nn as nn

    # 간단한 테스트 모델
    class SimpleModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 모델 생성 및 저장
    print("테스트 모델 생성...")
    model = SimpleModel(4, 10, 2)
    model.eval()

    # TorchScript로 저장
    scripted = torch.jit.script(model)
    scripted.save("test_model.pt")
    print("모델 저장: test_model.pt")

    # 핸들러 테스트
    print("\n핸들러 테스트...")

    # Mock 컨텍스트
    class MockContext:
        manifest = {"model": {"serializedFile": "test_model.pt"}}
        system_properties = {"model_dir": ".", "gpu_id": None}

    handler = ChurnPredictionHandler()
    handler.initialize(MockContext())

    # 테스트 요청
    test_data = [
        {"data": [1.0, 2.0, 3.0, 4.0]},
        {"data": [5.0, 6.0, 7.0, 8.0]}
    ]

    results = handler.handle(test_data, MockContext())

    print("\n결과:")
    for i, result in enumerate(results):
        print(f"  샘플 {i+1}:")
        print(f"    예측: {result['prediction']}")
        print(f"    확률: {result['probabilities']}")
        print(f"    신뢰도: {result['confidence']:.4f}")

    # 정리
    import os
    os.remove("test_model.pt")
    print("\n테스트 완료!")
