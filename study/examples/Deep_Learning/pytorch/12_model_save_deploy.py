"""
12. 모델 저장 및 배포

PyTorch 모델 저장, TorchScript, ONNX 변환을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tempfile

print("=" * 60)
print("PyTorch 모델 저장 및 배포")
print("=" * 60)


# ============================================
# 1. 샘플 모델
# ============================================
print("\n[1] 샘플 모델")
print("-" * 40)

class SimpleClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_classes': num_classes
        }
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleClassifier()
print(f"모델 구조:\n{model}")
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")


# ============================================
# 2. state_dict 저장
# ============================================
print("\n[2] state_dict 저장")
print("-" * 40)

# 임시 디렉토리 사용
save_dir = tempfile.mkdtemp()

# 저장
weights_path = os.path.join(save_dir, 'model_weights.pth')
torch.save(model.state_dict(), weights_path)
print(f"저장: {weights_path}")
print(f"파일 크기: {os.path.getsize(weights_path) / 1024:.2f} KB")

# 로드
loaded_model = SimpleClassifier()
loaded_model.load_state_dict(torch.load(weights_path, weights_only=True))
loaded_model.eval()

# 검증
x = torch.randn(2, 1, 28, 28)
model.eval()
with torch.no_grad():
    original_out = model(x)
    loaded_out = loaded_model(x)
    diff = (original_out - loaded_out).abs().max().item()
    print(f"출력 차이: {diff:.10f}")


# ============================================
# 3. 체크포인트 저장
# ============================================
print("\n[3] 체크포인트 저장")
print("-" * 40)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 가짜 학습 상태
epoch = 10
loss = 0.123
best_acc = 0.95

# 체크포인트 저장
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_acc': best_acc,
    'model_config': model.config
}

checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
torch.save(checkpoint, checkpoint_path)
print(f"체크포인트 저장: {checkpoint_path}")

# 체크포인트 로드
loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
print(f"로드된 epoch: {loaded_checkpoint['epoch']}")
print(f"로드된 best_acc: {loaded_checkpoint['best_acc']}")
print(f"모델 설정: {loaded_checkpoint['model_config']}")


# ============================================
# 4. TorchScript - Tracing
# ============================================
print("\n[4] TorchScript - Tracing")
print("-" * 40)

model.eval()
example_input = torch.randn(1, 1, 28, 28)

# Trace
traced_model = torch.jit.trace(model, example_input)

# 저장
traced_path = os.path.join(save_dir, 'model_traced.pt')
traced_model.save(traced_path)
print(f"TorchScript 저장: {traced_path}")
print(f"파일 크기: {os.path.getsize(traced_path) / 1024:.2f} KB")

# 로드 및 검증
loaded_traced = torch.jit.load(traced_path)
with torch.no_grad():
    traced_out = loaded_traced(example_input)
    original_out = model(example_input)
    diff = (traced_out - original_out).abs().max().item()
    print(f"출력 차이: {diff:.10f}")


# ============================================
# 5. TorchScript - Scripting
# ============================================
print("\n[5] TorchScript - Scripting")
print("-" * 40)

class ConditionalModel(nn.Module):
    """조건문이 있는 모델"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, use_relu: bool = True):
        x = self.fc(x)
        if use_relu:
            x = F.relu(x)
        return x

cond_model = ConditionalModel()
scripted_model = torch.jit.script(cond_model)

scripted_path = os.path.join(save_dir, 'model_scripted.pt')
scripted_model.save(scripted_path)
print(f"Scripted 모델 저장: {scripted_path}")

# 조건부 실행 테스트
x = torch.randn(2, 10)
out_relu = scripted_model(x, True)
out_no_relu = scripted_model(x, False)
print(f"ReLU 적용: min={out_relu.min():.4f}")
print(f"ReLU 미적용: min={out_no_relu.min():.4f}")


# ============================================
# 6. ONNX 변환
# ============================================
print("\n[6] ONNX 변환")
print("-" * 40)

try:
    import onnx

    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)

    onnx_path = os.path.join(save_dir, 'model.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )

    print(f"ONNX 저장: {onnx_path}")
    print(f"파일 크기: {os.path.getsize(onnx_path) / 1024:.2f} KB")

    # 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델 검증 통과")

except ImportError:
    print("onnx 미설치 - 스킵")


# ============================================
# 7. ONNX Runtime 추론
# ============================================
print("\n[7] ONNX Runtime 추론")
print("-" * 40)

try:
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 추론
    input_data = np.random.randn(2, 1, 28, 28).astype(np.float32)
    result = session.run([output_name], {input_name: input_data})

    print(f"ONNX Runtime 출력: {result[0].shape}")

    # PyTorch 결과와 비교
    model.eval()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(input_data))
        diff = np.abs(result[0] - torch_out.numpy()).max()
        print(f"PyTorch vs ONNX 차이: {diff:.6f}")

except ImportError:
    print("onnxruntime 미설치 - 스킵")


# ============================================
# 8. 양자화
# ============================================
print("\n[8] 양자화 (Quantization)")
print("-" * 40)

# 동적 양자화
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 크기 비교
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
quantized_size = sum(
    p.numel() * p.element_size() for p in quantized_model.parameters()
    if p.dtype != torch.qint8
)

print(f"원본 모델 크기: {original_size / 1024:.2f} KB")
print(f"양자화 모델 (일부 층): 약 {original_size / 1024 * 0.25:.2f} KB (추정)")

# 추론 비교
x = torch.randn(100, 1, 28, 28)

model.eval()
quantized_model.eval()

import time

# 원본 모델
start = time.time()
for _ in range(10):
    with torch.no_grad():
        _ = model(x)
original_time = time.time() - start

# 양자화 모델
start = time.time()
for _ in range(10):
    with torch.no_grad():
        _ = quantized_model(x)
quantized_time = time.time() - start

print(f"원본 추론 시간: {original_time*1000:.2f} ms")
print(f"양자화 추론 시간: {quantized_time*1000:.2f} ms")


# ============================================
# 9. 추론 최적화
# ============================================
print("\n[9] 추론 최적화")
print("-" * 40)

model.eval()
x = torch.randn(100, 1, 28, 28)

# no_grad
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(x)
no_grad_time = time.time() - start

# inference_mode (더 빠름)
start = time.time()
for _ in range(100):
    with torch.inference_mode():
        _ = model(x)
inference_time = time.time() - start

print(f"no_grad 시간: {no_grad_time*1000:.2f} ms")
print(f"inference_mode 시간: {inference_time*1000:.2f} ms")
print(f"개선: {(no_grad_time - inference_time) / no_grad_time * 100:.1f}%")


# ============================================
# 10. 모바일 최적화
# ============================================
print("\n[10] 모바일 최적화")
print("-" * 40)

try:
    # 모바일용 최적화
    traced_model = torch.jit.trace(model.eval(), example_input)
    optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)

    mobile_path = os.path.join(save_dir, 'model_mobile.ptl')
    optimized_model._save_for_lite_interpreter(mobile_path)

    print(f"모바일 모델 저장: {mobile_path}")
    print(f"파일 크기: {os.path.getsize(mobile_path) / 1024:.2f} KB")
except Exception as e:
    print(f"모바일 최적화 스킵: {e}")


# ============================================
# 11. 저장된 파일 목록
# ============================================
print("\n[11] 저장된 파일 목록")
print("-" * 40)

print(f"저장 디렉토리: {save_dir}")
for f in os.listdir(save_dir):
    path = os.path.join(save_dir, f)
    size = os.path.getsize(path) / 1024
    print(f"  {f}: {size:.2f} KB")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("모델 저장 및 배포 정리")
print("=" * 60)

summary = """
저장 방법:

1. state_dict (권장)
   torch.save(model.state_dict(), 'model.pth')
   model.load_state_dict(torch.load('model.pth'))

2. 체크포인트
   checkpoint = {'model': model.state_dict(), 'optimizer': ...}
   torch.save(checkpoint, 'checkpoint.pth')

3. TorchScript
   traced = torch.jit.trace(model, example_input)
   traced.save('model.pt')

4. ONNX
   torch.onnx.export(model, input, 'model.onnx')

추론 최적화:
   - model.eval()
   - torch.inference_mode()
   - 양자화 (quantize_dynamic)

배포 옵션:
   - FastAPI/Flask: 웹 API
   - ONNX Runtime: 범용 추론
   - TorchScript: C++ 배포
   - PyTorch Mobile: 모바일 앱
"""
print(summary)
print("=" * 60)

# 임시 파일 정리 안내
print(f"\n임시 파일 위치: {save_dir}")
print("(자동 삭제되지 않음 - 필요시 수동 삭제)")
