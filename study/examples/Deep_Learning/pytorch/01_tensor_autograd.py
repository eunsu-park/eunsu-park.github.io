"""
01. 텐서와 오토그래드 - PyTorch 버전

PyTorch의 핵심 기능인 텐서 연산과 자동 미분을 학습합니다.
NumPy 버전(examples/numpy/01_tensor_basics.py)과 비교해 보세요.
"""

import torch
import numpy as np

print("=" * 60)
print("PyTorch 텐서와 오토그래드")
print("=" * 60)


# ============================================
# 1. 텐서 생성
# ============================================
print("\n[1] 텐서 생성")
print("-" * 40)

# 리스트에서 생성
tensor1 = torch.tensor([1, 2, 3, 4])
print(f"리스트 → 텐서: {tensor1}")
print(f"  shape: {tensor1.shape}, dtype: {tensor1.dtype}")

# 특수 텐서
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
rand = torch.randn(2, 3)  # 표준 정규 분포
arange = torch.arange(0, 10, 2)

print(f"zeros(3,4): shape {zeros.shape}")
print(f"randn(2,3):\n{rand}")

# dtype 지정
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"float32 텐서: {float_tensor}")


# ============================================
# 2. NumPy와 변환
# ============================================
print("\n[2] NumPy 변환")
print("-" * 40)

# NumPy → PyTorch
np_arr = np.array([1.0, 2.0, 3.0])
torch_from_np = torch.from_numpy(np_arr)
print(f"NumPy → PyTorch: {torch_from_np}")

# 주의: 메모리 공유됨
np_arr[0] = 100
print(f"NumPy 수정 후 PyTorch: {torch_from_np}")  # 같이 변경됨

# PyTorch → NumPy
pt_tensor = torch.tensor([4.0, 5.0, 6.0])
np_from_torch = pt_tensor.numpy()
print(f"PyTorch → NumPy: {np_from_torch}")


# ============================================
# 3. 텐서 연산
# ============================================
print("\n[3] 텐서 연산")
print("-" * 40)

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 요소별 연산
print(f"a + b:\n{a + b}")
print(f"a * b (요소별):\n{a * b}")

# 행렬 곱셈
print(f"a @ b (행렬 곱):\n{a @ b}")
print(f"torch.matmul(a, b):\n{torch.matmul(a, b)}")

# 통계
print(f"a.sum(): {a.sum()}")
print(f"a.mean(): {a.mean()}")
print(f"a.max(): {a.max()}")


# ============================================
# 4. 브로드캐스팅
# ============================================
print("\n[4] 브로드캐스팅")
print("-" * 40)

x = torch.tensor([[1], [2], [3]])  # (3, 1)
y = torch.tensor([10, 20, 30])     # (3,)

result = x + y  # (3, 3)으로 자동 확장
print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"x + y shape: {result.shape}")
print(f"x + y:\n{result}")


# ============================================
# 5. 자동 미분 (Autograd) 기초
# ============================================
print("\n[5] 자동 미분 (Autograd)")
print("-" * 40)

# requires_grad=True로 미분 추적 활성화
x = torch.tensor([2.0], requires_grad=True)
print(f"x: {x}, requires_grad: {x.requires_grad}")

# 순전파
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1
print(f"y = x² + 3x + 1 = {y.item()}")

# 역전파
y.backward()

# 기울기 확인 (dy/dx = 2x + 3 = 2*2 + 3 = 7)
print(f"dy/dx at x=2: {x.grad.item()}")
print("검증: dy/dx = 2x + 3 = 2*2 + 3 = 7 ✓")


# ============================================
# 6. 복잡한 함수의 자동 미분
# ============================================
print("\n[6] 복잡한 함수 미분")
print("-" * 40)

# f(x) = x³ + 2x² - 5x + 3
# f'(x) = 3x² + 4x - 5
# f'(2) = 12 + 8 - 5 = 15

x = torch.tensor([2.0], requires_grad=True)
f = x**3 + 2*x**2 - 5*x + 3

f.backward()
print(f"f(x) = x³ + 2x² - 5x + 3")
print(f"f(2) = {f.item()}")
print(f"f'(2) = {x.grad.item()}")
print("검증: f'(x) = 3x² + 4x - 5 = 12 + 8 - 5 = 15 ✓")


# ============================================
# 7. 다변수 함수의 미분 (Gradient)
# ============================================
print("\n[7] 다변수 함수 미분")
print("-" * 40)

# f(x, y) = x² + y² + xy
# ∂f/∂x = 2x + y
# ∂f/∂y = 2y + x

x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([4.0], requires_grad=True)

f = x**2 + y**2 + x*y
f.backward()

print(f"f(x, y) = x² + y² + xy")
print(f"f(3, 4) = {f.item()}")
print(f"∂f/∂x at (3,4) = {x.grad.item()}")  # 2*3 + 4 = 10
print(f"∂f/∂y at (3,4) = {y.grad.item()}")  # 2*4 + 3 = 11


# ============================================
# 8. 기울기 초기화
# ============================================
print("\n[8] 기울기 초기화")
print("-" * 40)

x = torch.tensor([1.0], requires_grad=True)

# 첫 번째 역전파
y1 = x * 2
y1.backward()
print(f"첫 번째 grad: {x.grad}")

# 기울기가 누적됨!
y2 = x * 3
y2.backward()
print(f"누적된 grad: {x.grad}")  # 2 + 3 = 5

# 초기화 후 다시
x.grad.zero_()  # 중요!
y3 = x * 4
y3.backward()
print(f"초기화 후 grad: {x.grad}")


# ============================================
# 9. GPU 연산
# ============================================
print("\n[9] GPU 연산")
print("-" * 40)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # GPU로 텐서 이동
    x_cpu = torch.randn(1000, 1000)
    x_gpu = x_cpu.to(device)

    # GPU에서 연산
    y_gpu = x_gpu @ x_gpu

    # 결과를 CPU로
    y_cpu = y_gpu.cpu()
    print(f"GPU 행렬 곱셈 완료: {y_cpu.shape}")
else:
    print("GPU 사용 불가, CPU 모드로 실행")


# ============================================
# 10. no_grad 컨텍스트
# ============================================
print("\n[10] no_grad 컨텍스트")
print("-" * 40)

x = torch.tensor([1.0], requires_grad=True)

# 일반 연산 (기울기 추적)
y = x * 2
print(f"일반 연산: requires_grad = {y.requires_grad}")

# no_grad 내부 (기울기 추적 안 함)
with torch.no_grad():
    z = x * 2
    print(f"no_grad 내부: requires_grad = {z.requires_grad}")

# detach로 분리
w = x.detach() * 2
print(f"detach 후: requires_grad = {w.requires_grad}")


print("\n" + "=" * 60)
print("PyTorch 텐서와 오토그래드 완료!")
print("NumPy 버전과 비교: examples/numpy/01_tensor_basics.py")
print("=" * 60)
