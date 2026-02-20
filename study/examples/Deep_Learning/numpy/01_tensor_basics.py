"""
01. 텐서 기초 - NumPy 버전

NumPy로 텐서 연산과 수동 미분을 구현합니다.
PyTorch 버전(examples/pytorch/01_tensor_autograd.py)과 비교해 보세요.

핵심 차이점:
- NumPy: 자동 미분 없음, 직접 미분 계산
- PyTorch: autograd로 자동 미분
"""

import numpy as np

print("=" * 60)
print("NumPy 텐서 기초와 수동 미분")
print("=" * 60)


# ============================================
# 1. 배열 생성 (텐서)
# ============================================
print("\n[1] 배열 생성")
print("-" * 40)

# 리스트에서 생성
arr1 = np.array([1, 2, 3, 4])
print(f"리스트 → 배열: {arr1}")
print(f"  shape: {arr1.shape}, dtype: {arr1.dtype}")

# 특수 배열
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
rand = np.random.randn(2, 3)  # 표준 정규 분포
arange = np.arange(0, 10, 2)

print(f"zeros(3,4): shape {zeros.shape}")
print(f"randn(2,3):\n{rand}")

# dtype 지정
float_arr = np.array([1, 2, 3], dtype=np.float32)
print(f"float32 배열: {float_arr}")


# ============================================
# 2. 배열 연산
# ============================================
print("\n[2] 배열 연산")
print("-" * 40)

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[5, 6], [7, 8]], dtype=np.float32)

# 요소별 연산
print(f"a + b:\n{a + b}")
print(f"a * b (요소별):\n{a * b}")

# 행렬 곱셈
print(f"a @ b (행렬 곱):\n{a @ b}")
print(f"np.dot(a, b):\n{np.dot(a, b)}")

# 통계
print(f"a.sum(): {a.sum()}")
print(f"a.mean(): {a.mean()}")
print(f"a.max(): {a.max()}")


# ============================================
# 3. 브로드캐스팅
# ============================================
print("\n[3] 브로드캐스팅")
print("-" * 40)

x = np.array([[1], [2], [3]])  # (3, 1)
y = np.array([10, 20, 30])     # (3,)

result = x + y  # (3, 3)으로 자동 확장
print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"x + y shape: {result.shape}")
print(f"x + y:\n{result}")


# ============================================
# 4. 수동 미분 - 기본
# ============================================
print("\n[4] 수동 미분 - 기본")
print("-" * 40)

# y = x² + 3x + 1
# dy/dx = 2x + 3

def f1(x):
    """순전파: y = x² + 3x + 1"""
    return x**2 + 3*x + 1

def df1(x):
    """수동 미분: dy/dx = 2x + 3"""
    return 2*x + 3

x = 2.0
print(f"f(x) = x² + 3x + 1")
print(f"f({x}) = {f1(x)}")
print(f"f'({x}) = {df1(x)}")  # 2*2 + 3 = 7
print("검증: dy/dx = 2x + 3 = 2*2 + 3 = 7 ✓")


# ============================================
# 5. 수동 미분 - 복잡한 함수
# ============================================
print("\n[5] 수동 미분 - 복잡한 함수")
print("-" * 40)

# f(x) = x³ + 2x² - 5x + 3
# f'(x) = 3x² + 4x - 5

def f2(x):
    """순전파"""
    return x**3 + 2*x**2 - 5*x + 3

def df2(x):
    """수동 미분"""
    return 3*x**2 + 4*x - 5

x = 2.0
print(f"f(x) = x³ + 2x² - 5x + 3")
print(f"f({x}) = {f2(x)}")
print(f"f'({x}) = {df2(x)}")  # 3*4 + 4*2 - 5 = 15
print("검증: f'(x) = 3x² + 4x - 5 = 12 + 8 - 5 = 15 ✓")


# ============================================
# 6. 수동 미분 - 다변수 함수
# ============================================
print("\n[6] 수동 미분 - 다변수 함수")
print("-" * 40)

# f(x, y) = x² + y² + xy
# ∂f/∂x = 2x + y
# ∂f/∂y = 2y + x

def f3(x, y):
    """순전파"""
    return x**2 + y**2 + x*y

def df3_dx(x, y):
    """편미분 ∂f/∂x"""
    return 2*x + y

def df3_dy(x, y):
    """편미분 ∂f/∂y"""
    return 2*y + x

x, y = 3.0, 4.0
print(f"f(x, y) = x² + y² + xy")
print(f"f({x}, {y}) = {f3(x, y)}")
print(f"∂f/∂x at ({x},{y}) = {df3_dx(x, y)}")  # 2*3 + 4 = 10
print(f"∂f/∂y at ({x},{y}) = {df3_dy(x, y)}")  # 2*4 + 3 = 11


# ============================================
# 7. 수치 미분 (Numerical Differentiation)
# ============================================
print("\n[7] 수치 미분")
print("-" * 40)

def numerical_gradient(f, x, h=1e-5):
    """
    중앙 차분법으로 수치 미분 계산
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# f(x) = x³ + 2x² - 5x + 3 테스트
x = 2.0
numerical_grad = numerical_gradient(f2, x)
analytical_grad = df2(x)

print(f"해석적 미분: {analytical_grad}")
print(f"수치 미분:   {numerical_grad:.10f}")
print(f"오차:        {abs(numerical_grad - analytical_grad):.2e}")


# ============================================
# 8. 벡터 입력에 대한 미분
# ============================================
print("\n[8] 벡터 입력 미분")
print("-" * 40)

def f_vec(x):
    """f(x) = sum(x²) = x₁² + x₂² + x₃²"""
    return np.sum(x**2)

def df_vec(x):
    """∇f = [2x₁, 2x₂, 2x₃]"""
    return 2 * x

x = np.array([1.0, 2.0, 3.0])
print(f"f(x) = sum(x²)")
print(f"x = {x}")
print(f"f(x) = {f_vec(x)}")
print(f"∇f(x) = {df_vec(x)}")


# ============================================
# 9. 체인 룰 (Chain Rule) 예시
# ============================================
print("\n[9] 체인 룰 (Chain Rule)")
print("-" * 40)

# h(x) = f(g(x))
# g(x) = x²
# f(u) = sin(u)
# h(x) = sin(x²)
# dh/dx = df/du * dg/dx = cos(x²) * 2x

def g(x):
    return x**2

def f(u):
    return np.sin(u)

def h(x):
    return f(g(x))  # h(x) = sin(x²)

def dh_dx(x):
    """체인 룰: dh/dx = cos(x²) * 2x"""
    return np.cos(x**2) * (2*x)

x = 1.0
print(f"g(x) = x², f(u) = sin(u)")
print(f"h(x) = f(g(x)) = sin(x²)")
print(f"h({x}) = {h(x):.6f}")
print(f"dh/dx at x={x}: {dh_dx(x):.6f}")
print("체인 룰: dh/dx = cos(x²) * 2x")


# ============================================
# 10. 손실 함수와 미분 예시
# ============================================
print("\n[10] 손실 함수와 미분")
print("-" * 40)

def mse_loss(y_pred, y_true):
    """MSE: L = (1/n) * Σ(y_pred - y_true)²"""
    return np.mean((y_pred - y_true)**2)

def mse_gradient(y_pred, y_true):
    """∂L/∂y_pred = (2/n) * (y_pred - y_true)"""
    n = len(y_pred)
    return (2/n) * (y_pred - y_true)

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.2, 2.8])

loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true)

print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"MSE Loss: {loss:.4f}")
print(f"Gradient: {grad}")


# ============================================
# NumPy vs PyTorch 정리
# ============================================
print("\n" + "=" * 60)
print("NumPy vs PyTorch 비교")
print("=" * 60)

comparison = """
| 기능        | NumPy                | PyTorch                    |
|-------------|----------------------|----------------------------|
| 배열 생성    | np.array()          | torch.tensor()             |
| 미분        | 직접 구현 필요        | .backward() 자동 계산       |
| GPU         | 지원 안 함           | .to('cuda') 지원           |
| 장점        | 알고리즘 원리 이해    | 빠른 개발, 자동 미분        |
"""
print(comparison)

print("NumPy 텐서 기초와 수동 미분 완료!")
print("PyTorch 버전과 비교: examples/pytorch/01_tensor_autograd.py")
print("=" * 60)
