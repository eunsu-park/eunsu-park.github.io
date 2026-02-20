"""
03. 역전파 (Backpropagation) - NumPy 버전

NumPy로 역전파를 직접 구현하여 원리를 이해합니다.
이 파일이 딥러닝 이해의 핵심입니다!

PyTorch에서는 loss.backward() 한 줄이지만,
여기서는 체인 룰을 직접 적용합니다.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("NumPy 역전파 (Backpropagation) from scratch")
print("=" * 60)


# ============================================
# 1. 활성화 함수와 그 미분
# ============================================
print("\n[1] 활성화 함수와 미분")
print("-" * 40)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    """relu'(x) = 1 if x > 0 else 0"""
    return (x > 0).astype(float)

# 테스트
x = np.array([-2, -1, 0, 1, 2])
print(f"x: {x}")
print(f"sigmoid(x): {sigmoid(x).round(4)}")
print(f"sigmoid'(x): {sigmoid_derivative(x).round(4)}")
print(f"relu(x): {relu(x)}")
print(f"relu'(x): {relu_derivative(x)}")


# ============================================
# 2. 단일 뉴런 역전파 (이해용)
# ============================================
print("\n[2] 단일 뉴런 역전파")
print("-" * 40)

class SingleNeuron:
    """
    단일 뉴런: y = sigmoid(w*x + b)
    손실: L = (y - target)^2
    """
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def forward(self, x, target):
        """순전파"""
        self.x = x
        self.target = target

        # 단계별 계산 (캐시에 저장)
        self.z = self.w * x + self.b      # 선형 변환
        self.a = sigmoid(self.z)           # 활성화
        self.loss = (self.a - target) ** 2 # MSE

        return self.a, self.loss

    def backward(self):
        """
        역전파: 체인 룰 적용

        dL/dw = (dL/da) * (da/dz) * (dz/dw)
        dL/db = (dL/da) * (da/dz) * (dz/db)
        """
        # 1. 손실 → 활성화
        dL_da = 2 * (self.a - self.target)

        # 2. 활성화 → 선형 (시그모이드 미분)
        da_dz = sigmoid_derivative(self.z)

        # 3. 선형 → 가중치/편향
        dz_dw = self.x
        dz_db = 1

        # 체인 룰 적용
        dL_dw = dL_da * da_dz * dz_dw
        dL_db = dL_da * da_dz * dz_db

        return dL_dw, dL_db

# 테스트
neuron = SingleNeuron()
x, target = 2.0, 1.0

print(f"입력: x={x}, target={target}")
print(f"초기 가중치: w={neuron.w:.4f}, b={neuron.b:.4f}")

pred, loss = neuron.forward(x, target)
print(f"예측: {pred:.4f}, 손실: {loss:.4f}")

dw, db = neuron.backward()
print(f"기울기: dL/dw={dw:.4f}, dL/db={db:.4f}")


# ============================================
# 3. 2층 MLP 역전파 (핵심!)
# ============================================
print("\n[3] 2층 MLP 역전파")
print("-" * 40)

class MLPFromScratch:
    """
    2층 MLP with 역전파

    구조: 입력 → [W1, b1] → ReLU → [W2, b2] → Sigmoid → 출력
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier 초기화
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        print(f"MLP 생성: {input_dim} → {hidden_dim} → {output_dim}")

    def forward(self, X):
        """순전파 (중간값 캐시)"""
        # 첫 번째 층
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        # 두 번째 층
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, X, y_true):
        """
        역전파: 체인 룰로 모든 기울기 계산

        핵심 공식:
        dL/dW2 = a1.T @ (dL/dz2)
        dL/dW1 = X.T @ (dL/dz1)
        """
        m = X.shape[0]  # 배치 크기

        # ===== 출력층 역전파 =====
        # dL/da2 = 2(a2 - y) for MSE
        dL_da2 = 2 * (self.a2 - y_true) / m

        # dL/dz2 = dL/da2 * sigmoid'(z2)
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)

        # dL/dW2 = a1.T @ dL/dz2
        dW2 = self.a1.T @ dL_dz2
        db2 = np.sum(dL_dz2, axis=0)

        # ===== 은닉층 역전파 =====
        # dL/da1 = dL/dz2 @ W2.T (기울기 역전파)
        dL_da1 = dL_dz2 @ self.W2.T

        # dL/dz1 = dL/da1 * relu'(z1)
        dL_dz1 = dL_da1 * relu_derivative(self.z1)

        # dL/dW1 = X.T @ dL/dz1
        dW1 = X.T @ dL_dz1
        db1 = np.sum(dL_dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, grads, lr):
        """경사 하강법으로 가중치 업데이트"""
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']

    def loss(self, y_pred, y_true):
        """MSE 손실"""
        return np.mean((y_pred - y_true) ** 2)


# ============================================
# 4. XOR 문제로 테스트
# ============================================
print("\n[4] XOR 문제 학습")
print("-" * 40)

# 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# 모델 생성
np.random.seed(42)
mlp = MLPFromScratch(input_dim=2, hidden_dim=8, output_dim=1)

# 학습
learning_rate = 1.0
epochs = 2000
losses = []

for epoch in range(epochs):
    # 순전파
    y_pred = mlp.forward(X)
    loss = mlp.loss(y_pred, y)
    losses.append(loss)

    # 역전파
    grads = mlp.backward(X, y)

    # 가중치 업데이트
    mlp.update(grads, learning_rate)

    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

# 결과 확인
print("\n학습 결과:")
y_final = mlp.forward(X)
for i in range(4):
    print(f"  {X[i]} → {y_final[i, 0]:.4f} (정답: {y[i, 0]})")

# 손실 그래프
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training Loss (NumPy Backprop)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('numpy_xor_loss.png', dpi=100)
plt.close()
print("\n손실 그래프 저장: numpy_xor_loss.png")


# ============================================
# 5. 기울기 검증 (Gradient Checking)
# ============================================
print("\n[5] 기울기 검증")
print("-" * 40)

def numerical_gradient(model, X, y, param_name, h=1e-5):
    """수치 미분으로 기울기 계산"""
    param = getattr(model, param_name)
    grad = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        # f(x + h)
        param[idx] = original + h
        loss_plus = model.loss(model.forward(X), y)

        # f(x - h)
        param[idx] = original - h
        loss_minus = model.loss(model.forward(X), y)

        # 수치 미분
        grad[idx] = (loss_plus - loss_minus) / (2 * h)

        param[idx] = original
        it.iternext()

    return grad

# 작은 모델로 테스트
np.random.seed(0)
small_mlp = MLPFromScratch(2, 4, 1)

# 순전파
y_pred = small_mlp.forward(X)

# 해석적 기울기 (역전파)
analytical_grads = small_mlp.backward(X, y)

# 수치적 기울기
numerical_W1 = numerical_gradient(small_mlp, X, y, 'W1')
numerical_W2 = numerical_gradient(small_mlp, X, y, 'W2')

# 비교
diff_W1 = np.linalg.norm(analytical_grads['W1'] - numerical_W1)
diff_W2 = np.linalg.norm(analytical_grads['W2'] - numerical_W2)

print(f"W1 기울기 차이: {diff_W1:.2e}")
print(f"W2 기울기 차이: {diff_W2:.2e}")

if diff_W1 < 1e-5 and diff_W2 < 1e-5:
    print("✓ 기울기 검증 통과!")
else:
    print("✗ 기울기 검증 실패")


# ============================================
# 6. 체인 룰 시각화
# ============================================
print("\n[6] 체인 룰 흐름")
print("-" * 40)

chain_rule_diagram = """
순전파 (Forward):
    x ──▶ z1=xW1+b1 ──▶ a1=relu(z1) ──▶ z2=a1W2+b2 ──▶ a2=σ(z2) ──▶ L=MSE

역전파 (Backward):
    dL/dW1 ◀── dL/dz1 ◀── dL/da1 ◀── dL/dz2 ◀── dL/da2 ◀── dL/dL=1

체인 룰 적용:
    dL/dW2 = (dL/da2) × (da2/dz2) × (dz2/dW2)
           = 2(a2-y) × σ'(z2) × a1.T

    dL/dW1 = (dL/da2) × (da2/dz2) × (dz2/da1) × (da1/dz1) × (dz1/dW1)
           = 2(a2-y) × σ'(z2) × W2.T × relu'(z1) × x.T
"""
print(chain_rule_diagram)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("역전파 핵심 정리")
print("=" * 60)

summary = """
1. 순전파: 입력 → 출력 방향으로 값 계산
2. 손실 계산: 예측과 정답의 차이
3. 역전파: 출력 → 입력 방향으로 기울기 계산 (체인 룰)
4. 업데이트: W = W - lr × (dL/dW)

핵심 공식:
- 출력층: dL/dz2 = dL/da2 × σ'(z2)
- 은닉층: dL/dz1 = (dL/dz2 @ W2.T) × relu'(z1)
- 가중치: dL/dW = 이전층출력.T @ 현재층기울기

PyTorch에서는:
    loss.backward()  # 이 한 줄이 위의 모든 과정을 자동 수행!

NumPy 구현의 가치:
1. 행렬 곱셈의 전치 방향 이해
2. 활성화 함수 미분의 역할 이해
3. 배치 처리에서 합산이 필요한 이유 이해
"""
print(summary)
print("=" * 60)
