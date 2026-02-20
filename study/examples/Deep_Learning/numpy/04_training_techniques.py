"""
04. 학습 기법 - NumPy 버전

다양한 최적화 기법과 정규화를 NumPy로 직접 구현합니다.
PyTorch 버전(examples/pytorch/04_training_techniques.py)과 비교해 보세요.

이 파일이 마지막 NumPy 구현입니다!
CNN부터는 PyTorch만 사용합니다.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("NumPy 학습 기법 (from scratch)")
print("=" * 60)


# ============================================
# 0. 기본 함수들
# ============================================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# ============================================
# 1. 옵티마이저 구현
# ============================================
print("\n[1] 옵티마이저 구현")
print("-" * 40)

class SGD:
    """기본 확률적 경사 하강법"""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]

class SGDMomentum:
    """모멘텀을 사용한 SGD"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        for key in params:
            self.v[key] = self.momentum * self.v[key] + grads[key]
            params[key] -= self.lr * self.v[key]

class Adam:
    """Adam 옵티마이저"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = {key: np.zeros_like(val) for key, val in params.items()}
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        self.t += 1

        for key in params:
            # 1차 모멘트 (평균)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # 2차 모멘트 (분산)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # 편향 보정
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 업데이트
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

print("SGD, SGDMomentum, Adam 클래스 구현 완료")


# ============================================
# 2. 학습률 스케줄러
# ============================================
print("\n[2] 학습률 스케줄러")
print("-" * 40)

class StepLR:
    """Step Decay 스케줄러"""
    def __init__(self, initial_lr, step_size=30, gamma=0.1):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))

class ExponentialLR:
    """지수 감쇠 스케줄러"""
    def __init__(self, initial_lr, gamma=0.95):
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** epoch)

class CosineAnnealingLR:
    """코사인 어닐링 스케줄러"""
    def __init__(self, initial_lr, T_max, eta_min=0):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, epoch):
        return self.eta_min + (self.initial_lr - self.eta_min) * \
               (1 + np.cos(np.pi * epoch / self.T_max)) / 2

# 시각화
epochs = np.arange(100)
schedulers = {
    'StepLR': StepLR(1.0, step_size=20, gamma=0.5),
    'ExponentialLR': ExponentialLR(1.0, gamma=0.95),
    'CosineAnnealingLR': CosineAnnealingLR(1.0, T_max=50),
}

plt.figure(figsize=(10, 5))
for name, scheduler in schedulers.items():
    lrs = [scheduler.get_lr(e) for e in epochs]
    plt.plot(lrs, label=name)
    print(f"{name}: 시작={lrs[0]:.4f}, 끝={lrs[-1]:.4f}")

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('NumPy Learning Rate Schedulers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('numpy_lr_schedulers.png', dpi=100)
plt.close()
print("그래프 저장: numpy_lr_schedulers.png")


# ============================================
# 3. Dropout 구현
# ============================================
print("\n[3] Dropout")
print("-" * 40)

def dropout(x, p=0.5, training=True):
    """
    Dropout 구현

    Args:
        x: 입력
        p: 드롭할 확률
        training: 훈련 모드 여부
    """
    if not training or p == 0:
        return x

    # 마스크 생성 (1-p 확률로 1)
    mask = (np.random.rand(*x.shape) > p).astype(float)

    # 역 드롭아웃 (inverted dropout): 스케일 보정
    return x * mask / (1 - p)

# 테스트
np.random.seed(42)
x = np.ones((1, 10))

print("입력:", x)
print("훈련 모드 (p=0.5):")
for i in range(3):
    out = dropout(x.copy(), p=0.5, training=True)
    active = np.sum(out != 0)
    print(f"  시도 {i+1}: 활성 뉴런 = {active}/10, 출력 = {out[0][:5]}...")

print("평가 모드:")
out = dropout(x.copy(), p=0.5, training=False)
print(f"  출력 = {out[0][:5]}...")


# ============================================
# 4. Batch Normalization 구현
# ============================================
print("\n[4] Batch Normalization")
print("-" * 40)

class BatchNorm:
    """배치 정규화 구현"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # 학습 가능한 파라미터
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # 이동 평균 (추론용)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # 역전파용 캐시
        self.cache = None

    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # 이동 평균 업데이트
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # 정규화
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # 스케일 및 시프트
        out = self.gamma * x_norm + self.beta

        # 역전파용 저장
        self.cache = (x, x_norm, mean, var)

        return out

    def backward(self, dout):
        x, x_norm, mean, var = self.cache
        N = x.shape[0]

        # 파라미터 기울기
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        # 입력 기울기
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * (-0.5) * (var + self.eps)**(-1.5), axis=0)
        dmean = np.sum(dx_norm * (-1 / np.sqrt(var + self.eps)), axis=0) + \
                dvar * np.mean(-2 * (x - mean), axis=0)
        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N

        return dx, dgamma, dbeta

# 테스트
np.random.seed(42)
bn = BatchNorm(num_features=4)
x_batch = np.random.randn(32, 4) * 5 + 10  # 평균 10, 표준편차 5

print(f"입력 통계: mean={x_batch.mean(axis=0).round(2)}, std={x_batch.std(axis=0).round(2)}")

out = bn.forward(x_batch, training=True)
print(f"출력 통계: mean={out.mean(axis=0).round(4)}, std={out.std(axis=0).round(4)}")


# ============================================
# 5. L2 정규화 (Weight Decay)
# ============================================
print("\n[5] Weight Decay (L2 정규화)")
print("-" * 40)

def compute_loss_with_l2(y_pred, y_true, weights, l2_lambda=0.01):
    """L2 정규화가 포함된 손실 계산"""
    # 기본 손실 (MSE)
    data_loss = np.mean((y_pred - y_true) ** 2)

    # L2 정규화 항
    l2_loss = 0
    for W in weights:
        l2_loss += np.sum(W ** 2)
    l2_loss *= l2_lambda / 2

    return data_loss + l2_loss, data_loss, l2_loss

# 예시
W1 = np.random.randn(10, 5)
W2 = np.random.randn(5, 1)
y_pred = np.random.randn(32, 1)
y_true = np.random.randn(32, 1)

for l2_lambda in [0, 0.01, 0.1]:
    total, data, reg = compute_loss_with_l2(y_pred, y_true, [W1, W2], l2_lambda)
    print(f"λ={l2_lambda}: 총 손실={total:.4f} (데이터={data:.4f} + 정규화={reg:.4f})")


# ============================================
# 6. 옵티마이저 비교 실험
# ============================================
print("\n[6] 옵티마이저 비교")
print("-" * 40)

class MLPWithOptimizer:
    """옵티마이저 테스트용 MLP"""
    def __init__(self, optimizer):
        np.random.seed(42)
        self.params = {
            'W1': np.random.randn(2, 16) * 0.5,
            'b1': np.zeros(16),
            'W2': np.random.randn(16, 1) * 0.5,
            'b2': np.zeros(1),
        }
        self.optimizer = optimizer

    def forward(self, X):
        self.z1 = X @ self.params['W1'] + self.params['b1']
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.params['W2'] + self.params['b2']
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]

        dL_da2 = 2 * (self.a2 - y) / m
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)

        grads = {
            'W2': self.a1.T @ dL_dz2,
            'b2': np.sum(dL_dz2, axis=0),
        }

        dL_da1 = dL_dz2 @ self.params['W2'].T
        dL_dz1 = dL_da1 * relu_derivative(self.z1)

        grads['W1'] = X.T @ dL_dz1
        grads['b1'] = np.sum(dL_dz1, axis=0)

        return grads

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = np.mean((y_pred - y) ** 2)
        grads = self.backward(X, y)
        self.optimizer.update(self.params, grads)
        return loss

# XOR 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# 옵티마이저 비교
optimizers = {
    'SGD': SGD(lr=0.5),
    'SGD+Momentum': SGDMomentum(lr=0.5, momentum=0.9),
    'Adam': Adam(lr=0.05),
}

results = {}
for name, opt in optimizers.items():
    model = MLPWithOptimizer(opt)
    losses = []
    for epoch in range(500):
        loss = model.train_step(X, y)
        losses.append(loss)
    results[name] = losses
    print(f"{name}: 최종 손실 = {losses[-1]:.6f}")

# 시각화
plt.figure(figsize=(10, 5))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NumPy Optimizer Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('numpy_optimizer_comparison.png', dpi=100)
plt.close()
print("그래프 저장: numpy_optimizer_comparison.png")


# ============================================
# 7. 전체 기법 적용
# ============================================
print("\n[7] 전체 기법 적용")
print("-" * 40)

class FullMLP:
    """모든 기법이 적용된 MLP"""
    def __init__(self, dropout_p=0.3, l2_lambda=0.01):
        np.random.seed(42)
        self.params = {
            'W1': np.random.randn(2, 32) * np.sqrt(2/2),
            'b1': np.zeros(32),
            'W2': np.random.randn(32, 16) * np.sqrt(2/32),
            'b2': np.zeros(16),
            'W3': np.random.randn(16, 1) * np.sqrt(2/16),
            'b3': np.zeros(1),
        }
        self.bn1 = BatchNorm(32)
        self.bn2 = BatchNorm(16)
        self.dropout_p = dropout_p
        self.l2_lambda = l2_lambda
        self.training = True

    def forward(self, X):
        # 첫 번째 층
        self.z1 = X @ self.params['W1'] + self.params['b1']
        self.bn1_out = self.bn1.forward(self.z1, self.training)
        self.a1 = relu(self.bn1_out)
        self.d1 = dropout(self.a1, self.dropout_p, self.training)

        # 두 번째 층
        self.z2 = self.d1 @ self.params['W2'] + self.params['b2']
        self.bn2_out = self.bn2.forward(self.z2, self.training)
        self.a2 = relu(self.bn2_out)
        self.d2 = dropout(self.a2, self.dropout_p, self.training)

        # 출력층
        self.z3 = self.d2 @ self.params['W3'] + self.params['b3']
        self.a3 = sigmoid(self.z3)

        return self.a3

    def loss(self, y_pred, y_true):
        # MSE 손실
        data_loss = np.mean((y_pred - y_true) ** 2)

        # L2 정규화
        l2_loss = 0
        for key in ['W1', 'W2', 'W3']:
            l2_loss += np.sum(self.params[key] ** 2)
        l2_loss *= self.l2_lambda / 2

        return data_loss + l2_loss

# 더 복잡한 데이터 생성
np.random.seed(42)
n_samples = 200
theta = np.random.uniform(0, 2*np.pi, n_samples)
r = np.random.uniform(0, 1, n_samples)
X_train = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y_train = (r > 0.5).astype(np.float64).reshape(-1, 1)

# 학습
model = FullMLP(dropout_p=0.3, l2_lambda=0.001)
optimizer = Adam(lr=0.01)

losses = []
for epoch in range(300):
    # 순전파
    y_pred = model.forward(X_train)
    loss = model.loss(y_pred, y_train)
    losses.append(loss)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Full MLP Training (with BN, Dropout, L2)')
plt.grid(True, alpha=0.3)
plt.savefig('numpy_full_training.png', dpi=100)
plt.close()
print("그래프 저장: numpy_full_training.png")


# ============================================
# NumPy vs PyTorch 비교
# ============================================
print("\n" + "=" * 60)
print("NumPy vs PyTorch 비교")
print("=" * 60)

comparison = """
| 항목           | NumPy (이 코드)            | PyTorch                    |
|----------------|---------------------------|----------------------------|
| Optimizer      | 클래스로 직접 구현          | torch.optim.Adam 등        |
| Scheduler      | 함수로 직접 계산            | lr_scheduler 모듈          |
| Dropout        | 마스크 × 스케일 직접 계산   | nn.Dropout                 |
| BatchNorm      | 평균/분산 직접 계산         | nn.BatchNorm1d             |
| Weight Decay   | 손실에 직접 추가            | optimizer의 weight_decay   |

NumPy 구현의 가치:
1. Adam의 m, v 업데이트 원리 이해
2. BatchNorm의 이동 평균 작동 방식
3. Dropout의 역드롭아웃(inverted) 이해
4. 정규화 항이 손실에 미치는 영향

이후 CNN부터는 PyTorch만 사용:
- 합성곱 연산의 NumPy 구현은 비효율적
- GPU 가속이 필수적
- 복잡한 아키텍처 관리 어려움
"""
print(comparison)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("NumPy 학습 기법 정리")
print("=" * 60)

summary = """
구현한 것들:
1. SGD, Momentum, Adam 옵티마이저
2. StepLR, ExponentialLR, CosineAnnealingLR 스케줄러
3. Dropout (역드롭아웃 포함)
4. Batch Normalization (순전파 + 역전파)
5. L2 정규화 (Weight Decay)

핵심 포인트:
- Adam: β₁=0.9, β₂=0.999로 1차/2차 모멘트 추정
- Dropout: training 모드에서만 적용, 스케일 보정 필수
- BatchNorm: 훈련 시 배치 통계, 추론 시 이동 평균 사용
- L2: 가중치 크기 제한으로 일반화 향상

다음 단계:
- CNN (05_CNN_기초.md)부터는 PyTorch만 사용
- NumPy로 충분히 원리를 이해했음!
"""
print(summary)
print("=" * 60)
