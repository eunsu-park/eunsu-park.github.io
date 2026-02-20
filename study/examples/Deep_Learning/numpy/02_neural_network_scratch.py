"""
02. 신경망 기초 - NumPy 버전 (from scratch)

NumPy만으로 MLP 순전파를 구현합니다.
PyTorch 버전(examples/pytorch/02_neural_network.py)과 비교해 보세요.

핵심: 역전파 없이 순전파만 구현합니다.
     역전파는 03_backprop_scratch.py에서 구현합니다.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("NumPy 신경망 기초 (from scratch)")
print("=" * 60)


# ============================================
# 1. 활성화 함수 구현
# ============================================
print("\n[1] 활성화 함수 구현")
print("-" * 40)

def sigmoid(x):
    """시그모이드: σ(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """시그모이드 미분: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 미분: 1 if x > 0 else 0"""
    return (x > 0).astype(float)

def tanh(x):
    """Tanh: (e^x - e^(-x)) / (e^x + e^(-x))"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh 미분: 1 - tanh²(x)"""
    return 1 - np.tanh(x)**2

def softmax(x):
    """Softmax: e^xi / Σe^xj"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 테스트
x_test = np.array([-2, -1, 0, 1, 2])
print(f"입력: {x_test}")
print(f"sigmoid: {sigmoid(x_test)}")
print(f"relu: {relu(x_test)}")
print(f"tanh: {tanh(x_test)}")

# 시각화
x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, sigmoid(x), label='Sigmoid')
axes[0, 0].plot(x, sigmoid_derivative(x), '--', label='Derivative')
axes[0, 0].set_title('Sigmoid and Derivative')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, tanh(x), label='Tanh')
axes[0, 1].plot(x, tanh_derivative(x), '--', label='Derivative')
axes[0, 1].set_title('Tanh and Derivative')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x, relu(x), label='ReLU')
axes[1, 0].plot(x, relu_derivative(x), '--', label='Derivative')
axes[1, 0].set_title('ReLU and Derivative')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

x_softmax = np.array([1, 2, 3, 4])
axes[1, 1].bar(range(4), softmax(x_softmax))
axes[1, 1].set_title(f'Softmax of {x_softmax}')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('numpy_activation_functions.png', dpi=100)
plt.close()
print("활성화 함수 그래프 저장: numpy_activation_functions.png")


# ============================================
# 2. 퍼셉트론 (단일 뉴런)
# ============================================
print("\n[2] 퍼셉트론 구현")
print("-" * 40)

class Perceptron:
    """단일 퍼셉트론"""

    def __init__(self, n_inputs):
        # 가중치 초기화 (작은 랜덤 값)
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

    def forward(self, x):
        """순전파: z = wx + b, y = activation(z)"""
        z = np.dot(x, self.weights) + self.bias
        return sigmoid(z)

# 테스트
perceptron = Perceptron(n_inputs=3)
x_input = np.array([1.0, 2.0, 3.0])
output = perceptron.forward(x_input)

print(f"입력: {x_input}")
print(f"가중치: {perceptron.weights}")
print(f"편향: {perceptron.bias}")
print(f"출력: {output:.4f}")


# ============================================
# 3. 다층 퍼셉트론 (MLP) 순전파
# ============================================
print("\n[3] MLP 순전파 구현")
print("-" * 40)

class MLPNumpy:
    """
    NumPy로 구현한 다층 퍼셉트론
    순전파만 구현 (역전파는 03에서)
    """

    def __init__(self, layer_sizes):
        """
        layer_sizes: [입력 차원, 은닉층1, 은닉층2, ..., 출력 차원]
        예: [784, 256, 128, 10] → 입력 784, 은닉 256/128, 출력 10
        """
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        # Xavier 초기화
        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier 초기화: std = sqrt(2 / (fan_in + fan_out))
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

        print(f"MLP 생성: {layer_sizes}")
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            print(f"  Layer {i+1}: W{W.shape}, b{b.shape}")

    def forward(self, x):
        """순전파"""
        activations = [x]

        for i in range(self.num_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]

            # 마지막 층은 활성화 없음 (또는 softmax)
            if i < self.num_layers - 1:
                a = relu(z)
            else:
                a = z  # 출력층

            activations.append(a)

        return activations[-1], activations

    def predict_proba(self, x):
        """분류 확률 (softmax)"""
        output, _ = self.forward(x)
        return softmax(output)

    def predict(self, x):
        """분류 예측"""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=-1)

# MLP 테스트
mlp = MLPNumpy([10, 32, 16, 3])

# 배치 입력 (4개 샘플, 10차원)
x_batch = np.random.randn(4, 10)
output, activations = mlp.forward(x_batch)

print(f"\n입력 shape: {x_batch.shape}")
print(f"출력 shape: {output.shape}")
print(f"출력 예시:\n{output}")

# 확률과 예측
proba = mlp.predict_proba(x_batch)
pred = mlp.predict(x_batch)
print(f"\nSoftmax 확률:\n{proba}")
print(f"예측 클래스: {pred}")


# ============================================
# 4. XOR 문제 - 순전파만
# ============================================
print("\n[4] XOR 문제 (순전파만)")
print("-" * 40)

# XOR 데이터
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# 수동으로 가중치 설정 (학습 없이)
# XOR을 해결하는 수동 설정 가중치
class XORNetManual:
    def __init__(self):
        # 은닉층: 2개 뉴런
        # 첫 번째 뉴런: AND처럼 동작 (둘 다 1일 때)
        # 두 번째 뉴런: OR처럼 동작 (하나라도 1일 때)
        self.W1 = np.array([[ 20,  20],   # x1에 대한 가중치
                           [ 20,  20]])   # x2에 대한 가중치
        self.b1 = np.array([-30, -10])    # AND: -30, OR: -10

        # 출력층: OR - AND = XOR
        self.W2 = np.array([[-20],        # AND 뉴런에 음수
                           [ 20]])        # OR 뉴런에 양수
        self.b2 = np.array([-10])

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = sigmoid(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)

        return a2

xor_manual = XORNetManual()

print("수동 설정 가중치로 XOR 해결:")
for i in range(4):
    x = X_xor[i:i+1]
    y_pred = xor_manual.forward(x)
    print(f"  {X_xor[i]} → {y_pred[0, 0]:.4f} (정답: {y_xor[i]})")


# ============================================
# 5. 순전파 과정 시각화
# ============================================
print("\n[5] 순전파 과정 시각화")
print("-" * 40)

def visualize_forward_pass(x, model):
    """순전파 과정의 값 변화 출력"""
    print(f"입력: {x}")

    a = x
    for i in range(model.num_layers):
        z = a @ model.weights[i] + model.biases[i]
        print(f"\nLayer {i+1}:")
        print(f"  z (선형 변환): {z[:5]}...")  # 처음 5개만

        if i < model.num_layers - 1:
            a = relu(z)
            print(f"  a (ReLU 후):    {a[:5]}...")
        else:
            a = z
            print(f"  출력:           {a}")

    return a

# 단일 샘플로 테스트
small_mlp = MLPNumpy([4, 8, 3])
x_single = np.array([1.0, 2.0, 3.0, 4.0])
output = visualize_forward_pass(x_single, small_mlp)


# ============================================
# 6. NumPy vs PyTorch 비교
# ============================================
print("\n" + "=" * 60)
print("NumPy vs PyTorch 비교")
print("=" * 60)

comparison = """
| 항목          | NumPy (이 코드)           | PyTorch                    |
|---------------|---------------------------|----------------------------|
| 순전파        | x @ W + b 직접 계산       | model(x) 자동 계산         |
| 활성화 함수   | np.maximum(0, x)          | F.relu(x)                  |
| 가중치 관리   | 리스트로 직접 관리        | model.parameters()         |
| 역전파        | ❌ (다음 레슨에서 구현)   | loss.backward() 자동       |
| 배치 처리     | 행렬 곱셈으로 직접        | DataLoader 자동            |

NumPy 구현의 장점:
1. 순전파의 수학적 원리 완전 이해
2. 행렬 연산의 의미 파악
3. 활성화 함수의 동작 이해

다음 단계 (03_backprop_scratch.py):
- 역전파 알고리즘 NumPy 구현
- 경사 하강법으로 가중치 업데이트
- XOR 문제 학습으로 해결
"""
print(comparison)

print("NumPy 신경망 기초 (순전파) 완료!")
print("PyTorch 버전과 비교: examples/pytorch/02_neural_network.py")
print("=" * 60)
