"""
03. 역전파 (Backpropagation) - PyTorch 버전

PyTorch의 autograd가 역전파를 자동으로 처리합니다.
NumPy 버전(examples/numpy/03_backprop_scratch.py)과 비교해 보세요.

핵심: loss.backward() 한 줄이 모든 기울기를 자동 계산!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch 역전파 (Backpropagation)")
print("=" * 60)


# ============================================
# 1. 자동 미분 복습
# ============================================
print("\n[1] 자동 미분 복습")
print("-" * 40)

# requires_grad=True로 기울기 추적
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 순전파
y = w * x + b
print(f"y = w*x + b = {w.item()}*{x.item()} + {b.item()} = {y.item()}")

# 역전파
y.backward()

print(f"dy/dw = x = {w.grad.item()}")
print(f"dy/dx = w = {x.grad.item()}")
print(f"dy/db = 1 = {b.grad.item()}")


# ============================================
# 2. 단일 뉴런 역전파
# ============================================
print("\n[2] 단일 뉴런 역전파")
print("-" * 40)

# 입력과 목표
x = torch.tensor([2.0], requires_grad=True)
target = torch.tensor([1.0])

# 가중치와 편향
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# 순전파
z = w * x + b
a = torch.sigmoid(z)
loss = (a - target) ** 2

print(f"입력: x={x.item()}, target={target.item()}")
print(f"가중치: w={w.item()}, b={b.item()}")
print(f"예측: a={a.item():.4f}")
print(f"손실: {loss.item():.4f}")

# 역전파 (자동!)
loss.backward()

print(f"\n자동 계산된 기울기:")
print(f"  dL/dw = {w.grad.item():.4f}")
print(f"  dL/db = {b.grad.item():.4f}")


# ============================================
# 3. 2층 MLP 역전파
# ============================================
print("\n[3] 2층 MLP 역전파")
print("-" * 40)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 모델 생성
torch.manual_seed(42)
model = SimpleMLP(2, 8, 1)
print(model)

# 파라미터 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터 수: {total_params}")

for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}")


# ============================================
# 4. XOR 문제로 역전파 확인
# ============================================
print("\n[4] XOR 문제 학습")
print("-" * 40)

# 데이터
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 모델, 손실 함수, 옵티마이저
torch.manual_seed(42)
mlp = SimpleMLP(2, 8, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=1.0)

# 학습
losses = []
for epoch in range(2000):
    # 순전파
    y_pred = mlp(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # 역전파 (핵심 3줄!)
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()        # 역전파 (자동 기울기 계산)
    optimizer.step()       # 가중치 업데이트

    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

# 결과 확인
print("\n학습 결과:")
mlp.eval()
with torch.no_grad():
    y_final = mlp(X)
    for i in range(4):
        print(f"  {X[i].tolist()} → {y_final[i, 0]:.4f} (정답: {y[i, 0]})")

# 손실 그래프
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training Loss (PyTorch Backprop)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('pytorch_xor_loss.png', dpi=100)
plt.close()
print("\n손실 그래프 저장: pytorch_xor_loss.png")


# ============================================
# 5. 기울기 흐름 시각화
# ============================================
print("\n[5] 기울기 흐름 확인")
print("-" * 40)

# 새 모델로 기울기 확인
torch.manual_seed(0)
test_model = SimpleMLP(2, 4, 1)

# 순전파
x_test = torch.tensor([[1.0, 0.0]])
y_test = torch.tensor([[1.0]])

y_pred = test_model(x_test)
loss = criterion(y_pred, y_test)

# 역전파 전 기울기 확인
print("역전파 전:")
for name, param in test_model.named_parameters():
    print(f"  {name}.grad: {param.grad}")

# 역전파
loss.backward()

# 역전파 후 기울기 확인
print("\n역전파 후:")
for name, param in test_model.named_parameters():
    grad_norm = param.grad.norm().item()
    print(f"  {name}.grad norm: {grad_norm:.6f}")


# ============================================
# 6. 계산 그래프 확인
# ============================================
print("\n[6] 계산 그래프")
print("-" * 40)

# 간단한 계산
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a + b
d = a * b
e = c * d

print(f"a = {a.item()}, b = {b.item()}")
print(f"c = a + b = {c.item()}")
print(f"d = a * b = {d.item()}")
print(f"e = c * d = {e.item()}")

# 역전파
e.backward()

print(f"\nde/da = {a.grad.item()}")  # d(c*d)/da = d + c*b = 6 + 5*3 = 21
print(f"de/db = {b.grad.item()}")  # d(c*d)/db = d + c*a = 6 + 5*2 = 16

# 수동 검증
print("\n수동 검증:")
print("e = (a+b) * (a*b)")
print("de/da = (a*b) + (a+b)*b = d + c*b")
print(f"     = {d.item()} + {c.item()}*{b.item()} = {d.item() + c.item()*b.item()}")


# ============================================
# 7. retain_graph와 기울기 누적
# ============================================
print("\n[7] 기울기 누적")
print("-" * 40)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# 첫 번째 backward
y.backward(retain_graph=True)
print(f"첫 번째 backward: dy/dx = {x.grad.item()}")

# 두 번째 backward (기울기 누적!)
y.backward(retain_graph=True)
print(f"두 번째 backward: dy/dx = {x.grad.item()} (누적됨!)")

# 기울기 초기화 후 다시
x.grad.zero_()
y.backward()
print(f"zero_grad() 후: dy/dx = {x.grad.item()}")


# ============================================
# 8. NumPy vs PyTorch 비교
# ============================================
print("\n" + "=" * 60)
print("NumPy vs PyTorch 역전파 비교")
print("=" * 60)

comparison = """
| 단계        | NumPy (수동)                    | PyTorch (자동)              |
|-------------|--------------------------------|----------------------------|
| 순전파      | z1 = X @ W1 + b1               | y = model(X)              |
|             | a1 = relu(z1)                  |                            |
|             | z2 = a1 @ W2 + b2              |                            |
|             | a2 = sigmoid(z2)               |                            |
| 손실        | loss = mean((a2 - y)**2)       | loss = criterion(y, target)|
| 역전파      | dL_da2 = 2*(a2-y)/m            | loss.backward()           |
|             | dL_dz2 = dL_da2 * σ'(z2)       | (자동!)                    |
|             | dW2 = a1.T @ dL_dz2            |                            |
|             | dL_da1 = dL_dz2 @ W2.T         |                            |
|             | dL_dz1 = dL_da1 * relu'(z1)    |                            |
|             | dW1 = X.T @ dL_dz1             |                            |
| 업데이트    | W1 -= lr * dW1                 | optimizer.step()          |
|             | W2 -= lr * dW2                 |                            |

NumPy 구현의 가치:
1. 체인 룰의 동작 원리 직접 체험
2. 행렬 전치(T)가 왜 필요한지 이해
3. 활성화 함수 미분의 역할 파악
4. 배치 처리의 수학적 의미 이해

PyTorch의 장점:
1. 코드 간결성 (3줄로 역전파 완료)
2. 계산 오류 없음 (자동 미분)
3. 복잡한 모델도 동일한 방식
4. GPU 가속 자동 지원
"""
print(comparison)


# ============================================
# 정리
# ============================================
print("=" * 60)
print("역전파 핵심 정리")
print("=" * 60)

summary = """
PyTorch 역전파 3줄:
    optimizer.zero_grad()  # 기울기 초기화 (필수!)
    loss.backward()        # 역전파 (모든 기울기 자동 계산)
    optimizer.step()       # W = W - lr * grad

주의사항:
1. zero_grad() 없으면 기울기가 누적됨
2. backward()는 기본적으로 그래프 삭제 (retain_graph=True로 유지)
3. torch.no_grad()로 추론 시 기울기 계산 비활성화

NumPy로 구현해보면:
- 체인 룰이 실제로 어떻게 적용되는지 이해
- backward()가 내부적으로 하는 일을 알게 됨
- 더 깊은 디버깅 능력 획득
"""
print(summary)
print("=" * 60)
