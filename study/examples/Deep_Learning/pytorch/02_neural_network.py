"""
02. 신경망 기초 - PyTorch 버전

nn.Module을 사용한 MLP 구현과 XOR 문제 해결.
NumPy 버전(examples/numpy/02_neural_network_scratch.py)과 비교해 보세요.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch 신경망 기초")
print("=" * 60)


# ============================================
# 1. 활성화 함수
# ============================================
print("\n[1] 활성화 함수")
print("-" * 40)

x = torch.linspace(-5, 5, 100)

# 활성화 함수 적용
sigmoid_out = torch.sigmoid(x)
tanh_out = torch.tanh(x)
relu_out = F.relu(x)
leaky_relu_out = F.leaky_relu(x, 0.1)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x.numpy(), sigmoid_out.numpy())
axes[0, 0].set_title('Sigmoid')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

axes[0, 1].plot(x.numpy(), tanh_out.numpy())
axes[0, 1].set_title('Tanh')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

axes[1, 0].plot(x.numpy(), relu_out.numpy())
axes[1, 0].set_title('ReLU')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)

axes[1, 1].plot(x.numpy(), leaky_relu_out.numpy())
axes[1, 1].set_title('Leaky ReLU (α=0.1)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=100)
plt.close()
print("활성화 함수 그래프 저장: activation_functions.png")


# ============================================
# 2. nn.Module로 MLP 정의
# ============================================
print("\n[2] nn.Module MLP")
print("-" * 40)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP(input_dim=10, hidden_dim=32, output_dim=3)
print(model)

# 파라미터 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")


# ============================================
# 3. nn.Sequential로 간단히 정의
# ============================================
print("\n[3] nn.Sequential")
print("-" * 40)

model_seq = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
print(model_seq)


# ============================================
# 4. XOR 문제 해결
# ============================================
print("\n[4] XOR 문제 해결")
print("-" * 40)

# 데이터
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print("XOR 데이터:")
print("  (0,0) → 0")
print("  (0,1) → 1")
print("  (1,0) → 1")
print("  (1,1) → 0")

# 모델 정의
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

xor_model = XORNet()

# 손실 함수와 옵티마이저
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(xor_model.parameters(), lr=0.1)

# 학습
losses = []
for epoch in range(1000):
    # 순전파
    pred = xor_model(X)
    loss = criterion(pred, y)

    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

# 결과 확인
print("\n학습 결과:")
xor_model.eval()
with torch.no_grad():
    predictions = xor_model(X)
    for i in range(4):
        print(f"  {X[i].numpy()} → {predictions[i].item():.4f} (정답: {y[i].item()})")

# 손실 그래프
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('xor_loss.png', dpi=100)
plt.close()
print("손실 그래프 저장: xor_loss.png")


# ============================================
# 5. 가중치 초기화
# ============================================
print("\n[5] 가중치 초기화")
print("-" * 40)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
        print(f"  Initialized: {m}")

model_init = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

print("가중치 초기화 전:")
print(f"  fc1 weight mean: {model_init[0].weight.mean().item():.6f}")

print("\n초기화 적용:")
model_init.apply(init_weights)

print("\n가중치 초기화 후:")
print(f"  fc1 weight mean: {model_init[0].weight.mean().item():.6f}")


# ============================================
# 6. 순전파 단계별 확인
# ============================================
print("\n[6] 순전파 단계별 확인")
print("-" * 40)

class VerboseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        print(f"  입력: {x.shape}")

        z1 = self.fc1(x)
        print(f"  fc1 후: {z1.shape}")

        a1 = F.relu(z1)
        print(f"  ReLU 후: {a1.shape}")

        z2 = self.fc2(a1)
        print(f"  fc2 후 (출력): {z2.shape}")

        return z2

verbose_model = VerboseMLP()
sample_input = torch.randn(2, 3)  # 배치 크기 2, 입력 차원 3
print("순전파 과정:")
output = verbose_model(sample_input)


# ============================================
# 7. 모델 저장 및 로드
# ============================================
print("\n[7] 모델 저장/로드")
print("-" * 40)

# 저장
torch.save(xor_model.state_dict(), 'xor_model.pth')
print("모델 저장: xor_model.pth")

# 새 모델에 로드
new_model = XORNet()
new_model.load_state_dict(torch.load('xor_model.pth', weights_only=True))
new_model.eval()
print("모델 로드 완료")

# 검증
with torch.no_grad():
    new_pred = new_model(X)
    print("로드된 모델 예측:")
    for i in range(4):
        print(f"  {X[i].numpy()} → {new_pred[i].item():.4f}")


print("\n" + "=" * 60)
print("PyTorch 신경망 기초 완료!")
print("NumPy 버전과 비교: examples/numpy/02_neural_network_scratch.py")
print("=" * 60)
