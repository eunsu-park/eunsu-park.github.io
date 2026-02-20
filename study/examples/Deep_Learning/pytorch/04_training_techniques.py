"""
04. 학습 기법 - PyTorch 버전

다양한 최적화 기법과 정규화를 PyTorch로 구현합니다.
NumPy 버전(examples/numpy/04_training_techniques.py)과 비교해 보세요.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch 학습 기법")
print("=" * 60)


# ============================================
# 1. 옵티마이저 비교
# ============================================
print("\n[1] 옵티마이저 비교")
print("-" * 40)

# 간단한 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# XOR 데이터
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

def train_with_optimizer(optimizer_class, **kwargs):
    """주어진 옵티마이저로 학습"""
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.BCELoss()

    losses = []
    for epoch in range(500):
        pred = model(X)
        loss = criterion(pred, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

# 다양한 옵티마이저 테스트
optimizers = {
    'SGD (lr=0.5)': (torch.optim.SGD, {'lr': 0.5}),
    'SGD+Momentum': (torch.optim.SGD, {'lr': 0.5, 'momentum': 0.9}),
    'Adam': (torch.optim.Adam, {'lr': 0.01}),
    'RMSprop': (torch.optim.RMSprop, {'lr': 0.01}),
}

results = {}
for name, (opt_class, params) in optimizers.items():
    losses = train_with_optimizer(opt_class, **params)
    results[name] = losses
    print(f"{name}: 최종 손실 = {losses[-1]:.6f}")

# 시각화
plt.figure(figsize=(10, 5))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('optimizer_comparison.png', dpi=100)
plt.close()
print("그래프 저장: optimizer_comparison.png")


# ============================================
# 2. 학습률 스케줄러
# ============================================
print("\n[2] 학습률 스케줄러")
print("-" * 40)

# 스케줄러 테스트
def test_scheduler(scheduler_class, **kwargs):
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = scheduler_class(optimizer, **kwargs)

    lrs = []
    for epoch in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    return lrs

schedulers = {
    'StepLR': (torch.optim.lr_scheduler.StepLR, {'step_size': 20, 'gamma': 0.5}),
    'ExponentialLR': (torch.optim.lr_scheduler.ExponentialLR, {'gamma': 0.95}),
    'CosineAnnealingLR': (torch.optim.lr_scheduler.CosineAnnealingLR, {'T_max': 50}),
}

plt.figure(figsize=(10, 5))
for name, (sched_class, params) in schedulers.items():
    lrs = test_scheduler(sched_class, **params)
    plt.plot(lrs, label=name)
    print(f"{name}: 시작 {lrs[0]:.4f} → 끝 {lrs[-1]:.4f}")

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedulers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('lr_schedulers.png', dpi=100)
plt.close()
print("그래프 저장: lr_schedulers.png")


# ============================================
# 3. Dropout
# ============================================
print("\n[3] Dropout")
print("-" * 40)

class NetWithDropout(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Dropout 효과 확인
model = NetWithDropout(dropout_p=0.5)
x_test = torch.randn(1, 2)

model.train()
print("훈련 모드 (Dropout 활성):")
for i in range(3):
    out = model.fc1(x_test)
    out = F.relu(out)
    out = model.dropout(out)
    print(f"  시도 {i+1}: 활성 뉴런 = {(out != 0).sum().item()}/32")

model.eval()
print("\n평가 모드 (Dropout 비활성):")
out = model.fc1(x_test)
out = F.relu(out)
out = model.dropout(out)  # eval 모드에서는 전체 통과
print(f"  활성 뉴런 = {(out != 0).sum().item()}/32")


# ============================================
# 4. Batch Normalization
# ============================================
print("\n[4] Batch Normalization")
print("-" * 40)

class NetWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = torch.sigmoid(self.fc2(x))
        return x

bn_model = NetWithBatchNorm()
print(f"BatchNorm1d 파라미터:")
print(f"  weight (γ): {bn_model.bn1.weight.shape}")
print(f"  bias (β): {bn_model.bn1.bias.shape}")
print(f"  running_mean: {bn_model.bn1.running_mean.shape}")
print(f"  running_var: {bn_model.bn1.running_var.shape}")

# 훈련 vs 평가 모드
x_batch = torch.randn(32, 2)

bn_model.train()
out_train = bn_model.fc1(x_batch)
out_train = bn_model.bn1(out_train)
print(f"\n훈련 모드 - 출력 통계:")
print(f"  mean: {out_train.mean(dim=0)[:3].tolist()}")
print(f"  std: {out_train.std(dim=0)[:3].tolist()}")

bn_model.eval()
out_eval = bn_model.fc1(x_batch)
out_eval = bn_model.bn1(out_eval)
print(f"평가 모드 - 출력 통계:")
print(f"  mean: {out_eval.mean(dim=0)[:3].tolist()}")


# ============================================
# 5. Weight Decay (L2 정규화)
# ============================================
print("\n[5] Weight Decay")
print("-" * 40)

def train_with_weight_decay(weight_decay):
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(500):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 가중치 크기 확인
    weight_norm = sum(p.norm().item() for p in model.parameters())
    return loss.item(), weight_norm

for wd in [0, 0.01, 0.1]:
    loss, w_norm = train_with_weight_decay(wd)
    print(f"Weight Decay={wd}: 손실={loss:.4f}, 가중치 norm={w_norm:.4f}")


# ============================================
# 6. 조기 종료 (Early Stopping)
# ============================================
print("\n[6] 조기 종료")
print("-" * 40)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

# 데모 (시뮬레이션된 검증 손실)
early_stopping = EarlyStopping(patience=5)
val_losses = [1.0, 0.9, 0.85, 0.8, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87]

model = SimpleNet()
for epoch, val_loss in enumerate(val_losses):
    early_stopping(val_loss, model)
    status = "STOP" if early_stopping.early_stop else f"patience={early_stopping.counter}"
    print(f"Epoch {epoch+1}: val_loss={val_loss:.2f}, {status}")
    if early_stopping.early_stop:
        break


# ============================================
# 7. 전체 학습 예제
# ============================================
print("\n[7] 전체 학습 예제")
print("-" * 40)

# 더 큰 데이터셋 생성
np.random.seed(42)
n_samples = 200

# 원형 데이터 (비선형 문제)
theta = np.random.uniform(0, 2*np.pi, n_samples)
r = np.random.uniform(0, 1, n_samples)
X_train = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y_train = (r > 0.5).astype(np.float32)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# 검증 데이터
X_val = X_train[:40]
y_val = y_train[:40]
X_train = X_train[40:]
y_train = y_train[40:]

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# 모델 초기화
torch.manual_seed(42)
model = FullModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
early_stopping = EarlyStopping(patience=20)

# 학습
train_losses = []
val_losses = []

for epoch in range(200):
    # 훈련
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # 검증
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)

    # 스케줄러 업데이트
    scheduler.step(val_loss)

    # 조기 종료 체크
    early_stopping(val_loss, model)

    if (epoch + 1) % 40 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={val_loss:.4f}, lr={lr:.6f}")

    if early_stopping.early_stop:
        print(f"조기 종료 at epoch {epoch+1}")
        break

# 최고 모델 복원
if early_stopping.best_model:
    model.load_state_dict(early_stopping.best_model)

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training with Regularization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('full_training.png', dpi=100)
plt.close()
print("그래프 저장: full_training.png")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("학습 기법 정리")
print("=" * 60)

summary = """
권장 기본 설정:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    EarlyStopping(patience=10)

정규화 조합:
    - Dropout (0.2~0.5): 과적합 방지
    - BatchNorm: 학습 안정화
    - Weight Decay (1e-4~1e-2): 가중치 크기 제한

학습 루프 체크리스트:
    1. model.train() / model.eval() 모드 전환
    2. optimizer.zero_grad() 호출
    3. loss.backward()
    4. optimizer.step()
    5. scheduler.step() (에폭 끝)
    6. EarlyStopping 체크
"""
print(summary)
print("=" * 60)
