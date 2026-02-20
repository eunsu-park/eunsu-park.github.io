"""
11. 학습 최적화

하이퍼파라미터 튜닝, Mixed Precision, Gradient Accumulation 등을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import time

print("=" * 60)
print("PyTorch 학습 최적화")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")


# ============================================
# 1. 재현성 설정
# ============================================
print("\n[1] 재현성 설정")
print("-" * 40)

def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
print("시드 설정 완료: 42")


# ============================================
# 2. 샘플 모델 및 데이터
# ============================================
print("\n[2] 샘플 모델 및 데이터")
print("-" * 40)

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 더미 데이터
X_train = torch.randn(1000, 1, 28, 28)
y_train = torch.randint(0, 10, (1000,))
X_val = torch.randn(200, 1, 28, 28)
y_val = torch.randint(0, 10, (200,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

print(f"훈련 데이터: {len(train_dataset)}")
print(f"검증 데이터: {len(val_dataset)}")


# ============================================
# 3. 학습률 스케줄러
# ============================================
print("\n[3] 학습률 스케줄러")
print("-" * 40)

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine Decay 스케줄러"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 테스트
model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=100, total_steps=1000)

lrs = []
for step in range(1000):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

print(f"Warmup 구간 (0-100): {lrs[0]:.6f} → {lrs[99]:.6f}")
print(f"Decay 구간 (100-1000): {lrs[100]:.6f} → {lrs[-1]:.6f}")


# ============================================
# 4. 조기 종료
# ============================================
print("\n[4] 조기 종료")
print("-" * 40)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_loss = val_loss
            self._save_checkpoint(model)
            self.counter = 0

    def _save_checkpoint(self, model):
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# 테스트
early_stopping = EarlyStopping(patience=3)
losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]

print("조기 종료 시뮬레이션:")
for epoch, loss in enumerate(losses):
    early_stopping(loss, model)
    status = "STOP" if early_stopping.early_stop else f"counter={early_stopping.counter}"
    print(f"  Epoch {epoch}: loss={loss:.2f}, {status}")
    if early_stopping.early_stop:
        break


# ============================================
# 5. Gradient Accumulation
# ============================================
print("\n[5] Gradient Accumulation")
print("-" * 40)

def train_with_accumulation(model, train_loader, optimizer, accumulation_steps=4):
    """Gradient Accumulation으로 학습"""
    model.train()
    optimizer.zero_grad()
    total_loss = 0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss = loss / accumulation_steps  # 스케일링
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(train_loader)

# 테스트
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

loss = train_with_accumulation(model, train_loader, optimizer, accumulation_steps=4)
print(f"Accumulation 학습 손실: {loss:.4f}")
print(f"효과적 배치 크기: 32 × 4 = 128")


# ============================================
# 6. Mixed Precision Training
# ============================================
print("\n[6] Mixed Precision Training")
print("-" * 40)

if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler

    def train_with_amp(model, train_loader, optimizer, scaler):
        """Mixed Precision 학습"""
        model.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    loss = train_with_amp(model, train_loader, optimizer, scaler)
    print(f"AMP 학습 손실: {loss:.4f}")
else:
    print("CUDA 미사용 - AMP 스킵")


# ============================================
# 7. Gradient Clipping
# ============================================
print("\n[7] Gradient Clipping")
print("-" * 40)

def train_with_clipping(model, train_loader, optimizer, max_norm=1.0):
    """Gradient Clipping으로 학습"""
    model.train()
    total_loss = 0
    grad_norms = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Gradient norm 기록 (클리핑 전)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader), grad_norms

model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss, norms = train_with_clipping(model, train_loader, optimizer, max_norm=1.0)
print(f"Clipping 학습 손실: {loss:.4f}")
print(f"평균 기울기 norm: {np.mean(norms):.4f}")
print(f"최대 기울기 norm: {np.max(norms):.4f}")


# ============================================
# 8. 하이퍼파라미터 탐색 (Random Search)
# ============================================
print("\n[8] 하이퍼파라미터 탐색")
print("-" * 40)

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

def train_with_config(lr, batch_size, dropout, epochs=5):
    """설정으로 학습"""
    set_seed(42)

    model = SimpleNet(dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

    return evaluate(model, val_loader)

# Random Search
import random
print("Random Search 실행 중...")

best_acc = 0
best_config = None
results = []

for trial in range(5):
    lr = 10 ** random.uniform(-4, -2)
    batch_size = random.choice([32, 64, 128])
    dropout = random.uniform(0.2, 0.5)

    acc = train_with_config(lr, batch_size, dropout, epochs=3)
    results.append((lr, batch_size, dropout, acc))

    if acc > best_acc:
        best_acc = acc
        best_config = (lr, batch_size, dropout)

    print(f"  Trial {trial+1}: lr={lr:.6f}, bs={batch_size}, dropout={dropout:.2f} → acc={acc:.4f}")

print(f"\n최적 설정: lr={best_config[0]:.6f}, bs={best_config[1]}, dropout={best_config[2]:.2f}")
print(f"최고 정확도: {best_acc:.4f}")


# ============================================
# 9. 전체 학습 파이프라인
# ============================================
print("\n[9] 전체 학습 파이프라인")
print("-" * 40)

def full_training_pipeline(config):
    """최적화 기법이 적용된 전체 학습 파이프라인"""
    set_seed(config['seed'])

    # 모델
    model = SimpleNet(dropout=config['dropout']).to(device)

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # 스케줄러
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 조기 종료
    early_stopping = EarlyStopping(patience=config['patience'])

    # AMP (CUDA인 경우)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 학습
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(config['epochs']):
        # 훈련
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 검증
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        # 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 조기 종료 체크
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"  조기 종료 at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    return model, history

# 설정
config = {
    'seed': 42,
    'lr': 1e-3,
    'batch_size': 64,
    'epochs': 20,
    'dropout': 0.3,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'patience': 5,
    'max_grad_norm': 1.0
}

print("전체 파이프라인 실행 중...")
model, history = full_training_pipeline(config)
print(f"\n최종 검증 정확도: {history['val_acc'][-1]:.4f}")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("학습 최적화 정리")
print("=" * 60)

summary = """
핵심 기법:

1. 학습률 스케줄링
   - Warmup: 초기 안정화
   - Cosine Decay: 점진적 감소
   - OneCycleLR: 배치마다 조정

2. Mixed Precision (AMP)
   - 메모리 절약, 속도 향상
   - autocast() + GradScaler()

3. Gradient Accumulation
   - 작은 배치 → 큰 배치 효과
   - loss /= accumulation_steps

4. Gradient Clipping
   - 기울기 폭발 방지
   - clip_grad_norm_(params, max_norm)

5. 조기 종료
   - 과적합 방지
   - 최적 가중치 복원

권장 설정:
    optimizer = AdamW(lr=1e-4, weight_decay=0.01)
    scheduler = OneCycleLR(max_lr=1e-3)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=10)
"""
print(summary)
print("=" * 60)
