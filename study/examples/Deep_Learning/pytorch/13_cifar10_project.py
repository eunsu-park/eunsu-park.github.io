"""
13. 실전 이미지 분류 프로젝트 (CIFAR-10)

CIFAR-10 분류를 위한 전체 학습 파이프라인을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("CIFAR-10 이미지 분류 프로젝트")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")


# ============================================
# 1. 데이터 준비
# ============================================
print("\n[1] 데이터 준비")
print("-" * 40)

try:
    from torchvision import datasets, transforms

    # CIFAR-10 정규화 값
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # 훈련 변환 (데이터 증강)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 테스트 변환
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 데이터셋 로드
    train_data = datasets.CIFAR10('data', train=True, download=True,
                                   transform=train_transform)
    test_data = datasets.CIFAR10('data', train=False,
                                  transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256)

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"훈련 데이터: {len(train_data)}")
    print(f"테스트 데이터: {len(test_data)}")
    print(f"클래스: {classes}")

    DATA_AVAILABLE = True

except Exception as e:
    print(f"데이터 로드 실패: {e}")
    print("더미 데이터로 진행합니다.")
    DATA_AVAILABLE = False


# ============================================
# 2. 모델 정의
# ============================================
print("\n[2] 모델 정의")
print("-" * 40)

class CIFAR10CNN(nn.Module):
    """CIFAR-10용 CNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32 → 16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: 16 → 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: 8 → 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetCIFAR(nn.Module):
    """CIFAR용 ResNet"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 모델 생성
model = CIFAR10CNN().to(device)
print(f"CIFAR10CNN 파라미터: {sum(p.numel() for p in model.parameters()):,}")

resnet = ResNetCIFAR().to(device)
print(f"ResNetCIFAR 파라미터: {sum(p.numel() for p in resnet.parameters()):,}")


# ============================================
# 3. Mixup 데이터 증강
# ============================================
print("\n[3] Mixup 데이터 증강")
print("-" * 40)

def mixup_data(x, y, alpha=0.2):
    """Mixup 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 손실"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 테스트
x = torch.randn(4, 3, 32, 32)
y = torch.tensor([0, 1, 2, 3])
mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
print(f"Mixup lambda: {lam:.4f}")


# ============================================
# 4. 학습 함수
# ============================================
print("\n[4] 학습 함수")
print("-" * 40)

def train_epoch(model, loader, optimizer, criterion, use_mixup=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        if use_mixup:
            data, target_a, target_b, lam = mixup_data(data, target)

        optimizer.zero_grad()
        output = model(data)

        if use_mixup:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if not use_mixup:
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total if total > 0 else 0
    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ============================================
# 5. 전체 학습 파이프라인
# ============================================
print("\n[5] 학습 실행")
print("-" * 40)

def train_model(model, train_loader, test_loader, epochs=10, use_mixup=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, use_mixup
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        elapsed = time.time() - start_time

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch {epoch+1:3d}: Train Acc={train_acc:5.2f}%, "
              f"Test Acc={test_acc:5.2f}%, Time={elapsed:.1f}s")

    print(f"\n최고 테스트 정확도: {best_acc:.2f}%")
    return history

if DATA_AVAILABLE:
    # 짧은 학습 (데모)
    model = CIFAR10CNN().to(device)
    history = train_model(model, train_loader, test_loader, epochs=5)
else:
    print("데이터 없음 - 학습 스킵")
    history = None


# ============================================
# 6. 결과 시각화
# ============================================
print("\n[6] 결과 시각화")
print("-" * 40)

if history:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['test_acc'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cifar10_training.png', dpi=100)
    plt.close()
    print("그래프 저장: cifar10_training.png")


# ============================================
# 7. 클래스별 정확도
# ============================================
print("\n[7] 클래스별 분석")
print("-" * 40)

if DATA_AVAILABLE:
    def per_class_accuracy(model, loader, classes):
        model.eval()
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)

                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] += 1
                    if pred[i] == label:
                        class_correct[label] += 1

        print("클래스별 정확도:")
        for i, cls in enumerate(classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {cls:12s}: {acc:5.2f}%")

    per_class_accuracy(model, test_loader, classes)


# ============================================
# 8. 예측 시각화
# ============================================
print("\n[8] 예측 시각화")
print("-" * 40)

if DATA_AVAILABLE:
    def visualize_predictions(model, loader, classes, n=8):
        model.eval()
        data, target = next(iter(loader))
        data, target = data[:n].to(device), target[:n]

        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < n:
                img = data[i].cpu().numpy().transpose(1, 2, 0)
                # 역정규화
                img = img * np.array(std) + np.array(mean)
                img = np.clip(img, 0, 1)

                ax.imshow(img)
                color = 'green' if pred[i] == target[i] else 'red'
                ax.set_title(f"Pred: {classes[pred[i]]}\nTrue: {classes[target[i]]}",
                            color=color)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('cifar10_predictions.png', dpi=100)
        plt.close()
        print("예측 시각화 저장: cifar10_predictions.png")

    visualize_predictions(model, test_loader, classes)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("CIFAR-10 프로젝트 정리")
print("=" * 60)

summary = """
주요 기법:

1. 데이터 증강
   - RandomCrop, HorizontalFlip
   - ColorJitter
   - Mixup/CutMix

2. 모델 구조
   - Conv-BN-ReLU 블록
   - Dropout2d, Dropout
   - ResNet 블록 (Skip Connection)

3. 학습 설정
   - SGD + Momentum + Weight Decay
   - Cosine Annealing LR
   - Label Smoothing

예상 정확도:
   - 기본 CNN: 75-80%
   - + 데이터 증강: 80-85%
   - + Mixup: 85-88%
   - ResNet + 전이학습: 90%+

다음 단계:
   - 더 깊은 모델 (ResNet-50)
   - AutoAugment
   - Knowledge Distillation
"""
print(summary)
print("=" * 60)
