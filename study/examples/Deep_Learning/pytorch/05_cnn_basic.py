"""
05. CNN 기초 - PyTorch 버전

합성곱 신경망(CNN)을 PyTorch로 구현합니다.
MNIST와 CIFAR-10 분류를 수행합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("PyTorch CNN 기초")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")


# ============================================
# 1. 합성곱 연산 이해
# ============================================
print("\n[1] 합성곱 연산 이해")
print("-" * 40)

# Conv2d 기본
conv = nn.Conv2d(
    in_channels=1,    # 입력 채널
    out_channels=3,   # 필터 개수 (출력 채널)
    kernel_size=3,    # 필터 크기
    stride=1,         # 이동 간격
    padding=1         # 패딩
)

print(f"Conv2d 파라미터:")
print(f"  weight shape: {conv.weight.shape}")  # (out, in, H, W)
print(f"  bias shape: {conv.bias.shape}")       # (out,)

# 입력/출력 확인
x = torch.randn(1, 1, 8, 8)  # (batch, channel, H, W)
out = conv(x)
print(f"\n입력: {x.shape} → 출력: {out.shape}")


# 출력 크기 계산
def calc_output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1

print("\n출력 크기 공식: (입력 - 커널 + 2×패딩) / 스트라이드 + 1")
for k, s, p in [(3, 1, 0), (3, 1, 1), (3, 2, 0), (5, 1, 2)]:
    out_size = calc_output_size(32, k, s, p)
    print(f"  입력=32, kernel={k}, stride={s}, pad={p} → 출력={out_size}")


# ============================================
# 2. 풀링 연산
# ============================================
print("\n[2] 풀링 연산")
print("-" * 40)

# MaxPool2d
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.tensor([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=torch.float32)

print(f"입력:\n{x.squeeze()}")
print(f"\nMaxPool2d(2,2) 출력:\n{pool(x).squeeze()}")

# AvgPool2d
avg_pool = nn.AvgPool2d(2, 2)
print(f"\nAvgPool2d(2,2) 출력:\n{avg_pool(x).squeeze()}")


# ============================================
# 3. MNIST CNN
# ============================================
print("\n[3] MNIST CNN")
print("-" * 40)

class MNISTNet(nn.Module):
    """MNIST용 간단한 CNN"""
    def __init__(self):
        super().__init__()
        # Conv 블록 1: 1→32 채널, 28→14
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv 블록 2: 32→64 채널, 14→7
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # FC 블록
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MNISTNet()
print(model)

# 파라미터 수 계산
total = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터: {total:,}")


# ============================================
# 4. MNIST 학습
# ============================================
print("\n[4] MNIST 학습")
print("-" * 40)

# 데이터 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

try:
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    print(f"훈련 데이터: {len(train_data)} 샘플")
    print(f"테스트 데이터: {len(test_data)} 샘플")

    # 모델, 손실, 옵티마이저
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 학습
    epochs = 3
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

    # 테스트
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"\n테스트 정확도: {100. * correct / total:.2f}%")

except Exception as e:
    print(f"MNIST 로드 실패 (오프라인?): {e}")
    print("데모 모드로 진행합니다.")

    # 더미 데이터로 테스트
    x_dummy = torch.randn(4, 1, 28, 28)
    model = MNISTNet()
    out = model(x_dummy)
    print(f"더미 입력: {x_dummy.shape} → 출력: {out.shape}")


# ============================================
# 5. 특징 맵 시각화
# ============================================
print("\n[5] 특징 맵 시각화")
print("-" * 40)

def visualize_feature_maps(model, image, layer_name='conv1'):
    """특징 맵 시각화"""
    model.eval()

    # 훅으로 중간 출력 캡처
    activations = {}
    def hook_fn(module, input, output):
        activations['output'] = output.detach()

    hook = getattr(model, layer_name).register_forward_hook(hook_fn)

    with torch.no_grad():
        model(image)

    hook.remove()
    feature_maps = activations['output']

    # 시각화
    n_maps = min(16, feature_maps.shape[1])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i < n_maps:
            ax.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')

    plt.suptitle(f'{layer_name} Feature Maps')
    plt.tight_layout()
    plt.savefig('cnn_feature_maps.png', dpi=100)
    plt.close()
    print(f"특징 맵 저장: cnn_feature_maps.png")

# 시각화 (학습된 모델이 있는 경우)
try:
    sample_image = train_data[0][0].unsqueeze(0).to(device)
    visualize_feature_maps(model, sample_image, 'conv1')
except:
    print("시각화 스킵 (데이터 없음)")


# ============================================
# 6. 필터 시각화
# ============================================
print("\n[6] 필터 시각화")
print("-" * 40)

def visualize_filters(model, layer_name='conv1'):
    """Conv 필터 시각화"""
    filters = getattr(model, layer_name).weight.detach().cpu()

    # 첫 16개 필터
    n_filters = min(16, filters.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # 첫 번째 입력 채널의 필터
            ax.imshow(filters[i, 0].numpy(), cmap='gray')
        ax.axis('off')

    plt.suptitle(f'{layer_name} Filters')
    plt.tight_layout()
    plt.savefig('cnn_filters.png', dpi=100)
    plt.close()
    print(f"필터 저장: cnn_filters.png")

try:
    visualize_filters(model, 'conv1')
except:
    print("필터 시각화 스킵")


# ============================================
# 7. CIFAR-10 CNN
# ============================================
print("\n[7] CIFAR-10 CNN")
print("-" * 40)

class CIFAR10Net(nn.Module):
    """CIFAR-10용 CNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 블록 1: 3→64, 32→16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # 블록 2: 64→128, 16→8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # 블록 3: 128→256, 8→4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cifar_model = CIFAR10Net()
print(cifar_model)

# 파라미터 수
total = sum(p.numel() for p in cifar_model.parameters())
print(f"\n총 파라미터: {total:,}")

# 테스트
x_test = torch.randn(2, 3, 32, 32)
out = cifar_model(x_test)
print(f"입력: {x_test.shape} → 출력: {out.shape}")


# ============================================
# 8. 데이터 증강
# ============================================
print("\n[8] 데이터 증강")
print("-" * 40)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
])

print("훈련 변환: RandomCrop, Flip, ColorJitter, Normalize")
print("테스트 변환: ToTensor, Normalize")


# ============================================
# 9. 모델 저장/로드
# ============================================
print("\n[9] 모델 저장/로드")
print("-" * 40)

# 저장
torch.save(cifar_model.state_dict(), 'cifar_cnn.pth')
print("모델 저장: cifar_cnn.pth")

# 로드
loaded_model = CIFAR10Net()
loaded_model.load_state_dict(torch.load('cifar_cnn.pth', weights_only=True))
loaded_model.eval()
print("모델 로드 완료")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("CNN 기초 정리")
print("=" * 60)

summary = """
CNN 구성요소:
1. Conv2d: 지역 패턴 추출
2. BatchNorm2d: 학습 안정화
3. ReLU: 비선형성
4. MaxPool2d: 공간 축소
5. Dropout2d: 과적합 방지
6. Flatten + Linear: 분류

출력 크기 공식:
    output = (input - kernel + 2*padding) / stride + 1

일반적인 패턴:
    Conv → BN → ReLU → Pool (반복) → Flatten → FC

권장 설정:
- kernel_size=3, padding=1 (same padding)
- 채널 증가: 64 → 128 → 256
- Pool로 공간 축소
- FC 앞에 Dropout
"""
print(summary)
print("=" * 60)
