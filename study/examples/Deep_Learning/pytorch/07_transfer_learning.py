"""
07. 전이학습 (Transfer Learning)

사전 학습된 모델을 활용한 전이학습을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

print("=" * 60)
print("PyTorch 전이학습 (Transfer Learning)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")


# ============================================
# 1. 사전 학습 모델 로드
# ============================================
print("\n[1] 사전 학습 모델 로드")
print("-" * 40)

try:
    import torchvision.models as models

    # 다양한 사전 학습 모델
    print("사용 가능한 사전 학습 모델:")
    pretrained_models = {
        'ResNet-18': lambda: models.resnet18(weights='IMAGENET1K_V1'),
        'ResNet-50': lambda: models.resnet50(weights='IMAGENET1K_V2'),
        'EfficientNet-B0': lambda: models.efficientnet_b0(weights='IMAGENET1K_V1'),
        'MobileNet-V2': lambda: models.mobilenet_v2(weights='IMAGENET1K_V1'),
    }

    for name, loader in pretrained_models.items():
        model = loader()
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} 파라미터")

    TORCHVISION_AVAILABLE = True
except ImportError:
    print("torchvision이 설치되지 않았습니다. 데모 모드로 진행합니다.")
    TORCHVISION_AVAILABLE = False


# ============================================
# 2. 특성 추출 (Feature Extraction)
# ============================================
print("\n[2] 특성 추출 (Feature Extraction)")
print("-" * 40)

if TORCHVISION_AVAILABLE:
    # ResNet-18 로드
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 원래 분류기 확인
    print(f"원래 FC 층: {model.fc}")

    # 모든 가중치 고정
    for param in model.parameters():
        param.requires_grad = False

    # 마지막 층 교체
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 10)  # 10 클래스
    )

    print(f"새 FC 층: {model.fc}")

    # 학습 가능한 파라미터 확인
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"학습 가능 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ============================================
# 3. 미세 조정 (Fine-tuning)
# ============================================
print("\n[3] 미세 조정 (Fine-tuning)")
print("-" * 40)

if TORCHVISION_AVAILABLE:
    # 새로운 모델 로드
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 마지막 층 교체
    model.fc = nn.Linear(model.fc.in_features, 10)

    # 전체 학습 가능 (기본)
    print("전체 미세 조정:")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  학습 가능 파라미터: {trainable:,}")


# ============================================
# 4. 점진적 해동 (Gradual Unfreezing)
# ============================================
print("\n[4] 점진적 해동 (Gradual Unfreezing)")
print("-" * 40)

if TORCHVISION_AVAILABLE:
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 1단계: 모든 층 고정
    for param in model.parameters():
        param.requires_grad = False

    # 마지막 층만 학습 가능
    model.fc = nn.Linear(model.fc.in_features, 10)

    def count_trainable(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("점진적 해동 과정:")
    print(f"  1단계 (FC만): {count_trainable(model):,} 파라미터")

    # 2단계: layer4 해동
    for param in model.layer4.parameters():
        param.requires_grad = True
    print(f"  2단계 (FC + layer4): {count_trainable(model):,} 파라미터")

    # 3단계: layer3 해동
    for param in model.layer3.parameters():
        param.requires_grad = True
    print(f"  3단계 (FC + layer4 + layer3): {count_trainable(model):,} 파라미터")

    # 4단계: 전체 해동
    for param in model.parameters():
        param.requires_grad = True
    print(f"  4단계 (전체): {count_trainable(model):,} 파라미터")


# ============================================
# 5. 차등 학습률 (Discriminative Learning Rates)
# ============================================
print("\n[5] 차등 학습률")
print("-" * 40)

if TORCHVISION_AVAILABLE:
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10)

    # 층별 다른 학습률
    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters(), 'lr': 1e-5},
        {'params': model.layer1.parameters(), 'lr': 2e-5},
        {'params': model.layer2.parameters(), 'lr': 5e-5},
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 2e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3},
    ])

    print("층별 학습률:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  그룹 {i}: lr = {group['lr']}")


# ============================================
# 6. 데이터 전처리 (ImageNet 정규화)
# ============================================
print("\n[6] ImageNet 정규화")
print("-" * 40)

try:
    from torchvision import transforms

    # ImageNet 정규화 값
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    print(f"ImageNet Mean: {imagenet_mean}")
    print(f"ImageNet Std: {imagenet_std}")
    print("훈련 변환: RandomResizedCrop, Flip, Normalize")
    print("검증 변환: Resize, CenterCrop, Normalize")
except:
    print("transforms 로드 실패")


# ============================================
# 7. 전이학습 전체 파이프라인
# ============================================
print("\n[7] 전이학습 전체 파이프라인")
print("-" * 40)

class TransferLearningPipeline:
    """전이학습 파이프라인"""

    def __init__(self, backbone='resnet18', num_classes=10, strategy='finetune'):
        self.strategy = strategy

        if TORCHVISION_AVAILABLE:
            # 백본 로드
            if backbone == 'resnet18':
                self.model = models.resnet18(weights='IMAGENET1K_V1')
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
            elif backbone == 'resnet50':
                self.model = models.resnet50(weights='IMAGENET1K_V2')
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Unknown backbone: {backbone}")

            # 전략에 따른 가중치 고정
            if strategy == 'feature_extract':
                self._freeze_backbone()
            elif strategy == 'finetune':
                pass  # 전체 학습 가능
            elif strategy == 'gradual':
                self._freeze_backbone()
        else:
            # 데모용 간단한 모델
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, num_classes)
            )

    def _freeze_backbone(self):
        """FC 제외 모든 층 고정"""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_layer(self, layer_name):
        """특정 층 해동"""
        layer = getattr(self.model, layer_name, None)
        if layer:
            for param in layer.parameters():
                param.requires_grad = True

    def get_optimizer(self, lr=1e-4):
        """최적화기 생성"""
        if self.strategy == 'feature_extract':
            # 학습 가능한 파라미터만
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            return torch.optim.Adam(params, lr=lr)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def summary(self):
        """모델 요약"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"전략: {self.strategy}")
        print(f"학습 가능: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# 테스트
print("\n전략별 비교:")
for strategy in ['feature_extract', 'finetune']:
    print(f"\n{strategy}:")
    pipeline = TransferLearningPipeline('resnet18', 10, strategy)
    pipeline.summary()


# ============================================
# 8. 더미 데이터로 학습 예시
# ============================================
print("\n[8] 학습 예시 (더미 데이터)")
print("-" * 40)

# 더미 데이터 생성
X_train = torch.randn(100, 3, 224, 224)
y_train = torch.randint(0, 10, (100,))

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 파이프라인 설정
pipeline = TransferLearningPipeline('resnet18', 10, 'feature_extract')
model = pipeline.model.to(device)
optimizer = pipeline.get_optimizer(lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 간단한 학습
model.train()
for epoch in range(2):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")


# ============================================
# 9. 전이학습 체크리스트
# ============================================
print("\n[9] 전이학습 체크리스트")
print("-" * 40)

checklist = """
✓ 사전 학습 모델 선택
  - 작업과 유사한 데이터로 학습된 모델
  - ImageNet 모델이 대부분의 경우 좋음

✓ 전처리
  - ImageNet 정규화 사용
  - 모델 입력 크기 맞추기 (보통 224×224)

✓ 전략 선택
  - 데이터 적음: 특성 추출 (FC만 학습)
  - 데이터 충분: 미세 조정 (전체 학습)
  - 중간: 점진적 해동

✓ 학습률
  - 특성 추출: 1e-3 ~ 1e-2
  - 미세 조정: 1e-5 ~ 1e-4
  - 차등 학습률 고려

✓ 정규화
  - Dropout, Weight Decay
  - 데이터 증강
  - 조기 종료

✓ 모드 전환
  - 훈련: model.train()
  - 평가: model.eval()
"""
print(checklist)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("전이학습 정리")
print("=" * 60)

summary = """
전이학습 전략:

1. 특성 추출 (Feature Extraction)
   - 사전 학습 가중치 고정
   - 마지막 층만 학습
   - 데이터 적을 때 적합

2. 미세 조정 (Fine-tuning)
   - 전체 네트워크 학습
   - 낮은 학습률 사용
   - 데이터 충분할 때

3. 점진적 해동 (Gradual Unfreezing)
   - 후반 층부터 순차적 해동
   - 균형 잡힌 접근

핵심 코드:
    # 가중치 고정
    for param in model.parameters():
        param.requires_grad = False

    # 마지막 층 교체
    model.fc = nn.Linear(in_features, num_classes)

    # ImageNet 정규화
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
"""
print(summary)
print("=" * 60)
