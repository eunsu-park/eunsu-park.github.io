"""
06. CNN 심화 - 유명 아키텍처

VGG, ResNet, EfficientNet 등 유명 아키텍처를 PyTorch로 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("PyTorch CNN 심화 - 유명 아키텍처")
print("=" * 60)


# ============================================
# 1. VGG 블록 및 모델
# ============================================
print("\n[1] VGG16 구현")
print("-" * 40)

def make_vgg_block(in_channels, out_channels, num_convs):
    """VGG 블록 생성"""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels if i == 0 else out_channels,
            out_channels, kernel_size=3, padding=1
        ))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG16 구현"""
    def __init__(self, num_classes=1000):
        super().__init__()
        # 특징 추출부
        self.features = nn.Sequential(
            make_vgg_block(3, 64, 2),    # 224→112
            make_vgg_block(64, 128, 2),  # 112→56
            make_vgg_block(128, 256, 3), # 56→28
            make_vgg_block(256, 512, 3), # 28→14
            make_vgg_block(512, 512, 3), # 14→7
        )

        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

vgg = VGG16(num_classes=10)
print(f"VGG16 파라미터: {sum(p.numel() for p in vgg.parameters()):,}")

# 테스트
x = torch.randn(1, 3, 224, 224)
out = vgg(x)
print(f"입력: {x.shape} → 출력: {out.shape}")


# ============================================
# 2. ResNet Basic Block
# ============================================
print("\n[2] ResNet 구현")
print("-" * 40)

class BasicBlock(nn.Module):
    """ResNet Basic Block (ResNet-18, 34용)"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection!
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck Block (ResNet-50, 101, 152용)"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet 구현"""
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # 초기 층
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # ResNet 층
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# 테스트
resnet = resnet18(num_classes=10)
print(f"ResNet-18 파라미터: {sum(p.numel() for p in resnet.parameters()):,}")

x = torch.randn(1, 3, 224, 224)
out = resnet(x)
print(f"입력: {x.shape} → 출력: {out.shape}")


# ============================================
# 3. SE Block (Squeeze-and-Excitation)
# ============================================
print("\n[3] SE Block")
print("-" * 40)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

# 테스트
se = SEBlock(64)
x = torch.randn(2, 64, 32, 32)
out = se(x)
print(f"SE Block: {x.shape} → {out.shape}")


# ============================================
# 4. MBConv (EfficientNet 블록)
# ============================================
print("\n[4] MBConv Block (EfficientNet)")
print("-" * 40)

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, expand_ratio=6,
                 stride=1, se_ratio=0.25):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_skip = stride == 1 and in_channels == out_channels

        layers = []

        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        self.conv = nn.Sequential(*layers)

        # SE
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, hidden_dim, 1),
            nn.Sigmoid()
        )

        # Project
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = out * self.se(out)
        out = self.project(out)

        if self.use_skip:
            out = out + identity
        return out

# 테스트
mbconv = MBConv(32, 32, expand_ratio=6)
x = torch.randn(2, 32, 28, 28)
out = mbconv(x)
print(f"MBConv: {x.shape} → {out.shape}")


# ============================================
# 5. 사전 학습 모델 사용
# ============================================
print("\n[5] torchvision 사전 학습 모델")
print("-" * 40)

try:
    import torchvision.models as models

    # 다양한 사전 학습 모델
    model_names = ['resnet18', 'resnet50', 'vgg16', 'mobilenet_v2']

    for name in model_names:
        model = getattr(models, name)(weights=None)  # 가중치 없이 구조만
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} 파라미터")

    # 사전 학습된 ResNet50 로드
    print("\n사전 학습된 ResNet50 로드:")
    resnet50_pretrained = models.resnet50(weights='IMAGENET1K_V2')
    print(f"  마지막 층: {resnet50_pretrained.fc}")

    # 전이 학습을 위한 수정
    resnet50_pretrained.fc = nn.Linear(2048, 10)  # 10 클래스로 변경
    print(f"  수정된 마지막 층: {resnet50_pretrained.fc}")

except ImportError:
    print("torchvision이 설치되지 않았습니다.")


# ============================================
# 6. 모델 비교
# ============================================
print("\n[6] 모델 비교")
print("-" * 40)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def measure_forward_time(model, input_shape, iterations=100):
    import time
    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        # 워밍업
        for _ in range(10):
            _ = model(x)
        # 측정
        start = time.time()
        for _ in range(iterations):
            _ = model(x)
        end = time.time()
    return (end - start) / iterations * 1000  # ms

# 간단한 모델들 비교
models_to_compare = {
    'VGG16 (simple)': VGG16(num_classes=10),
    'ResNet-18': resnet18(num_classes=10),
    'ResNet-50': resnet50(num_classes=10),
}

print(f"{'Model':<20} {'Params':>12} {'Time (ms)':>12}")
print("-" * 46)

for name, model in models_to_compare.items():
    params = count_parameters(model)
    try:
        time_ms = measure_forward_time(model, (1, 3, 224, 224), iterations=10)
        print(f"{name:<20} {params:>12,} {time_ms:>12.2f}")
    except:
        print(f"{name:<20} {params:>12,} {'N/A':>12}")


# ============================================
# 7. 간단한 ResNet 실험
# ============================================
print("\n[7] Skip Connection 효과 실험")
print("-" * 40)

class ResBlockWithoutSkip(nn.Module):
    """Skip Connection 없는 블록"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out)  # Skip 없음!

class ResBlockWithSkip(nn.Module):
    """Skip Connection 있는 블록"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)  # Skip 있음!

# 깊은 네트워크 비교
def make_deep_net(block_class, num_blocks, channels=64):
    layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU()]
    for _ in range(num_blocks):
        layers.append(block_class(channels))
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(channels, 10))
    return nn.Sequential(*layers)

# 기울기 확인
def check_gradient_flow(model, depth):
    model.train()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # 첫 번째 Conv 기울기 확인
    first_conv_grad = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if module.weight.grad is not None:
                first_conv_grad = module.weight.grad.abs().mean().item()
                break

    return first_conv_grad

print("기울기 흐름 비교 (깊은 네트워크):")
for depth in [5, 10, 20]:
    net_no_skip = make_deep_net(ResBlockWithoutSkip, depth)
    net_with_skip = make_deep_net(ResBlockWithSkip, depth)

    grad_no_skip = check_gradient_flow(net_no_skip, depth)
    grad_with_skip = check_gradient_flow(net_with_skip, depth)

    print(f"  깊이 {depth:2d}: Skip 없음 = {grad_no_skip:.6f}, Skip 있음 = {grad_with_skip:.6f}")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("CNN 아키텍처 정리")
print("=" * 60)

summary = """
주요 아키텍처:

1. VGG (2014)
   - 3×3 Conv만 사용
   - 깊이 = 성능 (단순하지만 파라미터 많음)

2. ResNet (2015)
   - Skip Connection으로 기울기 소실 해결
   - 100+ 층도 학습 가능
   - 가장 널리 사용됨

3. EfficientNet (2019)
   - Compound Scaling
   - MBConv (Depthwise Separable + SE)
   - 효율적인 파라미터 사용

핵심 기법:
- Batch Normalization
- Skip Connection (Residual)
- Depthwise Separable Conv
- Squeeze-and-Excitation

실전 선택:
- 빠른 추론: MobileNet, EfficientNet-B0
- 높은 정확도: EfficientNet-B4~B7
- 균형: ResNet-50
"""
print(summary)
print("=" * 60)
