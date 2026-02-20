"""
PyTorch Low-Level ResNet 구현

nn.Conv2d, nn.BatchNorm2d 대신 F.conv2d, 수동 BN 사용
BasicBlock과 Bottleneck 모두 구현
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, List, Dict, Optional, Literal


class BatchNorm2dManual:
    """수동 Batch Normalization"""

    def __init__(self, num_features: int, device: torch.device):
        self.gamma = torch.ones(num_features, requires_grad=True, device=device)
        self.beta = torch.zeros(num_features, requires_grad=True, device=device)
        self.running_mean = torch.zeros(num_features, device=device)
        self.running_var = torch.ones(num_features, device=device)
        self.momentum = 0.1
        self.eps = 1e-5

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean +
                    self.momentum * mean.squeeze()
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var +
                    self.momentum * var.squeeze()
                )
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        return gamma * x_norm + beta

    def parameters(self) -> List[torch.Tensor]:
        return [self.gamma, self.beta]


class ConvBN:
    """Conv + BatchNorm 조합"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        device: torch.device = None
    ):
        self.stride = stride
        self.padding = padding

        # Kaiming 초기화
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)

        self.weight = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size,
            requires_grad=True, device=device
        ) * std

        # Conv에 bias 없음 (BN이 있으므로)
        self.bn = BatchNorm2dManual(out_channels, device)

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        x = F.conv2d(x, self.weight, None, self.stride, self.padding)
        x = self.bn(x, training)
        return x

    def parameters(self) -> List[torch.Tensor]:
        return [self.weight] + self.bn.parameters()


class BasicBlock:
    """
    ResNet BasicBlock (ResNet-18, 34용)

    구조: Conv3×3 → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        device: torch.device = None
    ):
        self.conv1 = ConvBN(in_channels, out_channels, 3, stride, 1, device)
        self.conv2 = ConvBN(out_channels, out_channels, 3, 1, 1, device)

        # Shortcut (차원이 다를 때만)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, 1, stride, 0, device)

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        identity = x

        out = self.conv1(x, training)
        out = F.relu(out)
        out = self.conv2(out, training)

        if self.shortcut is not None:
            identity = self.shortcut(x, training)

        out = out + identity  # Skip connection!
        out = F.relu(out)

        return out

    def parameters(self) -> List[torch.Tensor]:
        params = self.conv1.parameters() + self.conv2.parameters()
        if self.shortcut is not None:
            params += self.shortcut.parameters()
        return params


class Bottleneck:
    """
    ResNet Bottleneck (ResNet-50, 101, 152용)

    구조: Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+shortcut) → ReLU
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        device: torch.device = None
    ):
        # Bottleneck: 채널 축소 → 3×3 → 채널 복원
        self.conv1 = ConvBN(in_channels, out_channels, 1, 1, 0, device)  # 축소
        self.conv2 = ConvBN(out_channels, out_channels, 3, stride, 1, device)  # 주요 연산
        self.conv3 = ConvBN(out_channels, out_channels * self.expansion, 1, 1, 0, device)  # 복원

        # Shortcut
        self.shortcut = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = ConvBN(
                in_channels, out_channels * self.expansion, 1, stride, 0, device
            )

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        identity = x

        out = self.conv1(x, training)
        out = F.relu(out)

        out = self.conv2(out, training)
        out = F.relu(out)

        out = self.conv3(out, training)

        if self.shortcut is not None:
            identity = self.shortcut(x, training)

        out = out + identity  # Skip connection!
        out = F.relu(out)

        return out

    def parameters(self) -> List[torch.Tensor]:
        params = self.conv1.parameters() + self.conv2.parameters() + self.conv3.parameters()
        if self.shortcut is not None:
            params += self.shortcut.parameters()
        return params


class ResNetLowLevel:
    """
    ResNet Low-Level 구현

    nn.Module 미사용, 수동 파라미터 관리
    """

    CONFIGS = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3]),
    }

    def __init__(
        self,
        config_name: str = 'resnet50',
        num_classes: int = 1000,
        input_channels: int = 3
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        block_class, num_blocks = self.CONFIGS[config_name]
        self.expansion = block_class.expansion

        # Stem: Conv7×7 + BN + ReLU + MaxPool
        fan_in = input_channels * 7 * 7
        std = math.sqrt(2.0 / fan_in)
        self.conv1_weight = torch.randn(
            64, input_channels, 7, 7,
            requires_grad=True, device=self.device
        ) * std
        self.bn1 = BatchNorm2dManual(64, self.device)

        # Residual Layers
        self.in_channels = 64
        self.layer1 = self._make_layer(block_class, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_class, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, num_blocks[3], stride=2)

        # Classifier
        fc_in = 512 * self.expansion
        std = math.sqrt(2.0 / (fc_in + num_classes))
        self.fc_weight = torch.randn(
            num_classes, fc_in,
            requires_grad=True, device=self.device
        ) * std
        self.fc_bias = torch.zeros(num_classes, requires_grad=True, device=self.device)

    def _make_layer(
        self,
        block_class,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> List:
        """레이어 (여러 블록) 생성"""
        blocks = []

        # 첫 번째 블록: stride 적용, 차원 변경
        blocks.append(block_class(
            self.in_channels, out_channels, stride, self.device
        ))
        self.in_channels = out_channels * self.expansion

        # 나머지 블록: stride=1
        for _ in range(1, num_blocks):
            blocks.append(block_class(
                self.in_channels, out_channels, 1, self.device
            ))

        return blocks

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (N, C, H, W) 입력 이미지
            training: 학습 모드

        Returns:
            logits: (N, num_classes)
        """
        # Stem
        x = F.conv2d(x, self.conv1_weight, None, stride=2, padding=3)
        x = self.bn1(x, training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Residual Layers
        for block in self.layer1:
            x = block(x, training)
        for block in self.layer2:
            x = block(x, training)
        for block in self.layer3:
            x = block(x, training)
        for block in self.layer4:
            x = block(x, training)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Classifier
        x = torch.matmul(x, self.fc_weight.t()) + self.fc_bias

        return x

    def parameters(self) -> List[torch.Tensor]:
        """학습 가능한 파라미터 반환"""
        params = [self.conv1_weight] + self.bn1.parameters()

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                params += block.parameters()

        params += [self.fc_weight, self.fc_bias]
        return params

    def zero_grad(self):
        """Gradient 초기화"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def to(self, device):
        """Device 이동"""
        self.device = device

        # Stem
        self.conv1_weight = self.conv1_weight.to(device)
        self.bn1.gamma = self.bn1.gamma.to(device)
        self.bn1.beta = self.bn1.beta.to(device)
        self.bn1.running_mean = self.bn1.running_mean.to(device)
        self.bn1.running_var = self.bn1.running_var.to(device)

        # FC
        self.fc_weight = self.fc_weight.to(device)
        self.fc_bias = self.fc_bias.to(device)

        return self

    def count_parameters(self) -> int:
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters())


class ResNetSmall(ResNetLowLevel):
    """CIFAR-10용 작은 ResNet"""

    def __init__(
        self,
        config_name: str = 'resnet18',
        num_classes: int = 10,
        input_channels: int = 3
    ):
        # 부모 초기화 건너뛰고 직접 구성
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        block_class, num_blocks = self.CONFIGS[config_name]
        self.expansion = block_class.expansion

        # Stem: 3×3 Conv (7×7 대신)
        fan_in = input_channels * 3 * 3
        std = math.sqrt(2.0 / fan_in)
        self.conv1_weight = torch.randn(
            64, input_channels, 3, 3,
            requires_grad=True, device=self.device
        ) * std
        self.bn1 = BatchNorm2dManual(64, self.device)

        # MaxPool 없음 (32×32 입력이므로)

        # Residual Layers
        self.in_channels = 64
        self.layer1 = self._make_layer(block_class, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_class, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, num_blocks[3], stride=2)

        # Classifier
        fc_in = 512 * self.expansion
        std = math.sqrt(2.0 / (fc_in + num_classes))
        self.fc_weight = torch.randn(
            num_classes, fc_in,
            requires_grad=True, device=self.device
        ) * std
        self.fc_bias = torch.zeros(num_classes, requires_grad=True, device=self.device)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Stem (MaxPool 없음)
        x = F.conv2d(x, self.conv1_weight, None, stride=1, padding=1)
        x = self.bn1(x, training)
        x = F.relu(x)

        # Residual Layers
        for block in self.layer1:
            x = block(x, training)
        for block in self.layer2:
            x = block(x, training)
        for block in self.layer3:
            x = block(x, training)
        for block in self.layer4:
            x = block(x, training)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Classifier
        x = torch.matmul(x, self.fc_weight.t()) + self.fc_bias

        return x


def sgd_step_with_momentum(
    params: List[torch.Tensor],
    velocities: List[torch.Tensor],
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-4
):
    """Momentum SGD with Weight Decay"""
    with torch.no_grad():
        for param, velocity in zip(params, velocities):
            if param.grad is not None:
                param.grad.add_(param, alpha=weight_decay)
                velocity.mul_(momentum).add_(param.grad)
                param.sub_(velocity, alpha=lr)


def train_epoch(
    model: ResNetLowLevel,
    dataloader,
    lr: float,
    velocities: List[torch.Tensor]
) -> Tuple[float, float]:
    """한 에폭 학습"""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(model.device)
        labels = labels.to(model.device)

        logits = model.forward(images, training=True)
        loss = F.cross_entropy(logits, labels)

        model.zero_grad()
        loss.backward()

        sgd_step_with_momentum(model.parameters(), velocities, lr)

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model: ResNetLowLevel, dataloader) -> Tuple[float, float]:
    """평가"""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(model.device)
        labels = labels.to(model.device)

        logits = model.forward(images, training=False)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def visualize_gradient_flow(model: ResNetLowLevel):
    """각 레이어의 gradient 크기 시각화"""
    import matplotlib.pyplot as plt

    gradients = []
    names = []

    for i, block in enumerate(model.layer1 + model.layer2 + model.layer3 + model.layer4):
        for j, param in enumerate(block.parameters()):
            if param.grad is not None:
                gradients.append(param.grad.abs().mean().item())
                names.append(f"block{i}_param{j}")

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(gradients)), gradients)
    plt.xlabel('Layer')
    plt.ylabel('Mean |Gradient|')
    plt.title('Gradient Flow through ResNet')
    plt.tight_layout()
    plt.savefig('gradient_flow.png')
    print("Saved gradient_flow.png")


def main():
    """CIFAR-10으로 ResNet 학습 데모"""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("=== ResNet Low-Level Training (CIFAR-10) ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터 전처리
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    # 모델
    model = ResNetSmall(config_name='resnet18', num_classes=10)
    model.to(device)

    print(f"ResNet-18 for CIFAR-10")
    print(f"Total parameters: {model.count_parameters():,}\n")

    # Velocity 초기화
    velocities = [torch.zeros_like(p) for p in model.parameters()]

    # 학습
    epochs = 100
    lr = 0.1

    for epoch in range(epochs):
        # Learning rate schedule
        if epoch in [30, 60, 80]:
            lr *= 0.1
            print(f"LR → {lr}")

        train_loss, train_acc = train_epoch(model, train_loader, lr, velocities)

        if (epoch + 1) % 10 == 0:
            test_loss, test_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}\n")

    final_loss, final_acc = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
