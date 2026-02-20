"""
PyTorch Low-Level VGG 구현

nn.Conv2d, nn.Linear 대신 F.conv2d, torch.matmul 사용
파라미터를 수동으로 관리하며 블록 단위로 구성
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, List, Dict, Optional


# VGG 설정: 숫자 = 출력 채널, 'M' = MaxPool
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGLowLevel:
    """
    VGG Low-Level 구현

    nn.Module 미사용, F.conv2d 등 기본 연산만 사용
    """

    def __init__(
        self,
        config_name: str = 'VGG16',
        num_classes: int = 1000,
        input_channels: int = 3,
        use_bn: bool = False
    ):
        """
        Args:
            config_name: VGG 변형 ('VGG11', 'VGG13', 'VGG16', 'VGG19')
            num_classes: 출력 클래스 수
            input_channels: 입력 채널 수 (RGB=3)
            use_bn: Batch Normalization 사용 여부
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = VGG_CONFIGS[config_name]
        self.use_bn = use_bn

        # Feature extractor 파라미터
        self.conv_params = []
        self.bn_params = [] if use_bn else None
        self._build_features(input_channels)

        # Classifier 파라미터
        self._build_classifier(num_classes)

    def _init_conv_weight(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kaiming 초기화로 Conv 가중치 생성"""
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)

        weight = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size,
            requires_grad=True, device=self.device
        ) * std
        bias = torch.zeros(out_channels, requires_grad=True, device=self.device)

        return weight, bias

    def _init_bn_params(self, num_features: int) -> Dict[str, torch.Tensor]:
        """BatchNorm 파라미터 초기화"""
        return {
            'gamma': torch.ones(num_features, requires_grad=True, device=self.device),
            'beta': torch.zeros(num_features, requires_grad=True, device=self.device),
            'running_mean': torch.zeros(num_features, device=self.device),
            'running_var': torch.ones(num_features, device=self.device),
        }

    def _init_linear_weight(
        self,
        in_features: int,
        out_features: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Xavier 초기화로 Linear 가중치 생성"""
        std = math.sqrt(2.0 / (in_features + out_features))

        weight = torch.randn(
            out_features, in_features,
            requires_grad=True, device=self.device
        ) * std
        bias = torch.zeros(out_features, requires_grad=True, device=self.device)

        return weight, bias

    def _build_features(self, input_channels: int):
        """Feature extractor (Conv layers) 구축"""
        in_channels = input_channels

        for v in self.config:
            if v == 'M':
                # MaxPool은 파라미터 없음
                self.conv_params.append('M')
                if self.use_bn:
                    self.bn_params.append(None)
            else:
                out_channels = v
                weight, bias = self._init_conv_weight(in_channels, out_channels, 3)
                self.conv_params.append({'weight': weight, 'bias': bias})

                if self.use_bn:
                    bn = self._init_bn_params(out_channels)
                    self.bn_params.append(bn)

                in_channels = out_channels

    def _build_classifier(self, num_classes: int):
        """Classifier (FC layers) 구축"""
        # 7×7×512 = 25088 (224×224 입력 기준)
        # CIFAR-10 (32×32) 사용시 1×1×512 = 512

        # FC1: 25088 → 4096
        self.fc1_weight, self.fc1_bias = self._init_linear_weight(512 * 7 * 7, 4096)

        # FC2: 4096 → 4096
        self.fc2_weight, self.fc2_bias = self._init_linear_weight(4096, 4096)

        # FC3: 4096 → num_classes
        self.fc3_weight, self.fc3_bias = self._init_linear_weight(4096, num_classes)

    def _batch_norm(
        self,
        x: torch.Tensor,
        bn_params: Dict[str, torch.Tensor],
        training: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5
    ) -> torch.Tensor:
        """수동 Batch Normalization"""
        if training:
            # 현재 배치의 mean, var 계산
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            # Running statistics 업데이트
            with torch.no_grad():
                bn_params['running_mean'] = (
                    (1 - momentum) * bn_params['running_mean'] +
                    momentum * mean.squeeze()
                )
                bn_params['running_var'] = (
                    (1 - momentum) * bn_params['running_var'] +
                    momentum * var.squeeze()
                )
        else:
            mean = bn_params['running_mean'].view(1, -1, 1, 1)
            var = bn_params['running_var'].view(1, -1, 1, 1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Scale and shift
        gamma = bn_params['gamma'].view(1, -1, 1, 1)
        beta = bn_params['beta'].view(1, -1, 1, 1)

        return gamma * x_norm + beta

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (N, C, H, W) 입력 이미지
            training: 학습 모드 (BN, Dropout에 영향)

        Returns:
            logits: (N, num_classes)
        """
        # Feature extraction
        for i, params in enumerate(self.conv_params):
            if params == 'M':
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            else:
                x = F.conv2d(x, params['weight'], params['bias'],
                            stride=1, padding=1)

                if self.use_bn and self.bn_params[i] is not None:
                    x = self._batch_norm(x, self.bn_params[i], training)

                x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classifier
        # FC1
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)
        if training:
            x = F.dropout(x, p=0.5, training=True)

        # FC2
        x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias
        x = F.relu(x)
        if training:
            x = F.dropout(x, p=0.5, training=True)

        # FC3
        x = torch.matmul(x, self.fc3_weight.t()) + self.fc3_bias

        return x

    def parameters(self) -> List[torch.Tensor]:
        """학습 가능한 파라미터 반환"""
        params = []

        # Conv 파라미터
        for p in self.conv_params:
            if p != 'M':
                params.extend([p['weight'], p['bias']])

        # BN 파라미터
        if self.use_bn:
            for bn in self.bn_params:
                if bn is not None:
                    params.extend([bn['gamma'], bn['beta']])

        # FC 파라미터
        params.extend([
            self.fc1_weight, self.fc1_bias,
            self.fc2_weight, self.fc2_bias,
            self.fc3_weight, self.fc3_bias,
        ])

        return params

    def zero_grad(self):
        """Gradient 초기화"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def to(self, device):
        """Device 이동"""
        self.device = device

        # Conv 파라미터
        for p in self.conv_params:
            if p != 'M':
                p['weight'] = p['weight'].to(device)
                p['bias'] = p['bias'].to(device)

        # BN 파라미터
        if self.use_bn:
            for bn in self.bn_params:
                if bn is not None:
                    for key in bn:
                        bn[key] = bn[key].to(device)

        # FC 파라미터
        for attr in ['fc1_weight', 'fc1_bias', 'fc2_weight',
                     'fc2_bias', 'fc3_weight', 'fc3_bias']:
            tensor = getattr(self, attr)
            setattr(self, attr, tensor.to(device))

        return self

    def count_parameters(self) -> int:
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters())


class VGGSmall(VGGLowLevel):
    """
    CIFAR-10용 작은 VGG

    입력: 32×32 → 출력 feature map: 1×1×512
    """

    def _build_classifier(self, num_classes: int):
        """작은 입력에 맞는 Classifier"""
        # 32×32 입력 → 5번 풀링 → 1×1×512
        self.fc1_weight, self.fc1_bias = self._init_linear_weight(512, 512)
        self.fc2_weight, self.fc2_bias = self._init_linear_weight(512, 512)
        self.fc3_weight, self.fc3_bias = self._init_linear_weight(512, num_classes)


def sgd_step_with_momentum(
    params: List[torch.Tensor],
    velocities: List[torch.Tensor],
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 5e-4
):
    """Momentum SGD with Weight Decay"""
    with torch.no_grad():
        for param, velocity in zip(params, velocities):
            if param.grad is not None:
                # Weight decay
                param.grad.add_(param, alpha=weight_decay)

                # Momentum update
                velocity.mul_(momentum).add_(param.grad)
                param.sub_(velocity, alpha=lr)


def train_epoch(
    model: VGGLowLevel,
    dataloader,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4
) -> Tuple[float, float]:
    """한 에폭 학습"""
    # Velocity 초기화 (첫 에폭)
    if not hasattr(train_epoch, 'velocities') or len(train_epoch.velocities) != len(model.parameters()):
        train_epoch.velocities = [torch.zeros_like(p) for p in model.parameters()]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(model.device)
        labels = labels.to(model.device)

        # Forward
        logits = model.forward(images, training=True)

        # Loss
        loss = F.cross_entropy(logits, labels)

        # Backward
        model.zero_grad()
        loss.backward()

        # Update
        sgd_step_with_momentum(
            model.parameters(),
            train_epoch.velocities,
            lr, momentum, weight_decay
        )

        # Metrics
        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model: VGGLowLevel, dataloader) -> Tuple[float, float]:
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


def visualize_features(model: VGGLowLevel, image: torch.Tensor) -> List[torch.Tensor]:
    """
    각 블록의 feature map 추출

    Returns:
        List of feature maps after each conv block (before pooling)
    """
    features = []
    x = image.to(model.device)

    for i, params in enumerate(model.conv_params):
        if params == 'M':
            features.append(x.clone())  # Pool 전 저장
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        else:
            x = F.conv2d(x, params['weight'], params['bias'],
                        stride=1, padding=1)
            x = F.relu(x)

    features.append(x)  # 마지막 블록
    return features


def main():
    """CIFAR-10으로 VGG 학습 데모"""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("=== VGG Low-Level Training (CIFAR-10) ===\n")

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

    # CIFAR-10 데이터셋
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

    # 모델 (CIFAR용 작은 VGG)
    model = VGGSmall(config_name='VGG16', num_classes=10, use_bn=True)
    model.to(device)

    print(f"VGG16-BN for CIFAR-10")
    print(f"Total parameters: {model.count_parameters():,}\n")

    # 학습
    epochs = 100
    lr = 0.1

    for epoch in range(epochs):
        # Learning rate schedule
        if epoch in [30, 60, 80]:
            lr *= 0.1
            print(f"LR → {lr}")

        train_loss, train_acc = train_epoch(model, train_loader, lr)

        # 10 에폭마다 평가
        if (epoch + 1) % 10 == 0:
            test_loss, test_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}\n")

    # 최종 평가
    final_loss, final_acc = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
