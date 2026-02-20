"""
NumPy로 구현한 LeNet-5

원본 논문: LeCun et al. (1998)
"Gradient-Based Learning Applied to Document Recognition"
"""

import numpy as np
from typing import Tuple, List
from conv_numpy import Conv2dNumpy, im2col, col2im


class AvgPool2dNumpy:
    """Average Pooling Layer"""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = {}

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass"""
        N, C, H, W = input.shape
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        # im2col 변환
        col = im2col(input, (K, K), S, padding=0)
        col = col.reshape(N, C, K * K, H_out * W_out)

        # 평균
        output = np.mean(col, axis=2)
        output = output.reshape(N, C, H_out, W_out)

        # 캐시
        self.cache['input_shape'] = input.shape

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        N, C, H_out, W_out = grad_output.shape
        input_shape = self.cache['input_shape']
        K = self.kernel_size

        # 각 원소에 1/(K*K) 만큼 분배
        grad_output_expanded = grad_output.reshape(N, C, 1, H_out * W_out)
        grad_col = np.repeat(grad_output_expanded, K * K, axis=2) / (K * K)
        grad_col = grad_col.reshape(N, C * K * K, H_out * W_out)

        grad_input = col2im(
            grad_col, input_shape, (K, K),
            self.stride, padding=0
        )

        return grad_input


class MaxPool2dNumpy:
    """Max Pooling Layer"""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = {}

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass"""
        N, C, H, W = input.shape
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        # im2col
        col = im2col(input, (K, K), S, padding=0)
        col = col.reshape(N, C, K * K, H_out * W_out)

        # Max
        max_idx = np.argmax(col, axis=2)
        output = np.max(col, axis=2)
        output = output.reshape(N, C, H_out, W_out)

        # 캐시
        self.cache['input_shape'] = input.shape
        self.cache['max_idx'] = max_idx
        self.cache['col_shape'] = (N, C, K * K, H_out * W_out)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        N, C, H_out, W_out = grad_output.shape
        input_shape = self.cache['input_shape']
        max_idx = self.cache['max_idx']
        K = self.kernel_size

        # Max 위치에만 gradient 전달
        grad_col = np.zeros((N, C, K * K, H_out * W_out))

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        idx = max_idx[n, c, h * W_out + w]
                        grad_col[n, c, idx, h * W_out + w] = grad_output[n, c, h, w]

        grad_col = grad_col.reshape(N, C * K * K, H_out * W_out)

        grad_input = col2im(
            grad_col, input_shape, (K, K),
            self.stride, padding=0
        )

        return grad_input


class FlattenNumpy:
    """Flatten Layer"""

    def __init__(self):
        self.cache = {}

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.cache['input_shape'] = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self.cache['input_shape'])


class LinearNumpy:
    """Fully Connected Layer"""

    def __init__(self, in_features: int, out_features: int):
        # Xavier 초기화
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features) * scale
        self.bias = np.zeros(out_features)

        self.weight_grad = None
        self.bias_grad = None
        self.cache = {}

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Y = XW^T + b"""
        self.cache['input'] = input
        return input @ self.weight.T + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        input = self.cache['input']

        # Gradients
        self.weight_grad = grad_output.T @ input
        self.bias_grad = np.sum(grad_output, axis=0)

        # Input gradient
        grad_input = grad_output @ self.weight

        return grad_input

    def update(self, lr: float):
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad


class TanhNumpy:
    """Tanh Activation"""

    def __init__(self):
        self.cache = {}

    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.tanh(input)
        self.cache['output'] = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        output = self.cache['output']
        return grad_output * (1 - output ** 2)


class ReLUNumpy:
    """ReLU Activation"""

    def __init__(self):
        self.cache = {}

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.cache['input'] = input
        return np.maximum(0, input)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        input = self.cache['input']
        return grad_output * (input > 0)


class SoftmaxCrossEntropyNumpy:
    """Softmax + Cross Entropy Loss"""

    def __init__(self):
        self.cache = {}

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Args:
            logits: (N, num_classes)
            labels: (N,) - 클래스 인덱스

        Returns:
            loss: scalar
        """
        N = logits.shape[0]

        # Softmax (수치 안정성)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Cross entropy
        correct_probs = probs[np.arange(N), labels]
        loss = -np.mean(np.log(correct_probs + 1e-10))

        # 캐시
        self.cache['probs'] = probs
        self.cache['labels'] = labels

        return loss

    def backward(self) -> np.ndarray:
        """Gradient: softmax(x) - one_hot(y)"""
        probs = self.cache['probs']
        labels = self.cache['labels']
        N = probs.shape[0]

        grad = probs.copy()
        grad[np.arange(N), labels] -= 1
        grad /= N

        return grad


class LeNet5Numpy:
    """
    LeNet-5 NumPy 구현

    아키텍처:
    Input (1, 32, 32)
    → Conv1 (6, 5, 5) → Tanh → AvgPool
    → Conv2 (16, 5, 5) → Tanh → AvgPool
    → Conv3 (120, 5, 5) → Tanh
    → FC1 (120 → 84) → Tanh
    → FC2 (84 → 10)
    """

    def __init__(self, num_classes: int = 10, use_relu: bool = False):
        """
        Args:
            num_classes: 출력 클래스 수
            use_relu: True면 ReLU, False면 Tanh (원본)
        """
        Activation = ReLUNumpy if use_relu else TanhNumpy

        # Layer 1: Conv + Pool
        self.conv1 = Conv2dNumpy(1, 6, kernel_size=5, stride=1, padding=0)
        self.act1 = Activation()
        self.pool1 = AvgPool2dNumpy(kernel_size=2, stride=2)

        # Layer 2: Conv + Pool
        self.conv2 = Conv2dNumpy(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = Activation()
        self.pool2 = AvgPool2dNumpy(kernel_size=2, stride=2)

        # Layer 3: Conv (→ 1x1)
        self.conv3 = Conv2dNumpy(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = Activation()

        # Flatten
        self.flatten = FlattenNumpy()

        # FC Layers
        self.fc1 = LinearNumpy(120, 84)
        self.act4 = Activation()
        self.fc2 = LinearNumpy(84, num_classes)

        # Loss
        self.criterion = SoftmaxCrossEntropyNumpy()

        # Layer 리스트 (update용)
        self.layers = [
            self.conv1, self.conv2, self.conv3,
            self.fc1, self.fc2
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        # Layer 1
        x = self.conv1.forward(x)     # (N, 6, 28, 28)
        x = self.act1.forward(x)
        x = self.pool1.forward(x)     # (N, 6, 14, 14)

        # Layer 2
        x = self.conv2.forward(x)     # (N, 16, 10, 10)
        x = self.act2.forward(x)
        x = self.pool2.forward(x)     # (N, 16, 5, 5)

        # Layer 3
        x = self.conv3.forward(x)     # (N, 120, 1, 1)
        x = self.act3.forward(x)

        # Flatten + FC
        x = self.flatten.forward(x)   # (N, 120)
        x = self.fc1.forward(x)       # (N, 84)
        x = self.act4.forward(x)
        x = self.fc2.forward(x)       # (N, 10)

        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass"""
        # FC layers (역순)
        grad = self.fc2.backward(grad)
        grad = self.act4.backward(grad)
        grad = self.fc1.backward(grad)

        # Unflatten
        grad = self.flatten.backward(grad)

        # Conv layers (역순)
        grad = self.act3.backward(grad)
        grad = self.conv3.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.act2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.act1.backward(grad)
        grad = self.conv1.backward(grad)

        return grad

    def train_step(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01
    ) -> Tuple[float, float]:
        """
        단일 학습 스텝

        Returns:
            (loss, accuracy)
        """
        # Forward
        logits = self.forward(images)

        # Loss
        loss = self.criterion.forward(logits, labels)

        # Accuracy
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == labels)

        # Backward
        grad = self.criterion.backward()
        self.backward(grad)

        # Update
        for layer in self.layers:
            layer.update(lr)

        return loss, accuracy

    def predict(self, images: np.ndarray) -> np.ndarray:
        """예측"""
        logits = self.forward(images)
        return np.argmax(logits, axis=1)


def load_mnist_subset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MNIST 데이터셋 로드 (간단한 버전)

    실제로는 torchvision이나 keras를 사용
    여기서는 예시용으로 랜덤 데이터 생성
    """
    print("Note: 실제 MNIST 대신 랜덤 데이터 사용")

    # 학습 데이터
    X_train = np.random.randn(1000, 1, 32, 32).astype(np.float32)
    y_train = np.random.randint(0, 10, 1000)

    # 테스트 데이터
    X_test = np.random.randn(200, 1, 32, 32).astype(np.float32)
    y_test = np.random.randint(0, 10, 200)

    return X_train, y_train, X_test, y_test


def train_lenet():
    """LeNet-5 학습"""
    print("=== LeNet-5 NumPy Training ===\n")

    # 데이터
    X_train, y_train, X_test, y_test = load_mnist_subset()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 모델
    model = LeNet5Numpy(num_classes=10, use_relu=True)

    # 하이퍼파라미터
    epochs = 5
    batch_size = 32
    lr = 0.01
    num_batches = len(X_train) // batch_size

    # 학습
    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0.0
        epoch_acc = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            images = X_train[start:end]
            labels = y_train[start:end]

            loss, acc = model.train_step(images, labels, lr)

            epoch_loss += loss
            epoch_acc += acc

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{num_batches}, "
                      f"Loss: {loss:.4f}, Acc: {acc:.4f}")

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        # 검증
        predictions = model.predict(X_test)
        test_acc = np.mean(predictions == y_test)
        print(f"  Test Acc: {test_acc:.4f}")
        print()

    print("Training complete!")


if __name__ == "__main__":
    train_lenet()
